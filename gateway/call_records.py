"""Publish an LlmCallRecord to NATS for every proxied LLM call.

The publisher uses a bounded in-memory buffer with an on-disk spool so that a
NATS outage never delays or drops a caller's response.  A background thread
connects (and reconnects) to NATS, drains the buffer via JetStream, and replays
spooled records once the connection is restored.

LiteLLM callbacks ``on_success`` / ``on_failure`` are registered in
``config/litellm_config.yaml`` and hand each event to the publisher without
blocking the request path.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("gateway.call_records")

DEFAULT_NATS_URL = "nats://localhost:4222"
DEFAULT_SUBJECT = "petrosa.llm.call_record.v1"
MAX_BUFFER_SIZE = 10_000
MAX_SPOOL_FILES = 1_000
SPOOL_DIR_ENV = "CALL_RECORD_SPOOL_DIR"


class CallRecordPublisher:
    """Bounded in-memory buffer, on-disk spool, background NATS JetStream publish."""

    def __init__(
        self,
        nats_url: str | None = None,
        subject: str | None = None,
        buffer_size: int = MAX_BUFFER_SIZE,
        max_spool_files: int = MAX_SPOOL_FILES,
        spool_dir: str | Path | None = None,
        connect_fn: Any = None,
    ) -> None:
        self._nats_url = nats_url or os.environ.get("NATS_URL", DEFAULT_NATS_URL)
        self._subject = subject or os.environ.get(
            "CALL_RECORD_SUBJECT", DEFAULT_SUBJECT
        )
        self._buffer: deque[dict] = deque(maxlen=buffer_size)
        self._max_spool_files = max_spool_files
        default_spool = Path(tempfile.gettempdir()) / "call_records"
        self._spool_dir = Path(
            spool_dir or os.environ.get(SPOOL_DIR_ENV) or default_spool
        )
        self._lock = threading.Lock()
        self._nats_conn: Any = None
        self._publish_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connect_fn = connect_fn or self._default_connect

    def start(self) -> None:
        if self._publish_thread is not None and self._publish_thread.is_alive():
            return
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._publish_thread = threading.Thread(target=self._run, daemon=True)
        self._publish_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=5.0)
        self._drain_to_nats()
        if self._nats_conn is not None:
            try:
                self._nats_conn.flush(timeout=2_000)
                self._nats_conn.close()
            except Exception:
                pass
            self._nats_conn = None

    def publish(self, record_dict: dict) -> None:
        """Accept a serialised record; never blocks the caller."""
        with self._lock:
            if len(self._buffer) >= self._buffer.maxlen:  # type: ignore[union-attr]
                self._spool(record_dict)
            else:
                self._buffer.append(record_dict)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._nats_conn is None or not self._is_connected():
                    self._connect()
                self._drain_to_nats()
                self._replay_spool()
            except Exception as exc:
                logger.debug("publish loop error (will retry): %s", exc)
            self._stop_event.wait(timeout=0.1)

    def _is_connected(self) -> bool:
        return self._nats_conn is not None and self._nats_conn.is_connected()

    def _default_connect(self) -> Any:
        import nats

        if self._nats_conn is not None:
            try:
                self._nats_conn.close()
            except Exception:
                pass
        return nats.connect(
            [self._nats_url],
            wait=2_000,
            allow_reconnect=True,
            max_reconnect_attempts=-1,
            reconnect_time_wait=500,
        )

    def _connect(self) -> None:
        self._nats_conn = self._connect_fn()

    def _drain_to_nats(self) -> None:
        while self._buffer:
            record = self._buffer.popleft()
            payload = json.dumps(record).encode("utf-8")
            try:
                self._nats_conn.publish_jetstream(self._subject, payload)
            except Exception as exc:
                logger.debug("jetstream publish failed, falling back to core: %s", exc)
                try:
                    self._nats_conn.publish(self._subject, payload)
                except Exception:
                    self._spool(record)

    def _replay_spool(self) -> None:
        if not self._is_connected():
            return
        files = sorted(self._spool_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)[
            : self._max_spool_files
        ]
        for path in files:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                payload = json.dumps(record).encode("utf-8")
                self._nats_conn.publish_jetstream(self._subject, payload)
                path.unlink()
            except Exception as exc:
                logger.debug("spool replay failed for %s: %s", path.name, exc)
                break

    def _spool(self, record_dict: dict) -> None:
        try:
            existing = list(self._spool_dir.glob("*.json"))
            if len(existing) >= self._max_spool_files:
                existing.sort(key=lambda p: p.stat().st_mtime)
                for old in existing[: len(existing) - self._max_spool_files + 1]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            name = f"{int(time.time() * 1_000_000)}_{id(record_dict)}.json"
            (self._spool_dir / name).write_text(
                json.dumps(record_dict), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("failed to spool record: %s", exc)


_publisher: CallRecordPublisher | None = None
_publisher_lock = threading.Lock()


def _get_publisher() -> CallRecordPublisher:
    global _publisher
    with _publisher_lock:
        if _publisher is None:
            _publisher = CallRecordPublisher()
            _publisher.start()
        return _publisher


def on_success(event: Any) -> None:
    """LiteLLM success callback: build and publish an LlmCallRecord."""
    try:
        record = _build_record(event, outcome="success")
        if record is not None:
            _get_publisher().publish(record)
    except Exception as exc:
        logger.warning("call_records on_success failed (non-fatal): %s", exc)


def on_failure(event: Any) -> None:
    """LiteLLM failure callback: build and publish an LlmCallRecord."""
    try:
        record = _build_record(event, outcome="failure")
        if record is not None:
            _get_publisher().publish(record)
    except Exception as exc:
        logger.warning("call_records on_failure failed (non-fatal): %s", exc)


def _extract_field(source: Any, *names: str) -> Any:
    for name in names:
        if hasattr(source, name):
            val = getattr(source, name)
            if val is not None:
                return val
        if isinstance(source, dict) and name in source:
            val = source[name]
            if val is not None:
                return val
    return None


def _build_record(event: Any, outcome: str) -> dict | None:
    from petrosa_contracts import LlmCallRecord

    model_alias = _extract_field(event, "model", "model_name")
    if not model_alias:
        return None

    now = datetime.now(UTC)
    start_time = _extract_field(event, "start_time")
    end_time = _extract_field(event, "end_time")
    latency_ms = 0.0
    if start_time is not None and end_time is not None:
        try:
            delta = (end_time - start_time).total_seconds() * 1000.0
            latency_ms = max(0.0, delta)
        except Exception:
            pass

    call_outcome = _resolve_outcome(event, outcome)

    metadata = _extract_field(event, "metadata", "custom_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    def meta(*names: str) -> Any:
        for n in names:
            if n in metadata and metadata[n] is not None:
                return metadata[n]
        return _extract_field(event, *names)

    request_id = str(meta("request_id", "litellm_request_id") or "")[:128]
    attempt_no = max(1, min(int(meta("attempt_no", "attempt") or 1), 16))
    caller = str(meta("caller", "user_id", "user") or "unknown")[:64]
    priority_class = str(meta("priority_class", "priority") or "standard")[:32]

    record_kwargs: dict[str, Any] = {
        "request_id": request_id or "unknown",
        "attempt_no": attempt_no,
        "caller": caller,
        "priority_class": priority_class,
        "model_alias": str(model_alias)[:64],
        "started_at": start_time or now,
        "latency_ms": latency_ms,
        "outcome": call_outcome,
    }

    provider_model = _extract_field(event, "provider_model", "actual_model")
    if provider_model:
        record_kwargs["provider_model"] = str(provider_model)[:128]

    queue_wait = meta("queue_wait_ms")
    if queue_wait is not None:
        try:
            record_kwargs["queue_wait_ms"] = max(0.0, float(queue_wait))
        except (TypeError, ValueError):
            pass

    provider_status = _extract_field(event, "status_code", "provider_status")
    if provider_status is not None and outcome != "success":
        try:
            code = int(provider_status)
            if 100 <= code <= 599:
                record_kwargs["provider_status"] = code
        except (TypeError, ValueError):
            pass

    prompt_tokens = _extract_field(event, "prompt_tokens")
    if prompt_tokens is not None:
        try:
            record_kwargs["prompt_tokens"] = max(0, int(prompt_tokens))
        except (TypeError, ValueError):
            pass

    completion_tokens = _extract_field(event, "completion_tokens")
    if completion_tokens is not None:
        try:
            record_kwargs["completion_tokens"] = max(0, int(completion_tokens))
        except (TypeError, ValueError):
            pass

    decision_id = meta("decision_id")
    if decision_id:
        record_kwargs["decision_id"] = str(decision_id)

    investigation_id = meta("investigation_id")
    if investigation_id:
        record_kwargs["investigation_id"] = str(investigation_id)

    prompt_version = meta("prompt_version")
    if prompt_version:
        record_kwargs["prompt_version"] = str(prompt_version)

    try:
        validated = LlmCallRecord(**record_kwargs)
    except Exception as exc:
        logger.warning("LlmCallRecord validation failed: %s", exc)
        return None

    return validated.model_dump(mode="json")


def _resolve_outcome(event: Any, outcome: str) -> Any:
    from petrosa_contracts import LlmCallOutcome

    if outcome == "success":
        return LlmCallOutcome.SUCCESS
    status_code = _extract_field(event, "status_code", "provider_status")
    exception = _extract_field(event, "exception", "error")
    if status_code is not None:
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            code = 0
        if code == 429:
            return LlmCallOutcome.THROTTLED
        if 100 <= code <= 599:
            return LlmCallOutcome.ERROR
    if _is_timeout(exception):
        return LlmCallOutcome.TIMEOUT
    return LlmCallOutcome.ERROR


def _is_timeout(exception: Any) -> bool:
    if exception is None:
        return False
    name = type(exception).__name__.lower()
    text = str(exception).lower()
    return "timeout" in name or "timed out" in text


def start_publisher() -> CallRecordPublisher:
    """Start the call-records publisher (called from the gateway entrypoint)."""
    pub = _get_publisher()
    logger.info(
        "call records publisher started (nats=%s, subject=%s)",
        pub._nats_url,
        pub._subject,
    )
    return pub


def stop_publisher() -> None:
    """Stop the call-records publisher (called on shutdown)."""
    global _publisher
    if _publisher is not None:
        _publisher.stop()
        logger.info("call records publisher stopped")
