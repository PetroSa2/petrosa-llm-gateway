"""Call records: one `LlmCallRecord` per provider attempt, published on NATS (petrosa-llm-gateway#4).

Never on the hot path. A LiteLLM success/failure event only builds a small dict and hands it to
`RecordPublisher.offer`, which never awaits and never blocks: a full in-memory buffer drops the
record and counts it. A background task publishes; when NATS is down or slow it writes records
to a bounded on-disk spool (JSON lines) and replays the spool, oldest first, once a publish works
again.

A record never carries a prompt, message, completion or response body: `build_record` reads only
metadata, timings, status and token counts. The schema is `petrosa_contracts.LlmCallRecord`
(petrosa-contracts >= 0.2.0); `RECORD_FIELDS` mirrors it and a contract test keeps them equal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gateway.attempts import THROTTLE_EXCEPTIONS, previous_failures

logger = logging.getLogger("gateway.call_records")

#: Request-metadata keys the pre-call hook sets once per request.
REQUEST_ID_KEY = "petrosa_request_id"
QUEUE_WAIT_KEY = "petrosa_queue_wait_s"
#: Correlation keys a caller may put in its request `metadata`.
CORRELATION_KEYS = ("decision_id", "investigation_id", "prompt_version")

DEFAULT_SUBJECT = "petrosa.llm.call_record.v1"
DEFAULT_SPOOL_PATH = "/var/spool/llm-gateway/call-records.jsonl"
DEFAULT_SPOOL_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_QUEUE_SIZE = 1000

#: The fields of petrosa_contracts.LlmCallRecord v1 (a contract test keeps them equal).
RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "attempt_no",
        "caller",
        "priority_class",
        "model_alias",
        "provider_model",
        "decision_id",
        "investigation_id",
        "prompt_version",
        "started_at",
        "queue_wait_ms",
        "latency_ms",
        "outcome",
        "provider_status",
        "prompt_tokens",
        "completion_tokens",
    }
)
_MAX_LEN = {
    "request_id": 128,
    "caller": 64,
    "priority_class": 32,
    "model_alias": 64,
    "provider_model": 128,
    "decision_id": 128,
    "investigation_id": 128,
    "prompt_version": 64,
}


def request_metadata(kwargs: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The request metadata of a LiteLLM logging event (where the pre-call hook wrote ours)."""
    kwargs = kwargs or {}
    nested = (kwargs.get("litellm_params") or {}).get("metadata")
    return nested if isinstance(nested, Mapping) else kwargs.get("metadata") or {}


def outcome_of(exception: BaseException | None) -> str:
    """`success`, `throttled` (provider 429), `timeout` or `error`."""
    if exception is None:
        return "success"
    status = getattr(exception, "status_code", None)
    name = type(exception).__name__
    if status == 429 or name in THROTTLE_EXCEPTIONS:
        return "throttled"
    if status in (408, 504) or "timeout" in name.lower():
        return "timeout"
    return "error"


def _text(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    return str(value)[: _MAX_LEN[field]]


def _utc(moment: Any) -> datetime:
    """LiteLLM passes naive local datetimes; records carry aware UTC."""
    if isinstance(moment, int | float):
        return datetime.fromtimestamp(moment, UTC)
    if isinstance(moment, datetime):
        return moment.astimezone(UTC)
    return datetime.now(UTC)


def _seconds(start: Any, end: Any) -> float:
    if isinstance(start, datetime) and isinstance(end, datetime):
        return max(0.0, (end - start).total_seconds())
    if isinstance(start, int | float) and isinstance(end, int | float):
        return max(0.0, float(end) - float(start))
    return 0.0


def _usage(response_obj: Any) -> tuple[int | None, int | None]:
    usage = (
        response_obj.get("usage")
        if isinstance(response_obj, Mapping)
        else getattr(response_obj, "usage", None)
    )
    if usage is None:
        return None, None

    def count(name: str) -> int | None:
        value = (
            usage.get(name)
            if isinstance(usage, Mapping)
            else getattr(usage, name, None)
        )
        return value if isinstance(value, int) and value >= 0 else None

    return count("prompt_tokens"), count("completion_tokens")


def _provider_status(exception: BaseException | None) -> int | None:
    if exception is None:
        return 200
    status = getattr(exception, "status_code", None)
    return status if isinstance(status, int) and 100 <= status <= 599 else None


def build_record(
    kwargs: Mapping[str, Any] | None,
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    exception: BaseException | None = None,
    alias: str | None = None,
) -> dict[str, Any]:
    """One attempt's record from a LiteLLM logging event: metadata and counts, never a body."""
    kwargs = kwargs or {}
    metadata = request_metadata(kwargs)
    attempt_no = min(16, 1 + len(previous_failures(metadata)))
    request_id = metadata.get(REQUEST_ID_KEY) or kwargs.get("litellm_call_id")
    queue_wait = metadata.get(QUEUE_WAIT_KEY)
    prompt_tokens, completion_tokens = _usage(response_obj)
    record: dict[str, Any] = {
        "schema_version": "v1",
        "request_id": _text(request_id, "request_id") or uuid.uuid4().hex,
        "attempt_no": attempt_no,
        "caller": _text(metadata.get("petrosa_caller"), "caller") or "gateway",
        "priority_class": _text(metadata.get("petrosa_class"), "priority_class")
        or "batch",
        "model_alias": _text(alias, "model_alias") or "other",
        "provider_model": _text(kwargs.get("model"), "provider_model"),
        "started_at": _utc(start_time).isoformat(),
        # The admission wait belongs to the first attempt; a fallback's own wait is not timed.
        "queue_wait_ms": (
            round(float(queue_wait) * 1000.0, 3)
            if attempt_no == 1 and isinstance(queue_wait, int | float)
            else None
        ),
        "latency_ms": round(_seconds(start_time, end_time) * 1000.0, 3),
        "outcome": outcome_of(exception),
        "provider_status": _provider_status(exception),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    for key in CORRELATION_KEYS:
        record[key] = _text(metadata.get(key), key)
    return record


Sender = Callable[[str, bytes], Awaitable[None]]


class RecordPublisher:
    """Bounded buffer -> NATS, with a bounded on-disk spool while NATS is unavailable."""

    def __init__(
        self,
        send: Sender,
        subject: str = DEFAULT_SUBJECT,
        spool_path: str | Path = DEFAULT_SPOOL_PATH,
        max_queue: int = DEFAULT_QUEUE_SIZE,
        max_spool_bytes: int = DEFAULT_SPOOL_MAX_BYTES,
        send_timeout_s: float = 2.0,
    ) -> None:
        self._send = send
        self.subject = subject
        self.spool_path = Path(spool_path)
        self.max_spool_bytes = max_spool_bytes
        self.send_timeout_s = send_timeout_s
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue)
        self.dropped = 0
        self.published = 0
        self.spooled = 0
        self._task: asyncio.Task[None] | None = None

    def offer(self, record: dict[str, Any]) -> bool:
        """Hand a record over without waiting; False when the buffer was full (dropped)."""
        self._ensure_running()
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1
            return False
        return True

    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._task = loop.create_task(self.run())

    async def _publish(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record, separators=(",", ":")).encode()
        await asyncio.wait_for(self._send(self.subject, payload), self.send_timeout_s)

    def _spool(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":")) + "\n"
        try:
            size = self.spool_path.stat().st_size if self.spool_path.exists() else 0
            if size + len(line) > self.max_spool_bytes:
                self.dropped += 1
                return
            self.spool_path.parent.mkdir(parents=True, exist_ok=True)
            with self.spool_path.open("a", encoding="utf-8") as spool:
                spool.write(line)
            self.spooled += 1
        except OSError as error:
            self.dropped += 1
            logger.warning("call record spool write failed: %s", error)

    async def replay_spool(self) -> int:
        """Publish spooled records oldest first; keep the rest on the first failure."""
        if not self.spool_path.exists():
            return 0
        lines = self.spool_path.read_text(encoding="utf-8").splitlines()
        sent = 0
        for line in lines:
            try:
                await self._publish(json.loads(line))
            except Exception:  # noqa: BLE001 - any failure keeps the rest spooled
                break
            sent += 1
        rest = lines[sent:]
        if rest:
            self.spool_path.write_text("\n".join(rest) + "\n", encoding="utf-8")
        else:
            self.spool_path.unlink(missing_ok=True)
        self.published += sent
        return sent

    async def run(self) -> None:
        """Publish forever: replay the spool after a success, spool on a failure."""
        while True:
            record = await self.queue.get()
            try:
                await self._publish(record)
            except Exception as error:  # noqa: BLE001 - NATS down, slow or refusing
                logger.debug("call record publish failed, spooling: %s", error)
                self._spool(record)
                continue
            self.published += 1
            if self.spool_path.exists():
                await self.replay_spool()


def nats_sender(servers: str) -> Sender:
    """A sender over nats-py with its own reconnects; connects lazily on first use."""
    state: dict[str, Any] = {}

    async def send(subject: str, payload: bytes) -> None:
        import nats  # in the image (requirements-image.txt); tests use fake senders

        client = state.get("client")
        if client is None or client.is_closed:
            client = await nats.connect(
                servers=[s.strip() for s in servers.split(",") if s.strip()],
                name="petrosa-llm-gateway",
                connect_timeout=2,
                allow_reconnect=True,
                max_reconnect_attempts=-1,
            )
            state["client"] = client
        await client.publish(subject, payload)

    return send


def build_publisher(environ: Mapping[str, str] | None = None) -> RecordPublisher | None:
    """The publisher from the environment; None (records off) without `GATEWAY_NATS_URL`."""
    environ = os.environ if environ is None else environ
    servers = environ.get("GATEWAY_NATS_URL", "").strip()
    if not servers:
        logger.warning("GATEWAY_NATS_URL is not set: call records are off")
        return None
    return RecordPublisher(
        nats_sender(servers),
        subject=environ.get("GATEWAY_CALL_RECORD_SUBJECT", DEFAULT_SUBJECT),
        spool_path=environ.get("GATEWAY_CALL_RECORD_SPOOL", DEFAULT_SPOOL_PATH),
        max_spool_bytes=int(
            environ.get("GATEWAY_CALL_RECORD_SPOOL_MAX_BYTES", DEFAULT_SPOOL_MAX_BYTES)
        ),
    )
