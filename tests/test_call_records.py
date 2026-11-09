"""Tests for gateway.call_records: NATS publishing, spool replay, schema safety."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gateway import call_records


class FakeNatsConn:
    """Minimal in-memory stand-in for a NATS JetStream connection."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def publish_jetstream(self, subject: str, payload: bytes) -> None:
        if not self.connected:
            raise ConnectionError("not connected")
        self.published.append((subject, payload))

    def publish(self, subject: str, payload: bytes) -> None:
        if not self.connected:
            raise ConnectionError("not connected")
        self.published.append((subject, payload))

    def flush(self, timeout: int = 0) -> None:
        pass

    def close(self) -> None:
        self.connected = False


def _make_event(
    model: str = "petrosa-primary",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    metadata: dict | None = None,
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 50,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "model": model,
        "start_time": start_time or (now - __import__("datetime").timedelta(seconds=1)),
        "end_time": end_time or now,
        "metadata": metadata or {},
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


class TestPublisherNonBlocking:
    """NATS outage never delays a call."""

    def test_publish_never_blocks_when_buffer_full(self, tmp_path):
        conn = FakeNatsConn()
        pub = call_records.CallRecordPublisher(
            nats_url="nats://fake",
            subject="petrosa.llm.call_record.v1",
            buffer_size=5,
            max_spool_files=100,
            spool_dir=tmp_path / "spool",
            connect_fn=lambda: conn,
        )

        for i in range(20):
            start = time.monotonic()
            pub.publish({"request_id": f"req-{i}", "attempt_no": 1})
            elapsed = time.monotonic() - start
            assert elapsed < 0.1, f"publish blocked for {elapsed:.3f}s"

    def test_publish_with_no_nats_connection_does_not_raise(self, tmp_path):
        pub = call_records.CallRecordPublisher(
            nats_url="nats://unreachable:9999",
            subject="petrosa.llm.call_record.v1",
            buffer_size=3,
            max_spool_files=50,
            spool_dir=tmp_path / "spool",
            connect_fn=lambda: (_ for _ in ()).throw(ConnectionError("nope")),
        )

        pub.publish({"request_id": "r1", "attempt_no": 1})
        assert len(pub._buffer) == 1


class TestSpoolReplay:
    """Spooled records replay after NATS reconnects."""

    def test_spool_replays_on_reconnect(self, tmp_path):
        conn = FakeNatsConn()
        spool_dir = tmp_path / "spool"
        spool_dir.mkdir(parents=True)

        pub = call_records.CallRecordPublisher(
            nats_url="nats://fake",
            subject="petrosa.llm.call_record.v1",
            buffer_size=2,
            max_spool_files=100,
            spool_dir=spool_dir,
            connect_fn=lambda: conn,
        )

        pub._spool({"request_id": "s1", "attempt_no": 1})
        pub._spool({"request_id": "s2", "attempt_no": 2})
        assert len(list(spool_dir.glob("*.json"))) == 2

        conn.connected = True
        pub._nats_conn = conn
        pub._replay_spool()

        assert len(conn.published) == 2
        subjects = [s for s, _ in conn.published]
        assert all(s == "petrosa.llm.call_record.v1" for s in subjects)
        payloads = [json.loads(p) for _, p in conn.published]
        ids = {p["request_id"] for p in payloads}
        assert ids == {"s1", "s2"}
        assert len(list(spool_dir.glob("*.json"))) == 0

    def test_spool_bounded_evicts_oldest(self, tmp_path):
        pub = call_records.CallRecordPublisher(
            nats_url="nats://fake",
            subject="petrosa.llm.call_record.v1",
            buffer_size=1,
            max_spool_files=3,
            spool_dir=tmp_path / "spool",
            connect_fn=lambda: FakeNatsConn(),
        )

        for i in range(5):
            pub._spool({"request_id": f"r{i}", "attempt_no": 1})

        files = list(pub._spool_dir.glob("*.json"))
        assert len(files) <= 3


class TestNoBodyFields:
    """The record schema forbids prompt and response bodies."""

    def test_schema_rejects_prompt_field(self):
        from petrosa_contracts import LlmCallRecord

        with pytest.raises(Exception, match="Extra inputs"):
            LlmCallRecord(
                request_id="r1",
                attempt_no=1,
                caller="test",
                priority_class="standard",
                model_alias="petrosa-primary",
                started_at=datetime.now(UTC),
                latency_ms=100.0,
                outcome=__import__("petrosa_contracts").LlmCallOutcome.SUCCESS,
                prompt="secret user prompt text",
            )

    def test_schema_rejects_response_field(self):
        from petrosa_contracts import LlmCallOutcome, LlmCallRecord

        with pytest.raises(Exception, match="Extra inputs"):
            LlmCallRecord(
                request_id="r1",
                attempt_no=1,
                caller="test",
                priority_class="standard",
                model_alias="petrosa-primary",
                started_at=datetime.now(UTC),
                latency_ms=100.0,
                outcome=LlmCallOutcome.SUCCESS,
                response_body={"choices": []},
            )

    def test_build_record_never_includes_bodies(self):
        event = _make_event(metadata={})
        record = call_records._build_record(event, outcome="success")
        assert record is not None
        assert "prompt" not in record
        assert "response" not in record
        assert "completion" not in record
        assert "messages" not in record


class TestAttemptsDistinct:
    """Multiple attempts of the same request produce distinct valid records."""

    def test_same_request_different_attempts(self):
        from petrosa_contracts import LlmCallOutcome, LlmCallRecord

        base = {
            "request_id": "req-123",
            "caller": "cio",
            "priority_class": "high",
            "model_alias": "petrosa-primary",
            "started_at": datetime.now(UTC),
            "latency_ms": 50.0,
            "outcome": LlmCallOutcome.SUCCESS,
        }

        rec1 = LlmCallRecord(attempt_no=1, **base)
        rec2 = LlmCallRecord(attempt_no=2, **base)

        assert rec1.attempt_no == 1
        assert rec2.attempt_no == 2
        assert rec1.request_id == rec2.request_id == "req-123"
        dump1 = rec1.model_dump(mode="json")
        dump2 = rec2.model_dump(mode="json")
        assert dump1 != dump2

    def test_build_record_preserves_attempt_no(self):
        event = _make_event(metadata={"request_id": "req-42", "attempt_no": 3})
        record = call_records._build_record(event, outcome="success")
        assert record is not None
        assert record["request_id"] == "req-42"
        assert record["attempt_no"] == 3


class TestBuildRecord:
    """Field extraction from LiteLLM callback events."""

    def test_success_event(self):
        event = _make_event(
            model="petrosa-primary",
            metadata={"request_id": "r-1", "caller": "cio"},
            prompt_tokens=200,
            completion_tokens=80,
        )
        record = call_records._build_record(event, outcome="success")
        assert record is not None
        assert record["model_alias"] == "petrosa-primary"
        assert record["request_id"] == "r-1"
        assert record["caller"] == "cio"
        assert record["outcome"] == "success"
        assert record["prompt_tokens"] == 200
        assert record["completion_tokens"] == 80
        assert record["latency_ms"] >= 0.0

    def test_failure_event_with_status(self):
        event = _make_event(
            model="petrosa-fallback",
            metadata={"request_id": "r-2"},
        )
        event["status_code"] = 429
        record = call_records._build_record(event, outcome="failure")
        assert record is not None
        assert record["outcome"] == "throttled"
        assert record["provider_status"] == 429

    def test_failure_event_timeout(self):
        event = _make_event(
            model="petrosa-primary",
            metadata={"request_id": "r-3"},
        )
        event["exception"] = TimeoutError("request timed out")
        record = call_records._build_record(event, outcome="failure")
        assert record is not None
        assert record["outcome"] == "timeout"

    def test_missing_model_returns_none(self):
        record = call_records._build_record({"foo": "bar"}, outcome="success")
        assert record is None


class TestCallbacks:
    """on_success / on_failure never raise."""

    def test_on_success_with_valid_event(self, tmp_path):
        with patch.object(call_records, "_get_publisher") as mock_get:
            pub = MagicMock()
            mock_get.return_value = pub
            event = _make_event(metadata={"request_id": "cb-1"})
            call_records.on_success(event)
            assert pub.publish.called

    def test_on_failure_with_broken_event_does_not_raise(self):
        call_records.on_failure(None)
        call_records.on_failure("not a dict")
        call_records.on_success(object())


class TestLifecycle:
    """start_publisher / stop_publisher manage the singleton."""

    def test_start_and_stop(self, tmp_path):
        with patch.object(call_records.CallRecordPublisher, "stop") as mock_stop:
            pub = call_records.start_publisher()
            assert isinstance(pub, call_records.CallRecordPublisher)
            call_records.stop_publisher()
            mock_stop.assert_called_once()
