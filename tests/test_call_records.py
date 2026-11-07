"""Call records (petrosa-llm-gateway#4): the acceptance criteria as tests."""

import asyncio
import json
import time
from datetime import datetime, timedelta

import pytest

from gateway.call_records import (
    QUEUE_WAIT_KEY,
    RECORD_FIELDS,
    REQUEST_ID_KEY,
    RecordPublisher,
    build_publisher,
    build_record,
    outcome_of,
)

START = datetime(2026, 11, 7, 15, 0, 0)


def _kwargs(previous=(), **metadata):
    meta = {
        REQUEST_ID_KEY: "req-42",
        QUEUE_WAIT_KEY: 0.25,
        "petrosa_caller": "cio",
        "petrosa_class": "trading",
        "decision_id": "dec-1",
        "previous_models": [{"exception_type": e} for e in previous],
        **metadata,
    }
    return {
        "model": "openai/nvidia/nemotron-3.5-super-vl-120b-a12b",
        "messages": [{"role": "user", "content": "SECRET PROMPT"}],
        "litellm_params": {"metadata": meta},
    }


class _Response:
    class usage:  # noqa: N801 - mirrors the LiteLLM attribute
        prompt_tokens = 1200
        completion_tokens = 80

    choices = [{"message": {"content": "SECRET ANSWER"}}]


class Throttle(Exception):
    status_code = 429


# --- records -------------------------------------------------------------------------


def test_a_record_has_exactly_the_contract_fields_and_no_body():
    record = build_record(
        _kwargs(),
        _Response(),
        START,
        START + timedelta(seconds=1.5),
        alias="petrosa-primary",
    )
    assert set(record) == RECORD_FIELDS
    text = json.dumps(record)
    assert "SECRET" not in text
    assert record["prompt_tokens"] == 1200 and record["completion_tokens"] == 80
    assert record["latency_ms"] == 1500.0
    assert record["queue_wait_ms"] == 250.0
    assert record["outcome"] == "success" and record["provider_status"] == 200
    assert record["started_at"].endswith("+00:00")


def test_attempts_and_requests_are_distinct():
    first = build_record(
        _kwargs(), None, START, START, exception=TimeoutError(), alias="petrosa-primary"
    )
    fallback = build_record(
        _kwargs(previous=["Timeout"]),
        _Response(),
        START,
        START,
        alias="petrosa-fallback",
    )
    assert first["request_id"] == fallback["request_id"] == "req-42"
    assert (first["attempt_no"], fallback["attempt_no"]) == (1, 2)
    # The admission wait is counted once, on the first attempt.
    assert first["queue_wait_ms"] == 250.0 and fallback["queue_wait_ms"] is None
    assert first["outcome"] == "timeout" and fallback["outcome"] == "success"


@pytest.mark.parametrize(
    "exception,outcome",
    [
        (None, "success"),
        (Throttle(), "throttled"),
        (TimeoutError(), "timeout"),
        (ValueError(), "error"),
    ],
)
def test_outcomes(exception, outcome):
    assert outcome_of(exception) == outcome


def test_records_validate_against_the_shared_contract():
    import petrosa_contracts as contracts  # pinned in requirements-dev.txt (0.2.0)

    record = build_record(
        _kwargs(previous=["RateLimitError"]),
        None,
        START,
        START,
        exception=Throttle(),
        alias="petrosa-fallback",
    )
    model = contracts.LlmCallRecord.model_validate(record)
    assert set(contracts.LlmCallRecord.model_fields) == RECORD_FIELDS
    assert model.outcome == "throttled" and model.provider_status == 429


# --- publisher -----------------------------------------------------------------------


class FakeNats:
    def __init__(self):
        self.up = True
        self.hang = False
        self.sent = []

    async def send(self, subject, payload):
        if self.hang:
            await asyncio.sleep(3600)
        if not self.up:
            raise ConnectionError("nats down")
        self.sent.append((subject, json.loads(payload)))


async def _drain(publisher):
    for _ in range(200):
        if publisher.queue.empty():
            await asyncio.sleep(0.01)
            if publisher.queue.empty():
                return
        await asyncio.sleep(0.01)


def test_a_nats_outage_never_delays_a_call(tmp_path):
    async def scenario():
        nats = FakeNats()
        nats.hang = True  # NATS accepts the connection and never answers
        publisher = RecordPublisher(
            nats.send, spool_path=tmp_path / "s.jsonl", max_queue=5, send_timeout_s=0.05
        )
        started = time.perf_counter()
        results = [publisher.offer({"n": i}) for i in range(50)]
        elapsed = time.perf_counter() - started
        assert elapsed < 0.05  # 50 offers, no awaiting
        assert results.count(True) == 5 and publisher.dropped == 45
        await _drain(publisher)
        await asyncio.sleep(0.4)
        assert publisher.spooled >= 1  # timed-out publishes went to the spool

    asyncio.run(scenario())


def test_the_spool_replays_after_reconnect_in_order(tmp_path):
    async def scenario():
        nats = FakeNats()
        nats.up = False
        spool = tmp_path / "s.jsonl"
        publisher = RecordPublisher(nats.send, spool_path=spool, send_timeout_s=0.5)
        for i in range(3):
            publisher.offer({"n": i})
        await _drain(publisher)
        assert publisher.spooled == 3 and len(spool.read_text().splitlines()) == 3
        nats.up = True
        publisher.offer({"n": 3})
        await _drain(publisher)
        await asyncio.sleep(0.05)
        assert [p["n"] for _s, p in nats.sent] == [3, 0, 1, 2]
        assert not spool.exists()

    asyncio.run(scenario())


def test_the_spool_is_bounded(tmp_path):
    async def scenario():
        nats = FakeNats()
        nats.up = False
        spool = tmp_path / "s.jsonl"
        publisher = RecordPublisher(nats.send, spool_path=spool, max_spool_bytes=100)
        for i in range(20):
            publisher.offer({"n": i, "pad": "x" * 10})
        await _drain(publisher)
        assert spool.stat().st_size <= 100
        assert publisher.dropped > 0

    asyncio.run(scenario())


def test_records_are_off_without_a_nats_url():
    assert build_publisher({}) is None
    publisher = build_publisher({"GATEWAY_NATS_URL": "nats://nats.example:4222"})
    assert publisher is not None and publisher.subject == "petrosa.llm.call_record.v1"


# --- edge paths ----------------------------------------------------------------------


def test_numeric_and_missing_timestamps():
    record = build_record(_kwargs(), None, 1_794_000_000.0, 1_794_000_002.5)
    assert record["started_at"].startswith("2026-") and record["started_at"].endswith(
        "+00:00"
    )
    assert record["latency_ms"] == 2500.0
    unknown = build_record(_kwargs(), None, None, None)
    assert unknown["latency_ms"] == 0.0 and unknown["started_at"].endswith("+00:00")


def test_offer_outside_a_running_loop_only_buffers(tmp_path):
    publisher = RecordPublisher(FakeNats().send, spool_path=tmp_path / "s.jsonl")
    assert publisher.offer({"n": 1}) is True  # no event loop: buffered, no task started
    assert publisher.queue.qsize() == 1 and publisher._task is None


def test_a_spool_write_failure_drops_and_counts(tmp_path):
    publisher = RecordPublisher(FakeNats().send, spool_path=tmp_path)  # a directory
    publisher._spool({"n": 1})
    assert publisher.dropped == 1 and publisher.spooled == 0


def test_replay_keeps_the_rest_after_a_failure(tmp_path):
    async def scenario():
        spool = tmp_path / "s.jsonl"
        sent = []

        async def flaky(subject, payload):
            n = json.loads(payload)["n"]
            if n == 1:
                raise ConnectionError("down again")
            sent.append(n)

        publisher = RecordPublisher(flaky, spool_path=spool)
        assert await publisher.replay_spool() == 0  # nothing spooled yet
        spool.write_text("".join(json.dumps({"n": i}) + "\n" for i in range(3)))
        assert await publisher.replay_spool() == 1
        assert sent == [0]
        assert [json.loads(line)["n"] for line in spool.read_text().splitlines()] == [
            1,
            2,
        ]

    asyncio.run(scenario())


class _FakeNatsClient:
    def __init__(self, closed=False):
        self.is_closed = closed
        self.published = []

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


def _fake_nats(monkeypatch, closed):
    import sys
    import types

    connects = []

    async def connect(**kwargs):
        connects.append(kwargs)
        return _FakeNatsClient(closed=closed)

    monkeypatch.setitem(sys.modules, "nats", types.SimpleNamespace(connect=connect))
    return connects


def test_nats_sender_connects_lazily_once_and_parses_servers(monkeypatch):
    from gateway.call_records import nats_sender

    connects = _fake_nats(monkeypatch, closed=False)
    send = nats_sender("nats://a:4222, nats://b:4222,")

    async def scenario():
        await send("s", b"1")
        await send("s", b"2")

    asyncio.run(scenario())
    assert len(connects) == 1  # the open client is reused
    assert connects[0]["servers"] == ["nats://a:4222", "nats://b:4222"]
    assert connects[0]["max_reconnect_attempts"] == -1


def test_nats_sender_reconnects_when_the_client_is_closed(monkeypatch):
    from gateway.call_records import nats_sender

    connects = _fake_nats(monkeypatch, closed=True)
    send = nats_sender("nats://c:4222")

    async def scenario():
        await send("s", b"x")
        await send("s", b"y")

    asyncio.run(scenario())
    assert len(connects) == 2 and connects[0]["servers"] == ["nats://c:4222"]
