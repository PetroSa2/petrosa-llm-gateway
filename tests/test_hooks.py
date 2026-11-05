"""The pre-call hook's admission logic (petrosa-llm-gateway#2), without LiteLLM."""

from __future__ import annotations

import pytest

from gateway import hooks
from gateway.redis_store import RedisStore
from gateway.scheduler import (
    InMemoryStore,
    Scheduler,
    SchedulingConfig,
    SchedulingRefused,
)
from tests.test_scheduler import RAW, VirtualTime


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-petrosa-deadline-ms": "2500"}, 2.5),
        ({"X-Petrosa-Deadline-Ms": 800}, 0.8),
        ({"x-petrosa-deadline-ms": "0"}, None),
        ({"x-petrosa-deadline-ms": "-5"}, None),
        ({"x-petrosa-deadline-ms": "soon"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_budget_from_headers(headers, expected):
    assert hooks.budget_from_headers(headers) == expected


def test_request_class_comes_from_the_auth_metadata():
    config = SchedulingConfig.from_mapping(RAW)

    assert hooks.request_class({"priority_class": "trading"}, config) == "trading"
    assert hooks.request_class({"priority_class": "nope"}, config) == "batch"
    assert hooks.request_class(None, config) == "batch"


@pytest.mark.asyncio
async def test_admit_request_uses_class_and_deadline():
    config = SchedulingConfig.from_mapping({**RAW, "rate_per_window": 1})
    vt = VirtualTime()
    scheduler = Scheduler(config, InMemoryStore(), vt.clock, vt.sleep)

    ticket = await hooks.admit_request(
        scheduler, {"caller": "cio", "priority_class": "trading"}, None
    )
    assert ticket.cls == "trading"

    with pytest.raises(SchedulingRefused) as refused:
        await hooks.admit_request(
            scheduler,
            {"caller": "cio", "priority_class": "trading"},
            {"x-petrosa-deadline-ms": "100"},
        )
    assert refused.value.code == "deadline_unmeetable"


def _config_file(tmp_path):
    path = tmp_path / "scheduling.yaml"
    path.write_text(
        "classes:\n"
        "  trading: {rank: 0, max_queue: 5, max_age_s: 30}\n"
        "  batch: {rank: 1, max_queue: 5, max_age_s: 60}\n"
        "callers: {cio: trading}\n"
        "rate_per_window: 10\nwindow_s: 60\nreserved_class: trading\n"
        "reserved_share: 0.5\nrank_gap_s: 60\npoll_s: 0.25\n"
    )
    return str(path)


def test_build_scheduler_is_local_without_redis(tmp_path):
    scheduler = hooks.build_scheduler({"GATEWAY_SCHEDULING": _config_file(tmp_path)})

    assert isinstance(scheduler.store, InMemoryStore)
    assert scheduler.config.rate_per_window == 10


def test_build_scheduler_shares_state_in_redis_when_configured(tmp_path):
    scheduler = hooks.build_scheduler(
        {
            "GATEWAY_SCHEDULING": _config_file(tmp_path),
            "REDIS_URL": "redis://redis.invalid:6379/0",
            "GATEWAY_REDIS_PREFIX": "gw",
        }
    )

    assert isinstance(scheduler.store, RedisStore)
    assert scheduler.store.queue_key == "gw:queue"
    assert scheduler.store.stale_after_s == hooks.STALE_AFTER_POLLS * 0.25


def test_the_shipped_config_loads():
    scheduler = hooks.build_scheduler({"GATEWAY_SCHEDULING": "config/scheduling.yaml"})

    assert set(scheduler.config.classes) == {"trading", "ops", "batch"}
    assert scheduler.config.reserved_class == "trading"
