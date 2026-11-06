"""The hooks' admission and rate logic (petrosa-llm-gateway#2 and #3), without LiteLLM."""

from __future__ import annotations

import logging

import fakeredis
import pytest
from prometheus_client import CollectorRegistry

from gateway import hooks
from gateway.adaptive_rate import (
    COLD_START_RPM,
    InMemoryRateStore,
    RateState,
    RedisRateStore,
)
from gateway.attempts import DEADLINE_AT_KEY
from gateway.metrics import GatewayMetrics
from gateway.redis_store import RedisStore
from gateway.scheduler import (
    InMemoryStore,
    Scheduler,
    SchedulingConfig,
    SchedulingRefused,
)
from tests.test_scheduler import RAW, VirtualTime, fill_window


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


# --- gw#3: attempts, account throttling and the learned rate --------------------------

FALLBACKS = {"petrosa-primary": ["petrosa-fallback"]}
TRADING = {"caller": "cio", "priority_class": "trading"}


class RateLimitError(Exception):
    """Stands in for litellm.RateLimitError: the name and status LiteLLM gives a 429."""

    status_code = 429

    def __init__(self, headers=None):
        super().__init__("429")
        self.litellm_response_headers = headers or {}


def policy(rates=None, raw=None):
    config = SchedulingConfig.from_mapping(raw or RAW)
    vt = VirtualTime()
    rates = rates or InMemoryRateStore()

    async def learned():
        return (await rates.load()).rpm

    store = InMemoryStore()
    scheduler = Scheduler(config, store, vt.clock, vt.sleep, learned)
    return hooks.GatewayPolicy(scheduler, rates, vt.clock, FALLBACKS), vt, store


def _previous(*types):
    return [{"model_group": "petrosa-primary", "exception_type": t} for t in types]


@pytest.mark.asyncio
async def test_a_request_gets_its_class_deadline_and_a_timeout_that_leaves_room_to_fall_back():
    gateway, vt, store = policy()
    vt.now = 100.0
    data = {"model": "petrosa-primary"}

    await gateway.on_request(TRADING, {"x-petrosa-deadline-ms": "8000"}, data)

    assert data["metadata"][hooks.CLASS_KEY] == "trading"
    assert data["metadata"][DEADLINE_AT_KEY] == 108.0
    assert data["timeout"] == 4.0  # half the budget: the primary, then the fallback
    assert len(store.window) == 1

    plain = {"model": "petrosa-ops"}
    await gateway.on_request(TRADING, None, plain)
    assert "timeout" not in plain and DEADLINE_AT_KEY not in plain["metadata"]


@pytest.mark.asyncio
async def test_every_later_attempt_takes_its_own_slot():
    gateway, vt, store = policy()
    data = {"model": "petrosa-primary"}
    await gateway.on_request(TRADING, None, data)
    metadata = data["metadata"]

    await gateway.on_attempt(metadata)  # the first attempt: already admitted
    assert len(store.window) == 1

    await gateway.on_attempt({**metadata, "previous_models": _previous("Timeout")})
    assert len(store.window) == 2
    assert store.window[-1].cls == "trading"


@pytest.mark.asyncio
async def test_no_attempt_follows_a_429_on_the_same_account():
    gateway, _, store = policy()

    with pytest.raises(SchedulingRefused) as refused:
        await gateway.on_attempt({"previous_models": _previous("RateLimitError")})

    assert refused.value.code == "account_throttled"
    assert store.window == []


@pytest.mark.asyncio
async def test_a_fallback_with_no_budget_left_is_refused():
    gateway, vt, _ = policy()
    vt.now = 50.0

    with pytest.raises(SchedulingRefused) as refused:
        await gateway.on_attempt(
            {DEADLINE_AT_KEY: 49.0, "previous_models": _previous("Timeout")}
        )

    assert refused.value.code == "deadline_unmeetable"


@pytest.mark.asyncio
async def test_a_429_halves_the_rate_and_the_next_request_is_refused_at_once(caplog):
    gateway, vt, store = policy()
    vt.now = 10.0
    headers = {
        "retry-after": "7",
        "x-ratelimit-remaining-requests": "0",
        "authorization": "Bearer never-logged",
    }

    with caplog.at_level(logging.WARNING, logger="gateway.hooks"):
        assert await gateway.on_failure(RateLimitError(headers)) is True

    state = await gateway.rates.load()
    assert state.rpm == COLD_START_RPM / 2 and state.cooldown_until == 17.0
    assert "x-ratelimit-remaining-requests" in caplog.text
    assert "never-logged" not in caplog.text

    with pytest.raises(SchedulingRefused) as refused:
        await gateway.on_request(TRADING, None, {"model": "petrosa-primary"})
    assert refused.value.code == "account_throttled"
    assert store.window == []  # no slot spent, no provider call

    vt.now = 17.0
    await gateway.on_request(TRADING, None, {"model": "petrosa-primary"})
    assert len(store.window) == 1


@pytest.mark.asyncio
async def test_other_failures_leave_the_rate_alone():
    gateway, _, _ = policy()

    assert await gateway.on_failure(TimeoutError()) is False
    assert await gateway.on_failure(None) is False
    assert await gateway.rates.load() == RateState()


@pytest.mark.asyncio
async def test_the_rate_grows_only_while_it_is_the_binding_limit():
    rates = InMemoryRateStore(RateState(rpm=4, tpm=8_000, last_change_at=0.0))
    gateway, vt, store = policy(rates)
    writes = []
    update = rates.update

    async def counting_update(change):
        writes.append(1)
        return await update(change)

    rates.update = counting_update  # type: ignore[method-assign]

    vt.now = 30.0
    await gateway.on_success()  # less than a clean minute: nothing to do, no write
    assert writes == [] and (await rates.load()).rpm == 4

    vt.now = 125.0
    fill_window(store, "ops", 2, at=100.0)  # 2 of 4 used: demand is below the rate
    await gateway.on_success()
    assert writes == [] and (await rates.load()).rpm == 4

    fill_window(store, "trading", 2, at=110.0)  # 4 of 4: the rate is the limit
    await gateway.on_success()
    assert len(writes) == 1 and (await rates.load()).rpm == 5


@pytest.mark.asyncio
async def test_a_cold_record_starts_its_clock_on_the_first_success():
    gateway, vt, _ = policy()
    vt.now = 1_700_000_000.0

    await gateway.on_success()

    state = await gateway.rates.load()
    assert state.rpm == COLD_START_RPM and state.last_change_at == 1_700_000_000.0


@pytest.mark.asyncio
async def test_a_429_on_one_replica_throttles_the_other():
    server = fakeredis.FakeServer()
    one, vt_one, _ = policy(
        RedisRateStore(fakeredis.FakeAsyncRedis(server=server), "gw:rate")
    )
    two, vt_two, _ = policy(
        RedisRateStore(fakeredis.FakeAsyncRedis(server=server), "gw:rate")
    )

    await one.on_failure(RateLimitError({"retry-after": "30"}))

    with pytest.raises(SchedulingRefused) as refused:
        await two.on_request(TRADING, None, {"model": "petrosa-primary"})
    assert refused.value.code == "account_throttled"
    assert (
        await two.scheduler.effective_config()
    ).rate_per_window == COLD_START_RPM / 2


def test_response_headers_and_throttle_detection():
    class WithResponse(Exception):
        response = type("R", (), {"headers": {"retry-after": "3"}})()

    assert hooks.response_headers(WithResponse()) == {"retry-after": "3"}
    assert hooks.response_headers(ValueError()) == {}
    assert hooks.is_throttle(RateLimitError())
    assert not hooks.is_throttle(TimeoutError())


def test_load_fallbacks(tmp_path):
    path = tmp_path / "litellm.yaml"
    path.write_text(
        "router_settings:\n"
        "  fallbacks:\n"
        "    - {petrosa-primary: [petrosa-fallback]}\n"
        "    - {petrosa-ops: [petrosa-fallback]}\n"
    )
    assert hooks.load_fallbacks(path) == {
        "petrosa-primary": ["petrosa-fallback"],
        "petrosa-ops": ["petrosa-fallback"],
    }
    empty = tmp_path / "empty.yaml"
    empty.write_text("model_list: []\n")
    assert hooks.load_fallbacks(empty) == {}


def test_build_policy_without_redis(tmp_path):
    gateway = hooks.build_policy(
        {
            "GATEWAY_SCHEDULING": _config_file(tmp_path),
            "GATEWAY_CONFIG": "config/litellm_config.yaml",
        }
    )

    assert isinstance(gateway.rates, InMemoryRateStore)
    assert gateway.scheduler.rate is not None


def test_build_rates_uses_redis_when_configured():
    rates = hooks.build_rates(
        {"REDIS_URL": "redis://redis.invalid:6379/0", "GATEWAY_REDIS_PREFIX": "gw"}
    )
    assert isinstance(rates, RedisRateStore) and rates.key == "gw:rate"


ALIASES = ["petrosa-primary", "petrosa-fallback", "petrosa-ops"]


def measured_policy():
    registry = CollectorRegistry()
    gateway, vt, store = policy()
    gateway.metrics = GatewayMetrics(registry)
    gateway.aliases = frozenset(ALIASES)
    gateway.callers = frozenset({"cio", "holmes"})
    return gateway, vt, store, registry.get_sample_value


@pytest.mark.asyncio
async def test_metrics_follow_requests_attempts_and_throttles():
    gateway, vt, _, value = measured_policy()
    data = {"model": "petrosa-primary"}

    await gateway.on_request(TRADING, None, data)
    await gateway.on_attempt(data["metadata"], "petrosa-primary")
    await gateway.on_failure(RateLimitError({"retry-after": "5"}), "petrosa-primary")
    with pytest.raises(SchedulingRefused):
        await gateway.on_attempt(
            {**data["metadata"], "previous_models": _previous("RateLimitError")},
            "petrosa-fallback",
        )
    with pytest.raises(SchedulingRefused):
        await gateway.on_request(TRADING, None, {"model": "petrosa-primary"})

    def attempts(alias, outcome):
        labels = {"caller": "cio", "class": "trading", "alias": alias}
        return value("llm_gateway_attempts_total", {**labels, "outcome": outcome})

    assert attempts("petrosa-primary", "started") == 1
    assert attempts("petrosa-fallback", "account_throttled") == 1
    assert attempts("petrosa-primary", "account_throttled") == 1  # refused pre-call
    assert (
        value("llm_gateway_provider_throttles_total", {"alias": "petrosa-primary"}) == 1
    )
    assert value("llm_gateway_rate_rpm") == COLD_START_RPM / 2
    assert value("llm_gateway_cooldown_seconds") == 5.0
    assert value("llm_gateway_queue_depth", {"class": "trading"}) == 0
    started = {"caller": "cio", "class": "trading", "outcome": "started"}
    assert value("llm_gateway_wait_seconds_count", started) == 1


@pytest.mark.asyncio
async def test_metric_labels_stay_bounded():
    gateway, _, _, value = measured_policy()

    await gateway.on_request(
        {"caller": "someone-new", "priority_class": "trading"},
        None,
        {"model": "not-a-petrosa-alias"},
    )
    await gateway.on_attempt({}, "not-a-petrosa-alias")

    assert (
        value(
            "llm_gateway_attempts_total",
            {
                "caller": "gateway",
                "class": "batch",
                "alias": "other",
                "outcome": "started",
            },
        )
        == 1
    )


def test_build_policy_reads_aliases_and_callers_from_the_shipped_configs():
    gateway = hooks.build_policy(
        {
            "GATEWAY_SCHEDULING": "config/scheduling.yaml",
            "GATEWAY_CONFIG": "config/litellm_config.yaml",
        }
    )

    assert gateway.aliases == frozenset(ALIASES)
    assert gateway.callers == frozenset({"cio", "holmes"})
    assert gateway.fallbacks == {
        "petrosa-primary": ["petrosa-fallback"],
        "petrosa-ops": ["petrosa-fallback"],
    }
