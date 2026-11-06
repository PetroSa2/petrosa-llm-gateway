"""Adaptive provider rate (petrosa-llm-gateway#3): AIMD, cool-down, shared state."""

from __future__ import annotations

import fakeredis
import pytest

from gateway.adaptive_rate import (
    COLD_START_RPM,
    COLD_START_TPM,
    DEFAULT_COOLDOWN_S,
    MIN_RPM,
    InMemoryRateStore,
    RateState,
    RedisRateStore,
    on_clean,
    on_throttled,
    rate_headers,
    retry_after_seconds,
    throttled,
)


def test_the_cold_start_is_the_labelled_fallback():
    state = RateState()

    assert (state.rpm, state.tpm) == (COLD_START_RPM, COLD_START_TPM)
    assert not throttled(state, 0.0)


def test_a_429_halves_the_rate_and_it_recovers_by_one_per_clean_minute():
    state = on_throttled(RateState(rpm=12, tpm=24_000), now=100.0, retry_after_s=30)

    assert (state.rpm, state.tpm) == (6, 12_000)
    assert throttled(state, 129.0) and not throttled(state, 130.0)
    assert on_clean(state, 129.0) == state  # no growth during the cool-down
    assert on_clean(state, 189.0) == state  # less than one clean minute after it
    one = on_clean(state, 190.0)
    assert (one.rpm, one.tpm) == (7, 14_000)
    assert on_clean(one, 249.0) == one  # the next step needs another clean minute
    two = on_clean(one, 250.0)
    assert two.rpm == 8
    assert on_clean(two, 250.0) == two  # idempotent within the minute


def test_idle_minutes_add_nothing_and_a_cold_record_only_starts_the_clock():
    """A quiet hour is not evidence the account takes more (smoke test, 2026-11-06: a cold
    record counted every minute since the epoch as clean)."""
    cold = RateState()
    started = on_clean(cold, 1_700_000_000.0)
    assert started.rpm == COLD_START_RPM and started.last_change_at == 1_700_000_000.0

    after_an_idle_hour = on_clean(started, 1_700_003_600.0)
    assert after_an_idle_hour.rpm == COLD_START_RPM + 1  # one step, not sixty

    not_binding = on_clean(after_an_idle_hour, 1_700_003_700.0, saturated=False)
    assert not_binding == after_an_idle_hour  # the rate was not the limit: no growth


def test_repeated_429s_never_go_below_the_floor():
    state = RateState(rpm=2, tpm=4_000)
    for i in range(5):
        state = on_throttled(state, now=float(i), retry_after_s=None)

    assert state.rpm == MIN_RPM
    assert state.throttles == 5
    assert state.cooldown_until == 4.0 + DEFAULT_COOLDOWN_S


def test_a_later_shorter_retry_after_does_not_shorten_the_cool_down():
    state = on_throttled(RateState(), now=0.0, retry_after_s=120)
    state = on_throttled(state, now=10.0, retry_after_s=5)

    assert state.cooldown_until == 120.0


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "7"}, 7.0),
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "soon"}, None),
        ({"retry-after": "0"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_retry_after(headers, expected):
    assert retry_after_seconds(headers) == expected


def test_only_rate_headers_are_kept_for_logging():
    headers = {
        "Retry-After": "7",
        "x-ratelimit-remaining-requests": "0",
        "X-RateLimit-Limit-Tokens": "40000",
        "authorization": "Bearer secret",
        "content-type": "application/json",
    }

    assert rate_headers(headers) == {
        "retry-after": "7",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-limit-tokens": "40000",
    }


def test_state_round_trips_and_empty_is_cold_start():
    state = on_throttled(RateState(), now=5.0, retry_after_s=10)

    assert RateState.from_json(state.to_json()) == state
    assert RateState.from_json(None) == RateState()


@pytest.mark.asyncio
async def test_in_memory_store_applies_changes():
    store = InMemoryRateStore()

    await store.update(lambda s: on_throttled(s, 0.0, 30))

    assert (await store.load()).rpm == COLD_START_RPM / 2


@pytest.mark.asyncio
async def test_two_replicas_share_the_state_and_it_survives_a_restart():
    server = fakeredis.FakeServer()
    one = RedisRateStore(fakeredis.FakeAsyncRedis(server=server), "gw:rate")
    two = RedisRateStore(fakeredis.FakeAsyncRedis(server=server), "gw:rate")

    await one.update(lambda s: on_throttled(s, 0.0, 30))
    seen_by_two = await two.load()
    await two.update(
        lambda s: on_clean(s, 90.0)
    )  # one clean minute after the cool-down

    restarted = RedisRateStore(fakeredis.FakeAsyncRedis(server=server), "gw:rate")
    after_restart = await restarted.load()
    assert seen_by_two.rpm == COLD_START_RPM / 2
    assert throttled(seen_by_two, 10.0)
    assert after_restart.rpm == COLD_START_RPM / 2 + 1
    assert after_restart.throttles == 1


@pytest.mark.asyncio
async def test_conflicting_updates_are_retried_then_given_up(monkeypatch):
    from redis.exceptions import WatchError

    redis = fakeredis.FakeAsyncRedis()
    store = RedisRateStore(redis, "gw:rate", retries=3)
    real_pipeline = redis.pipeline
    attempts: list[int] = []
    conflicts = {"left": 2}

    def pipeline(*args, **kwargs):
        pipe = real_pipeline(*args, **kwargs)
        execute = pipe.execute

        async def conflicting_execute(*eargs, **ekwargs):
            attempts.append(1)
            if conflicts["left"] > 0:
                conflicts["left"] -= 1
                raise WatchError("another replica wrote")
            return await execute(*eargs, **ekwargs)

        pipe.execute = conflicting_execute
        return pipe

    monkeypatch.setattr(redis, "pipeline", pipeline)

    state = await store.update(lambda s: on_throttled(s, 0.0, 30))
    assert len(attempts) == 3 and state.throttles == 1

    conflicts["left"] = 10
    with pytest.raises(RuntimeError):
        await store.update(lambda s: on_throttled(s, 1.0, 30))


def test_no_headers_means_nothing_to_log():
    assert rate_headers(None) == {}
    assert rate_headers({}) == {}
