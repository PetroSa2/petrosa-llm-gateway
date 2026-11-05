"""Two gateway replicas sharing Redis act as one scheduler (petrosa-llm-gateway#2)."""

from __future__ import annotations

import asyncio

import fakeredis
import pytest

from gateway.redis_store import RedisStore
from gateway.scheduler import (
    Scheduler,
    SchedulingConfig,
    SchedulingRefused,
    Ticket,
    queue_key,
)
from tests.test_scheduler import RAW, VirtualTime

LONG_AGES = {
    "classes": {
        name: {**spec, "max_age_s": 10_000} for name, spec in RAW["classes"].items()
    }
}


def replicas(raw, count=2, stale_after_s=5.0):
    """``count`` schedulers, each with its own Redis client on one shared server."""
    config = SchedulingConfig.from_mapping(raw)
    vt = VirtualTime()
    server = fakeredis.FakeServer()
    schedulers = [
        Scheduler(
            config,
            RedisStore(
                fakeredis.FakeAsyncRedis(server=server), "gw-test", stale_after_s
            ),
            vt.clock,
            vt.sleep,
        )
        for _ in range(count)
    ]
    return config, vt, schedulers


@pytest.mark.asyncio
async def test_two_replicas_never_exceed_the_configured_rate():
    config, vt, (one, two) = replicas({**RAW, **LONG_AGES})
    starts: list[tuple[float, str]] = []

    async def call(scheduler, cls, name):
        await scheduler.admit(cls)
        starts.append((vt.now, name))

    tasks = [
        asyncio.create_task(call(one if i % 2 else two, cls, f"{cls}{i}"))
        for i in range(24)
        for cls in ("trading", "ops")
    ]
    await vt.run_until(lambda: len(starts) == len(tasks), limit_s=2000.0)

    assert len(starts) == len(tasks)
    times = [at for at, _ in starts]
    for at in times:
        in_window = [s for s in times if at - config.window_s < s <= at]
        assert len(in_window) <= config.rate_per_window
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_trading_on_one_replica_goes_before_ops_on_the_other():
    config, vt, (one, two) = replicas({**RAW, "rate_per_window": 1, "window_s": 10})
    await one.admit("batch")  # fills the window until t=10
    order: list[str] = []

    async def call(scheduler, cls):
        await scheduler.admit(cls)
        order.append(cls)

    ops = asyncio.create_task(call(two, "ops"))
    await vt.settle()
    vt.now = 1.0
    trading = asyncio.create_task(call(one, "trading"))
    await vt.run_until(lambda: len(order) == 2, limit_s=100.0)

    assert order == ["trading", "ops"]
    await asyncio.gather(ops, trading)


@pytest.mark.asyncio
async def test_queue_limit_counts_both_replicas():
    raw = {**RAW, "rate_per_window": 1, "window_s": 1000}
    config, vt, (one, two) = replicas(raw)
    await one.admit("trading")  # window full for the rest of the test
    waiters = [
        asyncio.create_task((one if i % 2 else two).admit("batch")) for i in range(5)
    ]
    await vt.settle()

    with pytest.raises(SchedulingRefused) as refused:
        await one.admit("batch")

    assert refused.value.code == "queue_full"
    for waiter in waiters:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    assert await one.store.redis.hlen("gw-test:queue") == 0


@pytest.mark.asyncio
async def test_a_dead_replicas_ticket_does_not_block_the_queue():
    config, vt, (one,) = replicas(RAW, count=1, stale_after_s=5.0)
    ghost = Ticket("ghost", "trading", 0.0, None, queue_key(config, "trading", 0.0))
    await one.store.enqueue(config, ghost, 0.0)  # its replica never polls again
    vt.now = 1.0

    started = asyncio.create_task(one.admit("ops"))
    await vt.run_until(started.done, limit_s=100.0)

    ticket = await started
    assert ticket.cls == "ops"
    assert vt.now >= 5.0  # served once the ghost went stale, not before
    assert await one.store.redis.hexists("gw-test:queue", "ghost") is False


@pytest.mark.asyncio
async def test_unmeetable_deadline_is_refused_across_replicas():
    config, vt, (one, two) = replicas({**RAW, "rate_per_window": 1, "window_s": 60})
    await one.admit("trading")
    queued = asyncio.create_task(two.admit("trading"))
    await vt.settle()

    with pytest.raises(SchedulingRefused) as refused:
        await one.admit("trading", budget_s=1.0)

    assert refused.value.code == "deadline_unmeetable"
    queued.cancel()
    await asyncio.gather(queued, return_exceptions=True)


def test_stale_after_must_be_positive():
    with pytest.raises(ValueError):
        RedisStore(fakeredis.FakeAsyncRedis(), "x", 0)
