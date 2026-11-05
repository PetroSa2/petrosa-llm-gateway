"""Priority scheduling under virtual time (petrosa-llm-gateway#2). No wall-clock sleeps."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time

import pytest

from gateway.scheduler import (
    InMemoryStore,
    Scheduler,
    SchedulingConfig,
    SchedulingRefused,
    Started,
    Ticket,
    capacity,
    may_start,
    queue_key,
)

RAW = {
    "classes": {
        "trading": {"rank": 0, "max_queue": 50, "max_age_s": 30},
        "ops": {"rank": 1, "max_queue": 50, "max_age_s": 120},
        "batch": {"rank": 2, "max_queue": 5, "max_age_s": 600},
    },
    "rate_per_window": 10,
    "window_s": 60,
    "reserved_class": "trading",
    "reserved_share": 0.3,
    "rank_gap_s": 20,
    "poll_s": 0.5,
}


class VirtualTime:
    """A clock whose sleeps wake only when the test advances time."""

    def __init__(self) -> None:
        self.now = 0.0
        self._wakeups: list[tuple[float, int, asyncio.Future]] = []
        self._seq = itertools.count()

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._wakeups, (self.now + seconds, next(self._seq), future))
        await future

    async def settle(self) -> None:
        for _ in range(20):
            await asyncio.sleep(0)

    async def run_until(self, done, limit_s: float = 3600.0) -> None:
        """Advance virtual time wake-up by wake-up until ``done()`` or the limit."""
        await self.settle()
        while not done() and self._wakeups and self.now <= limit_s:
            at, _, future = heapq.heappop(self._wakeups)
            self.now = max(self.now, at)
            if not future.done():
                future.set_result(None)
            await self.settle()


def make(raw=None):
    config = SchedulingConfig.from_mapping(raw or RAW)
    vt = VirtualTime()
    return config, vt, Scheduler(config, InMemoryStore(), vt.clock, vt.sleep)


def fill_window(store: InMemoryStore, cls: str, count: int, at: float = 0.0) -> None:
    store.window.extend(Started(f"pre-{cls}-{i}", cls, at) for i in range(count))


def test_config_validation():
    with pytest.raises(ValueError):
        SchedulingConfig.from_mapping({**RAW, "reserved_share": 1.0})
    with pytest.raises(ValueError):
        SchedulingConfig.from_mapping({**RAW, "reserved_class": "nope"})
    with pytest.raises(ValueError):
        SchedulingConfig.from_mapping({**RAW, "rate_per_window": 0})


def test_others_cannot_use_the_reserved_share():
    config = SchedulingConfig.from_mapping(RAW)
    window = [Started(f"o{i}", "ops", 0.0) for i in range(7)]  # 7 = 10 x (1 - 0.3)

    allowed = capacity(config, window, now=1.0)

    assert allowed == {"trading": True, "ops": False, "batch": False}
    assert (
        capacity(config, window, now=61.0)["ops"] is True
    )  # the window slid past them


def test_rank_then_aging_orders_the_queue():
    config = SchedulingConfig.from_mapping(RAW)
    old_ops = Ticket("a", "ops", 0.0, None, queue_key(config, "ops", 0.0))
    new_trading = Ticket("b", "trading", 5.0, None, queue_key(config, "trading", 5.0))
    late_trading = Ticket(
        "c", "trading", 25.0, None, queue_key(config, "trading", 25.0)
    )

    assert may_start(config, new_trading, [old_ops, new_trading], [], 5.0)
    # ops enqueued at 0 has aged past a trading arrival 20 s (one rank gap) later.
    assert may_start(config, old_ops, [old_ops, late_trading], [], 25.0)


@pytest.mark.asyncio
async def test_trading_is_served_before_ops_under_contention():
    config, vt, scheduler = make()
    fill_window(scheduler.store, "trading", 10, at=-40.0)  # full until t=20
    order: list[str] = []

    async def call(cls, name):
        await scheduler.admit(cls)
        order.append(name)

    tasks = [asyncio.create_task(call("ops", f"ops{i}")) for i in range(3)]
    await vt.settle()
    vt.now = 1.0
    tasks += [asyncio.create_task(call("trading", f"trading{i}")) for i in range(3)]
    await vt.run_until(lambda: len(order) == 6)

    assert order[:3] == ["trading0", "trading1", "trading2"]
    assert sorted(order[3:]) == ["ops0", "ops1", "ops2"]
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_an_unmeetable_deadline_is_refused_at_once():
    config, vt, scheduler = make()
    fill_window(scheduler.store, "trading", 10)
    for i in range(4):
        await scheduler.store.enqueue(
            config,
            Ticket(f"q{i}", "trading", 0.0, None, queue_key(config, "trading", 0.0)),
            0.0,
        )

    started = time.perf_counter()
    with pytest.raises(SchedulingRefused) as refused:
        await scheduler.admit("trading", budget_s=2.0)  # 5 spacings of 6 s ahead
    elapsed = time.perf_counter() - started

    assert refused.value.code == "deadline_unmeetable"
    assert elapsed < 0.05  # the acceptance bound: refused within 50 ms
    assert vt.now == 0.0  # refused without waiting any (virtual) time
    assert len(scheduler.store.queue) == 4  # and never enqueued


@pytest.mark.asyncio
async def test_a_meetable_deadline_is_admitted_immediately_with_room():
    _, vt, scheduler = make()

    ticket = await scheduler.admit("trading", budget_s=0.5)

    assert ticket.cls == "trading" and vt.now == 0.0


@pytest.mark.asyncio
async def test_ops_is_served_within_its_bound_despite_a_stream_of_trading():
    config, vt, scheduler = make({**RAW, "rate_per_window": 2, "reserved_share": 0.0})
    served: dict[str, float] = {}
    stop = False

    async def one_trading():
        try:
            await scheduler.admit("trading")
        except SchedulingRefused:
            pass  # trading tickets beyond the rate expire; only ops' wait matters here

    async def trading_stream():
        while not stop:
            asyncio.create_task(one_trading())
            await vt.sleep(10.0)

    async def ops_call():
        await scheduler.admit("ops")
        served["ops"] = vt.now

    stream = asyncio.create_task(trading_stream())
    await vt.settle()
    ops = asyncio.create_task(ops_call())
    await vt.run_until(lambda: "ops" in served, limit_s=600.0)
    stop = True

    assert "ops" in served
    assert served["ops"] <= config.policy("ops").max_age_s
    await ops
    stream.cancel()


@pytest.mark.asyncio
async def test_a_bounded_queue_refuses_when_full():
    config, vt, scheduler = make()
    fill_window(scheduler.store, "trading", 10)
    waiters = [asyncio.create_task(scheduler.admit("batch")) for _ in range(5)]
    await vt.settle()

    with pytest.raises(SchedulingRefused) as refused:
        await scheduler.admit("batch")

    assert refused.value.code == "queue_full"
    for waiter in waiters:
        waiter.cancel()
    await asyncio.gather(*waiters, return_exceptions=True)
    assert scheduler.store.queue == {}  # cancelled waiters leave the queue


@pytest.mark.asyncio
async def test_a_queued_request_expires_at_its_max_age():
    config, vt, scheduler = make({**RAW, "rate_per_window": 1, "window_s": 1000})
    fill_window(scheduler.store, "trading", 1, at=0.0)  # full until t=1000
    waiter = asyncio.create_task(scheduler.admit("batch"))
    await vt.run_until(waiter.done, limit_s=1200.0)

    with pytest.raises(SchedulingRefused) as refused:
        await waiter
    assert refused.value.code == "queue_expired"
    assert config.policy("batch").max_age_s <= vt.now < 1000.0
    assert scheduler.store.queue == {}


@pytest.mark.asyncio
async def test_the_rate_is_never_exceeded():
    long_ages = {
        name: {**spec, "max_age_s": 10_000} for name, spec in RAW["classes"].items()
    }
    config, vt, scheduler = make({**RAW, "classes": long_ages})
    starts: list[float] = []

    async def call(cls):
        await scheduler.admit(cls)
        starts.append(vt.now)

    tasks = [
        asyncio.create_task(call("trading" if i % 2 else "ops")) for i in range(30)
    ]
    await vt.run_until(lambda: len(starts) == 30, limit_s=1000.0)

    assert len(starts) == 30
    for at in starts:
        in_window = [s for s in starts if at - config.window_s < s <= at]
        assert len(in_window) <= config.rate_per_window
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_unknown_class_is_refused():
    _, _, scheduler = make()

    with pytest.raises(SchedulingRefused) as refused:
        await scheduler.admit("nobody")

    assert refused.value.code == "unknown_class"
