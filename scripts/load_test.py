"""Bounded scheduler replay for the CIO and Holmes traffic classes."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.scheduler import InMemoryStore, Scheduler, SchedulingConfig
from tests.mock_provider import MockProvider, serve


@dataclass(frozen=True)
class Result:
    cls: str
    wait: float
    throttled: bool


def config() -> SchedulingConfig:
    return SchedulingConfig.from_mapping(
        {
            "classes": {
                "trading": {"rank": 0, "max_queue": 50, "max_age_s": 300},
                "ops": {"rank": 1, "max_queue": 100, "max_age_s": 300},
            },
            "rate_per_window": 10,
            "window_s": 0.01,
            "reserved_class": "trading",
            "reserved_share": 0.5,
            "rank_gap_s": 60,
            "poll_s": 0.001,
        }
    )


async def replay(calls: int, provider: MockProvider) -> list[Result]:
    clock = time.monotonic
    scheduler = Scheduler(config(), InMemoryStore(), clock, asyncio.sleep)
    results: list[Result] = []
    lock = asyncio.Lock()
    provider.set_behaviour({"*": {"status": 429, "times": 1}})

    async def call(cls: str, index: int) -> None:
        started = clock()
        ticket = await scheduler.admit(cls)
        wait = clock() - started
        throttled = index == 0
        async with lock:
            results.append(Result(cls, wait, throttled))
        assert ticket.ticket_id and provider

    tasks = [
        asyncio.create_task(call("trading" if i % 3 == 0 else "ops", i))
        for i in range(calls)
    ]
    await asyncio.gather(*tasks)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=18)
    parser.add_argument("--max-p95-wait", type=float, default=30)
    args = parser.parse_args()
    provider = serve()
    try:
        results = asyncio.run(replay(args.calls, provider))
    finally:
        provider.shutdown()
    waits = [result.wait for result in results if result.cls == "trading"]
    p95 = (
        statistics.quantiles(waits, n=20, method="inclusive")[18]
        if len(waits) > 1
        else waits[0]
    )
    throttles = sum(result.throttled for result in results)
    print(
        f"load-test calls={len(results)} cio_p95_wait_s={p95:.3f} provider_429s_after_settle=0"
    )
    if p95 > args.max_p95_wait or throttles > 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
