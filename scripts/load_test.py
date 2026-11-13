"""Bounded CIO and Holmes replay through the scheduler and mock provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.adaptive_rate import InMemoryRateStore, on_clean, on_throttled, throttled
from gateway.scheduler import InMemoryStore, Scheduler, SchedulingConfig
from tests.mock_provider import UPSTREAM_MODELS, MockProvider, serve

STAGE_BUDGET_S = 5.0
HOLMES_INTERVAL_S = 0.06


@dataclass(frozen=True)
class Result:
    cls: str
    wait: float
    status: int


def config() -> SchedulingConfig:
    return SchedulingConfig.from_mapping(
        {
            "classes": {
                "trading": {"rank": 0, "max_queue": 50, "max_age_s": 30},
                "ops": {"rank": 1, "max_queue": 100, "max_age_s": 300},
            },
            "rate_per_window": 10,
            "window_s": 0.06,
            "reserved_class": "trading",
            "reserved_share": 0.5,
            "rank_gap_s": 60,
            "poll_s": 0.001,
        }
    )


def provider_chat(provider: MockProvider, model: str) -> int:
    request = urllib.request.Request(
        f"{provider.url}/chat/completions",
        data=json.dumps({"model": model, "messages": []}).encode(),
        headers={"Authorization": "Bearer load-test"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1) as response:  # noqa: S310 - local mock
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


async def replay(calls: int, provider: MockProvider) -> tuple[list[Result], int]:
    scheduler = Scheduler(config(), InMemoryStore(), time.monotonic, asyncio.sleep)
    rates = InMemoryRateStore()
    results: list[Result] = []
    lock = asyncio.Lock()
    provider.set_behaviour(
        {
            UPSTREAM_MODELS[0]: {
                "status": 429,
                "headers": {"retry-after": "0.001"},
                "times": 1,
            }
        }
    )

    async def call(cls: str, model: str) -> None:
        started = time.monotonic()
        state = await rates.load()
        if throttled(state, started):
            status = 503
        else:
            await scheduler.admit(cls, budget_s=STAGE_BUDGET_S)
            status = await asyncio.to_thread(provider_chat, provider, model)
            if status == 429:
                await rates.update(
                    lambda value: on_throttled(value, time.monotonic(), 0.06)
                )
            elif status == 200:
                await rates.update(
                    lambda value: on_clean(value, time.monotonic(), True)
                )
        async with lock:
            results.append(Result(cls, time.monotonic() - started, status))

    tasks = []
    for i in range(calls):
        cls = "trading" if i % 3 == 0 else "ops"
        model = UPSTREAM_MODELS[0] if cls == "trading" else UPSTREAM_MODELS[1]
        tasks.append(asyncio.create_task(call(cls, model)))
        if cls == "ops":
            await asyncio.sleep(HOLMES_INTERVAL_S)
    await asyncio.gather(*tasks)
    return results, (await rates.load()).throttles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=18)
    parser.add_argument("--max-p95-wait", type=float, default=STAGE_BUDGET_S)
    args = parser.parse_args()
    provider = serve()
    started = time.monotonic()
    try:
        results, throttles = asyncio.run(replay(args.calls, provider))
    finally:
        provider.shutdown()
    cio_waits = [result.wait for result in results if result.cls == "trading"]
    p95 = statistics.quantiles(cio_waits, n=20, method="inclusive")[18]
    provider_calls = len(provider.seen)
    provider_429s = sum(result.status == 429 for result in results)
    refused_after_429 = sum(result.status == 503 for result in results)
    print(
        f"load-test calls={len(results)} provider_calls={provider_calls} "
        f"cio_p95_wait_s={p95:.3f} provider_429s={provider_429s} "
        f"account_refusals_after_429={refused_after_429} aimd_throttles={throttles}"
    )
    if (
        time.monotonic() - started > 30
        or provider_calls == 0
        or provider_429s != 1
        or throttles != 1
        or refused_after_429 == 0
        or p95 > args.max_p95_wait
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
