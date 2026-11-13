"""Bounded replay through ``GatewayPolicy`` and the programmable mock provider.

The replay compresses one minute of Holmes traffic to ``TIME_SCALE`` seconds. The
relative 11-12 calls/minute shape is retained while CI stays bounded. CIO's stage
budget is the 30-second admission budget used by the caller contract.
"""

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

from gateway.adaptive_rate import InMemoryRateStore
from gateway.hooks import GatewayPolicy
from gateway.scheduler import InMemoryStore, Scheduler, SchedulingConfig
from tests.mock_provider import UPSTREAM_MODELS, MockProvider, serve

TIME_SCALE = 0.02
STAGE_BUDGET_S = 30.0 * TIME_SCALE
HOLMES_INTERVAL_S = 60.0 * TIME_SCALE / 12.0


@dataclass(frozen=True)
class Result:
    cls: str
    wait: float
    status: int


class Provider429(Exception):
    status_code = 429

    def __init__(self, headers: dict[str, str]) -> None:
        super().__init__("mock provider 429")
        self.response = type("Response", (), {"headers": headers})()


def make_policy() -> GatewayPolicy:
    config = SchedulingConfig.from_mapping(
        {
            "classes": {
                "trading": {"rank": 0, "max_queue": 50, "max_age_s": 30 * TIME_SCALE},
                "ops": {"rank": 1, "max_queue": 100, "max_age_s": 300 * TIME_SCALE},
            },
            "rate_per_window": 10,
            "window_s": 60 * TIME_SCALE,
            "reserved_class": "trading",
            "reserved_share": 0.5,
            "rank_gap_s": 60 * TIME_SCALE,
            "poll_s": 0.001,
        }
    )
    scheduler = Scheduler(config, InMemoryStore(), time.monotonic, asyncio.sleep)
    return GatewayPolicy(
        scheduler,
        InMemoryRateStore(),
        time.monotonic,
        aliases=("petrosa-primary", "petrosa-ops"),
        callers=("cio", "holmes"),
    )


def provider_chat(provider: MockProvider, model: str) -> tuple[int, dict[str, str]]:
    request = urllib.request.Request(
        f"{provider.url}/chat/completions",
        data=json.dumps({"model": model, "messages": []}).encode(),
        headers={"Authorization": "Bearer load-test"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1) as response:  # noqa: S310 - local mock
            return response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers)


async def replay(calls: int, provider: MockProvider) -> list[Result]:
    policy = make_policy()
    results: list[Result] = []
    lock = asyncio.Lock()
    provider.set_behaviour(
        {
            UPSTREAM_MODELS[0]: {
                "status": 429,
                "headers": {"retry-after": str(0.2 * TIME_SCALE)},
                "times": 1,
            }
        }
    )

    async def call(cls: str, caller: str, model: str) -> None:
        started = time.monotonic()
        data = {"model": "petrosa-primary" if cls == "trading" else "petrosa-ops"}
        headers = {"x-petrosa-deadline-ms": str(STAGE_BUDGET_S * 1000)}
        metadata = {"caller": caller, "priority_class": cls}
        try:
            await policy.on_request(metadata, headers, data)
            await policy.on_attempt(data["metadata"], data["model"])
            status, response_headers = await asyncio.to_thread(
                provider_chat, provider, model
            )
            if status == 429:
                await policy.on_failure(Provider429(response_headers), data["model"])
            else:
                await policy.on_success()
        except Exception as error:  # policy refusals are observable outcomes
            status = 503 if getattr(error, "code", None) else 500
        async with lock:
            results.append(Result(cls, time.monotonic() - started, status))

    tasks = []
    for i in range(calls):
        cls = "trading" if i % 3 == 0 else "ops"
        caller = "cio" if cls == "trading" else "holmes"
        model = UPSTREAM_MODELS[0] if cls == "trading" else UPSTREAM_MODELS[1]
        tasks.append(asyncio.create_task(call(cls, caller, model)))
        if cls == "ops":
            await asyncio.sleep(HOLMES_INTERVAL_S)
    await asyncio.gather(*tasks)
    # A post-cooldown clean wave proves convergence instead of counting only refusals.
    await asyncio.sleep(0.25 * TIME_SCALE)
    await call("trading", "cio", UPSTREAM_MODELS[0])
    await call("ops", "holmes", UPSTREAM_MODELS[1])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=18)
    parser.add_argument("--max-p95-wait", type=float, default=STAGE_BUDGET_S)
    args = parser.parse_args()
    provider = serve()
    started = time.monotonic()
    try:
        results = asyncio.run(replay(args.calls, provider))
    finally:
        provider.shutdown()
    cio_waits = [result.wait for result in results if result.cls == "trading"]
    p95 = statistics.quantiles(cio_waits, n=20, method="inclusive")[18]
    provider_calls = len(provider.seen)
    provider_429s = sum(result.status == 429 for result in results)
    settled = results[-2:]
    print(
        f"load-test calls={len(results)} provider_calls={provider_calls} "
        f"cio_p95_wait_s={p95:.3f} provider_429s={provider_429s} "
        f"settled_statuses={[result.status for result in settled]}"
    )
    if (
        time.monotonic() - started > 30
        or provider_calls < 3
        or provider_429s != 1
        or any(result.status != 200 for result in settled)
        or p95 > args.max_p95_wait
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
