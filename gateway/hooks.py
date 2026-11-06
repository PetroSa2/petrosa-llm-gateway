"""LiteLLM hooks: every provider attempt waits for its turn within the learned rate.

The caller and its class come from the custom auth (``gateway.callers``); the caller's
remaining time comes from the ``x-petrosa-deadline-ms`` request header. A request the
gateway refuses gets HTTP 503 with a stable ``error`` code, so callers can tell a full
queue, an unmeetable deadline or a throttled account from a provider failure.

- Pre-call (once per request): during a cool-down after a provider 429, answer
  ``account_throttled`` at once, with no provider call; otherwise wait for a slot
  (petrosa-llm-gateway#2) and give the attempt a timeout from the remaining budget.
- Deployment hook (once per provider attempt, fallbacks included): a later attempt
  takes its own slot, or is refused after a 429 (``gateway.attempts``) (gw#3).
- Failure and success events: AIMD on the shared rate (``gateway.adaptive_rate``).

Scheduler and rate state are shared in Redis when ``REDIS_URL`` is set (two replicas
act as one); without it, one replica keeps them in memory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from gateway.adaptive_rate import (
    InMemoryRateStore,
    RateStore,
    RedisRateStore,
    on_clean,
    on_throttled,
    rate_headers,
    retry_after_seconds,
    throttled,
)
from gateway.attempts import (
    DEADLINE_AT_KEY,
    THROTTLE_EXCEPTIONS,
    Attempt,
    attempt_timeout,
    decide,
    remaining_budget,
)
from gateway.scheduler import (
    InMemoryStore,
    Scheduler,
    SchedulingConfig,
    SchedulingRefused,
    Store,
)

logger = logging.getLogger("gateway.hooks")

DEADLINE_HEADER = "x-petrosa-deadline-ms"
#: Requests without a caller class (the gateway's own admin calls) queue as the lowest class.
DEFAULT_CLASS = "batch"
#: A replica that stops polling for this many polls has died; its tickets go stale.
STALE_AFTER_POLLS = 20
#: Request metadata key carrying the admitted class to the later attempts.
CLASS_KEY = "petrosa_class"


def budget_from_headers(headers: Mapping[str, Any] | None) -> float | None:
    """The caller's remaining time in seconds, or None when absent or not a positive number."""
    if not headers:
        return None
    lowered = {str(k).lower(): v for k, v in headers.items()}
    raw = lowered.get(DEADLINE_HEADER)
    try:
        millis = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    if millis is None or millis <= 0:
        return None
    return millis / 1000.0


def request_class(metadata: Mapping[str, Any] | None, config: SchedulingConfig) -> str:
    """The priority class the custom auth attached, or the default class."""
    cls = (metadata or {}).get("priority_class")
    return cls if isinstance(cls, str) and cls in config.classes else DEFAULT_CLASS


async def admit_request(
    scheduler: Scheduler,
    metadata: Mapping[str, Any] | None,
    headers: Mapping[str, Any] | None,
):
    """Wait for this request's turn; ``SchedulingRefused`` when it never comes."""
    cls = request_class(metadata, scheduler.config)
    return await scheduler.admit(cls, budget_from_headers(headers))


def load_fallbacks(path: str | Path) -> dict[str, list[str]]:
    """The router's fallback chain per alias, from the LiteLLM config."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    chains: dict[str, list[str]] = {}
    for entry in (raw.get("router_settings") or {}).get("fallbacks") or []:
        for alias, targets in (entry or {}).items():
            chains[str(alias)] = [str(t) for t in targets or []]
    return chains


def is_throttle(exception: BaseException | None) -> bool:
    """A provider 429: the account is throttled."""
    if exception is None:
        return False
    return (
        getattr(exception, "status_code", None) == 429
        or type(exception).__name__ in THROTTLE_EXCEPTIONS
    )


def response_headers(exception: BaseException | None) -> dict[str, Any]:
    """The provider response headers LiteLLM attaches to an exception, if any."""
    headers = getattr(exception, "litellm_response_headers", None)
    if headers is None:
        response = getattr(exception, "response", None)
        headers = getattr(response, "headers", None)
    try:
        return dict(headers or {})
    except (TypeError, ValueError):
        return {}


class GatewayPolicy:
    """The admission and rate decisions behind the LiteLLM hooks, testable without LiteLLM."""

    def __init__(
        self,
        scheduler: Scheduler,
        rates: RateStore,
        clock: Any,
        fallbacks: Mapping[str, list[str]] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.rates = rates
        self.clock = clock
        self.fallbacks = dict(fallbacks or {})

    async def on_request(
        self,
        auth_metadata: Mapping[str, Any] | None,
        headers: Mapping[str, Any] | None,
        data: dict[str, Any],
    ) -> None:
        """Refuse during a cool-down; otherwise wait for a slot and set the attempt timeout."""
        now = self.clock()
        state = await self.rates.load()
        if throttled(state, now):
            raise SchedulingRefused(
                "account_throttled",
                f"provider account cooling down for {state.cooldown_until - now:.1f}s",
            )
        cls = request_class(auth_metadata, self.scheduler.config)
        budget = budget_from_headers(headers)
        await self.scheduler.admit(cls, budget)
        metadata = data.setdefault("metadata", {})
        metadata[CLASS_KEY] = cls
        if budget is not None:
            metadata[DEADLINE_AT_KEY] = now + budget
            remaining = max(0.0, now + budget - self.clock())
            models = 1 + len(self.fallbacks.get(str(data.get("model")), []))
            timeout = attempt_timeout(remaining, models)
            if timeout is not None:
                data["timeout"] = timeout

    async def on_attempt(self, metadata: Mapping[str, Any] | None) -> None:
        """Count a later provider attempt in the rate, or refuse it after a 429."""
        decision = decide(metadata)
        if decision is Attempt.FIRST:
            return
        if decision is Attempt.REFUSE_THROTTLED:
            raise SchedulingRefused(
                "account_throttled", "provider 429: no fallback on the same account"
            )
        has_deadline, remaining = remaining_budget(metadata, self.clock())
        if has_deadline and not remaining:
            raise SchedulingRefused(
                "deadline_unmeetable", "no budget left for a fallback"
            )
        cls = (metadata or {}).get(CLASS_KEY)
        if not isinstance(cls, str) or cls not in self.scheduler.config.classes:
            cls = DEFAULT_CLASS
        await self.scheduler.admit(cls, remaining)

    async def on_failure(self, exception: BaseException | None) -> bool:
        """A provider 429 halves the shared rate and starts the cool-down; True when it did."""
        if not is_throttle(exception):
            return False
        headers = response_headers(exception)
        now = self.clock()
        retry_after = retry_after_seconds(headers)
        state = await self.rates.update(lambda s: on_throttled(s, now, retry_after))
        logger.warning(
            "provider throttled: rpm=%.1f tpm=%.0f cooldown_s=%.1f throttles=%d headers=%s",
            state.rpm,
            state.tpm,
            max(0.0, state.cooldown_until - now),
            state.throttles,
            rate_headers(headers),
        )
        return True

    async def on_success(self) -> None:
        """A clean call: grow the rate by one per clean minute since the last change."""
        now = self.clock()
        state = await self.rates.load()
        if on_clean(state, now) != state:
            await self.rates.update(lambda s: on_clean(s, now))


def build_scheduler(
    environ: Mapping[str, str], rates: RateStore | None = None
) -> Scheduler:
    """The process scheduler from the mounted config, with the Redis store when configured."""
    path = Path(environ.get("GATEWAY_SCHEDULING", "/etc/llm-gateway/scheduling.yaml"))
    config = SchedulingConfig.from_mapping(yaml.safe_load(path.read_text()))
    store: Store
    redis_url = environ.get("REDIS_URL")
    if redis_url:
        from redis.asyncio import Redis

        from gateway.redis_store import RedisStore

        store = RedisStore(
            Redis.from_url(redis_url),
            environ.get("GATEWAY_REDIS_PREFIX", "llm-gateway"),
            stale_after_s=STALE_AFTER_POLLS * config.poll_s,
        )
    else:
        logger.warning("REDIS_URL is not set: scheduler state is local to this replica")
        store = InMemoryStore()

    async def learned_rpm() -> float:
        assert rates is not None
        return (await rates.load()).rpm

    # Wall-clock time: replicas compare queue and window times through Redis.
    return Scheduler(
        config, store, time.time, asyncio.sleep, learned_rpm if rates else None
    )


def build_rates(environ: Mapping[str, str]) -> RateStore:
    """The shared rate record: one Redis key when configured, else this process."""
    redis_url = environ.get("REDIS_URL")
    if redis_url:
        from redis.asyncio import Redis

        prefix = environ.get("GATEWAY_REDIS_PREFIX", "llm-gateway")
        return RedisRateStore(Redis.from_url(redis_url), f"{prefix}:rate")
    return InMemoryRateStore()


def build_policy(environ: Mapping[str, str]) -> GatewayPolicy:
    """Scheduler, rate store and fallback chains, from the mounted configs."""
    rates = build_rates(environ)
    fallbacks = load_fallbacks(
        environ.get("GATEWAY_CONFIG", "/etc/llm-gateway/litellm_config.yaml")
    )
    return GatewayPolicy(build_scheduler(environ, rates), rates, time.time, fallbacks)


try:  # pragma: no cover - the adapter needs LiteLLM, present only in the image
    from fastapi import HTTPException
    from litellm.integrations.custom_logger import CustomLogger

    class SchedulingHook(CustomLogger):
        """Admits each provider attempt through the shared scheduler and learned rate."""

        def __init__(self) -> None:
            super().__init__()
            self._policy: GatewayPolicy | None = None

        @property
        def policy(self) -> GatewayPolicy:
            if self._policy is None:
                self._policy = build_policy(os.environ)
            return self._policy

        @staticmethod
        def _refusal(refused: SchedulingRefused) -> HTTPException:
            return HTTPException(
                status_code=503,
                detail={"error": refused.code, "detail": refused.detail},
            )

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            headers = (data.get("proxy_server_request") or {}).get("headers")
            try:
                await self.policy.on_request(
                    getattr(user_api_key_dict, "metadata", None), headers, data
                )
            except SchedulingRefused as refused:
                raise self._refusal(refused) from None
            return data

        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            try:
                await self.policy.on_attempt(kwargs.get("metadata"))
            except SchedulingRefused as refused:
                raise self._refusal(refused) from None
            return None

        async def async_log_failure_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            await self.policy.on_failure(kwargs.get("exception"))

        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            await self.policy.on_success()

    proxy_handler_instance: Any = SchedulingHook()
except ImportError:  # pragma: no cover - tests run without LiteLLM
    proxy_handler_instance = None
