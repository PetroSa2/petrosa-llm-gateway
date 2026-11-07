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
import uuid
from collections.abc import Iterable, Mapping
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
from gateway.call_records import (
    QUEUE_WAIT_KEY,
    REQUEST_ID_KEY,
    RecordPublisher,
    build_publisher,
    build_record,
)
from gateway.metrics import GatewayMetrics
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
#: Request metadata keys carrying the admitted class and caller to the later attempts.
CLASS_KEY = "petrosa_class"
CALLER_KEY = "petrosa_caller"


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


def load_aliases(path: str | Path) -> list[str]:
    """The Petrosa aliases callers may use, from the LiteLLM config's model list."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return [
        str(entry["model_name"])
        for entry in raw.get("model_list") or []
        if isinstance(entry, dict) and entry.get("model_name")
    ]


def alias_of(kwargs: Mapping[str, Any] | None) -> str | None:
    """The Petrosa alias (model group) of a LiteLLM call, wherever this event carries it.

    The deployment hook has it in ``metadata``; the logging events have it at the top level
    or under ``litellm_params.metadata``.
    """
    kwargs = kwargs or {}
    candidates = (
        kwargs.get("model_group"),
        (kwargs.get("metadata") or {}).get("model_group"),
        ((kwargs.get("litellm_params") or {}).get("metadata") or {}).get("model_group"),
        (kwargs.get("standard_logging_object") or {}).get("model_group"),
    )
    return next((c for c in candidates if isinstance(c, str) and c), None)


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
        metrics: GatewayMetrics | None = None,
        aliases: Iterable[str] = (),
        callers: Iterable[str] = (),
    ) -> None:
        self.scheduler = scheduler
        self.rates = rates
        self.clock = clock
        self.fallbacks = dict(fallbacks or {})
        self.metrics = metrics
        self.aliases = frozenset(aliases)
        self.callers = frozenset(callers)

    def _alias(self, alias: Any) -> str:
        """A bounded metric label: a configured Petrosa alias, else ``other``."""
        return alias if isinstance(alias, str) and alias in self.aliases else "other"

    def _caller(self, caller: Any) -> str:
        """A bounded metric label: a configured caller, else ``gateway`` (the master key)."""
        return (
            caller if isinstance(caller, str) and caller in self.callers else "gateway"
        )

    async def _observe(self, now: float) -> None:
        """Refresh the rate, cool-down and queue-depth gauges from the shared state."""
        if self.metrics is None:
            return
        state = await self.rates.load()
        self.metrics.set_rate(state.rpm, state.tpm, state.cooldown_until - now)
        queue, _window = await self.scheduler.store.snapshot(now)
        waiting: dict[str, int] = {}
        for ticket in queue:
            waiting[ticket.cls] = waiting.get(ticket.cls, 0) + 1
        self.metrics.set_queue_depth(self.scheduler.config.classes, waiting)

    async def on_request(
        self,
        auth_metadata: Mapping[str, Any] | None,
        headers: Mapping[str, Any] | None,
        data: dict[str, Any],
    ) -> None:
        """Refuse during a cool-down; otherwise wait for a slot and set the attempt timeout."""
        now = self.clock()
        caller = self._caller((auth_metadata or {}).get("caller"))
        cls = request_class(auth_metadata, self.scheduler.config)
        alias = self._alias(data.get("model"))
        try:
            state = await self.rates.load()
            if throttled(state, now):
                raise SchedulingRefused(
                    "account_throttled",
                    f"provider account cooling down for {state.cooldown_until - now:.1f}s",
                )
            budget = budget_from_headers(headers)
            await self.scheduler.admit(cls, budget)
        except SchedulingRefused as refused:
            if self.metrics is not None:
                self.metrics.admission(caller, cls, refused.code, self.clock() - now)
                self.metrics.attempt(caller, cls, alias, refused.code)
                await self._observe(self.clock())
            raise
        if self.metrics is not None:
            self.metrics.admission(caller, cls, "started", self.clock() - now)
            await self._observe(self.clock())
        metadata = data.setdefault("metadata", {})
        metadata[CLASS_KEY] = cls
        metadata[CALLER_KEY] = caller
        # Call records (#4): every attempt of this request shares the id; the wait is
        # the time this request spent queued for admission.
        metadata[REQUEST_ID_KEY] = uuid.uuid4().hex
        metadata[QUEUE_WAIT_KEY] = max(0.0, self.clock() - now)
        if budget is not None:
            metadata[DEADLINE_AT_KEY] = now + budget
            remaining = max(0.0, now + budget - self.clock())
            models = 1 + len(self.fallbacks.get(str(data.get("model")), []))
            timeout = attempt_timeout(remaining, models)
            if timeout is not None:
                data["timeout"] = timeout

    async def on_attempt(
        self, metadata: Mapping[str, Any] | None, alias: Any = None
    ) -> None:
        """Count a provider attempt; a later one takes a slot, or is refused after a 429."""
        metadata = metadata or {}
        cls = metadata.get(CLASS_KEY)
        if not isinstance(cls, str) or cls not in self.scheduler.config.classes:
            cls = DEFAULT_CLASS
        caller = self._caller(metadata.get(CALLER_KEY))
        label = self._alias(alias or metadata.get("model_group"))
        try:
            decision = decide(metadata)
            if decision is Attempt.REFUSE_THROTTLED:
                raise SchedulingRefused(
                    "account_throttled", "provider 429: no fallback on the same account"
                )
            if decision is Attempt.ADMIT:
                has_deadline, remaining = remaining_budget(metadata, self.clock())
                if has_deadline and not remaining:
                    raise SchedulingRefused(
                        "deadline_unmeetable", "no budget left for a fallback"
                    )
                await self.scheduler.admit(cls, remaining)
        except SchedulingRefused as refused:
            if self.metrics is not None:
                self.metrics.attempt(caller, cls, label, refused.code)
            raise
        if self.metrics is not None:
            self.metrics.attempt(caller, cls, label, "started")

    async def on_failure(
        self, exception: BaseException | None, alias: Any = None
    ) -> bool:
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
        if self.metrics is not None:
            self.metrics.throttled(self._alias(alias))
            await self._observe(now)
        return True

    async def on_success(self) -> None:
        """A clean call: one rpm back per clean minute, while the rate is the binding limit.

        The window is read only when a step is due, so most calls cost one rate read.
        """
        now = self.clock()
        state = await self.rates.load()
        if on_clean(state, now, saturated=True) == state:
            return
        config = await self.scheduler.effective_config()
        _queue, window = await self.scheduler.store.snapshot(now)
        recent = [s for s in window if s.at > now - config.window_s]
        saturated = len(recent) >= config.rate_per_window
        if on_clean(state, now, saturated) != state:
            await self.rates.update(lambda s: on_clean(s, now, saturated))
            await self._observe(now)


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


def build_policy(
    environ: Mapping[str, str], metrics: GatewayMetrics | None = None
) -> GatewayPolicy:
    """Scheduler, rate store, fallback chains and aliases, from the mounted configs."""
    rates = build_rates(environ)
    litellm_config = environ.get(
        "GATEWAY_CONFIG", "/etc/llm-gateway/litellm_config.yaml"
    )
    scheduling = Path(
        environ.get("GATEWAY_SCHEDULING", "/etc/llm-gateway/scheduling.yaml")
    )
    callers = (yaml.safe_load(scheduling.read_text()) or {}).get("callers") or {}
    return GatewayPolicy(
        build_scheduler(environ, rates),
        rates,
        time.time,
        load_fallbacks(litellm_config),
        metrics,
        load_aliases(litellm_config),
        callers,
    )


def record_attempt(
    publisher: RecordPublisher | None,
    kwargs: Mapping[str, Any] | None,
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    exception: BaseException | None = None,
) -> None:
    """Hand one attempt's record to the publisher; a record problem never touches the call."""
    if publisher is None:
        return
    try:
        publisher.offer(
            build_record(
                kwargs, response_obj, start_time, end_time, exception, alias_of(kwargs)
            )
        )
    except Exception:  # noqa: BLE001 - records are best effort by design (#4)
        logger.exception("call record dropped")


try:  # pragma: no cover - the adapter needs LiteLLM, present only in the image
    from fastapi import HTTPException
    from litellm.integrations.custom_logger import CustomLogger

    class SchedulingHook(CustomLogger):
        """Admits each provider attempt through the shared scheduler and learned rate."""

        def __init__(self) -> None:
            super().__init__()
            self._policy: GatewayPolicy | None = None
            self._metrics = GatewayMetrics()
            self._records = build_publisher(os.environ)
            from gateway.metrics import serve

            serve(os.environ)

        @property
        def policy(self) -> GatewayPolicy:
            if self._policy is None:
                self._policy = build_policy(os.environ, self._metrics)
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
                await self.policy.on_attempt(kwargs.get("metadata"), alias_of(kwargs))
            except SchedulingRefused as refused:
                raise self._refusal(refused) from None
            return None

        async def async_log_failure_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            await self.policy.on_failure(kwargs.get("exception"), alias_of(kwargs))
            record_attempt(
                self._records,
                kwargs,
                response_obj,
                start_time,
                end_time,
                kwargs.get("exception"),
            )

        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            await self.policy.on_success()
            record_attempt(self._records, kwargs, response_obj, start_time, end_time)

    proxy_handler_instance: Any = SchedulingHook()
except ImportError:  # pragma: no cover - tests run without LiteLLM
    proxy_handler_instance = None
