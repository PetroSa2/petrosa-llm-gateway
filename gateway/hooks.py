"""LiteLLM pre-call hook: every provider call waits for its turn (petrosa-llm-gateway#2).

The caller and its class come from the custom auth (``gateway.callers``); the caller's
remaining time comes from the ``x-petrosa-deadline-ms`` request header. A request the
scheduler refuses gets HTTP 503 with a stable ``error`` code, so callers can tell a full
queue or an unmeetable deadline from a provider failure.

Scheduler state is shared in Redis when ``REDIS_URL`` is set (two replicas act as one);
without it, one replica keeps it in memory.
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


def build_scheduler(environ: Mapping[str, str]) -> Scheduler:
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
    # Wall-clock time: replicas compare queue and window times through Redis.
    return Scheduler(config, store, time.time, asyncio.sleep)


try:  # pragma: no cover - the adapter needs LiteLLM, present only in the image
    from fastapi import HTTPException
    from litellm.integrations.custom_logger import CustomLogger

    class SchedulingHook(CustomLogger):
        """Admits each provider call through the shared scheduler."""

        def __init__(self) -> None:
            super().__init__()
            self._scheduler: Scheduler | None = None

        @property
        def scheduler(self) -> Scheduler:
            if self._scheduler is None:
                self._scheduler = build_scheduler(os.environ)
            return self._scheduler

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            headers = (data.get("proxy_server_request") or {}).get("headers")
            try:
                await admit_request(
                    self.scheduler,
                    getattr(user_api_key_dict, "metadata", None),
                    headers,
                )
            except SchedulingRefused as refused:
                raise HTTPException(
                    status_code=503,
                    detail={"error": refused.code, "detail": refused.detail},
                ) from None
            return data

    proxy_handler_instance = SchedulingHook()
except ImportError:  # pragma: no cover - tests run without LiteLLM
    proxy_handler_instance = None
