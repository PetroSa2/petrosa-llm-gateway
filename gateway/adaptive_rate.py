"""Adaptive provider rate for the shared account (petrosa-llm-gateway#3).

AIMD on requests and tokens per minute: a provider 429 halves both and starts a cool-down
(at least the provider's ``retry-after``); every clean minute since the last change adds
one request per minute (and the matching token share) back. The cold start is a labelled
fallback measured on 2026-11-05 (about 10 requests per minute for the shared account).

Any provider 429 means the ACCOUNT is throttled: one key, one account, and the fallback
model shares it. During the cool-down callers get an immediate ``account_throttled``
instead of a call that would only add another 429.

The state is a small record, so the in-memory store (one process) and the Redis store
(replicas, and survival across restarts) keep the same thing; the decisions are pure
functions of the record and the time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

#: Fallback only, used before anything has been learned: measured 2026-11-05 on the
#: shared account (the incident assistant's configured rpm 10; its 429s appeared above it).
COLD_START_RPM = 10.0
#: Fallback tokens per minute before anything is learned; replaced by the learned value.
COLD_START_TPM = 20_000.0
#: The rate never drops below one request per minute: halving keeps probing the account.
MIN_RPM = 1.0
#: Tokens per minute that one request per minute is worth when the rate grows back.
TOKENS_PER_REQUEST_FALLBACK = COLD_START_TPM / COLD_START_RPM
#: Cool-down after a 429 that names no retry-after: one minute, the rate window.
DEFAULT_COOLDOWN_S = 60.0
CLEAN_PERIOD_S = 60.0


@dataclass(frozen=True)
class RateState:
    """The learned rate and the cool-down, shared by every replica."""

    rpm: float = COLD_START_RPM
    tpm: float = COLD_START_TPM
    cooldown_until: float = 0.0
    last_change_at: float = 0.0
    throttles: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes | None) -> RateState:
        if not raw:
            return cls()
        return cls(**json.loads(raw))


def on_throttled(
    state: RateState, now: float, retry_after_s: float | None
) -> RateState:
    """A provider 429: halve both rates (never below the floor) and start the cool-down."""
    cooldown = (
        retry_after_s if retry_after_s and retry_after_s > 0 else DEFAULT_COOLDOWN_S
    )
    factor = max(state.rpm / 2.0, MIN_RPM) / state.rpm if state.rpm > 0 else 1.0
    return replace(
        state,
        rpm=max(state.rpm / 2.0, MIN_RPM),
        tpm=state.tpm * factor,
        cooldown_until=max(state.cooldown_until, now + cooldown),
        last_change_at=now,
        throttles=state.throttles + 1,
    )


def on_clean(state: RateState, now: float) -> RateState:
    """+1 request per minute (and its token share) for every clean minute since the last change."""
    if now < state.cooldown_until:
        return state
    since = max(state.last_change_at, state.cooldown_until)
    minutes = int((now - since) // CLEAN_PERIOD_S)
    if minutes <= 0:
        return state
    tokens_per_request = (
        state.tpm / state.rpm if state.rpm > 0 else TOKENS_PER_REQUEST_FALLBACK
    )
    return replace(
        state,
        rpm=state.rpm + minutes,
        tpm=state.tpm + minutes * tokens_per_request,
        last_change_at=since + minutes * CLEAN_PERIOD_S,
    )


def throttled(state: RateState, now: float) -> bool:
    """True while the account is cooling down after a 429."""
    return now < state.cooldown_until


def retry_after_seconds(headers: dict[str, Any] | None) -> float | None:
    """The provider's ``retry-after`` (seconds, or the ``-ms`` variant), if any."""
    if not headers:
        return None
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = lowered.get(name)
        if raw is None:
            continue
        try:
            value = float(raw) * scale
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def rate_headers(headers: dict[str, Any] | None) -> dict[str, str]:
    """The raw provider rate headers worth logging: retry-after and x-ratelimit-*."""
    if not headers:
        return {}
    return {
        str(k).lower(): str(v)
        for k, v in headers.items()
        if str(k).lower().startswith(("retry-after", "x-ratelimit-"))
    }


class RateStore(Protocol):
    async def load(self) -> RateState: ...

    async def update(self, change) -> RateState: ...


class InMemoryRateStore:
    """One process; the Redis store shares the same record across replicas."""

    def __init__(self, state: RateState | None = None) -> None:
        self.state = state or RateState()

    async def load(self) -> RateState:
        return self.state

    async def update(self, change) -> RateState:
        self.state = change(self.state)
        return self.state


class RedisRateStore:
    """The rate record in one Redis key: shared by replicas and kept across restarts."""

    def __init__(self, redis, key: str, retries: int = 20) -> None:
        self.redis = redis
        self.key = key
        self.retries = retries

    async def load(self) -> RateState:
        return RateState.from_json(await self.redis.get(self.key))

    async def update(self, change) -> RateState:
        from redis.exceptions import WatchError

        for _ in range(self.retries):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(self.key)
                    new = change(RateState.from_json(await pipe.get(self.key)))
                    pipe.multi()
                    pipe.set(self.key, new.to_json())
                    await pipe.execute()
                    return new
                except WatchError:
                    continue
        raise RuntimeError("rate state update kept conflicting")
