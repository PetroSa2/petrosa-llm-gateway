"""Per-attempt decisions for provider calls (petrosa-llm-gateway#3).

One request can make several provider ATTEMPTS: the router falls back to
``petrosa-fallback`` when the primary fails or is too slow. Every attempt spends the
shared account's rate, so the scheduler must count attempts, not requests:

- the first attempt was admitted by the pre-call hook;
- a later attempt after a provider 429 is refused: the ACCOUNT is throttled and the
  fallback model shares it, so another call only adds a 429;
- any other later attempt (a timeout or a provider error on one model) takes its own
  slot in the rate window.

LiteLLM records each earlier attempt in the request metadata's ``previous_models``
(with its ``exception_type``) before the next attempt's deployment hook runs, so the
decision needs no timing assumption (checked on the pinned v1.106.2, 2026-11-06).

Router retries of the same model are turned off (``num_retries: 0``): a refused or
throttled attempt would otherwise be retried with backoff for seconds.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from typing import Any

#: Exception types LiteLLM records for a provider 429.
THROTTLE_EXCEPTIONS = frozenset({"RateLimitError"})
#: Metadata key for the request's absolute deadline (epoch seconds), set at admission.
DEADLINE_AT_KEY = "petrosa_deadline_at"


class Attempt(enum.Enum):
    FIRST = "first"  # admitted by the pre-call hook
    REFUSE_THROTTLED = "refuse_throttled"
    ADMIT = "admit"  # a later attempt: one more slot in the rate window


def previous_failures(metadata: Mapping[str, Any] | None) -> list[str]:
    """The exception type of every earlier attempt of this request, oldest first."""
    return [
        str((entry or {}).get("exception_type") or "")
        for entry in (metadata or {}).get("previous_models") or []
    ]


def decide(metadata: Mapping[str, Any] | None) -> Attempt:
    """What the deployment hook does with this attempt."""
    failures = previous_failures(metadata)
    if not failures:
        return Attempt.FIRST
    if any(failure in THROTTLE_EXCEPTIONS for failure in failures):
        return Attempt.REFUSE_THROTTLED
    return Attempt.ADMIT


def remaining_budget(
    metadata: Mapping[str, Any] | None, now: float
) -> tuple[bool, float | None]:
    """``(has_deadline, seconds left)``; seconds left is None without a deadline."""
    deadline_at = (metadata or {}).get(DEADLINE_AT_KEY)
    if not isinstance(deadline_at, int | float):
        return False, None
    return True, max(0.0, deadline_at - now)


def attempt_timeout(remaining_s: float | None, models_left: int) -> float | None:
    """A per-attempt timeout that leaves the same share of the budget to each model left.

    With a 10 s budget and the primary plus one fallback, the primary gets 5 s, so a
    slow primary still leaves the fallback time to answer. Without a budget, no timeout
    is set (LiteLLM's own default applies).
    """
    if remaining_s is None or remaining_s <= 0 or models_left < 1:
        return None
    return remaining_s / models_left
