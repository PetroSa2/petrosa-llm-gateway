"""Per-attempt decisions: count attempts, refuse a retry into a throttled account (gw#3)."""

from __future__ import annotations

import pytest

from gateway.attempts import (
    DEADLINE_AT_KEY,
    Attempt,
    attempt_timeout,
    decide,
    previous_failures,
    remaining_budget,
)


def _previous(*types: str) -> dict:
    # The shape LiteLLM v1.106.2 records (observed 2026-11-06), trimmed to what we read.
    return {
        "previous_models": [
            {"model_group": "petrosa-primary", "exception_type": t} for t in types
        ]
    }


def test_the_first_attempt_was_admitted_by_the_pre_call_hook():
    assert decide({}) is Attempt.FIRST
    assert decide(None) is Attempt.FIRST
    assert decide({"previous_models": []}) is Attempt.FIRST


def test_an_attempt_after_a_429_is_refused():
    assert decide(_previous("RateLimitError")) is Attempt.REFUSE_THROTTLED
    assert decide(_previous("Timeout", "RateLimitError")) is Attempt.REFUSE_THROTTLED


@pytest.mark.parametrize("failure", ["Timeout", "InternalServerError", "APIError"])
def test_an_attempt_after_another_failure_takes_its_own_slot(failure):
    assert decide(_previous(failure)) is Attempt.ADMIT


def test_previous_failures_tolerates_odd_entries():
    assert previous_failures(
        {"previous_models": [None, {}, {"exception_type": "X"}]}
    ) == [
        "",
        "",
        "X",
    ]


def test_remaining_budget():
    assert remaining_budget({DEADLINE_AT_KEY: 110.0}, 100.0) == (True, 10.0)
    assert remaining_budget({DEADLINE_AT_KEY: 90.0}, 100.0) == (True, 0.0)
    assert remaining_budget({}, 100.0) == (False, None)
    assert remaining_budget({DEADLINE_AT_KEY: "soon"}, 100.0) == (False, None)


def test_attempt_timeout_shares_the_budget_between_the_models_left():
    assert attempt_timeout(10.0, 2) == 5.0
    assert attempt_timeout(4.0, 1) == 4.0
    assert attempt_timeout(None, 2) is None
    assert attempt_timeout(0.0, 2) is None
    assert attempt_timeout(10.0, 0) is None
