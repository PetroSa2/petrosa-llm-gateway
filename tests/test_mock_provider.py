"""The mock provider's programmable behaviour (used by the gw#3 and gw#5 checks)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from tests import mock_provider

PRIMARY = mock_provider.UPSTREAM_MODELS[0]
FALLBACK = mock_provider.UPSTREAM_MODELS[1]


@pytest.fixture
def server():
    srv = mock_provider.serve()
    yield srv
    srv.shutdown()


def _post(server, path, body):
    request = urllib.request.Request(
        server.url.removesuffix("/v1") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - local mock
            return response.status, dict(response.headers), json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), json.load(error)


def _chat(server, model):
    return _post(server, "/v1/chat/completions", {"model": model, "messages": []})


def test_answers_ok_by_default(server):
    status, _, body = _chat(server, PRIMARY)

    assert status == 200 and body["choices"][0]["message"]["content"] == "ok"
    assert server.seen[-1] == ("/v1/chat/completions", PRIMARY, False)


def test_a_rule_returns_429_with_headers_for_its_model_only(server):
    _post(
        server,
        "/_control",
        {
            "rules": {
                PRIMARY: {"status": 429, "headers": {"retry-after": "7"}, "times": 2}
            }
        },
    )

    first = _chat(server, PRIMARY)
    other = _chat(server, FALLBACK)
    second = _chat(server, PRIMARY)
    third = _chat(server, PRIMARY)

    assert first[0] == 429 and first[1]["retry-after"] == "7"
    assert other[0] == 200
    assert second[0] == 429
    assert third[0] == 200  # the rule was used up


def test_a_catch_all_rule_and_reset(server):
    _post(server, "/_control", {"rules": {"*": {"status": 503}}})
    assert _chat(server, FALLBACK)[0] == 503

    _post(server, "/_control", {"reset": True})
    assert _chat(server, FALLBACK)[0] == 200


def test_a_delay_rule_delays_the_answer(server):
    _post(server, "/_control", {"rules": {PRIMARY: {"delay_s": 0.01, "times": 1}}})

    status, _, _ = _chat(server, PRIMARY)

    assert status == 200
    assert server.behaviour == {}


def test_seen_counts_provider_calls_per_model(server):
    _chat(server, PRIMARY)
    _chat(server, PRIMARY)
    _chat(server, FALLBACK)
    with urllib.request.urlopen(  # noqa: S310 - local mock
        server.url.removesuffix("/v1") + "/_seen", timeout=5
    ) as response:
        assert json.load(response) == {"calls": {PRIMARY: 2, FALLBACK: 1}}
