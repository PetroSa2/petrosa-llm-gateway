"""Gateway metrics: names, bounded labels, and the scrape endpoint (gw#3)."""

from __future__ import annotations

import socket
import urllib.request

import pytest
from prometheus_client import CollectorRegistry

from gateway import metrics as gm


@pytest.fixture
def registry():
    return CollectorRegistry()


def test_rate_queue_wait_attempts_and_throttles(registry):
    m = gm.GatewayMetrics(registry)

    m.set_rate(5.0, 10_000.0, 2.5)
    m.set_queue_depth(["trading", "ops", "batch"], {"ops": 3})
    m.admission("cio", "trading", "started", 0.2)
    m.admission("holmes", "ops", "queue_full", -1.0)
    m.attempt("cio", "trading", "petrosa-primary", "started")
    m.throttled("petrosa-primary")

    value = registry.get_sample_value
    assert value("llm_gateway_rate_rpm") == 5.0
    assert value("llm_gateway_rate_tpm") == 10_000.0
    assert value("llm_gateway_cooldown_seconds") == 2.5
    assert value("llm_gateway_queue_depth", {"class": "ops"}) == 3
    assert value("llm_gateway_queue_depth", {"class": "trading"}) == 0
    assert (
        value(
            "llm_gateway_wait_seconds_count",
            {"caller": "cio", "class": "trading", "outcome": "started"},
        )
        == 1
    )
    assert (
        value(
            "llm_gateway_wait_seconds_sum",
            {"caller": "holmes", "class": "ops", "outcome": "queue_full"},
        )
        == 0.0
    )
    assert (
        value(
            "llm_gateway_attempts_total",
            {
                "caller": "cio",
                "class": "trading",
                "alias": "petrosa-primary",
                "outcome": "started",
            },
        )
        == 1
    )
    assert (
        value("llm_gateway_provider_throttles_total", {"alias": "petrosa-primary"}) == 1
    )


def test_labels_are_only_caller_class_alias_and_outcome(registry):
    gm.GatewayMetrics(registry)
    allowed = {"caller", "class", "alias", "outcome", "le"}

    for family in registry.collect():
        for sample in family.samples:
            assert set(sample.labels) <= allowed, (family.name, sample.labels)


def test_cooldown_never_reads_negative(registry):
    m = gm.GatewayMetrics(registry)
    m.set_rate(1.0, 2_000.0, -4.0)
    assert registry.get_sample_value("llm_gateway_cooldown_seconds") == 0.0


@pytest.mark.parametrize(
    ("env", "port"),
    [
        ({}, gm.METRICS_PORT_FALLBACK),
        ({gm.METRICS_PORT_ENV: "9191"}, 9191),
        ({gm.METRICS_PORT_ENV: "nine"}, gm.METRICS_PORT_FALLBACK),
        ({gm.METRICS_PORT_ENV: "0"}, gm.METRICS_PORT_FALLBACK),
        ({gm.METRICS_PORT_ENV: "70000"}, gm.METRICS_PORT_FALLBACK),
    ],
)
def test_metrics_port(env, port):
    assert gm.metrics_port(env) == port


def test_serve_exposes_the_registry(registry):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    m = gm.GatewayMetrics(registry)
    m.throttled("petrosa-fallback")

    assert gm.serve({gm.METRICS_PORT_ENV: str(free)}, registry) == free
    with urllib.request.urlopen(  # noqa: S310 - local test server
        f"http://127.0.0.1:{free}/metrics", timeout=5
    ) as response:
        body = response.read().decode()
    assert 'llm_gateway_provider_throttles_total{alias="petrosa-fallback"} 1.0' in body
