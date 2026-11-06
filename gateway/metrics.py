"""Gateway metrics (petrosa-llm-gateway#3), served on their own port for the cluster scrape.

Labels are only ``caller``, ``class``, ``alias`` (the Petrosa model alias) and ``outcome``:
never a key, a prompt, a ticket id or anything else unbounded.

The port is separate from the proxy so the proxy's caller-key route checks stay as they
are; the cluster scrapes ``/metrics`` on ``GATEWAY_METRICS_PORT`` (9090, the annotation
convention used by the other Petrosa services).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

logger = logging.getLogger("gateway.metrics")

METRICS_PORT_ENV = "GATEWAY_METRICS_PORT"
#: The scrape port the other Petrosa services annotate; used when the env var is unset.
METRICS_PORT_FALLBACK = 9090
#: Admission waits span instant starts to a class's maximum queue age (minutes).
WAIT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900)


class GatewayMetrics:
    """The metric set; one per registry so tests can use their own."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.rate_rpm = Gauge(
            "llm_gateway_rate_rpm",
            "Learned provider rate in requests per minute (AIMD).",
            registry=registry,
        )
        self.rate_tpm = Gauge(
            "llm_gateway_rate_tpm",
            "Learned provider rate in tokens per minute (AIMD).",
            registry=registry,
        )
        self.cooldown_seconds = Gauge(
            "llm_gateway_cooldown_seconds",
            "Seconds left in the account cool-down after a provider 429 (0 when none).",
            registry=registry,
        )
        self.queue_depth = Gauge(
            "llm_gateway_queue_depth",
            "Requests waiting for a slot, per class (all replicas when Redis is shared).",
            ["class"],
            registry=registry,
        )
        self.wait_seconds = Histogram(
            "llm_gateway_wait_seconds",
            "Time from arrival to a provider slot, or to a refusal.",
            ["caller", "class", "outcome"],
            buckets=WAIT_BUCKETS,
            registry=registry,
        )
        self.attempts = Counter(
            "llm_gateway_attempts_total",
            "Provider attempts and refusals per alias.",
            ["caller", "class", "alias", "outcome"],
            registry=registry,
        )
        self.provider_throttles = Counter(
            "llm_gateway_provider_throttles_total",
            "Provider 429 responses (account throttled).",
            ["alias"],
            registry=registry,
        )

    def set_rate(self, rpm: float, tpm: float, cooldown_s: float) -> None:
        self.rate_rpm.set(rpm)
        self.rate_tpm.set(tpm)
        self.cooldown_seconds.set(max(0.0, cooldown_s))

    def set_queue_depth(
        self, classes: Iterable[str], waiting: Mapping[str, int]
    ) -> None:
        """Every class gets a value, so an emptied queue reads 0 rather than stale."""
        for cls in classes:
            self.queue_depth.labels(cls).set(waiting.get(cls, 0))

    def admission(self, caller: str, cls: str, outcome: str, wait_s: float) -> None:
        self.wait_seconds.labels(caller, cls, outcome).observe(max(0.0, wait_s))

    def attempt(self, caller: str, cls: str, alias: str, outcome: str) -> None:
        self.attempts.labels(caller, cls, alias, outcome).inc()

    def throttled(self, alias: str) -> None:
        self.provider_throttles.labels(alias).inc()


def metrics_port(environ: Mapping[str, str]) -> int:
    """The scrape port from the environment, or the labelled fallback."""
    raw = environ.get(METRICS_PORT_ENV)
    try:
        port = int(raw) if raw is not None else METRICS_PORT_FALLBACK
    except ValueError:
        logger.warning(
            "%s=%r is not a port; using %d",
            METRICS_PORT_ENV,
            raw,
            METRICS_PORT_FALLBACK,
        )
        return METRICS_PORT_FALLBACK
    if not 0 < port < 65536:
        logger.warning(
            "%s=%r is out of range; using %d",
            METRICS_PORT_ENV,
            raw,
            METRICS_PORT_FALLBACK,
        )
        return METRICS_PORT_FALLBACK
    return port


def serve(environ: Mapping[str, str], registry: CollectorRegistry = REGISTRY) -> int:
    """Serve ``/metrics`` on the configured port; returns the port."""
    port = metrics_port(environ)
    start_http_server(port, registry=registry)
    logger.info("metrics on :%d/metrics", port)
    return port
