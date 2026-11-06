# LiteLLM proxy v1.106.2 (released 2026-10-21), pinned by version and index digest.
FROM ghcr.io/berriai/litellm:v1.106.2@sha256:7294b90f868c5c38c99b83021c7ba46f9b439e95f8e32964c925c7d2563a03f7

# Not secret: the provider endpoint (overridden in tests by the mock provider).
ENV PROVIDER_API_BASE=https://integrate.api.nvidia.com/v1 \
    GATEWAY_CONFIG=/etc/llm-gateway/litellm_config.yaml \
    GATEWAY_PORT=4000 \
    GATEWAY_SCHEDULING=/etc/llm-gateway/scheduling.yaml \
    PYTHONPATH=/opt/llm-gateway \
    PYTHONDONTWRITEBYTECODE=1

COPY gateway/ /opt/llm-gateway/gateway/
COPY config/litellm_config.yaml /etc/llm-gateway/litellm_config.yaml
COPY config/scheduling.yaml /etc/llm-gateway/scheduling.yaml

USER 65532:65532
EXPOSE 4000

# Keys are read from mounted Secret files by the entrypoint; none is baked in.
ENTRYPOINT ["/app/.venv/bin/python", "-m", "gateway.entrypoint"]
CMD []
