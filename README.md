# petrosa-llm-gateway

In-cluster LLM gateway: priority queue and adaptive rate for the shared AI provider account; reports every call to petrosa-data-manager.

## What it runs

A pinned [LiteLLM](https://github.com/BerriAI/litellm) proxy (see `Dockerfile`) with an OpenAI-compatible API on port 4000. Callers ask for one of the Petrosa model aliases defined in `config/litellm_config.yaml`:

| Alias | Used by |
|---|---|
| `petrosa-primary` | CIO decision stages (primary model) |
| `petrosa-fallback` | CIO when the primary model is slow |
| `petrosa-ops` | the incident assistant (holmes) |

## Credentials

The container reads two mounted Secret files at start-up and logs only a short SHA-256 fingerprint of each:

| Setting | Default path |
|---|---|
| `PROVIDER_KEY_FILE` | `/var/run/secrets/llm-gateway/provider-api-key` |
| `MASTER_KEY_FILE` | `/var/run/secrets/llm-gateway/master-key` |

Setting the keys directly in the environment is refused. `PROVIDER_API_BASE` (not secret) selects the provider endpoint.

## Local checks

```bash
make pipeline   # format, lint, tests, security scan
make smoke      # docker build, then /health and /v1/models against the mock provider
```
