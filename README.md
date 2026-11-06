# petrosa-llm-gateway

In-cluster LLM gateway: priority queue and adaptive rate for the shared AI provider account; reports every call to petrosa-data-manager.

## What it runs

A pinned [LiteLLM](https://github.com/BerriAI/litellm) proxy (see `Dockerfile`) with an OpenAI-compatible API on port 4000. Callers ask for one of the Petrosa model aliases defined in `config/litellm_config.yaml`:

| Alias | Used by |
|---|---|
| `petrosa-primary` | CIO decision stages (primary model) |
| `petrosa-fallback` | CIO when the primary model is slow |
| `petrosa-ops` | the incident assistant (holmes) |

## Calling the gateway

Each caller has its own key, a mounted file under `CALLER_KEYS_DIR` (default `/var/run/secrets/llm-gateway/callers/<caller>`). `config/scheduling.yaml` maps the caller to its priority class: `cio` is `trading`, `holmes` is `ops`. Caller keys may use only the inference routes (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, with or without `/v1`). Every other route answers 403.

Send the time you have left as `x-petrosa-deadline-ms`. A request that cannot start in time is refused at once instead of queueing. Each provider attempt gets an equal share of the remaining budget as its timeout, so a slow primary still leaves the fallback time to answer.

**Answers that mean "do not call again now":**

| Answer | Meaning | Caller should |
|---|---|---|
| **429** | This request hit the provider's 429. It made one attempt and no fallback, because the fallback shares the account. | treat it as **account throttled** |
| **503** `{"error": "account_throttled"}` | The account is cooling down after a 429. No provider call was made. | treat it as **account throttled** |
| **503** `deadline_unmeetable` / `queue_expired` | The request could not start within its deadline or its class's queue age. | treat it as out of time |
| **503** `queue_full` | Too many requests of this class are waiting. | back off |

Both throttle answers mean the same thing: **the account is throttled**. Go to your non-LLM path and never retry into `petrosa-fallback` yourself; it shares the account. The gateway does not retry either: a slow or failing model falls back once, inside the gateway, unless the failure was a 429.

## Ports

| Port | Serves |
|---|---|
| 4000 | the proxy; probes `GET /health/liveliness` and `GET /health/readiness` need no key |
| 9090 (`GATEWAY_METRICS_PORT`) | `/metrics`: learned rate, cool-down, queue depth, wait, attempts and provider 429s |

Set `REDIS_URL` (and optionally `GATEWAY_REDIS_PREFIX`) so the replicas share the queue and the learned rate. Without it, each replica keeps its own.

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
make smoke      # docker build, then auth, scheduling, fallback, throttling and metrics against the mock provider
```
