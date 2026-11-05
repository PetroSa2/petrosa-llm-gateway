# Agent instructions: petrosa-llm-gateway

In-cluster LLM gateway for the shared AI provider account. It runs a pinned LiteLLM proxy, exposes an OpenAI-compatible API with Petrosa model aliases, and (in later steps) queues callers by priority, adapts the request rate to the provider's limits and reports every call to petrosa-data-manager.

Ecosystem rules (data pillars, commit and PR process, wording, memory) are in the umbrella [AGENTS.md](https://github.com/PetroSa2/petrosa/blob/main/AGENTS.md); this file covers this repository only. Only statements that can be checked against the repository are listed as facts.

## Commands (from the Makefile)

| Command | Purpose |
|---|---|
| `make setup` | Install development dependencies and the pre-commit hooks |
| `make format` | Format code with ruff |
| `make lint` | Run linting checks with ruff |
| `make type-check` | Run type checking with mypy |
| `make test` | Run all tests with coverage |
| `make security` | Run security scans (Bandit) |
| `make pipeline` | Run the complete CI pipeline locally |
| `make smoke` | Build the image and check it against the mock provider |

Run `make pipeline` before opening a pull request, and `make smoke` when the `Dockerfile`, the proxy config or the entrypoint changes.

## Facts

- Python: `requires-python = ">=3.11"` for the repository's own code and tests; the image runs the Python bundled with the LiteLLM base image.
- The proxy image is the upstream LiteLLM image (ghcr.io/berriai/litellm), pinned by version and digest in `Dockerfile`.
- Model aliases are defined in `config/litellm_config.yaml`: `petrosa-primary`, `petrosa-fallback` and `petrosa-ops`. Callers use only these names.
- `gateway/entrypoint.py` reads the provider key and the proxy master key from mounted Secret files, logs only their fingerprints, and then starts the proxy.
- `tests/mock_provider.py` is an OpenAI-compatible mock provider used by the tests and by `scripts/smoke.sh`.
- `tests/test_no_provider_key.py` guards against provider keys in the repository and in the proxy config.
- Lint and format: ruff (config in `ruff.toml`). Tests: pytest, in `tests/`.
- CI workflows: `.github/workflows/ci-checks.yml`, `.github/workflows/agent-checks.yml`.

## Rules (policy)

- Credentials only by reference: the provider key and the master key come from mounted Secret files, never from the repository, the image, environment literals or logs.
- The LiteLLM base image stays pinned by version and digest to a release at least two weeks old.
- Do not add database drivers or connections. Call records go to petrosa-data-manager, which owns all storage.
- Never log or store prompt or completion bodies.
- Timing in tests is injected (clocks, sleeps, budgets), never real waits.
- Text that leaves the repository (PR titles and bodies, commit messages, code comments) uses generic roles such as Agentic Developer.
- Do not commit logs, drafts, scratch files or generated working notes. GitHub and the memory server are the record.
