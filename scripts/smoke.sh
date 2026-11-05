#!/usr/bin/env bash
# Build the gateway image and check it against the mock provider (petrosa-llm-gateway#1):
#   - /health/liveliness answers; /health reports every alias healthy against the mock;
#   - /v1/models lists exactly the Petrosa aliases and refuses a request without the master key;
#   - a chat completion through an alias reaches the mock;
#   - neither generated key nor the prompt text appears in the container logs or the image.
# The keys are random per run and exist only in a temporary directory mounted read-only.
set -euo pipefail

IMAGE="${IMAGE_NAME:-petrosa-llm-gateway}:smoke"
PORT="${SMOKE_PORT:-14000}"
RUN_ID="gwsmoke-$$"
SECRETS_DIR="$(mktemp -d)"
PROVIDER_KEY="smoke-provider-$(openssl rand -hex 16)"
MASTER_KEY="sk-smoke-$(openssl rand -hex 16)"
PROMPT="smoke-prompt-$(openssl rand -hex 8)"

cleanup() {
  docker rm -f "${RUN_ID}-mock" "${RUN_ID}-gateway" >/dev/null 2>&1 || true
  docker network rm "${RUN_ID}" >/dev/null 2>&1 || true
  rm -rf "${SECRETS_DIR}"
}
trap cleanup EXIT

fail() { echo "SMOKE FAIL: $*" >&2; docker logs "${RUN_ID}-gateway" 2>&1 | tail -20 >&2 || true; exit 1; }

printf '%s' "${PROVIDER_KEY}" > "${SECRETS_DIR}/provider-api-key"
printf '%s' "${MASTER_KEY}" > "${SECRETS_DIR}/master-key"
chmod 755 "${SECRETS_DIR}"
chmod 644 "${SECRETS_DIR}"/*

docker build -q -t "${IMAGE}" . >/dev/null
docker network create "${RUN_ID}" >/dev/null
docker run -d --name "${RUN_ID}-mock" --network "${RUN_ID}" \
  -v "$(pwd)/tests:/mock:ro" --entrypoint /app/.venv/bin/python \
  "${IMAGE}" /mock/mock_provider.py --port 8080 >/dev/null
docker run -d --name "${RUN_ID}-gateway" --network "${RUN_ID}" -p "127.0.0.1:${PORT}:4000" \
  -v "${SECRETS_DIR}:/var/run/secrets/llm-gateway:ro" \
  -e PROVIDER_API_BASE="http://${RUN_ID}-mock:8080/v1" "${IMAGE}" >/dev/null

BASE="http://127.0.0.1:${PORT}"
for _ in $(seq 1 90); do
  curl -sf "${BASE}/health/liveliness" >/dev/null && break
  sleep 2
done
curl -sf "${BASE}/health/liveliness" >/dev/null || fail "the proxy did not come up"

AUTH=(-H "Authorization: Bearer ${MASTER_KEY}")

MODELS="$(curl -sf "${AUTH[@]}" "${BASE}/v1/models")" || fail "/v1/models failed"
IDS="$(python3 -c 'import json,sys; print(" ".join(sorted(m["id"] for m in json.load(sys.stdin)["data"])))' <<<"${MODELS}")"
[ "${IDS}" = "petrosa-fallback petrosa-ops petrosa-primary" ] || fail "/v1/models listed: ${IDS}"

CODE="$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/v1/models")"
[ "${CODE}" = "401" ] || fail "/v1/models without the master key answered ${CODE}"

HEALTH="$(curl -sf "${AUTH[@]}" "${BASE}/health")" || fail "/health failed"
COUNTS="$(python3 -c 'import json,sys; h=json.load(sys.stdin); print(h["healthy_count"], h["unhealthy_count"])' <<<"${HEALTH}")"
[ "${COUNTS}" = "3 0" ] || fail "/health healthy/unhealthy: ${COUNTS}"

REPLY="$(curl -sf "${AUTH[@]}" -H 'Content-Type: application/json' "${BASE}/v1/chat/completions" \
  -d "{\"model\":\"petrosa-primary\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}]}")" \
  || fail "chat completion failed"
python3 -c 'import json,sys; assert json.load(sys.stdin)["choices"][0]["message"]["content"] == "ok"' <<<"${REPLY}" \
  || fail "unexpected chat reply"

LOGS="$(docker logs "${RUN_ID}-gateway" 2>&1)"
for secret in "${PROVIDER_KEY}" "${MASTER_KEY}" "${PROMPT}"; do
  ! grep -qF -- "${secret}" <<<"${LOGS}" || fail "a key or the prompt text appeared in the logs"
done
grep -q "PROVIDER_API_KEY loaded from .* (sha256:" <<<"${LOGS}" || fail "no provider key fingerprint in the logs"

IMAGE_ENV="$(docker image inspect "${IMAGE}" --format '{{json .Config.Env}}')"
! grep -qE "PROVIDER_API_KEY|LITELLM_MASTER_KEY" <<<"${IMAGE_ENV}" || fail "a key variable is baked into the image"

echo "smoke OK: aliases=[${IDS}] health=${COUNTS} auth=401-without-key logs=no-keys"
