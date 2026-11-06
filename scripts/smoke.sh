#!/usr/bin/env bash
# Build the gateway image and check it against the mock provider (petrosa-llm-gateway#1):
#   - /health/liveliness answers; /health reports every alias healthy against the mock;
#   - /v1/models lists exactly the Petrosa aliases and refuses a request without the master key;
#   - a chat completion through an alias reaches the mock;
#   - a slow primary falls back within the budget; a 429 makes one attempt, no fallback, and
#     the next request is refused at once with 503 account_throttled (gw#3);
#   - neither generated key nor the prompt text appears in the container logs or the image.
# The keys are random per run and exist only in a temporary directory mounted read-only.
set -euo pipefail

IMAGE="${IMAGE_NAME:-petrosa-llm-gateway}:smoke"
PORT="${SMOKE_PORT:-14000}"
RUN_ID="gwsmoke-$$"
SECRETS_DIR="$(mktemp -d)"
PROVIDER_KEY="smoke-provider-$(openssl rand -hex 16)"
MASTER_KEY="sk-smoke-$(openssl rand -hex 16)"
CIO_KEY="cio-smoke-$(openssl rand -hex 16)"
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
mkdir -p "${SECRETS_DIR}/callers"
printf '%s' "${CIO_KEY}" > "${SECRETS_DIR}/callers/cio"
chmod -R a+rX "${SECRETS_DIR}"

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

# Kubernetes probes (petrosa_k8s#1465): liveness and readiness answer with no key at all;
# /health (which calls every model) still needs the master key.
for probe in /health/liveliness /health/readiness; do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' "${BASE}${probe}")"
  [ "${CODE}" = "200" ] || fail "probe ${probe} without a key answered ${CODE}"
done
CODE="$(curl -s -o /dev/null -w '%{http_code}' "${BASE}/health")"
[ "${CODE}" = "401" ] || fail "/health without a key answered ${CODE}"

REPLY="$(curl -sf "${AUTH[@]}" -H 'Content-Type: application/json' "${BASE}/v1/chat/completions" \
  -d "{\"model\":\"petrosa-primary\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}]}")" \
  || fail "chat completion failed"
python3 -c 'import json,sys; assert json.load(sys.stdin)["choices"][0]["message"]["content"] == "ok"' <<<"${REPLY}" \
  || fail "unexpected chat reply"

CALLER=(-H "Authorization: Bearer ${CIO_KEY}")
CHAT="{\"model\":\"petrosa-primary\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}]}"
CODE="$(curl -s -o /dev/null -w '%{http_code}' "${CALLER[@]}" -H 'Content-Type: application/json' "${BASE}/v1/chat/completions" -d "${CHAT}")"
[ "${CODE}" = "200" ] || fail "the cio caller key got ${CODE} on chat completions"
for route in /key/generate /model/new /config/update; do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST "${CALLER[@]}" -H 'Content-Type: application/json' "${BASE}${route}" -d '{}')"
  [ "${CODE}" = "403" ] || fail "the cio caller key got ${CODE} on POST ${route}"
done
for route in /model/info /health; do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' "${CALLER[@]}" "${BASE}${route}")"
  [ "${CODE}" = "403" ] || fail "the cio caller key got ${CODE} on GET ${route}"
done
INFO="$(curl -sf "${AUTH[@]}" "${BASE}/model/info")" || fail "/model/info failed for the master key"
for secret in "${PROVIDER_KEY}" "${MASTER_KEY}"; do
  ! grep -qF -- "${secret}" <<<"${INFO}" || fail "/model/info returned a key"
done
# gw#3: provider attempts against the mock's programmable behaviour.
PRIMARY_UP="nvidia/nemotron-3.5-super-vl-120b-a12b"
FALLBACK_UP="nvidia/nemotron-3.5-lightning-30b-a3b"
mock() { # mock <method> <path> [json]: the mock's control API, from inside its container
  docker exec "${RUN_ID}-mock" /app/.venv/bin/python -c '
import json, sys, urllib.request
method, path = sys.argv[1], sys.argv[2]
body = sys.argv[3].encode() if len(sys.argv) > 3 else None
req = urllib.request.Request("http://127.0.0.1:8080" + path, data=body, method=method,
                             headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req, timeout=5).read().decode())' "$@"
}
calls() { mock GET /_seen | python3 -c 'import json,sys; c=json.load(sys.stdin)["calls"]; print(c.get(sys.argv[1], 0), c.get(sys.argv[2], 0))' "${PRIMARY_UP}" "${FALLBACK_UP}"; }

# A slow primary goes to the fallback inside the caller's budget: the attempt timeout is
# half the remaining budget, so the fallback still has its half.
mock POST /_control "{\"reset\": true, \"rules\": {\"${PRIMARY_UP}\": {\"delay_s\": 8, \"times\": 1}}}" >/dev/null
BEFORE="$(calls)"
REPLY="$(curl -s -w ' HTTP%{http_code} %{time_total}' "${CALLER[@]}" -H 'Content-Type: application/json' -H 'x-petrosa-deadline-ms: 6000' "${BASE}/v1/chat/completions" -d "${CHAT}")"
grep -q " HTTP200 " <<<"${REPLY}" || fail "a slow primary did not fall back: ${REPLY: -40}"
SLOW_S="${REPLY##* }"
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 6.0 else 1)" "${SLOW_S}" || fail "the fallback answered after the 6 s budget (${SLOW_S}s)"
read -r P0 F0 <<<"${BEFORE}"; read -r P1 F1 <<<"$(calls)"
[ "$((P1 - P0)) $((F1 - F0))" = "1 1" ] || fail "slow primary: expected one call to each model, got $((P1 - P0)) and $((F1 - F0))"

# A provider 429 throttles the ACCOUNT: one attempt, no fallback (it shares the account),
# and while the cool-down lasts the next request is refused at once with no provider call.
mock POST /_control "{\"reset\": true, \"rules\": {\"${PRIMARY_UP}\": {\"status\": 429, \"headers\": {\"retry-after\": \"3\", \"x-ratelimit-remaining-requests\": \"0\"}, \"times\": 1}}}" >/dev/null
read -r P0 F0 <<<"$(calls)"
CODE="$(curl -s -o /dev/null -w '%{http_code}' "${CALLER[@]}" -H 'Content-Type: application/json' "${BASE}/v1/chat/completions" -d "${CHAT}")"
[ "${CODE}" = "429" ] || fail "the throttled request answered ${CODE}, not the provider's 429"
read -r P1 F1 <<<"$(calls)"
[ "$((P1 - P0)) $((F1 - F0))" = "1 0" ] || fail "429: expected one primary call and no fallback, got $((P1 - P0)) and $((F1 - F0))"
REPLY="$(curl -s -w ' HTTP%{http_code} %{time_total}' "${CALLER[@]}" -H 'Content-Type: application/json' "${BASE}/v1/chat/completions" -d "${CHAT}")"
grep -q " HTTP503 " <<<"${REPLY}" && grep -q "account_throttled" <<<"${REPLY}" \
  || fail "a request during the cool-down was not refused with 503 account_throttled: ${REPLY: -60}"
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 1.0 else 1)" "${REPLY##* }" || fail "account_throttled took ${REPLY##* }s"
read -r P2 F2 <<<"$(calls)"
[ "$((P2 - P1)) $((F2 - F1))" = "0 0" ] || fail "a refused request still reached the provider"
THROTTLE_LOG="$(docker logs "${RUN_ID}-gateway" 2>&1)" # captured: grep -q on a pipe trips pipefail
grep -q "provider throttled: rpm=5.0 .*x-ratelimit-remaining-requests" <<<"${THROTTLE_LOG}" \
  || fail "the 429 was not logged with the halved rate and its rate headers"
METRICS="$(docker exec "${RUN_ID}-gateway" /app/.venv/bin/python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:9090/metrics", timeout=5).read().decode())')" \
  || fail "no metrics on :9090"
for expected in \
  'llm_gateway_provider_throttles_total{alias="petrosa-primary"} 1.0' \
  'llm_gateway_rate_rpm 5.0' \
  'llm_gateway_attempts_total{alias="petrosa-fallback",caller="cio",class="trading",outcome="account_throttled"} 1.0' \
  'llm_gateway_attempts_total{alias="petrosa-fallback",caller="cio",class="trading",outcome="started"} 1.0'; do
  grep -qF -- "${expected}" <<<"${METRICS}" \
    || fail "metrics lack: ${expected}; have: $(grep -E '^llm_gateway_(attempts|provider|rate_rpm)' <<<"${METRICS}" | tr '\n' ' ')"
done
for secret in "${PROVIDER_KEY}" "${MASTER_KEY}" "${CIO_KEY}" "${PROMPT}"; do
  ! grep -qF -- "${secret}" <<<"${METRICS}" || fail "a key or the prompt text appeared in the metrics"
done
sleep 3.5 # the provider's retry-after
mock POST /_control '{"reset": true}' >/dev/null

# Requests with a 100 ms budget run while the rate window has room; once it is full
# (10 per window in config/scheduling.yaml) the next one is refused at once.
REFUSED=""
for _ in $(seq 1 15); do
  REPLY="$(curl -s -w ' HTTP%{http_code}' "${CALLER[@]}" -H 'Content-Type: application/json' -H 'x-petrosa-deadline-ms: 100' "${BASE}/v1/chat/completions" -d "${CHAT}")"
  if grep -q "HTTP503" <<<"${REPLY}"; then REFUSED="${REPLY}"; break; fi
  grep -q "HTTP200" <<<"${REPLY}" || fail "a request inside the rate failed: ${REPLY: -12}"
done
grep -q "deadline_unmeetable" <<<"${REFUSED}" \
  || fail "a full rate window did not refuse an unmeetable deadline with 503 deadline_unmeetable"

LOGS="$(docker logs "${RUN_ID}-gateway" 2>&1)"
for secret in "${PROVIDER_KEY}" "${MASTER_KEY}" "${CIO_KEY}" "${PROMPT}"; do
  ! grep -qF -- "${secret}" <<<"${LOGS}" || fail "a key or the prompt text appeared in the logs"
done
grep -q "PROVIDER_API_KEY loaded from .* (sha256:" <<<"${LOGS}" || fail "no provider key fingerprint in the logs"

IMAGE_ENV="$(docker image inspect "${IMAGE}" --format '{{json .Config.Env}}')"
! grep -qE "PROVIDER_API_KEY|LITELLM_MASTER_KEY" <<<"${IMAGE_ENV}" || fail "a key variable is baked into the image"

echo "smoke OK: aliases=[${IDS}] health=${COUNTS} auth=401-without-key caller=200-chat/403-admin deadline=503 slow-primary=fallback 429=one-attempt+account_throttled metrics=:9090 probes=200-no-key logs=no-keys"
