#!/usr/bin/env bash
# Non-blocking mypy pre-commit hook (PetroSa2/petrosa_k8s#1052 epic / #180).
#
# CI's own "Run Mypy" step is `continue-on-error: true` today — mypy blocking
# is a separate, later epic step (step 1/2: baseline the per-repo error count,
# then flip to blocking, shrink-only). Until that baseline lands, this hook
# must mirror CI's non-blocking behavior: run mypy, surface findings, but
# never fail the (currently unconditionally blocking) pre-commit gate.
set -uo pipefail

mypy . "$@"
status=$?

if [ "${status}" -ne 0 ]; then
  echo "⚠️ Mypy found issues but continuing (non-blocking; petrosa_k8s#1052 epic step 2 will baseline + flip to blocking)"
fi

exit 0
