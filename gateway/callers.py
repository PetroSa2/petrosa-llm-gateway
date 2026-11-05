"""Caller identity without a database (petrosa-llm-gateway#2).

LiteLLM virtual keys need Postgres, and the gateway owns no database, so callers are
identified by static keys read from mounted Secret files: one file per caller in
``CALLER_KEYS_DIR`` (file name = caller name), each caller mapped to its priority class
in ``config/scheduling.yaml``. Only fingerprints are logged.

With ``general_settings.custom_auth`` set, LiteLLM trusts what this hook returns on every
route and runs none of its own route checks. So this module enforces them: a caller key
is accepted only on the inference routes below; everything else (``/key``, ``/config``,
``/model``, ``/health``, the UI...) needs the master key, which only the gateway holds.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from gateway.entrypoint import CredentialError, fingerprint, read_secret

logger = logging.getLogger("gateway.callers")

DEFAULT_CALLER_KEYS_DIR = "/var/run/secrets/llm-gateway/callers"
DEFAULT_SCHEDULING_CONFIG = "/etc/llm-gateway/scheduling.yaml"

#: Routes a caller key may use: inference and the alias listing, nothing administrative.
CALLER_ROUTES = frozenset(
    {
        "/chat/completions",
        "/v1/chat/completions",
        "/completions",
        "/v1/completions",
        "/models",
        "/v1/models",
    }
)


class AuthRefused(Exception):
    """A request without a valid key (401) or with a caller key on a non-caller route (403)."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class Identity:
    """Who is calling: a caller with a priority class, or the gateway's own admin."""

    caller: str
    priority_class: str | None
    admin: bool


@dataclass(frozen=True)
class CallerKeys:
    """The accepted keys: caller key -> (caller, class), and the master key."""

    by_key: Mapping[str, tuple[str, str]]
    master_key: str

    def authorize(self, path: str, presented: str | None) -> Identity:
        """The identity for ``presented`` on ``path``, or ``AuthRefused``."""
        key = (presented or "").removeprefix("Bearer ").strip()
        if not key:
            raise AuthRefused(401, "no key")
        if hmac.compare_digest(key, self.master_key):
            return Identity(caller="gateway-admin", priority_class=None, admin=True)
        for candidate, (caller, cls) in self.by_key.items():
            if hmac.compare_digest(key, candidate):
                if path.rstrip("/") not in CALLER_ROUTES:
                    raise AuthRefused(403, f"caller {caller} may not use {path}")
                return Identity(caller=caller, priority_class=cls, admin=False)
        raise AuthRefused(401, "unknown key")


def load_caller_keys(
    environ: Mapping[str, str], master_key: str
) -> tuple[CallerKeys, list[str]]:
    """Read each configured caller's key file; return the keys and fingerprint log lines."""
    config_path = Path(environ.get("GATEWAY_SCHEDULING", DEFAULT_SCHEDULING_CONFIG))
    callers: Mapping[str, str] = yaml.safe_load(config_path.read_text())["callers"]
    keys_dir = Path(environ.get("CALLER_KEYS_DIR", DEFAULT_CALLER_KEYS_DIR))
    by_key: dict[str, tuple[str, str]] = {}
    lines = []
    for caller, cls in sorted(callers.items()):
        path = keys_dir / caller
        if not path.exists():
            lines.append(f"caller {caller}: no key file at {path}; caller disabled")
            continue
        key = read_secret(path)
        if hmac.compare_digest(key, master_key):
            raise CredentialError(f"caller {caller}'s key equals the master key")
        if key in by_key:
            raise CredentialError(f"callers {by_key[key][0]} and {caller} share a key")
        by_key[key] = (caller, cls)
        lines.append(f"caller {caller} ({cls}) loaded from {path} ({fingerprint(key)})")
    return CallerKeys(by_key=by_key, master_key=master_key), lines


_KEYS: CallerKeys | None = None


def _keys() -> CallerKeys:
    """The process-wide keys, loaded on first use from the mounted files."""
    global _KEYS
    if _KEYS is None:
        master = os.environ.get("LITELLM_MASTER_KEY", "")
        if not master:
            raise CredentialError("LITELLM_MASTER_KEY is not set")
        _KEYS, lines = load_caller_keys(os.environ, master)
        for line in lines:
            logger.info(line)
    return _KEYS


async def user_api_key_auth(
    request, api_key: str
):  # pragma: no cover - LiteLLM adapter
    """LiteLLM ``custom_auth`` hook: map the key to an identity, refuse otherwise."""
    from fastapi import HTTPException
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    try:
        identity = _keys().authorize(request.url.path, api_key)
    except AuthRefused as refused:
        raise HTTPException(status_code=refused.status, detail=refused.reason) from None
    if identity.admin:
        return UserAPIKeyAuth(
            api_key=api_key,
            user_id=identity.caller,
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
    return UserAPIKeyAuth(
        user_id=identity.caller,
        user_role=LitellmUserRoles.INTERNAL_USER,
        metadata={"caller": identity.caller, "priority_class": identity.priority_class},
    )
