"""Start the LiteLLM proxy with credentials read from mounted Secret files only.

The provider key and the proxy master key are read from files (Kubernetes Secret
volumes), passed to the proxy process through its environment, and never logged:
only a short SHA-256 fingerprint of each is written, so a rotation can be confirmed
without exposing the value. Setting either key directly in the container
environment is refused, so the key cannot come from a manifest literal.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("gateway.entrypoint")

DEFAULT_CONFIG = "/etc/llm-gateway/litellm_config.yaml"
DEFAULT_PORT = "4000"


@dataclass(frozen=True)
class MountedSecret:
    """One credential: the env var the proxy reads and the setting naming its file."""

    env_name: str
    file_setting: str
    default_path: str


SECRETS = (
    MountedSecret(
        "PROVIDER_API_KEY",
        "PROVIDER_KEY_FILE",
        "/var/run/secrets/llm-gateway/provider-api-key",
    ),
    MountedSecret(
        "LITELLM_MASTER_KEY",
        "MASTER_KEY_FILE",
        "/var/run/secrets/llm-gateway/master-key",
    ),
)


class CredentialError(RuntimeError):
    """A credential is missing, empty or was supplied outside a mounted file."""


def fingerprint(value: str) -> str:
    """A short, non-reversible identifier of a credential, safe to log."""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def read_secret(path: Path) -> str:
    """The stripped content of a mounted Secret file; refuse a missing or empty one."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CredentialError(
            f"cannot read the mounted secret file {path}: {exc.strerror}"
        ) from None
    if not value:
        raise CredentialError(f"the mounted secret file {path} is empty")
    return value


def proxy_environment(environ: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """The proxy's environment plus one log line per credential (fingerprint only)."""
    env = dict(environ)
    lines = []
    for secret in SECRETS:
        if secret.env_name in environ:
            raise CredentialError(
                f"{secret.env_name} is set in the environment; mount it as a "
                f"Secret file and point {secret.file_setting} at it instead"
            )
        path = Path(environ.get(secret.file_setting, secret.default_path))
        value = read_secret(path)
        env[secret.env_name] = value
        lines.append(f"{secret.env_name} loaded from {path} ({fingerprint(value)})")
    return env, lines


def proxy_command(environ: Mapping[str, str], extra: Sequence[str]) -> list[str]:
    """The LiteLLM proxy command line."""
    return [
        "litellm",
        "--config",
        environ.get("GATEWAY_CONFIG", DEFAULT_CONFIG),
        "--port",
        environ.get("GATEWAY_PORT", DEFAULT_PORT),
        *extra,
    ]


ExecFn = Callable[[str, list[str], dict[str, str]], object]


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    execute: ExecFn = os.execvpe,
) -> int:
    """Load the credentials, log their fingerprints and replace this process by the proxy."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    environ = os.environ if environ is None else environ
    try:
        env, lines = proxy_environment(environ)
    except CredentialError as exc:
        logger.error("refusing to start: %s", exc)
        return 2
    for line in lines:
        logger.info(line)
    command = proxy_command(environ, list(sys.argv[1:] if argv is None else argv))
    logger.info("starting the proxy: %s", " ".join(command))
    execute(command[0], command, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
