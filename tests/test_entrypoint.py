"""The entrypoint reads both keys from mounted files and never logs them (petrosa-llm-gateway#1)."""

from __future__ import annotations

import logging

import pytest

from gateway import entrypoint

PROVIDER_KEY = "test-provider-key-0123456789abcdef"
MASTER_KEY = "test-master-key-fedcba9876543210"


@pytest.fixture
def mounted(tmp_path):
    provider = tmp_path / "provider-api-key"
    master = tmp_path / "master-key"
    provider.write_text(PROVIDER_KEY + "\n")
    master.write_text(MASTER_KEY)
    return {
        "PROVIDER_KEY_FILE": str(provider),
        "MASTER_KEY_FILE": str(master),
        "PATH": "/usr/bin",
    }


def run(environ, argv=()):
    calls = []
    code = entrypoint.main(
        list(argv),
        environ,
        lambda file, args, env: calls.append((file, args, env)),
    )
    return code, calls


def test_starts_the_proxy_with_both_keys_from_files(mounted):
    code, calls = run(mounted, ["--num_workers", "1"])

    assert code == 0
    [(file, args, env)] = calls
    assert file == "litellm"
    assert args == [
        "litellm",
        "--config",
        entrypoint.DEFAULT_CONFIG,
        "--port",
        entrypoint.DEFAULT_PORT,
        "--num_workers",
        "1",
    ]
    assert env["PROVIDER_API_KEY"] == PROVIDER_KEY
    assert env["LITELLM_MASTER_KEY"] == MASTER_KEY
    assert env["PATH"] == "/usr/bin"


def test_config_and_port_can_be_overridden(mounted):
    _, [(_, args, _)] = run(
        {**mounted, "GATEWAY_CONFIG": "/tmp/c.yaml", "GATEWAY_PORT": "4100"}
    )

    assert args[1:5] == ["--config", "/tmp/c.yaml", "--port", "4100"]


def test_logs_fingerprints_and_never_the_keys(mounted, caplog):
    caplog.set_level(logging.INFO, logger="gateway.entrypoint")

    run(mounted)

    text = caplog.text
    assert PROVIDER_KEY not in text
    assert MASTER_KEY not in text
    assert entrypoint.fingerprint(PROVIDER_KEY) in text
    assert entrypoint.fingerprint(MASTER_KEY) in text


def test_fingerprint_is_short_and_stable():
    first = entrypoint.fingerprint("abc")

    assert first == entrypoint.fingerprint("abc")
    assert first != entrypoint.fingerprint("abd")
    assert first.startswith("sha256:") and len(first) == len("sha256:") + 12


@pytest.mark.parametrize("name", ["PROVIDER_API_KEY", "LITELLM_MASTER_KEY"])
def test_refuses_a_key_set_in_the_environment(mounted, caplog, name):
    code, calls = run({**mounted, name: "literal-value-in-manifest"})

    assert code == 2
    assert calls == []
    assert "literal-value-in-manifest" not in caplog.text
    assert name in caplog.text


def test_refuses_a_missing_file(mounted, tmp_path):
    code, calls = run({**mounted, "PROVIDER_KEY_FILE": str(tmp_path / "absent")})

    assert (code, calls) == (2, [])


def test_refuses_an_empty_file(mounted, tmp_path):
    empty = tmp_path / "empty"
    empty.write_text(" \n")

    code, calls = run({**mounted, "MASTER_KEY_FILE": str(empty)})

    assert (code, calls) == (2, [])


def test_default_paths_are_secret_volume_paths():
    paths = {secret.default_path for secret in entrypoint.SECRETS}

    assert paths == {
        "/var/run/secrets/llm-gateway/provider-api-key",
        "/var/run/secrets/llm-gateway/master-key",
    }
