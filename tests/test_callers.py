"""Caller identity from mounted static keys; callers never reach admin routes (gw#2)."""

from __future__ import annotations

import pytest

from gateway.callers import AuthRefused, CallerKeys, load_caller_keys
from gateway.entrypoint import CredentialError, fingerprint

MASTER = "master-key-only-the-gateway-holds-0001"
CIO = "cio-caller-key-0002"
HOLMES = "holmes-caller-key-0003"


@pytest.fixture
def keys():
    return CallerKeys(
        by_key={CIO: ("cio", "trading"), HOLMES: ("holmes", "ops")}, master_key=MASTER
    )


def test_the_cio_key_is_tagged_caller_cio_class_trading(keys):
    identity = keys.authorize("/v1/chat/completions", f"Bearer {CIO}")

    assert (identity.caller, identity.priority_class, identity.admin) == (
        "cio",
        "trading",
        False,
    )
    assert keys.authorize("/chat/completions", HOLMES).priority_class == "ops"


@pytest.mark.parametrize(
    "path",
    [
        "/key/generate",
        "/key/list",
        "/config/update",
        "/model/new",
        "/model/info",
        "/v1/model/info",
        "/health",
        "/user/new",
        "/global/spend",
        "/ui",
    ],
)
def test_a_caller_key_is_refused_on_admin_and_info_routes(keys, path):
    with pytest.raises(AuthRefused) as refused:
        keys.authorize(path, CIO)

    assert refused.value.status == 403


def test_the_master_key_is_admin_everywhere(keys):
    for path in ("/key/generate", "/model/info", "/v1/chat/completions"):
        assert keys.authorize(path, f"Bearer {MASTER}").admin is True


@pytest.mark.parametrize("presented", [None, "", "Bearer ", "wrong-key", CIO + "x"])
def test_missing_or_unknown_keys_are_401(keys, presented):
    with pytest.raises(AuthRefused) as refused:
        keys.authorize("/v1/chat/completions", presented)

    assert refused.value.status == 401


def test_a_refusal_never_contains_the_presented_key(keys):
    with pytest.raises(AuthRefused) as refused:
        keys.authorize("/key/generate", CIO)

    assert CIO not in str(refused.value)


def _setup(tmp_path, files):
    (tmp_path / "callers").mkdir()
    for name, value in files.items():
        (tmp_path / "callers" / name).write_text(value + "\n")
    config = tmp_path / "scheduling.yaml"
    config.write_text("callers:\n  cio: trading\n  holmes: ops\n")
    return {
        "CALLER_KEYS_DIR": str(tmp_path / "callers"),
        "GATEWAY_SCHEDULING": str(config),
    }


def test_keys_load_from_mounted_files_and_log_only_fingerprints(tmp_path):
    env = _setup(tmp_path, {"cio": CIO, "holmes": HOLMES, "stranger": "x" * 20})

    keys, lines = load_caller_keys(env, MASTER)

    assert keys.by_key == {CIO: ("cio", "trading"), HOLMES: ("holmes", "ops")}
    text = "\n".join(lines)
    assert CIO not in text and HOLMES not in text
    assert fingerprint(CIO) in text and fingerprint(HOLMES) in text
    assert "stranger" not in text  # only configured callers are loaded


def test_a_caller_without_a_key_file_is_disabled(tmp_path):
    env = _setup(tmp_path, {"cio": CIO})

    keys, lines = load_caller_keys(env, MASTER)

    assert keys.by_key == {CIO: ("cio", "trading")}
    assert any("holmes" in line and "disabled" in line for line in lines)


def test_a_caller_key_equal_to_the_master_key_is_refused(tmp_path):
    env = _setup(tmp_path, {"cio": MASTER})

    with pytest.raises(CredentialError):
        load_caller_keys(env, MASTER)


def test_two_callers_sharing_a_key_is_refused(tmp_path):
    env = _setup(tmp_path, {"cio": CIO, "holmes": CIO})

    with pytest.raises(CredentialError):
        load_caller_keys(env, MASTER)


def test_process_keys_load_once_from_the_environment(tmp_path, monkeypatch):
    from gateway import callers

    env = _setup(tmp_path, {"cio": CIO})
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LITELLM_MASTER_KEY", MASTER)
    monkeypatch.setattr(callers, "_KEYS", None)

    first = callers._keys()

    assert first.by_key == {CIO: ("cio", "trading")}
    assert callers._keys() is first  # cached for the process


def test_process_keys_need_the_master_key(monkeypatch):
    from gateway import callers

    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    monkeypatch.setattr(callers, "_KEYS", None)

    with pytest.raises(CredentialError):
        callers._keys()
