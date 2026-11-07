"""Guard: no provider key in the repository, the proxy config or the image (petrosa-llm-gateway#1)."""

from __future__ import annotations

import re
import subprocess  # nosec B404 - lists tracked files only
from pathlib import Path

import yaml

from tests import mock_provider

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "litellm_config.yaml"
ALIASES = {"petrosa-primary", "petrosa-fallback", "petrosa-ops"}

# Key shapes of the providers the gateway can front, plus any literal assignment.
KEY_SHAPES = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{32,}"),
    re.compile(
        r"(?:PROVIDER_API_KEY|LITELLM_MASTER_KEY|NVIDIA_API_KEY)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    ),
]


def tracked_files() -> list[Path]:
    out = subprocess.run(  # nosec B603 B607 - fixed git command
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    return [ROOT / name for name in out.decode().split("\0") if name]


def test_no_key_shaped_string_in_tracked_files():
    hits = []
    for path in tracked_files():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits += [
            f"{path.relative_to(ROOT)}: {p.pattern}"
            for p in KEY_SHAPES
            if p.search(text)
        ]

    assert hits == []


def test_key_shapes_catch_real_looking_keys():
    samples = [
        "nvapi-" + "A1b2C3d4" * 4,
        "sk-" + "x" * 40,
        "PROVIDER_API_KEY=" + "abcd1234" * 3,
    ]

    assert all(any(p.search(s) for p in KEY_SHAPES) for s in samples)


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_aliases_are_exactly_the_petrosa_names():
    names = [m["model_name"] for m in load_config()["model_list"]]

    assert sorted(names) == sorted(ALIASES)
    assert all(
        "*" not in m["litellm_params"]["model"] for m in load_config()["model_list"]
    )


def test_credentials_and_endpoint_come_from_the_environment_only():
    config = load_config()

    for model in config["model_list"]:
        params = model["litellm_params"]
        assert params["api_key"] == "os.environ/PROVIDER_API_KEY"
        assert params["api_base"] == "os.environ/PROVIDER_API_BASE"
    assert config["general_settings"]["master_key"] == "os.environ/LITELLM_MASTER_KEY"


def test_bodies_are_not_logged():
    assert load_config()["litellm_settings"]["turn_off_message_logging"] is True


def test_upstream_models_are_ones_the_mock_serves():
    upstream = {
        m["litellm_params"]["model"].removeprefix("openai/")
        for m in load_config()["model_list"]
    }

    assert upstream <= set(mock_provider.UPSTREAM_MODELS)


def test_dockerfile_bakes_no_key_and_pins_the_base_image():
    text = (ROOT / "Dockerfile").read_text()

    assert not re.search(
        r"^\s*(ENV|ARG)\b.*(PROVIDER_API_KEY|LITELLM_MASTER_KEY)", text, re.M
    )
    assert re.search(
        r"^FROM ghcr\.io/berriai/litellm:v[0-9.]+@sha256:[0-9a-f]{64}$", text, re.M
    )
    assert re.search(r"^USER (?!0\b|root\b)", text, re.M)


def test_image_context_holds_only_code_and_config():
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert lines[0] == "*"
    assert {line for line in lines if line.startswith("!")} == {
        "!gateway/",
        "!config/litellm_config.yaml",
        "!config/scheduling.yaml",
        # Hash-pinned runtime dependencies for the builder stage (#4); no secrets.
        "!requirements-image.txt",
    }


def test_image_requirements_are_exact_and_hash_pinned():
    text = (ROOT / "requirements-image.txt").read_text()
    entries = " ".join(
        line.strip().rstrip("\\").strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ).split()
    names = [e for e in entries if not e.startswith("--hash=")]
    hashes = [e for e in entries if e.startswith("--hash=sha256:")]
    assert names and all(
        re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][0-9A-Za-z.]*", n) for n in names
    )
    assert len(hashes) >= len(names)
    assert all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", h) for h in hashes)
