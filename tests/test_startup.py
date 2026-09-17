"""Tests for startup warmup and the readiness reporting in /health."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import analyze as analyze_module
from backend import main as main_module
from backend import transcribe as transcribe_module
from backend.analyze import model_matches, ollama_model_available, ollama_models
from backend.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def test_warmup_runs_on_startup_when_enabled(monkeypatch) -> None:
    """The model should load at startup, not on the user's first recording."""
    warmed: list[str] = []

    def fake_warm(settings=None):  # type: ignore[no-untyped-def]
        warmed.append((settings or get_settings()).whisper_model)

    monkeypatch.setenv("VPB_WARMUP", "true")
    monkeypatch.setenv("VPB_WHISPER_MODEL", "tiny")
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "warm_up", fake_warm)

    try:
        with TestClient(main_module.app):
            pass  # Entering the context runs the lifespan.
    finally:
        get_settings.cache_clear()

    assert warmed == ["tiny"]


def test_warmup_is_skipped_when_disabled(monkeypatch) -> None:
    warmed: list[str] = []
    monkeypatch.setenv("VPB_WARMUP", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "warm_up", lambda *a, **k: warmed.append("x"))

    try:
        with TestClient(main_module.app):
            pass
    finally:
        get_settings.cache_clear()

    assert warmed == []


def test_a_failing_warmup_does_not_stop_the_server(monkeypatch) -> None:
    """A model hub that cannot be reached must not prevent startup."""
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("no network")

    monkeypatch.setattr(transcribe_module, "get_model", explode)

    # warm_up swallows and logs; the call must simply return.
    transcribe_module.warm_up(Settings(whisper_model="tiny"))

    assert transcribe_module.model_is_warming() is False


def test_warming_flag_clears_after_warmup(monkeypatch) -> None:
    monkeypatch.setattr(transcribe_module, "get_model", lambda *a, **k: object())

    transcribe_module.warm_up(Settings(whisper_model="tiny"))

    assert transcribe_module.model_is_warming() is False


# ---------------------------------------------------------------------------
# Ollama model availability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configured,available,expected",
    [
        ("llama3.2:3b", "llama3.2:3b", True),
        ("llama3.2", "llama3.2:latest", True),   # Untagged pulls become :latest.
        ("llama3.2:3b", "llama3.2:1b", False),
        ("qwen2.5:7b", "llama3.2:3b", False),
        ("llama3.2:latest", "llama3.2", False),  # Not symmetric, by design.
    ],
)
def test_model_matching(configured: str, available: str, expected: bool) -> None:
    assert model_matches(configured, available) is expected


def _fake_tags(payload: object, status: int = 200):
    def fake_get(*_args: object, **_kwargs: object) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return fake_get


def test_missing_model_is_reported_even_when_ollama_is_up(monkeypatch) -> None:
    """A running Ollama without the model fails only after a long wait."""
    monkeypatch.setattr(
        httpx, "get", _fake_tags({"models": [{"name": "some-other-model:7b"}]})
    )
    settings = Settings(ollama_model="llama3.2:3b")

    assert ollama_models(settings) == ["some-other-model:7b"]
    assert ollama_model_available(settings) is False


def test_present_model_is_reported_available(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", _fake_tags({"models": [{"name": "llama3.2:3b"}]}))

    assert ollama_model_available(Settings(ollama_model="llama3.2:3b")) is True


def test_unreachable_ollama_returns_none_not_empty(monkeypatch) -> None:
    """None means unreachable; [] means running but nothing pulled."""
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refuse)

    assert ollama_models(Settings()) is None
    assert ollama_model_available(Settings()) is False


def test_malformed_tags_response_is_survivable(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", _fake_tags({"unexpected": "shape"}))

    assert ollama_models(Settings()) == []


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_model_availability(monkeypatch) -> None:
    monkeypatch.setattr(
        analyze_module.httpx, "get", _fake_tags({"models": [{"name": "llama3.2:3b"}]})
    )
    monkeypatch.setenv("VPB_OLLAMA_MODEL", "llama3.2:3b")
    get_settings.cache_clear()

    try:
        client = TestClient(main_module.app)
        body = client.get("/health").json()
        assert body["ollama"] is True
        assert body["ollama_model_available"] is True
        assert "model_warming" in body
    finally:
        get_settings.cache_clear()


def test_health_flags_a_model_that_is_not_pulled(monkeypatch) -> None:
    monkeypatch.setattr(analyze_module.httpx, "get", _fake_tags({"models": []}))
    monkeypatch.setenv("VPB_OLLAMA_MODEL", "qwen2.5:7b")
    get_settings.cache_clear()

    try:
        client = TestClient(main_module.app)
        body = client.get("/health").json()
        assert body["ollama"] is True
        assert body["ollama_model_available"] is False
    finally:
        get_settings.cache_clear()
