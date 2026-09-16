"""Tests for the HTTP surface."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from tests.conftest import requires_ffmpeg, requires_say

client = TestClient(app)


def test_health_reports_ffmpeg_and_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["ffmpeg"], bool)
    assert body["whisper_model"]
    # The UI builds its model dropdown from this list.
    assert "small" in body["available_models"]


def test_index_serves_the_recording_page() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Voice Prompt Builder" in response.text


def test_transcribe_requires_a_file() -> None:
    response = client.post("/transcribe")

    assert response.status_code == 422


@requires_ffmpeg
def test_transcribe_rejects_undecodable_audio() -> None:
    """Garbage in gives a 400 with a readable message, not a 500."""
    response = client.post(
        "/transcribe", files={"audio": ("junk.webm", b"not audio", "audio/webm")}
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "conversion_failed"
    assert body["detail"]


@requires_ffmpeg
@requires_say
def test_transcribe_returns_text(monkeypatch, sample_webm: Path) -> None:
    """The full route, pinned to the tiny model so it stays quick."""
    from backend.config import get_settings

    monkeypatch.setenv("VPB_WHISPER_MODEL", "tiny")
    monkeypatch.setenv("VPB_LANGUAGE", "en")
    get_settings.cache_clear()

    try:
        response = client.post(
            "/transcribe",
            files={"audio": ("recording.webm", sample_webm.read_bytes(), "audio/webm")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "fox" in body["text"].lower()
        assert body["model"] == "tiny"
        assert body["duration"] > 0
        assert body["segments"]
    finally:
        get_settings.cache_clear()


def test_transcribe_rejects_unknown_model() -> None:
    """A bad model name is a 400, not an attempt to download something odd."""
    response = client.post(
        "/transcribe",
        files={"audio": ("a.webm", b"x", "audio/webm")},
        data={"model": "large-v3"},
    )

    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


@requires_ffmpeg
@requires_say
def test_vocabulary_improves_rare_words(sample_webm: Path) -> None:
    """The vocabulary hint reaches the decoder and the request still succeeds."""
    response = client.post(
        "/transcribe",
        files={"audio": ("recording.webm", sample_webm.read_bytes(), "audio/webm")},
        data={"model": "tiny", "vocabulary": "brown fox, lazy dog"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "tiny"
    assert "fox" in body["text"].lower()


def test_static_assets_are_revalidated() -> None:
    """Stale JS against fresh HTML looks like a broken UI, so force revalidation.

    Without Cache-Control the browser applies heuristic caching and can keep
    running an old app.js after an edit.
    """
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers.get("etag"), "ETag keeps revalidation cheap"


def test_index_is_revalidated() -> None:
    response = client.get("/")

    assert response.headers["cache-control"] == "no-cache"


def test_frontend_and_backend_agree_on_models() -> None:
    """The dropdown's built-in fallback list must match what the API accepts.

    The UI populates itself before /health answers, so a mismatch would offer
    a model the backend then rejects.
    """
    from backend.config import ALLOWED_MODELS, FRONTEND_DIR

    app_js = (FRONTEND_DIR / "app.js").read_text()
    speed_table = app_js.split("const MODEL_SPEED = {")[1].split("};")[0]

    for name in ALLOWED_MODELS:
        assert f"{name}:" in speed_table, f"{name} missing from MODEL_SPEED"
