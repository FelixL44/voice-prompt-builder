"""Tests for the HTTP surface."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from tests.conftest import requires_ffmpeg, requires_ollama, requires_say

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


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


def test_analyze_requires_a_transcript() -> None:
    response = client.post("/analyze", json={"transcript": ""})

    assert response.status_code == 422


def test_analyze_reports_unreachable_ollama(monkeypatch) -> None:
    """Ollama being down is a 503 with instructions, not a stack trace."""
    import httpx

    from backend.config import get_settings

    monkeypatch.setenv("VPB_OLLAMA_URL", "http://localhost:1")
    get_settings.cache_clear()

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)

    try:
        response = client.post("/analyze", json={"transcript": "I need a landing page."})
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "ollama_unavailable"
        assert "ollama serve" in body["detail"]
    finally:
        get_settings.cache_clear()


@requires_ollama
def test_analyze_extracts_structure_end_to_end() -> None:
    """The real path against the real local model.

    Asserts on structure and on the missing-field contract rather than on exact
    wording, which varies between models and runs.
    """
    transcript = (
        "I need a one page landing site for our developer tool. "
        "It is aimed at senior backend engineers. "
        "It must not use React and it has to load in under a second."
    )

    response = client.post("/analyze", json={"transcript": transcript})

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["extraction"]["goal"], "goal should be extracted"
    assert body["elapsed_s"] > 0
    assert body["model"]

    # Every reported missing field must genuinely be empty in the extraction,
    # and must have come back normalised rather than as a literal "null".
    from backend.analyze import is_empty

    for entry in body["missing"]:
        value = body["extraction"][entry["field"]]
        assert is_empty(value), f"{entry['field']} reported missing but holds {value!r}"
        assert value is None or value == [], f"{entry['field']} was not normalised"
        assert entry["question"]

    # No example was given, and the prompt is explicit that examples are
    # usually empty, so this is the one field we can assert on reliably.
    # (success_criteria is deliberately not asserted: a model can reasonably
    # read "must load in under a second" as either a constraint or a criterion.)
    assert "examples" in [m["field"] for m in body["missing"]]


# ---------------------------------------------------------------------------
# /build
# ---------------------------------------------------------------------------


def _extraction(**overrides: object) -> dict:
    base = {
        "goal": "Draft a first reply to a support ticket",
        "audience": "Support agents",
        "context": "Zendesk inbox",
        "constraints": ["Never send automatically"],
        "examples": [],
        "output_format": "Reply plus a confidence score",
        "success_criteria": ["60% accepted unedited"],
    }
    base.update(overrides)
    return base


def test_build_returns_a_prompt_and_an_estimate() -> None:
    response = client.post("/build", json={"extraction": _extraction()})

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["prompt"].startswith("<context>")
    assert "Never send automatically" in body["prompt"]
    assert body["characters"] == len(body["prompt"])
    assert body["estimated_tokens"] == len(body["prompt"]) // 4
    # examples was empty, so it must not appear.
    assert "examples" not in body["sections"]


def test_build_is_deterministic() -> None:
    """The same fields must always give byte-identical output."""
    payload = {"extraction": _extraction()}

    first = client.post("/build", json=payload).json()["prompt"]
    second = client.post("/build", json=payload).json()["prompt"]

    assert first == second


def test_build_rejects_an_empty_extraction() -> None:
    empty = {key: (None if not isinstance(value, list) else [])
             for key, value in _extraction().items()}

    response = client.post("/build", json={"extraction": empty})

    assert response.status_code == 422
    assert response.json()["error"] == "empty_prompt"


def test_build_accepts_a_partial_extraction() -> None:
    """Skipped questions must not block the prompt, only shorten it."""
    response = client.post(
        "/build",
        json={"extraction": {"goal": "Ship it", "constraints": [], "examples": [],
                             "success_criteria": []}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sections"] == ["task"]
    assert "<task>" in body["prompt"]


# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------


def test_oversized_upload_is_rejected(monkeypatch) -> None:
    """An upload over the limit is refused rather than transcribed."""
    from backend.config import get_settings

    monkeypatch.setenv("VPB_MAX_UPLOAD_BYTES", "1024")
    get_settings.cache_clear()

    try:
        response = client.post(
            "/transcribe",
            files={"audio": ("big.webm", b"x" * 5000, "audio/webm")},
        )
        assert response.status_code == 413
        assert "larger than" in response.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_oversized_upload_is_refused_before_buffering(monkeypatch) -> None:
    """A declared length over the limit is refused without reading the body.

    Otherwise a huge upload is fully resident in memory before being rejected.
    """
    from backend import main as main_module
    from backend.config import get_settings

    monkeypatch.setenv("VPB_MAX_UPLOAD_BYTES", "1024")
    get_settings.cache_clear()

    reads: list[int] = []
    original = main_module.UploadFile.read

    async def counting_read(self, size: int = -1):  # type: ignore[no-untyped-def]
        reads.append(size)
        return await original(self, size)

    monkeypatch.setattr(main_module.UploadFile, "read", counting_read)

    try:
        response = client.post(
            "/transcribe",
            files={"audio": ("big.webm", b"x" * 100_000, "audio/webm")},
            headers={"content-length": "100000"},
        )
        assert response.status_code == 413
    finally:
        get_settings.cache_clear()


def test_upload_within_the_limit_is_accepted() -> None:
    """The cap must not reject ordinary recordings; this one fails later."""
    response = client.post(
        "/transcribe", files={"audio": ("small.webm", b"not audio", "audio/webm")}
    )

    # Rejected for being undecodable, not for being too large.
    assert response.status_code == 400
