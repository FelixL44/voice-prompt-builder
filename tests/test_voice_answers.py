"""Tests for answering a follow-up question out loud.

A spoken answer is short, so Whisper has little context to work with. The
question it is answering is passed as decoding context, which narrows the
vocabulary considerably.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import MAX_CONTEXT_CHARS, FRONTEND_DIR, Settings, for_request
from backend.main import app
from tests.conftest import requires_ffmpeg, requires_say

client = TestClient(app)


# ---------------------------------------------------------------------------
# Decoding context
# ---------------------------------------------------------------------------


def test_context_reaches_the_decoding_hint() -> None:
    result = for_request(Settings(), context="Who is this for?")

    assert result.initial_prompt.startswith("Who is this for?")


def test_vocabulary_comes_before_context() -> None:
    """Whisper weights the start of the hint most; names matter more."""
    result = for_request(Settings(), vocabulary="Kubernetes", context="Who is this for?")

    assert result.initial_prompt.index("Kubernetes") < result.initial_prompt.index("Who is")


def test_context_is_truncated() -> None:
    """The hint shares a 224-token window with the audio context."""
    result = for_request(Settings(), context="word " * 500)

    used = result.initial_prompt.split(". A spoken brain-dump")[0]
    assert len(used) <= MAX_CONTEXT_CHARS + 1   # Plus the sentence-ending dot.


def test_punctuation_is_not_doubled() -> None:
    result = for_request(Settings(), context="Who is this for?")

    assert "?." not in result.initial_prompt


def test_a_context_free_request_is_unchanged() -> None:
    base = Settings()

    assert for_request(base, context="   ") is base


# ---------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------


@requires_ffmpeg
@requires_say
def test_an_answer_can_be_transcribed_with_its_question(sample_webm: Path) -> None:
    response = client.post(
        "/transcribe",
        files={"audio": ("answer.webm", sample_webm.read_bytes(), "audio/webm")},
        data={"model": "tiny", "context": "Who is this for?"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["text"]


@requires_ffmpeg
@requires_say
def test_the_job_route_accepts_context_too(sample_webm: Path) -> None:
    """The UI uses the job route, so it must take the same fields."""
    response = client.post(
        "/jobs/transcribe",
        files={"audio": ("answer.webm", sample_webm.read_bytes(), "audio/webm")},
        data={"model": "tiny", "context": "Who is this for?"},
    )

    assert response.status_code == 202
    assert response.json()["job_id"]


@requires_ffmpeg
@requires_say
def test_a_short_answer_transcribes_with_context(tmp_path: Path) -> None:
    """A short spoken reply still transcribes when a question is supplied.

    Deliberately not named "improves": measured on this project, the question
    made no difference on its own. Vocabulary and model size are the levers
    that do. This asserts only that supplying context does not break anything.
    """
    spoken = tmp_path / "answer.wav"
    subprocess.run(
        ["say", "-o", str(spoken), "--data-format=LEI16@16000",
         "Senior backend engineers evaluating a migration."],
        check=True, capture_output=True, timeout=60,
    )

    response = client.post(
        "/transcribe",
        files={"audio": ("answer.wav", spoken.read_bytes(), "audio/wav")},
        data={"model": "tiny", "context": "Who is this for?"},
    )

    assert response.status_code == 200
    text = response.json()["text"].strip()

    # Deliberately not asserting particular words: `tiny` mishears this clip
    # some of the time, which made an earlier version of this test flaky. What
    # is being checked is that context is accepted and a short clip round-trips.
    assert len(text) > 10, f"expected a transcript, got {text!r}"


# ---------------------------------------------------------------------------
# Frontend behaviour
# ---------------------------------------------------------------------------


def _js_function(name: str) -> str:
    source = (FRONTEND_DIR / "app.js").read_text()
    return f"function {name}(" + source.split(f"function {name}(")[1].split("\n}")[0] + "\n}"


requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to run the frontend logic"
)


@requires_node
@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("Fast. Cheap. Local.", ["Fast", "Cheap", "Local"]),
        ("No React", ["No React"]),
        ("It must be fast! And cheap?", ["It must be fast", "And cheap"]),
        ("One thing.\nAnother thing.", ["One thing", "Another thing"]),
        ("   ", []),
    ],
)
def test_a_spoken_list_answer_becomes_one_item_per_sentence(
    spoken: str, expected: list[str]
) -> None:
    """People say list items as sentences, not as newline-separated lines.

    Runs the real function out of app.js rather than a copy of its regex, so
    the test cannot pass against logic the app does not use.
    """
    script = (
        _js_function("splitSpokenList")
        + f"\nconsole.log(JSON.stringify(splitSpokenList({json.dumps(spoken)})));"
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )

    assert json.loads(result.stdout) == expected


def test_every_field_has_a_microphone() -> None:
    """A question you cannot answer aloud defeats the feature."""
    source = (FRONTEND_DIR / "app.js").read_text()

    assert "buildFieldMic(name, question)" in source
    # Built for every field in the render loop, not only the missing ones.
    render = source.split("function renderFields(")[1].split("\n}")[0]
    assert "buildFieldMic" in render
    assert "for (const [name, spec] of Object.entries(FIELDS))" in render


def test_answers_are_appended_not_replaced() -> None:
    """A second answer must not destroy the first, nor a typed correction."""
    body = _js_function("applyAnswer")

    assert "existing" in body
    assert "input.value = existing ?" in body or "[existing," in body


def test_only_one_recording_at_a_time() -> None:
    """Two recorders would fight over the microphone and the whisper slot."""
    body = _js_function("startFieldAnswer")

    assert "if (answer.field ||" in body
    assert "rec.recorder" in body, "must also refuse while the main recorder runs"
