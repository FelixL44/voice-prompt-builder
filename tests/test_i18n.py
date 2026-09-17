"""Tests for German/English follow-up questions.

Only the questions are translated on the server. Field *values* follow the
language the person actually spoke, which the extraction prompt handles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.analyze import (
    QUESTIONS,
    QUESTIONS_BY_LANGUAGE,
    QUESTIONS_DE,
    find_missing,
    questions_for,
)
from backend.config import FRONTEND_DIR
from backend.main import app
from backend.schemas import Extraction

client = TestClient(app)


def test_both_languages_cover_every_field() -> None:
    """A field with no question in one language would be a dead end there."""
    for language, table in QUESTIONS_BY_LANGUAGE.items():
        assert set(table) == set(Extraction.model_fields), f"{language} is incomplete"


def test_german_questions_are_actually_german() -> None:
    """Guard against an untranslated entry being copied over."""
    for field, text in QUESTIONS_DE.items():
        assert text != QUESTIONS[field], f"{field} was not translated"
        assert "?" in text


@pytest.mark.parametrize(
    "requested,expected",
    [("de", QUESTIONS_DE), ("DE", QUESTIONS_DE), ("de-DE", QUESTIONS_DE),
     ("en", QUESTIONS), ("fr", QUESTIONS), ("", QUESTIONS), (None, QUESTIONS)],
)
def test_language_resolution(requested: str | None, expected: dict) -> None:
    """An unknown language falls back to English rather than failing."""
    assert questions_for(requested) == expected


def test_missing_fields_carry_the_requested_language() -> None:
    missing = find_missing(Extraction(goal="x"), "de")

    by_field = {m.field: m.question for m in missing}
    assert by_field["audience"] == QUESTIONS_DE["audience"]


def test_field_order_is_the_same_in_both_languages() -> None:
    """The UI renders in this order; it must not shift with language."""
    english = [m.field for m in find_missing(Extraction(), "en")]
    german = [m.field for m in find_missing(Extraction(), "de")]

    assert english == german


def test_analyze_endpoint_accepts_a_language(monkeypatch) -> None:
    import httpx

    class Stream:
        status_code = 200
        text = ""

        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return b""

        def iter_lines(self):
            yield json.dumps({"response": json.dumps({"goal": "Etwas bauen"}), "done": True})

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Stream())

    response = client.post(
        "/analyze", json={"transcript": "Ich brauche eine Webseite.", "language": "de"}
    )

    assert response.status_code == 200
    questions = {m["field"]: m["question"] for m in response.json()["missing"]}
    assert questions["audience"] == QUESTIONS_DE["audience"]


def test_analyze_defaults_to_english(monkeypatch) -> None:
    import httpx

    class Stream:
        status_code = 200
        text = ""

        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return b""

        def iter_lines(self):
            yield json.dumps({"response": json.dumps({"goal": "Build it"}), "done": True})

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Stream())

    response = client.post("/analyze", json={"transcript": "I need a website."})

    questions = {m["field"]: m["question"] for m in response.json()["missing"]}
    assert questions["audience"] == QUESTIONS["audience"]


# ---------------------------------------------------------------------------
# Frontend parity
# ---------------------------------------------------------------------------


def _frontend_languages() -> dict[str, set[str]]:
    """Language keys declared in the frontend STRINGS table."""
    source = (FRONTEND_DIR / "app.js").read_text()
    block = source.split("const STRINGS = {")[1].split("\n};")[0]
    langs: dict[str, set[str]] = {}
    for match in re.finditer(r"^  (\w+): \{(.*?)^  \},", block, re.S | re.M):
        langs[match.group(1)] = set(re.findall(r"^    ([\w.]+):", match.group(2), re.M))
    return langs


def test_frontend_offers_the_same_languages_as_the_backend() -> None:
    assert set(_frontend_languages()) == set(QUESTIONS_BY_LANGUAGE)


def test_frontend_translations_have_no_gaps() -> None:
    """A missing key renders as a raw identifier in the UI."""
    languages = _frontend_languages()
    english = languages["en"]

    for name, keys in languages.items():
        assert keys == english, (
            f"{name} differs from en: missing {sorted(english - keys)}, "
            f"extra {sorted(keys - english)}"
        )
