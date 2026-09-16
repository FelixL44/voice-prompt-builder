"""Tests for the extraction module.

These do not call Ollama: the network boundary is stubbed so the tests stay
fast and deterministic. Only the live-model test at the bottom talks to Ollama,
and it skips when Ollama is not running.
"""

from __future__ import annotations

import json

import httpx
import pytest

from backend.analyze import (
    QUESTIONS,
    normalize,
    InvalidExtractionError,
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    _parse_extraction,
    analyze_transcript,
    find_missing,
    is_empty,
    load_prompt,
    ollama_available,
)
from backend.config import Settings
from backend.schemas import Extraction

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Missing-field detection (the part that must never be delegated to the model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True), ("", True), ("   ", True), ([], True), ({}, True),
        ("goal", False), (["a"], False), (0, False),
    ],
)
def test_is_empty(value: object, expected: bool) -> None:
    assert is_empty(value) is expected


@pytest.mark.parametrize(
    "value",
    ["null", "Null", "NULL", "none", "none.", "N/A", "n/a", "-",
     "unknown", "not specified", "Not mentioned", ["null"], [""]],
)
def test_model_null_sentinels_count_as_empty(value: object) -> None:
    """llama3.2:3b writes the word "null" rather than emitting JSON null.

    Observed in practice: {"output_format": "null"}. Treating that as answered
    would leave the v0.3 question loop with nothing to ask about.
    """
    assert is_empty(value) is True


def test_sentinel_in_a_field_is_reported_missing() -> None:
    missing = find_missing(Extraction(goal="Ship it", output_format="null"))

    assert "output_format" in [m.field for m in missing]
    assert "goal" not in [m.field for m in missing]


def test_everything_missing_on_a_blank_extraction() -> None:
    missing = find_missing(Extraction())

    assert [m.field for m in missing] == list(QUESTIONS)


def test_nothing_missing_when_all_fields_are_filled() -> None:
    full = Extraction(
        goal="Ship it", audience="Engineers", context="Legacy system",
        constraints=["No React"], examples=["like this"],
        output_format="Markdown", success_criteria=["Tests pass"],
    )

    assert find_missing(full) == []


def test_whitespace_only_counts_as_missing() -> None:
    """A model that answers with a space has not answered."""
    missing = find_missing(Extraction(goal="   "))

    assert "goal" in [m.field for m in missing]


def test_every_field_has_a_question() -> None:
    """A missing field the UI cannot ask about is a dead end."""
    for name in Extraction.model_fields:
        assert name in QUESTIONS, f"no question defined for {name}"
        # Phrased as a question, though it may close with a clarifying aside.
        assert "?" in QUESTIONS[name], f"{name} question is not a question"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parses_a_well_formed_response() -> None:
    payload = {"response": json.dumps({"goal": "Ship it", "constraints": ["fast"]})}

    extraction = _parse_extraction(payload)

    assert extraction.goal == "Ship it"
    assert extraction.constraints == ["fast"]
    assert extraction.audience is None  # Absent fields fall back to the default.


@pytest.mark.parametrize(
    "response", ["", "   ", "not json at all", "[1, 2, 3]", '"a string"']
)
def test_rejects_unusable_responses(response: str) -> None:
    with pytest.raises(InvalidExtractionError):
        _parse_extraction({"response": response})


def test_rejects_wrong_types() -> None:
    """A string where a list belongs must not slip through."""
    payload = {"response": json.dumps({"constraints": "should be a list"})}

    with pytest.raises(InvalidExtractionError):
        _parse_extraction(payload)


# ---------------------------------------------------------------------------
# Transport errors
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    return Settings(ollama_url="http://localhost:1", ollama_timeout_s=1)


def test_empty_transcript_is_rejected_before_calling_the_model() -> None:
    with pytest.raises(InvalidExtractionError, match="empty"):
        analyze_transcript("   ", _settings())


def test_unreachable_ollama_gives_an_actionable_error(monkeypatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)

    with pytest.raises(OllamaUnavailableError, match="ollama serve"):
        analyze_transcript("something", _settings())


def test_timeout_is_reported_as_such(monkeypatch) -> None:
    def slow(*_args: object, **_kwargs: object) -> None:
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(httpx, "post", slow)

    with pytest.raises(OllamaTimeoutError):
        analyze_transcript("something", _settings())


def test_missing_model_tells_you_how_to_pull_it(monkeypatch) -> None:
    def not_found(*_args: object, **_kwargs: object) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    monkeypatch.setattr(httpx, "post", not_found)

    with pytest.raises(OllamaModelMissingError, match="ollama pull"):
        analyze_transcript("something", _settings())


def test_unavailable_ollama_is_not_an_exception_for_health(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("no"))
    )

    assert ollama_available(_settings()) is False


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_extraction_prompt_is_loadable_and_warns_against_invention() -> None:
    """The anti-hallucination rule is load-bearing; a 3B model needs it."""
    prompt = load_prompt("extract")

    assert "null" in prompt
    assert "never invent" in prompt.lower()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_turns_string_null_into_none() -> None:
    """Otherwise "null" reaches the UI and the v0.4 template verbatim."""
    result = normalize(Extraction(output_format="null", goal="Ship it"))

    assert result.output_format is None
    assert result.goal == "Ship it"


def test_normalize_drops_filler_list_entries() -> None:
    result = normalize(Extraction(constraints=["No React", "n/a", "", "Fast"]))

    assert result.constraints == ["No React", "Fast"]


def test_normalize_leaves_real_content_alone() -> None:
    full = Extraction(
        goal="Ship it", audience="Engineers", context="Legacy",
        constraints=["No React"], examples=["like this"],
        output_format="Markdown", success_criteria=["Tests pass"],
    )

    assert normalize(full) == full
