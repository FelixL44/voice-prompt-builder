"""Tests for settings and per-request overrides."""

from __future__ import annotations

import pytest

from backend.config import (
    DEFAULT_INITIAL_PROMPT,
    MAX_VOCABULARY_CHARS,
    Settings,
    for_request,
)


def test_for_request_without_overrides_returns_same_settings() -> None:
    base = Settings()
    assert for_request(base) is base


def test_model_override_is_applied() -> None:
    result = for_request(Settings(), model="small")
    assert result.whisper_model == "small"


@pytest.mark.parametrize("bad", ["large-v3", "../etc/passwd", "gpt-4"])
def test_unknown_models_are_rejected(bad: str) -> None:
    """The model name reaches the model hub, so it must be allow-listed."""
    with pytest.raises(ValueError, match="Unknown model"):
        for_request(Settings(), model=bad)


def test_vocabulary_is_prepended_to_the_hint() -> None:
    """User terms come first: Whisper weights the start of the hint most."""
    result = for_request(Settings(), vocabulary="Kubernetes, Pydantic")

    assert result.initial_prompt.startswith("Kubernetes, Pydantic")
    assert DEFAULT_INITIAL_PROMPT in result.initial_prompt


def test_vocabulary_is_truncated() -> None:
    """The hint shares Whisper's 224-token window; an essay would crowd it out."""
    result = for_request(Settings(), vocabulary="word, " * 500)

    user_part = result.initial_prompt.split(". A spoken brain-dump")[0]
    assert len(user_part) <= MAX_VOCABULARY_CHARS


def test_blank_vocabulary_is_ignored() -> None:
    assert for_request(Settings(), vocabulary="   ").initial_prompt == DEFAULT_INITIAL_PROMPT
