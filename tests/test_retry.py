"""Tests for the extraction retry.

A malformed reply used to end the run after the user had already waited.
Retrying only helps if the second attempt differs from the first: at
temperature 0 an identical request reproduces the identical failure. What
changes depends on how it failed, which Ollama reports via ``done_reason``.
"""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from backend.analyze import (
    MALFORMED,
    RETRY_NOTE,
    TRUNCATED,
    AnalysisCancelledError,
    InvalidExtractionError,
    OllamaUnavailableError,
    _failure_kind,
    _remaining_room,
    _request_body,
    analyze_transcript,
)
from backend.config import Settings

VALID = json.dumps({"goal": "Ship it", "constraints": ["fast"]})


class Reply:
    """One scripted Ollama response."""

    def __init__(self, body: str, done_reason: str = "stop") -> None:
        self.body = body
        self.done_reason = done_reason

    def lines(self) -> list[str]:
        return [json.dumps({"response": self.body, "done": True,
                            "done_reason": self.done_reason})]


class ScriptedOllama:
    """Returns each scripted reply in turn, recording the requests made."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, _method, _url, **kwargs):
        self.requests.append(kwargs["json"])
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        return _Stream(reply.lines())


class _Stream:
    status_code = 200
    text = ""

    def __init__(self, lines): self._lines = lines
    def __enter__(self): return self
    def __exit__(self, *_a): return False
    def read(self): return b""
    def iter_lines(self): yield from self._lines


def _settings(**overrides) -> Settings:
    return Settings(ollama_url="http://localhost:1", ollama_timeout_s=5, **overrides)


# ---------------------------------------------------------------------------
# What a retry changes
# ---------------------------------------------------------------------------


def test_first_attempt_is_deterministic() -> None:
    body = _request_body("hello", _settings())

    assert body["options"]["temperature"] == 0.0
    assert RETRY_NOTE not in body["system"]


def test_retry_after_truncation_asks_for_more_room_not_more_warmth() -> None:
    """It ran out of space; a different sample would run out too."""
    settings = _settings()
    first = _request_body("hello", settings)
    retry = _request_body("hello", settings, TRUNCATED)

    assert retry["options"]["num_predict"] > first["options"]["num_predict"]
    assert retry["options"]["temperature"] == 0.0
    assert RETRY_NOTE not in retry["system"]


def test_retry_after_truncation_uses_the_room_that_is_actually_left() -> None:
    """Doubling is a guess; a guess that is still too small fails again."""
    settings = _settings(ollama_response_reserve_tokens=24)

    retry = _request_body("a short transcript", settings, TRUNCATED)

    assert retry["options"]["num_predict"] == _remaining_room("a short transcript", settings)
    assert retry["options"]["num_predict"] > 1000


def test_retry_after_malformed_changes_the_sample_and_says_why() -> None:
    retry = _request_body("hello", _settings(), MALFORMED)

    assert retry["options"]["temperature"] > 0
    assert RETRY_NOTE in retry["system"]


def test_num_predict_makes_the_reserve_real() -> None:
    """Otherwise the reserved tokens are an assumption, not a budget."""
    settings = _settings(ollama_response_reserve_tokens=333)

    assert _request_body("hello", settings)["options"]["num_predict"] == 333


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"done_reason": "length"}, TRUNCATED),
        ({"done_reason": "stop"}, MALFORMED),
        ({}, MALFORMED),
    ],
)
def test_failure_classification(payload: dict, expected: str) -> None:
    assert _failure_kind(payload) == expected


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_a_truncated_first_attempt_is_recovered(monkeypatch) -> None:
    ollama = ScriptedOllama(
        Reply('{"goal": "Ship i', done_reason="length"),   # Cut mid-string.
        Reply(VALID),
    )
    monkeypatch.setattr(httpx, "stream", ollama)

    extraction, _ = analyze_transcript("a transcript", _settings(analysis_retries=1))

    assert extraction.goal == "Ship it"
    assert len(ollama.requests) == 2
    # The retry must have reacted to the truncation, not merely repeated.
    assert ollama.requests[1]["options"]["num_predict"] > ollama.requests[0]["options"]["num_predict"]


def test_a_malformed_first_attempt_is_recovered(monkeypatch) -> None:
    ollama = ScriptedOllama(Reply("not json at all"), Reply(VALID))
    monkeypatch.setattr(httpx, "stream", ollama)

    extraction, _ = analyze_transcript("a transcript", _settings(analysis_retries=1))

    assert extraction.goal == "Ship it"
    assert ollama.requests[1]["options"]["temperature"] > 0


def test_a_good_first_attempt_does_not_retry(monkeypatch) -> None:
    """The user waits 20-90s per attempt; do not spend one needlessly."""
    ollama = ScriptedOllama(Reply(VALID))
    monkeypatch.setattr(httpx, "stream", ollama)

    analyze_transcript("a transcript", _settings(analysis_retries=1))

    assert len(ollama.requests) == 1


def test_retries_are_bounded(monkeypatch) -> None:
    ollama = ScriptedOllama(Reply("garbage"))
    monkeypatch.setattr(httpx, "stream", ollama)

    with pytest.raises(InvalidExtractionError):
        analyze_transcript("a transcript", _settings(analysis_retries=2))

    assert len(ollama.requests) == 3   # The first attempt plus two retries.


def test_retries_can_be_disabled(monkeypatch) -> None:
    ollama = ScriptedOllama(Reply("garbage"))
    monkeypatch.setattr(httpx, "stream", ollama)

    with pytest.raises(InvalidExtractionError):
        analyze_transcript("a transcript", _settings(analysis_retries=0))

    assert len(ollama.requests) == 1


def test_transport_failures_are_not_retried(monkeypatch) -> None:
    """Retrying an unreachable server just doubles the wait for the same error."""
    calls = []

    def refuse(*_a, **_k):
        calls.append(1)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "stream", refuse)

    with pytest.raises(OllamaUnavailableError):
        analyze_transcript("a transcript", _settings(analysis_retries=2))

    assert len(calls) == 1


def test_cancellation_stops_between_attempts(monkeypatch) -> None:
    cancel = threading.Event()
    ollama = ScriptedOllama(Reply("garbage"))

    def stream(*args, **kwargs):
        cancel.set()   # Cancelled while the first attempt is in flight.
        return ollama(*args, **kwargs)

    monkeypatch.setattr(httpx, "stream", stream)

    with pytest.raises(AnalysisCancelledError):
        analyze_transcript("a transcript", _settings(analysis_retries=2), cancel=cancel)

    assert len(ollama.requests) == 1


def test_progress_names_the_retry(monkeypatch) -> None:
    """The user should know why it is taking longer than usual."""
    monkeypatch.setattr(httpx, "stream", ScriptedOllama(Reply("garbage"), Reply(VALID)))
    notes: list[str] = []

    analyze_transcript(
        "a transcript", _settings(analysis_retries=1), on_progress=lambda f, n: notes.append(n)
    )

    assert any("retrying" in note for note in notes)
