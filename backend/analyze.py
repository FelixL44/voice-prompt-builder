"""Extract a transcript into structured fields using a local LLM.

Ollama is asked for JSON constrained by the schema of ``Extraction``, via its
``format`` parameter. That guarantees parseable output, but it guarantees
nothing about honesty: small models will happily fill an empty field with a
plausible guess, so the prompt pushes hard against that and everything the
model returns is still validated here.

Which fields count as missing is decided by :func:`find_missing` -- by code,
never by the model. The model is not asked what it left out, because a model
that invents an answer will not report it as absent.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from functools import lru_cache

import httpx
from pydantic import ValidationError

from backend.config import Settings, get_settings
from backend.schemas import Extraction, MissingField
from backend.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# (fraction complete or None when unknowable, human-readable note)
ProgressFn = Callable[[float | None, str], None]


class AnalysisError(Exception):
    """Base class for analysis failures we can explain to the user."""

    code = "analysis_error"


class OllamaUnavailableError(AnalysisError):
    """Ollama is not reachable."""

    code = "ollama_unavailable"


class OllamaModelMissingError(AnalysisError):
    """The configured model has not been pulled."""

    code = "ollama_model_missing"


class OllamaTimeoutError(AnalysisError):
    """Generation took longer than the configured timeout."""

    code = "ollama_timeout"


class InvalidExtractionError(AnalysisError):
    """The model returned something that does not fit the schema."""

    code = "invalid_extraction"


class AnalysisCancelledError(AnalysisError):
    """The caller asked for the extraction to stop."""

    code = "cancelled"


class TranscriptTooLongError(AnalysisError):
    """The transcript will not fit the model's context window.

    Refusing is the whole point. Ollama accepts an oversized prompt, returns
    HTTP 200, and silently drops whatever did not fit -- measured: a 6,600
    token prompt sent with num_ctx 2048 was evaluated at 1,026 tokens with no
    warning. It truncates from the *front*, which is where a brain-dump states
    its goal, so the result is a confident extraction built on the wrong half
    of the recording.
    """

    code = "transcript_too_long"


# The question asked when a field comes back empty. Fixed text, because the
# point of these is to be predictable -- not to spend another CPU-minute asking
# the model to phrase a question it might also invent, in a language it might
# also get wrong.
QUESTIONS: dict[str, str] = {
    "goal": "What do you actually want to happen? One sentence is enough.",
    "audience": "Who is this for?",
    "context": "What background would someone need to do this well?",
    "constraints": "Any hard limits? Budget, tone, length, tech, things to avoid?",
    "examples": "Do you have an example of what good looks like?",
    "output_format": "What shape should the result take? Email, JSON, table, memo?",
    "success_criteria": "How would you know this worked?",
}

QUESTIONS_DE: dict[str, str] = {
    "goal": "Was soll konkret passieren? Ein Satz reicht.",
    "audience": "F\u00fcr wen ist das gedacht?",
    "context": "Welchen Hintergrund br\u00e4uchte jemand, um das gut zu machen?",
    "constraints": "Gibt es harte Vorgaben? Budget, Ton, L\u00e4nge, Technik, Tabus?",
    "examples": "Hast du ein Beispiel daf\u00fcr, wie ein gutes Ergebnis aussieht?",
    "output_format": "Welche Form soll das Ergebnis haben? E-Mail, JSON, Tabelle, Memo?",
    "success_criteria": "Woran w\u00fcrdest du merken, dass es funktioniert hat?",
}

QUESTIONS_BY_LANGUAGE: dict[str, dict[str, str]] = {"en": QUESTIONS, "de": QUESTIONS_DE}

DEFAULT_LANGUAGE = "en"


def questions_for(language: str | None) -> dict[str, str]:
    """Follow-up questions in the requested language, falling back to English.

    Only the questions are translated here. Field *values* follow the language
    the person actually spoke, which the extraction prompt handles.
    """
    if not language:
        return QUESTIONS
    return QUESTIONS_BY_LANGUAGE.get(language.strip().lower()[:2], QUESTIONS)


@lru_cache(maxsize=4)
def load_prompt(name: str) -> str:
    """Read a system prompt from ``backend/prompts``. Cached after first read."""
    path = get_settings().prompts_dir / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AnalysisError(f"Could not read prompt {name!r}: {exc}") from exc


# Small models routinely write the *word* null instead of emitting JSON null,
# even under a schema that permits null. Observed from llama3.2:3b:
# {"output_format": "null"}. Left untreated these sail past the missing check,
# and the question loop then never asks about a field the user never answered.
NULL_STRINGS = frozenset(
    {
        "null", "none", "nil", "n/a", "na", "-", "--", "",
        "unknown", "unspecified", "not specified", "not mentioned",
        "not stated", "not provided", "not applicable", "no information",
    }
)


def is_empty(value: object) -> bool:
    """True if a field carries no information the user actually supplied.

    Treats a model's stand-in for absence ("null", "n/a") as absent, since the
    alternative is a field that looks answered but says nothing.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower().strip(".") in NULL_STRINGS
    if isinstance(value, (list, tuple)):
        # A list of nothing-in-particular is still nothing.
        return all(is_empty(item) for item in value)
    if isinstance(value, dict):
        return len(value) == 0
    return False


def normalize(extraction: Extraction) -> Extraction:
    """Replace the model's stand-ins for absence with real absence.

    Turns ``"null"`` into ``None`` and drops filler entries from lists, so a
    field that says nothing also *looks* like it says nothing -- in the raw
    JSON panel, and in the prompt the template builder assembles in v0.4.
    """
    data = extraction.model_dump()
    cleaned: dict[str, object] = {}

    for name, value in data.items():
        if isinstance(value, list):
            cleaned[name] = [item for item in value if not is_empty(item)]
        else:
            cleaned[name] = None if is_empty(value) else value

    return Extraction.model_validate(cleaned)


def find_missing(
    extraction: Extraction, language: str | None = None
) -> list[MissingField]:
    """Return the empty fields, in the order they are worth asking about.

    This is deliberately dumb and deterministic: a field is missing when it is
    null, blank, or an empty list. Keeping the rule in code means the question
    loop cannot be talked out of asking by a confident-sounding model.
    """
    data = extraction.model_dump()
    questions = questions_for(language)
    return [
        MissingField(field=name, question=questions[name])
        for name in QUESTIONS  # QUESTIONS defines both membership and priority.
        if name in data and is_empty(data[name])
    ]


def transcript_budget(settings: Settings | None = None) -> int:
    """How many transcript tokens fit, after the system prompt and the reply."""
    settings = settings or get_settings()
    overhead = estimate_tokens(load_prompt("extract")) + settings.ollama_response_reserve_tokens
    return max(0, settings.ollama_num_ctx - overhead)


def check_transcript_fits(transcript: str, settings: Settings | None = None) -> int:
    """Refuse a transcript that cannot fit the context window.

    Returns:
        The estimated token count, for logging.

    Raises:
        TranscriptTooLongError: it would be silently truncated.
    """
    settings = settings or get_settings()
    tokens = estimate_tokens(transcript)
    budget = transcript_budget(settings)

    if tokens > budget:
        # Suggest a window that would actually work, rounded to something sane.
        needed = settings.ollama_num_ctx + (tokens - budget)
        suggested = min(131072, 1 << (needed - 1).bit_length())
        raise TranscriptTooLongError(
            f"The transcript is about {tokens:,} tokens but only {budget:,} fit "
            f"the context window. Ollama would silently drop the beginning of it, "
            f"which is usually where the goal is. Shorten the recording, or "
            f"restart the server with VPB_OLLAMA_NUM_CTX={suggested}."
        )
    return tokens


def _request_body(transcript: str, settings: Settings) -> dict[str, object]:
    """Build the Ollama generate payload, constrained to the Extraction schema."""
    return {
        "model": settings.ollama_model,
        "stream": True,
        "format": Extraction.model_json_schema(),
        "system": load_prompt("extract"),
        # Tagged rather than bare, so a transcript that contains instructions
        # reads as data instead of as something to obey.
        "prompt": f"<transcript>\n{transcript.strip()}\n</transcript>",
        "options": {
            "temperature": 0,  # Extraction, not creativity.
            "num_ctx": settings.ollama_num_ctx,
        },
    }


def _raise_for_status(status: int, detail: str, settings: Settings) -> None:
    if status == 404:
        raise OllamaModelMissingError(
            f"Model {settings.ollama_model!r} is not available. "
            f"Pull it with: ollama pull {settings.ollama_model}"
        )
    if status >= 400:
        raise AnalysisError(f"Ollama returned {status}: {detail[:200]}")


def _post_to_ollama(
    body: dict[str, object],
    settings: Settings,
    on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, object]:
    """Call Ollama, translating transport failures into typed errors.

    Streamed rather than awaited whole: generation on a CPU takes tens of
    seconds, and streaming is what makes it interruptible and observable. The
    chunks are reassembled here, so callers still receive one response object.
    """
    url = f"{settings.ollama_url}/api/generate"
    pieces: list[str] = []
    final: dict[str, object] = {}

    try:
        with httpx.stream(
            "POST", url, json=body, timeout=settings.ollama_timeout_s
        ) as response:
            if response.status_code >= 400:
                response.read()
                _raise_for_status(response.status_code, response.text, settings)

            for line in response.iter_lines():
                if cancel is not None and cancel.is_set():
                    raise AnalysisCancelledError("Extraction cancelled.")
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue   # Ollama occasionally emits keep-alive blanks.

                piece = chunk.get("response")
                if isinstance(piece, str):
                    pieces.append(piece)
                    if on_progress and len(pieces) % 8 == 0:
                        # Total length is unknowable mid-stream, so report work
                        # done rather than a fraction that would be invented.
                        on_progress(None, f"{len(pieces)} tokens generated")
                if chunk.get("done"):
                    final = chunk
    except (AnalysisCancelledError, AnalysisError):
        raise
    except httpx.TimeoutException as exc:
        raise OllamaTimeoutError(
            f"The model took longer than {settings.ollama_timeout_s}s. "
            "Try a shorter recording or a smaller model."
        ) from exc
    except httpx.RequestError as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {settings.ollama_url}. "
            "Start it with: ollama serve"
        ) from exc

    final["response"] = "".join(pieces)
    return final


def _parse_extraction(payload: dict[str, object]) -> Extraction:
    """Validate the model's JSON against the schema it was given."""
    raw = payload.get("response")
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidExtractionError("The model returned an empty response.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Should not happen with a schema-constrained request, but a malformed
        # response must not surface as a 500.
        raise InvalidExtractionError(
            f"The model returned text that is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise InvalidExtractionError("The model returned JSON that is not an object.")

    try:
        return Extraction.model_validate(data)
    except ValidationError as exc:
        raise InvalidExtractionError(
            f"The model's JSON did not match the expected schema: "
            f"{exc.error_count()} problem(s)."
        ) from exc


def analyze_transcript(
    transcript: str,
    settings: Settings | None = None,
    on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> tuple[Extraction, float]:
    """Extract structure from a transcript.

    Returns:
        The extraction and the wall-clock time it took.

    Raises:
        AnalysisError: for an empty transcript, an unreachable or slow Ollama,
            or a response that does not fit the schema.
    """
    settings = settings or get_settings()
    if not transcript or not transcript.strip():
        raise InvalidExtractionError("The transcript is empty, so there is nothing to analyse.")

    tokens = check_transcript_fits(transcript, settings)

    started = time.monotonic()
    if on_progress:
        on_progress(None, "sending the transcript to the model")
    payload = _post_to_ollama(
        _request_body(transcript, settings), settings, on_progress, cancel
    )
    _warn_if_truncated(payload, transcript, settings)
    extraction = normalize(_parse_extraction(payload))
    elapsed = time.monotonic() - started

    logger.info(
        "Analysed %d chars (~%d tokens) in %.1fs with %s",
        len(transcript), tokens, elapsed, settings.ollama_model,
    )
    return extraction, round(elapsed, 2)


def _warn_if_truncated(
    payload: dict[str, object], transcript: str, settings: Settings
) -> None:
    """Log loudly if Ollama evaluated far fewer tokens than we sent.

    A safety net behind ``check_transcript_fits``: the estimate is rough, and
    silent truncation is severe enough to be worth catching after the fact too.
    The threshold is deliberately slack so a rough estimate cannot cry wolf.
    """
    seen = payload.get("prompt_eval_count")
    if not isinstance(seen, int) or seen <= 0:
        return

    sent = estimate_tokens(load_prompt("extract")) + estimate_tokens(transcript)
    if sent > 0 and seen < sent * 0.6:
        logger.warning(
            "Ollama evaluated only %d tokens of an estimated %d: the transcript "
            "was probably truncated. Raise VPB_OLLAMA_NUM_CTX (currently %d).",
            seen, sent, settings.ollama_num_ctx,
        )


def ollama_models(settings: Settings | None = None) -> list[str] | None:
    """Names of the models Ollama has pulled, or None if it is unreachable.

    None and [] mean different things: unreachable versus running but empty.
    """
    settings = settings or get_settings()
    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=2.0)
        if response.status_code != 200:
            return None
        payload = response.json()
    except (httpx.RequestError, ValueError):
        return None

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    return [m["name"] for m in models if isinstance(m, dict) and "name" in m]


def model_matches(configured: str, available: str) -> bool:
    """True if an available model satisfies the configured name.

    Ollama stores an untagged pull as ``name:latest``, so ``llama3.2`` in the
    config is satisfied by ``llama3.2:latest`` on disk.
    """
    if configured == available:
        return True
    return ":" not in configured and available == f"{configured}:latest"


def ollama_model_available(
    settings: Settings | None = None, models: list[str] | None = None
) -> bool:
    """Whether the configured model has actually been pulled.

    Checked separately from reachability: a running Ollama without the model
    fails only once the user has already recorded and waited.
    """
    settings = settings or get_settings()
    if models is None:
        models = ollama_models(settings)
    if not models:
        return False
    return any(model_matches(settings.ollama_model, name) for name in models)


def ollama_available(settings: Settings | None = None) -> bool:
    """Quick reachability check. Says nothing about which models exist."""
    return ollama_models(settings) is not None
