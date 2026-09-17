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
import time
from functools import lru_cache

import httpx
from pydantic import ValidationError

from backend.config import Settings, get_settings
from backend.schemas import Extraction, MissingField

logger = logging.getLogger(__name__)


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


# The question asked when a field comes back empty. Fixed text, because the
# point of these is to be predictable -- not to spend another CPU-minute
# asking the model to phrase a question it might also invent.
QUESTIONS: dict[str, str] = {
    "goal": "What do you actually want to happen? One sentence is enough.",
    "audience": "Who is this for?",
    "context": "What background would someone need to do this well?",
    "constraints": "Any hard limits? Budget, tone, length, tech, things to avoid?",
    "examples": "Do you have an example of what good looks like?",
    "output_format": "What shape should the result take? Email, JSON, table, memo?",
    "success_criteria": "How would you know this worked?",
}


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


def find_missing(extraction: Extraction) -> list[MissingField]:
    """Return the empty fields, in the order they are worth asking about.

    This is deliberately dumb and deterministic: a field is missing when it is
    null, blank, or an empty list. Keeping the rule in code means the question
    loop cannot be talked out of asking by a confident-sounding model.
    """
    data = extraction.model_dump()
    return [
        MissingField(field=name, question=QUESTIONS[name])
        for name in QUESTIONS  # QUESTIONS defines both membership and priority.
        if name in data and is_empty(data[name])
    ]


def _request_body(transcript: str, settings: Settings) -> dict[str, object]:
    """Build the Ollama generate payload, constrained to the Extraction schema."""
    return {
        "model": settings.ollama_model,
        "stream": False,
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


def _post_to_ollama(body: dict[str, object], settings: Settings) -> dict[str, object]:
    """Call Ollama, translating transport failures into typed errors."""
    url = f"{settings.ollama_url}/api/generate"
    try:
        response = httpx.post(url, json=body, timeout=settings.ollama_timeout_s)
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

    if response.status_code == 404:
        raise OllamaModelMissingError(
            f"Model {settings.ollama_model!r} is not available. "
            f"Pull it with: ollama pull {settings.ollama_model}"
        )
    if response.status_code >= 400:
        raise AnalysisError(
            f"Ollama returned {response.status_code}: {response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise AnalysisError("Ollama returned a response that was not JSON.") from exc


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
    transcript: str, settings: Settings | None = None
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

    started = time.monotonic()
    payload = _post_to_ollama(_request_body(transcript, settings), settings)
    extraction = normalize(_parse_extraction(payload))
    elapsed = time.monotonic() - started

    logger.info(
        "Analysed %d chars in %.1fs with %s",
        len(transcript), elapsed, settings.ollama_model,
    )
    return extraction, round(elapsed, 2)


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
