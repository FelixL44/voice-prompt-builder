"""Pydantic models for the API surface.

Kept in one place so the frontend contract is readable at a glance. Analysis
models (v0.2) will join these rather than living in ``analyze.py``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """One timestamped chunk of transcribed speech."""

    start: float = Field(description="Segment start in seconds.")
    end: float = Field(description="Segment end in seconds.")
    text: str


class TranscriptionResponse(BaseModel):
    """Successful result of ``POST /transcribe``."""

    text: str = Field(description="Full transcript, whitespace-normalised.")
    language: str = Field(description="Detected or forced ISO language code.")
    duration: float = Field(description="Audio duration in seconds.")
    model: str = Field(description="Whisper model size that produced this.")
    elapsed_s: float = Field(description="Wall-clock transcription time.")
    segments: list[Segment] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Result of ``GET /health``, used by the UI to warn before recording."""

    status: str
    ffmpeg: bool = Field(description="Whether the ffmpeg binary was found.")
    ffmpeg_version: str | None = None
    whisper_model: str = Field(description="Default model size.")
    available_models: list[str] = Field(
        default_factory=list, description="Model sizes the UI may request."
    )
    loaded_models: list[str] = Field(
        default_factory=list, description="Sizes already warm in memory."
    )
    model_loaded: bool = Field(
        description="False until the first transcription warms the model cache."
    )
    model_warming: bool = Field(
        default=False, description="True while the model is loading in the background."
    )
    ollama: bool = Field(default=False, description="Whether Ollama is reachable.")
    ollama_model: str = Field(default="", description="Configured extraction model.")
    ollama_model_available: bool = Field(
        default=False,
        description="Whether the configured model has actually been pulled.",
    )


class ErrorResponse(BaseModel):
    """Error body returned for every non-2xx response."""

    error: str = Field(description="Machine-readable error code.")
    detail: str = Field(description="Human-readable message, safe to show in the UI.")


# ---------------------------------------------------------------------------
# Analysis (v0.2)
# ---------------------------------------------------------------------------


class Extraction(BaseModel):
    """The structured form of a brain-dump.

    This model is the contract with the local LLM: its JSON schema is handed to
    Ollama's ``format`` parameter, so the field names and types here are what
    the model is constrained to produce.

    Every field is optional by design. An empty field is a real answer meaning
    "the speaker did not say", and the question loop in v0.3 depends on that
    being honest rather than filled in with a guess.
    """

    goal: str | None = Field(default=None, description="What they want to happen.")
    audience: str | None = Field(default=None, description="Who the output is for.")
    context: str | None = Field(default=None, description="Background the model needs.")
    constraints: list[str] = Field(
        default_factory=list, description="Hard limits: budget, stack, tone, length."
    )
    examples: list[str] = Field(
        default_factory=list, description="Concrete samples the speaker gave."
    )
    output_format: str | None = Field(
        default=None, description="Shape of the deliverable: email, JSON, table."
    )
    success_criteria: list[str] = Field(
        default_factory=list, description="How they would know it worked."
    )


class AnalysisRequest(BaseModel):
    """Body of ``POST /analyze``.

    Takes the transcript rather than the audio, so whatever the user corrected
    in the transcript box is what gets analysed.
    """

    transcript: str = Field(min_length=1, description="The (possibly edited) transcript.")


class MissingField(BaseModel):
    """One empty field, with the question the UI should ask about it."""

    field: str
    question: str


class AnalysisResponse(BaseModel):
    """Result of ``POST /analyze``."""

    extraction: Extraction
    missing: list[MissingField] = Field(
        default_factory=list,
        description="Empty fields, determined by code rather than by the model.",
    )
    model: str = Field(description="Ollama model that produced the extraction.")
    elapsed_s: float


# ---------------------------------------------------------------------------
# Prompt building (v0.4)
# ---------------------------------------------------------------------------


class BuildRequest(BaseModel):
    """Body of ``POST /build``.

    Carries the extraction as the user edited it, not as the model returned it.
    """

    extraction: Extraction


class BuildResponse(BaseModel):
    """The assembled prompt."""

    prompt: str = Field(description="The final prompt, ready to paste.")
    estimated_tokens: int = Field(description="Rough count: characters / 4.")
    characters: int
    sections: list[str] = Field(
        default_factory=list, description="XML tags present, in order."
    )
