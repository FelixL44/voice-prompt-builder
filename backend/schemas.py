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


class ErrorResponse(BaseModel):
    """Error body returned for every non-2xx response."""

    error: str = Field(description="Machine-readable error code.")
    detail: str = Field(description="Human-readable message, safe to show in the UI.")
