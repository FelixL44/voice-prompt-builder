"""Runtime configuration, read once from environment variables.

Every setting has a working default so the app runs with no setup. Override any
of them by exporting the matching ``VPB_*`` variable before starting the server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Only sizes we are willing to load. Guards the request-level model override:
# the value reaches the Hugging Face hub, so it must never be free-form.
ALLOWED_MODELS = ("tiny", "base", "small", "medium")

# Whisper conditions decoding on this text, which sharply improves rare words.
# It shares the 224-token window with the audio context, so it stays short.
DEFAULT_INITIAL_PROMPT = (
    "A spoken brain-dump describing a task for a large language model. "
    "Likely vocabulary: API, LLM, prompt, backend, frontend, schema, JSON, "
    "Python, TypeScript, Docker, latency, benchmark, deploy, roadmap."
)

MAX_VOCABULARY_CHARS = 600


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    whisper_model: str = "base"
    """Whisper model size: tiny, base, small, medium. Larger is slower."""

    compute_type: str = "int8"
    """CTranslate2 quantisation. int8 is the only sane choice on a CPU-only Mac."""

    device: str = "cpu"

    language: str | None = None
    """Force a language code (e.g. "en"), or None to auto-detect."""

    initial_prompt: str = DEFAULT_INITIAL_PROMPT
    """Decoding hint. The single cheapest accuracy win: it costs no extra time
    but reliably rescues product names, acronyms and technical terms."""

    vad_filter: bool = True
    """Drop silence before transcribing. Speeds up rambling brain-dumps a lot."""

    beam_size: int = 5

    max_upload_bytes: int = 100 * 1024 * 1024
    """Reject uploads larger than this. ~5 min of Opus is well under 10 MB."""

    max_audio_seconds: int = 1800
    """Longest audio accepted, in seconds (30 minutes).

    A size cap alone is not a length cap: 100 MB of 24 kbps Opus is nearly ten
    hours, which would occupy the machine for hours and then produce a
    transcript far too long to analyse. This bounds both.
    """

    ffmpeg_path: str = "ffmpeg"

    ffmpeg_timeout_s: int = 120

    warmup_on_startup: bool = True
    """Load the Whisper model in the background as the server starts.

    Without it the first recording pays the full load (~40s for base, ~90s for
    small) while the user waits with no idea why.
    """

    max_concurrent_transcriptions: int = 1
    """Transcriptions allowed to run at once.

    Measured on this machine: two concurrent runs finish only 1.29x faster than
    two serial ones, while making each individual request ~50% slower. Queueing
    gives a more predictable wait than thrashing a CPU that has no headroom.
    """

    tmp_dir: Path = PROJECT_ROOT / "tmp"

    # --- Ollama (v0.2) ---

    ollama_url: str = "http://localhost:11434"

    ollama_model: str = "llama3.2:3b"
    """Small on purpose: this runs on CPU. 3B-8B is the usable range."""

    ollama_timeout_s: int = 600
    """Generous: a 5-minute transcript takes 1-2 minutes on a CPU-only Mac."""

    ollama_num_ctx: int = 8192
    """Must fit the transcript plus the system prompt. ~750 words per 5 min."""

    ollama_response_reserve_tokens: int = 512
    """Held back from the context window for the model's own JSON output."""

    prompts_dir: Path = PROJECT_ROOT / "backend" / "prompts"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings from the environment. Cached, so the env is read once."""
    language = os.environ.get("VPB_LANGUAGE", "").strip()
    return Settings(
        whisper_model=_env_str("VPB_WHISPER_MODEL", "base"),
        compute_type=_env_str("VPB_COMPUTE_TYPE", "int8"),
        device=_env_str("VPB_DEVICE", "cpu"),
        language=language or None,
        initial_prompt=_env_str("VPB_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPT),
        vad_filter=_env_bool("VPB_VAD_FILTER", True),
        beam_size=_env_int("VPB_BEAM_SIZE", 5),
        max_upload_bytes=_env_int("VPB_MAX_UPLOAD_BYTES", 100 * 1024 * 1024),
        max_audio_seconds=_env_int("VPB_MAX_AUDIO_SECONDS", 1800),
        ffmpeg_path=_env_str("VPB_FFMPEG_PATH", "ffmpeg"),
        ffmpeg_timeout_s=_env_int("VPB_FFMPEG_TIMEOUT_S", 120),
        max_concurrent_transcriptions=_env_int("VPB_MAX_CONCURRENT_TRANSCRIPTIONS", 1),
        warmup_on_startup=_env_bool("VPB_WARMUP", True),
        tmp_dir=Path(_env_str("VPB_TMP_DIR", str(PROJECT_ROOT / "tmp"))),
        ollama_url=_env_str("VPB_OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=_env_str("VPB_OLLAMA_MODEL", "llama3.2:3b"),
        ollama_timeout_s=_env_int("VPB_OLLAMA_TIMEOUT_S", 600),
        ollama_num_ctx=_env_int("VPB_OLLAMA_NUM_CTX", 8192),
        ollama_response_reserve_tokens=_env_int("VPB_OLLAMA_RESPONSE_RESERVE", 512),
    )


def for_request(
    settings: Settings,
    model: str | None = None,
    vocabulary: str | None = None,
) -> Settings:
    """Apply per-request overrides on top of the base settings.

    The UI lets each recording pick a model size and supply domain vocabulary,
    so those two are resolved per request rather than per process.

    Raises:
        ValueError: the requested model is not one we allow.
    """
    overrides: dict[str, object] = {}

    if model:
        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"Unknown model {model!r}. Choose one of: {', '.join(ALLOWED_MODELS)}."
            )
        overrides["whisper_model"] = model

    if vocabulary and vocabulary.strip():
        # Prepend the user's terms: Whisper weights the start of the hint most,
        # and truncating protects the shared 224-token context window.
        terms = vocabulary.strip()[:MAX_VOCABULARY_CHARS]
        overrides["initial_prompt"] = f"{terms}. {settings.initial_prompt}"

    return replace(settings, **overrides) if overrides else settings
