"""Audio conversion and speech-to-text.

The browser hands us WebM/Opus, which Whisper cannot read directly, so every
request goes through ffmpeg to 16 kHz mono PCM first. Transcription runs on the
CPU via CTranslate2 (faster-whisper) with int8 quantisation -- no torch, which
matters because recent torch releases have no Intel-Mac builds.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from backend.config import Settings, get_settings
from backend.schemas import Segment

logger = logging.getLogger(__name__)

# Whisper wants 16 kHz mono; anything else it resamples internally anyway.
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class TranscriptionError(Exception):
    """Base class for failures we can explain to the user."""

    code = "transcription_error"


class FfmpegMissingError(TranscriptionError):
    """The ffmpeg binary is not on PATH."""

    code = "ffmpeg_missing"


class ConversionError(TranscriptionError):
    """ffmpeg ran but could not decode the audio."""

    code = "conversion_failed"


class EmptyAudioError(TranscriptionError):
    """The audio decoded fine but contains no speech."""

    code = "empty_audio"


class ModelLoadError(TranscriptionError):
    """The Whisper model could not be loaded or downloaded."""

    code = "model_load_failed"


# --------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------


def ffmpeg_available(settings: Settings | None = None) -> bool:
    """Return True if the configured ffmpeg binary can be found."""
    settings = settings or get_settings()
    return shutil.which(settings.ffmpeg_path) is not None


def ffmpeg_version(settings: Settings | None = None) -> str | None:
    """Return ffmpeg's version string, or None if it is missing or mute."""
    settings = settings or get_settings()
    if not ffmpeg_available(settings):
        return None
    try:
        proc = subprocess.run(
            [settings.ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
    return first_line or None


def convert_to_wav(src: Path, dst: Path, settings: Settings | None = None) -> Path:
    """Decode any ffmpeg-readable audio into 16 kHz mono PCM WAV at ``dst``.

    Raises:
        FfmpegMissingError: ffmpeg is not installed.
        ConversionError: the input could not be decoded.
    """
    settings = settings or get_settings()
    if not ffmpeg_available(settings):
        raise FfmpegMissingError(
            "ffmpeg was not found on PATH. Install it with: brew install ffmpeg"
        )

    cmd = [
        settings.ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-vn",                          # ignore any video/cover-art stream
        "-ac", str(TARGET_CHANNELS),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(dst),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(
            f"ffmpeg timed out after {settings.ffmpeg_timeout_s}s. "
            "The recording may be far longer than expected."
        ) from exc
    except OSError as exc:
        raise ConversionError(f"Could not run ffmpeg: {exc}") from exc

    if proc.returncode != 0:
        # ffmpeg's last stderr line is usually the actionable part.
        detail = (proc.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit code {proc.returncode}"
        raise ConversionError(f"ffmpeg could not decode the audio: {reason}")

    if not dst.exists() or dst.stat().st_size == 0:
        raise ConversionError("ffmpeg produced an empty file. Is the recording silent?")

    return dst


# --------------------------------------------------------------------------
# Whisper
# --------------------------------------------------------------------------

# Keyed by (size, device, compute_type). Holding several lets the UI switch
# between sizes without paying the load cost each time; int8 models are small
# enough (~150 MB base, ~500 MB small) that keeping a few resident is fine.
#
# Requests run in a threadpool, so this dict is touched concurrently: the lock
# stops two requests loading the same model twice, which would double both the
# wait and the memory.
_models: dict[tuple[str, str, str], WhisperModel] = {}
_models_lock = threading.Lock()

# Limits how many transcriptions run at once. Built lazily because its size
# comes from settings, and guarded by its own lock for the same reason as above.
_slots: threading.BoundedSemaphore | None = None
_slots_size: int | None = None
_slots_lock = threading.Lock()


def _transcription_slots(settings: Settings) -> threading.BoundedSemaphore:
    """The semaphore limiting concurrent transcriptions."""
    global _slots, _slots_size

    with _slots_lock:
        if _slots is None or _slots_size != settings.max_concurrent_transcriptions:
            _slots = threading.BoundedSemaphore(settings.max_concurrent_transcriptions)
            _slots_size = settings.max_concurrent_transcriptions
        return _slots


def model_is_loaded() -> bool:
    """True once any model has been loaded into the process cache."""
    return bool(_models)


def loaded_models() -> list[str]:
    """Names of the models currently resident, for /health."""
    return sorted(key[0] for key in _models)


def get_model(settings: Settings | None = None) -> WhisperModel:
    """Return the cached Whisper model, loading it on first use.

    The first load of a given size downloads its weights (tens of seconds);
    afterwards it is reused process-wide.
    """
    settings = settings or get_settings()
    key = (settings.whisper_model, settings.device, settings.compute_type)

    # Checked before taking the lock, so warm requests never serialise on it.
    cached = _models.get(key)
    if cached is not None:
        return cached

    with _models_lock:
        # Re-checked: another thread may have loaded it while we waited.
        cached = _models.get(key)
        if cached is not None:
            return cached

        logger.info("Loading Whisper model %s (%s, %s)", *key)
        started = time.monotonic()
        try:
            model = WhisperModel(
                settings.whisper_model,
                device=settings.device,
                compute_type=settings.compute_type,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
            raise ModelLoadError(
                f"Could not load Whisper model {settings.whisper_model!r}: {exc}"
            ) from exc

        logger.info(
            "Model %s ready in %.1fs", settings.whisper_model, time.monotonic() - started
        )
        _models[key] = model
        return model


@dataclass(frozen=True)
class TranscriptionResult:
    """Everything the API needs from one transcription run."""

    text: str
    language: str
    duration: float
    model: str
    elapsed_s: float
    segments: list[Segment]


def transcribe_wav(wav_path: Path, settings: Settings | None = None) -> TranscriptionResult:
    """Transcribe a 16 kHz mono WAV file.

    Raises:
        ModelLoadError: the model could not be loaded.
        EmptyAudioError: no speech was found.
        TranscriptionError: decoding failed for any other reason.
    """
    settings = settings or get_settings()
    model = get_model(settings)

    started = time.monotonic()
    try:
        segment_iter, info = model.transcribe(
            str(wav_path),
            language=settings.language,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            initial_prompt=settings.initial_prompt or None,
        )
        # faster-whisper is lazy: the real work happens while draining this.
        segments = [
            Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
            for s in segment_iter
        ]
    except TranscriptionError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    elapsed = time.monotonic() - started
    text = " ".join(s.text for s in segments if s.text).strip()
    if not text:
        raise EmptyAudioError(
            "No speech was detected. Check your microphone level and try again."
        )

    return TranscriptionResult(
        text=text,
        language=info.language or settings.language or "unknown",
        duration=round(info.duration, 2),
        model=settings.whisper_model,
        elapsed_s=round(elapsed, 2),
        segments=segments,
    )


@contextmanager
def _scratch_dir(settings: Settings) -> Iterator[Path]:
    """A per-request temp directory, removed even if transcription blows up."""
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    work = settings.tmp_dir / f"job-{uuid.uuid4().hex[:12]}"
    work.mkdir()
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


def transcribe_upload(
    data: bytes,
    filename: str,
    settings: Settings | None = None,
) -> TranscriptionResult:
    """Convert raw uploaded audio bytes and transcribe them.

    The uploaded bytes and the converted WAV are both deleted before returning,
    whatever the outcome: nothing recorded is kept on disk.
    """
    settings = settings or get_settings()
    if not data:
        raise EmptyAudioError("The uploaded file was empty.")

    # Keep the original extension; ffmpeg sniffs content but the hint helps.
    suffix = Path(filename).suffix or ".bin"

    # Queue rather than thrash: a CPU with no headroom serves two concurrent
    # transcriptions more slowly than two consecutive ones.
    with _transcription_slots(settings):
        with _scratch_dir(settings) as work:
            raw_path = work / f"input{suffix}"
            wav_path = work / "audio.wav"
            raw_path.write_bytes(data)
            convert_to_wav(raw_path, wav_path, settings)
            return transcribe_wav(wav_path, settings)
