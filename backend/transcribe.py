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
import wave
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from backend.config import Settings, get_settings
from backend.schemas import Segment

# (fraction complete 0..1, human-readable note)
ProgressFn = Callable[[float, str], None]

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


class AudioTooLongError(TranscriptionError):
    """The recording exceeds the configured length limit."""

    code = "audio_too_long"


class CancelledError(TranscriptionError):
    """The caller asked for the work to stop."""

    code = "cancelled"


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


def _ffprobe_path(settings: Settings) -> str:
    """ffprobe ships with ffmpeg, so derive it from the configured path."""
    configured = Path(settings.ffmpeg_path)
    sibling = configured.with_name(configured.name.replace("ffmpeg", "ffprobe"))
    return str(sibling) if configured.name != str(configured) else "ffprobe"


def probe_duration(src: Path, settings: Settings | None = None) -> float | None:
    """Return the audio duration in seconds, or None if it cannot be read.

    Uses ffprobe, which reads container metadata instead of decoding, so this
    costs milliseconds even for a multi-hour file. None is not an error: some
    inputs genuinely have no duration metadata, and the post-conversion check
    catches those.
    """
    settings = settings or get_settings()
    probe = _ffprobe_path(settings)
    if shutil.which(probe) is None:
        return None

    try:
        proc = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    raw = (proc.stdout or "").strip()
    try:
        duration = float(raw)
    except ValueError:
        return None
    return duration if duration > 0 else None


def _check_duration(seconds: float, settings: Settings) -> None:
    """Refuse audio longer than the configured limit.

    A size cap is not a length cap: 100 MB of low-bitrate Opus is hours of
    speech, which would hold the machine for hours and then produce a
    transcript too long to analyse.
    """
    if seconds <= settings.max_audio_seconds:
        return
    raise AudioTooLongError(
        f"The audio is {seconds / 60:.0f} minutes long, over the "
        f"{settings.max_audio_seconds / 60:.0f} minute limit. Record or upload "
        "something shorter, or raise VPB_MAX_AUDIO_SECONDS."
    )


def wav_duration(path: Path) -> float | None:
    """Exact duration of a PCM WAV, read from its header."""
    try:
        with wave.open(str(path)) as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (OSError, wave.Error, EOFError):
        # EOFError comes from a truncated header, which is exactly what a
        # half-written or corrupt file looks like.
        return None


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


# Set while a warmup is in flight, so /health can distinguish "not loaded yet"
# from "loading right now" and the UI can say which.
_warming = threading.Event()


def model_is_warming() -> bool:
    """True while a background warmup is still running."""
    return _warming.is_set()


def warm_up(settings: Settings | None = None) -> None:
    """Load the configured model so the first real request does not have to.

    Failures are logged and swallowed: a warmup that cannot reach the model hub
    must not stop the server starting, and the same error will surface properly
    on the first real request.
    """
    settings = settings or get_settings()
    _warming.set()
    try:
        get_model(settings)
        logger.info("Warmup complete: %s ready", settings.whisper_model)
    except Exception as exc:  # noqa: BLE001 - startup must not fail on this
        logger.warning("Warmup failed for %s: %s", settings.whisper_model, exc)
    finally:
        _warming.clear()


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


def transcribe_wav(
    wav_path: Path,
    settings: Settings | None = None,
    on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> TranscriptionResult:
    """Transcribe a 16 kHz mono WAV file.

    Args:
        on_progress: called with a 0..1 fraction as each segment lands.
        cancel: checked between segments; set it to stop early.

    Raises:
        ModelLoadError: the model could not be loaded.
        EmptyAudioError: no speech was found.
        CancelledError: ``cancel`` was set.
        TranscriptionError: decoding failed for any other reason.
    """
    settings = settings or get_settings()
    if on_progress:
        # Loading can take a while on a cold model, and the note would
        # otherwise still claim the audio is being converted.
        on_progress(0.0, "loading model" if not model_is_loaded() else "transcribing")
    model = get_model(settings)

    started = time.monotonic()
    if on_progress:
        # Whisper yields nothing until it has processed the first segment, so
        # this is the last honest update for a few seconds on a long file.
        on_progress(0.0, "transcribing")
    try:
        segment_iter, info = model.transcribe(
            str(wav_path),
            language=settings.language,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            initial_prompt=settings.initial_prompt or None,
        )
        # faster-whisper is lazy: the real work happens while draining this,
        # which is what makes per-segment progress and cancellation possible.
        segments: list[Segment] = []
        for raw in segment_iter:
            if cancel is not None and cancel.is_set():
                raise CancelledError("Transcription cancelled.")

            segments.append(
                Segment(start=round(raw.start, 2), end=round(raw.end, 2), text=raw.text.strip())
            )
            if on_progress and info.duration:
                on_progress(min(1.0, raw.end / info.duration), f"{len(segments)} segments")
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


SCRATCH_PREFIX = "job-"
STAGED_PREFIX = "upload-"


def sweep_scratch(settings: Settings | None = None) -> int:
    """Delete leftover working files. Returns how many were removed.

    Cleanup normally happens in a ``finally``, but that cannot run if the
    process is killed outright or the machine loses power, which would leave
    the user's audio on disk. Nothing is in flight at startup, so anything
    still here is an orphan.
    """
    settings = settings or get_settings()
    if not settings.tmp_dir.exists():
        return 0

    removed = 0
    for entry in settings.tmp_dir.iterdir():
        if not entry.name.startswith((SCRATCH_PREFIX, STAGED_PREFIX)):
            continue   # Not ours; leave it alone.
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            logger.warning("Could not remove %s: %s", entry, exc)

    if removed:
        logger.info("Swept %d orphaned working file(s) from %s", removed, settings.tmp_dir)
    return removed


def stage_upload(data: bytes, filename: str, settings: Settings | None = None) -> Path:
    """Write an upload to disk and return its path.

    Queued jobs can wait minutes behind the transcription slot. Holding the
    bytes in memory for that whole time means several queued uploads add up to
    however much audio was sent; on disk they cost nothing but space, and the
    sweep above cleans them up if we die holding one.
    """
    settings = settings or get_settings()
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix or ".bin"
    path = settings.tmp_dir / f"{STAGED_PREFIX}{uuid.uuid4().hex[:12]}{suffix}"
    path.write_bytes(data)
    return path


@contextmanager
def _scratch_dir(settings: Settings) -> Iterator[Path]:
    """A per-request temp directory, removed even if transcription blows up."""
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    work = settings.tmp_dir / f"{SCRATCH_PREFIX}{uuid.uuid4().hex[:12]}"
    work.mkdir()
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)

def transcribe_upload(
    data: bytes,
    filename: str,
    settings: Settings | None = None,
    on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> TranscriptionResult:
    """Stage uploaded audio to disk and transcribe it.

    Everything written is deleted before returning, whatever the outcome:
    nothing recorded is kept on disk.
    """
    settings = settings or get_settings()
    if not data:
        raise EmptyAudioError("The uploaded file was empty.")

    staged = stage_upload(data, filename, settings)
    return transcribe_file(staged, settings, on_progress, cancel)


def transcribe_file(
    source: Path,
    settings: Settings | None = None,
    on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file already on disk, then delete it.

    ``source`` is always removed, whatever the outcome. Queued jobs can wait
    minutes behind the transcription slot, so the audio waits on disk rather
    than in memory, where several queued uploads would add up.
    """
    settings = settings or get_settings()

    try:
        # Checked before the queue: a two-hour upload should be refused
        # immediately, not after waiting behind someone else's job.
        probed = probe_duration(source, settings)
        if probed is not None:
            _check_duration(probed, settings)

        if on_progress:
            on_progress(0.0, "waiting for a transcription slot")

        # Queue rather than thrash: a CPU with no headroom serves two
        # concurrent transcriptions more slowly than two consecutive ones.
        with _transcription_slots(settings):
            if cancel is not None and cancel.is_set():
                raise CancelledError("Transcription cancelled before it started.")

            with _scratch_dir(settings) as work:
                wav_path = work / "audio.wav"
                if on_progress:
                    on_progress(0.0, "converting audio")
                convert_to_wav(source, wav_path, settings)

                # Backstop for inputs whose metadata lied or was absent.
                # Converting is far cheaper than transcribing, so this still
                # saves the worst case.
                actual = wav_duration(wav_path)
                if actual is not None:
                    _check_duration(actual, settings)

                return transcribe_wav(wav_path, settings, on_progress, cancel)
    finally:
        source.unlink(missing_ok=True)
