"""Tests for the conversion and transcription pipeline."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from backend.config import Settings
from backend.transcribe import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    ConversionError,
    EmptyAudioError,
    FfmpegMissingError,
    convert_to_wav,
    ffmpeg_available,
    transcribe_upload,
)
from tests.conftest import requires_ffmpeg, requires_say


# ---------------------------------------------------------------------------
# ffmpeg conversion
# ---------------------------------------------------------------------------


@requires_ffmpeg
@requires_say
def test_convert_produces_16khz_mono_wav(sample_webm: Path, tmp_path: Path) -> None:
    """WebM/Opus in, exactly the PCM format Whisper wants out."""
    out = convert_to_wav(sample_webm, tmp_path / "out.wav")

    assert out.exists() and out.stat().st_size > 0
    with wave.open(str(out)) as handle:
        assert handle.getframerate() == TARGET_SAMPLE_RATE
        assert handle.getnchannels() == TARGET_CHANNELS
        assert handle.getsampwidth() == 2  # 16-bit PCM
        assert handle.getnframes() > 0


@requires_ffmpeg
def test_convert_rejects_non_audio(tmp_path: Path) -> None:
    """A file that is not audio fails loudly, not silently."""
    junk = tmp_path / "junk.webm"
    junk.write_bytes(b"this is definitely not audio")

    with pytest.raises(ConversionError):
        convert_to_wav(junk, tmp_path / "out.wav")


def test_convert_reports_missing_ffmpeg(tmp_path: Path) -> None:
    """A missing binary gives an actionable message, not a stack trace."""
    src = tmp_path / "in.webm"
    src.write_bytes(b"x")
    settings = Settings(ffmpeg_path="ffmpeg-that-does-not-exist")

    with pytest.raises(FfmpegMissingError, match="brew install ffmpeg"):
        convert_to_wav(src, tmp_path / "out.wav", settings)


# ---------------------------------------------------------------------------
# End-to-end transcription
# ---------------------------------------------------------------------------


@requires_ffmpeg
@requires_say
def test_transcribe_upload_roundtrip(sample_webm: Path, tiny_settings: Settings) -> None:
    """The headline path: browser-shaped bytes in, readable transcript out."""
    result = transcribe_upload(
        sample_webm.read_bytes(), "recording.webm", tiny_settings
    )

    words = result.text.lower()
    # Assert on content words rather than the exact string: even a correct
    # transcript varies in punctuation and casing between model versions.
    for expected in ("quick", "brown", "fox", "lazy", "dog"):
        assert expected in words, f"{expected!r} missing from {result.text!r}"

    assert result.language == "en"
    assert result.duration > 0
    assert result.model == "tiny"
    assert result.segments, "expected at least one timestamped segment"
    assert result.segments[0].end > result.segments[0].start


@requires_ffmpeg
@requires_say
def test_transcribe_upload_cleans_up_temp_files(
    sample_webm: Path, tiny_settings: Settings, tmp_path: Path
) -> None:
    """Nothing the user said is left on disk afterwards."""
    settings = Settings(
        whisper_model=tiny_settings.whisper_model,
        language="en",
        tmp_dir=tmp_path / "scratch",
    )

    transcribe_upload(sample_webm.read_bytes(), "recording.webm", settings)

    leftovers = list((tmp_path / "scratch").iterdir())
    assert leftovers == [], f"temp files left behind: {leftovers}"


@requires_ffmpeg
def test_transcribe_upload_rejects_empty_bytes(tiny_settings: Settings) -> None:
    """An empty upload is a user-facing error, not a crash."""
    with pytest.raises(EmptyAudioError):
        transcribe_upload(b"", "recording.webm", tiny_settings)


@requires_ffmpeg
@requires_say
def test_silence_reports_no_speech(tmp_path: Path, tiny_settings: Settings) -> None:
    """Silence is detected as such rather than returning an empty transcript."""
    silent = tmp_path / "silence.wav"
    import subprocess

    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3",
         "-c:a", "pcm_s16le", str(silent)],
        check=True, capture_output=True, timeout=60,
    )

    with pytest.raises(EmptyAudioError):
        transcribe_upload(silent.read_bytes(), "silence.wav", tiny_settings)


def test_ffmpeg_available_reflects_configured_path() -> None:
    """Detection follows the configured binary, not a hardcoded name."""
    assert ffmpeg_available(Settings(ffmpeg_path="definitely-not-a-binary")) is False
