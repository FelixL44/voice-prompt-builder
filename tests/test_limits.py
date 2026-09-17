"""Tests for the audio-length and context-window limits.

These exist because of a measured failure: Ollama accepts an oversized prompt,
returns HTTP 200, and silently discards whatever did not fit. It truncates from
the front, which is where a spoken brain-dump states its goal, so the result is
a confident extraction built on the wrong half of the recording.

A size cap does not prevent that, because 100 MB of low-bitrate Opus is hours
of speech. Length is the limit that matters.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest

from backend.analyze import (
    TranscriptTooLongError,
    _warn_if_truncated,
    check_transcript_fits,
    transcript_budget,
)
from backend.config import Settings
from backend.transcribe import (
    AudioTooLongError,
    probe_duration,
    transcribe_upload,
    wav_duration,
)
from tests.conftest import requires_ffmpeg


@pytest.fixture(scope="session")
def long_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """40 minutes of (silent) Opus: tiny on disk, far over the length limit."""
    path = tmp_path_factory.mktemp("long") / "long.webm"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2400",
         "-c:a", "libopus", "-b:a", "12k", str(path)],
        check=True, capture_output=True, timeout=180,
    )
    return path


# ---------------------------------------------------------------------------
# Duration probing
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_probe_reads_duration_without_decoding(long_audio: Path) -> None:
    """ffprobe reads metadata, so this is fast even for a long file."""
    duration = probe_duration(long_audio)

    assert duration is not None
    assert 2350 < duration < 2450


@requires_ffmpeg
def test_probe_returns_none_for_junk(tmp_path: Path) -> None:
    """An unreadable file is not an error here; the later check catches it."""
    junk = tmp_path / "junk.webm"
    junk.write_bytes(b"not audio at all")

    assert probe_duration(junk) is None


@requires_ffmpeg
def test_wav_duration_reads_the_header(sample_wav: Path) -> None:
    duration = wav_duration(sample_wav)

    assert duration is not None
    with wave.open(str(sample_wav)) as handle:
        assert duration == pytest.approx(handle.getnframes() / handle.getframerate())


def test_wav_duration_of_a_non_wav_is_none(tmp_path: Path) -> None:
    path = tmp_path / "not.wav"
    path.write_bytes(b"nope")

    assert wav_duration(path) is None


# ---------------------------------------------------------------------------
# The length limit
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_overlong_audio_is_refused(long_audio: Path) -> None:
    """A 40-minute upload must be refused, not transcribed for 16 minutes."""
    settings = Settings(max_audio_seconds=1800, whisper_model="tiny")

    with pytest.raises(AudioTooLongError, match="minute limit"):
        transcribe_upload(long_audio.read_bytes(), "long.webm", settings)


@requires_ffmpeg
def test_the_error_names_the_actual_length(long_audio: Path) -> None:
    settings = Settings(max_audio_seconds=60, whisper_model="tiny")

    with pytest.raises(AudioTooLongError) as excinfo:
        transcribe_upload(long_audio.read_bytes(), "long.webm", settings)

    assert "40 minutes" in str(excinfo.value)
    assert "VPB_MAX_AUDIO_SECONDS" in str(excinfo.value)


@requires_ffmpeg
def test_audio_within_the_limit_still_works(sample_webm: Path, tiny_settings: Settings) -> None:
    """The cap must not reject an ordinary recording."""
    from dataclasses import replace

    result = transcribe_upload(
        sample_webm.read_bytes(), "s.webm", replace(tiny_settings, max_audio_seconds=1800)
    )

    assert result.text


# ---------------------------------------------------------------------------
# The context-window limit
# ---------------------------------------------------------------------------


def test_budget_leaves_room_for_prompt_and_reply() -> None:
    settings = Settings(ollama_num_ctx=8192, ollama_response_reserve_tokens=512)

    budget = transcript_budget(settings)

    assert 0 < budget < settings.ollama_num_ctx - settings.ollama_response_reserve_tokens


def test_a_normal_transcript_fits() -> None:
    """Five minutes of speech is roughly a thousand tokens."""
    transcript = "word " * 750

    assert check_transcript_fits(transcript, Settings()) > 0


def test_an_overlong_transcript_is_refused_before_the_model_is_called() -> None:
    with pytest.raises(TranscriptTooLongError) as excinfo:
        check_transcript_fits("word " * 20_000, Settings(ollama_num_ctx=8192))

    message = str(excinfo.value)
    assert "silently drop" in message
    assert "VPB_OLLAMA_NUM_CTX=" in message


def test_the_suggested_window_would_actually_fit() -> None:
    """A suggestion that still truncates would be worse than none."""
    transcript = "word " * 8000
    settings = Settings(ollama_num_ctx=8192)

    with pytest.raises(TranscriptTooLongError) as excinfo:
        check_transcript_fits(transcript, settings)

    suggested = int(str(excinfo.value).split("VPB_OLLAMA_NUM_CTX=")[1].rstrip("."))
    from dataclasses import replace

    # The same transcript must pass with the suggested window.
    assert check_transcript_fits(transcript, replace(settings, ollama_num_ctx=suggested)) > 0


def test_raising_the_window_raises_the_budget() -> None:
    small = transcript_budget(Settings(ollama_num_ctx=4096))
    large = transcript_budget(Settings(ollama_num_ctx=32768))

    assert large > small


# ---------------------------------------------------------------------------
# Truncation detection, as a net behind the estimate
# ---------------------------------------------------------------------------


def test_truncation_is_logged_when_ollama_reads_far_less(caplog) -> None:
    transcript = "word " * 4000

    with caplog.at_level("WARNING"):
        _warn_if_truncated({"prompt_eval_count": 100}, transcript, Settings())

    assert "truncated" in caplog.text
    assert "VPB_OLLAMA_NUM_CTX" in caplog.text


def test_no_warning_when_the_counts_are_close(caplog) -> None:
    transcript = "word " * 100

    with caplog.at_level("WARNING"):
        _warn_if_truncated({"prompt_eval_count": 600}, transcript, Settings())

    assert "truncated" not in caplog.text


def test_missing_eval_count_is_not_a_warning(caplog) -> None:
    """Older or unusual responses simply carry no count."""
    with caplog.at_level("WARNING"):
        _warn_if_truncated({}, "word " * 100, Settings())

    assert "truncated" not in caplog.text
