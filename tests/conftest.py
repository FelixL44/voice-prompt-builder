"""Shared fixtures.

The sample audio is generated at test time with macOS ``say`` rather than
committed as a binary blob, which keeps the repo text-only. Tests that need it
skip cleanly on machines without ``say`` or ``ffmpeg``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SAMPLE_SENTENCE = "The quick brown fox jumps over the lazy dog."


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


requires_ffmpeg = pytest.mark.skipif(
    not _have("ffmpeg"), reason="ffmpeg is not installed"
)
requires_say = pytest.mark.skipif(
    not _have("say"), reason="macOS 'say' is needed to generate sample audio"
)


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 16 kHz mono WAV of SAMPLE_SENTENCE, spoken by the system voice."""
    if not _have("say"):
        pytest.skip("macOS 'say' is needed to generate sample audio")

    path = tmp_path_factory.mktemp("audio") / "sample.wav"
    subprocess.run(
        ["say", "-o", str(path), "--data-format=LEI16@16000", SAMPLE_SENTENCE],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return path


@pytest.fixture(scope="session")
def sample_webm(sample_wav: Path) -> Path:
    """The sample re-encoded as WebM/Opus, matching what the browser uploads."""
    if not _have("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    path = sample_wav.parent / "sample.webm"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(sample_wav), "-c:a", "libopus", "-b:a", "24k", str(path)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return path


def _ollama_up() -> bool:
    """True if Ollama answers, so live tests can skip rather than fail."""
    import httpx

    try:
        from backend.config import get_settings

        url = get_settings().ollama_url
        return httpx.get(f"{url}/api/tags", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_up(), reason="Ollama is not running"
)


@pytest.fixture(scope="session")
def tiny_settings():
    """Settings pinned to the 'tiny' model so the suite stays fast."""
    from backend.config import Settings

    return Settings(whisper_model="tiny", language="en")
