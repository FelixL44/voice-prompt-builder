"""Tests for working-file cleanup, job limits, retention and WAL.

The theme is that nothing the user said should outlive the work it was needed
for: not on disk after a crash, not in memory after the result is stored.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import store
from backend.config import Settings, get_settings
from backend.jobs import JobRegistry, JobState, TooManyJobsError
from backend.main import app
from backend.transcribe import (
    SCRATCH_PREFIX,
    STAGED_PREFIX,
    stage_upload,
    sweep_scratch,
    transcribe_file,
)
from tests.conftest import requires_ffmpeg, requires_say


# ---------------------------------------------------------------------------
# Orphaned working files
# ---------------------------------------------------------------------------


def test_sweep_removes_orphaned_job_directories(tmp_path: Path) -> None:
    """A kill -9 leaves these behind, holding the user's audio."""
    settings = Settings(tmp_dir=tmp_path)
    orphan = tmp_path / f"{SCRATCH_PREFIX}deadbeef"
    orphan.mkdir()
    (orphan / "audio.wav").write_bytes(b"pretend audio")

    assert sweep_scratch(settings) == 1
    assert not orphan.exists()


def test_sweep_removes_staged_uploads(tmp_path: Path) -> None:
    settings = Settings(tmp_dir=tmp_path)
    (tmp_path / f"{STAGED_PREFIX}cafe.webm").write_bytes(b"pretend audio")

    assert sweep_scratch(settings) == 1
    assert list(tmp_path.iterdir()) == []


def test_sweep_leaves_unrelated_files_alone(tmp_path: Path) -> None:
    """The directory is configurable and might not be ours exclusively."""
    settings = Settings(tmp_dir=tmp_path)
    keeper = tmp_path / "notes.txt"
    keeper.write_text("keep me")

    assert sweep_scratch(settings) == 0
    assert keeper.exists()


def test_sweep_on_a_missing_directory_is_harmless(tmp_path: Path) -> None:
    assert sweep_scratch(Settings(tmp_dir=tmp_path / "does-not-exist")) == 0


def test_startup_sweeps_orphans(tmp_path: Path, monkeypatch) -> None:
    """The guarantee: whatever a crash left behind goes on the next start."""
    monkeypatch.setenv("VPB_TMP_DIR", str(tmp_path))
    monkeypatch.setenv("VPB_WARMUP", "false")
    get_settings.cache_clear()

    orphan = tmp_path / f"{SCRATCH_PREFIX}fromcrash"
    orphan.mkdir()
    (orphan / "audio.wav").write_bytes(b"pretend audio")

    try:
        with TestClient(app):
            pass   # Entering runs the lifespan.
    finally:
        get_settings.cache_clear()

    assert not orphan.exists()


@requires_ffmpeg
@requires_say
def test_staged_audio_is_deleted_even_when_transcription_fails(
    tmp_path: Path, tiny_settings: Settings
) -> None:
    """The finally must hold on the error path too."""
    from dataclasses import replace

    settings = replace(tiny_settings, tmp_dir=tmp_path)
    staged = stage_upload(b"this is not audio", "junk.webm", settings)
    assert staged.exists()

    with pytest.raises(Exception):
        transcribe_file(staged, settings)

    assert not staged.exists()
    assert list(tmp_path.iterdir()) == []


@requires_ffmpeg
@requires_say
def test_nothing_is_left_after_a_successful_run(
    sample_webm: Path, tiny_settings: Settings, tmp_path: Path
) -> None:
    from dataclasses import replace

    settings = replace(tiny_settings, tmp_dir=tmp_path)
    staged = stage_upload(sample_webm.read_bytes(), "s.webm", settings)

    transcribe_file(staged, settings)

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Job limits and retention
# ---------------------------------------------------------------------------


def test_active_jobs_are_capped() -> None:
    registry = JobRegistry()
    for _ in range(3):
        registry.create("test", max_active=3)

    with pytest.raises(TooManyJobsError, match="already running"):
        registry.create("test", max_active=3)


def test_finished_jobs_do_not_count_towards_the_cap() -> None:
    registry = JobRegistry()
    first = registry.create("test", max_active=1)
    registry.finish(first.id, state=JobState.DONE, result="x")

    registry.create("test", max_active=1)   # Must not raise.


def test_releasing_drops_the_result_immediately() -> None:
    """The result is the transcript; it should not linger once stored."""
    registry = JobRegistry()
    job = registry.create("test")
    registry.finish(job.id, state=JobState.DONE, result={"text": "something private"})

    assert registry.release(job.id) is True
    assert registry.get(job.id) is None


def test_an_unfinished_job_cannot_be_released() -> None:
    registry = JobRegistry()
    job = registry.create("test")

    assert registry.release(job.id) is False
    assert registry.get(job.id) is not None


def test_retention_is_configurable() -> None:
    registry = JobRegistry(retention_seconds=999)
    registry.set_retention(0)

    job = registry.create("test")
    registry.finish(job.id, state=JobState.DONE, result="x")
    time.sleep(0.01)
    registry.create("other")   # Creating sweeps.

    assert registry.get(job.id) is None


def test_default_retention_is_short() -> None:
    """A long window would leave transcripts in memory for no good reason."""
    assert Settings().job_retention_s <= 300


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_job_cap_is_reported_as_429(monkeypatch) -> None:
    import httpx

    monkeypatch.setenv("VPB_MAX_ACTIVE_JOBS", "1")
    get_settings.cache_clear()

    class Endless:
        status_code = 200
        text = ""

        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return b""

        def iter_lines(self):
            import json as _json
            while True:
                time.sleep(0.01)
                yield _json.dumps({"response": "x"})

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Endless())
    client = TestClient(app)

    try:
        first = client.post("/jobs/analyze", json={"transcript": "hello"})
        assert first.status_code == 202

        second = client.post("/jobs/analyze", json={"transcript": "hello again"})
        assert second.status_code == 429
        assert second.json()["error"] == "too_many_jobs"
    finally:
        client.post(f"/jobs/{first.json()['job_id']}/cancel")
        get_settings.cache_clear()


def test_releasing_over_http(monkeypatch) -> None:
    import json as _json

    import httpx

    class Quick:
        status_code = 200
        text = ""

        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return b""

        def iter_lines(self):
            yield _json.dumps({"response": _json.dumps({"goal": "x"}), "done": True})

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Quick())
    client = TestClient(app)

    job_id = client.post("/jobs/analyze", json={"transcript": "hello"}).json()["job_id"]
    for _ in range(100):
        if client.get(f"/jobs/{job_id}").json()["state"] == "done":
            break
        time.sleep(0.02)

    assert client.delete(f"/jobs/{job_id}").status_code == 200
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_releasing_an_unknown_job_is_404() -> None:
    assert TestClient(app).delete("/jobs/job-nope").status_code == 404


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def test_database_uses_wal(tmp_path: Path) -> None:
    """Readers and a writer overlap on a threadpool; WAL lets them proceed."""
    settings = Settings(db_path=tmp_path / "s.db")
    store.save_session({"title": "first"}, settings)

    with sqlite3.connect(settings.db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_concurrent_writes_do_not_lock(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    settings = Settings(db_path=tmp_path / "s.db")
    store.save_session({"title": "seed"}, settings)

    def write(index: int) -> dict:
        return store.save_session({"title": f"session {index}"}, settings)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(24)))

    assert len(results) == 24
    assert len(store.list_sessions(settings=settings)) == 25
