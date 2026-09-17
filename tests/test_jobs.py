"""Tests for the background job registry and its endpoints."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.jobs import JobRegistry, JobState, run_job
from backend.main import app
from backend.transcribe import CancelledError

client = TestClient(app)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_a_job_runs_and_records_its_result() -> None:
    registry = JobRegistry()
    job = registry.create("test")

    with _patched(registry):
        run_job(job, lambda _j: {"answer": 42})
        assert _wait_for(lambda: registry.get(job.id).state is JobState.DONE)

    assert registry.get(job.id).result == {"answer": 42}
    assert registry.get(job.id).snapshot()["progress"] == 1.0


def test_a_failing_job_records_the_error() -> None:
    """An exception must not vanish into a dead thread."""
    registry = JobRegistry()
    job = registry.create("test")

    def boom(_job):
        raise ValueError("it broke")

    with _patched(registry):
        run_job(job, boom)
        assert _wait_for(lambda: registry.get(job.id).state is JobState.ERROR)

    snapshot = registry.get(job.id).snapshot()
    assert "it broke" in snapshot["error"]
    assert snapshot["error_code"] == "internal_error"


def test_cancellation_is_cooperative() -> None:
    registry = JobRegistry()
    job = registry.create("test")

    def slow(current):
        for _ in range(200):
            if current.cancel.is_set():
                raise CancelledError("stopped")
            time.sleep(0.01)
        return {"finished": True}

    with _patched(registry):
        run_job(job, slow)
        assert _wait_for(lambda: registry.get(job.id).state is JobState.RUNNING)
        assert registry.cancel(job.id) is True
        assert _wait_for(lambda: registry.get(job.id).state is JobState.CANCELLED)

    assert registry.get(job.id).snapshot()["error_code"] == "cancelled"


def test_cancelling_a_finished_job_is_refused() -> None:
    registry = JobRegistry()
    job = registry.create("test")

    with _patched(registry):
        run_job(job, lambda _j: "done")
        assert _wait_for(lambda: registry.get(job.id).state is JobState.DONE)

    assert registry.cancel(job.id) is False


def test_unknown_job_cancel_is_refused() -> None:
    assert JobRegistry().cancel("job-nope") is False


def test_finished_jobs_are_swept_after_retention() -> None:
    """Transcripts should not linger in memory indefinitely."""
    registry = JobRegistry(retention_seconds=0)
    job = registry.create("test")
    registry.finish(job.id, state=JobState.DONE, result="x")

    time.sleep(0.01)
    registry.create("another")   # Creating sweeps.

    assert registry.get(job.id) is None


def test_progress_updates_are_visible_while_running() -> None:
    registry = JobRegistry()
    job = registry.create("test")
    gate = threading.Event()

    def work(current):
        registry.update(current.id, progress=0.5, note="halfway")
        gate.wait(timeout=2)
        return "ok"

    with _patched(registry):
        run_job(job, work)
        assert _wait_for(lambda: registry.get(job.id).progress == 0.5)
        assert registry.get(job.id).note == "halfway"
        gate.set()


class _patched:
    """run_job writes to the module-level registry; point it at ours."""

    def __init__(self, registry: JobRegistry) -> None:
        self.registry = registry

    def __enter__(self):
        from backend import jobs

        self._original = jobs.registry
        jobs.registry = self.registry
        return self.registry

    def __exit__(self, *_exc):
        from backend import jobs

        jobs.registry = self._original
        return False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_polling_an_unknown_job_is_404() -> None:
    response = client.get("/jobs/job-does-not-exist")

    assert response.status_code == 404


def test_cancelling_an_unknown_job_is_404() -> None:
    response = client.post("/jobs/job-does-not-exist/cancel")

    assert response.status_code == 404


def test_analyze_job_is_accepted_immediately(monkeypatch) -> None:
    """The endpoint must return before the work finishes, not after."""
    import httpx

    started = threading.Event()
    release = threading.Event()

    class SlowStream:
        status_code = 200
        text = ""

        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return b""

        def iter_lines(self):
            started.set()
            release.wait(timeout=5)
            import json as _json
            yield _json.dumps({"response": '{"goal": "x"}', "done": True})

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: SlowStream())

    response = client.post("/jobs/analyze", json={"transcript": "a short transcript"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    assert started.wait(timeout=5), "work should have begun"
    assert client.get(f"/jobs/{job_id}").json()["state"] in {"queued", "running"}

    release.set()
    assert _wait_for(lambda: client.get(f"/jobs/{job_id}").json()["state"] == "done")
    assert client.get(f"/jobs/{job_id}").json()["result"]["extraction"]["goal"] == "x"


def test_a_job_can_be_cancelled_through_the_api(monkeypatch) -> None:
    import httpx
    import json as _json

    class EndlessStream:
        status_code = 200
        text = ""

        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return b""

        def iter_lines(self):
            while True:
                time.sleep(0.01)
                yield _json.dumps({"response": "x"})

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: EndlessStream())

    job_id = client.post("/jobs/analyze", json={"transcript": "hello"}).json()["job_id"]
    assert _wait_for(lambda: client.get(f"/jobs/{job_id}").json()["state"] == "running")

    client.post(f"/jobs/{job_id}/cancel")

    assert _wait_for(lambda: client.get(f"/jobs/{job_id}").json()["state"] == "cancelled")
