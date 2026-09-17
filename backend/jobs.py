"""An in-memory registry for long-running work.

Transcription and extraction both take tens of seconds to minutes on a CPU.
Run synchronously they give the user a spinner, no sense of progress, and no
way out: closing the tab abandons the request but the machine keeps working.

So they run as jobs. The caller gets an id immediately, polls for progress, and
can cancel. State lives in memory rather than in the database because it is
worthless after a restart -- a job whose thread died cannot be resumed, and
pretending otherwise would be a lie told by the UI.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Fallback retention when no settings are supplied. The real value comes from
# VPB_JOB_RETENTION_S: a finished job's result is the transcript, so this is a
# privacy window, not merely a cache.
RETENTION_SECONDS = 120


class TooManyJobsError(Exception):
    """The active-job limit is already reached."""

    code = "too_many_jobs"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in {JobState.DONE, JobState.ERROR, JobState.CANCELLED}


@dataclass
class Job:
    """One unit of background work and everything the UI needs to show it."""

    id: str
    kind: str
    state: JobState = JobState.QUEUED
    progress: float | None = None
    """0..1 where it can be known, None where it genuinely cannot."""
    note: str = ""
    result: Any = None
    error: str | None = None
    error_code: str | None = None
    created: float = field(default_factory=time.monotonic)
    updated: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        """A plain dict for the API. Never hands out the cancel event."""
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state.value,
            "progress": self.progress,
            "note": self.note,
            "result": self.result,
            "error": self.error,
            "error_code": self.error_code,
            "elapsed_s": round((self.finished_at or time.monotonic()) - self.created, 2),
        }


class JobRegistry:
    """Thread-safe store of jobs, swept of stale entries as it is used."""

    def __init__(self, retention_seconds: int = RETENTION_SECONDS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._retention = retention_seconds

    def create(self, kind: str, max_active: int | None = None) -> Job:
        """Register a new job.

        Raises:
            TooManyJobsError: ``max_active`` unfinished jobs already exist.
        """
        job = Job(id=f"job-{uuid.uuid4().hex[:12]}", kind=kind)
        with self._lock:
            self._sweep_locked()

            if max_active is not None:
                active = sum(1 for j in self._jobs.values() if not j.state.finished)
                if active >= max_active:
                    raise TooManyJobsError(
                        f"{active} jobs are already running or queued. "
                        "Wait for one to finish, or cancel it."
                    )

            self._jobs[job.id] = job
        return job

    def release(self, job_id: str) -> bool:
        """Forget a finished job now that the caller has its result.

        Called once the client has stored the outcome, so the transcript is not
        left sitting in memory for the whole retention window.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.state.finished:
                return False
            del self._jobs[job_id]
            return True

    def active(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if not job.state.finished)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. Returns False if it is unknown or already over."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state.finished:
                return False
            job.cancel.set()
            job.note = "cancelling"
            job.updated = time.monotonic()
            return True

    def update(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        progress: float | None = None,
        note: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if state is not None:
                job.state = state
            if progress is not None:
                job.progress = progress
            if note is not None:
                job.note = note
            job.updated = time.monotonic()

    def finish(
        self,
        job_id: str,
        *,
        state: JobState,
        result: Any = None,
        error: str | None = None,
        error_code: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.state = state
            job.result = result
            job.error = error
            job.error_code = error_code
            job.progress = 1.0 if state is JobState.DONE else job.progress
            job.finished_at = time.monotonic()
            job.updated = job.finished_at

    def set_retention(self, seconds: int) -> None:
        """Adopt the configured retention. Settings are not known at import."""
        self._retention = seconds

    def _sweep_locked(self) -> None:
        """Drop finished jobs past their retention. Caller holds the lock."""
        now = time.monotonic()
        stale = [
            key for key, job in self._jobs.items()
            if job.finished_at is not None and now - job.finished_at > self._retention
        ]
        for key in stale:
            del self._jobs[key]

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)


registry = JobRegistry()


def run_job(job: Job, work: Callable[[Job], Any]) -> None:
    """Execute ``work`` on a worker thread, recording how it ended.

    ``work`` receives the job so it can report progress and watch ``cancel``.
    Every outcome is captured: an uncaught exception here would otherwise
    vanish into a dead thread and leave the UI polling a job that never moves.
    """
    def target() -> None:
        registry.update(job.id, state=JobState.RUNNING, note="starting")
        try:
            result = work(job)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            code = getattr(exc, "code", None)
            if code == "cancelled" or job.cancel.is_set():
                logger.info("Job %s cancelled", job.id)
                registry.finish(job.id, state=JobState.CANCELLED, error="Cancelled.",
                                error_code="cancelled")
            else:
                logger.warning("Job %s failed: %s", job.id, exc)
                registry.finish(job.id, state=JobState.ERROR, error=str(exc),
                                error_code=code or "internal_error")
            return

        if job.cancel.is_set():
            registry.finish(job.id, state=JobState.CANCELLED, error="Cancelled.",
                            error_code="cancelled")
        else:
            registry.finish(job.id, state=JobState.DONE, result=result)

    threading.Thread(target=target, name=f"vpb-{job.kind}", daemon=True).start()
