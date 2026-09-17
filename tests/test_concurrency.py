"""Tests for the backend staying responsive and safe under concurrent use.

The bug these exist for: `/transcribe` was declared `async def` while doing
fully blocking work, so FastAPI ran it on the event loop and every other
request -- including `/health` -- waited for the whole transcription.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest

from backend import transcribe as transcribe_module
from backend.config import Settings
from backend.transcribe import TranscriptionResult, _transcription_slots, get_model


# ---------------------------------------------------------------------------
# The event loop must stay free
# ---------------------------------------------------------------------------


BLOCKING_SECONDS = 3.0


def _free_port() -> int:
    """Claim a free port, so a stray server from an earlier run cannot clash."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _canned_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="canned", language="en", duration=1.0,
        model="tiny", elapsed_s=BLOCKING_SECONDS, segments=[],
    )


def test_health_stays_responsive_during_a_transcription(monkeypatch) -> None:
    """A transcription must not block unrelated requests.

    The work is stubbed with a fixed sleep rather than a real transcription:
    a real one on a short clip can finish before the probe lands, which made an
    earlier version of this test pass even with the bug present.

    Runs against a real ASGI server, because TestClient drives the app
    synchronously and would hide the bug entirely.
    """
    import httpx
    import uvicorn

    from backend import main as main_module

    def slow_transcribe(*_args: object, **_kwargs: object) -> TranscriptionResult:
        time.sleep(BLOCKING_SECONDS)
        return _canned_result()

    # Both the sync and job routes funnel through this one call.
    monkeypatch.setattr(main_module, "transcribe_file", slow_transcribe)

    port = _free_port()
    config = uvicorn.Config(main_module.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):  # Wait for startup.
            try:
                httpx.get(f"{base}/health", timeout=1.0)
                break
            except httpx.RequestError:
                time.sleep(0.1)
        else:
            pytest.skip("test server did not start")

        with ThreadPoolExecutor(max_workers=2) as pool:
            job = pool.submit(
                httpx.post,
                f"{base}/transcribe",
                files={"audio": ("s.webm", b"pretend audio", "audio/webm")},
                timeout=30.0,
            )
            time.sleep(0.5)  # Well inside the 3s window.

            started = time.monotonic()
            health = httpx.get(f"{base}/health", timeout=30.0)
            latency = time.monotonic() - started

            assert job.result().status_code == 200

        assert health.status_code == 200
        # On the event loop this waits out the full BLOCKING_SECONDS.
        assert latency < BLOCKING_SECONDS / 2, (
            f"/health took {latency:.1f}s during a {BLOCKING_SECONDS}s transcription "
            "-- the blocking work is running on the event loop"
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------


def test_concurrent_get_model_loads_only_once(monkeypatch) -> None:
    """Two simultaneous requests must not each load their own copy.

    Loading is made slow enough that the second caller is certain to arrive
    while the first is still constructing. Without the lock both would pass the
    cache check and build their own model.
    """
    loads: list[str] = []
    entered = threading.Event()

    class FakeModel:
        def __init__(self, name: str, **_kwargs: object) -> None:
            entered.set()
            time.sleep(0.3)  # Wide enough for the other thread to arrive.
            loads.append(name)

    monkeypatch.setattr(transcribe_module, "WhisperModel", FakeModel)
    monkeypatch.setattr(transcribe_module, "_models", {})

    settings = Settings(whisper_model="fake-model-for-test")

    def first() -> object:
        return get_model(settings)

    def second() -> object:
        entered.wait(timeout=5)  # Arrive mid-construction.
        return get_model(settings)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(first), pool.submit(second)]
        results = [f.result(timeout=10) for f in futures]

    assert len(loads) == 1, f"model loaded {len(loads)} times, expected 1"
    assert results[0] is results[1], "both callers must get the same instance"


def test_model_cache_returns_the_same_instance() -> None:
    settings = Settings(whisper_model="tiny")

    assert get_model(settings) is get_model(settings)


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------


def test_slots_match_the_configured_limit() -> None:
    slots = _transcription_slots(Settings(max_concurrent_transcriptions=3))

    acquired = [slots.acquire(blocking=False) for _ in range(4)]
    for ok in acquired[:3]:
        assert ok, "should allow up to the configured limit"
    assert acquired[3] is False, "should refuse beyond the limit"

    for ok in acquired:
        if ok:
            slots.release()


def test_slots_are_rebuilt_when_the_limit_changes() -> None:
    """Settings are per-request, so a changed limit must take effect."""
    first = _transcription_slots(Settings(max_concurrent_transcriptions=1))
    second = _transcription_slots(Settings(max_concurrent_transcriptions=2))

    assert first is not second
