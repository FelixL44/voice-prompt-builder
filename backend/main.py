"""FastAPI application: routes and error handling.

Everything here runs locally. No audio or text is sent anywhere off this
machine -- the only network traffic is the one-time Whisper model download.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import store
from backend.analyze import (
    AnalysisError,
    analyze_transcript,
    find_missing,
    ollama_model_available,
    ollama_models,
)
from backend.jobs import Job, TooManyJobsError, registry, run_job
from backend.builder import (
    EmptyPromptError,
    build_prompt,
    estimate_tokens,
    used_sections,
)
from backend.config import ALLOWED_MODELS, FRONTEND_DIR, for_request, get_settings
from backend.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    BuildRequest,
    BuildResponse,
    ErrorResponse,
    HealthResponse,
    JobAccepted,
    JobStatus,
    SessionRecord,
    SessionSaveRequest,
    SessionSummary,
    StoreStats,
    TranscriptionResponse,
)
from backend.transcribe import (
    TranscriptionError,
    ffmpeg_available,
    ffmpeg_version,
    loaded_models,
    model_is_loaded,
    model_is_warming,
    stage_upload,
    sweep_scratch,
    transcribe_file,
    warm_up,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vpb")

# Uploads are read in pieces so an oversized file is rejected partway through
# rather than after it is all in memory.
UPLOAD_CHUNK_BYTES = 1024 * 1024

class NoCacheStaticFiles(StaticFiles):
    """Serve static assets with revalidation forced.

    Without an explicit Cache-Control header, browsers fall back to heuristic
    caching and can keep running a stale app.js after an edit. That surfaces as
    a broken UI rather than as a caching problem, which is a miserable thing to
    debug. The ETag still makes revalidation cheap: unchanged files come back
    as a 304 with no body.
    """

    def file_response(self, *args: object, **kwargs: object) -> Response:
        response = super().file_response(*args, **kwargs)  # type: ignore[arg-type]
        response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm the Whisper model in the background while the server starts.

    Deliberately not awaited: the server must accept requests immediately, and
    /health reports `model_warming` so the UI can explain the wait rather than
    appearing to hang on the first recording.
    """
    settings = get_settings()
    task: asyncio.Task[None] | None = None

    # Anything left in the working directory is an orphan: cleanup runs in a
    # finally, which cannot happen if the process was killed outright.
    registry.set_retention(settings.job_retention_s)
    sweep_scratch(settings)

    if settings.warmup_on_startup:
        logger.info("Warming up %s in the background", settings.whisper_model)
        task = asyncio.create_task(asyncio.to_thread(warm_up, settings))

    try:
        yield
    finally:
        if task is not None and not task.done():
            task.cancel()
        # Best effort: workers are daemon threads and may die mid-write, so the
        # startup sweep remains the real guarantee.
        sweep_scratch(settings)


app = FastAPI(
    title="Voice Prompt Builder",
    version="0.1.0",
    description="Turn a spoken brain-dump into a structured prompt, fully locally.",
    lifespan=lifespan,
)

# Which error maps to which status code. Anything unlisted is a 500.
_STATUS_BY_CODE = {
    "ffmpeg_missing": 503,
    "model_load_failed": 503,
    "empty_audio": 422,
    "conversion_failed": 400,
    "ollama_unavailable": 503,
    "ollama_model_missing": 503,
    "ollama_timeout": 504,
    "invalid_extraction": 422,
    "empty_prompt": 422,
    "audio_too_long": 413,
    "transcript_too_long": 422,
    "cancelled": 499,
    "store_unavailable": 503,
    "too_many_jobs": 429,
}


async def _typed_error_handler(
    _request: Request, exc: TranscriptionError | AnalysisError
) -> JSONResponse:
    """Turn our typed errors into readable JSON the UI can display as-is."""
    status = _STATUS_BY_CODE.get(exc.code, 500)
    logger.warning("%s (%s): %s", type(exc).__name__, exc.code, exc)
    body = ErrorResponse(error=exc.code, detail=str(exc))
    return JSONResponse(status_code=status, content=body.model_dump())


app.add_exception_handler(TranscriptionError, _typed_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(AnalysisError, _typed_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(EmptyPromptError, _typed_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(store.StoreError, _typed_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(TooManyJobsError, _typed_error_handler)  # type: ignore[arg-type]


def _too_large(limit: int) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"Recording is larger than the {limit / 1_048_576:.0f} MB limit.",
    )


async def _read_upload(audio: UploadFile, request: Request, limit: int) -> bytes:
    """Read an upload into memory, refusing anything over ``limit``.

    The declared length is checked first so an oversized upload is refused
    outright, and the body is then read in chunks so a missing or dishonest
    Content-Length cannot still fill memory.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise _too_large(limit)

    buffer = bytearray()
    while chunk := await audio.read(UPLOAD_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise _too_large(limit)
    return bytes(buffer)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the machine is actually ready to transcribe.

    The UI calls this on load so a missing ffmpeg shows up as a banner rather
    than as a failure after someone has already recorded five minutes.
    """
    settings = get_settings()
    has_ffmpeg = ffmpeg_available(settings)

    # One call answers both questions: reachability and whether the configured
    # model is actually there.
    models = ollama_models(settings)
    return HealthResponse(
        status="ok" if has_ffmpeg else "degraded",
        ffmpeg=has_ffmpeg,
        model_warming=model_is_warming(),
        ollama=models is not None,
        ollama_model=settings.ollama_model,
        ollama_model_available=ollama_model_available(settings, models),
        ffmpeg_version=ffmpeg_version(settings),
        whisper_model=settings.whisper_model,
        available_models=list(ALLOWED_MODELS),
        loaded_models=loaded_models(),
        model_loaded=model_is_loaded(),
    )


# ---------------------------------------------------------------------------
# Shared work
#
# Each operation exists twice at the HTTP layer: once synchronously for scripts
# and curl, once as a job for the UI, which needs progress and cancellation.
# Only the waiting differs, so the work itself lives here once. Keeping two
# copies had already let them drift -- the synchronous upload path missed a
# size-check fix that the job path received.
# ---------------------------------------------------------------------------


async def _staged_upload(
    request: Request,
    audio: UploadFile,
    model: str | None,
    vocabulary: str | None,
) -> tuple[Settings, Path, str]:
    """Resolve per-request settings and put the upload on disk.

    Returns:
        The settings for this request, the staged file, and its original name.
    """
    try:
        settings = for_request(get_settings(), model, vocabulary)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = await _read_upload(audio, request, settings.max_upload_bytes)
    filename = audio.filename or "recording.webm"

    logger.info(
        "Transcribing %s (%.1f KB) with %s%s",
        filename, len(data) / 1024, settings.whisper_model,
        " + vocabulary" if vocabulary else "",
    )
    return settings, stage_upload(data, filename, settings), filename


def _transcription_response(
    staged: Path,
    settings: Settings,
    on_progress: object = None,
    cancel: object = None,
) -> TranscriptionResponse:
    """Transcribe a staged file and shape the reply."""
    result = transcribe_file(staged, settings, on_progress, cancel)  # type: ignore[arg-type]

    logger.info(
        "Done: %.1fs of audio in %.1fs (%d chars)",
        result.duration, result.elapsed_s, len(result.text),
    )
    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
        model=result.model,
        elapsed_s=result.elapsed_s,
        segments=result.segments,
    )


def _analysis_response(
    request: AnalysisRequest,
    settings: Settings,
    on_progress: object = None,
    cancel: object = None,
) -> AnalysisResponse:
    """Extract a transcript and shape the reply.

    Missing fields are computed here, from the extraction -- the model is never
    asked what it failed to find.
    """
    extraction, elapsed = analyze_transcript(
        request.transcript, settings, on_progress, cancel  # type: ignore[arg-type]
    )
    missing = find_missing(extraction, request.language)
    logger.info("Extraction complete, %d field(s) missing", len(missing))

    return AnalysisResponse(
        extraction=extraction,
        missing=missing,
        model=settings.ollama_model,
        elapsed_s=elapsed,
    )


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
    model: str | None = Form(default=None),
    vocabulary: str | None = Form(default=None),
) -> TranscriptionResponse:
    """Transcribe an uploaded recording and wait for the result.

    The job variant is what the UI uses; this one is for scripts and curl.

    Args:
        model: Optional size override (tiny/base/small/medium).
        vocabulary: Optional domain terms. Whisper is conditioned on these,
            which rescues names and acronyms at no cost in time.
    """
    settings, staged, _ = await _staged_upload(request, audio, model, vocabulary)

    # ffmpeg and Whisper both block for the whole recording. Run them in a
    # worker thread: on the event loop they would freeze every other request,
    # including /health, for the entire transcription.
    return await run_in_threadpool(_transcription_response, staged, settings)


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Extract a transcript into structured fields and wait for the result.

    The transcript arrives as JSON rather than as the recording, so the user's
    edits in the transcript box are what gets analysed.
    """
    return _analysis_response(request, get_settings())



def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Extract a transcript into structured fields with the local LLM.

    The transcript arrives as JSON rather than as the recording, so the user's
    edits in the transcript box are what gets analysed.
    """
    settings = get_settings()
    extraction, elapsed = analyze_transcript(request.transcript, settings)

    # Missing fields are computed here, from the extraction -- the model is
    # never asked what it failed to find.
    missing = find_missing(extraction, request.language)
    logger.info("Extraction complete, %d field(s) missing", len(missing))

    return AnalysisResponse(
        extraction=extraction,
        missing=missing,
        model=settings.ollama_model,
        elapsed_s=elapsed,
    )


@app.post(
    "/build",
    response_model=BuildResponse,
    responses={422: {"model": ErrorResponse}},
)
def build(request: BuildRequest) -> BuildResponse:
    """Assemble the final prompt from the edited fields.

    Pure string assembly, no model: the same extraction always produces the
    same prompt, so the output is reproducible and reviewable.
    """
    prompt = build_prompt(request.extraction)

    return BuildResponse(
        prompt=prompt,
        estimated_tokens=estimate_tokens(prompt),
        characters=len(prompt),
        sections=used_sections(request.extraction),
    )


# ---------------------------------------------------------------------------
# Jobs: long work, made observable and interruptible
# ---------------------------------------------------------------------------


@app.post("/jobs/transcribe", response_model=JobAccepted, status_code=202)
async def start_transcription(
    request: Request,
    audio: UploadFile = File(...),
    model: str | None = Form(default=None),
    vocabulary: str | None = Form(default=None),
) -> JobAccepted:
    """Queue a transcription and return immediately with its job id."""
    # Staged before the job is created: a queued job can wait minutes for a
    # slot, and several waiting uploads held in memory would add up.
    settings, staged, filename = await _staged_upload(request, audio, model, vocabulary)
    job = registry.create("transcribe", settings.max_active_jobs)

    def work(current: Job) -> dict[str, object]:
        def progress(fraction: float, note: str) -> None:
            registry.update(current.id, progress=fraction, note=note)

        return _transcription_response(staged, settings, progress, current.cancel).model_dump()

    run_job(job, work)
    logger.info("Queued transcription %s for %s", job.id, filename)
    return JobAccepted(job_id=job.id, kind=job.kind)


@app.post("/jobs/analyze", response_model=JobAccepted, status_code=202)
def start_analysis(request: AnalysisRequest) -> JobAccepted:
    """Queue an extraction and return immediately with its job id."""
    settings = get_settings()
    job = registry.create("analyze", settings.max_active_jobs)

    def work(current: Job) -> dict[str, object]:
        def progress(fraction: float | None, note: str) -> None:
            registry.update(current.id, progress=fraction, note=note)

        return _analysis_response(request, settings, progress, current.cancel).model_dump()

    run_job(job, work)
    return JobAccepted(job_id=job.id, kind=job.kind)


@app.get("/jobs/{job_id}", response_model=JobStatus, responses={404: {"model": ErrorResponse}})
def job_status(job_id: str) -> JobStatus:
    """Poll a job. Finished jobs stay readable for a while, then are swept."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job, or it has expired.")
    return JobStatus(**job.snapshot())


@app.delete("/jobs/{job_id}", responses={404: {"model": ErrorResponse}})
def release_job(job_id: str) -> dict[str, str]:
    """Forget a finished job.

    The client calls this once it has stored the result, so a transcript is not
    left in memory for the whole retention window.
    """
    if not registry.release(job_id):
        raise HTTPException(
            status_code=404, detail="No such job, or it has not finished yet."
        )
    return {"status": "released"}


@app.post("/jobs/{job_id}/cancel", response_model=JobStatus,
          responses={404: {"model": ErrorResponse}})
def cancel_job(job_id: str) -> JobStatus:
    """Ask a job to stop.

    Cancellation is cooperative: the worker checks between Whisper segments and
    between streamed model tokens, so it takes effect within a second or two
    rather than instantly.
    """
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job, or it has expired.")

    registry.cancel(job_id)
    return JobStatus(**job.snapshot())


# ---------------------------------------------------------------------------
# Sessions: history that outlives the browser
# ---------------------------------------------------------------------------


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions(limit: int = 100) -> list[SessionSummary]:
    """Session summaries, newest first."""
    return [SessionSummary(**row) for row in store.list_sessions(min(limit, 500))]


@app.get("/sessions/stats", response_model=StoreStats)
def session_stats() -> StoreStats:
    """Count and on-disk size, for the cache panel."""
    return StoreStats(**store.store_stats())


@app.get("/sessions/{session_id}", response_model=SessionRecord,
         responses={404: {"model": ErrorResponse}})
def read_session(session_id: str) -> SessionRecord:
    row = store.get_session(session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such session.")
    return SessionRecord(**row)


@app.put("/sessions", response_model=SessionRecord)
def save_session(request: SessionSaveRequest) -> SessionRecord:
    """Create or update a session. The UI calls this after each stage."""
    payload = request.model_dump()
    if payload.get("extraction") is not None:
        payload["extraction"] = request.extraction.model_dump() if request.extraction else None
    return SessionRecord(**store.save_session(payload))


@app.delete("/sessions/{session_id}", responses={404: {"model": ErrorResponse}})
def remove_session(session_id: str) -> dict[str, str]:
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="No such session.")
    return {"status": "deleted"}


@app.delete("/sessions")
def clear_all_sessions() -> dict[str, int]:
    """Delete every stored session."""
    return {"deleted": store.clear_sessions()}


@app.exception_handler(HTTPException)
async def _http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Give HTTP errors the same body shape as transcription errors."""
    body = ErrorResponse(error=f"http_{exc.status_code}", detail=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the recording page."""
    return FileResponse(
        FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"}
    )


# Mounted last so it cannot shadow the API routes above.
app.mount("/static", NoCacheStaticFiles(directory=FRONTEND_DIR), name="static")
