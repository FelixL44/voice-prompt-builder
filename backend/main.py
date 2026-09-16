"""FastAPI application: routes and error handling.

Everything here runs locally. No audio or text is sent anywhere off this
machine -- the only network traffic is the one-time Whisper model download.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.analyze import (
    AnalysisError,
    analyze_transcript,
    find_missing,
    ollama_available,
)
from backend.config import ALLOWED_MODELS, FRONTEND_DIR, for_request, get_settings
from backend.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    TranscriptionResponse,
)
from backend.transcribe import (
    EmptyAudioError,
    FfmpegMissingError,
    TranscriptionError,
    ffmpeg_available,
    ffmpeg_version,
    loaded_models,
    model_is_loaded,
    transcribe_upload,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vpb")

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


app = FastAPI(
    title="Voice Prompt Builder",
    version="0.1.0",
    description="Turn a spoken brain-dump into a structured prompt, fully locally.",
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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the machine is actually ready to transcribe.

    The UI calls this on load so a missing ffmpeg shows up as a banner rather
    than as a failure after someone has already recorded five minutes.
    """
    settings = get_settings()
    has_ffmpeg = ffmpeg_available(settings)
    return HealthResponse(
        status="ok" if has_ffmpeg else "degraded",
        ffmpeg=has_ffmpeg,
        ollama=ollama_available(settings),
        ollama_model=settings.ollama_model,
        ffmpeg_version=ffmpeg_version(settings),
        whisper_model=settings.whisper_model,
        available_models=list(ALLOWED_MODELS),
        loaded_models=loaded_models(),
        model_loaded=model_is_loaded(),
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
    audio: UploadFile = File(...),
    model: str | None = Form(default=None),
    vocabulary: str | None = Form(default=None),
) -> TranscriptionResponse:
    """Transcribe an uploaded recording.

    Accepts anything ffmpeg can decode: the browser's WebM/Opus blobs as well
    as m4a/mp3/wav files picked from disk.

    Args:
        model: Optional size override (tiny/base/small/medium).
        vocabulary: Optional domain terms. Whisper is conditioned on these,
            which rescues names and acronyms at no cost in time.
    """
    base_settings = get_settings()
    try:
        settings = for_request(base_settings, model, vocabulary)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = await audio.read()
    if len(data) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes / 1_048_576
        raise HTTPException(
            status_code=413,
            detail=f"Recording is larger than the {limit_mb:.0f} MB limit.",
        )

    filename = audio.filename or "recording.webm"
    logger.info(
        "Transcribing %s (%.1f KB) with %s%s",
        filename, len(data) / 1024, settings.whisper_model,
        " + vocabulary" if vocabulary else "",
    )

    # TranscriptionError subclasses are handled by the exception handler above.
    result = transcribe_upload(data, filename, settings)

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
    """Extract a transcript into structured fields with the local LLM.

    The transcript arrives as JSON rather than as the recording, so the user's
    edits in the transcript box are what gets analysed.
    """
    settings = get_settings()
    extraction, elapsed = analyze_transcript(request.transcript, settings)

    # Missing fields are computed here, from the extraction -- the model is
    # never asked what it failed to find.
    missing = find_missing(extraction)
    logger.info("Extraction complete, %d field(s) missing", len(missing))

    return AnalysisResponse(
        extraction=extraction,
        missing=missing,
        model=settings.ollama_model,
        elapsed_s=elapsed,
    )


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
