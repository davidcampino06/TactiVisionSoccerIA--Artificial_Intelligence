"""TactiVision AI Service - FastAPI entry point.

Only the Backend calls this service. The Frontend must never call it
directly; when AI_SERVICE_API_KEY is set every analysis request must send
it in the ``X-API-Key`` header.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import threading
import uuid

import cv2
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from starlette.responses import Response
from config import SERVICE_NAME, SERVICE_VERSION, settings
from detectors import SUPPORTED_MODELS, DetectorFactory, ModelLoadError, ObjectDetector
from log_setup import configure_logging
from pipeline import AnalysisOptions, VideoAnalysisPipeline, VideoProcessingError
from request_guard import SlidingWindowRateLimiter, matches_extension
from schemas import AnalysisResponse, HealthResponse, ModelInfoResponse
from simulation import run_simulation
from tactical_analysis import ATTACK_LEFT_TO_RIGHT, ATTACK_RIGHT_TO_LEFT, ATTACK_UNKNOWN
from team_assignment import HEX_COLOR

configure_logging()
if settings.inference_threads > 0:
    cv2.setNumThreads(settings.inference_threads)
logger = logging.getLogger("tactivision.ai")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
UPLOAD_CHUNK_BYTES = 1024 * 1024

class DetectorProvider:
    """Loads the configured detector once and shares it between requests."""
    def __init__(self) -> None:
        self._detector: ObjectDetector | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        # One analysis at a time: the model runs on CPU and is not shared safely between threads.
        self.analysis_lock = threading.Lock()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def detector(self) -> ObjectDetector | None:
        return self._detector

    def get(self) -> ObjectDetector:
        """Returns the detector, loading it once on first use (thread-safe); raises ModelLoadError on failure."""
        if self._detector is not None:
            return self._detector
        with self._load_lock:
            if self._detector is None:
                try:
                    self._detector = DetectorFactory.create(
                        settings.model_name, settings.model_runtime, settings.model_dir, settings.model_auto_download
                    )
                    self._load_error = None
                    logger.info("Model %s loaded (%s)", settings.model_name, settings.model_runtime)
                except ModelLoadError as error:
                    self._load_error = str(error)
                    logger.error("Model load failed: %s", error)
                    raise
        return self._detector


detector_provider = DetectorProvider()

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Creates the temporary folder, optionally preloads the model and cleans up on shutdown."""
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    if settings.load_model_on_startup:
        try:
            detector_provider.get()
        except ModelLoadError:
            pass  # reported by GET /api/model; the service stays up for health checks
    yield
    shutil.rmtree(settings.temp_dir, ignore_errors=True)

app = FastAPI(title="TactiVision AI Service", version=SERVICE_VERSION, lifespan=lifespan)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Adds the security headers to every response that does not set them itself."""
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


analysis_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute, window_seconds=60)


def client_id(request: Request) -> str:
    """The caller's address. Behind Render's proxy the real one is the first X-Forwarded-For entry."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Refuses the request with 429 and Retry-After when the client went over its per-minute limit."""
    wait = analysis_limiter.check(client_id(request))
    if wait:
        logger.warning("Rate limit reached for %s", client_id(request))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many analysis requests. Try again later.",
            headers={"Retry-After": SlidingWindowRateLimiter.retry_after_header(wait)},
        )

def verify_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Requires the shared X-API-Key (constant-time comparison) when AI_SERVICE_API_KEY is set."""
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        logger.warning("Rejected request to %s from %s: %s API key", request.url.path, client_id(request),
                       "missing" if not x_api_key else "invalid")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")

@app.get("/api/health", response_model=HealthResponse)
def get_health() -> dict:
    """Liveness check used by the backend and by Render."""
    return {"service": SERVICE_NAME, "status": "OK"}

@app.get("/api/model", response_model=ModelInfoResponse)
def get_model() -> dict:
    """Describes the configured model, its runtime, the tracker and whether it loaded."""
    spec = SUPPORTED_MODELS.get(settings.model_name)
    detector = detector_provider.detector
    info = {
        "model": settings.model_name,
        "runtime": settings.model_runtime,
        "tracker": settings.tracker_name,
        "loaded": detector is not None,
        "load_error": detector_provider.load_error,
        "supported_models": DetectorFactory.supported_models(),
        "family": spec.family if spec else None,
        "description": spec.description if spec else None,
    }
    if detector is not None:
        description = detector.describe()
        info["model_version"] = description["model_version"]
        info["detected_classes"] = description["detected_classes"]
    return info

def _save_upload(upload: UploadFile) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{extension}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )
    first_chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
    if not first_chunk:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty video file.")
    # The extension is chosen by the uploader: the first bytes must really be that kind of video.
    if not matches_extension(first_chunk[:16], extension):
        logger.warning("Rejected upload: content is not a %s video", extension)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The file content is not a valid {extension} video.",
        )
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    target = settings.temp_dir / f"{uuid.uuid4()}{extension}"
    written = 0
    with target.open("wb") as output:
        chunk = first_chunk
        while chunk:
            written += len(chunk)
            if written > max_bytes:
                output.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Video larger than {settings.max_upload_size_mb} MB.",
                )
            output.write(chunk)
            chunk = upload.file.read(UPLOAD_CHUNK_BYTES)
    return target

@app.post("/api/analyze", response_model=AnalysisResponse, dependencies=[Depends(rate_limit), Depends(verify_api_key)])
def analyze_video(
    video: UploadFile = File(...),
    confidence_threshold: float = Form(settings.confidence_threshold, ge=0.05, le=0.95),
    frame_stride: int = Form(settings.default_frame_stride, ge=1, le=60),
    max_frames: int = Form(settings.default_max_frames, ge=1, le=5000),
    team_a_attack_direction: str = Form(ATTACK_UNKNOWN),
    pitch_filter: bool = Form(True),
    include_detections: bool = Form(True),
    own_kit_color: str | None = Form(None),
) -> dict:
    """Analyses an uploaded match video and returns detections, tracks, indicators and possible issues."""
    if team_a_attack_direction not in (ATTACK_UNKNOWN, ATTACK_LEFT_TO_RIGHT, ATTACK_RIGHT_TO_LEFT):
        raise HTTPException(status_code=400, detail="Invalid team_a_attack_direction.")
    if own_kit_color and not HEX_COLOR.match(own_kit_color):
        raise HTTPException(status_code=400, detail="own_kit_color must be a #rrggbb colour.")

    try:
        detector = detector_provider.get()
    except ModelLoadError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    video_path = _save_upload(video)
    options = AnalysisOptions(
        confidence_threshold=confidence_threshold,
        ball_confidence_threshold=min(settings.ball_confidence_threshold, confidence_threshold),
        frame_stride=frame_stride,
        max_frames=max_frames,
        team_a_attack_direction=team_a_attack_direction,
        pitch_filter=pitch_filter,
        include_detections=include_detections,
        own_kit_color=own_kit_color or None,
    )
    size_mb = video_path.stat().st_size / 1_048_576
    logger.info("Analysis started: %s %.1f MB, stride=%s, max_frames=%s, direction=%s",
                video_path.suffix, size_mb, frame_stride, max_frames, team_a_attack_direction)
    try:
        with detector_provider.analysis_lock:
            result = VideoAnalysisPipeline(detector).analyze_video(video_path, options)
        logger.info("Analysis finished: %s frames in %s s, %s possible issues", result["frames_processed"],
                    result["performance"]["processing_seconds"], len(result["possible_issues"]))
        return result
    except VideoProcessingError as error:
        logger.warning("Analysis failed: %s", error)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    finally:
        video_path.unlink(missing_ok=True)

@app.post("/api/analyze/simulation", response_model=AnalysisResponse,
          dependencies=[Depends(rate_limit), Depends(verify_api_key)])
def analyze_simulation(frames: int = Form(120, ge=10, le=2000), seed: int = Form(7)) -> dict:
    """SIMULATION MODE: synthetic positions, always labelled as such, never a real analysis."""
    return run_simulation(frames=frames, seed=seed)
