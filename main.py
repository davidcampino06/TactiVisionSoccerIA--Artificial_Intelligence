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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from config import SERVICE_NAME, SERVICE_VERSION, settings
from detectors import SUPPORTED_MODELS, DetectorFactory, ModelLoadError, ObjectDetector
from pipeline import AnalysisOptions, VideoAnalysisPipeline, VideoProcessingError
from schemas import AnalysisResponse, HealthResponse, ModelInfoResponse
from simulation import run_simulation
from tactical_analysis import ATTACK_LEFT_TO_RIGHT, ATTACK_RIGHT_TO_LEFT, ATTACK_UNKNOWN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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
async def lifespan(_: FastAPI):
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    if settings.load_model_on_startup:
        try:
            detector_provider.get()
        except ModelLoadError:
            pass  # reported by GET /api/model; the service stays up for health checks
    yield
    shutil.rmtree(settings.temp_dir, ignore_errors=True)

app = FastAPI(title="TactiVision AI Service", version=SERVICE_VERSION, lifespan=lifespan)

def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")

@app.get("/api/health", response_model=HealthResponse)
def get_health() -> dict:
    return {"service": SERVICE_NAME, "status": "OK"}

@app.get("/api/model", response_model=ModelInfoResponse)
def get_model() -> dict:
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
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    target = settings.temp_dir / f"{uuid.uuid4()}{extension}"
    written = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                output.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Video larger than {settings.max_upload_size_mb} MB.",
                )
            output.write(chunk)
    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty video file.")
    return target

@app.post("/api/analyze", response_model=AnalysisResponse, dependencies=[Depends(verify_api_key)])
def analyze_video(
    video: UploadFile = File(...),
    confidence_threshold: float = Form(settings.confidence_threshold, ge=0.05, le=0.95),
    frame_stride: int = Form(settings.default_frame_stride, ge=1, le=60),
    max_frames: int = Form(settings.default_max_frames, ge=1, le=5000),
    team_a_attack_direction: str = Form(ATTACK_UNKNOWN),
    pitch_filter: bool = Form(True),
    include_detections: bool = Form(True),
) -> dict:
    if team_a_attack_direction not in (ATTACK_UNKNOWN, ATTACK_LEFT_TO_RIGHT, ATTACK_RIGHT_TO_LEFT):
        raise HTTPException(status_code=400, detail="Invalid team_a_attack_direction.")

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
    )
    try:
        with detector_provider.analysis_lock:
            return VideoAnalysisPipeline(detector).analyze_video(video_path, options)
    except VideoProcessingError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    finally:
        video_path.unlink(missing_ok=True)

@app.post("/api/analyze/simulation", response_model=AnalysisResponse, dependencies=[Depends(verify_api_key)])
def analyze_simulation(frames: int = Form(120, ge=10, le=2000), seed: int = Form(7)) -> dict:
    return run_simulation(frames=frames, seed=seed)
