"""Runtime configuration for the TactiVision AI Service.

Every value comes from an environment variable so that no secret or
machine-specific path is hard-coded in the source code.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SERVICE_NAME = "TactiVision AI"
SERVICE_VERSION = "0.5.0"

def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    return float(raw_value) if raw_value not in (None, "") else default

def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value not in (None, "") else default

def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    return raw_value.strip().lower() in ("1", "true", "yes", "on")

@dataclass(frozen=True)
class Settings:
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "YOLO26n"))
    # "onnx" -> onnxruntime (light, no PyTorch). "ultralytics" -> .pt weights with PyTorch.
    model_runtime: str = field(default_factory=lambda: os.getenv("MODEL_RUNTIME", "onnx").lower())
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "models")))
    model_auto_download: bool = field(default_factory=lambda: _env_bool("MODEL_AUTO_DOWNLOAD", True))
    tracker_name: str = field(default_factory=lambda: os.getenv("TRACKER_NAME", "ByteTrack"))
    confidence_threshold: float = field(default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.25))
    ball_confidence_threshold: float = field(
        default_factory=lambda: _env_float("BALL_CONFIDENCE_THRESHOLD", 0.15)
    )
    default_frame_stride: int = field(default_factory=lambda: _env_int("FRAME_STRIDE", 3))
    default_max_frames: int = field(default_factory=lambda: _env_int("MAX_FRAMES", 300))
    max_upload_size_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_SIZE_MB", 200))
    api_key: str = field(default_factory=lambda: os.getenv("AI_SERVICE_API_KEY", ""))
    temp_dir: Path = field(default_factory=lambda: Path(os.getenv("TEMP_DIR", "tmp")))
    load_model_on_startup: bool = field(default_factory=lambda: _env_bool("LOAD_MODEL_ON_STARTUP", True))

settings = Settings()
