"""Visual detection layer (step 1 of the AI pipeline).

This module only answers the question "what objects are visible in this
frame?". It does not know anything about tracking or tactics.

Design patterns:
- Factory Method: ``DetectorFactory`` builds the right ``ObjectDetector``
  for the configured model and runtime, so the rest of the service never
  depends on a concrete model implementation (Dependency Inversion).
- Polymorphism / Liskov: ``OnnxYoloDetector`` and ``UltralyticsDetector``
  are interchangeable behind the ``ObjectDetector`` interface.
"""

from __future__ import annotations

import ast
import logging
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("tactivision.detectors")

# COCO class ids used by the pre-trained models.
COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32

OBJECT_TYPE_PLAYER = "PLAYER"
OBJECT_TYPE_BALL = "BALL"

CLASS_ID_TO_OBJECT_TYPE = {
    COCO_PERSON_CLASS_ID: OBJECT_TYPE_PLAYER,
    COCO_SPORTS_BALL_CLASS_ID: OBJECT_TYPE_BALL,
}

ASSETS_BASE_URL = "https://github.com/ultralytics/assets/releases/download"

@dataclass(frozen=True)
class ModelSpec:
    """Static description of a supported detection model."""
    name: str
    family: str
    onnx_url: str | None
    weights_url: str | None
    end_to_end: bool
    description: str

    @property
    def onnx_file_name(self) -> str | None:
        return self.onnx_url.rsplit("/", 1)[-1] if self.onnx_url else None

    @property
    def weights_file_name(self) -> str | None:
        return self.weights_url.rsplit("/", 1)[-1] if self.weights_url else None

SUPPORTED_MODELS: dict[str, ModelSpec] = {
    "YOLO26n": ModelSpec(
        name="YOLO26n",
        family="YOLO26",
        onnx_url=f"{ASSETS_BASE_URL}/v8.4.0/yolo26n.onnx",
        weights_url=f"{ASSETS_BASE_URL}/v8.4.0/yolo26n.pt",
        end_to_end=True,
        description="Ultralytics YOLO26 nano, NMS-free end-to-end detector (COCO pre-trained).",
    ),
    "YOLO11n": ModelSpec(
        name="YOLO11n",
        family="YOLO11",
        onnx_url=f"{ASSETS_BASE_URL}/v8.3.0/yolo11n.onnx",
        weights_url=f"{ASSETS_BASE_URL}/v8.3.0/yolo11n.pt",
        end_to_end=False,
        description="Ultralytics YOLO11 nano baseline detector (COCO pre-trained, requires NMS).",
    ),
    "RT-DETR-l": ModelSpec(
        name="RT-DETR-l",
        family="RT-DETR",
        onnx_url=None,
        weights_url=f"{ASSETS_BASE_URL}/v8.3.0/rtdetr-l.pt",
        end_to_end=True,
        description="Transformer-based detector. Only available with MODEL_RUNTIME=ultralytics.",
    ),
}

@dataclass(frozen=True)
class Detection:
    """One object detected in one frame, in original frame pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    object_type: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: the best 2D proxy of a player's position on the pitch."""
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

class ModelLoadError(RuntimeError):
    """Raised when a model cannot be found, downloaded or loaded."""

class ObjectDetector(ABC):
    """Abstraction used by the pipeline. Concrete detectors hide runtime details."""
    def __init__(self, spec: ModelSpec, runtime: str) -> None:
        self.spec = spec
        self.runtime = runtime
        self.model_version = "unknown"
        self.last_inference_ms = 0.0

    @property
    def model_name(self) -> str:
        return self.spec.name

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float) -> list[Detection]:
        """Return players and balls found in a BGR frame."""

    def describe(self) -> dict:
        return {
            "model": self.spec.name,
            "family": self.spec.family,
            "runtime": self.runtime,
            "model_version": self.model_version,
            "end_to_end": self.spec.end_to_end,
            "description": self.spec.description,
            "detected_classes": {"person": OBJECT_TYPE_PLAYER, "sports ball": OBJECT_TYPE_BALL},
        }

def ensure_model_file(url: str | None, model_dir: Path, auto_download: bool) -> Path:
    """Return the local model path, downloading the official file once if allowed."""
    if not url:
        raise ModelLoadError("The selected model has no file for this runtime.")

    model_dir.mkdir(parents=True, exist_ok=True)
    target_path = model_dir / url.rsplit("/", 1)[-1]
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    if not auto_download:
        raise ModelLoadError(
            f"Model file '{target_path}' not found and MODEL_AUTO_DOWNLOAD is disabled."
        )

    logger.info("Downloading model from %s", url)
    partial_path = target_path.with_suffix(target_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial_path)  # noqa: S310 - fixed official URL
        partial_path.replace(target_path)
    except Exception as error:  # network or file system problems
        partial_path.unlink(missing_ok=True)
        raise ModelLoadError(f"Could not download model from {url}: {error}") from error
    return target_path

def letterbox(frame_bgr: np.ndarray, target_size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio and pad to a square, as YOLO expects."""
    frame_height, frame_width = frame_bgr.shape[:2]
    scale = min(target_size / frame_height, target_size / frame_width)
    resized_width, resized_height = round(frame_width * scale), round(frame_height * scale)
    resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_size - resized_width) / 2.0
    pad_y = (target_size - resized_height) / 2.0
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, float(left), float(top)

class OnnxYoloDetector(ObjectDetector):
    """Runs official Ultralytics ONNX exports with onnxruntime (no PyTorch required)."""

    def __init__(self, spec: ModelSpec, model_path: Path, input_size: int = 640) -> None:
        super().__init__(spec, runtime="onnx")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ModelLoadError("onnxruntime is not installed.") from error

        try:
            self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        except Exception as error:
            raise ModelLoadError(f"onnxruntime could not load '{model_path}': {error}") from error

        self._input_name = self._session.get_inputs()[0].name
        self._input_size = input_size
        metadata = self._session.get_modelmeta().custom_metadata_map
        self.model_version = f"ultralytics-{metadata.get('version', 'unknown')}"
        self._end_to_end = metadata.get("end2end", str(spec.end_to_end)).lower() == "true"
        self._class_names = self._parse_class_names(metadata.get("names"))
        self._verify_expected_classes()

    @staticmethod
    def _parse_class_names(raw_names: str | None) -> dict[int, str]:
        if not raw_names:
            return {}
        try:
            return {int(key): str(value) for key, value in ast.literal_eval(raw_names).items()}
        except (ValueError, SyntaxError):
            return {}

    def _verify_expected_classes(self) -> None:
        if not self._class_names:
            return
        if self._class_names.get(COCO_PERSON_CLASS_ID) != "person":
            raise ModelLoadError("Model class 0 is not 'person'; a COCO model is required.")
        if self._class_names.get(COCO_SPORTS_BALL_CLASS_ID) != "sports ball":
            raise ModelLoadError("Model class 32 is not 'sports ball'; a COCO model is required.")

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float) -> list[Detection]:
        padded, scale, pad_x, pad_y = letterbox(frame_bgr, self._input_size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[np.newaxis]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0

        started = time.perf_counter()
        raw_output = self._session.run(None, {self._input_name: blob})[0]
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        if self._end_to_end:
            boxes, scores, class_ids = self._decode_end_to_end(raw_output)
        else:
            boxes, scores, class_ids = self._decode_with_nms(raw_output, confidence_threshold)

        frame_height, frame_width = frame_bgr.shape[:2]
        detections: list[Detection] = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            object_type = CLASS_ID_TO_OBJECT_TYPE.get(int(class_id))
            if object_type is None or score < confidence_threshold:
                continue
            x1 = float(np.clip((box[0] - pad_x) / scale, 0, frame_width))
            y1 = float(np.clip((box[1] - pad_y) / scale, 0, frame_height))
            x2 = float(np.clip((box[2] - pad_x) / scale, 0, frame_width))
            y2 = float(np.clip((box[3] - pad_y) / scale, 0, frame_height))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            detections.append(Detection(x1, y1, x2, y2, float(score), int(class_id), object_type))
        return detections

    @staticmethod
    def _decode_end_to_end(raw_output: np.ndarray):
        """YOLO26 export: (1, max_detections, 6) -> x1, y1, x2, y2, score, class_id."""
        rows = raw_output[0]
        return rows[:, :4], rows[:, 4], rows[:, 5].astype(int)

    @staticmethod
    def _decode_with_nms(raw_output: np.ndarray, confidence_threshold: float):
        """YOLO11 export: (1, 4 + classes, anchors) -> boxes in cx, cy, w, h plus class scores."""
        predictions = raw_output[0].T
        class_scores = predictions[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_scores)), class_ids]

        relevant = np.isin(class_ids, list(CLASS_ID_TO_OBJECT_TYPE)) & (scores >= confidence_threshold)
        predictions, scores, class_ids = predictions[relevant], scores[relevant], class_ids[relevant]
        if len(predictions) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        centers_x, centers_y, widths, heights = predictions[:, :4].T
        boxes_xyxy = np.stack(
            [centers_x - widths / 2, centers_y - heights / 2, centers_x + widths / 2, centers_y + heights / 2],
            axis=1,
        )
        boxes_xywh = np.stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1], widths, heights], axis=1)
        kept_indices = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(), scores.tolist(), class_ids.tolist(), confidence_threshold, 0.45
        )
        kept_indices = np.array(kept_indices).flatten().astype(int)
        return boxes_xyxy[kept_indices], scores[kept_indices], class_ids[kept_indices]

class UltralyticsDetector(ObjectDetector):
    """Uses the ultralytics package and .pt weights (requires PyTorch).

    Useful for model evaluation (mAP with a labelled dataset) and for RT-DETR.
    """

    def __init__(self, spec: ModelSpec, weights_path: Path) -> None:
        super().__init__(spec, runtime="ultralytics")
        try:
            import ultralytics
            from ultralytics import RTDETR, YOLO
        except ImportError as error:
            raise ModelLoadError(
                "MODEL_RUNTIME=ultralytics requires 'pip install -r requirements-eval.txt'."
            ) from error

        model_class = RTDETR if spec.family == "RT-DETR" else YOLO
        try:
            self._model = model_class(str(weights_path))
        except Exception as error:
            raise ModelLoadError(f"Ultralytics could not load '{weights_path}': {error}") from error
        self.model_version = f"ultralytics-{ultralytics.__version__}"

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float) -> list[Detection]:
        started = time.perf_counter()
        results = self._model.predict(
            frame_bgr,
            conf=confidence_threshold,
            classes=list(CLASS_ID_TO_OBJECT_TYPE),
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        detections: list[Detection] = []
        boxes = results[0].boxes
        for box, score, class_id in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
        ):
            object_type = CLASS_ID_TO_OBJECT_TYPE.get(int(class_id))
            if object_type:
                detections.append(
                    Detection(*map(float, box), float(score), int(class_id), object_type)
                )
        return detections

class DetectorFactory:
    """Factory Method: creates the detector that matches model name and runtime."""

    @staticmethod
    def supported_models() -> list[str]:
        return list(SUPPORTED_MODELS)

    @staticmethod
    def create(model_name: str, runtime: str, model_dir: Path, auto_download: bool) -> ObjectDetector:
        spec = SUPPORTED_MODELS.get(model_name)
        if spec is None:
            raise ModelLoadError(
                f"Unsupported model '{model_name}'. Supported: {', '.join(SUPPORTED_MODELS)}"
            )

        if runtime == "onnx":
            if spec.onnx_url is None:
                raise ModelLoadError(
                    f"{model_name} has no official ONNX file. Use MODEL_RUNTIME=ultralytics."
                )
            model_path = ensure_model_file(spec.onnx_url, model_dir, auto_download)
            return OnnxYoloDetector(spec, model_path)

        if runtime == "ultralytics":
            weights_path = ensure_model_file(spec.weights_url, model_dir, auto_download)
            return UltralyticsDetector(spec, weights_path)

        raise ModelLoadError(f"Unsupported runtime '{runtime}'. Use 'onnx' or 'ultralytics'.")
