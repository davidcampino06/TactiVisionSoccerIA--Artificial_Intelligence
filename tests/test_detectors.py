from pathlib import Path

import numpy as np
import pytest

from detectors import DetectorFactory, ModelLoadError, letterbox

def test_unknown_model_is_rejected():
    with pytest.raises(ModelLoadError):
        DetectorFactory.create("YOLO99", "onnx", Path("models"), False)

def test_rtdetr_has_no_onnx_runtime():
    with pytest.raises(ModelLoadError, match="ultralytics"):
        DetectorFactory.create("RT-DETR-l", "onnx", Path("models"), False)

def test_missing_file_without_download_is_reported(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        DetectorFactory.create("YOLO26n", "onnx", tmp_path, False)

def test_letterbox_keeps_aspect_ratio():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = letterbox(frame, 640)
    assert padded.shape == (640, 640, 3)
    assert scale == pytest.approx(1.0)
    assert pad_x == 0 and pad_y == 140
