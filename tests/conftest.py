import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))
os.environ.setdefault("AI_SERVICE_API_KEY", "test-key")
os.environ.setdefault("MODEL_DIR", str(SERVICE_DIR / "models"))


@pytest.fixture(scope="session")
def synthetic_pitch_video(tmp_path_factory) -> Path:
    """A short video: green pitch with two groups of coloured rectangles.

    It has no real people, so YOLO is expected to find few or none; it is used
    to exercise decoding, sampling and error handling, not detection quality.
    """
    path = tmp_path_factory.mktemp("videos") / "pitch.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for frame_index in range(30):
        frame = np.full((360, 640, 3), (40, 140, 40), dtype=np.uint8)
        for player in range(5):
            x = 60 + player * 40 + frame_index
            cv2.rectangle(frame, (x, 100 + player * 30), (x + 12, 130 + player * 30), (0, 0, 220), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def people_video(tmp_path_factory) -> Path | None:
    """Video built from a real photo with people (Ultralytics sample 'zidane.jpg') if available."""
    source = os.getenv("PEOPLE_IMAGE_PATH")
    if not source or not Path(source).exists():
        return None
    image = cv2.resize(cv2.imread(source), (640, 360))
    path = tmp_path_factory.mktemp("videos") / "people.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for shift in range(20):
        writer.write(np.roll(image, shift * 2, axis=1))
    writer.release()
    return path
