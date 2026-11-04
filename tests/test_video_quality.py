"""Clip check before the analysis (video_quality.py and POST /api/video-quality)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from detectors import OBJECT_TYPE_BALL, OBJECT_TYPE_PLAYER, Detection
from main import app
from pipeline import AnalysisOptions, _FrameRecord, coverage_warning
from video_quality import Mode, assess, coverage_check

API_HEADERS = {"X-API-Key": "test-key"}
FAST, FULL = Mode(5, 120), Mode(3, 300)


class FakeDetector:
    """Twelve players standing on the grass and the ball: what a usable broadcast frame gives."""
    model_name, model_version, runtime, last_inference_ms = "fake", "0", "cpu", 0.0

    def __init__(self, players: int = 12, ball: bool = True):
        self.players, self.ball = players, ball

    def detect(self, frame, confidence):
        height, width = frame.shape[:2]
        found = [Detection(40 + i * 45, height * 0.55, 52 + i * 45, height * 0.55 + height * 0.09, 0.8, 0,
                           OBJECT_TYPE_PLAYER) for i in range(self.players)]
        if self.ball:
            found.append(Detection(300, 250, 306, 256, 0.4, 32, OBJECT_TYPE_BALL))
        return found


def write_video(path, seconds: float, fps: int = 25, size=(640, 360), cut_at: float | None = None,
                brightness: float = 1.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), (40, 140, 40), dtype=np.uint8)
        cv2.rectangle(frame, (index % 200, 40), (index % 200 + 30, 60), (230, 230, 230), -1)  # a slow pan
        if cut_at is not None and index / fps >= cut_at:
            frame = np.full((size[1], size[0], 3), (60, 60, 200), dtype=np.uint8)  # another shot: a close-up
        writer.write(cv2.convertScaleAbs(frame, alpha=brightness))
    writer.release()
    return path


def by_code(result):
    return {check["code"]: check for check in result["checks"]}


def test_coverage_depends_on_the_mode_frame_cap():
    assert coverage_check(18, 30, FAST, FULL)["status"] == "ok"       # fast covers 20 s
    check = coverage_check(26, 30, FAST, FULL)
    assert check["status"] == "warn" and check["value"]["fast_seconds"] == 20.0
    check = coverage_check(60, 30, FAST, FULL)
    assert check["status"] == "bad" and "Recorta" in check["detail"] and check["value"]["full_seconds"] == 30.0
    assert coverage_check(None, 30, FAST, FULL)["status"] == "warn"


def test_a_good_clip_passes_every_check(tmp_path):
    result = assess(write_video(tmp_path / "ok.mp4", 6), FakeDetector(), FAST, FULL)
    checks = by_code(result)
    assert result["verdict"] == "ok", [c for c in result["checks"] if c["status"] != "ok"]
    assert checks["players"]["value"] == 12 and checks["ball"]["value"] == result["sampled_frames"] == 8
    assert checks["cuts"]["measured"] is False and checks["players"]["measured"] is True


def test_cuts_few_players_dark_image_and_no_ball_are_flagged(tmp_path):
    path = write_video(tmp_path / "bad.mp4", 6, cut_at=3.0, brightness=0.2)
    checks = by_code(assess(path, FakeDetector(players=5, ball=False), FAST, FULL))
    assert checks["cuts"]["status"] == "warn" and checks["cuts"]["value"] == [3.0]
    assert checks["players"]["status"] == "bad"
    assert checks["ball"]["status"] == "warn"
    assert checks["light"]["status"] == "warn"


def test_low_resolution_and_low_fps_are_flagged(tmp_path):
    checks = by_code(assess(write_video(tmp_path / "small.mp4", 3, fps=12, size=(320, 180)), FakeDetector(),
                            FAST, FULL))
    assert checks["resolution"]["status"] == "warn" and checks["fps"]["status"] == "warn"


def test_the_analysis_says_when_the_end_of_the_clip_was_not_analysed():
    video = {"fps": 30.0, "duration_seconds": 60.0}
    records = [_FrameRecord(frame_number=i * 5, timestamp_seconds=i * 5 / 30, tracked_players=[], ball=None)
               for i in range(120)]
    warning = coverage_warning(records, video, AnalysisOptions(frame_stride=5, max_frames=120))
    assert warning.startswith("Solo se analizaron los primeros 20 s de un video de 60 s")
    assert coverage_warning(records, {"fps": 30.0, "duration_seconds": 20.0},
                            AnalysisOptions(frame_stride=5, max_frames=120)) is None


def test_endpoint_requires_the_api_key(synthetic_pitch_video):
    with TestClient(app) as client, synthetic_pitch_video.open("rb") as video:
        assert client.post("/api/video-quality", files={"video": ("p.mp4", video, "video/mp4")}).status_code == 401


def test_endpoint_checks_a_clip(synthetic_pitch_video):
    with TestClient(app) as client, synthetic_pitch_video.open("rb") as video:
        response = client.post("/api/video-quality", headers=API_HEADERS,
                               files={"video": ("p.mp4", video, "video/mp4")})
    if response.status_code == 503:
        pytest.skip("model not available in this environment")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verdict"] in ("ok", "warn", "bad") and {c["code"] for c in body["checks"]} >= {"coverage", "players", "cuts"}
