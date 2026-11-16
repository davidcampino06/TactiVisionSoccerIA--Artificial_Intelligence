"""One photo per track (track_crops.py): the staff recognises the player and links him to the squad."""

import base64

import cv2
import numpy as np

from detectors import OBJECT_TYPE_PLAYER, Detection
from pipeline import AnalysisOptions, VideoAnalysisPipeline
from track_crops import MAX_HEIGHT_PX, MIN_HEIGHT_PX, TrackCropCollector
from tracking import TrackedDetection


def player(x1, y1, x2, y2, confidence=0.8):
    return Detection(x1, y1, x2, y2, confidence, 0, OBJECT_TYPE_PLAYER)


def tracked(track_id, box):
    return TrackedDetection(track_id, box)


def frame(width=640, height=360, color=(40, 140, 40)):
    return np.full((height, width, 3), color, dtype=np.uint8)


def decode(item):
    return cv2.imdecode(np.frombuffer(base64.b64decode(item["image_jpeg_base64"]), np.uint8), cv2.IMREAD_COLOR)


def test_keeps_the_clearest_view_of_each_track():
    collector = TrackCropCollector()
    collector.collect(frame(), [tracked(1, player(100, 100, 130, 170, 0.5))], 0, 0.0)
    collector.collect(frame(), [tracked(1, player(100, 100, 140, 190, 0.9))], 5, 0.2)  # bigger and surer
    collector.collect(frame(), [tracked(1, player(100, 100, 130, 160, 0.6))], 10, 0.4)
    (item,) = collector.payload()
    assert (item["track_id"], item["frame_number"], item["timestamp"], item["confidence"]) == (1, 5, 0.2, 0.9)


def test_a_player_cut_by_the_border_loses_to_a_whole_one():
    collector = TrackCropCollector()
    collector.collect(frame(), [tracked(7, player(0, 100, 40, 190, 0.9))], 0, 0.0)      # touches the left border
    collector.collect(frame(), [tracked(7, player(100, 100, 138, 186, 0.9))], 5, 0.2)   # a bit smaller, whole
    assert collector.payload()[0]["frame_number"] == 5


def test_a_crop_shared_with_another_player_loses_to_a_clean_one():
    collector = TrackCropCollector()
    # Frame 0: track 1 is almost hidden behind track 2. Frame 5: alone.
    collector.collect(frame(), [tracked(1, player(100, 100, 140, 190)), tracked(2, player(105, 100, 145, 190))], 0, 0.0)
    collector.collect(frame(), [tracked(1, player(100, 100, 136, 184))], 5, 0.2)
    first = {item["track_id"]: item["frame_number"] for item in collector.payload()}
    assert first[1] == 5


def test_tiny_boxes_are_never_kept_and_photos_are_small():
    collector = TrackCropCollector()
    collector.collect(frame(), [tracked(3, player(10, 10, 20, 10 + MIN_HEIGHT_PX - 1))], 0, 0.0)
    assert collector.payload() == []
    collector.collect(frame(height=1080, width=1920), [tracked(4, player(500, 100, 700, 700))], 0, 0.0)
    (item,) = collector.payload()
    assert item["height"] <= MAX_HEIGHT_PX and decode(item).shape[:2] == (item["height"], item["width"])
    assert len(base64.b64decode(item["image_jpeg_base64"])) < 20_000


def test_the_photo_shows_the_player_not_the_background():
    collector = TrackCropCollector()
    image = frame()
    cv2.rectangle(image, (200, 100), (240, 190), (0, 0, 230), -1)  # a red shirt on green grass
    collector.collect(image, [tracked(9, player(200, 100, 241, 191))], 0, 0.0)
    blue, green, red = decode(collector.payload()[0]).reshape(-1, 3).mean(axis=0)
    assert red > 150 and green < 110  # mostly the red player, only the thin margin is grass


def test_the_pipeline_returns_the_photo_of_each_track_from_the_real_frame(tmp_path):
    """Two players with distinct colours walk through a video: each track's photo has its own colour."""
    path = tmp_path / "two.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for index in range(40):
        image = frame()
        cv2.rectangle(image, (80 + index, 120), (120 + index, 210), (0, 0, 230), -1)    # red, left
        cv2.rectangle(image, (400 - index, 120), (440 - index, 210), (230, 60, 0), -1)  # blue, right
        writer.write(image)
    writer.release()

    class Detector:
        model_name, model_version, runtime, last_inference_ms = "fake", "0", "cpu", 0.0

        def detect(self, image, confidence):
            found = []
            for point, color in (((80, 120), (0, 0, 230)), ((400, 120), (230, 60, 0))):
                mask = cv2.inRange(image, np.array(color) - 5, np.array(color) + 5)
                ys, xs = np.nonzero(mask)
                if len(xs):
                    found.append(player(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.85))
            return found

    result = VideoAnalysisPipeline(Detector()).analyze_video(
        path, AnalysisOptions(frame_stride=1, max_frames=40, pitch_filter=False))
    photos = result["track_snapshots"]
    assert len(photos) == 2 and {p["track_id"] for p in photos} == {t["track_id"] for t in result["tracking"]}
    means = sorted((decode(p).reshape(-1, 3).mean(axis=0) for p in photos), key=lambda bgr: bgr[2])
    assert means[0][0] > 150 and means[1][2] > 150  # one photo is the blue player, the other the red one
