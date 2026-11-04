"""Exports the real detections of a clip for the landing page hero (frontend/public/hero).

    python tools/landing_detections.py <clip.mp4> <out.json> [--stride 2]

Runs the same detector (YOLO26n, ONNX, CPU) and tracker (ByteTrack) as the analysis, on 1 of every
``stride`` frames, and writes each frame's player boxes as fractions of the image with their track
id (null when ByteTrack has not confirmed the track yet) and confidence, plus the totals shown under
the hero. Nothing is drawn or adjusted by hand: the landing shows exactly what the model found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detectors import OBJECT_TYPE_PLAYER, DetectorFactory  # noqa: E402
from tracking import TrackerFactory  # noqa: E402

CONFIDENCE = 0.25


def _iou(a, b) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    return inter / (a.width * a.height + b.width * b.height - inter + 1e-9)


def export(clip: Path, out: Path, stride: int) -> dict:
    detector = DetectorFactory.create("YOLO26n", "onnx", Path(__file__).resolve().parent.parent / "models", True)
    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = TrackerFactory.create("ByteTrack", frame_rate=fps / stride)
    frames, counts, confidences, track_ids, index = [], [], [], set(), 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            players = [d for d in detector.detect(frame, CONFIDENCE)
                       if d.object_type == OBJECT_TYPE_PLAYER and d.confidence >= CONFIDENCE]
            tracked = tracker.update(players)
            boxes = []
            for detection in players:
                match = max(tracked, key=lambda t: _iou(t.detection, detection), default=None)
                track_id = int(match.track_id) if match is not None and _iou(match.detection, detection) > 0.6 else None
                if track_id is not None:
                    track_ids.add(track_id)
                confidences.append(detection.confidence)
                boxes.append([round(detection.x1 / width, 4), round(detection.y1 / height, 4),
                              round(detection.width / width, 4), round(detection.height / height, 4),
                              track_id, round(detection.confidence, 2)])
            frames.append(boxes)
            counts.append(len(players))
        index += 1
    capture.release()
    result = {
        "source": "Detección real de YOLO26n + ByteTrack sobre este clip, en CPU (ai/tools/landing_detections.py).",
        "model": "YOLO26n", "tracker": "ByteTrack", "fps": fps, "stride": stride,
        "aspect": round(width / height, 4),
        "players_median": float(np.median(counts)) if counts else 0.0,
        "tracks": len(track_ids),
        "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0,
        "frames": frames,
    }
    out.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    summary = export(args.clip, args.out, args.stride)
    print(f"{len(summary['frames'])} frames, {summary['players_median']:.0f} players per frame (median), "
          f"{summary['tracks']} tracks, mean confidence {summary['mean_confidence']:.2f}")
