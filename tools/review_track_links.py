"""Checks the suggested track links of a real clip by eye: the two photos of every link side by side.

    python tools/review_track_links.py <clip.mp4> <out.jpg> [--stride 3] [--max-frames 300]

Runs the whole analysis (YOLO26n ONNX on CPU, ByteTrack, team colours, track stitching), exactly as
the service does, and draws one row per suggested link: the player in the LAST frame of the track
that ends, the player in the FIRST frame of the track that starts (cut from the video with some
context around and his box drawn), and the numbers behind the suggestion. A person looks at the
sheet and counts how many links are right; that count is what goes in MODEL_EVALUATION.md, not a
number made up here. It also prints the links as JSON so they can be compared between runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detectors import DetectorFactory  # noqa: E402
from pipeline import AnalysisOptions, VideoAnalysisPipeline  # noqa: E402

PHOTO_HEIGHT = 140
TEXT_WIDTH = 330
GREEN = (60, 180, 60)
WHITE = (240, 240, 240)


def _read_frame(clip: Path, frame_number: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(clip))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _photo(clip: Path, box: dict | None) -> np.ndarray:
    """The player at that moment, with one box of context on every side and his box in green."""
    frame = None if box is None else _read_frame(clip, box["frame_number"])
    if frame is None:
        blank = np.full((PHOTO_HEIGHT, 70, 3), 40, np.uint8)
        cv2.putText(blank, "sin foto", (4, PHOTO_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
        return blank
    x, y = box["bounding_box_x"], box["bounding_box_y"]
    w, h = box["bounding_box_width"], box["bounding_box_height"]
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), GREEN, 2)
    top, bottom = max(0, int(y - h * 0.5)), min(frame.shape[0], int(y + h * 1.5))
    left, right = max(0, int(x - w)), min(frame.shape[1], int(x + w * 2))
    crop = frame[top:bottom, left:right]
    scale = PHOTO_HEIGHT / crop.shape[0]
    return cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), PHOTO_HEIGHT))


def contact_sheet(clip: Path, result: dict) -> np.ndarray:
    boxes = {(item["track_id"], item["frame_number"]): item for item in result["detections"]
             if item["track_id"] is not None}
    teams = {item["track_id"]: item["team"] for item in result["tracking"]}
    rows = []
    for link in result["track_links"]["links"]:
        before = _photo(clip, boxes.get((link["from_track"], link["from_last_frame"])))
        after = _photo(clip, boxes.get((link["to_track"], link["to_first_frame"])))
        text = np.full((PHOTO_HEIGHT, TEXT_WIDTH, 3), 25, np.uint8)
        lines = [
            f"ID {link['from_track']} -> ID {link['to_track']}   conf {link['confidence']:.2f}",
            f"cuadro {link['from_last_frame']} -> {link['to_first_frame']}  ({link['gap_seconds']:.2f} s)",
            f"distancia {link['distance_heights']:.2f} alturas",
            f"color {link['color_distance']}   equipo {teams.get(link['from_track'])}/{teams.get(link['to_track'])}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(text, line, (8, 26 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        arrow = np.full((PHOTO_HEIGHT, 30, 3), 25, np.uint8)
        cv2.arrowedLine(arrow, (4, PHOTO_HEIGHT // 2), (26, PHOTO_HEIGHT // 2), GREEN, 2, tipLength=0.4)
        rows.append(np.hstack([before, arrow, after, text]))
    if not rows:
        empty = np.full((60, 420, 3), 25, np.uint8)
        cv2.putText(empty, "Sin enlaces sugeridos en este clip", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        return empty
    width = max(row.shape[1] for row in rows)
    padded = [np.pad(row, ((0, 6), (0, width - row.shape[1]), (0, 0)), constant_values=12) for row in rows]
    return np.vstack(padded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    detector = DetectorFactory.create("YOLO26n", "onnx", ROOT / "models", True)
    result = VideoAnalysisPipeline(detector).analyze_video(
        args.clip, AnalysisOptions(frame_stride=args.stride, max_frames=args.max_frames))
    cv2.imwrite(str(args.out), contact_sheet(args.clip, result))
    print(json.dumps({
        "clip": args.clip.name,
        "tracks": len(result["tracking"]),
        "links": result["track_links"]["links"],
        "groups": result["track_links"]["groups"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
