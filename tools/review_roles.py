"""Checks the referee / goalkeeper suggestions of a real clip by eye: one row per kit with its photos.

    python tools/review_roles.py <clip.mp4> <out.jpg> [--stride 3] [--max-frames 300]

Runs the whole analysis exactly as the service does and draws, for every kit of no team, the photo
of each of its tracks (track_snapshots) with the suggestion and the kit colour. A person then says
whether every photo in a row is the same kind of person (all the referee, all one goalkeeper...);
that count goes in MODEL_EVALUATION.md. The suggestions are also printed as JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detectors import DetectorFactory  # noqa: E402
from pipeline import AnalysisOptions, VideoAnalysisPipeline  # noqa: E402

PHOTO_HEIGHT = 120
WHITE = (240, 240, 240)


def _photo(snapshot: dict, track_id: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(base64.b64decode(snapshot["image_jpeg_base64"]), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (max(1, round(image.shape[1] * PHOTO_HEIGHT / image.shape[0])), PHOTO_HEIGHT))
    tile = np.full((PHOTO_HEIGHT + 22, max(image.shape[1], 60), 3), 20, np.uint8)
    tile[:PHOTO_HEIGHT, :image.shape[1]] = image
    cv2.putText(tile, f"ID {track_id}", (3, PHOTO_HEIGHT + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    return np.pad(tile, ((0, 0), (0, 4), (0, 0)), constant_values=12)


def contact_sheet(result: dict) -> np.ndarray:
    photos = {item["track_id"]: item for item in result.get("track_snapshots", [])}
    kits: dict[int, list[dict]] = {}
    for suggestion in result["track_role_suggestions"]:
        kits.setdefault(suggestion["kit_group"], []).append(suggestion)
    rows = []
    for group in sorted(kits):
        members = kits[group]
        hex_color = members[0]["kit_color"]
        swatch = np.full((PHOTO_HEIGHT + 22, 190, 3), 25, np.uint8)
        red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        cv2.rectangle(swatch, (8, 10), (40, 42), (blue, green, red), -1)
        for index, line in enumerate([f"kit {group}  {hex_color}", members[0]["suggestion"],
                                      f"{len(members)} trayectorias"]):
            cv2.putText(swatch, line, (8, 64 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        tiles = [_photo(photos[m["track_id"]], m["track_id"]) for m in members if m["track_id"] in photos]
        rows.append(np.hstack([swatch, *tiles]))
    if not rows:
        empty = np.full((60, 460, 3), 25, np.uint8)
        cv2.putText(empty, "Ninguna camiseta fuera de los dos equipos", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        return empty
    width = max(row.shape[1] for row in rows)
    return np.vstack([np.pad(row, ((0, 6), (0, width - row.shape[1]), (0, 0)), constant_values=12) for row in rows])


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
    cv2.imwrite(str(args.out), contact_sheet(result))
    print(json.dumps({"clip": args.clip.name, "tracks": len(result["tracking"]),
                      "suggestions": result["track_role_suggestions"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
