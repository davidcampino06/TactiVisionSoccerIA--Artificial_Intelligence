"""One photo per track, so a person can tell who is who (step between tracking and the result).

A track id is only a number. To link it to a squad player the staff has to recognise the player, and
for that they need to SEE him. While the video is decoded (the frames are already in memory) this
keeps, for every track, the best crop of the video itself: the clearest view of that player.

"Best" is the crop with the highest score = confidence x height of the box, discounted when:
  - the box touches the border of the frame (the player is cut off),
  - other players overlap it (the photo would show somebody else: score x (1 - largest overlap)).
Boxes shorter than MIN_HEIGHT_PX are never kept: nobody can be recognised in 20 pixels.

Only one crop per track is held in memory at a time (a replacement happens only when the score
improves), and at the end each one is scaled to at most MAX_HEIGHT_PX and encoded as JPEG, a few
kilobytes each. No face recognition and no biometrics: it is a picture cut out of the video, the
same one the user uploaded, and the person decides who it is.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from detectors import Detection
from tracking import TrackedDetection

MIN_HEIGHT_PX = 40
MAX_HEIGHT_PX = 160
JPEG_QUALITY = 82
# Margin around the box, as a fraction of its size: a bit of context makes the player easier to recognise.
PADDING = 0.08
BORDER_PX = 2
BORDER_PENALTY = 0.5


def _overlap(a: Detection, b: Detection) -> float:
    """Share of box ``a`` covered by box ``b`` (0 to 1)."""
    width = min(a.x2, b.x2) - max(a.x1, b.x1)
    height = min(a.y2, b.y2) - max(a.y1, b.y1)
    if width <= 0 or height <= 0:
        return 0.0
    return (width * height) / max(a.width * a.height, 1e-9)


@dataclass
class _Candidate:
    score: float
    frame_number: int
    timestamp: float
    confidence: float
    crop: np.ndarray


class TrackCropCollector:
    """Keeps the best crop of every track while the video is read, frame by frame."""

    def __init__(self) -> None:
        self._best: dict[int, _Candidate] = {}

    def collect(self, frame: np.ndarray, tracked_players: list[TrackedDetection], frame_number: int,
                timestamp: float) -> None:
        height, width = frame.shape[:2]
        boxes = [tracked.detection for tracked in tracked_players]
        for tracked in tracked_players:
            box = tracked.detection
            if box.height < MIN_HEIGHT_PX or box.width <= 0:
                continue
            score = box.confidence * box.height
            if box.x1 <= BORDER_PX or box.y1 <= BORDER_PX or box.x2 >= width - BORDER_PX or box.y2 >= height - BORDER_PX:
                score *= BORDER_PENALTY
            covered = max((_overlap(box, other) for other in boxes if other is not box), default=0.0)
            score *= 1.0 - min(covered, 0.95)
            current = self._best.get(tracked.track_id)
            if current is not None and current.score >= score:
                continue
            pad_x, pad_y = box.width * PADDING, box.height * PADDING
            left, top = max(0, int(box.x1 - pad_x)), max(0, int(box.y1 - pad_y))
            right, bottom = min(width, int(box.x2 + pad_x) + 1), min(height, int(box.y2 + pad_y) + 1)
            if right - left < 4 or bottom - top < 4:
                continue
            self._best[tracked.track_id] = _Candidate(score, frame_number, timestamp, box.confidence,
                                                      frame[top:bottom, left:right].copy())

    def payload(self) -> list[dict]:
        """One entry per track with a usable crop, ordered by track id; the JPEG goes in base64."""
        items = []
        for track_id in sorted(self._best):
            candidate = self._best[track_id]
            crop = candidate.crop
            if crop.shape[0] > MAX_HEIGHT_PX:
                scale = MAX_HEIGHT_PX / crop.shape[0]
                crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), MAX_HEIGHT_PX),
                                  interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            items.append({
                "track_id": track_id,
                "frame_number": candidate.frame_number,
                "timestamp": round(candidate.timestamp, 4),
                "confidence": round(candidate.confidence, 4),
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
                "image_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            })
        return items
