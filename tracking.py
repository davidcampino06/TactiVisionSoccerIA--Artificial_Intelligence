"""Tracking layer (step 2 of the AI pipeline).

Detecting players frame by frame is not enough: tactical metrics need to
know that a box in frame 10 and a box in frame 13 are the same player.
The tracker assigns a temporal ``track_id`` (Player Track 1, 2, 3...).

No face recognition and no biometrics are used. A track id is only a
temporary number inside one video; it never identifies a real person.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from detectors import Detection

@dataclass(frozen=True)
class TrackedDetection:
    """A player detection with its temporal identity."""

    track_id: int
    detection: Detection

class ObjectTracker(ABC):
    """Abstraction so ByteTrack can be swapped by BoT-SORT later (Open/Closed)."""

    name: str = "abstract"
    @abstractmethod
    def update(self, player_detections: list[Detection]) -> list[TrackedDetection]:
        """Associate this frame's detections with existing tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Forget all tracks (one tracker instance per video)."""

class ByteTrackTracker(ObjectTracker):
    """ByteTrack implementation provided by the ``supervision`` package (NumPy only)."""

    name = "ByteTrack"
    def __init__(self, frame_rate: float, lost_track_buffer: int = 30, activation_threshold: float = 0.25):
        import supervision as sv  # imported lazily to keep module import cheap

        self._sv = sv
        self._frame_rate = max(frame_rate, 1.0)
        self._lost_track_buffer = lost_track_buffer
        self._activation_threshold = activation_threshold
        self._tracker = self._build()

    def _build(self):
        return self._sv.ByteTrack(
            track_activation_threshold=self._activation_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=self._frame_rate,
            minimum_consecutive_frames=1,
        )

    def reset(self) -> None:
        self._tracker = self._build()

    def update(self, player_detections: list[Detection]) -> list[TrackedDetection]:
        if not player_detections:
            empty = self._sv.Detections.empty()
            self._tracker.update_with_detections(empty)
            return []

        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in player_detections], dtype=np.float32)
        detections = self._sv.Detections(
            xyxy=boxes,
            confidence=np.array([d.confidence for d in player_detections], dtype=np.float32),
            class_id=np.array([d.class_id for d in player_detections], dtype=int),
            data={"source_index": np.arange(len(player_detections))},
        )
        tracked = self._tracker.update_with_detections(detections)

        results: list[TrackedDetection] = []
        for source_index, tracker_id in zip(tracked.data["source_index"], tracked.tracker_id):
            if tracker_id is None or tracker_id < 0:
                continue
            results.append(TrackedDetection(int(tracker_id), player_detections[int(source_index)]))
        return results

class TrackerFactory:
    """Creates trackers by name. Only ByteTrack is implemented in this phase."""

    @staticmethod
    def create(tracker_name: str, frame_rate: float) -> ObjectTracker:
        if tracker_name.lower() == "bytetrack":
            return ByteTrackTracker(frame_rate=frame_rate)
        raise ValueError(f"Unsupported tracker '{tracker_name}'. Available: ByteTrack")
