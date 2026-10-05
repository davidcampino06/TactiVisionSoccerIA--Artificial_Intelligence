"""Team assignment by jersey colour (between tracking and tactical metrics).

Tactical indicators such as team width or compactness need to know which
players belong to the same team. A COCO detector only says "person", so we
cluster the dominant shirt colour of every track into two groups.

Only clothing colour is used. No faces, no biometrics.

Known limitations (reported in the analysis warnings):
- referees and goalkeepers wear different colours; tracks far from both
  team colours are marked as UNASSIGNED instead of being forced into a team;
- very similar kits (e.g. two dark kits) reduce the separation quality.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import cv2
import numpy as np
from tracking import TrackedDetection

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNASSIGNED = "UNASSIGNED"

MIN_TRACKS_FOR_CLUSTERING = 4
OUTLIER_DISTANCE_FACTOR = 2.5


@dataclass
class TeamAssignmentResult:
    track_teams: dict[int, str]
    team_colors_hex: dict[str, str]
    separation_score: float
    warnings: list[str]

def _torso_color_lab(frame_bgr: np.ndarray, tracked: TrackedDetection) -> np.ndarray | None:
    """Mean Lab colour of the shirt area, ignoring green (pitch) pixels."""
    box = tracked.detection
    height, width = box.height, box.width
    x1 = int(box.x1 + width * 0.25)
    x2 = int(box.x2 - width * 0.25)
    y1 = int(box.y1 + height * 0.15)
    y2 = int(box.y1 + height * 0.50)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    crop = frame_bgr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)) > 0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    shirt_pixels = lab[~green_mask.reshape(-1)]
    if len(shirt_pixels) < 4:
        return None
    return shirt_pixels.mean(axis=0)

def _lab_to_hex(lab_color: np.ndarray) -> str:
    lab_pixel = np.clip(lab_color, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    blue, green, red = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"

class JerseyColorTeamClassifier:
    """Collects colour samples while the video is processed, then clusters once."""

    def __init__(self) -> None:
        self._samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._mean_x: dict[int, list[float]] = defaultdict(list)

    def collect(self, frame_bgr: np.ndarray, tracked_players: list[TrackedDetection]) -> None:
        for tracked in tracked_players:
            color = _torso_color_lab(frame_bgr, tracked)
            if color is not None:
                self._samples[tracked.track_id].append(color)
                self._mean_x[tracked.track_id].append(tracked.detection.foot_point[0])

    def assign(self) -> TeamAssignmentResult:
        track_ids = [track_id for track_id, samples in self._samples.items() if samples]
        if len(track_ids) < MIN_TRACKS_FOR_CLUSTERING:
            return TeamAssignmentResult(
                track_teams={track_id: UNASSIGNED for track_id in track_ids},
                team_colors_hex={},
                separation_score=0.0,
                warnings=[
                    f"Only {len(track_ids)} player tracks with visible shirts; "
                    "team separation was not possible."
                ],
            )

        features = np.array([np.median(self._samples[t], axis=0) for t in track_ids], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        # Deterministic naming: TEAM_A is the cluster that appears more on the left of the image.
        cluster_mean_x = [
            np.mean([np.mean(self._mean_x[t]) for t, label in zip(track_ids, labels) if label == cluster])
            if np.any(labels == cluster) else np.inf
            for cluster in (0, 1)
        ]
        team_a_cluster = int(np.argmin(cluster_mean_x))
        cluster_to_team = {team_a_cluster: TEAM_A, 1 - team_a_cluster: TEAM_B}

        distances = np.linalg.norm(features - centers[labels], axis=1)
        typical_distance = float(np.median(distances)) or 1.0
        center_gap = float(np.linalg.norm(centers[0] - centers[1]))
        separation_score = round(min(center_gap / (typical_distance * 4.0), 1.0), 4)

        track_teams: dict[int, str] = {}
        outliers = 0
        for track_id, label, distance in zip(track_ids, labels, distances):
            if distance > OUTLIER_DISTANCE_FACTOR * typical_distance and distance > 12.0:
                track_teams[track_id] = UNASSIGNED
                outliers += 1
            else:
                track_teams[track_id] = cluster_to_team[int(label)]

        warnings: list[str] = []
        if separation_score < 0.5:
            warnings.append(
                f"Low colour separation between teams (score {separation_score}); "
                "team-based indicators may be unreliable."
            )
        if outliers:
            warnings.append(
                f"{outliers} tracks did not match either team colour (possible referees or goalkeepers)."
            )

        return TeamAssignmentResult(
            track_teams=track_teams,
            team_colors_hex={cluster_to_team[c]: _lab_to_hex(centers[c]) for c in (0, 1)},
            separation_score=separation_score,
            warnings=warnings,
        )
