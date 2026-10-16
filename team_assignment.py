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
from dataclasses import dataclass, field
from typing import Hashable
import cv2
import numpy as np
from detectors import Detection
from tracking import TrackedDetection

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNASSIGNED = "UNASSIGNED"

MIN_TRACKS_FOR_CLUSTERING = 4
OUTLIER_DISTANCE_FACTOR = 2.5


@dataclass
class TeamAssignmentResult:
    """Team of every track, the two jersey colours and how well they separate."""
    track_teams: dict[int, str]
    team_colors_hex: dict[str, str]
    separation_score: float
    warnings: list[str]
    # Team of every player seen without a track id (key chosen by the caller, e.g. (frame, index)).
    detection_teams: dict[Hashable, str] = field(default_factory=dict)

def _torso_color_lab(frame_bgr: np.ndarray, box: Detection) -> np.ndarray | None:
    """Mean Lab colour of the shirt area, ignoring green (pitch) pixels."""
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
    """Collects colour samples while the video is processed, then clusters once.

    The two team colours come only from tracks (many samples each, so a shadow or a blurry frame
    does not move them). Players the tracker could not follow are then given the nearest team
    colour, with the same rule for referees and goalkeepers (too far from both -> UNASSIGNED)."""

    def __init__(self) -> None:
        self._samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._mean_x: dict[int, list[float]] = defaultdict(list)
        self._loose: dict[Hashable, np.ndarray] = {}

    def collect(self, frame_bgr: np.ndarray, tracked_players: list[TrackedDetection]) -> None:
        """Samples the torso colour of every tracked player in this frame."""
        for tracked in tracked_players:
            color = _torso_color_lab(frame_bgr, tracked.detection)
            if color is not None:
                self._samples[tracked.track_id].append(color)
                self._mean_x[tracked.track_id].append(tracked.detection.foot_point[0])

    def collect_untracked(self, frame_bgr: np.ndarray, key: Hashable, detection: Detection) -> None:
        """Samples the shirt of a player seen in one frame only (no track id yet)."""
        color = _torso_color_lab(frame_bgr, detection)
        if color is not None:
            self._loose[key] = color

    def assign(self) -> TeamAssignmentResult:
        """Clusters the colour samples in two teams (k-means); tracks in neither stay UNASSIGNED."""
        track_ids = [track_id for track_id, samples in self._samples.items() if samples]
        if len(track_ids) < MIN_TRACKS_FOR_CLUSTERING:
            return TeamAssignmentResult(
                track_teams={track_id: UNASSIGNED for track_id in track_ids},
                team_colors_hex={},
                detection_teams={key: UNASSIGNED for key in self._loose},
                separation_score=0.0,
                warnings=[
                    f"Solo {len(track_ids)} trayectorias de jugadores con camiseta visible; "
                    "no fue posible separar los equipos."
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

        def is_outlier(distance: float) -> bool:
            return distance > OUTLIER_DISTANCE_FACTOR * typical_distance and distance > 12.0

        track_teams: dict[int, str] = {}
        outliers = 0
        for track_id, label, distance in zip(track_ids, labels, distances):
            if is_outlier(distance):
                track_teams[track_id] = UNASSIGNED
                outliers += 1
            else:
                track_teams[track_id] = cluster_to_team[int(label)]

        detection_teams: dict[Hashable, str] = {}
        for key, color in self._loose.items():
            to_centers = np.linalg.norm(centers - color, axis=1)
            nearest = int(np.argmin(to_centers))
            detection_teams[key] = UNASSIGNED if is_outlier(float(to_centers[nearest])) else cluster_to_team[nearest]

        warnings: list[str] = []
        if separation_score < 0.5:
            warnings.append(
                f"Baja separación de color entre equipos (puntaje {separation_score}); "
                "los indicadores por equipo podrían no ser confiables."
            )
        if outliers:
            warnings.append(
                f"{outliers} trayectorias no coincidieron con el color de ningún equipo (posibles árbitros o arqueros)."
            )

        return TeamAssignmentResult(
            track_teams=track_teams,
            team_colors_hex={cluster_to_team[c]: _lab_to_hex(centers[c]) for c in (0, 1)},
            separation_score=separation_score,
            warnings=warnings,
            detection_teams=detection_teams,
        )
