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
import re
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
# The user's kit colour belongs to a team only if it is clearly closer to it than to the other one.
# With the colour weights (stage 7), real kits on 7 broadcast clips and white / yellow kits on 6
# eFootball clips gave at most 0.654 for the right team; a green kit against red and blue teams
# must not be forced into the red one.
OWN_TEAM_MAX_RATIO = 0.68
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
OUTLIER_DISTANCE_FACTOR = 2.5
# Weights of L (lightness), a and b when comparing shirts. Shadows and the camera change lightness a
# lot, the colour itself much less: with full weight on L, a white team and a yellow one fell in the
# same cluster and the split was "players in shadow" against "players in the light" (eFootball clips,
# CALIBRATION.md stage 7). Lightness still counts, so white against black kits stays apart.
COLOR_WEIGHTS = np.array([0.35, 1.0, 1.0], dtype=np.float32)
# Tracks seen in fewer frames do not choose the team colours (one blurry sample is noise); they are
# then given the nearest team like any other player.
MIN_SAMPLES_TO_CLUSTER = 2
# A "team" holding less than this share of the shirt samples is not a team: it is the goalkeeper,
# the referee or a colourful outlier that pulled k-means to itself (a lone orange keeper against
# white and black teams). Those tracks are set aside and the two teams are searched again.
MIN_TEAM_SHARE = 0.2
MAX_TRIMS = 2


@dataclass
class TeamAssignmentResult:
    """Team of every track, the two jersey colours and how well they separate."""
    track_teams: dict[int, str]
    team_colors_hex: dict[str, str]
    separation_score: float
    warnings: list[str]
    # Team of every player seen without a track id (key chosen by the caller, e.g. (frame, index)).
    detection_teams: dict[Hashable, str] = field(default_factory=dict)
    # Which detected team wears the kit colour the user gave (None when not given or not clear).
    own_team: str | None = None


def hex_to_lab(hex_color: str) -> np.ndarray:
    """"#rrggbb" -> Lab in OpenCV's 8-bit scale, the same space as the shirt samples."""
    red, green, blue = (int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    pixel = np.uint8([[[blue, green, red]]])
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def match_own_team(team_centers: dict[str, np.ndarray], kit_color: str) -> tuple[str | None, float]:
    """(team whose shirt colour is clearly the closest to ``kit_color`` or None, distance ratio).

    The shirts measured in the video are an average with shadows, duller than the real kit, so no
    absolute distance is required: the kit only has to be much closer to one team than to the other."""
    target = hex_to_lab(kit_color)
    # Same weights as the clustering: the measured shirts are darker than the kit (shadows).
    distances = {team: float(np.linalg.norm((center - target) * COLOR_WEIGHTS)) for team, center in team_centers.items()}
    if len(distances) != 2:
        return None, 1.0
    nearest, farthest = sorted(distances, key=distances.get)
    ratio = distances[nearest] / distances[farthest] if distances[farthest] else 1.0
    return (nearest if ratio <= OWN_TEAM_MAX_RATIO else None), round(ratio, 3)

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

    def assign(self, own_kit_color: str | None = None) -> TeamAssignmentResult:
        """Clusters the colour samples in two teams (k-means); tracks in neither stay UNASSIGNED.

        The teams are always found from the video alone. ``own_kit_color`` (#rrggbb, optional) only
        says which of the two is the user's team; if it matches neither clearly, nothing is marked."""
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

        colors = np.array([np.median(self._samples[t], axis=0) for t in track_ids], dtype=np.float32)
        features = colors * COLOR_WEIGHTS
        steady = np.array([len(self._samples[t]) >= MIN_SAMPLES_TO_CLUSTER for t in track_ids])
        if steady.sum() < MIN_TRACKS_FOR_CLUSTERING:
            steady[:] = True
        weights = np.array([len(self._samples[t]) for t in track_ids], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        used = steady.copy()
        for _ in range(MAX_TRIMS + 1):
            cv2.setRNGSeed(42)
            _, used_labels, centers = cv2.kmeans(features[used], 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
            used_labels = used_labels.flatten()
            shares = [weights[used][used_labels == cluster].sum() / weights[used].sum() for cluster in (0, 1)]
            small = int(np.argmin(shares))
            if shares[small] >= MIN_TEAM_SHARE or used.sum() - (used_labels == small).sum() < MIN_TRACKS_FOR_CLUSTERING:
                break
            used[np.flatnonzero(used)[used_labels == small]] = False  # keeper / referee: out, and search again
        labels = np.argmin(np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2), axis=1)
        for cluster in (0, 1):  # every cluster keeps at least one track (k-means guarantees it on the steady ones)
            if not np.any(labels == cluster):
                labels[int(np.argmin(np.linalg.norm(features - centers[cluster], axis=1)))] = cluster
        # Team colours in plain Lab (for the swatches and the kit-colour match), from their tracks.
        lab_centers = np.array([colors[labels == cluster].mean(axis=0) for cluster in (0, 1)], dtype=np.float32)

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
            to_centers = np.linalg.norm(centers - color * COLOR_WEIGHTS, axis=1)
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

        own_team = None
        if own_kit_color and HEX_COLOR.match(own_kit_color):
            own_team, ratio = match_own_team({cluster_to_team[c]: lab_centers[c] for c in (0, 1)}, own_kit_color)
            if own_team is None:
                warnings.append(
                    f"El color de tu equipo ({own_kit_color.lower()}) no se parece claramente a ninguno de los dos "
                    "equipos del video: se separaron los equipos automáticamente, sin marcar cuál es el tuyo."
                )

        return TeamAssignmentResult(
            track_teams=track_teams,
            team_colors_hex={cluster_to_team[c]: _lab_to_hex(lab_centers[c]) for c in (0, 1)},
            separation_score=separation_score,
            warnings=warnings,
            detection_teams=detection_teams,
            own_team=own_team,
        )
