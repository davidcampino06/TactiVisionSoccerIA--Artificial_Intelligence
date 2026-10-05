"""Tactical metrics (step 3 of the AI pipeline).

Converts tracked positions into tactical indicators. These are MEASUREMENTS
taken from the image, not absolute truths:

- positions are image coordinates (no pitch calibration / homography yet);
- the main broadcast camera is assumed: the pitch length runs along the
  image X axis and the pitch width along the image Y axis;
- only visible players are measured.

Units:
- ``frame_width_ratio``: fraction of the image width (0..1). Vertical
  differences are converted with the image aspect ratio so distances are
  comparable in both axes.
- ``frame_height_ratio``: fraction of the image height (0..1).
- ``frame_area_ratio``: fraction of the image area (0..1).

Design pattern - Strategy: each ``IndicatorStrategy`` computes one family of
indicators. ``TacticalAnalyzer`` runs whatever strategies it receives, so new
metrics can be added without modifying existing code (Open/Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from team_assignment import TEAM_A, TEAM_B

MIN_PLAYERS_PER_TEAM = 4
MIN_PLAYERS_FOR_LINES = 6

ATTACK_LEFT_TO_RIGHT = "left_to_right"
ATTACK_RIGHT_TO_LEFT = "right_to_left"
ATTACK_UNKNOWN = "unknown"

# Heuristic reference thresholds for a main broadcast camera.
# They are configurable baselines that must be calibrated with real footage.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "team_width": 0.30,                    # below -> possible insufficient width
    "average_player_distance": 0.30,       # above -> possible low compactness
    "team_compactness_area": 0.25,         # above -> possible low compactness
    "nearest_teammate_distance": 0.15,     # above -> possible excessive spacing
    "defensive_line_position": 0.15,       # below -> possible line too deep (image relative)
    "defensive_midfield_distance": 0.22,   # above -> possible excessive distance between lines
    "largest_line_gap": 0.22,              # above -> same issue when direction is unknown
    "max_zone_concentration": 0.45,        # above -> possible high concentration in one zone
    "max_zone_numerical_difference": 3.0,  # at/above -> possible numerical imbalance
}

@dataclass(frozen=True)
class PlayerPosition:
    track_id: int
    team: str
    x: float  # 0..1 of image width (foot point)
    y: float  # 0..1 of image height (foot point)
    confidence: float

@dataclass
class FrameSnapshot:
    frame_number: int
    timestamp_seconds: float
    players: list[PlayerPosition]
    ball: tuple[float, float] | None = None

    def team_players(self, team: str) -> list[PlayerPosition]:
        return [player for player in self.players if player.team == team]

@dataclass
class AnalysisContext:
    snapshots: list[FrameSnapshot]
    aspect_ratio: float  # image height / image width
    team_a_attack_direction: str = ATTACK_UNKNOWN
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    def own_goal_side(self, team: str) -> str | None:
        """Return 'left' or 'right' for the team's own goal, or None if unknown."""
        if self.team_a_attack_direction == ATTACK_UNKNOWN:
            return None
        team_a_goal = "left" if self.team_a_attack_direction == ATTACK_LEFT_TO_RIGHT else "right"
        if team == TEAM_A:
            return team_a_goal
        return "right" if team_a_goal == "left" else "left"

    def to_metric_points(self, players: list[PlayerPosition]) -> np.ndarray:
        """Points where both axes are expressed in frame-width units."""
        return np.array([[p.x, p.y * self.aspect_ratio] for p in players], dtype=np.float64)

@dataclass
class IndicatorResult:
    """Aggregated indicator over all valid frames."""

    name: str
    team: str
    unit: str
    values: list[float]
    threshold: float | None = None
    threshold_direction: str | None = None  # "above" or "below" means "problem when value is ..."
    description: str = ""
    mean_detection_confidence: float = 0.0

    @property
    def samples(self) -> int:
        return len(self.values)

    @property
    def value(self) -> float:
        return round(float(np.mean(self.values)), 4) if self.values else 0.0

    @property
    def exceed_ratio(self) -> float:
        """Fraction of frames where the value was on the problematic side of the threshold."""
        if not self.values or self.threshold is None or self.threshold_direction is None:
            return 0.0
        array = np.array(self.values)
        if self.threshold_direction == "above":
            exceeded = array >= self.threshold
        else:
            exceeded = array <= self.threshold
        return round(float(exceeded.mean()), 4)

    @property
    def full_name(self) -> str:
        return f"{self.team.lower()}.{self.name}"

    def to_dict(self) -> dict:
        array = np.array(self.values) if self.values else np.array([0.0])
        return {
            "name": self.full_name,
            "indicator": self.name,
            "team": self.team,
            "value": self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "threshold_direction": self.threshold_direction,
            "exceed_ratio": self.exceed_ratio,
            "min": round(float(array.min()), 4),
            "max": round(float(array.max()), 4),
            "std": round(float(array.std()), 4),
            "samples": self.samples,
            "description": self.description,
        }

def _mean_confidence(players: list[PlayerPosition]) -> float:
    return float(np.mean([p.confidence for p in players])) if players else 0.0

class IndicatorStrategy(ABC):
    """One family of tactical indicators."""

    teams = (TEAM_A, TEAM_B)

    @abstractmethod
    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        """Return indicators computed from the context."""

    @staticmethod
    def _new(context, name, team, unit, description, direction=None) -> IndicatorResult:
        return IndicatorResult(
            name=name,
            team=team,
            unit=unit,
            values=[],
            threshold=context.thresholds.get(name),
            threshold_direction=direction,
            description=description,
        )

    @staticmethod
    def _finalize(results: list[IndicatorResult], confidences: dict[str, list[float]]):
        for result in results:
            values = confidences.get(result.full_name, [])
            result.mean_detection_confidence = round(float(np.mean(values)), 4) if values else 0.0
        return [result for result in results if result.samples > 0]

class TeamShapeStrategy(IndicatorStrategy):
    """Team width (touchline to touchline) and team length (goal to goal)."""

    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        results, confidences = [], {}
        for team in self.teams:
            width = self._new(context, "team_width", team, "frame_height_ratio",
                                "Vertical spread of the team in the image (pitch width).", "below")
            length = self._new(context, "team_length", team, "frame_width_ratio",
                                "Horizontal spread of the team in the image (pitch length).")
            for snapshot in context.snapshots:
                players = snapshot.team_players(team)
                if len(players) < MIN_PLAYERS_PER_TEAM:
                    continue
                xs, ys = [p.x for p in players], [p.y for p in players]
                width.values.append(max(ys) - min(ys))
                length.values.append(max(xs) - min(xs))
                confidences.setdefault(width.full_name, []).append(_mean_confidence(players))
                confidences.setdefault(length.full_name, []).append(_mean_confidence(players))
            results += [width, length]
        return self._finalize(results, confidences)

class CompactnessStrategy(IndicatorStrategy):
    """Average distance between team-mates, convex-hull area and nearest team-mate distance."""

    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        results, confidences = [], {}
        for team in self.teams:
            average_distance = self._new(
                context, "average_player_distance", team, "frame_width_ratio",
                "Mean distance between every pair of team-mates.", "above")
            hull_area = self._new(
                context, "team_compactness_area", team, "frame_area_ratio",
                "Area of the convex hull that contains the team (smaller = more compact).", "above")
            nearest = self._new(
                context, "nearest_teammate_distance", team, "frame_width_ratio",
                "Mean distance from each player to the closest team-mate.", "above")

            for snapshot in context.snapshots:
                players = snapshot.team_players(team)
                if len(players) < MIN_PLAYERS_PER_TEAM:
                    continue
                points = context.to_metric_points(players)
                pair_distances = [np.linalg.norm(a - b) for a, b in combinations(points, 2)]
                average_distance.values.append(float(np.mean(pair_distances)))

                image_points = np.array([[p.x, p.y] for p in players], dtype=np.float32)
                hull = cv2.convexHull(image_points)
                hull_area.values.append(float(cv2.contourArea(hull)))

                distance_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
                np.fill_diagonal(distance_matrix, np.inf)
                nearest.values.append(float(distance_matrix.min(axis=1).mean()))

                confidence = _mean_confidence(players)
                for indicator in (average_distance, hull_area, nearest):
                    confidences.setdefault(indicator.full_name, []).append(confidence)
            results += [average_distance, hull_area, nearest]
        return self._finalize(results, confidences)


def _split_into_lines(sorted_xs: list[float]) -> list[list[float]]:
    """Split outfield x positions (already sorted from own goal outwards) into 3 lines
    by cutting at the two largest gaps."""
    gaps = np.diff(sorted_xs)
    cut_indices = sorted(np.argsort(gaps)[-2:])
    first_cut, second_cut = cut_indices[0] + 1, cut_indices[1] + 1
    return [sorted_xs[:first_cut], sorted_xs[first_cut:second_cut], sorted_xs[second_cut:]]

class LinesStrategy(IndicatorStrategy):
    """Defensive line position and distance between lines."""

    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        results, confidences = [], {}
        for team in self.teams:
            goal_side = context.own_goal_side(team)
            if goal_side:
                defensive_line = self._new(
                    context, "defensive_line_position", team, "frame_width_ratio",
                    "Distance from the team's own-goal image edge to its last line "
                    "(image relative; reliable only with a wide or static camera).", "below")
                defense_midfield = self._new(
                    context, "defensive_midfield_distance", team, "frame_width_ratio",
                    "Distance between the defensive line and the midfield line.", "above")
                midfield_attack = self._new(
                    context, "midfield_attack_distance", team, "frame_width_ratio",
                    "Distance between the midfield line and the attacking line.")
                tracked = [defensive_line, defense_midfield, midfield_attack]
            else:
                largest_gap = self._new(
                    context, "largest_line_gap", team, "frame_width_ratio",
                    "Largest horizontal gap between consecutive team lines "
                    "(attack direction unknown).", "above")
                tracked = [largest_gap]

            for snapshot in context.snapshots:
                players = snapshot.team_players(team)
                if len(players) < MIN_PLAYERS_FOR_LINES:
                    continue
                # Distance from the own goal side, so index 0 is always the deepest player.
                if goal_side == "right":
                    depth_values = sorted(1.0 - p.x for p in players)
                else:
                    depth_values = sorted(p.x for p in players)
                # Exclude a probable goalkeeper: an isolated deepest player.
                if len(depth_values) > MIN_PLAYERS_FOR_LINES and depth_values[1] - depth_values[0] > 0.08:
                    depth_values = depth_values[1:]
                lines = _split_into_lines(depth_values)
                line_means = [float(np.mean(line)) for line in lines]
                confidence = _mean_confidence(players)

                if goal_side:
                    defensive_line.values.append(line_means[0])
                    defense_midfield.values.append(line_means[1] - line_means[0])
                    midfield_attack.values.append(line_means[2] - line_means[1])
                else:
                    largest_gap.values.append(max(line_means[1] - line_means[0], line_means[2] - line_means[1]))
                for indicator in tracked:
                    confidences.setdefault(indicator.full_name, []).append(confidence)
            results += tracked
        return self._finalize(results, confidences)

class ZoneOccupationStrategy(IndicatorStrategy):
    """Player density, distribution by thirds and numerical balance by zone."""

    GRID_COLUMNS = 3
    GRID_ROWS = 3

    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        results, confidences = [], {}
        for team in self.teams:
            concentration = self._new(
                context, "max_zone_concentration", team, "player_ratio",
                "Share of the team's visible players inside the most crowded zone of a 3x3 grid.", "above")
            density = self._new(
                context, "player_density", team, "players_per_frame_area",
                "Visible players divided by the area of the team's bounding box.")
            thirds = [
                self._new(context, f"distribution_third_{index + 1}", team, "player_ratio",
                            f"Share of the team's visible players in image third {index + 1} (left to right).")
                for index in range(3)
            ]
            for snapshot in context.snapshots:
                players = snapshot.team_players(team)
                if len(players) < MIN_PLAYERS_PER_TEAM:
                    continue
                cells = [self._cell(p) for p in players]
                counts = np.bincount(cells, minlength=self.GRID_COLUMNS * self.GRID_ROWS)
                concentration.values.append(float(counts.max() / len(players)))

                xs, ys = [p.x for p in players], [p.y for p in players]
                box_area = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-3)
                density.values.append(float(len(players) / box_area))

                third_counts = np.bincount([min(int(p.x * 3), 2) for p in players], minlength=3)
                for index, third in enumerate(thirds):
                    third.values.append(float(third_counts[index] / len(players)))

                confidence = _mean_confidence(players)
                for indicator in [concentration, density, *thirds]:
                    confidences.setdefault(indicator.full_name, []).append(confidence)
            results += [concentration, density, *thirds]

        results += self._numerical_balance(context, confidences)
        return self._finalize(results, confidences)

    def _cell(self, player: PlayerPosition) -> int:
        column = min(int(player.x * self.GRID_COLUMNS), self.GRID_COLUMNS - 1)
        row = min(int(player.y * self.GRID_ROWS), self.GRID_ROWS - 1)
        return row * self.GRID_COLUMNS + column

    def _numerical_balance(self, context: AnalysisContext, confidences) -> list[IndicatorResult]:
        imbalance = self._new(
            context, "max_zone_numerical_difference", "MATCH", "players",
            "Largest difference in visible players between both teams inside one image third.", "above")
        for snapshot in context.snapshots:
            team_a = snapshot.team_players(TEAM_A)
            team_b = snapshot.team_players(TEAM_B)
            if len(team_a) < MIN_PLAYERS_PER_TEAM or len(team_b) < MIN_PLAYERS_PER_TEAM:
                continue
            a_counts = np.bincount([min(int(p.x * 3), 2) for p in team_a], minlength=3)
            b_counts = np.bincount([min(int(p.x * 3), 2) for p in team_b], minlength=3)
            imbalance.values.append(float(np.abs(a_counts - b_counts).max()))
            confidences.setdefault(imbalance.full_name, []).append(_mean_confidence(team_a + team_b))
        return [imbalance]

class BallStrategy(IndicatorStrategy):
    """Ball position and how often the ball was visible."""

    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        position_x = self._new(context, "ball_position_x", "MATCH", "frame_width_ratio",
                                "Mean horizontal ball position in the image (0 = left).")
        position_y = self._new(context, "ball_position_y", "MATCH", "frame_height_ratio",
                                "Mean vertical ball position in the image (0 = top).")
        visibility = self._new(context, "ball_detection_rate", "MATCH", "frame_ratio",
                                "Fraction of analysed frames where the ball was detected.")
        for snapshot in context.snapshots:
            visibility.values.append(1.0 if snapshot.ball else 0.0)
            if snapshot.ball:
                position_x.values.append(snapshot.ball[0])
                position_y.values.append(snapshot.ball[1])
        return [result for result in (position_x, position_y, visibility) if result.samples > 0]

def default_strategies() -> list[IndicatorStrategy]:
    return [TeamShapeStrategy(), CompactnessStrategy(), LinesStrategy(), ZoneOccupationStrategy(), BallStrategy()]

class TacticalAnalyzer:
    """Runs a configurable list of indicator strategies."""

    def __init__(self, strategies: list[IndicatorStrategy] | None = None) -> None:
        self._strategies = strategies if strategies is not None else default_strategies()

    def analyze(self, context: AnalysisContext) -> list[IndicatorResult]:
        indicators: list[IndicatorResult] = []
        for strategy in self._strategies:
            indicators.extend(strategy.compute(context))
        return indicators
