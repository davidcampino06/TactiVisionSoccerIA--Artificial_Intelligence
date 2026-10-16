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

from proximity_graph import LINK_DISTANCE, proximity_graph
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
    "network_components": 2.0,             # at/above -> team split into groups with no support between them
    "isolated_players": 2.0,               # at/above -> several players with no team-mate close by
    "pressure_on_ball": 0.5,               # at/below -> nobody from the defending team near the ball
    "attacking_width": 0.35,               # at/below -> narrow attack (pitch width in the image)
    "players_ahead_of_ball": 0.5,          # at/below -> no attacker ahead of the ball (no depth)
}

# Automatic attack direction (detect_attack_direction).
MIN_DIRECTION_FRAMES = 8        # frames where both teams are visible and clearly apart
MIN_DIRECTION_AGREEMENT = 0.7   # share of those frames that must agree
MIN_CENTROID_GAP = 0.03         # teams closer than this (image width) do not vote

# Possession phases (PossessionPhaseStrategy), in frame-width units.
BALL_CONTROL_DISTANCE = 0.06  # the nearest player closer than this has the ball; otherwise it is loose
PRESSURE_RADIUS = 0.10        # defenders closer than this to the ball are pressing it

@dataclass(frozen=True)
class PlayerPosition:
    """A tracked player in one frame, in image-relative coordinates (0..1)."""
    track_id: int
    team: str
    x: float  # 0..1 of image width (foot point)
    y: float  # 0..1 of image height (foot point)
    confidence: float

@dataclass
class FrameSnapshot:
    """Everything the indicators need from one analysed frame."""
    frame_number: int
    timestamp_seconds: float
    players: list[PlayerPosition]
    ball: tuple[float, float] | None = None

    def team_players(self, team: str) -> list[PlayerPosition]:
        return [player for player in self.players if player.team == team]

@dataclass
class AnalysisContext:
    """Input shared by every indicator strategy: the snapshots and the analysis settings."""
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
        """Serialises the indicator for the JSON response, with its distribution."""
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
    def _new(context: AnalysisContext, name: str, team: str, unit: str, description: str,
             direction: str | None = None) -> IndicatorResult:
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
    def _finalize(results: list[IndicatorResult], confidences: dict[str, list[float]]) -> list[IndicatorResult]:
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

    def _numerical_balance(self, context: AnalysisContext,
                           confidences: dict[str, list[float]]) -> list[IndicatorResult]:
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

class ProximityNetworkStrategy(IndicatorStrategy):
    """Graph of supports between team-mates (proximity_graph.py), explored with BFS each frame."""

    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        results, confidences = [], {}
        for team in self.teams:
            components = self._new(
                context, "network_components", team, "groups",
                "Groups of team-mates not linked to each other (link = closer than "
                f"{LINK_DISTANCE} of the image width). 1 = the whole team is connected.", "above")
            isolated = self._new(
                context, "isolated_players", team, "players",
                "Players with no team-mate inside the link distance.", "above")
            links = self._new(
                context, "network_average_links", team, "links",
                "Mean number of close team-mates per player (average degree of the graph).")
            diameter = self._new(
                context, "network_diameter", team, "hops",
                "Longest shortest route, in short passes, between two linked team-mates (BFS).")
            for snapshot in context.snapshots:
                players = snapshot.team_players(team)
                if len(players) < MIN_PLAYERS_PER_TEAM:
                    continue
                graph = proximity_graph(((p.track_id, p.x, p.y) for p in players), context.aspect_ratio)
                components.values.append(float(len(graph.connected_components())))
                isolated.values.append(float(len(graph.isolated())))
                links.values.append(2 * graph.edge_count / len(players))
                diameter.values.append(float(graph.diameter()))
                confidence = _mean_confidence(players)
                for indicator in (components, isolated, links, diameter):
                    confidences.setdefault(indicator.full_name, []).append(confidence)
            results += [components, isolated, links, diameter]
        return self._finalize(results, confidences)


class PossessionPhaseStrategy(IndicatorStrategy):
    """Indicators that depend on which team has the ball.

    The ball is controlled by the team of the player nearest to it, when that player is closer
    than BALL_CONTROL_DISTANCE; frames with a loose or invisible ball are skipped. Then:

        pressure_on_ball       defending team: players within PRESSURE_RADIUS of the ball
        attacking_width        attacking team: vertical spread in the image (pitch width)
        players_ahead_of_ball  attacking team: players beyond the ball towards the rival goal
                               (needs the attack direction, given or detected)
    """

    def compute(self, context: AnalysisContext) -> list[IndicatorResult]:
        confidences: dict[str, list[float]] = {}
        pressure = {team: self._new(
            context, "pressure_on_ball", team, "players",
            f"Players of the team without the ball closer than {PRESSURE_RADIUS} of the image width "
            "to the ball.", "below") for team in self.teams}
        width = {team: self._new(
            context, "attacking_width", team, "frame_height_ratio",
            "Vertical spread of the team in the image while it has the ball (attacking width).",
            "below") for team in self.teams}
        ahead = {team: self._new(
            context, "players_ahead_of_ball", team, "players",
            "Players of the team with the ball placed beyond it towards the rival goal (depth).",
            "below") for team in self.teams}

        for snapshot in context.snapshots:
            if not snapshot.ball:
                continue
            squads = {team: snapshot.team_players(team) for team in self.teams}
            if any(len(players) < MIN_PLAYERS_PER_TEAM for players in squads.values()):
                continue
            owner = self.ball_owner(context, snapshot)
            if owner is None:
                continue
            defender = TEAM_B if owner == TEAM_A else TEAM_A
            ball = np.array([snapshot.ball[0], snapshot.ball[1] * context.aspect_ratio])

            defenders = context.to_metric_points(squads[defender])
            pressure[defender].values.append(float((np.linalg.norm(defenders - ball, axis=1) < PRESSURE_RADIUS).sum()))
            confidences.setdefault(pressure[defender].full_name, []).append(_mean_confidence(squads[defender]))

            attackers = squads[owner]
            ys = [player.y for player in attackers]
            width[owner].values.append(max(ys) - min(ys))
            confidences.setdefault(width[owner].full_name, []).append(_mean_confidence(attackers))

            goal_side = context.own_goal_side(owner)
            if goal_side:
                towards_rival = 1 if goal_side == "left" else -1
                beyond = sum(1 for player in attackers if (player.x - snapshot.ball[0]) * towards_rival > 0)
                ahead[owner].values.append(float(beyond))
                confidences.setdefault(ahead[owner].full_name, []).append(_mean_confidence(attackers))

        results = [indicator for team in self.teams for indicator in (pressure[team], width[team], ahead[team])]
        return self._finalize(results, confidences)

    @staticmethod
    def ball_owner(context: AnalysisContext, snapshot: FrameSnapshot) -> str | None:
        """Team of the player nearest to the ball, or None when the ball is loose."""
        if not snapshot.ball or not snapshot.players:
            return None
        ball = np.array([snapshot.ball[0], snapshot.ball[1] * context.aspect_ratio])
        points = context.to_metric_points(snapshot.players)
        distances = np.linalg.norm(points - ball, axis=1)
        nearest = int(distances.argmin())
        return snapshot.players[nearest].team if distances[nearest] < BALL_CONTROL_DISTANCE else None


def network_snapshot(context: AnalysisContext) -> dict | None:
    """The proximity network of the frame with most visible players, ready to draw.

    For each team: nodes (track, position, degree, group), links, groups, isolated players,
    diameter and the BFS hops between every pair of linked players."""
    candidates = [snapshot for snapshot in context.snapshots if snapshot.players]
    if not candidates:
        return None
    best = max(candidates, key=lambda snapshot: len(snapshot.players))
    teams = {}
    for team in (TEAM_A, TEAM_B):
        players = best.team_players(team)
        if len(players) < 2:
            continue
        graph = proximity_graph(((p.track_id, p.x, p.y) for p in players), context.aspect_ratio)
        groups = graph.connected_components()
        group_of = {node: index for index, group in enumerate(groups) for node in group}
        teams[team] = {
            "nodes": [{"track_id": p.track_id, "x": round(p.x, 4), "y": round(p.y, 4),
                       "degree": graph.degree(p.track_id), "group": group_of[p.track_id]} for p in players],
            "links": [[a, b] for a, b in graph.edges()],
            "groups": len(groups),
            "isolated": graph.isolated(),
            "diameter": graph.diameter(),
            "hops": {str(source): {str(target): hops for target, hops in reached.items()}
                     for source, reached in graph.all_hops().items()},
        }
    return {"frame_number": best.frame_number, "timestamp_seconds": round(best.timestamp_seconds, 3),
            "link_distance": LINK_DISTANCE, "teams": teams}


@dataclass(frozen=True)
class DirectionEstimate:
    """Result of the automatic attack direction vote for TEAM_A."""
    direction: str      # ATTACK_LEFT_TO_RIGHT, ATTACK_RIGHT_TO_LEFT or ATTACK_UNKNOWN (for TEAM_A)
    agreement: float    # share of the voting frames that agree with it (0..1)
    frames: int         # frames that voted


def detect_attack_direction(snapshots: list[FrameSnapshot]) -> DirectionEstimate:
    """Guess where TEAM_A attacks when the user did not say it.

    Each team keeps its shape between the ball and its own goal, so on average the team that
    defends the left goal stands further left than its rival. Every frame where both teams are
    visible and their mean x positions are clearly apart votes for one side; the direction is
    accepted only with enough votes and a clear majority, otherwise it stays unknown. O(frames * players).
    """
    votes: list[bool] = []
    for snapshot in snapshots:
        team_a, team_b = snapshot.team_players(TEAM_A), snapshot.team_players(TEAM_B)
        if len(team_a) < MIN_PLAYERS_PER_TEAM or len(team_b) < MIN_PLAYERS_PER_TEAM:
            continue
        gap = float(np.mean([p.x for p in team_b]) - np.mean([p.x for p in team_a]))
        if abs(gap) >= MIN_CENTROID_GAP:
            votes.append(gap > 0)  # True: TEAM_A is on the left, so it defends the left goal
    if len(votes) < MIN_DIRECTION_FRAMES:
        return DirectionEstimate(ATTACK_UNKNOWN, 0.0, len(votes))
    left_share = sum(votes) / len(votes)
    if left_share >= MIN_DIRECTION_AGREEMENT:
        return DirectionEstimate(ATTACK_LEFT_TO_RIGHT, round(left_share, 4), len(votes))
    if 1 - left_share >= MIN_DIRECTION_AGREEMENT:
        return DirectionEstimate(ATTACK_RIGHT_TO_LEFT, round(1 - left_share, 4), len(votes))
    return DirectionEstimate(ATTACK_UNKNOWN, round(max(left_share, 1 - left_share), 4), len(votes))


DIRECTION_TEXT = {ATTACK_LEFT_TO_RIGHT: "de izquierda a derecha", ATTACK_RIGHT_TO_LEFT: "de derecha a izquierda"}


def resolve_attack_direction(requested: str, snapshots: list[FrameSnapshot]) -> tuple[str, str, float, str | None]:
    """Direction to use: the user's, or a detected one. Returns (direction, source, agreement, warning).

    source is "user", "detected" or "unknown"; the warning (in Spanish, shown to the user) explains
    a detected or missing direction."""
    if requested != ATTACK_UNKNOWN:
        return requested, "user", 1.0, None
    estimate = detect_attack_direction(snapshots)
    if estimate.direction != ATTACK_UNKNOWN:
        percent = f"{estimate.agreement * 100:.0f}"
        return estimate.direction, "detected", estimate.agreement, (
            f"No se indicó la dirección de ataque: se detectó automáticamente que el Equipo A ataca "
            f"{DIRECTION_TEXT[estimate.direction]} (lo indica el {percent} % de {estimate.frames} cuadros). "
            "Si no es así, indícala al analizar para corregir los indicadores de línea defensiva."
        )
    return ATTACK_UNKNOWN, "unknown", estimate.agreement, (
        "No se indicó la dirección de ataque y no se pudo detectar con seguridad: los indicadores de "
        "línea defensiva se reemplazaron por el mayor espacio entre líneas."
    )


def default_strategies() -> list[IndicatorStrategy]:
    """The indicator families computed by default, one Strategy each."""
    return [TeamShapeStrategy(), CompactnessStrategy(), LinesStrategy(), ZoneOccupationStrategy(), BallStrategy(),
            ProximityNetworkStrategy(), PossessionPhaseStrategy()]

class TacticalAnalyzer:
    """Runs a configurable list of indicator strategies."""

    def __init__(self, strategies: list[IndicatorStrategy] | None = None) -> None:
        self._strategies = strategies if strategies is not None else default_strategies()

    def analyze(self, context: AnalysisContext) -> list[IndicatorResult]:
        """Runs every strategy on the context and joins their indicators."""
        indicators: list[IndicatorResult] = []
        for strategy in self._strategies:
            indicators.extend(strategy.compute(context))
        return indicators
