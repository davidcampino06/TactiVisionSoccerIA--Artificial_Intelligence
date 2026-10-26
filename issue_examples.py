"""One real frame that shows each possible tactical issue (or strength) on the pitch.

For every recommendation the analysis keeps the snapshot (players, teams, ball) of a frame where
the rule was really crossed, and a small overlay that explains it: the nearest team-mate of each
player, the groups of the proximity network, the gap between lines, the ball and the pressure
radius... The frontend draws it on a pitch.

Which frame: among the frames of the right phase of play where the threshold was crossed, the
one whose value is closest to the average reported in the evidence (representative, not the most
extreme). Positions stay image-relative (0..1): there is no pitch calibration, so the drawing is
"as seen by the camera", and the frontend says so.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import cv2
import numpy as np

from proximity_graph import proximity_graph
from tactical_analysis import (
    MIN_PLAYERS_FOR_LINES,
    PRESSURE_RADIUS,
    TEAM_A,
    TEAM_B,
    AnalysisContext,
    CompactnessStrategy,
    FrameSnapshot,
    IndicatorResult,
    IndicatorStrategy,
    LinesStrategy,
    PossessionPhaseStrategy,
    ProximityNetworkStrategy,
    TeamShapeStrategy,
    ZoneOccupationStrategy,
    _split_into_lines,
    in_phase,
    possession_by_frame,
)
from tactical_strengths import RULE_BY_CODE, is_good

STRATEGY_OF: dict[str, type[IndicatorStrategy]] = {
    "team_width": TeamShapeStrategy,
    "average_player_distance": CompactnessStrategy,
    "team_compactness_area": CompactnessStrategy,
    "nearest_teammate_distance": CompactnessStrategy,
    "defensive_line_position": LinesStrategy,
    "defensive_midfield_distance": LinesStrategy,
    "largest_line_gap": LinesStrategy,
    "max_zone_concentration": ZoneOccupationStrategy,
    "max_zone_numerical_difference": ZoneOccupationStrategy,
    "network_components": ProximityNetworkStrategy,
    "isolated_players": ProximityNetworkStrategy,
    "pressure_on_ball": PossessionPhaseStrategy,
    "attacking_width": PossessionPhaseStrategy,
    "players_ahead_of_ball": PossessionPhaseStrategy,
}


def _r(value: float) -> float:
    return round(float(value), 4)


def frame_value(context: AnalysisContext, snapshot: FrameSnapshot, indicator: IndicatorResult) -> float | None:
    """Value of one indicator in one frame (None when it cannot be measured there)."""
    strategy = STRATEGY_OF.get(indicator.name)
    if strategy is None:
        return None
    for result in strategy().compute(replace(context, snapshots=[snapshot])):
        if result.full_name == indicator.full_name and result.values:
            return float(result.values[0])
    return None


def _crosses(value: float, indicator: IndicatorResult) -> bool:
    if indicator.threshold is None:
        return False
    return value >= indicator.threshold if indicator.threshold_direction == "above" else value <= indicator.threshold


def representative_frame(context: AnalysisContext, indicator: IndicatorResult,
                         owners: list[str | None], shows: Callable[[float], bool] | None = None,
                         target: float | None = None) -> tuple[int, float] | None:
    """(snapshot index, value) of the crossing frame that best shows the issue.

    Only frames where most of the team is visible compete (at least 3/4 of the best one), so the
    drawing shows the whole picture; among them, the value closest to the reported average wins
    (representative, not the most extreme), then the earliest frame. ``shows`` and ``target`` pick
    other frames, e.g. the ones that show a strength (value on the good side) and their average."""
    shows = shows or (lambda value: _crosses(value, indicator))
    candidates = []
    for index, (snapshot, owner) in enumerate(zip(context.snapshots, owners)):
        if indicator.phase and not in_phase(indicator.phase, owner, indicator.team):
            continue
        value = frame_value(context, snapshot, indicator)
        if value is not None and shows(value):
            visible = len(snapshot.players) if indicator.team == "MATCH" else len(snapshot.team_players(indicator.team))
            candidates.append((abs(value - indicator.value), index, value, visible))
    if not candidates:
        return None
    if target is not None:
        candidates = [(abs(value - target), index, value, visible) for _, index, value, visible in candidates]
    most = max(visible for *_, visible in candidates)
    best = min(item for item in candidates if item[3] >= 0.75 * most)
    return best[1], best[2]


# ------------------------------------------------------------------------------ overlays
def _team(snapshot: FrameSnapshot, team: str):
    return snapshot.team_players(team)


def _nearest_pairs(context, snapshot, team, threshold):
    players = _team(snapshot, team)
    points = context.to_metric_points(players)
    pairs = []
    for index, player in enumerate(players):
        distances = np.linalg.norm(points - points[index], axis=1)
        distances[index] = np.inf
        nearest = int(distances.argmin())
        pairs.append({"from": player.track_id, "to": players[nearest].track_id, "distance": _r(distances[nearest]),
                      "too_far": bool(threshold is not None and distances[nearest] >= threshold)})
    return {"kind": "nearest", "pairs": pairs}


def _hull(context, snapshot, team, threshold):
    players = _team(snapshot, team)
    points = np.array([[p.x, p.y] for p in players], dtype=np.float32)
    hull = cv2.convexHull(points).reshape(-1, 2)
    return {"kind": "hull", "polygon": [[_r(x), _r(y)] for x, y in hull]}


def _network(context, snapshot, team, threshold):
    players = _team(snapshot, team)
    graph = proximity_graph(((p.track_id, p.x, p.y) for p in players), context.aspect_ratio)
    return {"kind": "network", "links": [[a, b] for a, b in graph.edges()],
            "groups": [sorted(group) for group in graph.connected_components()], "isolated": graph.isolated()}


def _lines(context, snapshot, team, threshold):
    players = _team(snapshot, team)
    if len(players) < MIN_PLAYERS_FOR_LINES:
        return None
    goal_side = context.own_goal_side(team)
    to_depth = (lambda x: 1.0 - x) if goal_side == "right" else (lambda x: x)
    depths = sorted(to_depth(p.x) for p in players)
    if len(depths) > MIN_PLAYERS_FOR_LINES and depths[1] - depths[0] > 0.08:
        depths = depths[1:]                         # probable goalkeeper, as in LinesStrategy
    means = [float(np.mean(line)) for line in _split_into_lines(depths)]
    gaps = [means[1] - means[0], means[2] - means[1]]
    widest = 0 if goal_side else int(np.argmax(gaps))
    return {"kind": "lines", "lines": [_r(to_depth(m)) for m in means],
            "gap": [_r(to_depth(means[widest])), _r(to_depth(means[widest + 1]))], "own_goal": goal_side}


def _zone(context, snapshot, team, threshold):
    players = _team(snapshot, team)
    cells = [min(int(p.y * 3), 2) * 3 + min(int(p.x * 3), 2) for p in players]
    counts = np.bincount(cells, minlength=9)
    cell = int(counts.argmax())
    return {"kind": "zone", "column": cell % 3, "row": cell // 3, "players": int(counts[cell]), "of": len(players)}


def _thirds(context, snapshot, team, threshold):
    counts = {t: np.bincount([min(int(p.x * 3), 2) for p in _team(snapshot, t)], minlength=3) for t in (TEAM_A, TEAM_B)}
    third = int(np.abs(counts[TEAM_A] - counts[TEAM_B]).argmax())
    return {"kind": "third", "third": third, "team_a": int(counts[TEAM_A][third]), "team_b": int(counts[TEAM_B][third])}


def _width(context, snapshot, team, threshold):
    ys = [p.y for p in _team(snapshot, team)]
    return {"kind": "width", "from": _r(min(ys)), "to": _r(max(ys))}


def _pressure(context, snapshot, team, threshold):
    if not snapshot.ball:
        return None
    ball = np.array([snapshot.ball[0], snapshot.ball[1] * context.aspect_ratio])
    defenders = _team(snapshot, team)
    near = [p.track_id for p, d in zip(defenders, np.linalg.norm(context.to_metric_points(defenders) - ball, axis=1))
            if d < PRESSURE_RADIUS]
    # The circle is drawn in image units: PRESSURE_RADIUS is a share of the image width.
    return {"kind": "pressure", "radius": PRESSURE_RADIUS, "pressing": near}


def _ahead(context, snapshot, team, threshold):
    goal_side = context.own_goal_side(team)
    if not snapshot.ball or not goal_side:
        return None
    towards = 1 if goal_side == "left" else -1
    ahead = [p.track_id for p in _team(snapshot, team) if (p.x - snapshot.ball[0]) * towards > 0]
    return {"kind": "ahead", "ball_x": _r(snapshot.ball[0]), "towards": "right" if towards == 1 else "left",
            "ahead": ahead}


OVERLAYS: dict[str, Callable] = {
    "nearest_teammate_distance": _nearest_pairs,
    "average_player_distance": _hull,
    "team_compactness_area": _hull,
    "network_components": _network,
    "isolated_players": _network,
    "defensive_midfield_distance": _lines,
    "largest_line_gap": _lines,
    "defensive_line_position": _lines,
    "max_zone_concentration": _zone,
    "max_zone_numerical_difference": _thirds,
    "team_width": _width,
    "attacking_width": _width,
    "pressure_on_ball": _pressure,
    "players_ahead_of_ball": _ahead,
}


def build_example(context: AnalysisContext, indicator: IndicatorResult, owners: list[str | None],
                  shows: Callable[[float], bool] | None = None, target: float | None = None) -> dict | None:
    """Snapshot + overlay of a representative frame for one indicator, ready to draw (or None)."""
    found = representative_frame(context, indicator, owners, shows, target)
    if found is None:
        return None
    index, value = found
    snapshot = context.snapshots[index]
    overlay = OVERLAYS[indicator.name](context, snapshot, indicator.team, indicator.threshold) \
        if indicator.name in OVERLAYS else None
    return {
        "frame_number": snapshot.frame_number,
        "timestamp_seconds": _r(snapshot.timestamp_seconds),
        "aspect_ratio": _r(context.aspect_ratio),   # image height / width, to draw distances right
        "team": indicator.team,
        "value": _r(value),
        "players": [{"id": p.track_id, "team": p.team, "x": _r(p.x), "y": _r(p.y)} for p in snapshot.players],
        "ball": [_r(snapshot.ball[0]), _r(snapshot.ball[1])] if snapshot.ball else None,
        "owner": owners[index],
        "overlay": overlay,
    }


def attach_examples(context: AnalysisContext, indicators: list[IndicatorResult], recommendations: list[dict]) -> None:
    """Adds ``explanation.example`` to every recommendation that has an explanation."""
    by_name = {indicator.full_name: indicator for indicator in indicators}
    owners = possession_by_frame(context)
    for recommendation in recommendations:
        explanation = recommendation.get("explanation")
        names = recommendation.get("related_indicators") or []
        indicator = by_name.get(names[0]) if names else None
        if explanation is None or indicator is None:
            continue
        explanation["example"] = build_example(context, indicator, owners)


def attach_strength_examples(context: AnalysisContext, indicators: list[IndicatorResult], strengths: list[dict]) -> None:
    """Adds ``explanation.example`` to every strength: a frame on the good side, close to their average."""
    by_name = {indicator.full_name: indicator for indicator in indicators}
    owners = possession_by_frame(context)
    for strength in strengths:
        rule = RULE_BY_CODE.get(strength["code"])
        names = strength.get("related_indicators") or []
        indicator = by_name.get(names[0]) if names else None
        if rule is None or indicator is None or strength.get("explanation") is None:
            continue
        good_values = [value for value in indicator.values if is_good(value, rule)]
        target = float(np.mean(good_values)) if good_values else None
        strength["explanation"]["example"] = build_example(
            context, indicator, owners, shows=lambda value, rule=rule: is_good(value, rule), target=target)
