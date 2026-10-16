"""One test per tactical indicator, with hand-made positions whose expected values are easy to check."""

import math

import pytest

from tactical_analysis import (
    ATTACK_LEFT_TO_RIGHT,
    ATTACK_RIGHT_TO_LEFT,
    ATTACK_UNKNOWN,
    AnalysisContext,
    FrameSnapshot,
    PlayerPosition,
    PossessionPhaseStrategy,
    TacticalAnalyzer,
    detect_attack_direction,
    resolve_attack_direction,
)
from tactical_insights import TacticalInsightGenerator
from team_assignment import TEAM_A, TEAM_B

SQUARE = [(0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3)]
FAR_SQUARE = [(0.7, 0.7), (0.9, 0.7), (0.7, 0.9), (0.9, 0.9)]
# Three lines at x = 0.10, 0.45 and 0.60.
STRETCHED = [(0.10, 0.2), (0.10, 0.5), (0.10, 0.8), (0.45, 0.3), (0.45, 0.6), (0.60, 0.4), (0.60, 0.6)]


def snapshot(team_a, team_b, ball=(0.5, 0.5), frame=0):
    players = [PlayerPosition(i, TEAM_A, x, y, 0.9) for i, (x, y) in enumerate(team_a)]
    players += [PlayerPosition(100 + i, TEAM_B, x, y, 0.8) for i, (x, y) in enumerate(team_b)]
    return FrameSnapshot(frame, frame / 25.0, players, ball=ball)


def analyze(snapshots, aspect_ratio=1.0, direction="unknown"):
    context = AnalysisContext(snapshots, aspect_ratio=aspect_ratio, team_a_attack_direction=direction)
    return {indicator.full_name: indicator for indicator in TacticalAnalyzer().analyze(context)}


# ---------------------------------------------------------------- shape and compactness

def test_average_player_distance_of_a_square():
    indicators = analyze([snapshot(SQUARE, FAR_SQUARE)])
    # 4 sides of 0.2 and 2 diagonals of 0.2 * sqrt(2).
    assert indicators["team_a.average_player_distance"].value == pytest.approx((4 * 0.2 + 2 * 0.2 * math.sqrt(2)) / 6, abs=1e-4)


def test_vertical_distances_use_the_aspect_ratio():
    # In a 16:9 image a vertical step of 0.4 of the height is 0.4 * 0.5 = 0.2 of the width.
    tall = [(0.1, 0.1), (0.3, 0.1), (0.1, 0.5), (0.3, 0.5)]
    indicators = analyze([snapshot(tall, FAR_SQUARE)], aspect_ratio=0.5)
    assert indicators["team_a.nearest_teammate_distance"].value == pytest.approx(0.2)
    assert indicators["team_a.team_width"].value == pytest.approx(0.4)  # width stays in height units


def test_teams_with_too_few_players_are_not_measured():
    indicators = analyze([snapshot(SQUARE[:3], FAR_SQUARE)])
    assert "team_a.team_width" not in indicators
    assert "team_b.team_width" in indicators


# ---------------------------------------------------------------- lines

def test_largest_line_gap_without_attack_direction():
    indicators = analyze([snapshot(STRETCHED, FAR_SQUARE)])
    assert indicators["team_a.largest_line_gap"].value == pytest.approx(0.35)
    assert "team_a.defensive_line_position" not in indicators


def test_midfield_attack_distance():
    indicators = analyze([snapshot(STRETCHED, FAR_SQUARE)], direction=ATTACK_LEFT_TO_RIGHT)
    assert indicators["team_a.midfield_attack_distance"].value == pytest.approx(0.15)


def test_lines_are_measured_from_the_own_goal_when_attacking_right_to_left():
    indicators = analyze([snapshot(STRETCHED, FAR_SQUARE)], direction=ATTACK_RIGHT_TO_LEFT)
    # Own goal on the right: the last line is the one at x = 0.60, i.e. 0.40 from the right edge.
    assert indicators["team_a.defensive_line_position"].value == pytest.approx(0.40)
    assert indicators["team_a.defensive_midfield_distance"].value == pytest.approx(0.15)


def test_an_isolated_goalkeeper_is_not_the_defensive_line():
    with_keeper = [(0.0, 0.5), *STRETCHED]
    indicators = analyze([snapshot(with_keeper, FAR_SQUARE)], direction=ATTACK_LEFT_TO_RIGHT)
    assert indicators["team_a.defensive_line_position"].value == pytest.approx(0.10)


# ---------------------------------------------------------------- zones

def test_zone_concentration_density_and_thirds():
    indicators = analyze([snapshot(SQUARE, FAR_SQUARE)])
    assert indicators["team_a.max_zone_concentration"].value == pytest.approx(1.0)  # all in one cell
    assert indicators["team_a.player_density"].value == pytest.approx(4 / (0.2 * 0.2))
    thirds = [indicators[f"team_a.distribution_third_{n}"].value for n in (1, 2, 3)]
    assert thirds == [1.0, 0.0, 0.0]
    assert [indicators[f"team_b.distribution_third_{n}"].value for n in (1, 2, 3)] == [0.0, 0.0, 1.0]


# ---------------------------------------------------------------- ball

def test_ball_position_and_detection_rate():
    snapshots = [snapshot(SQUARE, FAR_SQUARE, ball=(0.4, 0.6), frame=0),
                 snapshot(SQUARE, FAR_SQUARE, ball=(0.6, 0.4), frame=1),
                 snapshot(SQUARE, FAR_SQUARE, ball=None, frame=2)]
    indicators = analyze(snapshots)
    assert indicators["match.ball_position_x"].value == pytest.approx(0.5)
    assert indicators["match.ball_position_y"].value == pytest.approx(0.5)
    assert indicators["match.ball_detection_rate"].value == pytest.approx(2 / 3, abs=1e-4)


# ---------------------------------------------------------------- possession phases (new indicators)

# Team A has the ball at (0.5, 0.5): its nearest player is 0.01 away.
ATTACKERS = [(0.49, 0.5), (0.60, 0.3), (0.70, 0.7), (0.65, 0.5), (0.20, 0.5)]
PRESSERS = [(0.55, 0.5), (0.50, 0.58), (0.90, 0.1), (0.90, 0.9)]


def test_ball_owner_is_the_team_of_the_nearest_player():
    context = AnalysisContext([], aspect_ratio=1.0)
    assert PossessionPhaseStrategy.ball_owner(context, snapshot(ATTACKERS, PRESSERS)) == TEAM_A
    loose = snapshot([(0.2, 0.2), *SQUARE], FAR_SQUARE, ball=(0.5, 0.5))
    assert PossessionPhaseStrategy.ball_owner(context, loose) is None
    assert PossessionPhaseStrategy.ball_owner(context, snapshot(ATTACKERS, PRESSERS, ball=None)) is None


def test_pressure_on_ball_counts_defenders_near_the_ball():
    indicators = analyze([snapshot(ATTACKERS, PRESSERS)], direction=ATTACK_LEFT_TO_RIGHT)
    assert indicators["team_b.pressure_on_ball"].value == 2.0
    assert "team_a.pressure_on_ball" not in indicators  # team A had the ball in every frame


def test_attacking_width_is_measured_only_with_the_ball():
    indicators = analyze([snapshot(ATTACKERS, PRESSERS)])
    assert indicators["team_a.attacking_width"].value == pytest.approx(0.4)
    assert "team_b.attacking_width" not in indicators


def test_players_ahead_of_ball_follow_the_attack_direction():
    left_to_right = analyze([snapshot(ATTACKERS, PRESSERS)], direction=ATTACK_LEFT_TO_RIGHT)
    right_to_left = analyze([snapshot(ATTACKERS, PRESSERS)], direction=ATTACK_RIGHT_TO_LEFT)
    unknown = analyze([snapshot(ATTACKERS, PRESSERS)])
    assert left_to_right["team_a.players_ahead_of_ball"].value == 3.0
    assert right_to_left["team_a.players_ahead_of_ball"].value == 2.0
    assert "team_a.players_ahead_of_ball" not in unknown


def test_frames_with_a_loose_ball_are_skipped():
    loose = [snapshot([(0.2, 0.2), *SQUARE], FAR_SQUARE, ball=(0.5, 0.5), frame=n) for n in range(5)]
    indicators = analyze(loose, direction=ATTACK_LEFT_TO_RIGHT)
    assert not any(name.endswith(("pressure_on_ball", "attacking_width", "players_ahead_of_ball")) for name in indicators)


def test_possession_issues_are_raised_with_evidence():
    narrow_attack = [(0.49, 0.5), (0.30, 0.45), (0.30, 0.55), (0.20, 0.5)]
    passive_defense = [(0.90, 0.1), (0.90, 0.9), (0.80, 0.2), (0.85, 0.8)]
    snapshots = [snapshot(narrow_attack, passive_defense, frame=n) for n in range(12)]
    context = AnalysisContext(snapshots, aspect_ratio=1.0, team_a_attack_direction=ATTACK_LEFT_TO_RIGHT)
    issues, _ = TacticalInsightGenerator().generate(TacticalAnalyzer().analyze(context), team_separation=0.9)
    raised = {(issue["code"], issue["team"]) for issue in issues}
    assert {("LOW_BALL_PRESSURE", TEAM_B), ("NARROW_ATTACK", TEAM_A), ("NO_ATTACKING_DEPTH", TEAM_A)} <= raised
    narrow = next(issue for issue in issues if issue["code"] == "NARROW_ATTACK")
    assert "10 % del ancho visible (umbral 35 %)" in narrow["evidence"]


# ---------------------------------------------------------------- automatic attack direction

LEFT_BLOCK = [(0.10, 0.3), (0.20, 0.5), (0.25, 0.7), (0.30, 0.4)]
RIGHT_BLOCK = [(0.70, 0.3), (0.75, 0.5), (0.80, 0.7), (0.90, 0.4)]


def frames(team_a, team_b, count, start=0):
    return [snapshot(team_a, team_b, frame=start + n) for n in range(count)]


def test_team_on_the_left_attacks_left_to_right():
    estimate = detect_attack_direction(frames(LEFT_BLOCK, RIGHT_BLOCK, 10))
    assert (estimate.direction, estimate.agreement, estimate.frames) == (ATTACK_LEFT_TO_RIGHT, 1.0, 10)
    assert detect_attack_direction(frames(RIGHT_BLOCK, LEFT_BLOCK, 10)).direction == ATTACK_RIGHT_TO_LEFT


def test_a_clear_majority_is_enough():
    snapshots = frames(LEFT_BLOCK, RIGHT_BLOCK, 8) + frames(RIGHT_BLOCK, LEFT_BLOCK, 2, start=8)
    estimate = detect_attack_direction(snapshots)
    assert (estimate.direction, estimate.agreement) == (ATTACK_LEFT_TO_RIGHT, 0.8)


def test_no_direction_when_frames_disagree_or_are_too_few():
    mixed = frames(LEFT_BLOCK, RIGHT_BLOCK, 6) + frames(RIGHT_BLOCK, LEFT_BLOCK, 6, start=6)
    assert detect_attack_direction(mixed).direction == ATTACK_UNKNOWN
    assert detect_attack_direction(frames(LEFT_BLOCK, RIGHT_BLOCK, 5)).direction == ATTACK_UNKNOWN
    mingled = frames(LEFT_BLOCK, [(x + 0.01, y) for x, y in LEFT_BLOCK], 12)  # teams on top of each other
    assert detect_attack_direction(mingled).frames == 0


def test_the_users_direction_always_wins():
    direction, source, agreement, warning = resolve_attack_direction(
        ATTACK_RIGHT_TO_LEFT, frames(LEFT_BLOCK, RIGHT_BLOCK, 10))
    assert (direction, source, agreement, warning) == (ATTACK_RIGHT_TO_LEFT, "user", 1.0, None)


def test_detected_direction_is_explained_to_the_user():
    direction, source, _, warning = resolve_attack_direction(ATTACK_UNKNOWN, frames(LEFT_BLOCK, RIGHT_BLOCK, 10))
    assert (direction, source) == (ATTACK_LEFT_TO_RIGHT, "detected")
    assert "se detectó automáticamente que el Equipo A ataca de izquierda a derecha" in warning
    assert "100 % de 10 cuadros" in warning
    direction, source, _, warning = resolve_attack_direction(ATTACK_UNKNOWN, frames(LEFT_BLOCK, RIGHT_BLOCK, 3))
    assert (direction, source) == (ATTACK_UNKNOWN, "unknown")
    assert "no se pudo detectar" in warning
