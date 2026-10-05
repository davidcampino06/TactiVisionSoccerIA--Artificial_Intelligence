import pytest

from tactical_analysis import (
    ATTACK_LEFT_TO_RIGHT,
    AnalysisContext,
    FrameSnapshot,
    PlayerPosition,
    TacticalAnalyzer,
    _split_into_lines,
)
from tactical_insights import TacticalInsightGenerator
from team_assignment import TEAM_A, TEAM_B

def _snapshot(team_a_points, team_b_points, frame=0):
    players = [PlayerPosition(i, TEAM_A, x, y, 0.9) for i, (x, y) in enumerate(team_a_points)]
    players += [PlayerPosition(100 + i, TEAM_B, x, y, 0.9) for i, (x, y) in enumerate(team_b_points)]
    return FrameSnapshot(frame, frame / 25.0, players, ball=(0.5, 0.5))

def _by_name(indicators):
    return {indicator.full_name: indicator for indicator in indicators}

SQUARE = [(0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3)]
STRETCHED_A = [(0.10, 0.2), (0.10, 0.5), (0.10, 0.8), (0.45, 0.3), (0.45, 0.6), (0.60, 0.4), (0.60, 0.6)]

def test_team_width_and_length_are_spreads():
    context = AnalysisContext([_snapshot(SQUARE, SQUARE)], aspect_ratio=1.0)
    indicators = _by_name(TacticalAnalyzer().analyze(context))
    assert indicators["team_a.team_width"].value == pytest.approx(0.2)
    assert indicators["team_a.team_length"].value == pytest.approx(0.2)

def test_compactness_area_of_square():
    context = AnalysisContext([_snapshot(SQUARE, SQUARE)], aspect_ratio=1.0)
    indicators = _by_name(TacticalAnalyzer().analyze(context))
    assert indicators["team_a.team_compactness_area"].value == pytest.approx(0.04, abs=1e-4)
    assert indicators["team_a.nearest_teammate_distance"].value == pytest.approx(0.2)

def test_split_into_lines_uses_two_largest_gaps():
    lines = _split_into_lines([0.10, 0.11, 0.12, 0.40, 0.41, 0.70])
    assert lines == [[0.10, 0.11, 0.12], [0.40, 0.41], [0.70]]

def test_lines_indicators_with_known_direction():
    snapshots = [_snapshot(STRETCHED_A, SQUARE, frame) for frame in range(12)]
    context = AnalysisContext(snapshots, aspect_ratio=0.5625, team_a_attack_direction=ATTACK_LEFT_TO_RIGHT)
    indicators = _by_name(TacticalAnalyzer().analyze(context))
    assert indicators["team_a.defensive_line_position"].value == pytest.approx(0.10)
    assert indicators["team_a.defensive_midfield_distance"].value == pytest.approx(0.35)

def test_issue_generated_with_evidence_and_bounded_confidence():
    snapshots = [_snapshot(STRETCHED_A, SQUARE, frame) for frame in range(20)]
    context = AnalysisContext(snapshots, aspect_ratio=0.5625, team_a_attack_direction=ATTACK_LEFT_TO_RIGHT)
    indicators = TacticalAnalyzer().analyze(context)
    issues, recommendations = TacticalInsightGenerator().generate(indicators, team_separation=0.9)

    lines_issue = next(i for i in issues if i["code"] == "EXCESSIVE_DISTANCE_BETWEEN_LINES" and i["team"] == TEAM_A)
    assert lines_issue["issue"].startswith("Possible tactical issue")
    assert "0.35" in lines_issue["evidence"]
    assert 0.0 < lines_issue["confidence"] <= 1.0
    assert lines_issue["related_indicators"] == ["team_a.defensive_midfield_distance"]
    assert any(r["issue_id"] == lines_issue["id"] for r in recommendations)

def test_no_issue_with_too_few_samples():
    context = AnalysisContext([_snapshot(STRETCHED_A, SQUARE)], aspect_ratio=0.5625,
                                team_a_attack_direction=ATTACK_LEFT_TO_RIGHT)
    issues, _ = TacticalInsightGenerator().generate(TacticalAnalyzer().analyze(context), team_separation=1.0)
    assert issues == []

def test_numerical_imbalance_is_match_level():
    many_left = [(0.1, 0.1 * i) for i in range(1, 7)]
    few_left = [(0.1, 0.5), (0.8, 0.2), (0.8, 0.5), (0.8, 0.8)]
    context = AnalysisContext([_snapshot(many_left, few_left)], aspect_ratio=1.0)
    indicators = _by_name(TacticalAnalyzer().analyze(context))
    assert indicators["match.max_zone_numerical_difference"].value == 5.0
