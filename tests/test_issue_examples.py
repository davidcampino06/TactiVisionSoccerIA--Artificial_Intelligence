"""Every possible issue comes with a plain explanation and one real frame that shows it."""

import pytest

from issue_examples import attach_examples, build_example, representative_frame
from tactical_analysis import (
    ATTACK_LEFT_TO_RIGHT,
    AnalysisContext,
    FrameSnapshot,
    PlayerPosition,
    TacticalAnalyzer,
    possession_by_frame,
)
from tactical_insights import TacticalInsightGenerator, plain_explanation
from team_assignment import TEAM_A, TEAM_B

COMPACT_B = [(0.70, 0.40), (0.75, 0.40), (0.70, 0.45), (0.75, 0.45)]


def frame(index, team_a, team_b=COMPACT_B, ball=None):
    players = [PlayerPosition(i + 1, TEAM_A, x, y, 0.9) for i, (x, y) in enumerate(team_a)]
    players += [PlayerPosition(100 + i, TEAM_B, x, y, 0.9) for i, (x, y) in enumerate(team_b)]
    return FrameSnapshot(index, index * 0.2, players, ball=ball)


def spread(gap):
    """Team A as a square of side ``gap``: nearest team-mate distance = gap."""
    return [(0.1, 0.1), (0.1 + gap, 0.1), (0.1, 0.1 + gap), (0.1 + gap, 0.1 + gap)]


def analyse(snapshots, direction="unknown"):
    context = AnalysisContext(snapshots, aspect_ratio=1.0, team_a_attack_direction=direction)
    indicators = TacticalAnalyzer().analyze(context)
    _, recommendations = TacticalInsightGenerator().generate(indicators, team_separation=1.0, seconds_per_frame=0.2)
    attach_examples(context, indicators, recommendations)
    return context, indicators, {r["issue_id"]: r for r in recommendations}


def test_the_example_is_a_representative_frame_not_the_most_extreme():
    # Team B has the ball (Team A defends) in every frame; Team A is spread 0.20, 0.30 or 0.40.
    gaps = [0.20, 0.30, 0.30, 0.40, 0.20, 0.30, 0.30, 0.40, 0.30, 0.30]
    snapshots = [frame(i, spread(gap), ball=(0.71, 0.40)) for i, gap in enumerate(gaps)]
    context, indicators, _ = analyse(snapshots)
    spacing = next(i for i in indicators if i.full_name == "team_a.nearest_teammate_distance")
    index, value = representative_frame(context, spacing, possession_by_frame(context))
    assert value == pytest.approx(0.30)          # the average is 0.30; 0.40 would be the extreme
    assert index == 1                            # first frame with that value


def test_spacing_example_draws_each_player_and_the_nearest_team_mate():
    snapshots = [frame(i, spread(0.30), ball=(0.71, 0.40)) for i in range(10)]
    _, _, issues = analyse(snapshots)
    issue = issues["EXCESSIVE_SPACING:TEAM_A:nearest_teammate_distance"]
    example = issue["explanation"]["example"]
    assert example["team"] == TEAM_A and example["owner"] == TEAM_B and example["ball"] == [0.71, 0.4]
    assert example["aspect_ratio"] == 1.0
    assert len(example["players"]) == 8
    pairs = example["overlay"]["pairs"]
    assert example["overlay"]["kind"] == "nearest" and len(pairs) == 4
    assert all(pair["too_far"] and pair["distance"] == pytest.approx(0.30) for pair in pairs)


def test_plain_explanation_says_it_like_a_coach():
    snapshots = [frame(i, spread(0.30), ball=(0.71, 0.40)) for i in range(10)]
    _, _, issues = analyse(snapshots)
    explanation = issues["EXCESSIVE_SPACING:TEAM_A:nearest_teammate_distance"]["explanation"]
    assert explanation["seen"] == (
        "Mientras el rival tenía el balón (unos 2 segundos de juego), en el Equipo A la distancia de cada "
        "jugador a su compañero más cercano fue en promedio el 30 % del ancho de la imagen; se considera un "
        "problema desde el 11,5 %, y lo alcanzó en el 100 % de esos momentos.")
    assert explanation["meaning"].startswith("Defendían con los jugadores muy separados")
    assert explanation["coach"].startswith("Juntar más las líneas")


def test_pressure_example_marks_the_ball_and_nobody_inside_the_radius():
    attackers = [(0.50, 0.50), (0.55, 0.30), (0.60, 0.70), (0.65, 0.50)]
    far_defenders = [(0.85, 0.10), (0.90, 0.30), (0.88, 0.70), (0.92, 0.90)]
    snapshots = [frame(i, attackers, far_defenders, ball=(0.51, 0.50)) for i in range(10)]
    _, _, issues = analyse(snapshots)
    issue = issues["LOW_BALL_PRESSURE:TEAM_B:pressure_on_ball"]
    overlay = issue["explanation"]["example"]["overlay"]
    assert overlay == {"kind": "pressure", "radius": 0.1, "pressing": []}
    assert "lo normal es al menos 1 jugador" in issue["explanation"]["seen"]


def test_a_split_team_example_lists_its_groups():
    two_blocks = [(0.10, 0.10), (0.12, 0.12), (0.60, 0.80), (0.62, 0.82)]
    snapshots = [frame(i, two_blocks, ball=(0.71, 0.40)) for i in range(10)]
    _, _, issues = analyse(snapshots)
    overlay = issues["DISCONNECTED_TEAM:TEAM_A:network_components"]["explanation"]["example"]["overlay"]
    assert overlay["kind"] == "network"
    assert sorted(overlay["groups"]) == [[1, 2], [3, 4]]
    assert sorted(overlay["links"]) == [[1, 2], [3, 4]]


def test_lines_example_marks_the_gap_between_defence_and_midfield():
    # Team A attacks left to right: defenders at x = 0.10, midfield at 0.45, attack at 0.60.
    stretched = [(0.10, 0.2), (0.10, 0.5), (0.10, 0.8), (0.45, 0.3), (0.45, 0.6), (0.60, 0.4), (0.60, 0.6)]
    snapshots = [frame(i, stretched, ball=(0.71, 0.40)) for i in range(10)]
    context = AnalysisContext(snapshots, aspect_ratio=1.0, team_a_attack_direction=ATTACK_LEFT_TO_RIGHT)
    indicator = next(i for i in TacticalAnalyzer().analyze(context) if i.full_name == "team_a.defensive_midfield_distance")
    overlay = build_example(context, indicator, possession_by_frame(context))["overlay"]
    assert overlay["lines"] == pytest.approx([0.10, 0.45, 0.60])
    assert overlay["gap"] == pytest.approx([0.10, 0.45])
    assert overlay["own_goal"] == "left"


def test_no_example_when_no_frame_crosses_the_threshold():
    snapshots = [frame(i, spread(0.05), ball=(0.71, 0.40)) for i in range(10)]
    context, indicators, _ = analyse(snapshots)
    spacing = next(i for i in indicators if i.full_name == "team_a.nearest_teammate_distance")
    assert build_example(context, spacing, possession_by_frame(context)) is None


def test_match_level_explanation_has_no_team():
    indicator = next(
        i for i in TacticalAnalyzer().analyze(AnalysisContext(
            [frame(i, [(0.1, 0.1 * k) for k in range(1, 7)], [(0.1, 0.5), (0.8, 0.2), (0.8, 0.5), (0.8, 0.8)])
             for i in range(10)], aspect_ratio=1.0))
        if i.full_name == "match.max_zone_numerical_difference")
    seen = plain_explanation(indicator, seconds_per_frame=0.2)["seen"]
    assert seen.startswith("En el partido (unos 2 segundos de juego), la diferencia de jugadores")
    assert "Equipo" not in seen
