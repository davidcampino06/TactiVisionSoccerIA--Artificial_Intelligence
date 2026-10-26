"""What the team did well: only clear, sustained strengths, explained in football words."""

import json
from pathlib import Path

import pytest

from calibration.calibrate import strength_scores
from issue_examples import attach_strength_examples
from simulation import run_simulation
from tactical_analysis import (
    ATTACK_LEFT_TO_RIGHT,
    AnalysisContext,
    FrameSnapshot,
    IndicatorResult,
    PlayerPosition,
    TacticalAnalyzer,
)
from tactical_insights import TacticalInsightGenerator
from tactical_strengths import STRENGTH_RULES, StrengthGenerator, good_share
from team_assignment import TEAM_A, TEAM_B

RULES = {rule.code: rule for rule in STRENGTH_RULES}


def indicator(name, values, team=TEAM_A, unit="players", phase="defending"):
    return IndicatorResult(name=name, team=team, unit=unit, values=values, mean_detection_confidence=0.8, phase=phase)


def test_a_strength_needs_most_moments_on_the_good_side():
    nobody_alone = indicator("isolated_players", [0.0] * 19 + [1.0])          # 95 % of the frames
    one_alone_often = indicator("isolated_players", [0.0] * 15 + [1.0] * 5)    # 75 %: not enough
    strengths = StrengthGenerator().generate([nobody_alone], team_separation=1.0)
    assert [s["code"] for s in strengths] == ["NOBODY_ISOLATED"]
    assert good_share(one_alone_often, RULES["NOBODY_ISOLATED"]) == 0.75
    assert StrengthGenerator().generate([one_alone_often], team_separation=1.0) == []


def test_too_few_frames_say_nothing():
    assert StrengthGenerator().generate([indicator("isolated_players", [0.0] * 5)], team_separation=1.0) == []


def test_never_a_strength_and_the_opposite_problem_for_the_same_team():
    nobody_alone = indicator("isolated_players", [0.0] * 20)
    issue = {"team": TEAM_A, "code": "DISCONNECTED_TEAM"}
    assert StrengthGenerator().generate([nobody_alone], 1.0, possible_issues=[issue]) == []
    rival_issue = {"team": TEAM_B, "code": "DISCONNECTED_TEAM"}
    assert len(StrengthGenerator().generate([nobody_alone], 1.0, possible_issues=[rival_issue])) == 1


def test_it_is_told_like_a_coach_would_with_the_real_numbers():
    lines = indicator("defensive_midfield_distance", [0.12] * 16 + [0.2] * 4, unit="frame_width_ratio")
    strength = StrengthGenerator().generate([lines], team_separation=1.0, seconds_per_frame=0.2)[0]
    assert strength["title"] == "Líneas juntas al defender"
    assert strength["explanation"]["seen"] == (
        "Mientras el rival tenía el balón (unos 4 segundos de juego), en el Equipo A el espacio entre la línea de "
        "defensas y la de mediocampistas fue menor al 15 % del ancho de la imagen en el 80\u00a0% de esos momentos "
        "(en promedio, el 13,6\u00a0%).")
    assert strength["explanation"]["meaning"].startswith("Defensa y mediocampo defendieron cerca")
    assert strength["explanation"]["keep"].startswith("Mantener esa distancia corta")
    assert 0 < strength["confidence"] <= 0.8


def test_the_example_frame_shows_the_good_side():
    # Team A (left to right) defends with its lines together; team B has the ball in every frame.
    tight = [(0.10, 0.2), (0.10, 0.5), (0.10, 0.8), (0.20, 0.3), (0.20, 0.6), (0.32, 0.4), (0.32, 0.6)]
    rival = [(0.70, 0.40), (0.75, 0.40), (0.70, 0.45), (0.75, 0.45)]
    snapshots = [FrameSnapshot(i, i * 0.2, [PlayerPosition(n + 1, TEAM_A, x, y, 0.9) for n, (x, y) in enumerate(tight)]
                               + [PlayerPosition(100 + n, TEAM_B, x, y, 0.9) for n, (x, y) in enumerate(rival)],
                               ball=(0.71, 0.40)) for i in range(12)]
    context = AnalysisContext(snapshots, aspect_ratio=1.0, team_a_attack_direction=ATTACK_LEFT_TO_RIGHT)
    indicators = TacticalAnalyzer().analyze(context)
    issues, _ = TacticalInsightGenerator().generate(indicators, team_separation=1.0)
    strengths = StrengthGenerator().generate(indicators, 1.0, issues, seconds_per_frame=0.2)
    attach_strength_examples(context, indicators, strengths)
    lines = next(s for s in strengths if s["code"] == "LINES_TOGETHER" and s["team"] == TEAM_A)
    example = lines["explanation"]["example"]
    assert example["overlay"]["kind"] == "lines"
    assert example["overlay"]["lines"] == pytest.approx([0.10, 0.20, 0.32])
    assert example["value"] <= RULES["LINES_TOGETHER"].limit


def test_simulation_reports_strengths_with_their_drawing():
    strengths = run_simulation()["strengths"]
    assert {s["code"] for s in strengths} >= {"LINES_TOGETHER", "NOBODY_ISOLATED"}
    assert all(s["explanation"]["example"] for s in strengths)


def test_no_strength_where_the_coach_saw_the_opposite_problem():
    """Calibration clips (CALIBRATION.md, stages 6 and 7): zero contradictions with the coach's labels,
    on the 7 broadcast clips and on the 6 eFootball clips."""
    here = Path(__file__).parents[1] / "calibration"
    measurements = json.loads((here / "measurements.json").read_text(encoding="utf-8"))
    labels = json.loads((here / "labels.json").read_text(encoding="utf-8"))
    scores = strength_scores(measurements, labels)
    assert all(row["contradictions"] == [] for row in scores.values())
    assert scores["NOBODY_ISOLATED"]["shown_no_problem"] == ["c8_tiro_esquina:TEAM_A"]
    # Strict on purpose: of the 13 strengths the coach listed in the eFootball clips none is shown
    # (the indicators do not back them up), and nothing is praised that the coach did not list
    # except the attacking depth of v1's white team, a question he was not asked.
    assert sum(row["found"] for row in scores.values()) == 0
    assert sum(row["shown_not_listed"] for row in scores.values()) == 1
