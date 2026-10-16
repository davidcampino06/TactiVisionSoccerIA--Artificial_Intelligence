"""Phase of play: who has the ball (with memory) and indicators judged only in their phase."""

import pytest

from pipeline import phase_warnings
from tactical_analysis import (
    PHASE_DEFENDING,
    PHASE_NOT_DEFENDING,
    POSSESSION_MEMORY_SECONDS,
    AnalysisContext,
    FrameSnapshot,
    PlayerPosition,
    TacticalAnalyzer,
    possession_by_frame,
)
from tactical_insights import TacticalInsightGenerator
from team_assignment import TEAM_A, TEAM_B

# Compact block: team-mates 0.05 apart. Spread block: 0.30 apart.
COMPACT = [(0.40, 0.40), (0.45, 0.40), (0.40, 0.45), (0.45, 0.45)]
SPREAD = [(0.10, 0.10), (0.40, 0.10), (0.10, 0.40), (0.40, 0.40)]


def frame(index, team_a, team_b, ball_with=None):
    """One snapshot every 0.2 s; the ball sits on the first player of ``ball_with`` (or nowhere)."""
    players = [PlayerPosition(i, TEAM_A, x, y, 0.9) for i, (x, y) in enumerate(team_a)]
    players += [PlayerPosition(100 + i, TEAM_B, x, y, 0.9) for i, (x, y) in enumerate(team_b)]
    ball = None
    if ball_with is not None:
        holder = (team_a if ball_with == TEAM_A else team_b)[0]
        ball = (holder[0] + 0.01, holder[1])
    return FrameSnapshot(index, index * 0.2, players, ball=ball)


def shifted(block, dx):
    return [(x + dx, y) for x, y in block]


def test_possession_is_remembered_for_a_while_and_then_forgotten():
    snapshots = [frame(0, COMPACT, shifted(COMPACT, 0.4), ball_with=TEAM_A)]
    steps_in_memory = int(POSSESSION_MEMORY_SECONDS / 0.2)
    snapshots += [frame(i, COMPACT, shifted(COMPACT, 0.4)) for i in range(1, steps_in_memory + 3)]
    owners = possession_by_frame(AnalysisContext(snapshots, aspect_ratio=1.0))
    assert owners[0] == TEAM_A
    assert owners[steps_in_memory] == TEAM_A      # ball not seen, still within the memory
    assert owners[-1] is None                      # too long without seeing it


def test_possession_changes_when_the_rival_is_seen_with_the_ball():
    snapshots = [frame(0, COMPACT, shifted(COMPACT, 0.4), ball_with=TEAM_A),
                 frame(1, COMPACT, shifted(COMPACT, 0.4), ball_with=TEAM_B)]
    assert possession_by_frame(AnalysisContext(snapshots, aspect_ratio=1.0)) == [TEAM_A, TEAM_B]


def test_shape_is_measured_only_while_the_team_defends():
    # Team A is spread while it attacks (frames 0-9) and compact while it defends (10-19).
    attacking = [frame(i, SPREAD, shifted(COMPACT, 0.4), ball_with=TEAM_A) for i in range(10)]
    defending = [frame(i, COMPACT, shifted(COMPACT, 0.4), ball_with=TEAM_B) for i in range(10, 20)]
    context = AnalysisContext(attacking + defending, aspect_ratio=1.0)

    by_phase = {i.full_name: i for i in TacticalAnalyzer().analyze(context)}
    plain = {i.full_name: i for i in TacticalAnalyzer(by_phase=False).analyze(context)}

    spacing = by_phase["team_a.nearest_teammate_distance"]
    assert spacing.phase == PHASE_DEFENDING
    assert spacing.samples == 10
    assert spacing.value == pytest.approx(0.05)  # only the compact, defending frames
    assert plain["team_a.nearest_teammate_distance"].value > spacing.value
    assert by_phase["team_a.team_width"].phase == PHASE_NOT_DEFENDING
    assert by_phase["team_a.team_width"].value == pytest.approx(0.30)
    assert by_phase["team_a.team_length"].phase is None  # not phase dependent: every frame
    assert by_phase["team_a.team_length"].samples == 20


def test_a_team_that_never_defends_is_not_judged_on_its_defensive_shape():
    frames = [frame(i, SPREAD, shifted(COMPACT, 0.4), ball_with=TEAM_A) for i in range(10)]
    indicators = {i.full_name: i for i in TacticalAnalyzer().analyze(AnalysisContext(frames, aspect_ratio=1.0))}
    assert "team_a.nearest_teammate_distance" not in indicators  # not measured, never "fine"
    assert "team_b.nearest_teammate_distance" in indicators
    assert indicators["team_a.team_width"].samples == 10


def test_the_analysis_says_which_team_was_not_judged_while_defending():
    frames = [frame(i, SPREAD, shifted(COMPACT, 0.4), ball_with=TEAM_A) for i in range(10)]
    warnings = phase_warnings(AnalysisContext(frames, aspect_ratio=1.0), {TEAM_A: "#ffffff", TEAM_B: "#ff0000"})
    assert len(warnings) == 1 and warnings[0].startswith("El Equipo A defendió solo 0 de 10")
    assert phase_warnings(AnalysisContext(frames, aspect_ratio=1.0), {}) == []  # no teams, nothing to say


def test_the_evidence_says_the_shape_was_judged_while_defending():
    spread_defending = [frame(i, SPREAD, shifted(COMPACT, 0.4), ball_with=TEAM_B) for i in range(10)]
    indicators = TacticalAnalyzer().analyze(AnalysisContext(spread_defending, aspect_ratio=1.0))
    issues, _ = TacticalInsightGenerator().generate(indicators, team_separation=1.0)
    spacing = next(issue for issue in issues if issue["code"] == "EXCESSIVE_SPACING")
    assert spacing["team"] == TEAM_A
    assert spacing["evidence"].endswith("Medido solo en los cuadros en que el rival tenía el balón (el equipo defendía).")
