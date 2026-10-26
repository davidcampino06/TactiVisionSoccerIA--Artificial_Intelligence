"""Team colours come from tracks; players the tracker did not follow get the nearest team colour."""

import numpy as np

from detectors import OBJECT_TYPE_PLAYER, Detection
from pipeline import VideoAnalysisPipeline, _FrameRecord, untracked_id
from team_assignment import TEAM_A, TEAM_B, UNASSIGNED, JerseyColorTeamClassifier, TeamAssignmentResult
from tracking import TrackedDetection

RED = (40, 40, 200)     # BGR
BLUE = (200, 60, 30)
YELLOW = (30, 220, 230)


def painted(boxes: list[tuple[int, tuple[int, int, int]]]) -> tuple[np.ndarray, list[Detection]]:
    """A frame with one solid-coloured player box every 60 px, left to right."""
    frame = np.full((200, 60 * len(boxes) + 20, 3), (60, 140, 60), dtype=np.uint8)
    detections = []
    for slot, (_, color) in enumerate(boxes):
        x = 10 + slot * 60
        frame[20:180, x:x + 40] = color
        detections.append(Detection(x, 20, x + 40, 180, 0.9, 0, OBJECT_TYPE_PLAYER))
    return frame, detections


def test_untracked_players_get_the_nearest_team_colour():
    teams = [(1, RED), (2, RED), (3, RED), (4, RED), (5, BLUE), (6, BLUE), (7, BLUE), (8, BLUE)]
    frame, detections = painted(teams)
    classifier = JerseyColorTeamClassifier()
    classifier.collect(frame, [TrackedDetection(track_id, d) for (track_id, _), d in zip(teams, detections)])

    loose_frame, loose = painted([(0, BLUE), (0, RED), (0, YELLOW)])
    for index, detection in enumerate(loose):
        classifier.collect_untracked(loose_frame, (0, index), detection)

    result = classifier.assign()
    assert result.track_teams[1] == TEAM_A and result.track_teams[5] == TEAM_B  # red tracks are on the left
    assert result.detection_teams == {(0, 0): TEAM_B, (0, 1): TEAM_A, (0, 2): UNASSIGNED}


def test_untracked_players_are_part_of_the_tactical_snapshot():
    tracked = Detection(100, 100, 140, 200, 0.9, 0, OBJECT_TYPE_PLAYER)
    loose = Detection(300, 100, 340, 200, 0.6, 0, OBJECT_TYPE_PLAYER)
    record = _FrameRecord(5, 0.2, [TrackedDetection(7, tracked)], None, [loose])
    teams = TeamAssignmentResult({7: TEAM_A}, {}, 1.0, [], detection_teams={(0, 0): TEAM_B})

    players = VideoAnalysisPipeline._build_snapshots([record], teams, {"width": 400, "height": 200})[0].players

    assert [(p.track_id, p.team) for p in players] == [(7, TEAM_A), (untracked_id(0, 0), TEAM_B)]
    assert untracked_id(0, 0) < 0  # never mistaken for a real track
    assert record.player_count() == 2


# ---------------------------------------------------------------- the user's kit colour (optional)
import json
from pathlib import Path

import pytest

from team_assignment import OWN_TEAM_MAX_RATIO, hex_to_lab, match_own_team


def _two_team_classifier():
    teams = [(1, RED), (2, RED), (3, RED), (4, RED), (5, BLUE), (6, BLUE), (7, BLUE), (8, BLUE)]
    frame, detections = painted(teams)
    classifier = JerseyColorTeamClassifier()
    classifier.collect(frame, [TrackedDetection(track_id, d) for (track_id, _), d in zip(teams, detections)])
    return classifier


def test_without_a_kit_colour_the_teams_are_still_found_and_none_is_marked():
    result = _two_team_classifier().assign()
    assert set(result.team_colors_hex) == {TEAM_A, TEAM_B}
    assert result.own_team is None


def test_the_kit_colour_marks_which_team_is_mine():
    assert _two_team_classifier().assign("#c8281e").own_team == TEAM_A   # a red kit -> the red team
    assert _two_team_classifier().assign("#1e3cc8").own_team == TEAM_B   # a blue kit -> the blue team


def test_a_kit_colour_like_neither_team_is_not_forced():
    result = _two_team_classifier().assign("#00c800")                     # green: neither red nor blue
    assert result.own_team is None
    assert any("no se parece claramente" in warning for warning in result.warnings)


# Real kits against the shirt colours the AI measured on the 7 calibration clips (CALIBRATION.md).
MEASURED = json.loads((Path(__file__).parents[1] / "calibration" / "measurements.json").read_text(encoding="utf-8"))
REAL_KITS = [
    ("c1_francia_marruecos", "#1f2a44", TEAM_A), ("c1_francia_marruecos", "#c1272d", TEAM_B),
    ("c3_contraataque_banda", "#ffffff", TEAM_A), ("c3_contraataque_banda", "#dc052d", TEAM_B),
    ("c6_pase_filtrado", "#8e1f2f", TEAM_A), ("c6_pase_filtrado", "#ffffff", TEAM_B),
    ("c7_presion", "#0e1f4d", TEAM_A), ("c7_presion", "#f2d335", TEAM_B),
    # eFootball: the white team against BVB's yellow (CALIBRATION.md, stage 7).
    ("v1_ataque_y_perdida", "#ffffff", TEAM_A), ("v1_ataque_y_perdida", "#fde100", TEAM_B),
    ("v4_presion_tras_perdida", "#ffffff", TEAM_B), ("v4_presion_tras_perdida", "#fde100", TEAM_A),
]


@pytest.mark.parametrize(("clip", "kit", "team"), REAL_KITS)
def test_real_kits_find_their_team_in_broadcast_clips(clip, kit, team):
    centers = {name: hex_to_lab(color) for name, color in MEASURED[clip]["colors"].items()}
    found, ratio = match_own_team(centers, kit)
    assert found == team and ratio < OWN_TEAM_MAX_RATIO


# ---------------------------------------------------------------- shade and keepers (stage 7)
WHITE, WHITE_IN_SHADE = (235, 235, 235), (120, 120, 120)          # BGR
YELLOW, YELLOW_IN_SHADE = (40, 215, 240), (20, 110, 125)
ORANGE_KEEPER = (20, 120, 250)


def test_shade_does_not_split_the_teams():
    """A white team and a yellow one, half of each in the shade: the split is by shirt, not by light."""
    players = [(1, WHITE), (2, WHITE_IN_SHADE), (3, WHITE), (4, WHITE_IN_SHADE),
               (5, YELLOW), (6, YELLOW_IN_SHADE), (7, YELLOW), (8, YELLOW_IN_SHADE)]
    frame, detections = painted(players)
    classifier = JerseyColorTeamClassifier()
    for _ in range(3):
        classifier.collect(frame, [TrackedDetection(track_id, d) for (track_id, _), d in zip(players, detections)])
    teams = classifier.assign().track_teams
    assert teams[1] == teams[2] == teams[3] == teams[4] != teams[5]
    assert teams[5] == teams[6] == teams[7] == teams[8]


def test_a_lone_keeper_does_not_become_a_team():
    """White and dark kits plus one orange keeper seen in every frame: the keeper is set aside."""
    dark = (40, 40, 40)
    players = [(1, WHITE), (2, WHITE), (3, WHITE), (4, WHITE), (5, dark), (6, dark), (7, dark), (8, dark),
               (9, ORANGE_KEEPER)]
    frame, detections = painted(players)
    classifier = JerseyColorTeamClassifier()
    for _ in range(3):
        classifier.collect(frame, [TrackedDetection(track_id, d) for (track_id, _), d in zip(players, detections)])
    teams = classifier.assign().track_teams
    assert teams[1] == teams[4] != teams[5] == teams[8]
    assert teams[9] != teams[1]
