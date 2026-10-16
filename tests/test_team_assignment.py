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
