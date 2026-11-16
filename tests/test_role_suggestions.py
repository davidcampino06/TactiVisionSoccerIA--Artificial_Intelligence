"""Referee and goalkeeper suggestions (role_suggestions.py): grouped by kit, decided by who is together."""

from role_suggestions import (
    GOALKEEPER_OR_REFEREE,
    MIN_TOGETHER_FRAMES,
    REFEREE,
    is_grass,
    most_at_once,
    suggest_roles,
)
from team_assignment import TEAM_A, TEAM_B, UNASSIGNED, TeamAssignmentResult, hex_to_lab

ORANGE = hex_to_lab("#d16422")   # a goalkeeper of a real clip
CYAN = hex_to_lab("#3bb3c2")     # the referee of the same clip
WHITE = hex_to_lab("#e8e8e8")


def result(tracks):
    """tracks: {id: (team, colour, distance to the teams, samples)}"""
    return TeamAssignmentResult(
        track_teams={t: team for t, (team, _, _, _) in tracks.items()},
        team_colors_hex={}, separation_score=1.0, warnings=[],
        track_colors={t: color for t, (_, color, _, _) in tracks.items()},
        track_color_distance={t: distance for t, (_, _, distance, _) in tracks.items()},
        track_samples={t: samples for t, (_, _, _, samples) in tracks.items()},
    )


def walk(first, last, x, y=300.0, height=80.0):
    return {frame: (x + frame, y, height) for frame in range(first, last + 1, 3)}


def test_one_goalkeeper_cut_in_three_tracks_is_one_person():
    tracks = {t: (UNASSIGNED, ORANGE + t * 0.5, 6.0, 10) for t in (15, 31, 48)}
    positions = {15: walk(0, 21, 50), 31: walk(45, 99, 60), 48: walk(120, 159, 70)}
    suggestions = suggest_roles(result(tracks), positions)
    assert [s["track_id"] for s in suggestions] == [15, 31, 48]
    assert {s["suggestion"] for s in suggestions} == {GOALKEEPER_OR_REFEREE}
    assert {s["kit_group"] for s in suggestions} == {1} and suggestions[0]["kit_tracks"] == 3
    assert "arquero o el árbitro" in suggestions[0]["reason"]


def test_two_people_with_the_same_kit_at_once_are_the_refereeing_crew():
    tracks = {3: (UNASSIGNED, CYAN, 6.0, 10), 9: (UNASSIGNED, CYAN + 1, 6.0, 10)}
    positions = {3: walk(0, 60, 100), 9: walk(30, 90, 600)}  # the referee and an assistant, far apart
    suggestions = suggest_roles(result(tracks), positions)
    assert {s["suggestion"] for s in suggestions} == {REFEREE}
    assert suggestions[0]["confidence"] == 0.75 and "2 personas a la vez" in suggestions[0]["reason"]


def test_a_second_box_on_top_of_the_same_person_is_not_a_second_person():
    tracks = {74: (UNASSIGNED, ORANGE, 6.0, 10), 80: (UNASSIGNED, ORANGE, 6.0, 4)}
    positions = {74: walk(225, 318, 100), 80: {f: (x + 10, y, h) for f, (x, y, h) in walk(243, 252, 100).items()}}
    assert most_at_once([74, 80], positions) == 1
    assert {s["suggestion"] for s in suggest_roles(result(tracks), positions)} == {GOALKEEPER_OR_REFEREE}


def test_being_together_must_last_a_few_frames():
    positions = {1: walk(0, 60, 0), 2: {60: (900.0, 300.0, 80.0)}}  # together in one frame only
    assert MIN_TOGETHER_FRAMES > 1 and most_at_once([1, 2], positions) == 1
    positions[2] = walk(51, 60, 900)
    assert len(positions[2]) >= MIN_TOGETHER_FRAMES and most_at_once([1, 2], positions) == 2


def test_different_kits_are_different_groups_and_players_are_left_alone():
    tracks = {
        1: (TEAM_A, WHITE, 0.8, 10),          # a player
        2: (TEAM_B, CYAN, 0.9, 10),           # a player whose shirt looks like the referee's: has a team
        5: (UNASSIGNED, ORANGE, 6.0, 10),
        6: (UNASSIGNED, CYAN, 6.0, 10),
        7: (UNASSIGNED, CYAN, 1.2, 10),       # unassigned but close to a team: not an outlier
        8: (UNASSIGNED, ORANGE, 6.0, 2),      # too few samples
    }
    positions = {t: walk(0, 30, 100 * t) for t in tracks}
    suggestions = suggest_roles(result(tracks), positions)
    assert [(s["track_id"], s["kit_group"]) for s in suggestions] == [(5, 1), (6, 2)]
    assert suggestions[0]["kit_color"] != suggestions[1]["kit_color"]


def test_a_pitch_green_shirt_is_the_grass_not_a_kit():
    assert is_grass(hex_to_lab("#142506")) and is_grass(hex_to_lab("#3c8c3c"))
    assert not is_grass(ORANGE) and not is_grass(CYAN) and not is_grass(hex_to_lab("#23323f"))
    tracks = {4: (UNASSIGNED, hex_to_lab("#142506"), 6.0, 10)}
    assert suggest_roles(result(tracks), {4: walk(0, 30, 100)}) == []


def test_nothing_to_suggest_when_the_teams_could_not_be_found():
    empty = TeamAssignmentResult({1: UNASSIGNED}, {}, 0.0, [], track_colors={1: ORANGE}, track_samples={1: 9})
    assert suggest_roles(empty, {1: walk(0, 30, 0)}) == []


def test_the_pipeline_suggests_the_referees_of_a_video_and_the_response_accepts_them(tmp_path):
    """Six red and six blue players (each shirt a bit different) stand still; two people in yellow run."""
    import cv2
    import numpy as np

    from detectors import OBJECT_TYPE_PLAYER, Detection
    from pipeline import AnalysisOptions, VideoAnalysisPipeline
    from schemas import AnalysisResponse

    red, blue, yellow = (40, 40, 200), (200, 60, 30), (30, 220, 230)
    def shade(color, i):
        return tuple(int(c + (i % 3 - 1) * 12) for c in color)

    people = ([(40 + 90 * i, 30, shade(red, i)) for i in range(6)]
              + [(40 + 90 * i, 360, shade(blue, i)) for i in range(6)])
    path = tmp_path / "referees.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))
    boxes_per_frame = []
    for index in range(30):
        image = np.full((480, 640, 3), (40, 140, 40), dtype=np.uint8)
        moving = [(20 + 2 * index, 195, yellow), (580 - 2 * index, 195, yellow)]
        boxes = []
        for x, y, color in people + moving:
            cv2.rectangle(image, (x, y), (x + 36, y + 90), color, -1)
            boxes.append(Detection(x, y, x + 37, y + 91, 0.9, 0, OBJECT_TYPE_PLAYER))
        boxes_per_frame.append(boxes)
        writer.write(image)
    writer.release()

    class Detector:
        model_name, model_version, runtime, last_inference_ms = "fake", "0", "cpu", 0.0
        frame = 0

        def detect(self, image, confidence):
            found = boxes_per_frame[min(self.frame, len(boxes_per_frame) - 1)]
            self.frame += 1
            return found

    raw = VideoAnalysisPipeline(Detector()).analyze_video(
        path, AnalysisOptions(frame_stride=1, max_frames=30, pitch_filter=False))
    suggestions = raw["track_role_suggestions"]
    assert len(suggestions) == 2 and {s["suggestion"] for s in suggestions} == {REFEREE}
    teams = raw["teams"]["track_teams"]
    assert all(teams[str(s["track_id"])] == UNASSIGNED for s in suggestions)
    assert sum(team in (TEAM_A, TEAM_B) for team in teams.values()) == 12
    AnalysisResponse(**raw)  # the service returns it as it is
