"""Cut tracks (track_stitching.py): the same player with two ids is suggested as one, and only then."""

import numpy as np

from detectors import OBJECT_TYPE_PLAYER, Detection
from team_assignment import TEAM_A, TEAM_B, UNASSIGNED
from track_stitching import (
    MAX_GAP_SECONDS,
    link_groups,
    score_link,
    stitch_tracks,
    stitching_payload,
    summarize_tracks,
)
from tracking import TrackedDetection

FPS = 25.0
RED = np.array([90.0, 170.0, 160.0])   # Lab, OpenCV 8-bit scale
BLUE = np.array([60.0, 140.0, 90.0])


def box(foot_x, foot_y, height=80.0):
    width = height * 0.4
    return Detection(foot_x - width / 2, foot_y - height, foot_x + width / 2, foot_y, 0.8, 0, OBJECT_TYPE_PLAYER)


def run(track_id, start_frame, end_frame, start, velocity, height=80.0):
    """A player running in a straight line at ``velocity`` px/s, seen from start_frame to end_frame."""
    out = []
    for frame in range(start_frame, end_frame + 1):
        seconds = (frame - start_frame) / FPS
        x, y = start[0] + velocity[0] * seconds, start[1] + velocity[1] * seconds
        out.append((frame, TrackedDetection(track_id, box(x, y, height))))
    return out


def frames(*runs):
    """(frame number, seconds, tracked players) for every frame, as the pipeline has them."""
    by_frame = {}
    for one in runs:
        for frame, tracked in one:
            by_frame.setdefault(frame, []).append(tracked)
    return [(frame, frame / FPS, by_frame[frame]) for frame in sorted(by_frame)]


def where(frame, start_frame, start, velocity):
    seconds = (frame - start_frame) / FPS
    return start[0] + velocity[0] * seconds, start[1] + velocity[1] * seconds


def test_a_player_hidden_for_half_a_second_is_the_same_player():
    # ID 12 runs right at 120 px/s, disappears at frame 50 behind a team-mate, ID 31 appears at frame 62
    # exactly where he would be.
    velocity = (120.0, 10.0)
    first = run(12, 0, 50, (100, 300), velocity)
    second = run(31, 62, 100, where(62, 0, (100, 300), velocity), velocity)
    summaries = summarize_tracks(frames(first, second))
    (link,) = stitch_tracks(summaries, {12: TEAM_A, 31: TEAM_A}, {12: RED, 31: RED + 2})
    assert (link.from_track, link.to_track, link.same_team) == (12, 31, True)
    assert abs(link.gap_seconds - 12 / FPS) < 1e-9 and link.distance_heights < 0.1 and link.confidence > 0.8


def test_the_speed_is_measured_at_the_end_of_the_track():
    # He walked left and then sprinted right: the prediction must use the sprint, not the average.
    walk = run(4, 0, 30, (400, 300), (-40.0, 0.0))
    turn = where(30, 0, (400, 300), (-40.0, 0.0))
    sprint = [(f + 31, t) for f, t in run(4, 0, 20, turn, (200.0, 0.0))]
    summary = summarize_tracks(frames(walk, sprint))[4]
    assert abs(summary.end_velocity[0] - 200.0) < 1.0 and abs(summary.start_velocity[0] + 40.0) < 1.0


def test_two_tracks_on_screen_at_the_same_time_are_two_players():
    a = run(1, 0, 40, (100, 300), (100.0, 0.0))
    b = run(2, 30, 70, where(30, 0, (100, 300), (100.0, 0.0)), (100.0, 0.0))  # starts before A ends
    assert stitch_tracks(summarize_tracks(frames(a, b))) == []


def test_too_long_a_gap_or_too_far_away_is_not_a_link():
    a = run(1, 0, 25, (100, 300), (100.0, 0.0))
    late_frame = 26 + int((MAX_GAP_SECONDS + 0.2) * FPS)
    late = run(2, late_frame, late_frame + 20, where(late_frame, 0, (100, 300), (100.0, 0.0)), (100.0, 0.0))
    far = run(3, 30, 50, (900, 300), (100.0, 0.0))  # 10 box heights away after 0.2 s
    assert stitch_tracks(summarize_tracks(frames(a, late))) == []
    assert stitch_tracks(summarize_tracks(frames(a, far))) == []


def test_other_team_other_shirt_or_other_size_rule_it_out():
    a = run(1, 0, 25, (100, 300), (100.0, 0.0))
    b = run(2, 32, 60, where(32, 0, (100, 300), (100.0, 0.0)), (100.0, 0.0))
    summaries = summarize_tracks(frames(a, b))
    assert score_link(summaries[1], summaries[2], TEAM_A, TEAM_A) is not None
    assert score_link(summaries[1], summaries[2], TEAM_A, TEAM_B) is None
    assert score_link(summaries[1], summaries[2], color_a=RED, color_b=BLUE) is None
    # A team that is not known (referee, keeper, too few frames) does not rule anything out.
    assert score_link(summaries[1], summaries[2], TEAM_A, UNASSIGNED).same_team is None
    small = run(3, 32, 60, where(32, 0, (100, 300), (100.0, 0.0)), (100.0, 0.0), height=40.0)
    assert score_link(summaries[1], summarize_tracks(frames(small))[3]) is None


def test_of_two_candidates_the_one_that_fits_is_chosen_and_each_track_is_used_once():
    velocity = (150.0, 0.0)
    a = run(10, 0, 40, (100, 300), velocity)
    right_place = run(20, 48, 80, where(48, 0, (100, 300), velocity), velocity)
    near = run(21, 50, 80, np.add(where(50, 0, (100, 300), velocity), (0, 60)), velocity)  # 0.75 heights off
    links = stitch_tracks(summarize_tracks(frames(a, right_place, near)), {}, {10: RED, 20: RED, 21: RED})
    assert [(link.from_track, link.to_track) for link in links] == [(10, 20)]


def test_chains_become_one_group_in_time_order():
    velocity = (100.0, 0.0)
    a = run(12, 0, 30, (100, 300), velocity)
    b = run(31, 40, 70, where(40, 0, (100, 300), velocity), velocity)
    c = run(40, 80, 110, where(80, 0, (100, 300), velocity), velocity)
    other = run(5, 0, 110, (100, 600), (0.0, 0.0))
    payload = stitching_payload(frames(a, b, c, other))
    assert payload["groups"] == [[12, 31, 40]]
    first = payload["links"][0] if payload["links"][0]["from_track"] == 12 else payload["links"][1]
    assert (first["from_last_frame"], first["to_first_frame"]) == (30, 40)
    assert set(first) >= {"gap_seconds", "distance_heights", "color_distance", "same_team", "confidence"}


def test_groups_from_links_without_tracks_on_both_ends():
    class Link:
        def __init__(self, a, b):
            self.from_track, self.to_track = a, b

    assert link_groups([Link(31, 40), Link(12, 31), Link(5, 9)]) == [[5, 9], [12, 31, 40]]
    assert link_groups([]) == []


def test_a_wild_speed_from_a_jumpy_box_is_capped_before_predicting():
    """A box that jumped in its last frames gives 5000 px/s: no player runs that fast (cap 4.5 heights/s)."""
    from track_stitching import MAX_SPEED_HEIGHTS, TrackSummary

    end = np.array([500.0, 300.0])
    jumpy = TrackSummary(1, 0, 50, 0.0, 2.0, np.array([300.0, 300.0]), end, np.zeros(2),
                         np.array([5000.0, 0.0]), 80.0, 20)
    back = TrackSummary(2, 55, 90, 2.2, 3.6, end + (10.0, 0.0), end + (60.0, 0.0), np.zeros(2), np.zeros(2), 80.0, 15)
    link = score_link(jumpy, back)
    assert link is not None
    # Forward 0.1 s at the capped speed (4.5 x 80 px/s = 36 px), not 500 px.
    assert abs(link.distance_heights - (MAX_SPEED_HEIGHTS * 80.0 * 0.1 - 10.0) / 80.0) < 1e-9


def test_a_track_seen_once_is_never_linked():
    a = run(1, 0, 0, (100, 300), (0.0, 0.0))
    b = run(2, 5, 30, (102, 300), (0.0, 0.0))
    assert stitch_tracks(summarize_tracks(frames(a, b))) == []


def many_tracks(count, seed=1):
    """Short random tracks over ten minutes, as a long video analysed frame by frame gives."""
    from track_stitching import TrackSummary

    rng = np.random.default_rng(seed)
    summaries = {}
    for track_id in range(count):
        start, length = rng.uniform(0, 600), rng.uniform(0.2, 5)
        point = rng.uniform(0, 1900, 2)
        summaries[track_id] = TrackSummary(
            track_id, int(start * FPS), int((start + length) * FPS), start, start + length, point,
            point + rng.normal(0, 50, 2), rng.normal(0, 100, 2), rng.normal(0, 100, 2), 80.0, 10)
    return summaries


def test_only_tracks_starting_right_after_are_compared_and_the_result_is_the_same():
    import time

    summaries = many_tracks(1500)
    every_pair = sorted(
        (link for a in summaries.values() for b in summaries.values() if a is not b
         for link in [score_link(a, b)] if link is not None and link.confidence >= 0.5),
        key=lambda link: (-link.confidence, link.gap_seconds, link.from_track, link.to_track))
    used_from, used_to, expected = set(), set(), []
    for link in every_pair:
        if link.from_track not in used_from and link.to_track not in used_to:
            used_from.add(link.from_track)
            used_to.add(link.to_track)
            expected.append((link.from_track, link.to_track))
    assert [(link.from_track, link.to_track) for link in stitch_tracks(summaries)] == expected

    large = many_tracks(4000, seed=2)
    started = time.perf_counter()
    stitch_tracks(large)
    assert time.perf_counter() - started < 1.5  # all pairs took about 3 s on the same machine


def hidden_player_video(tmp_path):
    """A red player walks, is hidden for 1.3 s (the tracker forgets him) and comes back; a blue one stands."""
    import cv2

    path = tmp_path / "hidden.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
    hidden = range(30, 63)
    for index in range(100):
        image = np.full((360, 640, 3), (40, 140, 40), dtype=np.uint8)
        x = 100 + index * 2
        if index not in hidden:
            cv2.rectangle(image, (x, 120), (x + 40, 210), (0, 0, 230), -1)
        cv2.rectangle(image, (520, 120), (560, 210), (230, 60, 0), -1)
        writer.write(image)
    writer.release()
    return path


class ColorDetector:
    """Finds the red and the blue player by their colour, as a model would find them."""
    model_name, model_version, runtime, last_inference_ms = "fake", "0", "cpu", 0.0

    def detect(self, image, confidence):
        import cv2

        found = []
        for color in ((0, 0, 230), (230, 60, 0)):
            ys, xs = np.nonzero(cv2.inRange(image, np.array(color) - 25, np.array(color) + 25))
            if len(xs) > 100:
                found.append(Detection(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.85, 0, OBJECT_TYPE_PLAYER))
        return found


def test_the_pipeline_suggests_joining_a_player_the_tracker_lost(tmp_path):
    from pipeline import AnalysisOptions, VideoAnalysisPipeline

    result = VideoAnalysisPipeline(ColorDetector()).analyze_video(
        hidden_player_video(tmp_path), AnalysisOptions(frame_stride=1, max_frames=100, pitch_filter=False))
    tracks = {t["track_id"]: t for t in result["tracking"]}
    assert len(tracks) == 3  # the red player got two ids, the blue one kept his
    (link,) = result["track_links"]["links"]
    before, after = tracks[link["from_track"]], tracks[link["to_track"]]
    assert before["last_frame"] == link["from_last_frame"] < link["to_first_frame"] == after["first_frame"]
    assert 1.0 < link["gap_seconds"] <= MAX_GAP_SECONDS and link["confidence"] >= 0.5
    assert result["track_links"]["groups"] == [[link["from_track"], link["to_track"]]]


def test_the_model_evaluation_counts_the_people_after_the_links(tmp_path):
    from evaluate_models import evaluate_on_video, to_markdown

    row = evaluate_on_video(ColorDetector(), hidden_player_video(tmp_path), frame_stride=1, max_frames=100)
    assert (row["unique_tracks"], row["identities_after_links"]) == (3, 2)
    header, _, line = to_markdown([{"model": "fake", **row}]).splitlines()
    position = header.split(" | ").index("identities_after_links")
    assert line.split(" | ")[position] == "2"
