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


def test_a_track_seen_once_is_never_linked():
    a = run(1, 0, 0, (100, 300), (0.0, 0.0))
    b = run(2, 5, 30, (102, 300), (0.0, 0.0))
    assert stitch_tracks(summarize_tracks(frames(a, b))) == []
