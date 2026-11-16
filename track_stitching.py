"""Links tracks that are the same player cut in two (step after tracking and the team colours).

ByteTrack loses a player when he is hidden behind another one, leaves the picture for a moment or
the camera pans fast; when he comes back he gets a NEW id. The video then shows "ID 12" and, a
second later and almost in the same place, "ID 31". This module finds those pairs so the staff can
label the player once and the label covers the whole play.

A track A that ends is linked to a track B that starts later when everything agrees:
  - time: B starts after A ends (never both on screen at once) and within MAX_GAP_SECONDS;
  - place: B starts where A would be if he kept running at his last speed. The error is measured
    in box heights (a player far from the camera is small, near it he is big) and the allowed
    radius grows with the gap, because in a second a player can change direction;
  - size: the boxes are of similar height (the same player at the same distance from the camera);
  - team: if both have a team, it is the same one;
  - shirt: the median shirt colours (Lab) are close.
Every candidate gets a confidence from 0 to 1 and the pairs are chosen greedily, the surest first,
so a track is continued by at most one track and continues at most one. Chains are allowed
(12 -> 31 -> 40): each one is listed in ``groups``.

Nothing is merged in the data: the links are suggestions with their reasons, and a person accepts
or rejects them. A wrong merge would mix two players' trajectories, which is worse than a cut.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from team_assignment import UNASSIGNED
from tracking import TrackedDetection

MAX_GAP_SECONDS = 1.5
# Allowed distance between the predicted and the real start of B, in box heights: a base error plus
# what a change of direction can add per second of gap. Measured on a real broadcast clip
# (MODEL_EVALUATION.md): the right links stayed under 0.85 heights, the wrong ones were above 1.0.
BASE_RADIUS_HEIGHTS = 0.6
RADIUS_PER_SECOND_HEIGHTS = 0.4
# The speed used for the prediction is capped: a fast sprint is about 4 body heights per second.
MAX_SPEED_HEIGHTS = 4.5
# Speed of the last part of A (and the first of B), to predict where he goes.
VELOCITY_WINDOW_SECONDS = 0.5
MAX_HEIGHT_RATIO = 1.6
# Shirt colours farther than this (Lab, OpenCV 8-bit scale) are different kits.
MAX_COLOR_DISTANCE = 28.0
MIN_CONFIDENCE = 0.5
# A track seen in fewer frames than this says too little about where it goes.
MIN_POINTS = 2


@dataclass
class TrackSummary:
    track_id: int
    first_frame: int
    last_frame: int
    first_time: float
    last_time: float
    first_point: np.ndarray
    last_point: np.ndarray
    start_velocity: np.ndarray  # px per second
    end_velocity: np.ndarray
    height: float               # median box height, px
    points: int


@dataclass
class TrackLink:
    from_track: int
    to_track: int
    gap_seconds: float
    distance_heights: float
    color_distance: float | None
    same_team: bool | None
    confidence: float

    def as_dict(self) -> dict:
        return {
            "from_track": self.from_track,
            "to_track": self.to_track,
            "gap_seconds": round(self.gap_seconds, 3),
            "distance_heights": round(self.distance_heights, 3),
            "color_distance": None if self.color_distance is None else round(self.color_distance, 2),
            "same_team": self.same_team,
            "confidence": round(self.confidence, 3),
        }


def _velocity(times: list[float], points: list[np.ndarray], from_end: bool) -> np.ndarray:
    """Mean speed (px/s) over the last (or first) VELOCITY_WINDOW_SECONDS of a track."""
    if len(times) < 2:
        return np.zeros(2)
    if from_end:
        edge_time, edge_point = times[-1], points[-1]
        index = len(times) - 2
        while index > 0 and edge_time - times[index - 1] <= VELOCITY_WINDOW_SECONDS:
            index -= 1
        other_time, other_point = times[index], points[index]
        elapsed, moved = edge_time - other_time, edge_point - other_point
    else:
        edge_time, edge_point = times[0], points[0]
        index = 1
        while index < len(times) - 1 and times[index + 1] - edge_time <= VELOCITY_WINDOW_SECONDS:
            index += 1
        other_time, other_point = times[index], points[index]
        elapsed, moved = other_time - edge_time, other_point - edge_point
    return moved / elapsed if elapsed > 0 else np.zeros(2)


def summarize_tracks(frames: Iterable[tuple[int, float, list[TrackedDetection]]]) -> dict[int, TrackSummary]:
    """Where and when every track starts and ends, from (frame number, seconds, tracked players)."""
    seen: dict[int, tuple[list[int], list[float], list[np.ndarray], list[float]]] = {}
    for frame_number, timestamp, tracked_players in frames:
        for tracked in tracked_players:
            frames_of, times, points, heights = seen.setdefault(tracked.track_id, ([], [], [], []))
            frames_of.append(frame_number)
            times.append(float(timestamp))
            points.append(np.array(tracked.detection.foot_point, dtype=np.float64))
            heights.append(float(tracked.detection.height))
    summaries = {}
    for track_id, (frames_of, times, points, heights) in seen.items():
        summaries[track_id] = TrackSummary(
            track_id=track_id,
            first_frame=frames_of[0], last_frame=frames_of[-1],
            first_time=times[0], last_time=times[-1],
            first_point=points[0], last_point=points[-1],
            start_velocity=_velocity(times, points, from_end=False),
            end_velocity=_velocity(times, points, from_end=True),
            height=float(np.median(heights)),
            points=len(times),
        )
    return summaries


def _capped(velocity: np.ndarray, height: float) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    limit = MAX_SPEED_HEIGHTS * height
    return velocity * (limit / speed) if speed > limit > 0 else velocity


def score_link(a: TrackSummary, b: TrackSummary, team_a: str | None = None, team_b: str | None = None,
               color_a: np.ndarray | None = None, color_b: np.ndarray | None = None) -> TrackLink | None:
    """The link "B continues A" with its confidence, or None when something rules it out."""
    gap = b.first_time - a.last_time
    if gap <= 0 or gap > MAX_GAP_SECONDS or a.points < MIN_POINTS or b.points < MIN_POINTS:
        return None
    if a.height <= 0 or b.height <= 0 or max(a.height, b.height) / min(a.height, b.height) > MAX_HEIGHT_RATIO:
        return None
    same_team = None
    if team_a not in (None, UNASSIGNED) and team_b not in (None, UNASSIGNED):
        if team_a != team_b:
            return None
        same_team = True

    scale = (a.height + b.height) / 2.0
    # Meet in the middle of the gap: A runs forward at his last speed, B runs back at his first one.
    half = gap / 2.0
    forward = a.last_point + _capped(a.end_velocity, a.height) * half
    backward = b.first_point - _capped(b.start_velocity, b.height) * half
    distance = float(np.linalg.norm(forward - backward)) / scale
    radius = BASE_RADIUS_HEIGHTS + RADIUS_PER_SECOND_HEIGHTS * gap
    if distance > radius:
        return None

    color_distance = None
    color_score = 0.5  # unknown shirt: neither helps nor hurts much
    if color_a is not None and color_b is not None:
        color_distance = float(np.linalg.norm(np.asarray(color_a, float) - np.asarray(color_b, float)))
        if color_distance > MAX_COLOR_DISTANCE:
            return None
        color_score = 1.0 - color_distance / MAX_COLOR_DISTANCE

    place_score = 1.0 - distance / radius
    time_score = 1.0 - gap / MAX_GAP_SECONDS
    size_score = 1.0 - (max(a.height, b.height) / min(a.height, b.height) - 1.0) / (MAX_HEIGHT_RATIO - 1.0)
    confidence = 0.45 * place_score + 0.2 * time_score + 0.25 * color_score + 0.1 * size_score
    if same_team:
        confidence = min(1.0, confidence + 0.05)
    return TrackLink(a.track_id, b.track_id, gap, distance, color_distance, same_team, confidence)


def stitch_tracks(summaries: dict[int, TrackSummary], track_teams: dict[int, str] | None = None,
                  track_colors: dict[int, np.ndarray] | None = None,
                  min_confidence: float = MIN_CONFIDENCE) -> list[TrackLink]:
    """The chosen links, surest first, each track used at most once as an end and once as a start."""
    track_teams = track_teams or {}
    track_colors = track_colors or {}
    # Only tracks that start within MAX_GAP_SECONDS after A ends can continue it: with the starts
    # sorted, each A looks at that window alone instead of every track (a long video in full mode has
    # thousands of tracks, and comparing all pairs took seconds).
    by_start = sorted(summaries.values(), key=lambda s: s.first_time)
    starts = [s.first_time for s in by_start]
    candidates = []
    for a in summaries.values():
        window = by_start[bisect_right(starts, a.last_time):bisect_right(starts, a.last_time + MAX_GAP_SECONDS)]
        for b in window:
            if b.track_id == a.track_id:
                continue
            link = score_link(a, b, track_teams.get(a.track_id), track_teams.get(b.track_id),
                              track_colors.get(a.track_id), track_colors.get(b.track_id))
            if link is not None and link.confidence >= min_confidence:
                candidates.append(link)
    candidates.sort(key=lambda link: (-link.confidence, link.gap_seconds, link.from_track, link.to_track))
    continued, started = set(), set()
    chosen = []
    for link in candidates:
        if link.from_track in continued or link.to_track in started:
            continue
        continued.add(link.from_track)
        started.add(link.to_track)
        chosen.append(link)
    return chosen


def link_groups(links: list[TrackLink]) -> list[list[int]]:
    """Chains of tracks that would be one player, in time order: [[12, 31, 40], [5, 9]]."""
    following = {link.from_track: link.to_track for link in links}
    heads = sorted(set(following) - set(following.values()))
    groups = []
    for head in heads:
        chain = [head]
        while chain[-1] in following:
            chain.append(following[chain[-1]])
        groups.append(chain)
    return groups


def stitching_payload(frames: Iterable[tuple[int, float, list[TrackedDetection]]],
                      track_teams: dict[int, str] | None = None,
                      track_colors: dict[int, np.ndarray] | None = None) -> dict:
    """What the analysis returns: the links with their reasons and the resulting groups."""
    summaries = summarize_tracks(frames)
    links = stitch_tracks(summaries, track_teams, track_colors)
    return {
        "links": [{**link.as_dict(),
                   "from_last_frame": summaries[link.from_track].last_frame,
                   "to_first_frame": summaries[link.to_track].first_frame} for link in links],
        "groups": link_groups(links),
    }
