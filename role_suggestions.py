"""Suggests which tracks are the referee or a goalkeeper (step after the team colours).

The team colours (team_assignment.py) leave UNASSIGNED the people whose shirt is far from both
teams: referees, goalkeepers, and sometimes somebody on the touchline. Those tracks are then
grouped by shirt colour, and the group says who they probably are:

  - two or more people with that shirt ON SCREEN AT THE SAME TIME: a team plays with one goalkeeper,
    so it is the refereeing crew (referee and assistants wear the same kit) -> "REFEREE";
  - only one person at a time with that shirt: a goalkeeper, or the referee when he is alone in the
    picture; the colour cannot tell them apart -> "GOALKEEPER_OR_REFEREE".

"At the same time" means in at least MIN_TOGETHER_FRAMES analysed frames, more than one box height
apart: one goalkeeper cut in five tracks (one after the other) is still one person, and a second box
the tracker opened on top of him for a moment is not a second person. Only tracks with enough shirt
samples take part (a colour seen in one or two frames is noise), and a "shirt" that is pitch green
is the grass seen through a small box, not a kit: the team colours already ignore green pixels, so
a green kit cannot be measured either.

These are suggestions with the reason written out; the staff confirms the role of each track
(referee, player or outsider) in the app, like the shirt number.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from team_assignment import COLOR_WEIGHTS, OUTLIER_DISTANCE_FACTOR, UNASSIGNED, TeamAssignmentResult, _lab_to_hex

REFEREE = "REFEREE"
GOALKEEPER_OR_REFEREE = "GOALKEEPER_OR_REFEREE"
# Two shirts closer than this (Lab with the team weights) are the same kit.
SAME_KIT_DISTANCE = 18.0
MIN_SAMPLES = 3
MIN_TOGETHER_FRAMES = 3
# Hue and saturation of the pitch mask of the shirt colours (team_assignment._torso_color_lab), but
# also dark: that mask keeps the grass in shadow, and a small box then averages to a dark green "shirt"
# (a white player of a real clip came out #142506).
GRASS_HSV_LOW, GRASS_HSV_HIGH = (35, 40, 15), (85, 255, 255)

# Where a track is in each analysed frame: frame number -> (foot x, foot y, box height), in pixels.
Positions = dict[int, dict[int, tuple[float, float, float]]]


@dataclass
class _Kit:
    tracks: list[int]
    color: np.ndarray
    together: int  # most people with this kit seen in the same frames


def _kits(candidates: list[int], colors: dict[int, np.ndarray]) -> list[list[int]]:
    """Groups of tracks with the same shirt (single linkage: a chain of close colours is one kit)."""
    weighted = {t: np.asarray(colors[t], dtype=np.float64) * COLOR_WEIGHTS for t in candidates}
    parent = {t: t for t in candidates}

    def root(t: int) -> int:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for index, a in enumerate(candidates):
        for b in candidates[index + 1:]:
            if np.linalg.norm(weighted[a] - weighted[b]) <= SAME_KIT_DISTANCE:
                parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for t in candidates:
        groups.setdefault(root(t), []).append(t)
    return [sorted(group) for group in groups.values()]


def is_grass(lab_color: np.ndarray) -> bool:
    pixel = np.clip(np.asarray(lab_color), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV)
    return bool(cv2.inRange(hsv, GRASS_HSV_LOW, GRASS_HSV_HIGH)[0, 0])


def _people_in_frame(boxes: list[tuple[float, float, float]]) -> int:
    """People among these boxes: boxes closer than one box height are the same person."""
    left = list(boxes)
    people = 0
    while left:
        x, y, height = left.pop()
        left = [(ox, oy, oh) for ox, oy, oh in left if np.hypot(ox - x, oy - y) > max(height, oh)]
        people += 1
    return people


def most_at_once(tracks: list[int], positions: Positions) -> int:
    """Most people seen together, counting only what happens in at least MIN_TOGETHER_FRAMES frames."""
    per_frame: dict[int, list[tuple[float, float, float]]] = {}
    for track_id in tracks:
        for frame, box in positions.get(track_id, {}).items():
            per_frame.setdefault(frame, []).append(box)
    counts = sorted((_people_in_frame(boxes) for boxes in per_frame.values()), reverse=True)
    if not counts:
        return 0
    return counts[MIN_TOGETHER_FRAMES - 1] if len(counts) >= MIN_TOGETHER_FRAMES else 1


def suggest_roles(team_result: TeamAssignmentResult, positions: Positions) -> list[dict]:
    """One suggestion per track that looks like a referee or a goalkeeper, ordered by track id."""
    candidates = sorted(
        track_id for track_id, team in team_result.track_teams.items()
        if team == UNASSIGNED
        and track_id in team_result.track_colors and track_id in positions
        and team_result.track_samples.get(track_id, 0) >= MIN_SAMPLES
        and team_result.track_color_distance.get(track_id, 0.0) > OUTLIER_DISTANCE_FACTOR
        and not is_grass(team_result.track_colors[track_id])
    )
    kits = []
    for tracks in _kits(candidates, team_result.track_colors):
        color = np.median([team_result.track_colors[t] for t in tracks], axis=0)
        kits.append(_Kit(tracks, color, max(1, most_at_once(tracks, positions))))

    suggestions = []
    for number, kit in enumerate(sorted(kits, key=lambda k: k.tracks[0]), start=1):
        hex_color = _lab_to_hex(kit.color)
        if kit.together >= 2:
            role, confidence = REFEREE, min(0.95, 0.6 + 0.15 * (kit.together - 1))
            reason = (f"Camiseta {hex_color}, que no es de ningún equipo, la llevan {kit.together} personas "
                      "a la vez: un equipo juega con un solo arquero, así que parece el cuerpo arbitral.")
        else:
            role, confidence = GOALKEEPER_OR_REFEREE, 0.5
            reason = (f"Camiseta {hex_color}, que no es de ningún equipo, y nunca hay dos personas con ella "
                      "a la vez: puede ser un arquero o el árbitro.")
        for track_id in kit.tracks:
            suggestions.append({
                "track_id": track_id,
                "suggestion": role,
                "kit_group": number,
                "kit_color": hex_color,
                "kit_tracks": len(kit.tracks),
                "confidence": round(confidence, 2),
                "reason": reason,
            })
    return sorted(suggestions, key=lambda item: item["track_id"])
