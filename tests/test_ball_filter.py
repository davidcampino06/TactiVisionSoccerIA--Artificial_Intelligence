"""Only a ball with grass around it is the match ball (TV graphics and the stands are ignored)."""

import numpy as np

from detectors import OBJECT_TYPE_BALL, Detection
from pipeline import _ball_on_pitch, _green_mask


def test_a_ball_without_grass_around_it_is_not_the_match_ball():
    frame = np.full((200, 400, 3), (60, 140, 60), dtype=np.uint8)  # grass (BGR)
    frame[:60, 300:] = (120, 40, 20)                                # TV scoreboard in a corner
    on_grass = Detection(100, 100, 110, 110, 0.5, 32, OBJECT_TYPE_BALL)
    on_scoreboard = Detection(340, 20, 350, 30, 0.5, 32, OBJECT_TYPE_BALL)  # e.g. a club crest
    green = _green_mask(frame)
    assert _ball_on_pitch(on_grass, green)
    assert not _ball_on_pitch(on_scoreboard, green)
