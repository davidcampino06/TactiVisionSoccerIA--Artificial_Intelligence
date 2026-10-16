"""The calibration script decides like the app and never trades a found problem for a new threshold."""

from calibration.calibrate import best_threshold, fires, score


def test_a_rule_fires_only_with_enough_frames_and_persistence():
    assert not fires([0.9] * 7, 0.5, "above")               # fewer than MIN_SAMPLES frames
    assert fires([0.9] * 4 + [0.1] * 4, 0.5, "above")       # crossed in half of the frames
    assert not fires([0.9] * 3 + [0.1] * 5, 0.5, "above")
    assert fires([0.1] * 8, 0.5, "below")


def test_best_threshold_keeps_hits_and_adds_no_false_alarm():
    items = [
        ("a:TEAM_A", [0.30] * 8, "above", True),   # problem the current threshold misses
        ("b:TEAM_A", [0.45] * 8, "above", True),   # problem already found
        ("c:TEAM_A", [0.20] * 8, "above", False),  # fine team: must stay quiet
    ]
    current = 0.40
    assert score(items, current)["agree"] == 2
    best = best_threshold(items, current)
    assert score(items, best) == {"tp": 2, "fp": 0, "fn": 0, "tn": 1, "agree": 3, "total": 3}
    assert 0.20 < best <= 0.30
