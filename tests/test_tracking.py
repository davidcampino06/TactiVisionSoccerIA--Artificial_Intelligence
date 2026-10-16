"""ByteTrack adapter: track ids stay stable between frames and empty results never break a frame."""

from detectors import OBJECT_TYPE_PLAYER, Detection
from tracking import ByteTrackTracker, TrackerFactory


def player(x: float, y: float, confidence: float = 0.9) -> Detection:
    return Detection(x, y, x + 40, y + 100, confidence, 0, OBJECT_TYPE_PLAYER)


def test_same_player_keeps_its_track_id_while_moving():
    tracker = ByteTrackTracker(frame_rate=5)
    first = tracker.update([player(100, 200), player(600, 220)])
    second = tracker.update([player(104, 202), player(606, 221)])
    assert len(first) == 2 and len(second) == 2
    by_x_first = {round(t.detection.x1, -2): t.track_id for t in first}
    by_x_second = {round(t.detection.x1, -2): t.track_id for t in second}
    assert by_x_first == by_x_second


def test_frame_with_only_weak_detections_returns_no_tracks():
    # ByteTrack answers with sv.Detections.empty() (no "source_index" column) when nothing is
    # confirmed; this used to raise KeyError and fail the whole analysis of a real video.
    tracker = ByteTrackTracker(frame_rate=5)
    assert tracker.update([player(100, 200, confidence=0.1)]) == []


def test_empty_frame_returns_no_tracks():
    tracker = ByteTrackTracker(frame_rate=5)
    assert tracker.update([]) == []


def test_factory_creates_bytetrack_by_name():
    assert isinstance(TrackerFactory.create("ByteTrack", frame_rate=25), ByteTrackTracker)
