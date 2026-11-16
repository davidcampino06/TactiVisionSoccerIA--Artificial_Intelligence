"""The review sheets (tools/) show what the analysis suggested, one row per link or per kit."""

import base64
import importlib.util
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_link_sheet_has_one_row_per_link_with_both_moments(tmp_path):
    from pipeline import AnalysisOptions, VideoAnalysisPipeline
    from test_track_stitching import ColorDetector, hidden_player_video

    links = load("review_track_links")
    path = hidden_player_video(tmp_path)
    result = VideoAnalysisPipeline(ColorDetector()).analyze_video(
        path, AnalysisOptions(frame_stride=1, max_frames=100, pitch_filter=False))
    sheet = links.contact_sheet(path, result)
    assert sheet.shape[0] == links.PHOTO_HEIGHT + 6  # one link -> one row
    photo = sheet[:links.PHOTO_HEIGHT, :60]
    assert photo[..., 2].max() > 200  # the red player is in the photo, not a blank tile

    result["track_links"] = {"links": [], "groups": []}
    assert links.contact_sheet(path, result).shape[0] == 60  # the "no links" message


def test_the_role_sheet_has_one_row_per_kit():
    roles = load("review_roles")
    tile = np.full((80, 30, 3), (34, 100, 209), np.uint8)
    jpeg = base64.b64encode(cv2.imencode(".jpg", tile)[1].tobytes()).decode()
    result = {
        "track_snapshots": [{"track_id": t, "image_jpeg_base64": jpeg} for t in (15, 31, 17)],
        "track_role_suggestions": [
            {"track_id": 15, "kit_group": 1, "kit_color": "#d16422", "suggestion": "GOALKEEPER_OR_REFEREE"},
            {"track_id": 31, "kit_group": 1, "kit_color": "#d16422", "suggestion": "GOALKEEPER_OR_REFEREE"},
            {"track_id": 17, "kit_group": 2, "kit_color": "#3bb3c2", "suggestion": "GOALKEEPER_OR_REFEREE"},
        ],
    }
    sheet = roles.contact_sheet(result)
    assert sheet.shape[0] == 2 * (roles.PHOTO_HEIGHT + 22 + 6)
    assert tuple(sheet[20, 20]) == (0x22, 0x64, 0xd1)  # the swatch of kit 1 in BGR
    assert roles.contact_sheet({"track_role_suggestions": []}).shape[0] == 60
