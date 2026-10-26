from fastapi.testclient import TestClient

from config import settings
from main import app

API_HEADERS = {"X-API-Key": "test-key"}


def test_health_returns_expected_payload():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"service": "TactiVision AI", "status": "OK"}


def test_model_reports_configuration():
    with TestClient(app) as client:
        body = client.get("/api/model").json()
    assert body["model"] == settings.model_name
    assert body["tracker"] == "ByteTrack"
    assert "YOLO26n" in body["supported_models"]


def test_analyze_requires_api_key(synthetic_pitch_video):
    with TestClient(app) as client:
        with synthetic_pitch_video.open("rb") as video:
            response = client.post("/api/analyze", files={"video": ("pitch.mp4", video, "video/mp4")})
    assert response.status_code == 401


def test_analyze_rejects_unsupported_extension():
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze", headers=API_HEADERS, files={"video": ("notes.txt", b"hello", "text/plain")}
        )
    assert response.status_code == 400


def test_analyze_rejects_corrupt_video():
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze", headers=API_HEADERS, files={"video": ("broken.mp4", b"not a video", "video/mp4")}
        )
    # Rejected by its first bytes before reaching OpenCV (503 only if the model could not load).
    assert response.status_code in (400, 503)


def test_analyze_rejects_truncated_mp4(tmp_path):
    """Real MP4 signature but no video inside: passes the type check and fails in the pipeline."""
    header = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64
    with TestClient(app) as client:
        response = client.post(
            "/api/analyze", headers=API_HEADERS, files={"video": ("cut.mp4", header, "video/mp4")}
        )
    assert response.status_code in (422, 503)


def test_analyze_real_pipeline_on_synthetic_video(synthetic_pitch_video):
    with TestClient(app) as client:
        if not client.get("/api/model").json()["loaded"]:
            import pytest
            pytest.skip("Model not available in this environment")
        with synthetic_pitch_video.open("rb") as video:
            response = client.post(
                "/api/analyze",
                headers=API_HEADERS,
                data={"frame_stride": "2", "max_frames": "10"},
                files={"video": ("pitch.mp4", video, "video/mp4")},
            )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "REAL_VIDEO_ANALYSIS"
    assert body["frames_processed"] == 10
    assert body["pitch_visible"] is True


def test_analyze_detects_people_in_real_photo_video(people_video):
    import pytest
    if people_video is None:
        pytest.skip("PEOPLE_IMAGE_PATH not provided")
    with TestClient(app) as client:
        if not client.get("/api/model").json()["loaded"]:
            pytest.skip("Model not available in this environment")
        with people_video.open("rb") as video:
            body = client.post(
                "/api/analyze",
                headers=API_HEADERS,
                data={"frame_stride": "1", "max_frames": "20"},
                files={"video": ("people.mp4", video, "video/mp4")},
            ).json()
    assert body["players_detected"] >= 1
    assert body["tracking"], "ByteTrack should produce at least one track"
    assert all(track["frames_seen"] >= 1 for track in body["tracking"])
    # Not a football pitch -> no tactical issues (and no strengths) must be invented.
    assert body["possible_issues"] == []
    assert body["strengths"] == []


def test_simulation_is_clearly_labelled():
    with TestClient(app) as client:
        body = client.post("/api/analyze/simulation", headers=API_HEADERS, data={"frames": "60"}).json()
    assert body["mode"] == "SIMULATION_MODE"
    assert body["model"] == "SIMULATION"
    assert any("MODO SIMULACIÓN" in warning for warning in body["warnings"])
    assert body["tactical_indicators"]


def test_simulation_recommendations_carry_their_explanation():
    """The API keeps the plain explanation and the example frame of every recommendation."""
    with TestClient(app) as client:
        body = client.post("/api/analyze/simulation", data={"frames": "60"}, headers=API_HEADERS).json()
    assert body["recommendations"], "the synthetic match always has at least one possible issue"
    for recommendation in body["recommendations"]:
        explanation = recommendation["explanation"]
        assert explanation["seen"] and explanation["meaning"] and explanation["coach"]
        assert explanation["example"]["players"], recommendation["issue_id"]


def test_simulation_strengths_reach_the_api_response():
    """What the team did well travels with its explanation and its example frame."""
    with TestClient(app) as client:
        body = client.post("/api/analyze/simulation", data={"frames": "60"}, headers=API_HEADERS).json()
    assert body["strengths"], "the synthetic match has at least one clear strength"
    for strength in body["strengths"]:
        assert strength["title"] and strength["team"] in ("TEAM_A", "TEAM_B")
        explanation = strength["explanation"]
        assert explanation["seen"] and explanation["meaning"] and explanation["keep"]
        assert explanation["example"]["players"], strength["id"]
