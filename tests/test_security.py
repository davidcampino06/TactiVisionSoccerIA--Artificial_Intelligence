"""Security of the AI service: API key, rate limit, real video type and HTTP headers."""

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from request_guard import SlidingWindowRateLimiter, detect_container, matches_extension

API_HEADERS = {"X-API-Key": "test-key"}

MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00"
MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00"
AVI = b"RIFF\x24\x10\x00\x00AVI LIST"
MKV = b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\xf7\x81"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------- rate limiter

def test_rate_limiter_allows_the_limit_then_blocks():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(3, window_seconds=60, clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    clock.now += 10
    assert limiter.check("a") == pytest.approx(50)  # the oldest request leaves the window in 50 s
    assert limiter.check("b") == 0.0  # every client has its own window


def test_rate_limiter_window_slides():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(2, window_seconds=60, clock=clock)
    limiter.check("a")
    clock.now += 30
    limiter.check("a")
    clock.now += 30  # the first request is now exactly 60 s old: it leaves the window
    assert limiter.check("a") == 0.0
    assert limiter.check("a") == pytest.approx(30)


def test_refused_requests_do_not_extend_the_wait():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(1, window_seconds=60, clock=clock)
    limiter.check("a")
    for _ in range(5):
        clock.now += 10
        limiter.check("a")
    clock.now += 10
    assert limiter.check("a") == 0.0


def test_idle_clients_are_forgotten_so_memory_stays_bounded():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(1, window_seconds=60, clock=clock, max_clients=2)
    limiter.check("a")
    limiter.check("b")
    clock.now += 61
    limiter.check("c")
    assert set(limiter._requests) == {"c"}


def test_rate_limiter_rejects_bad_settings():
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(0)
    assert SlidingWindowRateLimiter.retry_after_header(0.2) == "1"
    assert SlidingWindowRateLimiter.retry_after_header(12.1) == "13"


# ---------------------------------------------------------------- real video type

@pytest.mark.parametrize(("header", "family"), [(MP4, "iso-bmff"), (MOV, "iso-bmff"), (AVI, "avi"), (MKV, "ebml"),
                                                (PNG, None), (b"hello world!", None), (b"", None)])
def test_detect_container(header, family):
    assert detect_container(header) == family


def test_extension_must_match_the_content():
    assert matches_extension(MP4, ".mp4") and matches_extension(MOV, ".MOV")
    assert matches_extension(MKV, ".webm") and matches_extension(AVI, ".avi")
    assert not matches_extension(MP4, ".avi")
    assert not matches_extension(PNG, ".mp4")
    assert not matches_extension(MP4, ".exe")


@pytest.mark.parametrize(("name", "content"), [("photo.mp4", PNG), ("script.webm", b"#!/bin/sh\nrm -rf /"),
                                               ("clip.avi", MP4)])
def test_fake_videos_are_rejected_before_analysis(name, content):
    with TestClient(app) as client:
        response = client.post("/api/analyze", headers=API_HEADERS, files={"video": (name, content, "video/mp4")})
    if response.status_code == 503:
        pytest.skip("Model not available in this environment")
    assert response.status_code == 400
    assert "not a valid" in response.json()["detail"]


# ---------------------------------------------------------------- API key and rate limit on the endpoints

@pytest.fixture
def strict_limit(monkeypatch):
    limiter = SlidingWindowRateLimiter(2, window_seconds=60)
    monkeypatch.setattr(main, "analysis_limiter", limiter)
    return limiter


def test_wrong_or_missing_api_key_is_refused():
    with TestClient(app) as client:
        assert client.post("/api/analyze/simulation", data={"frames": "20"}).status_code == 401
        assert client.post("/api/analyze/simulation", headers={"X-API-Key": "test-key-x"},
                           data={"frames": "20"}).status_code == 401
        assert client.post("/api/analyze/simulation", headers=API_HEADERS, data={"frames": "20"}).status_code == 200


def test_too_many_requests_get_429_with_retry_after(strict_limit):
    with TestClient(app) as client:
        codes = [client.post("/api/analyze/simulation", headers=API_HEADERS, data={"frames": "20"}).status_code
                 for _ in range(3)]
        blocked = client.post("/api/analyze/simulation", headers=API_HEADERS, data={"frames": "20"})
    assert codes == [200, 200, 429]
    assert int(blocked.headers["Retry-After"]) >= 1


def test_guessing_the_api_key_is_throttled(strict_limit):
    with TestClient(app) as client:
        codes = [client.post("/api/analyze/simulation", headers={"X-API-Key": f"guess-{n}"},
                             data={"frames": "20"}).status_code for n in range(4)]
    assert codes == [401, 401, 429, 429]


def test_clients_are_told_apart_by_forwarded_address(strict_limit):
    with TestClient(app) as client:
        for _ in range(2):
            client.post("/api/analyze/simulation", headers={**API_HEADERS, "X-Forwarded-For": "10.0.0.1"},
                        data={"frames": "20"})
        other = client.post("/api/analyze/simulation", headers={**API_HEADERS, "X-Forwarded-For": "10.0.0.2, 1.1.1.1"},
                            data={"frames": "20"})
        same = client.post("/api/analyze/simulation", headers={**API_HEADERS, "X-Forwarded-For": "10.0.0.1"},
                           data={"frames": "20"})
    assert (other.status_code, same.status_code) == (200, 429)


def test_health_is_not_rate_limited(strict_limit):
    with TestClient(app) as client:
        assert {client.get("/api/health").status_code for _ in range(5)} == {200}


def test_security_headers():
    with TestClient(app) as client:
        headers = client.get("/api/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Cache-Control"] == "no-store"
