"""Logs of the AI service: secrets hidden, rejected requests visible."""

import logging

from fastapi.testclient import TestClient

from log_setup import REDACTED, redact
from main import app


def test_secrets_are_redacted():
    for text, secret in [("key X-API-Key: abc123secret", "abc123secret"),
                         ("Authorization: Bearer tok.en.value", "tok.en.value"),
                         ("db postgresql://u:hunter2@host/db", "hunter2")]:
        cleaned = redact(text)
        assert secret not in cleaned and REDACTED in cleaned
    assert redact("Analysis finished: 120 frames in 3.2 s") == "Analysis finished: 120 frames in 3.2 s"


def test_rejected_requests_are_logged_without_the_key(caplog):
    caplog.set_level(logging.INFO, logger="tactivision.ai")
    with TestClient(app) as client:
        client.post("/api/analyze/simulation", headers={"X-API-Key": "guessed-key-123"}, data={"frames": "20"})
        client.post("/api/analyze", headers={"X-API-Key": "test-key"},
                    files={"video": ("photo.mp4", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "video/mp4")})
    messages = [record.getMessage() for record in caplog.records if record.name == "tactivision.ai"]
    assert any("Rejected request to /api/analyze/simulation" in m and "invalid API key" in m for m in messages)
    assert "guessed-key-123" not in "\n".join(messages)
    if not any("Model" in m and "failed" in m for m in messages):
        assert any("Rejected upload: content is not a .mp4 video" in m for m in messages)
