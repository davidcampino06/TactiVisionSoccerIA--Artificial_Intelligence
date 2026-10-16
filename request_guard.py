"""Protections for the analysis endpoints: request rate limit and real video file type.

Rate limit (sliding window)
    Each client keeps a FIFO queue with the times of its last requests. Before a new request the
    times older than the window are dequeued from the front; if ``limit`` times are still inside
    the window the request is refused with 429 and ``Retry-After``. Every request is one enqueue
    and at most ``limit`` dequeues, so it costs O(1) amortised and O(limit) memory per client.
    It runs before the API key check, so guessing the key is throttled too.

Real file type (magic bytes)
    The file extension is chosen by whoever uploads the file, so the first bytes are checked too:
    MP4 and MOV start with an ISO-BMFF box (``ftyp``, ``moov``...), AVI with ``RIFF....AVI `` and
    MKV/WebM with the EBML signature. A ``.mp4`` that is really a text file or an image is rejected
    before it is written to disk or reaches OpenCV.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

# ---------------------------------------------------------------- rate limit


class SlidingWindowRateLimiter:
    """At most ``limit`` requests per client inside any ``window_seconds`` long window."""

    def __init__(self, limit: int, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic,
                 max_clients: int = 10_000) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("limit must be >= 1 and window_seconds > 0")
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._max_clients = max_clients
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str) -> float:
        """Registers a request. Returns 0 when it is allowed, otherwise the seconds to wait."""
        now = self._clock()
        with self._lock:
            times = self._requests.get(client)
            if times is None:
                if len(self._requests) >= self._max_clients:
                    self._forget_idle(now)
                times = self._requests[client] = deque()
            while times and now - times[0] >= self.window:
                times.popleft()
            if len(times) >= self.limit:
                return max(0.0, self.window - (now - times[0]))
            times.append(now)
            return 0.0

    def _forget_idle(self, now: float) -> None:
        """Drops clients without requests inside the window, so memory stays bounded."""
        for client in [key for key, times in self._requests.items() if not times or now - times[-1] >= self.window]:
            del self._requests[client]

    @staticmethod
    def retry_after_header(wait_seconds: float) -> str:
        return str(max(1, math.ceil(wait_seconds)))


# ---------------------------------------------------------------- real file type

ISO_BMFF_BOXES = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"}
EBML_SIGNATURE = b"\x1aE\xdf\xa3"

# Container family each allowed extension must contain.
FAMILY_BY_EXTENSION = {".mp4": "iso-bmff", ".mov": "iso-bmff", ".avi": "avi", ".mkv": "ebml", ".webm": "ebml"}


def detect_container(header: bytes) -> str | None:
    """Container family from the first bytes of the file, or None when it is not a known video."""
    if len(header) >= 8 and header[4:8] in ISO_BMFF_BOXES:
        return "iso-bmff"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "avi"
    if header[:4] == EBML_SIGNATURE:
        return "ebml"
    return None


def matches_extension(header: bytes, extension: str) -> bool:
    """True when the content really is the kind of video the extension says."""
    expected = FAMILY_BY_EXTENSION.get(extension.lower())
    return expected is not None and detect_container(header) == expected
