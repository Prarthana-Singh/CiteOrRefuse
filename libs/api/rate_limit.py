"""A minimal fixed-window rate limiter for `POST /ingest` specifically.

Ingestion is the expensive (real OpenAI embedding calls) and abusable
(arbitrary uploaded content) operation the API layer exposes -- `/answer`
against already-indexed content is comparatively cheap and isn't limited
here. This is deliberately hand-rolled rather than a dependency like
`slowapi`: the bookkeeping is a few lines, and at this project's scale (a
single-process demo, no distributed deployment) a library would be more
machinery than the problem needs.
"""
import time
from collections import defaultdict, deque


class FixedWindowRateLimiter:
    """Allows at most `max_calls` per `window_seconds`, per key (e.g. client
    IP). Not thread-safe beyond CPython's GIL-guarded list/dict operations,
    which is sufficient for this project's single-worker deployment.
    """

    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        calls = self._calls[key]
        while calls and now - calls[0] > self.window_seconds:
            calls.popleft()
        if len(calls) >= self.max_calls:
            return False
        calls.append(now)
        return True
