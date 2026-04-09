"""Centralized pyalex configuration and rate limiting.

Every module that uses pyalex should import it from here:

    from bookgap.pyalex_client import pyalex, RATE_LIMITER

This ensures consistent API credentials, retry settings, and cross-process
rate limiting.

Future consideration: if pyalex call patterns stabilize, it may be worth
wrapping entity types (Works, Topics, etc.) with functions that call
RATE_LIMITER.wait() automatically so callers don't need to remember.
"""
import fcntl
import os
import time
from pathlib import Path

import pyalex

from bookgap.util import CACHE_DIR

# -- Configuration (runs on import) ------------------------------------------

pyalex.config.email = "gchurchill@cs.stonybrook.edu"
if os.environ.get("OPENALEX_API_KEY"):
    pyalex.config.api_key = os.environ["OPENALEX_API_KEY"]
pyalex.config.max_retries = 5
pyalex.config.retry_backoff_factor = 0.5


# -- Cross-process rate limiter ----------------------------------------------

class _RateLimiter:
    """Cross-process rate limiter for OpenAlex API calls.

    Uses a file lock so multiple worker processes coordinate on API timing.
    Falls back gracefully when only one process is running.
    """

    def __init__(self, min_interval: float = 0.1):
        self._min_interval = min_interval

    def _lock_path(self) -> Path:
        return CACHE_DIR / ".rate_lock"

    def _ts_path(self) -> Path:
        return CACHE_DIR / ".rate_ts"

    def wait(self) -> None:
        lock_path = self._lock_path()
        ts_path = self._ts_path()
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            now = time.time()
            try:
                last = float(ts_path.read_text())
            except (FileNotFoundError, ValueError):
                last = 0.0
            elapsed = now - last
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            ts_path.write_text(str(time.time()))


RATE_LIMITER = _RateLimiter(min_interval=0.1)
