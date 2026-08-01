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
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import dotenv
import pyalex
import requests.adapters
import requests.exceptions
import urllib3.util.retry

from bookgap.util import CACHE_DIR, LOGGER

# -- Configuration (runs on import) ------------------------------------------

dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Contact address for OpenAlex's "polite pool", which gives faster, more reliable
# service in exchange for being identifiable. Sourced from the environment rather
# than hardcoded: the literal address was author-identifying, which matters for
# anonymous peer review and for publishing this repo at all.
#
# Unset is legal -- OpenAlex still serves requests, just from the common pool --
# so warn rather than fail, since most of the codebase never touches the API.
_OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL")
if _OPENALEX_EMAIL:
    pyalex.config.email = _OPENALEX_EMAIL
else:
    LOGGER.warning(
        "OPENALEX_EMAIL is not set; OpenAlex requests will use the common pool "
        "(slower, rate-limited more aggressively). Set it in .env to join the "
        "polite pool: https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication"
    )
if os.environ.get("OPENALEX_API_KEY"):
    pyalex.config.api_key = os.environ["OPENALEX_API_KEY"]
pyalex.config.max_retries = 5
pyalex.config.retry_backoff_factor = 0.5

# -- HTTP timeout -------------------------------------------------------------
# pyalex never sets a request timeout, so a dead socket hangs forever.
# Inject a default at the HTTPAdapter level so urllib3's Retry still works
# normally — each individual request gets a timeout, but retries and backoff
# proceed as configured.

_HTTP_TIMEOUT = 5  # seconds per HTTP request

_original_adapter_send = requests.adapters.HTTPAdapter.send

# Ignore server Retry-After entirely: a 17000s "come back later" would wedge the
# process, and transient throttling is already handled by urllib3's own
# exponential backoff_factor. Returning 0 makes urllib3 use its own backoff.
_original_get_retry_after = urllib3.util.retry.Retry.get_retry_after


def _ignore_retry_after(self, response):
    return 0


urllib3.util.retry.Retry.get_retry_after = _ignore_retry_after


def _strip_api_key(url: str) -> str:
    parts = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "api_key"]
    return urlunparse(parts._replace(query=urlencode(query)))


def _strip_auth(request) -> None:
    """Remove pyalex's Authorization header so the request goes to the polite pool."""
    request.headers.pop("Authorization", None)
    request.url = _strip_api_key(request.url)


def _is_budget_exhausted(response) -> bool:
    if response.status_code != 429:
        return False
    try:
        body = response.content[:1024]
    except Exception:
        return False
    return b"Insufficient budget" in body or b"insufficient budget" in body


def _fallback_to_polite_pool(request) -> None:
    LOGGER.warning(
        "OpenAlex rate/budget limit exhausted; dropping api_key and retrying in polite pool (mailto only) for the rest of this process"
    )
    pyalex.config.api_key = None
    _strip_auth(request)


def _adapter_send_with_timeout(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
    if timeout is None:
        timeout = _HTTP_TIMEOUT
    try:
        response = _original_adapter_send(self, request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies)
    except requests.exceptions.RetryError:
        if not pyalex.config.api_key:
            raise
        _fallback_to_polite_pool(request)
        response = _original_adapter_send(self, request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies)
        return response
    if _is_budget_exhausted(response) and pyalex.config.api_key:
        _fallback_to_polite_pool(request)
        response = _original_adapter_send(self, request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies)
    return response


requests.adapters.HTTPAdapter.send = _adapter_send_with_timeout


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
