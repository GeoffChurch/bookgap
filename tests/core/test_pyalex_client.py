"""Tests for the centralized pyalex client module."""


def test_email_configured():
    from bookgap.pyalex_client import pyalex
    assert pyalex.config.email == "gchurchill@cs.stonybrook.edu"


def test_max_retries_configured():
    from bookgap.pyalex_client import pyalex
    assert pyalex.config.max_retries == 5


def test_retry_backoff_factor_configured():
    from bookgap.pyalex_client import pyalex
    assert pyalex.config.retry_backoff_factor == 0.5


def test_rate_limiter_exported():
    from bookgap.pyalex_client import RATE_LIMITER
    assert hasattr(RATE_LIMITER, "wait")
