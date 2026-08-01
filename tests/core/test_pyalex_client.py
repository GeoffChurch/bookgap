"""Tests for the centralized pyalex client module."""


def test_email_sourced_from_environment():
    """The polite-pool contact comes from OPENALEX_EMAIL, never a hardcoded literal.

    This used to assert a specific address, which is exactly what made the module
    author-identifying. Assert the wiring instead: whatever the environment says
    (including nothing) is what pyalex gets.
    """
    import os

    from bookgap.pyalex_client import pyalex

    assert pyalex.config.email == os.environ.get("OPENALEX_EMAIL")


def test_max_retries_configured():
    from bookgap.pyalex_client import pyalex
    assert pyalex.config.max_retries == 5


def test_retry_backoff_factor_configured():
    from bookgap.pyalex_client import pyalex
    assert pyalex.config.retry_backoff_factor == 0.5


def test_rate_limiter_exported():
    from bookgap.pyalex_client import RATE_LIMITER
    assert hasattr(RATE_LIMITER, "wait")
