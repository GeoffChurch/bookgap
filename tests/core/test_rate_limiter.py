"""Tests for the cross-process file-lock rate limiter in openalex_ingest."""

import multiprocessing
import time
from pathlib import Path

import pytest

import bookgap.openalex_ingest as _oa_mod
from bookgap.openalex_ingest import _RateLimiter


@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    """Monkeypatch CACHE_DIR in bookgap.openalex_ingest and return a fresh _RateLimiter."""
    monkeypatch.setattr(_oa_mod, "CACHE_DIR", tmp_path)
    limiter = _RateLimiter(min_interval=0.1)
    return limiter, tmp_path


def test_enforces_min_interval(tmp_cache):
    """3 sequential calls with 0.1s interval should take at least 0.18s total."""
    limiter, _ = tmp_cache
    start = time.time()
    limiter.wait()
    limiter.wait()
    limiter.wait()
    elapsed = time.time() - start
    assert elapsed >= 0.18, f"Expected >= 0.18s but got {elapsed:.3f}s"


def test_lock_files_created(tmp_cache):
    """After calling wait(), .rate_lock and .rate_ts files should exist in CACHE_DIR."""
    limiter, cache_dir = tmp_cache
    limiter.wait()
    assert (cache_dir / ".rate_lock").exists(), ".rate_lock not created"
    assert (cache_dir / ".rate_ts").exists(), ".rate_ts not created"


def _worker(cache_dir_str: str, result_queue: "multiprocessing.Queue[float]") -> None:
    """Worker that calls the rate limiter once and records the timestamp."""
    import bookgap.openalex_ingest as mod
    from pathlib import Path

    cache_dir = Path(cache_dir_str)
    mod.CACHE_DIR = cache_dir  # type: ignore[assignment]
    limiter = mod._RateLimiter(min_interval=0.1)
    limiter.wait()
    result_queue.put(time.time())


def test_cross_process_coordination(tmp_path):
    """Two worker processes sharing a lock file should both respect min_interval.

    All recorded timestamps should be separated by at least 0.08s.
    """
    ctx = multiprocessing.get_context("fork")
    result_queue: multiprocessing.Queue = ctx.Queue()

    p1 = ctx.Process(target=_worker, args=(str(tmp_path), result_queue))
    p2 = ctx.Process(target=_worker, args=(str(tmp_path), result_queue))

    p1.start()
    p2.start()
    p1.join(timeout=5)
    p2.join(timeout=5)

    assert p1.exitcode == 0, f"Worker 1 exited with code {p1.exitcode}"
    assert p2.exitcode == 0, f"Worker 2 exited with code {p2.exitcode}"

    timestamps = sorted([result_queue.get(), result_queue.get()])
    gap = timestamps[1] - timestamps[0]
    assert gap >= 0.08, f"Expected timestamps at least 0.08s apart, got {gap:.3f}s"
