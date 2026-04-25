from __future__ import annotations

import json
import os
import time
import pytest
from pathlib import Path


JOBS_DIR_NAME = ".jobs"


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    import bookgap.openalex_ingest as mod
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    return tmp_path


class TestJobManifest:
    def test_write_and_read_manifest(self, tmp_cache):
        from bookgap.jobs import write_manifest, read_manifest
        manifest = {
            "pid": 12345,
            "action": "fetch",
            "scope": "17/1702/T10181/article",
            "model": None,
            "status": "running",
            "started_at": "2026-04-07T14:30:00",
            "finished_at": None,
            "error": None,
        }
        job_id = "fetch_T10181_article_1712345678"
        write_manifest(job_id, manifest)

        loaded = read_manifest(job_id)
        assert loaded["pid"] == 12345
        assert loaded["status"] == "running"
        assert loaded["scope"] == "17/1702/T10181/article"

    def test_atomic_write(self, tmp_cache):
        from bookgap.jobs import write_manifest
        job_id = "fetch_T10181_article_123"
        write_manifest(job_id, {"pid": 1, "status": "running"})
        jobs_dir = tmp_cache / JOBS_DIR_NAME
        assert not list(jobs_dir.glob("*.tmp"))


class TestDiscoverJobs:
    def test_discovers_running_job(self, tmp_cache):
        from bookgap.jobs import write_manifest, discover_jobs
        write_manifest("fetch_T10181_article_100", {
            "pid": os.getpid(),
            "action": "fetch",
            "scope": "17/1702/T10181/article",
            "model": None,
            "status": "running",
            "started_at": "2026-04-07T14:30:00",
            "finished_at": None,
            "error": None,
        })
        jobs = discover_jobs()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "running"

    def test_detects_crashed_job(self, tmp_cache):
        from bookgap.jobs import write_manifest, discover_jobs
        write_manifest("fetch_T10181_article_200", {
            "pid": 99999999,
            "action": "fetch",
            "scope": "17/1702/T10181/article",
            "model": None,
            "status": "running",
            "started_at": "2026-04-07T14:30:00",
            "finished_at": None,
            "error": None,
        })
        jobs = discover_jobs()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "crashed"

    def test_skips_completed_job(self, tmp_cache):
        from bookgap.jobs import write_manifest, discover_jobs
        write_manifest("fetch_T10181_article_300", {
            "pid": 1,
            "action": "fetch",
            "scope": "17/1702/T10181/article",
            "model": None,
            "status": "complete",
            "started_at": "2026-04-07T14:30:00",
            "finished_at": "2026-04-07T14:35:00",
            "error": None,
        })
        jobs = discover_jobs()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "complete"


class TestDuplicatePrevention:
    def test_has_running_job(self, tmp_cache):
        from bookgap.jobs import write_manifest
        from bookgap.scope_ops import has_running_job
        write_manifest("fetch_T10181_article_100", {
            "pid": os.getpid(),
            "action": "fetch",
            "scope": "17/1702/T10181/article",
            "model": None,
            "status": "running",
            "started_at": "2026-04-07T14:30:00",
            "finished_at": None,
            "error": None,
        })
        assert has_running_job("fetch", "17/1702/T10181/article") is True
        assert has_running_job("fetch", "17/1702/T10191/article") is False
        assert has_running_job("embed", "17/1702/T10181/article") is False


class TestWorkerLifecycleWatchdog:
    """Watchdog must convert disk/RSS threshold crossings into a
    cooperative stop request, not into a hard error. Tests fake
    psutil/shutil so we don't need real disk pressure to exercise it."""

    def _seed_manifest(self, job_id: str) -> None:
        from bookgap.jobs import write_manifest
        write_manifest(job_id, {
            "pid": os.getpid(), "status": "running", "error": None,
            "started_at": "2026-04-25T00:00:00", "finished_at": None,
        })

    def test_disk_watchdog_requests_stop(self, tmp_cache, monkeypatch):
        from bookgap import jobs
        self._seed_manifest("disk_test")

        class FakeUsage:
            total = 1_000_000
            used = 950_000  # 95% — over 90% default
            free = 50_000

        monkeypatch.setattr("shutil.disk_usage", lambda p: FakeUsage())

        with jobs.worker_lifecycle(
            "disk_test", mem_limit_pct=None, disk_abort_pct=90.0,
            watchdog_interval=0.05,
        ):
            # Wait for watchdog to flip the flag.
            for _ in range(40):
                if jobs.stop_requested():
                    break
                time.sleep(0.05)

        assert jobs.stop_requested() is True
        reason = jobs.stop_reason() or ""
        assert "disk" in reason and "95.0" in reason

    def test_rss_watchdog_requests_stop(self, tmp_cache, monkeypatch):
        from bookgap import jobs
        self._seed_manifest("rss_test")

        # Drive the RSS check above the 95%-of-RLIMIT_AS threshold.
        # mem_limit_pct=0.5 → threshold = 0.5*100*0.95 = 47.5%. Fake RSS
        # = 60% of total RAM → triggers.
        class FakeMem:
            total = 1_000_000_000  # 1 GB
        class FakeProcMem:
            rss = 600_000_000  # 60% of total
        class FakeProc:
            def memory_info(self): return FakeProcMem()

        import psutil
        monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeMem())
        monkeypatch.setattr(psutil, "Process", lambda *a, **k: FakeProc())
        # Don't actually call setrlimit — keep the test process unaffected.
        import resource
        monkeypatch.setattr(resource, "setrlimit", lambda *a, **k: None)

        with jobs.worker_lifecycle(
            "rss_test", mem_limit_pct=0.5, disk_abort_pct=None,
            watchdog_interval=0.05,
        ):
            for _ in range(40):
                if jobs.stop_requested():
                    break
                time.sleep(0.05)

        assert jobs.stop_requested() is True
        reason = jobs.stop_reason() or ""
        assert "rss" in reason

    def test_no_watchdog_when_thresholds_disabled(self, tmp_cache, monkeypatch):
        from bookgap import jobs
        self._seed_manifest("nowatch_test")
        import resource
        monkeypatch.setattr(resource, "setrlimit", lambda *a, **k: None)

        with jobs.worker_lifecycle(
            "nowatch_test", mem_limit_pct=None,
            disk_abort_pct=None, watchdog_interval=0.05,
        ):
            time.sleep(0.15)
            assert jobs.stop_requested() is False
            assert jobs.stop_reason() is None

    def test_signal_handler_records_reason(self, tmp_cache, monkeypatch):
        from bookgap import jobs
        self._seed_manifest("signal_test")
        import resource
        monkeypatch.setattr(resource, "setrlimit", lambda *a, **k: None)

        # We exercise the handler path directly rather than sending a real
        # signal, since pytest doesn't like signal handlers in subthreads.
        with jobs.worker_lifecycle(
            "signal_test", mem_limit_pct=None, disk_abort_pct=None,
            watchdog_interval=0,  # disable watchdog
        ):
            handler = jobs._make_stop_handler("signal_test")
            handler(15, None)  # SIGTERM
            assert jobs.stop_requested() is True
            assert "signal 15" in (jobs.stop_reason() or "")
