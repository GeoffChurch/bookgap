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
    import bookgap.scope_ops as ops
    monkeypatch.setattr(ops, "_ingest_mod", mod)
    return tmp_path


class TestJobManifest:
    def test_write_and_read_manifest(self, tmp_cache):
        from bookgap.scope_ops import write_job_manifest, read_job_manifest
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
        write_job_manifest(job_id, manifest)

        loaded = read_job_manifest(job_id)
        assert loaded["pid"] == 12345
        assert loaded["status"] == "running"
        assert loaded["scope"] == "17/1702/T10181/article"

    def test_atomic_write(self, tmp_cache):
        from bookgap.scope_ops import write_job_manifest
        job_id = "fetch_T10181_article_123"
        write_job_manifest(job_id, {"pid": 1, "status": "running"})
        jobs_dir = tmp_cache / JOBS_DIR_NAME
        assert not list(jobs_dir.glob("*.tmp"))


class TestDiscoverJobs:
    def test_discovers_running_job(self, tmp_cache):
        from bookgap.scope_ops import write_job_manifest, discover_jobs
        write_job_manifest("fetch_T10181_article_100", {
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
        from bookgap.scope_ops import write_job_manifest, discover_jobs
        write_job_manifest("fetch_T10181_article_200", {
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
        from bookgap.scope_ops import write_job_manifest, discover_jobs
        write_job_manifest("fetch_T10181_article_300", {
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
        from bookgap.scope_ops import write_job_manifest, has_running_job
        write_job_manifest("fetch_T10181_article_100", {
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
