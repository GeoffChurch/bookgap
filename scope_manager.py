#!/usr/bin/env python3
"""Scope Manager TUI — browse OpenAlex taxonomy, view fetch/embed status, launch jobs."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pyalex
import pyarrow.parquet as pq

import bookgap.openalex_ingest as _ingest_mod
from bookgap.embedder import SentenceTransformerEmbedder as Embedder
from bookgap.openalex_ingest import ResolvedScope, _scope_parquet_path
from bookgap.util import WorkType

# ---------------------------------------------------------------------------
# Hierarchy resolution (PyAlex queries — returns (id, display_name) pairs)
# ---------------------------------------------------------------------------

def _suffix(raw_id: str) -> str:
    return raw_id.rsplit("/", 1)[-1]


def get_domain_fields(domain_id: str) -> list[tuple[str, str]]:
    """Return (field_id, display_name) pairs for a domain, sorted by name."""
    data: Any = pyalex.Domains()[domain_id]
    fields = data.get("fields", [])
    result = [(_suffix(f["id"]), f["display_name"]) for f in fields]
    return sorted(result, key=lambda x: x[1])


def get_field_subfields(field_id: str) -> list[tuple[str, str]]:
    """Return (subfield_id, display_name) pairs for a field, sorted by name."""
    data: Any = pyalex.Fields()[field_id]
    subfields = data.get("subfields", [])
    result = [(_suffix(sf["id"]), sf["display_name"]) for sf in subfields]
    return sorted(result, key=lambda x: x[1])


def get_subfield_topics_named(subfield_id: str) -> list[tuple[str, str]]:
    """Return (topic_id, display_name) pairs for a subfield, sorted by name, deduplicated."""
    result: list[tuple[str, str]] = []
    for page in pyalex.Topics().filter(**{"subfield.id": subfield_id}).paginate(per_page=200):
        for topic in page:
            tid = topic.get("id")
            name = topic.get("display_name", "")
            if isinstance(tid, str):
                result.append((_suffix(tid), name))
    # Deduplicate by ID, sort by name
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for tid, name in sorted(result, key=lambda x: x[1]):
        if tid not in seen:
            seen.add(tid)
            unique.append((tid, name))
    return unique


# ---------------------------------------------------------------------------
# Status queries (filesystem only — no API calls)
# ---------------------------------------------------------------------------

_MODEL_PATHS = {
    "gist": Embedder.ModelName.GIST_SMALL.value,   # "avsolatorio/GIST-small-Embedding-v0"
    "gemma": Embedder.ModelName.GEMMA_300M.value,   # "google/embeddinggemma-300m"
}


def _embedding_dir(scope: ResolvedScope, model_value: str) -> Path:
    """Return the embedding cache directory for a scope + model.

    Uses the same CACHE_DIR as bookgap.openalex_ingest so that test
    monkeypatches on that module-level variable are respected.
    """
    cache_dir: Path = _ingest_mod.CACHE_DIR
    return cache_dir / "embeddings" / model_value / scope.field / scope.subfield / scope.topic / scope.work_type


def get_scope_status(scope: ResolvedScope) -> dict[str, Any]:
    """Check fetch and embed status for a single scope. Filesystem only."""
    parquet_path = _scope_parquet_path(scope)
    fetched = parquet_path.exists()
    work_count = 0
    if fetched:
        try:
            work_count = pq.read_metadata(parquet_path).num_rows
        except Exception:
            work_count = -1  # corrupt file

    gist_dir = _embedding_dir(scope, _MODEL_PATHS["gist"])
    gemma_dir = _embedding_dir(scope, _MODEL_PATHS["gemma"])

    return {
        "fetched": fetched,
        "work_count": work_count,
        "gist_embedded": gist_dir.exists() and any(gist_dir.glob("batch_*_embeddings.npy")),
        "gemma_embedded": gemma_dir.exists() and any(gemma_dir.glob("batch_*_embeddings.npy")),
    }


# ---------------------------------------------------------------------------
# Job manifest management
# ---------------------------------------------------------------------------

JOBS_DIR = ".jobs"


def _jobs_path() -> Path:
    cache_dir: Path = _ingest_mod.CACHE_DIR
    d = cache_dir / JOBS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_job_manifest(job_id: str, manifest: dict[str, Any]) -> None:
    """Write a job manifest atomically (temp file + rename)."""
    dest = _jobs_path() / f"{job_id}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp, dest)


def read_job_manifest(job_id: str) -> dict[str, Any]:
    """Read a job manifest by ID."""
    path = _jobs_path() / f"{job_id}.json"
    return json.loads(path.read_text())


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def discover_jobs() -> list[dict[str, Any]]:
    """Scan job manifests, detect crashed workers, return all jobs with IDs."""
    jobs_dir = _jobs_path()
    results: list[dict[str, Any]] = []
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            manifest = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        manifest["_job_id"] = path.stem
        if manifest.get("status") == "running" and not _is_pid_alive(manifest.get("pid", -1)):
            manifest["status"] = "crashed"
            write_job_manifest(path.stem, manifest)
        results.append(manifest)
    return results


def has_running_job(action: str, scope_str: str, model: str | None = None) -> bool:
    """Check if there's already a running job for this action+scope+model."""
    for job in discover_jobs():
        if (
            job.get("status") == "running"
            and job.get("action") == action
            and job.get("scope") == scope_str
            and job.get("model") == model
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Worker launch + entry point
# ---------------------------------------------------------------------------

ALL_WORK_TYPES: list[str] = [wt.value for wt in WorkType]
MAX_CONCURRENT_WORKERS = 3


def _make_job_id(action: str, scope_str: str, model: str | None = None) -> str:
    parts = [action]
    scope_parts = scope_str.split("/")
    if len(scope_parts) >= 4:
        parts.append(scope_parts[2])  # topic
        parts.append(scope_parts[3])  # work_type
    else:
        parts.append(scope_str.replace("/", "_"))
    if model:
        parts.append(model.split("/")[-1][:8])
    parts.append(str(int(time.time())))
    return "_".join(parts)


def launch_worker(action: str, scope_str: str, model: str | None = None) -> str | None:
    """Launch a detached worker subprocess. Returns job_id or None if duplicate/at capacity."""
    if has_running_job(action, scope_str, model):
        return None
    jobs = discover_jobs()
    running_count = sum(1 for j in jobs if j.get("status") == "running")
    if running_count >= MAX_CONCURRENT_WORKERS:
        return None

    job_id = _make_job_id(action, scope_str, model)
    log_path = _jobs_path() / f"{job_id}.log"

    cmd = [sys.executable, os.path.abspath(__file__), "--worker", action, scope_str]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["--job-id", job_id])

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    manifest = {
        "pid": proc.pid,
        "action": action,
        "scope": scope_str,
        "model": model,
        "status": "running",
        "started_at": datetime.datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
    }
    write_job_manifest(job_id, manifest)
    return job_id


def run_worker(action: str, scope_str: str, model: str | None, job_id: str) -> None:
    """Worker entry point — called in the detached subprocess."""
    from bookgap.openalex_ingest import fetch_works, fetch_all_scopes, embed_works
    from bookgap.util import Filter

    # Update manifest with our actual PID
    manifest = read_job_manifest(job_id)
    manifest["pid"] = os.getpid()
    write_job_manifest(job_id, manifest)

    try:
        parts = scope_str.split("/")
        if action == "fetch":
            if len(parts) == 4:
                scope = ResolvedScope(field=parts[0], subfield=parts[1], topic=parts[2], work_type=parts[3])
                fetch_works(scope)
            elif len(parts) == 2:
                filt = Filter(field=parts[0], subfield=parts[1], work_types=tuple(ALL_WORK_TYPES))
                fetch_all_scopes(filt)
            elif len(parts) == 1:
                if any(d[0] == parts[0] for d in KNOWN_DOMAINS):
                    filt = Filter(domain=parts[0], work_types=tuple(ALL_WORK_TYPES))
                else:
                    filt = Filter(field=parts[0], work_types=tuple(ALL_WORK_TYPES))
                fetch_all_scopes(filt)
            else:
                raise ValueError(f"Unexpected scope format for fetch: {scope_str}")
        elif action == "embed":
            if len(parts) != 4:
                raise ValueError(f"Embed requires a full scope (field/subfield/topic/work_type), got: {scope_str}")
            if model is None:
                raise ValueError("Embed requires --model")
            scope = ResolvedScope(field=parts[0], subfield=parts[1], topic=parts[2], work_type=parts[3])
            model_enum = Embedder.ModelName(model)
            embed_works(scope, model_enum)
        else:
            raise ValueError(f"Unknown action: {action}")

        manifest = read_job_manifest(job_id)
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.datetime.now().isoformat()
        write_job_manifest(job_id, manifest)

    except Exception as e:
        manifest = read_job_manifest(job_id)
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        manifest["finished_at"] = datetime.datetime.now().isoformat()
        write_job_manifest(job_id, manifest)
        raise


# ---------------------------------------------------------------------------
# Known domains
# ---------------------------------------------------------------------------

KNOWN_DOMAINS: list[tuple[str, str]] = [
    ("3", "Physical Sciences"),
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scope Manager TUI")
    parser.add_argument("--worker", metavar="ACTION", help="Run in worker mode (fetch|embed)")
    parser.add_argument("scope", nargs="?", help="Scope string (e.g., 17/1702/T10181/article)")
    parser.add_argument("--model", help="Embedder model name (for embed action)")
    parser.add_argument("--job-id", help="Job ID (set by launcher)")
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.scope, args.model, args.job_id)
    else:
        # TUI mode — implemented in Task 4
        print("TUI not implemented yet. Use --worker for headless mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
