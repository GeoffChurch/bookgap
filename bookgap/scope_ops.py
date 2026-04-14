"""Scope operations library — hierarchy queries, status checks, job management.

Reusable by CLI, TUI, and other Python code. No presentation logic.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from bookgap.pyalex_client import pyalex
import pyarrow.parquet as pq

import bookgap.openalex_ingest as _ingest_mod
from bookgap.embedder import SentenceTransformerEmbedder as Embedder
from bookgap.openalex_ingest import ResolvedScope, _scope_parquet_path
from bookgap.util import WorkType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_DOMAINS: list[tuple[str, str]] = [
    ("3", "Physical Sciences"),
]

ALL_WORK_TYPES: list[str] = [wt.value for wt in WorkType]

MAX_CONCURRENT_WORKERS = 3

MODEL_PATHS = {
    "gist": Embedder.ModelName.GIST_SMALL.value,
    "gemma": Embedder.ModelName.GEMMA_300M.value,
}

JOBS_DIR = ".jobs"

# ---------------------------------------------------------------------------
# Hierarchy resolution (PyAlex queries)
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


def get_subfield_topics(subfield_id: str) -> list[tuple[str, str]]:
    """Return (topic_id, display_name) pairs for a subfield, sorted by name, deduplicated."""
    result: list[tuple[str, str]] = []
    for page in pyalex.Topics().filter(**{"subfield.id": subfield_id}).paginate(per_page=200):
        for topic in page:
            tid = topic.get("id")
            name = topic.get("display_name", "")
            if isinstance(tid, str):
                result.append((_suffix(tid), name))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for tid, name in sorted(result, key=lambda x: x[1]):
        if tid not in seen:
            seen.add(tid)
            unique.append((tid, name))
    return unique


def resolve_scopes(
    level: int, path: list[tuple[str, str]], node_id: str, work_types: list[str],
) -> list[ResolvedScope]:
    """Resolve all concrete scopes under a given hierarchy node."""
    if level == 0:
        fields = get_domain_fields(node_id)
        scopes: list[ResolvedScope] = []
        for field_id, _ in fields:
            subfields = get_field_subfields(field_id)
            for sf_id, _ in subfields:
                topics = get_subfield_topics(sf_id)
                for topic_id, _ in topics:
                    for wt in work_types:
                        scopes.append(ResolvedScope(field=field_id, subfield=sf_id, topic=topic_id, work_type=wt))
        return scopes
    elif level == 1:
        field_id = node_id
        subfields = get_field_subfields(field_id)
        scopes = []
        for sf_id, _ in subfields:
            topics = get_subfield_topics(sf_id)
            for topic_id, _ in topics:
                for wt in work_types:
                    scopes.append(ResolvedScope(field=field_id, subfield=sf_id, topic=topic_id, work_type=wt))
        return scopes
    elif level == 2:
        field_id = path[1][0]
        sf_id = node_id
        topics = get_subfield_topics(sf_id)
        scopes = []
        for topic_id, _ in topics:
            for wt in work_types:
                scopes.append(ResolvedScope(field=field_id, subfield=sf_id, topic=topic_id, work_type=wt))
        return scopes
    else:
        field_id = path[1][0]
        sf_id = path[2][0]
        return [
            ResolvedScope(field=field_id, subfield=sf_id, topic=node_id, work_type=wt)
            for wt in work_types
        ]


# ---------------------------------------------------------------------------
# Status queries (filesystem only)
# ---------------------------------------------------------------------------

def _embedding_dir(scope: ResolvedScope, model_value: str) -> Path:
    """Return the embedding cache directory for a scope + model."""
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
            work_count = -1
    gist_dir = _embedding_dir(scope, MODEL_PATHS["gist"])
    gemma_dir = _embedding_dir(scope, MODEL_PATHS["gemma"])
    return {
        "fetched": fetched,
        "work_count": work_count,
        "gist_embedded": gist_dir.exists() and any(gist_dir.glob("batch_*_embeddings.npy")),
        "gemma_embedded": gemma_dir.exists() and any(gemma_dir.glob("batch_*_embeddings.npy")),
    }


def get_aggregate_status(
    scopes: list[ResolvedScope],
    active_types: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute aggregate fetch/embed status per work type."""
    result: dict[str, dict[str, Any]] = {}
    for wt in active_types:
        wt_scopes = [s for s in scopes if s.work_type == wt]
        fetched = 0
        gist = 0
        gemma = 0
        total_works = 0
        for s in wt_scopes:
            status = get_scope_status(s)
            if status["fetched"]:
                fetched += 1
                total_works += status["work_count"]
            if status["gist_embedded"]:
                gist += 1
            if status["gemma_embedded"]:
                gemma += 1
        result[wt] = {
            "fetched": fetched,
            "total": len(wt_scopes),
            "gist_count": gist,
            "gemma_count": gemma,
            "total_works": total_works,
        }
    return result


def _count_glob_dirs(pattern: str) -> int:
    """Count unique parent directories matching a glob pattern."""
    cache_dir: Path = _ingest_mod.CACHE_DIR
    return len(set(p.parent for p in cache_dir.glob(pattern)))


def _count_glob_files(pattern: str) -> int:
    """Count files matching a glob pattern."""
    cache_dir: Path = _ingest_mod.CACHE_DIR
    return len(list(cache_dir.glob(pattern)))


def get_fast_aggregate_status(
    level: int, node_id: str, path: list[tuple[str, str]], active_types: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute aggregate status using pyalex count() + filesystem globs.

    Much faster than resolve_scopes + get_aggregate_status because
    it avoids enumerating every descendant scope via API calls.
    """
    if level == 0:
        api_filter_key = "domain.id"
        fetch_glob_prefix = "works_parquet/*/*/T*"
        embed_glob_prefix = "*/T*"
    elif level == 1:
        api_filter_key = "field.id"
        fetch_glob_prefix = f"works_parquet/{node_id}/*/T*"
        embed_glob_prefix = f"{node_id}/*/T*"
    elif level == 2:
        api_filter_key = "subfield.id"
        field_id = path[1][0]
        fetch_glob_prefix = f"works_parquet/{field_id}/{node_id}/T*"
        embed_glob_prefix = f"{field_id}/{node_id}/T*"
    else:
        return {}

    total_topics = pyalex.Topics().filter(**{api_filter_key: node_id}).count()

    result: dict[str, dict[str, Any]] = {}
    for wt in active_types:
        fetched = _count_glob_files(f"{fetch_glob_prefix}/{wt}.parquet")
        gist_model = MODEL_PATHS["gist"]
        gemma_model = MODEL_PATHS["gemma"]
        gist_count = _count_glob_dirs(
            f"embeddings/{gist_model}/{embed_glob_prefix}/{wt}/batch_*_embeddings.npy"
        )
        gemma_count = _count_glob_dirs(
            f"embeddings/{gemma_model}/{embed_glob_prefix}/{wt}/batch_*_embeddings.npy"
        )
        result[wt] = {
            "fetched": fetched,
            "total": total_topics,
            "gist_count": gist_count,
            "gemma_count": gemma_count,
        }
    return result


# ---------------------------------------------------------------------------
# System resource helper
# ---------------------------------------------------------------------------

def get_system_resources() -> str:
    """Return a formatted string of CPU/RAM/GPU usage."""
    parts: list[str] = []
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        cores = psutil.cpu_count()
        mem = psutil.virtual_memory()
        parts.append(f"CPU: {cpu:.0f}% ({cores} cores)")
        parts.append(f"RAM: {mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GB ({mem.percent:.0f}%)")
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                used, total = line.split(",")
                used_gb = int(used.strip()) / 1024
                total_gb = int(total.strip()) / 1024
                pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
                parts.append(f"GPU: {used_gb:.1f}/{total_gb:.1f} GB VRAM ({pct:.0f}%)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return " | ".join(parts) if parts else "System info unavailable"


# ---------------------------------------------------------------------------
# Job manifest management
# ---------------------------------------------------------------------------

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


def get_job_log_path(job_id: str) -> Path:
    """Return the log file path for a job."""
    return _jobs_path() / f"{job_id}.log"


def dismiss_finished_jobs() -> int:
    """Remove completed/failed/crashed job manifests. Returns count dismissed."""
    jobs = discover_jobs()
    dismissed = 0
    for job in jobs:
        if job.get("status") in ("complete", "failed", "crashed"):
            job_id = job["_job_id"]
            manifest_path = _jobs_path() / f"{job_id}.json"
            log_path = _jobs_path() / f"{job_id}.log"
            manifest_path.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)
            dismissed += 1
    return dismissed


# ---------------------------------------------------------------------------
# Worker launch + entry point
# ---------------------------------------------------------------------------

def _make_job_id(action: str, scope_str: str, model: str | None = None) -> str:
    parts = [action]
    scope_parts = scope_str.split("/")
    if len(scope_parts) >= 4:
        parts.append(scope_parts[2])
        parts.append(scope_parts[3])
    else:
        parts.append(scope_str.replace("/", "_"))
    if model:
        parts.append(model.split("/")[-1][:8])
    parts.append(str(int(time.time())))
    return "_".join(parts)


def launch_worker(
    action: str, scope_str: str, model: str | None = None, work_types: list[str] | None = None,
    snapshot_date: str | None = None,
) -> str | None:
    """Launch a detached worker subprocess. Returns job_id or None if duplicate/at capacity."""
    if has_running_job(action, scope_str, model):
        return None
    jobs = discover_jobs()
    running_count = sum(1 for j in jobs if j.get("status") == "running")
    if running_count >= MAX_CONCURRENT_WORKERS:
        return None

    job_id = _make_job_id(action, scope_str, model)
    log_path = _jobs_path() / f"{job_id}.log"
    scope_unit = f"bookgap-{job_id}.scope"

    # systemd-run --user --scope places the worker in its own transient
    # scope cgroup under user@.service, so it survives SSH logout. The
    # old double-fork + setsid didn't, because setsid doesn't escape the
    # session-N.scope cgroup that logind tears down on logout.
    cmd = [
        "systemd-run", "--user", "--scope", "--quiet", "--collect",
        f"--unit={scope_unit}",
        sys.executable, "-m", "scope_manager", "worker", action, scope_str,
    ]
    if model:
        cmd.extend(["--model", model])
    if work_types:
        cmd.extend(["--work-types", ",".join(work_types)])
    if snapshot_date:
        cmd.extend(["--snapshot-date", snapshot_date])
    cmd.extend(["--job-id", job_id])

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=log_fd,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_fd)

    manifest = {
        "pid": proc.pid,
        "scope_unit": scope_unit,
        "action": action,
        "scope": scope_str,
        "model": model,
        "snapshot_date": snapshot_date,
        "status": "running",
        "started_at": datetime.datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
    }
    write_job_manifest(job_id, manifest)
    return job_id


def run_worker(
    action: str, scope_str: str, model: str | None, job_id: str, work_types: list[str] | None = None,
    snapshot_date: str | None = None,
) -> None:
    """Worker entry point — called in the detached subprocess."""
    import faulthandler
    import resource
    import signal

    from bookgap.openalex_ingest import fetch_works, fetch_all_scopes, embed_works
    from bookgap.util import Filter

    faulthandler.enable()

    # Cap virtual memory at 80% of physical RAM so we get a catchable
    # MemoryError instead of an uncatchable SIGKILL from the OOM killer.
    import psutil
    total_ram = psutil.virtual_memory().total
    mem_limit = int(total_ram * 0.8)
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

    wt = tuple(work_types) if work_types else tuple(ALL_WORK_TYPES)

    manifest = read_job_manifest(job_id)
    manifest["pid"] = os.getpid()
    write_job_manifest(job_id, manifest)

    def _on_sigterm(signum: int, frame: Any) -> None:
        print(f"\nWorker killed by signal {signum}", flush=True)
        m = read_job_manifest(job_id)
        m["status"] = "failed"
        m["error"] = f"killed by signal {signum}"
        m["finished_at"] = datetime.datetime.now().isoformat()
        write_job_manifest(job_id, m)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        parts = scope_str.split("/")
        if action == "fetch":
            if len(parts) == 4:
                scope = ResolvedScope(field=parts[0], subfield=parts[1], topic=parts[2], work_type=parts[3])
                fetch_works(scope)
            elif len(parts) == 2:
                filt = Filter(field=parts[0], subfield=parts[1], work_types=wt)
                fetch_all_scopes(filt)
            elif len(parts) == 1:
                if any(d[0] == parts[0] for d in KNOWN_DOMAINS):
                    filt = Filter(domain=parts[0], work_types=wt)
                else:
                    filt = Filter(field=parts[0], work_types=wt)
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
        elif action == "snapshot":
            from bookgap.snapshot_ingest import run as snapshot_run
            from bookgap.configs.config import SnapshotConfig

            if len(parts) == 4:
                filt = Filter(field=parts[0], subfield=parts[1], topic=parts[2], work_types=(parts[3],))
            elif len(parts) == 3:
                filt = Filter(field=parts[0], subfield=parts[1], topic=parts[2], work_types=wt)
            elif len(parts) == 2:
                filt = Filter(field=parts[0], subfield=parts[1], work_types=wt)
            elif len(parts) == 1:
                if any(d[0] == parts[0] for d in KNOWN_DOMAINS):
                    filt = Filter(domain=parts[0], work_types=wt)
                else:
                    filt = Filter(field=parts[0], work_types=wt)
            else:
                raise ValueError(f"Unexpected scope format for snapshot: {scope_str}")

            # NOTE: We instantiate SnapshotConfig directly (not via Hydra compose) so
            # scope_manager can be invoked without the full experiment config stack.
            # Any Hydra YAML overrides for snapshot.* are ignored by this path — the
            # only override is the --snapshot-date CLI flag.
            cfg = SnapshotConfig()
            effective_date = snapshot_date or cfg.snapshot_date
            snapshot_run(
                work_filter=filt,
                snapshot_date=effective_date,
                bucket=cfg.bucket,
                prefix=cfg.prefix,
                shard_row_flush_threshold=cfg.shard_row_flush_threshold,
                max_resident_rows=cfg.max_resident_rows,
                disk_abort_pct=cfg.disk_abort_pct,
            )
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
