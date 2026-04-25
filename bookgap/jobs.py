"""Generic long-running-job plumbing: systemd-run launch + JSON manifests.

Used by:
- ``bookgap.scope_ops`` for taxonomy-scoped workers (fetch/snapshot/embed),
- ``bookgap.jobs`` CLI (``python -m bookgap.jobs``) for arbitrary pipelines,
- ``scope_manager`` TUI/CLI that surfaces jobs from the shared manifest dir.

The design: every long-running job is launched inside a ``systemd-run
--user --scope`` cgroup with a predictable unit name, logs to
``cache/.jobs/<job_id>.log``, and writes a JSON manifest at
``cache/.jobs/<job_id>.json``. Manifests track pid, scope_unit, status
(running/complete/failed/crashed/interrupted), timestamps, and any
caller-supplied metadata. ``systemctl --user stop`` on the scope unit
kills the whole process tree cleanly; worker_lifecycle() converts that
into an ``interrupted`` status while preserving partial state on disk.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import bookgap.openalex_ingest as _ingest_mod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOBS_DIR_NAME = ".jobs"
MAX_CONCURRENT_WORKERS = 3

# Serialise manifest writes within a process so the watchdog thread and
# the main worker thread can't collide on the shared ``<id>.tmp`` rename.
_manifest_lock = threading.Lock()

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_CRASHED = "crashed"
STATUS_INTERRUPTED = "interrupted"

_FINISHED_STATUSES = frozenset({STATUS_COMPLETE, STATUS_FAILED, STATUS_CRASHED, STATUS_INTERRUPTED})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def jobs_dir() -> Path:
    """Return the jobs directory, creating it if needed."""
    d = _ingest_mod.CACHE_DIR / JOBS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def log_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.log"


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def write_manifest(job_id: str, manifest: dict[str, Any]) -> None:
    """Atomically write a job manifest to disk (temp file + os.replace).

    Held under ``_manifest_lock`` so the watchdog thread can update
    the manifest concurrently with the main worker thread without
    racing on the shared ``<id>.tmp`` filename.
    """
    dest = manifest_path(job_id)
    tmp = dest.with_suffix(".tmp")
    with _manifest_lock:
        tmp.write_text(json.dumps(manifest, indent=2, default=str))
        os.replace(tmp, dest)


def read_manifest(job_id: str) -> dict[str, Any]:
    """Read a job manifest by ID."""
    return json.loads(manifest_path(job_id).read_text())


def update_manifest(job_id: str, **fields: Any) -> dict[str, Any]:
    """Read, merge `fields` in, atomically rewrite, return updated manifest."""
    m = read_manifest(job_id)
    m.update(fields)
    write_manifest(job_id, m)
    return m


# ---------------------------------------------------------------------------
# Process liveness + discovery
# ---------------------------------------------------------------------------

def is_pid_alive(pid: int) -> bool:
    """Return True if `pid` exists and we can signal it (signal 0 probe)."""
    if pid is None or pid < 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def discover_jobs() -> list[dict[str, Any]]:
    """Scan manifests, mark dead-pid 'running' jobs as 'crashed', return all.

    Each manifest has ``_job_id`` injected from its filename (not persisted
    to disk). The crashed-detection write-back means ``discover_jobs`` is
    mildly stateful, but keeping the authoritative-status invariant in one
    place is cheaper than audit trails.
    """
    results: list[dict[str, Any]] = []
    for path in sorted(jobs_dir().glob("*.json")):
        try:
            manifest = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        manifest["_job_id"] = path.stem
        if manifest.get("status") == STATUS_RUNNING and not is_pid_alive(manifest.get("pid", -1)):
            manifest["status"] = STATUS_CRASHED
            write_manifest(path.stem, {k: v for k, v in manifest.items() if k != "_job_id"})
        results.append(manifest)
    return results


def running_count() -> int:
    return sum(1 for j in discover_jobs() if j.get("status") == STATUS_RUNNING)


def at_capacity(max_concurrent: int = MAX_CONCURRENT_WORKERS) -> bool:
    return running_count() >= max_concurrent


def has_running_match(predicate: Callable[[dict[str, Any]], bool]) -> bool:
    """True if any running job's manifest satisfies `predicate`."""
    return any(
        j.get("status") == STATUS_RUNNING and predicate(j)
        for j in discover_jobs()
    )


def dismiss_finished_jobs() -> int:
    """Delete manifest + log for every job whose status is terminal.

    Returns the number of jobs dismissed.
    """
    dismissed = 0
    for job in discover_jobs():
        if job.get("status") in _FINISHED_STATUSES:
            jid = job["_job_id"]
            manifest_path(jid).unlink(missing_ok=True)
            log_path(jid).unlink(missing_ok=True)
            dismissed += 1
    return dismissed


# ---------------------------------------------------------------------------
# Launch / stop
# ---------------------------------------------------------------------------

def scope_unit_name(job_id: str) -> str:
    """Canonical systemd scope unit for a given job_id."""
    return f"bookgap-{job_id}.scope"


class JobCapacityError(RuntimeError):
    """Raised when running_count is already at the configured cap."""


def launch(
    *,
    job_id: str,
    command: list[str],
    metadata: Optional[dict[str, Any]] = None,
    log_path_override: Optional[Path] = None,
    env_extra: Optional[dict[str, str]] = None,
    enforce_capacity: bool = True,
    max_concurrent: int = MAX_CONCURRENT_WORKERS,
) -> dict[str, Any]:
    """Launch `command` in a transient systemd user scope.

    Writes ``cache/.jobs/<job_id>.json`` manifest and redirects the
    worker's stdout/stderr to ``cache/.jobs/<job_id>.log`` (or
    ``log_path_override``). Returns the written manifest.

    The systemd scope ensures the whole process tree (including any
    ``conda run`` wrapper that forks a child) shares one cgroup, so
    ``systemctl --user stop <scope_unit>`` cleanly kills everything.
    The scope also survives SSH logout — plain double-fork / setsid
    doesn't escape logind's session-N.scope.

    Callers are responsible for their own duplicate-detection policy
    (e.g., ``has_running_match``). The only built-in check is a
    concurrency cap, disabled by passing ``enforce_capacity=False``.
    """
    if enforce_capacity and at_capacity(max_concurrent):
        raise JobCapacityError(
            f"already at {max_concurrent} concurrent running jobs"
        )

    scope_unit = scope_unit_name(job_id)
    log_target = log_path_override or log_path(job_id)
    log_target.parent.mkdir(parents=True, exist_ok=True)

    systemd_cmd = [
        "systemd-run", "--user", "--scope", "--quiet", "--collect",
        f"--unit={scope_unit}",
        *command,
    ]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env_extra:
        env.update(env_extra)

    log_fd = os.open(str(log_target), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        proc = subprocess.Popen(
            systemd_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=log_fd,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_fd)

    manifest: dict[str, Any] = {
        "pid": proc.pid,
        "scope_unit": scope_unit,
        "status": STATUS_RUNNING,
        "started_at": datetime.datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
    }
    if metadata:
        # Caller-supplied metadata is merged in but cannot overwrite
        # reserved bookkeeping keys.
        reserved = set(manifest.keys())
        for k, v in metadata.items():
            if k in reserved:
                continue
            manifest[k] = v
    write_manifest(job_id, manifest)
    return {**manifest, "_job_id": job_id}


def stop(job_id: str, *, timeout: float = 10.0) -> None:
    """Gracefully stop a running job by sending SIGTERM to its scope cgroup.

    Returns when ``systemctl stop`` completes (which itself waits for the
    process tree to exit or escalates to SIGKILL). Workers that use
    :func:`worker_lifecycle` convert the resulting SIGTERM into an
    ``interrupted`` status; other workers will typically exit 15 and be
    marked ``failed``.
    """
    manifest = read_manifest(job_id)
    scope_unit = manifest.get("scope_unit") or scope_unit_name(job_id)
    subprocess.run(
        ["systemctl", "--user", "stop", scope_unit],
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Worker-side lifecycle
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False
# First reason captured when ``_STOP_REQUESTED`` flipped to True. Subsequent
# attempts to set it preserve the original cause (e.g., a watchdog disk-abort
# that triggered a Ctrl-C cascade should keep "disk 91.2%" as the reason,
# not the SIGINT that followed). Reset on entry to ``worker_lifecycle``.
_STOP_REASON: Optional[str] = None


def stop_requested() -> bool:
    """Poll whether a stop has been requested in the current worker.

    Sources include SIGTERM/SIGINT signal delivery and the
    ``worker_lifecycle`` watchdog (disk-usage / RSS thresholds). Only
    meaningful inside a :func:`worker_lifecycle` context; returns
    ``False`` outside one. Long inner loops (e.g., per-scope embedding,
    per-shard ingest, chunked corpus reads) should poll this between
    iterations and exit cleanly when it flips, so on-disk state stays
    consistent and the lifecycle records the cause.
    """
    return _STOP_REQUESTED


def stop_reason() -> Optional[str]:
    """The first cause recorded when ``_STOP_REQUESTED`` was set.

    Captured by the signal handler or the watchdog thread; consumed by
    ``worker_lifecycle`` when writing the manifest's error field.
    """
    return _STOP_REASON


def _request_stop(reason: str) -> None:
    """Idempotent stop-request helper. Sets the flag + first-cause reason."""
    global _STOP_REQUESTED, _STOP_REASON
    _STOP_REQUESTED = True
    if _STOP_REASON is None:
        _STOP_REASON = reason


def _make_stop_handler(job_id: str) -> Callable[[int, Any], None]:
    def _handler(signum: int, frame: Any) -> None:
        _request_stop(f"stop requested via signal {signum}")
        # Best-effort manifest status update so external observers
        # see "interrupted" immediately, even before the worker's
        # main body notices the flag. If this fails (e.g., disk
        # issue), we still let the flag do its job.
        try:
            update_manifest(job_id, status=STATUS_INTERRUPTED, error=_STOP_REASON)
        except Exception:
            pass
    return _handler


def _start_watchdog(
    job_id: str,
    *,
    mem_limit_pct: Optional[float],
    disk_abort_pct: Optional[float],
    interval: float,
) -> tuple[Optional[threading.Thread], threading.Event]:
    """Start a daemon watchdog that polls disk + RSS and requests stop on
    threshold crossings. Returns (thread, stop_event); caller must
    ``stop_event.set()`` and join the thread on lifecycle exit.

    Either threshold can be disabled by passing ``None`` / 0.
    """
    stop_event = threading.Event()
    if interval <= 0 or (not mem_limit_pct and not disk_abort_pct):
        return None, stop_event

    try:
        import psutil
    except ImportError:
        return None, stop_event

    cache_dir = _ingest_mod.CACHE_DIR
    rss_warn_pct = (mem_limit_pct * 100 * 0.95) if mem_limit_pct else None
    proc = psutil.Process()

    def _watchdog() -> None:
        # Run until the lifecycle wakes us or we trip a threshold.
        while not stop_event.wait(interval):
            try:
                if disk_abort_pct:
                    d = shutil.disk_usage(cache_dir)
                    used_pct = 100.0 * d.used / d.total
                    if used_pct >= disk_abort_pct:
                        reason = f"watchdog: disk {used_pct:.1f}% >= {disk_abort_pct:.1f}%"
                        _request_stop(reason)
                        try:
                            update_manifest(job_id, error=reason)
                        except Exception:
                            pass
                        return
                if rss_warn_pct:
                    rss = proc.memory_info().rss
                    rss_pct = 100.0 * rss / psutil.virtual_memory().total
                    if rss_pct >= rss_warn_pct:
                        reason = f"watchdog: rss {rss_pct:.1f}% >= {rss_warn_pct:.1f}% (95% of RLIMIT_AS)"
                        _request_stop(reason)
                        try:
                            update_manifest(job_id, error=reason)
                        except Exception:
                            pass
                        return
            except Exception:
                # Don't kill a long-running job because of a transient
                # /proc read or disk-usage stat error in the watchdog.
                continue

    t = threading.Thread(target=_watchdog, daemon=True, name=f"jobs-watchdog-{job_id}")
    t.start()
    return t, stop_event


@contextmanager
def worker_lifecycle(
    job_id: str,
    *,
    mem_limit_pct: Optional[float] = 0.80,
    disk_abort_pct: Optional[float] = 90.0,
    watchdog_interval: float = 10.0,
) -> Iterator[None]:
    """Wrap a worker's main() body to get manifest-driven lifecycle.

    On entry, records the current pid in the manifest, installs
    SIGTERM/SIGINT handlers that set :func:`stop_requested`, sets
    ``RLIMIT_AS`` to ``mem_limit_pct`` of total RAM (so OOM conditions
    raise a catchable ``MemoryError`` rather than getting an
    uncatchable kernel SIGKILL), and starts a daemon watchdog thread
    that polls disk usage + process RSS every ``watchdog_interval``
    seconds and requests a graceful stop if either crosses
    ``disk_abort_pct`` or 95% of the RLIMIT_AS ceiling.

    Inner loops should poll :func:`stop_requested` between iterations
    so the soft-abort actually fires before the kernel ceiling does.

    Lifecycle outcomes:
        - normal return → manifest ``complete``
        - ``KeyboardInterrupt`` (signal or watchdog) → ``interrupted``,
          error = the first :func:`stop_reason` captured
        - any other exception → ``failed``, error = type+message

    Pass ``mem_limit_pct=None`` or ``disk_abort_pct=None`` to disable
    either threshold; pass ``watchdog_interval<=0`` to disable the
    watchdog entirely (useful in tests).

    Usage::

        def main(job_id):
            with worker_lifecycle(job_id):
                do_work()  # may poll jobs.stop_requested() in inner loops
    """
    global _STOP_REQUESTED, _STOP_REASON
    _STOP_REQUESTED = False
    _STOP_REASON = None

    update_manifest(job_id, pid=os.getpid())

    if mem_limit_pct:
        try:
            import resource
            import psutil  # type: ignore[import-untyped]
            total_ram = psutil.virtual_memory().total
            mem_limit = int(total_ram * mem_limit_pct)
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        except (ImportError, ValueError, OSError):
            # resource is Unix-only; psutil may be missing; setrlimit can
            # fail under some sandboxed environments. None of these
            # should kill the run — we just lose the RLIMIT_AS backstop.
            pass

    watchdog_thread, stop_event = _start_watchdog(
        job_id,
        mem_limit_pct=mem_limit_pct,
        disk_abort_pct=disk_abort_pct,
        interval=watchdog_interval,
    )

    handler = _make_stop_handler(job_id)
    old_term = signal.signal(signal.SIGTERM, handler)
    old_int = signal.signal(signal.SIGINT, handler)
    now = lambda: datetime.datetime.now().isoformat()
    try:
        try:
            yield
        except BaseException as exc:
            # A stop signal/watchdog-trip can race with other downstream
            # exceptions (e.g. a subprocess exiting non-zero after
            # receiving the cgroup-wide SIGTERM). Whatever exception
            # surfaces, treat the exit as 'interrupted' when a stop was
            # requested so the manifest reflects the actual cause
            # (signal vs disk vs memory) rather than the propagated
            # symptom.
            if _STOP_REQUESTED:
                status = STATUS_INTERRUPTED
                error = _STOP_REASON or str(exc) or "interrupted"
            else:
                status = STATUS_FAILED
                error = f"{type(exc).__name__}: {exc}"
            update_manifest(job_id, status=status, error=error, finished_at=now())
            raise
        else:
            update_manifest(job_id, status=STATUS_COMPLETE, finished_at=now())
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        stop_event.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=5)


__all__ = [
    # Constants
    "JOBS_DIR_NAME",
    "MAX_CONCURRENT_WORKERS",
    "STATUS_RUNNING",
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_CRASHED",
    "STATUS_INTERRUPTED",
    # Paths
    "jobs_dir",
    "manifest_path",
    "log_path",
    "scope_unit_name",
    # Manifest I/O
    "write_manifest",
    "read_manifest",
    "update_manifest",
    # Discovery
    "is_pid_alive",
    "discover_jobs",
    "running_count",
    "at_capacity",
    "has_running_match",
    "dismiss_finished_jobs",
    # Launch/stop
    "JobCapacityError",
    "launch",
    "stop",
    # Worker-side
    "stop_reason",
    "stop_requested",
    "worker_lifecycle",
]


# ---------------------------------------------------------------------------
# CLI: ``python -m bookgap.jobs {run,list,stop,log,_exec}``
# ---------------------------------------------------------------------------


def _fmt_duration(started_at: Optional[str], finished_at: Optional[str]) -> str:
    if not started_at:
        return "?"
    start = datetime.datetime.fromisoformat(started_at)
    end = datetime.datetime.fromisoformat(finished_at) if finished_at else datetime.datetime.now()
    secs = int((end - start).total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _cmd_run(args: Any) -> int:
    """Launch `args.command` under a new systemd scope tracked by a manifest.

    The inner process is this same CLI invoked with ``_exec`` so it gets
    :func:`worker_lifecycle` around the user's command — meaning normal
    exit → ``complete``, SIGTERM/SIGINT → ``interrupted``, any other exit
    code → ``failed``.
    """
    if not args.command:
        print("error: no command given after `--`", file=sys.stderr)
        return 2
    job_id = f"{args.prefix}_{int(datetime.datetime.now().timestamp())}"
    exec_cmd = [
        sys.executable, "-m", "bookgap.jobs", "_exec",
        "--job-id", job_id, "--",
        *args.command,
    ]
    try:
        manifest = launch(
            job_id=job_id,
            command=exec_cmd,
            metadata={"prefix": args.prefix, "command": args.command},
            enforce_capacity=not args.no_cap,
        )
    except JobCapacityError as e:
        print(f"error: {e} (use --no-cap to override)", file=sys.stderr)
        return 1
    print(f"job_id:     {job_id}")
    print(f"scope:      {manifest['scope_unit']}")
    print(f"log:        {log_path(job_id)}")
    print(f"stop with:  python -m bookgap.jobs stop {job_id}")
    return 0


def _cmd_exec(args: Any) -> int:
    """Internal: wrap `args.command` in worker_lifecycle and exec it."""
    with worker_lifecycle(args.job_id):
        rc = subprocess.call(args.command)
        if rc != 0:
            raise SystemExit(rc)
    return 0


def _cmd_list(args: Any) -> int:
    rows = discover_jobs()
    if not rows:
        print("(no jobs)")
        return 0
    print(f"{'JOB_ID':<50} {'STATUS':<12} {'AGE':<10} {'PID':<8}")
    for m in rows:
        print(
            f"{m['_job_id']:<50} "
            f"{m.get('status', '?'):<12} "
            f"{_fmt_duration(m.get('started_at'), m.get('finished_at')):<10} "
            f"{m.get('pid', '?'):<8}"
        )
    return 0


def _cmd_stop(args: Any) -> int:
    try:
        stop(args.job_id)
    except FileNotFoundError:
        print(f"error: no manifest for job_id={args.job_id!r}", file=sys.stderr)
        return 1
    print(f"sent stop to {scope_unit_name(args.job_id)}")
    return 0


def _cmd_log(args: Any) -> int:
    path = log_path(args.job_id)
    if not path.exists():
        print(f"error: no log at {path}", file=sys.stderr)
        return 1
    if args.follow:
        os.execvp("tail", ["tail", "-n", str(args.tail), "-f", str(path)])
    else:
        os.execvp("tail", ["tail", "-n", str(args.tail), str(path)])


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_age_seconds(secs: float) -> str:
    if secs < 60:
        return f"{int(secs)}s"
    if secs < 3600:
        return f"{int(secs // 60)}m{int(secs % 60):02d}s"
    return f"{int(secs // 3600)}h{int((secs % 3600) // 60):02d}m"


def _systemd_show(scope_unit: str) -> dict[str, str]:
    """Read selected systemd properties for a transient scope unit."""
    try:
        out = subprocess.check_output(
            ["systemctl", "--user", "show", scope_unit,
             "--property=MemoryCurrent",
             "--property=MemoryPeak",
             "--property=CPUUsageNSec",
             "--property=ActiveEnterTimestamp",
             "--property=SubState"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    result: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k] = v
    return result


def _find_recent_log(glob_root: Path, pattern: str, started_at: datetime.datetime,
                    window_min: float = 5.0) -> Optional[Path]:
    """Find the most recent file under ``glob_root`` matching ``pattern``
    whose mtime falls within the job's window (started_at − window_min,
    now). Heuristic: assumes Hydra/curriculum-style logs that are written
    while the job runs, and that no other process touches them around
    the same time. Returns None if no candidate is found."""
    if not glob_root.exists():
        return None
    earliest = (started_at - datetime.timedelta(minutes=window_min)).timestamp()
    candidates = []
    for p in glob_root.rglob(pattern):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= earliest:
            candidates.append((m, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _tail_lines(path: Path, n: int) -> list[str]:
    """Return the last ``n`` lines of ``path``. Best-effort; returns []
    on any error so the status display degrades gracefully."""
    try:
        with path.open("r", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip() for ln in lines[-n:]]
    except OSError:
        return []


def _cmd_status(args: Any) -> int:
    """One-stop-shop status display for a single job.

    Combines: manifest (status, error, started/finished), wall-time age,
    systemd resource snapshot (memory, CPU), worker-log tail with stall
    detection, and best-effort auto-discovery of the Hydra/curriculum
    log files associated with the run.
    """
    job_id = args.job_id
    try:
        manifest = read_manifest(job_id)
    except FileNotFoundError:
        print(f"error: no manifest for job_id={job_id!r}", file=sys.stderr)
        return 1

    print(f"job_id:       {job_id}")
    status = manifest.get("status", "?")
    print(f"status:       {status}")
    started_str = manifest.get("started_at")
    started_dt: Optional[datetime.datetime] = None
    if started_str:
        try:
            started_dt = datetime.datetime.fromisoformat(started_str)
        except Exception:
            pass
    age = _fmt_duration(started_str, manifest.get("finished_at"))
    print(f"started:      {started_str or '?'}  ({age} ago)")
    if manifest.get("finished_at"):
        print(f"finished:     {manifest['finished_at']}")
    if manifest.get("error"):
        print(f"error:        {manifest['error']}")
    print(f"pid:          {manifest.get('pid', '?')}")
    scope_unit = manifest.get("scope_unit") or scope_unit_name(job_id)
    print(f"scope unit:   {scope_unit}")

    # Resource snapshot via systemd
    sysd = _systemd_show(scope_unit)
    sub_state = sysd.get("SubState", "")
    if sysd:
        mem_cur = sysd.get("MemoryCurrent", "")
        mem_peak = sysd.get("MemoryPeak", "")
        cpu_ns = sysd.get("CPUUsageNSec", "")
        print()
        print("resources:")
        if sub_state:
            print(f"  unit state:   {sub_state}")
        if mem_cur and mem_cur.isdigit():
            print(f"  memory cur:   {_fmt_bytes(int(mem_cur))}")
        if mem_peak and mem_peak.isdigit():
            print(f"  memory peak:  {_fmt_bytes(int(mem_peak))}")
        if cpu_ns and cpu_ns.isdigit():
            cpu_secs = int(cpu_ns) / 1e9
            print(f"  cpu time:     {_fmt_age_seconds(cpu_secs)}")

    # Worker log + stall detection
    wlog = log_path(job_id)
    print()
    print(f"worker log ({wlog}):")
    if wlog.exists():
        try:
            mtime = wlog.stat().st_mtime
            age_log = (datetime.datetime.now().timestamp() - mtime)
            tail = _tail_lines(wlog, args.tail_n)
            if tail:
                for ln in tail:
                    print(f"  {ln[:160]}")
            else:
                print("  (empty — likely buffered by `conda run`; see auto-discovered logs below)")
            print(f"  mtime: {_fmt_age_seconds(age_log)} ago")
        except OSError as exc:
            print(f"  (read error: {exc})")
    else:
        print("  (no log file)")

    # Auto-discover Hydra + curriculum logs by mtime-within-job-window
    if started_dt is not None:
        from bookgap.util import CACHE_DIR
        repo_root = CACHE_DIR.parent
        for label, root, pattern in [
            ("hydra log", repo_root / "outputs", "main.log"),
            ("curriculum pipeline log",
             repo_root / "artifacts" / "curriculum", "pipeline.log"),
        ]:
            found = _find_recent_log(root, pattern, started_dt)
            if found is None:
                continue
            print()
            print(f"{label} ({found}):")
            mtime = found.stat().st_mtime
            age_log = (datetime.datetime.now().timestamp() - mtime)
            for ln in _tail_lines(found, args.tail_n):
                print(f"  {ln[:160]}")
            print(f"  mtime: {_fmt_age_seconds(age_log)} ago")
    return 0


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="bookgap.jobs", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="launch a command as a tracked job")
    p_run.add_argument("--prefix", required=True, help="short label for the job_id")
    p_run.add_argument("--no-cap", action="store_true",
                       help=f"bypass the {MAX_CONCURRENT_WORKERS}-concurrent-job cap")
    p_run.add_argument("command", nargs=argparse.REMAINDER,
                       help="command and args (put after `--` to be safe)")
    p_run.set_defaults(func=_cmd_run)

    p_exec = sub.add_parser("_exec", help=argparse.SUPPRESS)
    p_exec.add_argument("--job-id", required=True)
    p_exec.add_argument("command", nargs=argparse.REMAINDER)
    p_exec.set_defaults(func=_cmd_exec)

    p_list = sub.add_parser("list", help="list all jobs")
    p_list.set_defaults(func=_cmd_list)

    p_stop = sub.add_parser("stop", help="gracefully stop a running job")
    p_stop.add_argument("job_id")
    p_stop.set_defaults(func=_cmd_stop)

    p_log = sub.add_parser("log", help="tail a job's log")
    p_log.add_argument("job_id")
    p_log.add_argument("-f", "--follow", action="store_true")
    p_log.add_argument("-n", "--tail", type=int, default=200)
    p_log.set_defaults(func=_cmd_log)

    p_status = sub.add_parser(
        "status",
        help="one-stop-shop view: manifest + resources + log tails (auto-discovered)",
    )
    p_status.add_argument("job_id")
    p_status.add_argument("-n", "--tail-n", type=int, default=10,
                          help="how many lines from each log to show")
    p_status.set_defaults(func=_cmd_status)

    args = parser.parse_args(argv)
    # argparse.REMAINDER sometimes keeps the separating "--" as the first
    # positional; strip it for convenience.
    if hasattr(args, "command") and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(_main())
