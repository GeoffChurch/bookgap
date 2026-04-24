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
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import bookgap.openalex_ingest as _ingest_mod


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOBS_DIR_NAME = ".jobs"
MAX_CONCURRENT_WORKERS = 3

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
    """Atomically write a job manifest to disk (temp file + os.replace)."""
    dest = manifest_path(job_id)
    tmp = dest.with_suffix(".tmp")
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


def stop_requested() -> bool:
    """Poll whether SIGTERM/SIGINT has been received in the current worker.

    Only meaningful inside a :func:`worker_lifecycle` context; returns
    ``False`` outside one. Long inner loops (e.g., per-scope embedding,
    per-shard ingest) should poll this between iterations and exit
    cleanly when it flips, so on-disk state stays consistent.
    """
    return _STOP_REQUESTED


def _make_stop_handler(job_id: str) -> Callable[[int, Any], None]:
    def _handler(signum: int, frame: Any) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        # Best-effort manifest status update so external observers
        # see "interrupted" immediately, even before the worker's
        # main body notices the flag. If this fails (e.g., disk
        # issue), we still let the flag do its job.
        try:
            update_manifest(
                job_id,
                status=STATUS_INTERRUPTED,
                error=f"stop requested via signal {signum}",
            )
        except Exception:
            pass
    return _handler


@contextmanager
def worker_lifecycle(job_id: str) -> Iterator[None]:
    """Wrap a worker's main() body to get manifest-driven lifecycle.

    On entry, records the current pid in the manifest and installs
    SIGTERM/SIGINT handlers that set :func:`stop_requested` and mark the
    manifest ``interrupted``. On normal exit the manifest becomes
    ``complete``; on ``KeyboardInterrupt`` (from either signal or a
    manual Ctrl-C) it stays ``interrupted``; any other exception yields
    ``failed`` with the exception message recorded.

    Usage::

        def main(job_id):
            with worker_lifecycle(job_id):
                do_work()  # may poll jobs.stop_requested() in inner loops
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = False

    update_manifest(job_id, pid=os.getpid())

    handler = _make_stop_handler(job_id)
    old_term = signal.signal(signal.SIGTERM, handler)
    old_int = signal.signal(signal.SIGINT, handler)
    now = lambda: datetime.datetime.now().isoformat()
    try:
        try:
            yield
        except BaseException as exc:
            # A stop signal can race with other downstream exceptions
            # (e.g. a subprocess exiting non-zero after receiving the
            # cgroup-wide SIGTERM). Whatever exception surfaces, treat
            # the exit as 'interrupted' when a stop was requested so
            # the manifest reflects user intent rather than the
            # propagated symptom.
            if _STOP_REQUESTED:
                status, error = STATUS_INTERRUPTED, str(exc) or "interrupted"
            else:
                status, error = STATUS_FAILED, f"{type(exc).__name__}: {exc}"
            update_manifest(job_id, status=status, error=error, finished_at=now())
            raise
        else:
            update_manifest(job_id, status=STATUS_COMPLETE, finished_at=now())
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


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

    args = parser.parse_args(argv)
    # argparse.REMAINDER sometimes keeps the separating "--" as the first
    # positional; strip it for convenience.
    if hasattr(args, "command") and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(_main())
