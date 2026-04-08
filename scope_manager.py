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
pyalex.config.email = "gchurchill@cs.stonybrook.edu"
import pyarrow.parquet as pq
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import DataTable, OptionList, Static
from textual.widgets.option_list import Option

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


def launch_worker(
    action: str, scope_str: str, model: str | None = None, work_types: list[str] | None = None,
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

    cmd = [sys.executable, os.path.abspath(__file__), "--worker", action, scope_str]
    if model:
        cmd.extend(["--model", model])
    if work_types:
        cmd.extend(["--work-types", ",".join(work_types)])
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


def run_worker(
    action: str, scope_str: str, model: str | None, job_id: str, work_types: list[str] | None = None,
) -> None:
    """Worker entry point — called in the detached subprocess."""
    from bookgap.openalex_ingest import fetch_works, fetch_all_scopes, embed_works
    from bookgap.util import Filter

    wt = tuple(work_types) if work_types else tuple(ALL_WORK_TYPES)

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
# Aggregate status helpers
# ---------------------------------------------------------------------------

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


def resolve_scopes_for_node(
    level: int, path: list[tuple[str, str]], node_id: str, work_types: list[str],
) -> list[ResolvedScope]:
    """Resolve all concrete scopes under a given hierarchy node."""
    if level == 0:
        fields = get_domain_fields(node_id)
        scopes: list[ResolvedScope] = []
        for field_id, _ in fields:
            subfields = get_field_subfields(field_id)
            for sf_id, _ in subfields:
                topics = get_subfield_topics_named(sf_id)
                for topic_id, _ in topics:
                    for wt in work_types:
                        scopes.append(ResolvedScope(field=field_id, subfield=sf_id, topic=topic_id, work_type=wt))
        return scopes
    elif level == 1:
        field_id = node_id
        subfields = get_field_subfields(field_id)
        scopes = []
        for sf_id, _ in subfields:
            topics = get_subfield_topics_named(sf_id)
            for topic_id, _ in topics:
                for wt in work_types:
                    scopes.append(ResolvedScope(field=field_id, subfield=sf_id, topic=topic_id, work_type=wt))
        return scopes
    elif level == 2:
        field_id = path[1][0]
        sf_id = node_id
        topics = get_subfield_topics_named(sf_id)
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
# Known domains
# ---------------------------------------------------------------------------

KNOWN_DOMAINS: list[tuple[str, str]] = [
    ("3", "Physical Sciences"),
]


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
# Textual TUI app
# ---------------------------------------------------------------------------

class ModelSelectScreen(ModalScreen[str | None]):
    """Modal popup for selecting an embedding model."""

    CSS = """
    ModelSelectScreen {
        align: center middle;
    }
    #model_list {
        width: 50;
        height: 8;
        border: solid $primary;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        yield OptionList(
            Option("GIST-small (384d)", id="gist"),
            Option("Gemma-300M (768d)", id="gemma"),
            id="model_list",
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        model_map = {
            "gist": Embedder.ModelName.GIST_SMALL.value,
            "gemma": Embedder.ModelName.GEMMA_300M.value,
        }
        if event.option.id is not None:
            self.dismiss(model_map.get(event.option.id))

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


class ScopeManagerApp(App):
    """TUI for browsing OpenAlex scopes, viewing status, and launching fetch/embed jobs."""

    TITLE = "Scope Manager"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "fetch", "Fetch"),
        Binding("e", "embed", "Embed"),
        Binding("d", "dismiss_jobs", "Dismiss"),
        Binding("l", "view_log", "Log"),
        Binding("1", "toggle_type_1", "article", show=False),
        Binding("2", "toggle_type_2", "book", show=False),
        Binding("3", "toggle_type_3", "book-chapter", show=False),
        Binding("4", "toggle_type_4", "preprint", show=False),
    ]

    CSS = """
    #breadcrumb {
        dock: top;
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }
    #resource_bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    #status_bar {
        dock: bottom;
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }
    #key_hints {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    DataTable {
        height: 1fr;
    }
    """

    def __init__(self):
        super().__init__()
        self._level: int = 0  # 0=domains, 1=fields, 2=subfields, 3=topics
        self._path: list[tuple[str, str]] = []  # [(id, name), ...] of navigation selections
        self._active_types: list[bool] = [True, True, True, True]
        self._rows: list[tuple[str, str]] = []  # [(id, display_name), ...]

    def compose(self) -> ComposeResult:
        yield Static("Scope Manager", id="breadcrumb")
        yield DataTable(id="scope_table")
        yield Static("", id="key_hints")
        yield Static("", id="status_bar")
        yield Static("", id="resource_bar")

    def on_mount(self) -> None:
        table = self.query_one("#scope_table", DataTable)
        table.cursor_type = "row"
        self._update_resource_bar()
        self.set_interval(3, self._update_resource_bar)
        self.set_interval(3, self._update_status_bar)
        self._render_current_level()

    def _update_resource_bar(self) -> None:
        self.query_one("#resource_bar", Static).update(get_system_resources())

    def _update_status_bar(self) -> None:
        from rich.markup import escape
        type_parts: list[str] = []
        for i, wt in enumerate(ALL_WORK_TYPES):
            if self._active_types[i]:
                type_parts.append(escape(f"[{wt}]"))
            else:
                type_parts.append(escape(wt))
        types_str = " ".join(type_parts)

        jobs = discover_jobs()
        running = sum(1 for j in jobs if j["status"] == "running")
        failed = sum(1 for j in jobs if j["status"] in ("failed", "crashed"))
        job_parts: list[str] = []
        if running:
            job_parts.append(f"{running} running")
        if failed:
            job_parts.append(f"{failed} failed")
        jobs_str = ", ".join(job_parts) if job_parts else "no jobs"

        self.query_one("#status_bar", Static).update(f"Types: {types_str} | Jobs: {jobs_str}")
        self.query_one("#key_hints", Static).update(
            escape("[f]etch [e]mbed [r]efresh [d]ismiss [l]og [1-4]toggle types [q]uit")
        )

    def _breadcrumb_text(self) -> str:
        parts = [name for _, name in self._path]
        if not parts:
            return "Domains"
        return " > ".join(parts)

    def _active_type_names(self) -> list[str]:
        return [ALL_WORK_TYPES[i] for i in range(4) if self._active_types[i]]

    def _render_current_level(self) -> None:
        self.query_one("#breadcrumb", Static).update(self._breadcrumb_text())
        table = self.query_one("#scope_table", DataTable)
        table.clear(columns=True)

        active_types = self._active_type_names()

        # Determine rows for this level
        if self._level == 0:
            self._rows = KNOWN_DOMAINS
        elif self._level == 1:
            domain_id = self._path[0][0]
            self._rows = get_domain_fields(domain_id)
        elif self._level == 2:
            field_id = self._path[1][0]
            self._rows = get_field_subfields(field_id)
        elif self._level == 3:
            subfield_id = self._path[2][0]
            self._rows = get_subfield_topics_named(subfield_id)

        # Build columns: Name + per-work-type status columns
        table.add_column("Name", key="name", width=30)
        for wt in active_types:
            table.add_column(f"{wt[:8]} fetch", key=f"{wt}_fetch", width=12)
            table.add_column(f"{wt[:8]} GIST", key=f"{wt}_gist", width=8)
            table.add_column(f"{wt[:8]} Gem", key=f"{wt}_gemma", width=8)

        # Populate rows
        for node_id, name in self._rows:
            if self._level == 3:
                # Topic level: per-work-type direct status
                cells: list[str] = [name]
                for wt in active_types:
                    scope = ResolvedScope(
                        field=self._path[1][0],
                        subfield=self._path[2][0],
                        topic=node_id,
                        work_type=wt,
                    )
                    status = get_scope_status(scope)
                    if status["fetched"]:
                        cells.append(str(status["work_count"]))
                    else:
                        cells.append("-")
                    cells.append("\u2713" if status["gist_embedded"] else "-")
                    cells.append("\u2713" if status["gemma_embedded"] else "-")
                table.add_row(*cells, key=node_id)
            else:
                # Aggregate level: fetch/embed counts across descendants
                cells = [name]
                scopes = resolve_scopes_for_node(self._level, self._path, node_id, active_types)
                agg = get_aggregate_status(scopes, active_types)
                for wt in active_types:
                    wt_agg = agg.get(wt, {"fetched": 0, "total": 0, "gist_count": 0, "gemma_count": 0})
                    cells.append(f"{wt_agg['fetched']}/{wt_agg['total']}")
                    cells.append(str(wt_agg["gist_count"]))
                    cells.append(str(wt_agg["gemma_count"]))
                table.add_row(*cells, key=node_id)

        self._update_status_bar()

    def _toggle_type(self, index: int) -> None:
        self._active_types[index] = not self._active_types[index]
        self._render_current_level()

    def action_toggle_type_1(self) -> None:
        self._toggle_type(0)

    def action_toggle_type_2(self) -> None:
        self._toggle_type(1)

    def action_toggle_type_3(self) -> None:
        self._toggle_type(2)

    def action_toggle_type_4(self) -> None:
        self._toggle_type(3)

    def action_refresh(self) -> None:
        self._render_current_level()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Drill down on Enter/click."""
        if self._level >= 3:
            return
        row_key = str(event.row_key.value)
        name = ""
        for rid, rname in self._rows:
            if rid == row_key:
                name = rname
                break
        self._path.append((row_key, name))
        self._level += 1
        self._render_current_level()

    def on_key(self, event) -> None:
        table = self.query_one("#scope_table", DataTable)
        if event.key == "left" or event.key == "backspace":
            if self._level > 0:
                self._path.pop()
                self._level -= 1
                self._render_current_level()
        elif event.key == "right":
            if self._level < 3 and table.cursor_row is not None:
                row_key = str(list(table.rows.keys())[table.cursor_row].value)
                name = ""
                for rid, rname in self._rows:
                    if rid == row_key:
                        name = rname
                        break
                self._path.append((row_key, name))
                self._level += 1
                self._render_current_level()
        elif event.key == "j":
            table.action_cursor_down()
        elif event.key == "k":
            table.action_cursor_up()

    def action_fetch(self) -> None:
        """Launch fetch jobs for the selected scope."""
        table = self.query_one("#scope_table", DataTable)
        if table.cursor_row is None:
            return
        row_key = str(list(table.rows.keys())[table.cursor_row].value)
        active_types = self._active_type_names()

        if self._level == 3:
            field_id = self._path[1][0]
            sf_id = self._path[2][0]
            launched = 0
            for wt in active_types:
                scope_str = f"{field_id}/{sf_id}/{row_key}/{wt}"
                job_id = launch_worker("fetch", scope_str)
                if job_id:
                    launched += 1
            self.notify(f"Launched {launched} fetch job(s)" if launched else "Already running or at capacity")
        else:
            if self._level == 0:
                scope_str = row_key  # domain ID
            elif self._level == 1:
                scope_str = row_key  # field ID
            elif self._level == 2:
                scope_str = f"{self._path[1][0]}/{row_key}"  # field/subfield
            job_id = launch_worker("fetch", scope_str, work_types=active_types)
            self.notify("Launched fetch job" if job_id else "Already running or at capacity")

        self._update_status_bar()

    def action_embed(self) -> None:
        """Prompt for model, then launch embed jobs."""
        table = self.query_one("#scope_table", DataTable)
        if table.cursor_row is None:
            return
        row_key = str(list(table.rows.keys())[table.cursor_row].value)

        def on_model_selected(model: str | None) -> None:
            if model is None:
                return
            active_types = self._active_type_names()
            if self._level == 3:
                field_id = self._path[1][0]
                sf_id = self._path[2][0]
                launched = 0
                for wt in active_types:
                    scope_str = f"{field_id}/{sf_id}/{row_key}/{wt}"
                    job_id = launch_worker("embed", scope_str, model=model)
                    if job_id:
                        launched += 1
                self.notify(f"Launched {launched} embed job(s)" if launched else "Already running or at capacity")
            else:
                self.notify("Embed is only supported at topic level")
            self._update_status_bar()

        self.push_screen(ModelSelectScreen(), callback=on_model_selected)

    def action_dismiss_jobs(self) -> None:
        """Remove completed/failed/crashed job manifests."""
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
        self.notify(f"Dismissed {dismissed} job(s)" if dismissed else "No jobs to dismiss")
        self._update_status_bar()

    def action_view_log(self) -> None:
        """Show path to the most recent running job's log."""
        jobs = discover_jobs()
        running = [j for j in jobs if j["status"] == "running"]
        if not running:
            self.notify("No running jobs")
            return
        log_path = _jobs_path() / f"{running[0]['_job_id']}.log"
        if log_path.exists():
            self.notify(f"Log: {log_path}")
        else:
            self.notify("No log file found")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scope Manager TUI")
    parser.add_argument("--worker", metavar="ACTION", help="Run in worker mode (fetch|embed)")
    parser.add_argument("scope", nargs="?", help="Scope string (e.g., 17/1702/T10181/article)")
    parser.add_argument("--model", help="Embedder model name (for embed action)")
    parser.add_argument("--work-types", help="Comma-separated work types (for aggregate fetch)")
    parser.add_argument("--job-id", help="Job ID (set by launcher)")
    args = parser.parse_args()

    if args.worker:
        wt = args.work_types.split(",") if args.work_types else None
        run_worker(args.worker, args.scope, args.model, args.job_id, work_types=wt)
    else:
        app = ScopeManagerApp()
        app.run()


if __name__ == "__main__":
    main()
