#!/usr/bin/env python3
"""Scope Manager — CLI and TUI for browsing OpenAlex taxonomy, viewing status, launching jobs."""
from __future__ import annotations

from typing import Optional

import typer

from bookgap.scope_ops import (
    ALL_WORK_TYPES,
    KNOWN_DOMAINS,
    MODEL_PATHS,
    discover_jobs,
    dismiss_finished_jobs,
    get_domain_fields,
    get_fast_aggregate_status,
    get_field_subfields,
    get_scope_status,
    get_subfield_topics,
    get_system_resources,
    has_running_job,
    get_job_log_path,
    launch_worker,
    run_worker,
)
from bookgap.embedder import SentenceTransformerEmbedder as Embedder
from bookgap.openalex_ingest import ResolvedScope

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, OptionList, RichLog, Static
from textual.widgets.option_list import Option


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


class JobsScreen(ModalScreen):
    """Modal screen showing jobs list and live log viewer."""

    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
        Binding("d", "dismiss_jobs", "Dismiss finished"),
        Binding("j", "cursor_down", "Down", key_display="j/\u2193"),
        Binding("k", "cursor_up", "Up", key_display="k/\u2191"),
    ]

    CSS = """
    JobsScreen {
        align: center middle;
    }
    #jobs_container {
        width: 95%;
        height: 90%;
        border: solid $primary;
        background: $surface;
    }
    #jobs_table {
        height: 1fr;
        max-height: 12;
    }
    #log_viewer {
        height: 3fr;
        border-top: solid $primary-background;
    }
    #log_header {
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }
    """

    def __init__(self):
        super().__init__()
        self._selected_job_id: str | None = None
        self._log_offset: int = 0
        self._timer = None

    def compose(self) -> ComposeResult:
        with Vertical(id="jobs_container"):
            yield DataTable(id="jobs_table")
            yield Static("Select a job to view its log", id="log_header")
            yield RichLog(id="log_viewer", wrap=True, highlight=True)

    def on_mount(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        table.cursor_type = "row"
        self._refresh_jobs_table()
        self._timer = self.set_interval(1, self._poll_log)

    def _refresh_jobs_table(self) -> None:
        table = self.query_one("#jobs_table", DataTable)
        table.clear(columns=True)
        table.add_column("Job ID", key="job_id", width=30)
        table.add_column("Action", key="action", width=8)
        table.add_column("Scope", key="scope", width=30)
        table.add_column("Status", key="status", width=10)
        table.add_column("Started", key="started", width=20)

        jobs = discover_jobs()
        for job in jobs:
            table.add_row(
                job.get("_job_id", "?"),
                job.get("action", "?"),
                job.get("scope", "?"),
                job.get("status", "?"),
                job.get("started_at", "?"),
                key=job.get("_job_id", ""),
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_job(str(event.row_key.value))

    def on_data_table_cursor_changed(self, event: DataTable.CursorChanged) -> None:
        if event.cursor_row is not None:
            table = self.query_one("#jobs_table", DataTable)
            keys = list(table.rows.keys())
            if 0 <= event.cursor_row < len(keys):
                self._select_job(str(keys[event.cursor_row].value))

    def _select_job(self, job_id: str) -> None:
        if job_id == self._selected_job_id:
            return
        self._selected_job_id = job_id
        self._log_offset = 0
        log_viewer = self.query_one("#log_viewer", RichLog)
        log_viewer.clear()
        self.query_one("#log_header", Static).update(f"Log: {job_id}")
        self._poll_log()

    def _poll_log(self) -> None:
        if not self._selected_job_id:
            return
        log_path = get_job_log_path(self._selected_job_id)
        if not log_path.exists():
            return
        try:
            with open(log_path) as f:
                f.seek(self._log_offset)
                new_data = f.read()
                if new_data:
                    self._log_offset = f.tell()
                    log_viewer = self.query_one("#log_viewer", RichLog)
                    for line in new_data.splitlines():
                        log_viewer.write(line)
        except OSError:
            pass

    def action_cursor_down(self) -> None:
        self.query_one("#jobs_table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#jobs_table", DataTable).action_cursor_up()

    def action_dismiss_jobs(self) -> None:
        dismissed = dismiss_finished_jobs()
        self.notify(f"Dismissed {dismissed} job(s)" if dismissed else "No jobs to dismiss")
        self._refresh_jobs_table()

    def action_dismiss(self) -> None:
        self.app.pop_screen()


class ScopeManagerApp(App):
    """TUI for browsing OpenAlex scopes, viewing status, and launching fetch/embed jobs."""

    TITLE = "Scope Manager"

    BINDINGS = [
        Binding("question_mark", "show_help_panel", "Help", key_display="?"),
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "fetch", "Fetch"),
        Binding("e", "embed", "Embed"),
        Binding("d", "dismiss_jobs", "Dismiss jobs"),
        Binding("l", "view_log", "Jobs/logs"),
        Binding("left", "nav_back", "Back", key_display="\u2190/BS"),
        Binding("backspace", "nav_back", "Back", show=False),
        Binding("right", "nav_forward", "Drill down", key_display="\u2192/Enter"),
        Binding("j", "cursor_down", "Down", key_display="j/\u2193"),
        Binding("k", "cursor_up", "Up", key_display="k/\u2191"),
        Binding("1", "toggle_type_1", "Toggle article"),
        Binding("2", "toggle_type_2", "Toggle book"),
        Binding("3", "toggle_type_3", "Toggle book-chapter"),
        Binding("4", "toggle_type_4", "Toggle preprint"),
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
    DataTable {
        height: 1fr;
    }
    """

    def __init__(self):
        super().__init__()
        self._level: int = 0
        self._path: list[tuple[str, str]] = []
        self._active_types: list[bool] = [True, True, True, True]
        self._rows: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Static("Scope Manager", id="breadcrumb")
        yield DataTable(id="scope_table")
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

        self.query_one("#status_bar", Static).update(
            f"Types: {types_str} | Jobs: {jobs_str} | Press [bold]?[/bold] for help"
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
            self._rows = get_subfield_topics(subfield_id)

        table.add_column("Name", key="name", width=30)
        for wt in active_types:
            table.add_column(f"{wt[:8]} fetch", key=f"{wt}_fetch", width=12)
            table.add_column(f"{wt[:8]} GIST", key=f"{wt}_gist", width=8)
            table.add_column(f"{wt[:8]} Gem", key=f"{wt}_gemma", width=8)

        for node_id, name in self._rows:
            if self._level == 3:
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
                cells = [name]
                agg = get_fast_aggregate_status(self._level, node_id, self._path, active_types)
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

    def action_cursor_down(self) -> None:
        self.query_one("#scope_table", DataTable).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#scope_table", DataTable).action_cursor_up()

    def action_nav_back(self) -> None:
        if self._level > 0:
            self._path.pop()
            self._level -= 1
            self._render_current_level()

    def action_nav_forward(self) -> None:
        table = self.query_one("#scope_table", DataTable)
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
                scope_str = row_key
            elif self._level == 1:
                scope_str = row_key
            elif self._level == 2:
                scope_str = f"{self._path[1][0]}/{row_key}"
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
        dismissed = dismiss_finished_jobs()
        self.notify(f"Dismissed {dismissed} job(s)" if dismissed else "No jobs to dismiss")
        self._update_status_bar()

    def action_view_log(self) -> None:
        """Open the jobs screen with live log viewer."""
        self.push_screen(JobsScreen())


# ---------------------------------------------------------------------------
# Typer CLI app
# ---------------------------------------------------------------------------
# Imports here to avoid loading Typer/Rich when only the TUI is needed
# (Textual has its own Rich dependency but Typer is separate)

def _build_cli():
    """Build and return the Typer app. Deferred to avoid import cost when running TUI directly."""
    import json as _json

    from rich.console import Console
    from rich.table import Table

    cli = typer.Typer(help="Scope Manager — browse OpenAlex taxonomy, check status, launch jobs.")
    console = Console()
    _use_json = False

    @cli.callback()
    def cli_main(
        json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    ):
        nonlocal _use_json
        _use_json = json_output

    # -- list subcommand --

    ENTITY_CHOICES = ["domains", "fields", "subfields", "topics"]

    @cli.command()
    def list(
        entity: str = typer.Argument(help="Entity type: domains, fields, subfields, topics"),
        parent_id: Optional[str] = typer.Argument(None, help="Parent ID (required for fields/subfields/topics)"),
    ):
        """List hierarchy entities (domains, fields, subfields, topics)."""
        if entity not in ENTITY_CHOICES:
            typer.echo(f"Unknown entity: {entity}. Choose from: {', '.join(ENTITY_CHOICES)}", err=True)
            raise typer.Exit(1)

        if entity == "domains":
            rows = KNOWN_DOMAINS
        elif entity == "fields":
            if not parent_id:
                typer.echo("parent_id (domain ID) is required for 'fields'", err=True)
                raise typer.Exit(1)
            rows = get_domain_fields(parent_id)
        elif entity == "subfields":
            if not parent_id:
                typer.echo("parent_id (field ID) is required for 'subfields'", err=True)
                raise typer.Exit(1)
            rows = get_field_subfields(parent_id)
        elif entity == "topics":
            if not parent_id:
                typer.echo("parent_id (subfield ID) is required for 'topics'", err=True)
                raise typer.Exit(1)
            rows = get_subfield_topics(parent_id)
        else:
            raise typer.Exit(1)  # unreachable — guarded by ENTITY_CHOICES check above

        if _use_json:
            console.print_json(_json.dumps([{"id": r[0], "name": r[1]} for r in rows]))
        else:
            table = Table(title=f"{entity.title()}")
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            for rid, name in rows:
                table.add_row(rid, name)
            console.print(table)

    # -- status subcommand --

    @cli.command()
    def status(
        scope_path: str = typer.Argument(help="Scope path (e.g., 17/1702/T10181/article or 17/1702)"),
        work_types: Optional[str] = typer.Option(None, "--work-types", "-w", help="Comma-separated work types"),
    ):
        """Show fetch/embed status for a scope (single or aggregate)."""
        parts = scope_path.split("/")
        active_types = work_types.split(",") if work_types else ALL_WORK_TYPES

        if len(parts) == 4:
            scope = ResolvedScope(field=parts[0], subfield=parts[1], topic=parts[2], work_type=parts[3])
            result = get_scope_status(scope)
            if _use_json:
                console.print_json(_json.dumps(result))
            else:
                table = Table(title=f"Status: {scope_path}")
                table.add_column("Property")
                table.add_column("Value")
                table.add_row("Fetched", str(result["fetched"]))
                table.add_row("Work count", str(result["work_count"]))
                table.add_row("GIST embedded", str(result["gist_embedded"]))
                table.add_row("Gemma embedded", str(result["gemma_embedded"]))
                console.print(table)
        elif len(parts) in (1, 2):
            if len(parts) == 1:
                level, node_id = 1, parts[0]
                path_tuples: list[tuple[str, str]] = [("", "")]
            else:
                level, node_id = 2, parts[1]
                path_tuples = [("", ""), (parts[0], "")]

            agg = get_fast_aggregate_status(level, node_id, path_tuples, active_types)
            if _use_json:
                console.print_json(_json.dumps(agg))
            else:
                table = Table(title=f"Aggregate status: {scope_path}")
                table.add_column("Work Type")
                table.add_column("Fetched")
                table.add_column("GIST")
                table.add_column("Gemma")
                for wt, data in agg.items():
                    table.add_row(
                        wt,
                        f"{data['fetched']}/{data['total']}",
                        str(data["gist_count"]),
                        str(data["gemma_count"]),
                    )
                console.print(table)
        else:
            typer.echo(f"Invalid scope path: {scope_path}", err=True)
            raise typer.Exit(1)

    # -- fetch subcommand --

    @cli.command()
    def fetch(
        scope_path: str = typer.Argument(help="Scope path (e.g., 17/1702/T10181/article or 17/1702)"),
        work_types: Optional[str] = typer.Option(None, "--work-types", "-w", help="Comma-separated work types"),
    ):
        """Launch a fetch job for the given scope."""
        wt = work_types.split(",") if work_types else None
        job_id = launch_worker("fetch", scope_path, work_types=wt)
        if job_id:
            result = {"job_id": job_id, "status": "launched"}
            if _use_json:
                console.print_json(_json.dumps(result))
            else:
                console.print(f"Launched fetch job: [cyan]{job_id}[/cyan]")
        else:
            if _use_json:
                console.print_json(_json.dumps({"error": "duplicate or at capacity"}))
            else:
                console.print("[red]Already running or at capacity[/red]")
            raise typer.Exit(1)

    # -- snapshot subcommand --

    @cli.command()
    def snapshot(
        scope_path: str = typer.Argument(help="Scope path (e.g., 3 or 31/3102/T10805/article)"),
        work_types: Optional[str] = typer.Option(None, "--work-types", "-w", help="Comma-separated work types"),
        snapshot_date: Optional[str] = typer.Option(None, "--snapshot-date", "-d", help="Pin to a specific OpenAlex snapshot date (YYYY-MM-DD). Defaults to SnapshotConfig.snapshot_date."),
    ):
        """Launch a snapshot-based ingest job for the given scope."""
        wt = work_types.split(",") if work_types else None
        job_id = launch_worker("snapshot", scope_path, work_types=wt, snapshot_date=snapshot_date)
        if job_id:
            result = {"job_id": job_id, "status": "launched"}
            if _use_json:
                console.print_json(_json.dumps(result))
            else:
                console.print(f"Launched snapshot job: [cyan]{job_id}[/cyan]")
        else:
            if _use_json:
                console.print_json(_json.dumps({"error": "duplicate or at capacity"}))
            else:
                console.print("[red]Already running or at capacity[/red]")
            raise typer.Exit(1)

    # -- embed subcommand --

    MODEL_ALIASES = {"gist": MODEL_PATHS["gist"], "gemma": MODEL_PATHS["gemma"]}

    @cli.command()
    def embed(
        scope_path: str = typer.Argument(help="Full scope path (field/subfield/topic/work_type)"),
        model: str = typer.Option(..., "--model", "-m", help="Model: gist or gemma"),
    ):
        """Launch an embed job for the given scope."""
        resolved_model = MODEL_ALIASES.get(model, model)
        job_id = launch_worker("embed", scope_path, model=resolved_model)
        if job_id:
            result = {"job_id": job_id, "status": "launched"}
            if _use_json:
                console.print_json(_json.dumps(result))
            else:
                console.print(f"Launched embed job: [cyan]{job_id}[/cyan]")
        else:
            if _use_json:
                console.print_json(_json.dumps({"error": "duplicate or at capacity"}))
            else:
                console.print("[red]Already running or at capacity[/red]")
            raise typer.Exit(1)

    # -- jobs subcommand --

    jobs_app = typer.Typer(help="Manage background jobs.")
    cli.add_typer(jobs_app, name="jobs", invoke_without_command=True)

    @jobs_app.callback(invoke_without_command=True)
    def jobs_list(ctx: typer.Context):
        """List all jobs (default when no subcommand given)."""
        if ctx.invoked_subcommand is not None:
            return
        all_jobs = discover_jobs()
        if _use_json:
            console.print_json(_json.dumps(all_jobs, default=str))
        else:
            if not all_jobs:
                console.print("No jobs found.")
                return
            table = Table(title="Jobs")
            table.add_column("Job ID", style="cyan")
            table.add_column("Action")
            table.add_column("Scope")
            table.add_column("Model")
            table.add_column("Status")
            table.add_column("Started")
            for job in all_jobs:
                table.add_row(
                    job.get("_job_id", "?"),
                    job.get("action", "?"),
                    job.get("scope", "?"),
                    job.get("model") or "-",
                    job.get("status", "?"),
                    job.get("started_at", "?"),
                )
            console.print(table)

    @jobs_app.command()
    def dismiss():
        """Remove completed/failed/crashed job manifests."""
        count = dismiss_finished_jobs()
        if _use_json:
            console.print_json(_json.dumps({"dismissed": count}))
        else:
            console.print(f"Dismissed {count} job(s)." if count else "No jobs to dismiss.")

    @jobs_app.command()
    def log(
        job_id: Optional[str] = typer.Argument(None, help="Job ID (default: most recent running job)"),
        follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output (like tail -f)"),
        tail: int = typer.Option(20, "--tail", "-n", help="Number of lines to show from end"),
    ):
        """Show or follow a job's log file."""
        import time as _time

        if job_id is None:
            jobs = discover_jobs()
            running = [j for j in jobs if j["status"] == "running"]
            if running:
                job_id = running[0]["_job_id"]
            elif jobs:
                job_id = jobs[-1]["_job_id"]
            else:
                typer.echo("No jobs found.", err=True)
                raise typer.Exit(1)

        log_path = get_job_log_path(job_id)
        if not log_path.exists():
            typer.echo(f"No log file for job: {job_id}", err=True)
            raise typer.Exit(1)

        if _use_json:
            with open(log_path) as f:
                lines = f.readlines()
            console.print_json(_json.dumps({
                "job_id": job_id,
                "log_path": str(log_path),
                "lines": [l.rstrip() for l in lines[-tail:]],
            }))
            return

        console.print(f"[dim]Log: {log_path}[/dim]")
        with open(log_path) as f:
            all_lines = f.readlines()
            start = max(0, len(all_lines) - tail)
            for line in all_lines[start:]:
                console.print(line.rstrip())

            if follow:
                console.print("[dim]--- following (Ctrl+C to stop) ---[/dim]")
                try:
                    while True:
                        line = f.readline()
                        if line:
                            console.print(line.rstrip())
                        else:
                            _time.sleep(0.5)
                except KeyboardInterrupt:
                    console.print("\n[dim]Stopped.[/dim]")

    # -- tui subcommand --

    @cli.command()
    def tui():
        """Launch the interactive TUI."""
        app = ScopeManagerApp()
        app.run()

    # -- worker subcommand (hidden, called by launch_worker) --

    @cli.command(hidden=True)
    def worker(
        action: str = typer.Argument(help="Action: fetch, embed, or snapshot"),
        scope: str = typer.Argument(help="Scope string"),
        model: Optional[str] = typer.Option(None, "--model"),
        work_types: Optional[str] = typer.Option(None, "--work-types"),
        snapshot_date: Optional[str] = typer.Option(None, "--snapshot-date"),
        job_id: str = typer.Option(..., "--job-id"),
    ):
        """Internal worker entry point (called by launch_worker)."""
        wt = work_types.split(",") if work_types else None
        run_worker(action, scope, model, job_id, work_types=wt, snapshot_date=snapshot_date)

    return cli


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli = _build_cli()
    cli()


if __name__ == "__main__":
    main()
