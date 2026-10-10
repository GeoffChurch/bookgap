"""One-shot migration of the October 2026 runs to regen_sweep's current layout.

regen_sweep used to name a run directory by its start time and cell
(``2026-10-07_12-40-37_gist_a10_b5_..._r0.75``), start a fresh one whenever a
cell had not finished, and write a flat provenance.json. It now names a run by
its cell and ``run_identity`` (``gist_a10_b5_..._r0.75-<hash>``), resumes a
cell in place, and records one provenance entry per session. This renames each
run under runs/ and radius_runs/ to its identity name, rewrites its
provenance.json into the per-session form (keeping the old name as
``renamed_from``), and marks a run whose scores are written SCORED, the state
that now means done (it used to stop at COMPLETED, before scoring). Runs
already in the new form are left alone, so a second invocation is a no-op.

Usage (from the bookgap root)::

    uv run python -m artifacts_scripts.paper.migrate_regen_run_names
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from artifacts_scripts.paper.regen_sweep import RADIUS_ROOT, TOPIC_SET_ROOTS, run_identity
from bookgap.llm.generators.run_state import RunStateManager, RunStep

TIMESTAMPED = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<slug>.+)$")
SESSION_KEYS = ("command", "bookgap_commit", "bookgap_dirty", "packages", "ollama_num_ctx", "started")


def migrate(run: Path) -> Path:
    slug = TIMESTAMPED.match(run.name)["slug"]
    old = json.loads((run / "provenance.json").read_text())
    new = run.parent / f"{slug}-{run_identity((run / 'experiment_config.yaml').read_text(), old['ollama_model_digest'])}"
    if new.exists():
        raise FileExistsError(f"{new} exists; refusing to overwrite it with {run}")
    provenance = {
        "cell": {**old["cell"], "root": str(run.parent)},
        "ollama_model": old["ollama_model"],
        "ollama_model_digest": old["ollama_model_digest"],
        "sessions": [{k: old[k] for k in SESSION_KEYS}],
        "renamed_from": run.name,
    }
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    state = RunStateManager(run)
    if state.state.step == RunStep.COMPLETED and (run / "gaps_analysis_results.csv").exists():
        state.mark_scored()
    run.rename(new)
    return new


def main() -> int:
    for root in (TOPIC_SET_ROOTS["random20"], RADIUS_ROOT):
        for run in sorted(p for p in root.iterdir() if p.is_dir() and TIMESTAMPED.match(p.name)):
            print(f"{run.name} -> {migrate(run).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
