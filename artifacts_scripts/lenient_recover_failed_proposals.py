"""Lenient-recover proposals from `proposal: None` records, then re-run
the canonical coverage + CSV pipeline.

For records where the strict parser rejected the model output (typically
because the model produced a JSON array `[{...}]` instead of a single
object), apply :func:`_try_parse_proposal_lenient` to pull the first
parseable Proposal out of the persisted ``raw_content``. Then re-invoke
``run_batch_workflow`` so the canonical post-loop step embeds the newly
populated proposals via the cloud's denoiser projection, computes
coverage / percentile / z-score for them, and rewrites
``gaps_analysis_results.csv``.

The strict pipeline is the source of truth for downstream artifacts;
this script just hands it a populated JSONL where the strict parser
gave up. Other records pass through untouched.

Usage::

    uv run python -m artifacts_scripts.lenient_recover_failed_proposals \\
        --run-dir outputs/gemma4_sweep/2026-05-22_11-11-43_gist_a10_b5_dropk10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from bookgap.llm.generators.ollama_local import _try_parse_proposal_lenient
from bookgap.llm.generators.run_state import RunStateManager
from bookgap.llm.generators.workflow_functions import run_batch_workflow
from bookgap.llm.protocols import LLMBackend, OpenAIModel
from bookgap.types import CellType, DeflationStrategy


def _patch_jsonl(jsonl_path: Path) -> list[tuple[str, str]]:
    """Update JSONL in-place: for every record with ``proposal is None``,
    lenient-parse the persisted ``raw_content`` and populate ``proposal``
    if successful. Returns the list of (custom_id, title) recovered.
    """
    recovered: list[tuple[str, str]] = []
    out_lines: list[str] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("proposal") is None:
            raw = rec.get("raw_content", "") or ""
            parsed = _try_parse_proposal_lenient(raw)
            if parsed is not None:
                rec["proposal"] = parsed.model_dump()
                rec.setdefault("metadata", {})["lenient_recovered"] = True
                recovered.append((rec["custom_id"], parsed.title[:90]))
        out_lines.append(json.dumps(rec))
    jsonl_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return recovered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    jsonl_path = run_dir / "ollama_results.jsonl"
    cfg_path = run_dir / "experiment_config.yaml"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = jsonl_path.with_name(f"ollama_results.jsonl.lenient_bak.{ts}")
    shutil.copy2(jsonl_path, backup)
    print(f"Backed up {jsonl_path} -> {backup}", flush=True)

    recovered = _patch_jsonl(jsonl_path)
    print(f"Lenient-recovered {len(recovered)} record(s) on this pass:",
          flush=True)
    for cid, title in recovered:
        print(f"  {cid}: {title}", flush=True)
    if not recovered:
        print("(No fresh recoveries; rerunning pipeline anyway to "
              "rebuild CSV from current JSONL.)", flush=True)

    # YAML serialization stores enum-typed fields as bare strings; restore
    # them so downstream pipeline code that does e.g. ``cell_type.value``
    # works. Structured-config merge would do this automatically, but it
    # also chokes on Hydra's ``_target_`` fields on frozen dataclasses,
    # so we do the narrow conversion explicitly.
    cfg = OmegaConf.load(cfg_path)
    cfg.gap.cell_type = CellType[cfg.gap.cell_type]
    cfg.gap.deflation_strategy = DeflationStrategy[cfg.gap.deflation_strategy]
    cfg.llm.backend = LLMBackend[cfg.llm.backend]
    cfg.llm.model = OpenAIModel[cfg.llm.model]
    state_manager = RunStateManager(run_dir)
    print(f"Re-running canonical pipeline against {run_dir}...", flush=True)
    results = run_batch_workflow(cfg, run_dir, state_manager)
    print(f"Done. {len(results.proposals)} proposals processed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
