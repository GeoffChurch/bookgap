#!/usr/bin/env python3
"""Backfill legacy batch outputs with resumable run_state metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from omegaconf import OmegaConf
from openai import OpenAI

from bookgap.llm.generators.run_state import RunState, RunStep

BATCH_ID_PATTERNS = [
    re.compile(r"Batch job created successfully:\s+(batch_[A-Za-z0-9]+)"),
    re.compile(r"Submitted batch job:\s+(batch_[A-Za-z0-9]+)"),
    re.compile(r"Using recorded batch job id:\s+(batch_[A-Za-z0-9]+)"),
]

ACTIVE_STATUSES = {"validating", "in_progress", "finalizing"}
SUCCESS_STATUSES = {"completed"}
FAILURE_STATUSES = {"failed", "expired", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer run_state.json and experiment_config.yaml for prior batch runs "
            "so the resumable workflow can continue from an existing output directory."
        )
    )
    parser.add_argument(
        "output_dirs",
        metavar="OUTPUT_DIR",
        nargs="*",
        help="Path(s) to Hydra output directories (defaults to all runs under outputs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run_state.json if present",
    )
    return parser.parse_args()


def locate_batch_id(log_path: Path) -> str:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing batch log: {log_path}")

    for line in reversed(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        for pattern in BATCH_ID_PATTERNS:
            match = pattern.search(line)
            if match:
                return match.group(1)
    raise ValueError(f"Could not locate a batch id inside {log_path}")


def snapshot_experiment_config(output_dir: Path, dry_run: bool) -> Optional[Tuple[Path, Any]]:
    target = output_dir / "experiment_config.yaml"
    hydra_cfg = output_dir / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        print(f"[WARN] No Hydra config found at {hydra_cfg}; skipping config snapshot", file=sys.stderr)
        return None

    cfg = OmegaConf.load(hydra_cfg)
    if not target.exists():
        if dry_run:
            print(f"[DRY] Would write {target}")
        else:
            OmegaConf.save(cfg, target)
            print(f"[OK ] Wrote {target}")
    return target, cfg


def fetch_batch(batch_id: str) -> Optional[dict]:
    client = OpenAI()
    try:
        batch = client.batches.retrieve(batch_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Unable to retrieve batch {batch_id}: {exc}", file=sys.stderr)
        return None

    if hasattr(batch, "model_dump"):
        return batch.model_dump()
    if hasattr(batch, "to_dict"):
        return batch.to_dict()
    return json.loads(batch.json())  # type: ignore[attr-defined]


def determine_step(status: Optional[str], *, local_results_present: bool) -> RunStep:
    if not status:
        return RunStep.COMPLETED if local_results_present else RunStep.SUBMITTED
    if status in SUCCESS_STATUSES:
        return RunStep.COMPLETED if local_results_present else RunStep.SUBMITTED
    if status in ACTIVE_STATUSES:
        return RunStep.SUBMITTED
    if status in FAILURE_STATUSES:
        return RunStep.SUBMITTED
    return RunStep.SUBMITTED


def build_details(
    batch: Optional[dict],
    config: Optional[dict],
    *,
    local_results_present: bool,
    missing_results_warning: bool,
) -> dict:
    details: dict = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }
    if batch:
        status = batch.get("status")
        details["batch_status"] = status
        for key in ("created_at", "completed_at", "expires_at"):
            if batch.get(key) is not None:
                details[key] = batch[key]

        counts = batch.get("request_counts") or {}
        if counts:
            details["request_counts"] = {
                "total": counts.get("total"),
                "completed": counts.get("completed"),
                "failed": counts.get("failed"),
            }

        if batch.get("metadata"):
            details["metadata"] = batch["metadata"]

        if batch.get("output_file_id"):
            details["output_file_id"] = batch["output_file_id"]

        if batch.get("error_file_id"):
            details["error_file_id"] = batch["error_file_id"]
    else:
        details.setdefault("warnings", []).append("Batch metadata unavailable via API")

    if config:
        cloud_cfg = config.get("cloud", {}) if isinstance(config, dict) else {}
        embedder = cloud_cfg.get("embedder_model") if isinstance(cloud_cfg, dict) else None
        denoiser = cloud_cfg.get("denoiser") if isinstance(cloud_cfg, dict) else None
        drop_k = cloud_cfg.get("drop_k") if isinstance(cloud_cfg, dict) else None

        if embedder is not None:
            details["embedder_model"] = str(embedder)
        if denoiser is not None:
            details["denoiser"] = str(denoiser)
        if drop_k is not None:
            details["drop_k"] = drop_k

    details["local_results_present"] = local_results_present
    if missing_results_warning:
        details.setdefault("warnings", []).append(
            "Remote batch completed but local batch_results.jsonl missing"
        )

    return details


def write_run_state(output_dir: Path, state: RunState, dry_run: bool, force: bool) -> None:
    target = output_dir / "run_state.json"
    if target.exists() and not force:
        print(f"[SKIP] run_state.json already exists at {target}; use --force to overwrite", file=sys.stderr)
        return

    state.last_updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if dry_run:
        print(f"[DRY] Would write {target} with state:\n{json.dumps(state.to_dict(), indent=2)}")
        return

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2, sort_keys=True)
    print(f"[OK ] Wrote {target}")


def process_output_dir(path: str, dry_run: bool, force: bool) -> None:
    output_dir = Path(path).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    log_path = output_dir / "batch_generator.log"
    try:
        batch_id = locate_batch_id(log_path)
    except Exception as exc:
        print(f"[SKIP] {exc}", file=sys.stderr)
        return
    print(f"[INFO] Found batch id {batch_id} in {log_path}")

    experiment_info = snapshot_experiment_config(output_dir, dry_run)
    experiment_path: Optional[Path] = None
    config_data: Optional[dict[str, Any]] = None
    if experiment_info:
        experiment_path, hydra_cfg = experiment_info
        print(f"[INFO] Experiment config located at {experiment_path}")
        config_data = OmegaConf.to_container(hydra_cfg, resolve=True)  # type: ignore[arg-type]

    batch = fetch_batch(batch_id)
    status = batch.get("status") if batch else None
    print(f"[INFO] Remote batch status: {status or 'unknown'}")

    results_path = output_dir / "batch_results.jsonl"
    local_results_present = results_path.exists()
    missing_results_warning = bool(status in SUCCESS_STATUSES and not local_results_present)

    run_step = determine_step(status, local_results_present=local_results_present)
    details = build_details(
        batch,
        config_data,
        local_results_present=local_results_present,
        missing_results_warning=missing_results_warning,
    )
    if experiment_path:
        details["experiment_config"] = experiment_path.name

    state = RunState(step=run_step, batch_job_id=batch_id, details=details)
    write_run_state(output_dir, state, dry_run=dry_run, force=force)


def main() -> int:
    args = parse_args()
    targets = args.output_dirs
    if not targets:
        default_root = Path("outputs")
        if not default_root.exists():
            print("[ERROR] No output directories provided and outputs/ does not exist.", file=sys.stderr)
            return 1
        targets = [str(p) for p in sorted(default_root.rglob("*")) if (p / "batch_generator.log").exists()]
        if not targets:
            print("[ERROR] No eligible Hydra runs found under outputs/", file=sys.stderr)
            return 1

    try:
        for path in targets:
            process_output_dir(path, dry_run=args.dry_run, force=args.force)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
