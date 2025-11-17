#!/usr/bin/env python3
"""List incomplete runs and prompt for a resumable target."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from omegaconf import OmegaConf

from bookgap.llm.generators.run_state import RunState, RunStep

OUTPUT_ROOT = Path("outputs")
RESUME_MARKER_FILENAME = "resumed.txt"


def discover_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if (path / "batch_generator.log").exists()
    )


def read_state(output_dir: Path) -> Optional[RunState]:
    state_path = output_dir / "run_state.json"
    if not state_path.exists():
        return None
    with open(state_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RunState.from_dict(data)


def read_resume_marker(output_dir: Path) -> Optional[str]:
    marker_path = output_dir / RESUME_MARKER_FILENAME
    if not marker_path.exists():
        return None
    return marker_path.read_text(encoding="utf-8")


def write_resume_marker(
    output_dir: Path,
    *,
    direct_cmd: str,
    nohup_cmd: str,
    log_filename: Path,
) -> None:
    marker_path = output_dir / RESUME_MARKER_FILENAME
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content = (
        f"Resumed at {timestamp} UTC\n"
        f"Command: {direct_cmd}\n"
        f"nohup: {nohup_cmd}\n"
        f"Log file: {log_filename}\n"
        "Remove this file if you need to resume this run again.\n"
    )
    marker_path.write_text(content, encoding="utf-8")


def extract_cloud_params_from_state(state: Optional[RunState]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if state is None:
        return (None, None, None)
    details = state.details or {}
    embedder = details.get("embedder_model")
    denoiser = details.get("denoiser")
    drop_k = details.get("drop_k")
    drop_k_str = str(drop_k) if drop_k is not None else None
    return (
        str(embedder) if embedder not in (None, "?") else None,
        str(denoiser) if denoiser not in (None, "?") else None,
        drop_k_str if drop_k_str not in (None, "?") else None,
    )


def extract_cloud_params_from_config(output_dir: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    for candidate in (
        output_dir / "experiment_config.yaml",
        output_dir / ".hydra" / "config.yaml",
    ):
        if not candidate.exists():
            continue
        cfg = OmegaConf.load(candidate)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        if not isinstance(cfg_dict, dict):
            continue
        cloud = cfg_dict.get("cloud")
        if not isinstance(cloud, dict):
            continue
        embedder = cloud.get("embedder_model")
        denoiser = cloud.get("denoiser")
        drop_k = cloud.get("drop_k")
        return (
            str(embedder) if embedder is not None else None,
            str(denoiser) if denoiser is not None else None,
            str(drop_k) if drop_k is not None else None,
        )
    return (None, None, None)


def resolve_cloud_params(output_dir: Path, state: Optional[RunState]) -> Tuple[str, str, str]:
    embedder, denoiser, drop_k = extract_cloud_params_from_state(state)
    if embedder is None or denoiser is None or drop_k is None:
        cfg_embedder, cfg_denoiser, cfg_drop_k = extract_cloud_params_from_config(output_dir)
        if embedder is None:
            embedder = cfg_embedder
        if denoiser is None:
            denoiser = cfg_denoiser
        if drop_k is None:
            drop_k = cfg_drop_k
    return (
        embedder if embedder is not None else "?",
        denoiser if denoiser is not None else "?",
        drop_k if drop_k is not None else "?",
    )


def slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in value)
    cleaned = cleaned.strip("-")
    return cleaned or "value"


def build_run_slug(output_dir: Path) -> str:
    if len(output_dir.parts) >= 2:
        candidate = f"{output_dir.parts[-2]}_{output_dir.parts[-1]}"
    else:
        candidate = output_dir.name
    return slugify(candidate)


def describe_state(output_dir: Path, state: Optional[RunState]) -> str:
    if state is None:
        return "missing run_state.json"
    batch_id = state.batch_job_id or "(no batch id)"
    batch_status = state.details.get("batch_status", "unknown")
    embedder, denoiser, drop_k = resolve_cloud_params(output_dir, state)
    return (
        f"step={state.step.value}, batch={batch_id}, remote={batch_status}, "
        f"embedder={embedder}, denoiser={denoiser}, drop_k={drop_k}"
    )


def main() -> int:
    run_dirs = discover_runs(OUTPUT_ROOT)
    candidates: list[tuple[int, Path, RunState]] = []
    missing: list[Path] = []
    already_resumed: list[tuple[Path, RunState, str]] = []

    for output_dir in run_dirs:
        state = read_state(output_dir)
        if state is None:
            missing.append(output_dir)
            continue
        if state.step == RunStep.COMPLETED:
            continue
        marker = read_resume_marker(output_dir)
        if marker:
            already_resumed.append((output_dir, state, marker))
            continue
        candidates.append((len(candidates) + 1, output_dir, state))

    if not candidates and not missing and not already_resumed:
        print("All runs under outputs/ are completed.")
        return 0

    if candidates:
        print("Resumable runs:")
        for idx, output_dir, state in candidates:
            print(f"  [{idx}] {output_dir} -> {describe_state(output_dir, state)}")
    else:
        print("No runs have resumable state.")

    if already_resumed:
        print("\nAlready resumed (marker present - remove marker to resume again):")
        for output_dir, state, marker in already_resumed:
            marker_line = marker.splitlines()[0] if marker else "(empty marker)"
            print(
                f"  [R] {output_dir} -> {describe_state(output_dir, state)} | {marker_line}"
            )

    if missing:
        print("\nRuns missing run_state.json (backfill required before resume):")
        for output_dir in missing:
            print(f"  [-] {output_dir}")

    if not candidates:
        return 0

    selection = input("\nEnter the number of the run to resume (or press Enter to cancel): ").strip()
    if not selection:
        print("Cancelled.")
        return 0

    try:
        choice = int(selection)
    except ValueError:
        print("Invalid selection.")
        return 1

    match = next((entry for entry in candidates if entry[0] == choice), None)
    if not match:
        print("Selection does not correspond to a resumable run.")
        return 1

    _, chosen_dir, state = match
    embedder, denoiser, drop_k = resolve_cloud_params(chosen_dir, state)
    direct_cmd = (
        f"python -m bookgap.batch_generator --resume-output-dir {chosen_dir}"
    )
    run_slug = build_run_slug(chosen_dir)
    log_filename =  f"nohup_resume_{run_slug}_{slugify(embedder)}_{slugify(denoiser)}_drop{slugify(drop_k)}.log"
    nohup_cmd = f"nohup {direct_cmd} > {log_filename} 2>&1 &"

    print("\nResume command:\n" + direct_cmd)
    print("\nnohup command:\n" + nohup_cmd)
    print(
        "\nDetails:\n"
        f"  step: {state.step.value}\n"
        f"  batch_job_id: {state.batch_job_id or '(missing)'}\n"
        f"  last_updated: {state.last_updated}\n"
        f"  embedder: {embedder}\n"
        f"  denoiser: {denoiser}\n"
        f"  drop_k: {drop_k}\n"
        f"  nohup log: {log_filename}\n"
    )

    write_resume_marker(
        chosen_dir,
        direct_cmd=direct_cmd,
        nohup_cmd=nohup_cmd,
        log_filename=log_filename,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
