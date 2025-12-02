#!/usr/bin/env python3
"""Resume a Hydra-managed batch run in-place using the original configuration."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from omegaconf import ListConfig, OmegaConf

from artifacts_scripts.resume import build_run_slug, slugify

HYDRA_DIRNAME = ".hydra"
HYDRA_BACKUP_SUFFIX = "_ORIG"
LOG_FILENAME = "batch_generator.log"


@dataclass(frozen=True)
class ResumeContext:
    output_dir: Path
    hydra_dir: Path
    hydra_backup: Optional[Path]
    log_path: Path
    log_backup: Optional[Path]
    overrides: list[str]
    embedder: Optional[str]
    denoiser: Optional[str]
    drop_k: Optional[str]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Hydra output directory to resume",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resume command without executing it",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for the resumed run",
    )
    return parser.parse_args(argv)


def _unique_backup_path(path: Path, suffix: str) -> Path:
    base = path.parent / f"{path.name}{suffix}"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = path.parent / f"{path.name}{suffix}{index}"
        if not candidate.exists():
            return candidate
        index += 1


def _load_overrides(hydra_dir: Path) -> list[str]:
    overrides_path = hydra_dir / "overrides.yaml"
    if not overrides_path.exists():
        raise FileNotFoundError(f"Missing overrides file: {overrides_path}")
    yaml_data = OmegaConf.load(overrides_path)
    if not isinstance(yaml_data, (list, ListConfig)):
        raise ValueError(
            f"Expected overrides.yaml to contain a list, got {type(yaml_data)!r}"
        )
    overrides: list[str] = []
    for entry in yaml_data:
        overrides.append(str(entry))
    return overrides


def _prepare_resume(output_dir: Path, *, mutate: bool) -> ResumeContext:
    resolved = output_dir.resolve()
    hydra_dir = resolved / HYDRA_DIRNAME
    if not hydra_dir.exists():
        raise FileNotFoundError(f"Hydra directory not found: {hydra_dir}")

    overrides = _load_overrides(hydra_dir)
    config_paths = [hydra_dir / "config.yaml", resolved / "experiment_config.yaml"]
    embedder: Optional[str] = None
    denoiser: Optional[str] = None
    drop_k: Optional[str] = None
    for config_path in config_paths:
        if not config_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            continue
        cloud_cfg = cfg_dict.get("cloud")
        if isinstance(cloud_cfg, dict):
            embedder_val = cloud_cfg.get("embedder_model")
            denoiser_val = cloud_cfg.get("denoiser")
            drop_k_val = cloud_cfg.get("drop_k")
            embedder = str(embedder_val) if embedder_val is not None else embedder
            denoiser = str(denoiser_val) if denoiser_val is not None else denoiser
            drop_k = str(drop_k_val) if drop_k_val is not None else drop_k
        if embedder and denoiser and drop_k:
            break

    hydra_backup: Optional[Path] = None
    log_path = resolved / LOG_FILENAME
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    log_backup: Optional[Path] = None

    if mutate:
        hydra_backup = _unique_backup_path(hydra_dir, HYDRA_BACKUP_SUFFIX)
        shutil.move(str(hydra_dir), str(hydra_backup))

        log_backup = _unique_backup_path(log_path, HYDRA_BACKUP_SUFFIX)
        shutil.copy2(log_path, log_backup)

    return ResumeContext(
        output_dir=resolved,
        hydra_dir=hydra_dir,
        hydra_backup=hydra_backup,
        log_path=log_path,
        log_backup=log_backup,
        overrides=overrides,
        embedder=embedder,
        denoiser=denoiser,
        drop_k=drop_k,
    )


def _build_command(ctx: ResumeContext, python_exe: str) -> list[str]:
    base = [python_exe, "-m", "bookgap.batch_generator"]
    base.append(f"hydra.run.dir={ctx.output_dir}")
    base.extend(ctx.overrides)
    return base


def _print_commands(ctx: ResumeContext, cmd: list[str]) -> None:
    run_slug = build_run_slug(ctx.output_dir)
    embedder = ctx.embedder or "embedder"
    denoiser = ctx.denoiser or "denoiser"
    drop_k = ctx.drop_k or "drop"
    log_name = (
        f"nohup_resume_{run_slug}_{slugify(embedder)}_"
        f"{slugify(denoiser)}_drop{slugify(drop_k)}.log"
    )
    nohup_cmd = [
        "nohup",
        *cmd,
        ">",
        log_name,
        "2>&1",
        "&",
    ]
    print("Resume command:")
    print(" ".join(cmd))
    print("\nnohup command:")
    print(" ".join(nohup_cmd))
    print("\nnohup log:", log_name)


def _run_command(cmd: list[str], dry_run: bool) -> int:
    if dry_run:
        print("DRY RUN - command:")
        print(" ".join(cmd))
        return 0
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _compare_hydra_dirs(original: Path, regenerated: Path) -> bool:
    cmp = filecmp.dircmp(original, regenerated)
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    for sub in cmp.subdirs.values():
        if not _compare_hydra_dirs(Path(sub.left), Path(sub.right)):
            return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    ctx = _prepare_resume(args.output_dir, mutate=not args.dry_run)
    cmd = _build_command(ctx, args.python)
    _print_commands(ctx, cmd)

    return_code = _run_command(cmd, args.dry_run)
    if return_code != 0:
        print(f"Resume command exited with status {return_code}")
        if not args.dry_run and ctx.hydra_backup is not None:
            # Restore original .hydra on failure
            if ctx.hydra_dir.exists():
                shutil.rmtree(ctx.hydra_dir)
            shutil.move(str(ctx.hydra_backup), str(ctx.hydra_dir))
        return return_code

    if not args.dry_run:
        if not ctx.hydra_dir.exists():
            print(f"ERROR: Hydra directory was not recreated at {ctx.hydra_dir}")
            return 1
        if ctx.hydra_backup is not None and not _compare_hydra_dirs(ctx.hydra_backup, ctx.hydra_dir):
            print("WARNING: Regenerated .hydra contents differ from original backup")
            print(f"Original: {ctx.hydra_backup}")
            print(f"Current : {ctx.hydra_dir}")
        else:
            print("Verified regenerated .hydra matches original backup")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
