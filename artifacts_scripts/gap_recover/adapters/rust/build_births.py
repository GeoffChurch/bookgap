"""Build births.npz for the Rust temporal-holdout (epoch-normalized cross-crate tag axis).

Mirrors the Python adapter's build_births but uses the tree-sitter Rust extractor (no
toolchain / no build, so re-extracting at each tag is cheap). Reuses the generic
epoch/birth helpers from the Python adapter. Rust ``extract_repo`` already namespaces
node names as ``{crate}::…`` (matching data.npz), so no extra prefixing is needed.

    python -m artifacts_scripts.gap_recover.adapters.rust.build_births \\
        --data cache/code_gap/rust/data --E 6
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from artifacts_scripts.gap_recover.adapters.python.build_births import (
    _compute_births_fast,
    select_epoch_tags,
    version_tags,
)
from artifacts_scripts.gap_recover.adapters.rust.corpus import PACKAGES, REPO_ROOT
from artifacts_scripts.gap_recover.adapters.rust.extract import extract_repo

LOGGER = logging.getLogger("rust_build_births")


def tag_namesets(repo_dir: Path, crate: str, selected: list[tuple[str, int]]) -> dict[int, set[str]]:
    """Checkout each (tag, epoch), tree-sitter-extract node names; always restore the ref."""
    epoch_namesets: dict[int, set[str]] = {}
    try:
        ref = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if ref == "HEAD":
            ref = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError as exc:
        LOGGER.error("[%s] could not determine starting ref: %s", crate, exc)
        return epoch_namesets
    try:
        for tag, epoch in selected:
            try:
                subprocess.run(["git", "-C", str(repo_dir), "checkout", "-q", tag],
                               capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as exc:
                LOGGER.warning("[%s] checkout %s failed: %s", crate, tag, exc)
                continue
            try:
                r = extract_repo(repo_dir, crate)        # names already "{crate}::…"
                epoch_namesets[epoch] = set(r.names)
                LOGGER.info("[%s] tag=%s epoch=%d  %d names", crate, tag, epoch, len(r.names))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[%s] extract at tag %s failed: %s", crate, tag, exc)
    finally:
        try:
            subprocess.run(["git", "-C", str(repo_dir), "checkout", "-q", ref],
                           capture_output=True, text=True, check=True)
            LOGGER.info("[%s] restored to %s", crate, ref)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("[%s] FAILED to restore ref %s: %s", crate, ref, exc)
    return epoch_namesets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "cache" / "code_gap" / "rust" / "data"))
    ap.add_argument("--E", type=int, default=6)
    args = ap.parse_args()
    data_dir = Path(args.data); E = args.E
    d = np.load(data_dir / "data.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    is_F = d["is_F"].astype(bool); is_B = d["is_B"].astype(bool)
    N = len(names)
    LOGGER.info("Loaded %d node names from %s", N, data_dir / "data.npz")

    global_epoch: dict[int, set[str]] = {}
    for crate, _url, _n in PACKAGES:
        repo_dir = REPO_ROOT / crate
        if not repo_dir.is_dir():
            LOGGER.warning("[%s] repo dir not found — skipping", crate); continue
        tags = version_tags(repo_dir)
        if not tags:
            LOGGER.warning("[%s] no semver tags — skipping", crate); continue
        selected = select_epoch_tags(tags, E)
        LOGGER.info("[%s] %d semver tags -> %s", crate, len(tags), selected)
        for epoch, ns in tag_namesets(repo_dir, crate, selected).items():
            global_epoch.setdefault(epoch, set()).update(ns)

    birth = _compute_births_fast(global_epoch, names, E)
    np.savez(data_dir / "births.npz", birth=birth,
             tags=np.array([f"e{i}" for i in range(E)], dtype=object))
    resolved = birth < E
    pct = lambda m: "n/a" if m.sum() == 0 else f"{resolved[m].mean()*100:.1f}%"
    print(f"\n[resolution] overall={resolved.mean()*100:.1f}%  F={pct(is_F)}  B={pct(is_B)}", flush=True)
    print(f"[births] E={E} resolved={int(resolved.sum())}/{N} unresolved={int((~resolved).sum())}",
          flush=True)


if __name__ == "__main__":
    main()
