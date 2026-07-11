"""Build births.npz for the Bazel temporal-holdout (epoch-normalized cross-repo tag axis).

For each Bazel repo, checkout its release tags and re-run ``bazel query`` to get the set
of target labels present at that tag; birth = earliest epoch a target appears. Reuses the
generic epoch/birth helpers. Robust: tags whose BUILD files don't load under the pinned
Bazel version are logged and skipped (old tags often need an ancient Bazel + WORKSPACE).

    python -m artifacts_scripts.gap_recover.adapters.bazel.build_births \\
        --data cache/code_gap/bazel/data --E 6
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from artifacts_scripts.gap_recover.adapters.python.build_births import (
    _compute_births_fast,
    select_epoch_tags,
    version_tags,
)
from artifacts_scripts.gap_recover.adapters.bazel.corpus import (
    BAZELISK, PACKAGES, REPO_ROOT,
)

LOGGER = logging.getLogger("bazel_build_births")
# Pin one Bazel version across all tags (avoid per-tag bazelisk downloads). Old tags that
# need a different Bazel/WORKSPACE simply fail to query and are skipped.
PIN_ENV = {**os.environ, "USE_BAZEL_VERSION": os.environ.get("USE_BAZEL_VERSION", "9.1.1")}


def query_labels(repo_dir: Path, scope: str) -> set[str] | None:
    try:
        r = subprocess.run(
            [BAZELISK, "query", f'kind("cc_(library|binary|test) rule", {scope})',
             "--output=xml", "--noimplicit_deps"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=600, env=PIN_ENV)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        root = ET.fromstring(r.stdout)
    except ET.ParseError:
        return None
    return {rule.get("name") for rule in root.findall("rule")}


def tag_namesets(repo_dir: Path, repo: str, scope: str,
                 selected: list[tuple[str, int]]) -> dict[int, set[str]]:
    epoch_namesets: dict[int, set[str]] = {}
    ref = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if ref == "HEAD":
        ref = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    try:
        for tag, epoch in selected:
            try:
                subprocess.run(["git", "-C", str(repo_dir), "checkout", "-q", "-f", tag],
                               capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                LOGGER.warning("[%s] checkout %s failed", repo, tag); continue
            labels = query_labels(repo_dir, scope)
            if labels is None:
                LOGGER.warning("[%s] query at %s failed/skipped (old BUILD?)", repo, tag); continue
            epoch_namesets[epoch] = {f"{repo}::{lab}" for lab in labels}
            LOGGER.info("[%s] tag=%s epoch=%d  %d targets", repo, tag, epoch, len(labels))
    finally:
        subprocess.run(["git", "-C", str(repo_dir), "checkout", "-q", "-f", ref],
                       capture_output=True, text=True)
        LOGGER.info("[%s] restored to %s", repo, ref)
    return epoch_namesets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "cache" / "code_gap" / "bazel" / "data"))
    ap.add_argument("--E", type=int, default=6)
    args = ap.parse_args()
    data_dir = Path(args.data); E = args.E
    d = np.load(data_dir / "data.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    is_F = d["is_F"].astype(bool); is_B = d["is_B"].astype(bool)
    N = len(names)
    LOGGER.info("Loaded %d node names", N)

    global_epoch: dict[int, set[str]] = {}
    for repo, _url, scope in PACKAGES:
        repo_dir = REPO_ROOT / repo
        if not repo_dir.is_dir():
            LOGGER.warning("[%s] not cloned — skipping", repo); continue
        tags = version_tags(repo_dir)
        if not tags:
            LOGGER.warning("[%s] no semver tags — skipping", repo); continue
        selected = select_epoch_tags(tags, E)
        LOGGER.info("[%s] %d semver tags -> %s", repo, len(tags), selected)
        for epoch, ns in tag_namesets(repo_dir, repo, scope, selected).items():
            global_epoch.setdefault(epoch, set()).update(ns)

    birth = _compute_births_fast(global_epoch, names, E)
    np.savez(data_dir / "births.npz", birth=birth,
             tags=np.array([f"e{i}" for i in range(E)], dtype=object))
    resolved = birth < E
    pct = lambda m: "n/a" if m.sum() == 0 else f"{resolved[m].mean()*100:.1f}%"
    print(f"\n[resolution] overall={resolved.mean()*100:.1f}%  F={pct(is_F)}  B={pct(is_B)}", flush=True)
    print(f"[births] E={E} resolved={int(resolved.sum())}/{N} epochs_seen={sorted(global_epoch)}",
          flush=True)


if __name__ == "__main__":
    main()
