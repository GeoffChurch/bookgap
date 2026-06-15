"""Build births.npz for the Go std temporal-holdout (epoch-normalized release-tag axis).

Mirrors the Python/Rust adapters' build_births, reusing the generic epoch/birth helpers
(``select_epoch_tags``, ``_compute_births_fast``) from the Python adapter. Differences
specific to Go std:

  * The corpus is a SINGLE repo (golang/go), not a set of namespaced packages, so node
    names carry NO repo prefix (the static Go dataset's names are unprefixed:
    ``net/http.(*Client).Do``). We clone golang/go blobless-with-history once.

  * Release tags match ``^go\\d+\\.\\d+(\\.\\d+)?$`` (e.g. ``go1.21``, ``go1.21.4``),
    NOT the python semver regex. Sorted by creatordate, mapped onto E shared epochs.

  * Names at each tag come from the go/parser-only dumper
    (``names_at/`` — type-check-free, so it works on any era's source) run over the
    checked-out ``src/`` tree. Built binary preferred; falls back to ``go run .``.

Because the static dataset is shared across the ``cha``/``rta`` resolutions (identical node
set, differing only in edges), the SAME births array is written into every ``data.npz``-
bearing subdir found under --out-root (default: data_cha, data_rta). The temporal holdout
reads births.npz from whichever ``--data`` dir it is pointed at.

CLI::
    python -m artifacts_scripts.curriculum.gap_recover.adapters.go.build_births \\
        --data cache/code_gap/go/data_rta --E 6
    # (writes births.npz into the sibling data_* dirs that share the node set)
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]  # .../src/bookgap
sys.path.insert(0, str(ROOT))

from artifacts_scripts.curriculum.gap_recover.adapters.python.build_births import (  # noqa: E402
    _compute_births_fast,
    select_epoch_tags,
)

LOGGER = logging.getLogger("go_build_births")

# Go release tags: go1, go1.2, go1.21.4 — NOT rc/beta (those carry a suffix).
_GO_TAG_RE = re.compile(r"^go\d+\.\d+(\.\d+)?$")

GO_REPO_URL = "https://github.com/golang/go"
NAMES_AT_DIR = Path(__file__).resolve().parent / "names_at"


# --------------------------------------------------------------------------- clone
def ensure_clone(repo_dir: Path) -> None:
    """Clone golang/go blobless WITH history into repo_dir if absent (one-time, large)."""
    if repo_dir.is_dir() and (repo_dir / ".git").exists():
        LOGGER.info("[clone] golang/go already present at %s", repo_dir)
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[clone] cloning %s --filter=blob:none -> %s (large, one-time)", GO_REPO_URL, repo_dir)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", GO_REPO_URL, str(repo_dir)],
        check=True,
    )


# --------------------------------------------------------------------------- tags
def go_release_tags(repo_dir: Path) -> list[str]:
    """Return golang/go release tags (``go<major>.<minor>[.<patch>]``) by creatordate."""
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "tag", "--sort=creatordate"],
        capture_output=True, text=True, check=True,
    )
    tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
    return [t for t in tags if _GO_TAG_RE.match(t)]


# --------------------------------------------------------------------------- dumper
def _names_at_invocation() -> list[str]:
    """Build (if needed) and return the argv prefix that runs the go/parser dumper.

    Prefers the compiled binary (``names_at/names_at``); falls back to ``go run .``.
    Requires Go on PATH (caller sets PATH/GOPATH); we attempt a build once.
    """
    binary = NAMES_AT_DIR / "names_at"
    if binary.exists() and os.access(binary, os.X_OK):
        return [str(binary)]
    # Try to build it.
    LOGGER.info("[dumper] building names_at in %s", NAMES_AT_DIR)
    build = subprocess.run(
        ["go", "build", "-o", "names_at", "."],
        cwd=str(NAMES_AT_DIR), capture_output=True, text=True,
    )
    if build.returncode == 0 and binary.exists():
        return [str(binary)]
    LOGGER.warning("[dumper] build failed (%s); falling back to `go run .`", build.stderr.strip())
    return ["go", "run", "."]


def names_at_tag(invocation: list[str], src_dir: Path) -> set[str]:
    """Run the dumper on src_dir, returning the set of emitted node names (unprefixed)."""
    # `go run .` must execute inside the module dir; a built binary can run anywhere.
    cwd = str(NAMES_AT_DIR) if invocation[0] == "go" else None
    proc = subprocess.run(
        [*invocation, "-src", str(src_dir)],
        cwd=cwd, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        LOGGER.warning("[dumper] non-zero exit on %s: %s", src_dir, proc.stderr.strip()[:500])
    return {ln for ln in proc.stdout.splitlines() if ln}


# --------------------------------------------------------------------------- births
def tag_namesets(
    repo_dir: Path,
    selected: list[tuple[str, int]],
    invocation: list[str],
) -> dict[int, set[str]]:
    """Checkout each (tag, epoch), dump names from src/; ALWAYS restore the starting ref."""
    epoch_namesets: dict[int, set[str]] = {}
    src_dir = repo_dir / "src"

    # Capture starting ref so we can always restore it.
    try:
        ref = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if ref == "HEAD":
            ref = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        LOGGER.error("[go] could not determine starting ref: %s", exc)
        return epoch_namesets

    try:
        for tag, epoch in selected:
            try:
                subprocess.run(
                    ["git", "-C", str(repo_dir), "checkout", "-q", "-f", tag],
                    capture_output=True, text=True, check=True,
                )
            except subprocess.CalledProcessError as exc:
                LOGGER.warning("[go] checkout %s failed: %s", tag, exc)
                continue
            if not src_dir.is_dir():
                LOGGER.warning("[go] tag=%s has no src/ dir — skipping", tag)
                continue
            ns = names_at_tag(invocation, src_dir)
            epoch_namesets[epoch] = ns
            LOGGER.info("[go] tag=%s epoch=%d  %d names", tag, epoch, len(ns))
    finally:
        try:
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", "-q", "-f", ref],
                capture_output=True, text=True, check=True,
            )
            LOGGER.info("[go] restored to %s", ref)
        except subprocess.CalledProcessError as exc:
            LOGGER.error("[go] FAILED to restore ref %s: %s", ref, exc)

    return epoch_namesets


# --------------------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description="Build births.npz (Go release-tag epoch axis) for the Go std adapter"
    )
    ap.add_argument(
        "--data",
        default=str(ROOT / "cache" / "code_gap" / "go" / "data_rta"),
        help="Dir containing data.npz (defines the node-name order). births.npz is "
        "written here AND into every sibling data_* dir that shares the node set.",
    )
    ap.add_argument("--E", type=int, default=6, help="Number of shared epochs (default: 6)")
    ap.add_argument(
        "--repo",
        default=str(ROOT / "cache" / "code_gap" / "go" / "repos" / "go"),
        help="golang/go clone dir (cloned blobless+history if absent)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data)
    repo_dir = Path(args.repo)
    E = args.E

    # Load data.npz names (and is_F/is_B for resolution-rate split).
    d = np.load(data_dir / "data.npz", allow_pickle=True)
    data_names: list[str] = [str(n) for n in d["names"]]
    is_F = d["is_F"].astype(bool)
    is_B = d["is_B"].astype(bool)
    N = len(data_names)
    LOGGER.info("Loaded %d node names from %s", N, data_dir / "data.npz")

    # Clone golang/go (one-time) and pick release-tag epochs.
    ensure_clone(repo_dir)
    tags = go_release_tags(repo_dir)
    if not tags:
        LOGGER.error("[go] no release tags matched %s — aborting", _GO_TAG_RE.pattern)
        sys.exit(1)
    selected = select_epoch_tags(tags, E)
    LOGGER.info("[go] %d release tags -> %d epochs: %s", len(tags), len(selected), selected)

    invocation = _names_at_invocation()
    LOGGER.info("[dumper] invocation: %s", " ".join(invocation))

    epoch_namesets = tag_namesets(repo_dir, selected, invocation)

    # Compute birth array (names are unprefixed — Go data names are unprefixed).
    birth = _compute_births_fast(epoch_namesets, data_names, E)
    tags_list = [f"e{i}" for i in range(E)]

    # Write births.npz into every sibling data_* dir that shares this node set.
    out_dirs = _sibling_data_dirs(data_dir, data_names)
    for out in out_dirs:
        np.savez(out / "births.npz", birth=birth, tags=np.array(tags_list, dtype=object))
        LOGGER.info("Wrote %s", out / "births.npz")

    # Resolution-rate report.
    resolved = birth < E

    def pct(mask: np.ndarray) -> str:
        return "n/a" if mask.sum() == 0 else f"{resolved[mask].mean() * 100:.1f}%"

    print(
        f"\n[resolution] overall={resolved.mean()*100:.1f}%  "
        f"F={pct(is_F)}  B={pct(is_B)}  "
        f"(nodes with birth==SENTINEL never appeared in any epoch tag)",
        flush=True,
    )
    print(
        f"[births] E={E}  SENTINEL={E}  resolved={int(resolved.sum())}/{N}  "
        f"unresolved={int((~resolved).sum())}  wrote_into={[str(o) for o in out_dirs]}",
        flush=True,
    )

    provenance = (
        "python -m artifacts_scripts.curriculum.gap_recover.adapters.go.build_births "
        + " ".join(sys.argv[1:])
    )
    print(f"\n[provenance] regenerate with:\n    {provenance}", flush=True)


def _sibling_data_dirs(data_dir: Path, data_names: list[str]) -> list[Path]:
    """Return data_dir plus any sibling ``data*`` dir whose data.npz has the same node
    names (the cha/rta resolutions share one node set, so one births array serves all)."""
    out = [data_dir]
    parent = data_dir.parent
    want = data_names
    for sib in sorted(parent.glob("data*")):
        if sib == data_dir or not (sib / "data.npz").exists():
            continue
        try:
            sn = [str(n) for n in np.load(sib / "data.npz", allow_pickle=True)["names"]]
        except Exception:  # noqa: BLE001
            continue
        if sn == want:
            out.append(sib)
        else:
            LOGGER.info("[go] sibling %s has a different node set — not writing births there", sib)
    return out


if __name__ == "__main__":
    main()
