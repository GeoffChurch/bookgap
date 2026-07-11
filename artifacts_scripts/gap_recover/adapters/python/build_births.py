"""Build births.npz for the Python temporal-holdout evaluation.

For each repo in PACKAGES, maps its version tags onto a shared E-epoch axis and
records the earliest epoch at which each named node (as seen in data.npz) was
present.  Because node names are globally namespaced as ``{github}::{local}``
the per-repo epoch sets are always disjoint — union across repos per epoch is safe.

CLI::
    python -m artifacts_scripts.gap_recover.adapters.python.build_births
    python -m artifacts_scripts.gap_recover.adapters.python.build_births \\
        --data cache/code_gap/python/data --E 6
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]  # .../src/bookgap
sys.path.insert(0, str(ROOT))

from artifacts_scripts.gap_recover.adapters.python.corpus import (
    PACKAGES,
    REPO_ROOT,
)
from artifacts_scripts.gap_recover.adapters.python.build_data import (
    discover_package,
)
from artifacts_scripts.gap_recover.adapters.python.extract import (
    extract_repo,
)

LOGGER = logging.getLogger("build_births")

# Semver-ish regex: vX.Y or vX.Y.Z, no pre-release suffixes.
_SEMVER_RE = re.compile(r"^v?\d+\.\d+(\.\d+)?$")


# ---------------------------------------------------------------------------
# Pure functions (tested without git)
# ---------------------------------------------------------------------------

def select_epoch_tags(sorted_tags: list[str], E: int = 6) -> list[tuple[str, int]]:
    """Map a repo's chronologically-ordered tags onto E shared epochs.

    Args:
        sorted_tags: Tags in chronological order (oldest first).
        E:           Number of epochs (default 6).

    Returns:
        List of ``(tag, epoch)`` pairs sorted by epoch.  Deduplication is
        applied so each epoch maps to exactly one tag; for the ``m <= E`` case,
        the tag with the higher index (later in time) wins on collision.

    Algorithm:
        - If ``m == 1``: single tag -> epoch ``E-1``.
        - If ``1 < m <= E``: assign epoch ``round(i / (m-1) * (E-1))`` to the
          i-th tag.  When two tags collide on the same epoch the later tag
          (higher i) overwrites the earlier one.
        - If ``m > E``: pick E evenly-spaced indices
          ``round(j / (E-1) * (m-1))`` for j in ``0 .. E-1`` and assign
          them epochs ``0 .. E-1`` (no collisions possible by construction).
    """
    m = len(sorted_tags)
    if m == 0:
        return []

    epoch_to_tag: dict[int, str] = {}

    if m == 1:
        epoch_to_tag[E - 1] = sorted_tags[0]
    elif m <= E:
        for i, tag in enumerate(sorted_tags):
            epoch = round(i / (m - 1) * (E - 1))
            # Later tag (higher i) wins on collision.
            epoch_to_tag[epoch] = tag
    else:
        for j in range(E):
            idx = round(j / (E - 1) * (m - 1))
            epoch_to_tag[j] = sorted_tags[idx]

    return [(tag, epoch) for epoch, tag in sorted(epoch_to_tag.items())]


def compute_births(
    epoch_namesets: dict[int, set[str]],
    data_names: list[str],
    E: int,
) -> np.ndarray:
    """Compute per-node birth epoch aligned to ``data_names``.

    Args:
        epoch_namesets: ``{epoch_index: set_of_namespaced_node_names}``.
        data_names:     Node names in data.npz order.
        E:              Number of epochs (SENTINEL value when never seen).

    Returns:
        int16 array of length ``len(data_names)``.  ``birth[i] = E`` means
        the node was never seen across all checked epochs (SENTINEL).
    """
    birth = np.full(len(data_names), E, dtype=np.int16)
    for epoch, names in epoch_namesets.items():
        for i, name in enumerate(data_names):
            if name in names and epoch < birth[i]:
                birth[i] = epoch
    return birth


def _compute_births_fast(
    epoch_namesets: dict[int, set[str]],
    data_names: list[str],
    E: int,
) -> np.ndarray:
    """Vectorised variant of compute_births (used in main for large corpora)."""
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(data_names)}
    birth = np.full(len(data_names), E, dtype=np.int16)
    for epoch, names in sorted(epoch_namesets.items()):
        for name in names:
            i = name_to_idx.get(name)
            if i is not None and epoch < birth[i]:
                birth[i] = epoch
    return birth


# ---------------------------------------------------------------------------
# Git-driving functions (not called in tests)
# ---------------------------------------------------------------------------

def version_tags(repo_dir: str | Path) -> list[str]:
    """Return semver-ish tags for ``repo_dir`` in chronological order.

    Runs ``git -C <repo> tag --sort=creatordate`` and keeps only tags matching
    ``^v?\\d+\\.\\d+(\\.\\d+)?$`` (no rc/alpha/beta/dev pre-releases).
    """
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "tag", "--sort=creatordate"],
        capture_output=True,
        text=True,
        check=True,
    )
    tags = [t.strip() for t in result.stdout.splitlines() if t.strip()]
    return [t for t in tags if _SEMVER_RE.match(t)]


def tag_namesets(
    repo_dir: str | Path,
    github_name: str,
    selected: list[tuple[str, int]],
) -> dict[int, set[str]]:
    """For each ``(tag, epoch)`` in ``selected``, checkout the tag, extract node names.

    The repo is ALWAYS restored to its original ref via a try/finally, even if
    individual checkouts or extractions fail (those are logged and skipped).

    Ref-capture strategy:
        ``git rev-parse --abbrev-ref HEAD`` is run up front.  If the result is
        ``"HEAD"`` (detached HEAD state), we fall back to ``git rev-parse HEAD``
        to get the commit hash so the restore checkout is unambiguous.

    Args:
        repo_dir:    Path to the cloned repo.
        github_name: Used to prefix names as ``{github_name}::{local_name}``.
        selected:    Output of :func:`select_epoch_tags`.

    Returns:
        ``{epoch: set_of_namespaced_names}``
    """
    repo_dir = Path(repo_dir)
    epoch_namesets: dict[int, set[str]] = {}

    # Capture starting ref so we can always restore it.
    try:
        ref_result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        starting_ref = ref_result.stdout.strip()
        if starting_ref == "HEAD":
            # Detached HEAD — use the commit hash instead.
            ref_result2 = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            starting_ref = ref_result2.stdout.strip()
    except subprocess.CalledProcessError as exc:
        LOGGER.error("[%s] could not determine starting ref: %s", github_name, exc)
        return epoch_namesets

    try:
        # Discover the package from HEAD (same discovery logic as build_data).
        pkg_found = discover_package(repo_dir)
        if pkg_found is None:
            LOGGER.warning("[%s] no package found; skipping all tags", github_name)
            return epoch_namesets
        import_name, _ = pkg_found

        for tag, epoch in selected:
            try:
                subprocess.run(
                    ["git", "-C", str(repo_dir), "checkout", "-q", tag],
                    capture_output=True, text=True, check=True,
                )
            except subprocess.CalledProcessError as exc:
                LOGGER.warning("[%s] checkout %s failed: %s", github_name, tag, exc)
                continue

            try:
                r = extract_repo(repo_dir, import_name)
                ns = {f"{github_name}::{n}" for n in r.names}
                epoch_namesets[epoch] = ns
                LOGGER.info(
                    "[%s] tag=%s epoch=%d  %d names",
                    github_name, tag, epoch, len(ns),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "[%s] extract_repo at tag %s failed: %s", github_name, tag, exc
                )
    finally:
        # Always restore — even if something above raised an unexpected exception.
        try:
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", "-q", starting_ref],
                capture_output=True, text=True, check=True,
            )
            LOGGER.info("[%s] restored to %s", github_name, starting_ref)
        except subprocess.CalledProcessError as exc:
            LOGGER.error(
                "[%s] FAILED to restore ref %s: %s", github_name, starting_ref, exc
            )

    return epoch_namesets


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Build births.npz (epoch-normalized cross-repo tag axis) for Python adapter"
    )
    ap.add_argument(
        "--data",
        default=str(ROOT / "cache" / "code_gap" / "python" / "data"),
        help="Directory containing data.npz (output also written here)",
    )
    ap.add_argument(
        "--E",
        type=int,
        default=6,
        help="Number of shared epochs (default: 6)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data)
    E = args.E

    # Load data.npz names (and is_F/is_B for resolution-rate split).
    d = np.load(data_dir / "data.npz", allow_pickle=True)
    data_names: list[str] = [str(n) for n in d["names"]]
    is_F: np.ndarray = d["is_F"].astype(bool)
    is_B: np.ndarray = d["is_B"].astype(bool)
    N = len(data_names)
    LOGGER.info("Loaded %d node names from %s", N, data_dir / "data.npz")

    # Global epoch_namesets: union across repos (disjoint by github:: prefix).
    global_epoch_namesets: dict[int, set[str]] = {}

    for github_name, _url, _n_tags in PACKAGES:
        repo_dir = REPO_ROOT / github_name
        if not repo_dir.is_dir():
            LOGGER.warning("[%s] repo dir not found at %s — skipping", github_name, repo_dir)
            continue

        tags = version_tags(repo_dir)
        if not tags:
            LOGGER.warning("[%s] no semver tags found — skipping", github_name)
            continue

        selected = select_epoch_tags(tags, E)
        LOGGER.info(
            "[%s] %d semver tags -> %d epochs: %s",
            github_name, len(tags), len(selected),
            [(t, e) for t, e in selected],
        )

        repo_ns = tag_namesets(repo_dir, github_name, selected)
        for epoch, ns in repo_ns.items():
            if epoch not in global_epoch_namesets:
                global_epoch_namesets[epoch] = set()
            global_epoch_namesets[epoch].update(ns)

    # Compute birth array.
    birth = _compute_births_fast(global_epoch_namesets, data_names, E)
    tags_list = [f"e{i}" for i in range(E)]

    # Save births.npz.
    out_path = data_dir / "births.npz"
    np.savez(out_path, birth=birth, tags=np.array(tags_list, dtype=object))
    LOGGER.info("Wrote %s", out_path)

    # Resolution-rate report.
    resolved = birth < E

    def pct(mask: np.ndarray) -> str:
        if mask.sum() == 0:
            return "n/a"
        return f"{resolved[mask].mean() * 100:.1f}%"

    print(
        f"\n[resolution] overall={resolved.mean()*100:.1f}%  "
        f"F={pct(is_F)}  B={pct(is_B)}  "
        f"(nodes with birth==SENTINEL never appeared in any epoch tag)",
        flush=True,
    )
    print(
        f"[births] E={E}  SENTINEL={E}  "
        f"resolved={int(resolved.sum())}/{N}  "
        f"unresolved={int((~resolved).sum())}",
        flush=True,
    )

    provenance = (
        "python -m artifacts_scripts.gap_recover.adapters.python.build_births "
        + " ".join(sys.argv[1:])
    )
    print(f"\n[provenance] regenerate with:\n    {provenance}", flush=True)


if __name__ == "__main__":
    main()
