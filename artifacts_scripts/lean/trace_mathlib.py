"""Trace mathlib4 with LeanDojo to produce the interactive ``Dojo`` environment.

This is the long-pole build for the *interactive* arm of the Lean proving
experiment: LeanDojo clones mathlib4 at a pinned commit, builds it from source
with instrumentation, and emits a ``TracedRepo`` whose per-theorem tactic
states can later be driven via ``Dojo.run_tac``. We trace the **same** commit
the decl-dependency graph was extracted from (mathlib4 ``v4.30.0`` /
``c5ea00351``; see ``docs/lean_decl_graph.md``) so the curriculum graph and the
prover stay on one mathlib version.

Compatibility (resolved 2026-05-31, lean-dojo 4.20.0):
  - ``lean_dojo.data_extraction.lean.is_supported_version`` accepts any Lean
    ``>= v4.3.0-rc2`` with no upper bound, so v4.30.0 passes cleanly.
  - LeanDojo 4.x traces **natively** (no Docker; the old ``DISABLE_DOCKER``
    knob is gone). It clones the repo from GitHub by commit into its own cache
    (``~/.cache/lean_dojo``, outside this git repo) and builds there -- it does
    NOT reuse ``~/src/mathlib4``.

This script is intentionally thin: configuration is fixed to the coherent
commit and the heavy lifting is LeanDojo's. Run it under ``bookgap.jobs`` (see
``run_trace.sh``) so the from-source build lives in a tracked systemd cgroup
with a flushed sidecar log.
"""

from __future__ import annotations

import os
import sys
import time

# Pinned to match cache/lean/decl_graph (docs/lean_decl_graph.md): mathlib4
# stable tag v4.30.0. The full hash LeanDojo resolves this to is
# c5ea00351c28e24afc9f0f84379aa41082b1188f.
MATHLIB_URL = "https://github.com/leanprover-community/mathlib4"
MATHLIB_COMMIT = "c5ea00351"


def _flush_print(*args: object) -> None:
    print(*args, flush=True)


def main() -> int:
    # Fail fast and loud if elan's lean/lake are not visible: LeanDojo shells
    # out to lake to build mathlib from source.
    import shutil

    for tool in ("lean", "lake", "elan", "git"):
        path = shutil.which(tool)
        _flush_print(f"[trace] {tool}: {path or '<MISSING>'}")
        if path is None and tool != "elan":
            _flush_print(f"[trace] FATAL: {tool} not on PATH; aborting.")
            return 2

    from lean_dojo import LeanGitRepo, trace

    repo = LeanGitRepo(MATHLIB_URL, MATHLIB_COMMIT)
    _flush_print(f"[trace] repo url       = {repo.url}")
    _flush_print(f"[trace] repo commit    = {repo.commit}")
    _flush_print(f"[trace] lean_version   = {repo.lean_version}")
    _flush_print(f"[trace] repo_type      = {repo.repo_type}")

    cache_dir = os.environ.get("CACHE_DIR", os.path.expanduser("~/.cache/lean_dojo"))
    _flush_print(f"[trace] lean_dojo cache = {cache_dir}")
    _flush_print(f"[trace] DISABLE_REMOTE_CACHE = {os.environ.get('DISABLE_REMOTE_CACHE', '<unset>')}")
    _flush_print(f"[trace] GITHUB_ACCESS_TOKEN set = {bool(os.environ.get('GITHUB_ACCESS_TOKEN'))}")

    t0 = time.time()
    _flush_print(f"[trace] starting trace() of full mathlib4 (build_deps=True) ...")
    # dst_dir=None -> LeanDojo stores the TracedRepo under its own cache and
    # returns an in-memory handle. The cache is the durable artifact.
    traced = trace(repo)
    dt = time.time() - t0
    _flush_print(f"[trace] trace() complete in {dt/3600:.2f}h")
    _flush_print(f"[trace] traced root = {getattr(traced, 'root_dir', '<unknown>')}")
    try:
        thms = traced.get_traced_theorems()
        _flush_print(f"[trace] traced theorems = {len(thms)}")
    except Exception as exc:  # noqa: BLE001 - best-effort summary only
        _flush_print(f"[trace] (could not count theorems: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
