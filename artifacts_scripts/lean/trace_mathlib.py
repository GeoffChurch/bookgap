"""Trace mathlib4 with LeanDojo to produce the interactive ``Dojo`` environment.

This is the long-pole build for the *interactive* arm of the Lean proving
experiment: LeanDojo clones mathlib4 at a pinned commit, builds it from source
with instrumentation, and emits a ``TracedRepo`` whose per-theorem tactic
states can later be driven via ``Dojo.run_tac``. We trace the **same** commit
the decl-dependency graph was extracted from (mathlib4 ``v4.20.0`` /
``c211948581``; see ``docs/lean_decl_graph.md``) so the curriculum graph and the
prover stay on one mathlib version.

Version choice (resolved 2026-05-31, lean-dojo 4.20.0):
  - LeanDojo 4.20.0's ``is_supported_version`` has no upper bound, so newer
    Lean *passes the version gate* but its bundled Lean metaprograms do not
    actually work on it. The v4.30.0 trace built mathlib from source then died
    at AST parse (``ast.py:1480 assert len(children) == 2`` in
    ``CommandSectionNode.from_data``) -- the Lean 4.25-4.27 String/AST
    overhaul changed the ``section`` command's syntax-tree shape, and the
    String-API change had already forced two earlier metaprogram patches
    (``ExtractData``/``Lean4Repl``). Upstream ``lean-dojo`` (PyPI 4.20.0 ==
    latest; ``main`` unchanged on this path) has no fix, and ``lean-dojo-v2``
    is a different package with an incompatible API.
  - The fix is to trace the version LeanDojo 4.20.0 is itself pinned to:
    its repo ``lean-toolchain`` is ``leanprover/lean4:v4.20.0``. mathlib4 tag
    ``v4.20.0`` (commit ``c211948581bde9846a99e32d97a03f0d5307c31e``) is on
    that exact toolchain, predates the String/AST overhaul, and traces with
    the **pristine** (unpatched) LeanDojo metaprograms -- verified on a toy
    v4.20.0 repo containing ``section`` blocks (parses past ``ast.py:1480``).
  - The decl-graph (``cache/lean/decl_graph``) is re-extracted from this same
    v4.20.0 commit for coherence (``docs/lean_decl_graph.md``).
  - LeanDojo 4.x traces **natively** (no Docker; the old ``DISABLE_DOCKER``
    knob is gone). It clones the repo from GitHub by commit into its own cache
    (``~/.cache/lean_dojo``, outside this git repo) and builds there -- it does
    NOT reuse ``~/src/mathlib4`` (which stays on v4.30.0 for the Goedel
    whole-proof verifier).

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
# stable tag v4.20.0 (== LeanDojo 4.20.0's own pinned lean-toolchain). The full
# hash this tag resolves to is c211948581bde9846a99e32d97a03f0d5307c31e.
MATHLIB_URL = "https://github.com/leanprover-community/mathlib4"
MATHLIB_COMMIT = "c211948581bde9846a99e32d97a03f0d5307c31e"


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
