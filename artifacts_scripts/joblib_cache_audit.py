"""One-off audit of bookgap's joblib cache to identify orphaned entries.

Walks ``cache/joblib`` and for each memoised function reports:
  * module.function path
  * total on-disk size + number of cached argshash entries
  * whether the function is still importable
  * whether joblib's stored ``func_code.py`` matches the current source

Categories printed at the end:

  1. **Dead function** — the target function no longer exists at
     ``<module>.<function>``. Safe to ``rm -rf`` the whole dir.
  2. **Source mismatch** — function exists but its source has changed.
     joblib would invalidate on the next call anyway; pre-deleting just
     skips the first-call penalty and reclaims disk immediately.
  3. **Live, source matches** — entries are reachable. Argument-key
     orphans (entries built against old config hashes that will never be
     requested again) cannot be detected from the cache alone; the
     pragmatic move when you know a config-level default has shifted
     (HNSW defaults, default deduplicator, etc.) is to clear the
     affected function dirs by hand using the per-function sizes below.

Usage::

    uv run python -m artifacts_scripts.joblib_cache_audit
    uv run python -m artifacts_scripts.joblib_cache_audit --prune-dead
    uv run python -m artifacts_scripts.joblib_cache_audit --prune-source-mismatch

``--prune-*`` modes delete the corresponding dirs in place; without a
flag the script is read-only.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bookgap.util import CACHE_DIR


@dataclass
class FuncCacheEntry:
    module: str
    func_name: str
    func_dir: Path
    size_bytes: int
    n_arg_entries: int
    importable: bool
    source_matches: Optional[bool]  # None if can't check (not importable)


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _fmt_size(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:6.1f} {unit}"
        b /= 1024
    return f"{b:6.1f} PB"


def _stored_source(func_code_path: Path) -> str:
    """Read joblib's stored source, stripping its ``# first line: N`` header."""
    text = func_code_path.read_text()
    lines = text.splitlines()
    if lines and lines[0].startswith("# first line:"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _live_source(module_name: str, func_name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None
    func = getattr(mod, func_name, None)
    if func is None:
        return None
    # Unwrap lru_cache/TIMER.time wrappers if present so we compare to the
    # underlying def that joblib hashed.
    unwrapped = inspect.unwrap(func)
    try:
        return inspect.getsource(unwrapped).strip()
    except (OSError, TypeError):
        return None


def audit() -> list[FuncCacheEntry]:
    root = CACHE_DIR / "joblib"
    if not root.exists():
        print(f"No joblib cache at {root}")
        return []

    entries: list[FuncCacheEntry] = []
    for func_code in root.rglob("func_code.py"):
        func_dir = func_code.parent
        rel_parts = func_dir.relative_to(root).parts
        module_name = ".".join(rel_parts[:-1])
        func_name = rel_parts[-1]

        size = _dir_size(func_dir)
        n_entries = sum(
            1 for child in func_dir.iterdir()
            if child.is_dir() and child.name != "__pycache__"
        )

        live_src = _live_source(module_name, func_name)
        importable = live_src is not None
        source_matches: Optional[bool] = None
        if importable:
            stored = _stored_source(func_code)
            source_matches = stored == live_src

        entries.append(FuncCacheEntry(
            module=module_name,
            func_name=func_name,
            func_dir=func_dir,
            size_bytes=size,
            n_arg_entries=n_entries,
            importable=importable,
            source_matches=source_matches,
        ))
    return entries


def _print_table(entries: list[FuncCacheEntry]) -> None:
    entries = sorted(entries, key=lambda e: -e.size_bytes)
    header = f"{'module.function':<70} {'size':>10} {'#args':>6} {'import':>7} {'src':>5}"
    print(header)
    print("-" * len(header))
    for e in entries:
        imp = "YES" if e.importable else "NO"
        if e.source_matches is None:
            src = "-"
        else:
            src = "YES" if e.source_matches else "NO"
        print(
            f"{e.module + '.' + e.func_name:<70} "
            f"{_fmt_size(e.size_bytes):>10} {e.n_arg_entries:>6d} "
            f"{imp:>7} {src:>5}"
        )


def _summarise_and_maybe_prune(
    entries: list[FuncCacheEntry],
    prune_dead: bool,
    prune_mismatch: bool,
) -> None:
    dead = [e for e in entries if not e.importable]
    mismatch = [e for e in entries if e.importable and e.source_matches is False]
    live_ok = [e for e in entries if e.importable and e.source_matches is True]

    print()
    print(f"Dead functions:     {len(dead):>3}  ({_fmt_size(sum(e.size_bytes for e in dead))})")
    print(f"Source mismatches:  {len(mismatch):>3}  ({_fmt_size(sum(e.size_bytes for e in mismatch))})")
    print(f"Live + source OK:   {len(live_ok):>3}  ({_fmt_size(sum(e.size_bytes for e in live_ok))})")

    if dead:
        print("\nDead (safe to delete):")
        for e in dead:
            print(f"  {e.module}.{e.func_name}  ({_fmt_size(e.size_bytes)})")
    if mismatch:
        print("\nSource mismatch (joblib will invalidate on next call; pre-delete to skip wait):")
        for e in mismatch:
            print(f"  {e.module}.{e.func_name}  ({_fmt_size(e.size_bytes)})")

    to_delete: list[FuncCacheEntry] = []
    if prune_dead:
        to_delete.extend(dead)
    if prune_mismatch:
        to_delete.extend(mismatch)

    if to_delete:
        total = sum(e.size_bytes for e in to_delete)
        print(f"\nDeleting {len(to_delete)} directories ({_fmt_size(total)})...")
        for e in to_delete:
            shutil.rmtree(e.func_dir)
            print(f"  removed {e.func_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prune-dead", action="store_true",
                        help="rm -rf directories whose function no longer exists")
    parser.add_argument("--prune-source-mismatch", action="store_true",
                        help="rm -rf directories whose stored source differs from live")
    args = parser.parse_args()

    entries = audit()
    if not entries:
        return 0
    _print_table(entries)
    _summarise_and_maybe_prune(entries, args.prune_dead, args.prune_source_mismatch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
