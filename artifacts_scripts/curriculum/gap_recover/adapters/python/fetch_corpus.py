"""Clone the curated Python corpus as blobless partial clones (history + tags,
blobs fetched on demand). Idempotent: re-running fetches updates instead of
re-cloning. Failures are logged and skipped (no silent caps — the summary lists
exactly which repos are missing).

    python -m artifacts_scripts.curriculum.gap_recover.adapters.python.fetch_corpus
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from .corpus import PACKAGES, REPO_ROOT


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-400:]


def _dir_size_mb(path: Path) -> float:
    rc, out = _run(["du", "-sm", str(path)], timeout=120)
    try:
        return float(out.split()[0])
    except Exception:
        return -1.0


def fetch_one(name: str, url: str) -> tuple[str, bool, str]:
    dest = REPO_ROOT / name
    if dest.exists():
        rc, msg = _run(["git", "-C", str(dest), "fetch", "--tags",
                        "--filter=blob:none", "origin"])
        action = "fetch"
    else:
        rc, msg = _run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        url, str(dest)])
        if rc == 0:
            # check out the default branch so HEAD has a working tree to scan
            rc, msg = _run(["git", "-C", str(dest), "checkout"])
        action = "clone"
    ok = rc == 0
    return action, ok, msg


def main() -> int:
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok_list: list[str] = []
    fail_list: list[tuple[str, str]] = []
    for i, (name, url, _ntags) in enumerate(PACKAGES, 1):
        ts = time.time()
        try:
            action, ok, msg = fetch_one(name, url)
        except subprocess.TimeoutExpired:
            action, ok, msg = "clone", False, "TIMEOUT"
        if ok:
            sz = _dir_size_mb(REPO_ROOT / name)
            ok_list.append(name)
            print(f"[{i:2d}/{len(PACKAGES)}] {name:14s} {action} OK "
                  f"({sz:.0f} MB, {time.time()-ts:.0f}s)", flush=True)
        else:
            fail_list.append((name, msg.strip().replace("\n", " ")[-200:]))
            print(f"[{i:2d}/{len(PACKAGES)}] {name:14s} {action} FAILED: {msg.strip()[-160:]}",
                  flush=True)
    print(f"\n[done] {len(ok_list)}/{len(PACKAGES)} repos ready in "
          f"{time.time()-t0:.0f}s -> {REPO_ROOT}", flush=True)
    if fail_list:
        print("[FAILED] " + ", ".join(n for n, _ in fail_list), flush=True)
        for n, m in fail_list:
            print(f"   {n}: {m}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
