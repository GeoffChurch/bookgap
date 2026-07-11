"""Blobless-clone the curated Rust corpus (history + tags, blobs on demand). Idempotent;
failures logged and skipped (no silent caps).

    python -m artifacts_scripts.gap_recover.adapters.rust.fetch_corpus
"""
from __future__ import annotations

import subprocess
import sys
import time

from .corpus import PACKAGES, REPO_ROOT


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-300:]


def fetch_one(name: str, url: str):
    dest = REPO_ROOT / name
    if dest.exists():
        rc, msg = _run(["git", "-C", str(dest), "fetch", "--tags",
                        "--filter=blob:none", "origin"])
        return "fetch", rc == 0, msg
    rc, msg = _run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)])
    if rc == 0:
        rc, msg = _run(["git", "-C", str(dest), "checkout"])
    return "clone", rc == 0, msg


def main() -> int:
    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok, fail = [], []
    for i, (name, url, _n) in enumerate(PACKAGES, 1):
        try:
            action, good, msg = fetch_one(name, url)
        except subprocess.TimeoutExpired:
            action, good, msg = "clone", False, "TIMEOUT"
        if good:
            ok.append(name)
            print(f"[{i:2d}/{len(PACKAGES)}] {name:12s} {action} OK", flush=True)
        else:
            fail.append((name, msg.strip().replace("\n", " ")[-160:]))
            print(f"[{i:2d}/{len(PACKAGES)}] {name:12s} {action} FAILED: {msg.strip()[-120:]}",
                  flush=True)
    print(f"\n[done] {len(ok)}/{len(PACKAGES)} repos in {time.time()-t0:.0f}s -> {REPO_ROOT}",
          flush=True)
    if fail:
        print("[FAILED] " + ", ".join(n for n, _ in fail), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
