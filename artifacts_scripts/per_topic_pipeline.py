"""Per-topic K-ball + LLR refinement runner for a subfield.

For each topic in the given subfield, generate a per-topic K-ball CSV (if
not already cached), then run LLR refinement with the v2-auto calibration
phase (joint K_query_stats + member subsample) and σ-stratified-dist
null. Outputs per-topic refined CSVs under
``artifacts/gap_kball_llr_refine/per_topic/``.

The per-topic embedding indices are scope-local, so K-ball outputs from a
subfield-scoped run can NOT be reused at per-topic scope — we regenerate
K-ball per topic. K-ball at per-topic scope is fast (~30-60s/topic).

Topic list is taken from the FIRST positional argument (a subfield-scoped
K-ball CSV) — just to enumerate which topics exist in the subfield. The
indices in that CSV are not used.

Usage::

    uv run python -u -m artifacts_scripts.per_topic_pipeline \\
        --subfield 17/1702 \\
        --topic-source artifacts/kball_gap_probe/17-1702__pca__drop0.csv \\
        --num-gaps 3000 \\
        --radius-fraction 0.75 \\
        --out-dir artifacts/gap_kball_llr_refine/per_topic
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

# The interpreter running this script, so subprocesses inherit the same venv
# without hardcoding a path to one particular checkout.
PY = sys.executable


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subfield", required=True, help="e.g. 17/1702")
    p.add_argument("--topic-source", required=True, type=Path,
                   help="Subfield-scoped K-ball CSV whose topic_id column "
                        "is used to enumerate topics. Article indices in "
                        "the file are NOT reused.")
    p.add_argument("--num-gaps", type=int, default=3000)
    p.add_argument("--radius-fraction", type=float, default=0.75)
    p.add_argument("--out-dir", type=Path,
                   default=Path("artifacts/gap_kball_llr_refine/per_topic"))
    p.add_argument("--kball-dir", type=Path,
                   default=Path("artifacts/kball_gap_probe"))
    p.add_argument("--start-from", type=int, default=0,
                   help="Skip the first N topics (for restart). Default 0.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N topics (for testing).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.kball_dir.mkdir(parents=True, exist_ok=True)

    topics: list[str] = []
    with open(args.topic_source) as f:
        seen: set[str] = set()
        for r in csv.DictReader(f):
            t = r["topic_id"]
            if t not in seen:
                seen.add(t)
                topics.append(t)
    topics.sort()
    print(f"[plan] {len(topics)} topics in {args.subfield}: "
          f"{topics[0]}, {topics[1]}, ..., {topics[-1]}", flush=True)

    if args.limit is not None:
        topics = topics[args.start_from : args.start_from + args.limit]
    else:
        topics = topics[args.start_from:]

    started_at = time.time()
    for i, topic in enumerate(topics, 1):
        scope = f"{args.subfield}/{topic}"
        safe_scope = scope.replace("/", "-")
        kball_csv = args.kball_dir / f"{safe_scope}__pca__drop0.csv"
        refined_csv = args.out_dir / f"{topic}__llr_refined.csv"

        elapsed = time.time() - started_at
        print(f"\n=== [{i}/{len(topics)}] topic {topic} "
              f"(elapsed {elapsed/3600:.1f}h) ===", flush=True)

        if refined_csv.exists() and refined_csv.stat().st_size > 5000:
            print(f"  refined CSV exists ({refined_csv.stat().st_size} bytes), "
                  f"skipping", flush=True)
            continue

        # Step 1: K-ball if not already cached. Check size > 1500 to avoid
        # accepting stale 5-gap placeholders that exist for some topics.
        need_kball = (
            not kball_csv.exists()
            or kball_csv.stat().st_size < 50000
        )
        if need_kball:
            print(f"  [kball] generating K-ball for {scope}...", flush=True)
            t0 = time.time()
            r = subprocess.run([
                PY, "-u", "-m", "artifacts_scripts.kball_gap_probe",
                "--scope", scope,
                "--denoiser", "pca", "--drop-k", "0",
                "--radius-fraction", str(args.radius_fraction),
                "--num-gaps", str(args.num_gaps),
            ])
            if r.returncode != 0:
                print(f"  [kball] FAILED (rc={r.returncode}); skipping topic",
                      flush=True)
                continue
            print(f"  [kball] done in {time.time() - t0:.0f}s", flush=True)
        else:
            print(f"  [kball] cached at {kball_csv}", flush=True)

        # Step 2: LLR refinement
        print(f"  [refine] running LLR refinement...", flush=True)
        t0 = time.time()
        r = subprocess.run([
            PY, "-u", "-m", "artifacts_scripts.gap_kball_llr_refine",
            "--scope", scope,
            "--kball-csv", str(kball_csv),
            "--out", str(refined_csv),
            "--null-method", "stein",
            # v2-auto defaults: --k-query-stats=auto, --member-subsample=auto,
            # --null-stratify=dist (now the default after this PR).
        ])
        if r.returncode != 0:
            print(f"  [refine] FAILED (rc={r.returncode})", flush=True)
            continue
        print(f"  [refine] done in {time.time() - t0:.0f}s", flush=True)

    total = time.time() - started_at
    print(f"\n[done] {len(topics)} topics in {total/3600:.2f}h "
          f"(mean {total/max(len(topics),1):.0f}s/topic)", flush=True)


if __name__ == "__main__":
    main()
