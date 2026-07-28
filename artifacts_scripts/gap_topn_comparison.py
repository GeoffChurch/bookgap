"""Top-N gap comparison: K-ball vs HDBSCAN maximal-bookless on a single topic.

Takes a K-ball CSV (from kball_gap_probe) and a maximal-bookless CSV
(from maximal_bookless_extractor), filters both to the largest N gaps,
and emits a side-by-side comparison of size distributions and structural
metrics.

The "largest-N" framing matches the paper's "largest gap per topic"
methodology and avoids min-size-floor arbitrariness.

Usage::

    python -m artifacts_scripts.gap_topn_comparison \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__kball__rf0.75__num200.csv \\
        --maximal-bookless-csv artifacts/maximal_bookless/T10181_no_denoise.csv \\
        --top-n 20

Output: stdout markdown table + saved comparison CSV at
``artifacts/topn_comparison/{topic_id}__top{N}.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _load_kball_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_idx"]),
                "size": int(r["gap_size"]),
                "topic_id": r["topic_id"],
                "topic_name": r.get("topic_name", ""),
                "percentile": float(r["gap_size_percentile"]) if r["gap_size_percentile"].lower() != "nan" else float("nan"),
                "zscore": float(r["gap_zscore"]) if r["gap_zscore"].lower() != "nan" else float("nan"),
            })
    return rows


def _load_maximal_bookless_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            article_idxs = json.loads(r["article_indices_json"])
            rows.append({
                "gap_id": int(r["gap_id"]),
                "size": int(r["n_articles"]),
                "n_boundary_books": int(r["n_boundary_books"]),
                "dominant_topic_id": r["dominant_topic_id"],
                "dominant_topic_count": int(r["dominant_topic_count"]),
                "n_topics_spanned": int(r["n_topics_spanned"]),
                "article_indices": article_idxs,
            })
    return rows


def _topic_id_from_path(path: Path) -> str:
    """Extract topic id from a filename like 17-1702-T10181__... or T10181_..."""
    stem = path.stem
    # Try 17-1702-T10181 form
    parts = stem.split("__")[0].split("-")
    if len(parts) >= 3 and parts[-1].startswith("T"):
        return parts[-1]
    # Try T10181_no_denoise
    if stem.startswith("T") and "_" in stem:
        return stem.split("_")[0]
    return stem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kball-csv", required=True)
    p.add_argument("--maximal-bookless-csv", required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--filter-topic", default=None,
                   help="Filter K-ball CSV to a single topic_id (e.g. T10181). "
                        "Useful when the K-ball CSV is from a multi-topic scope "
                        "but the max-bookless CSV is single-topic.")
    p.add_argument("--output-dir", default="artifacts/topn_comparison")
    args = p.parse_args()

    kball = _load_kball_csv(Path(args.kball_csv))
    mb = _load_maximal_bookless_csv(Path(args.maximal_bookless_csv))

    topic_id = args.filter_topic or _topic_id_from_path(Path(args.maximal_bookless_csv))
    if args.filter_topic is not None:
        kball = [r for r in kball if r["topic_id"] == args.filter_topic]

    # Sort descending by size, take top-N.
    kball_sorted = sorted(kball, key=lambda r: -r["size"])
    mb_sorted = sorted(mb, key=lambda r: -r["size"])
    kball_topn = kball_sorted[:args.top_n]
    mb_topn = mb_sorted[:args.top_n]

    # ------------------------------------------------------------------
    # Side-by-side stdout table
    # ------------------------------------------------------------------
    print(f"## Top-{args.top_n} gap comparison: K-ball vs HDBSCAN max-bookless")
    print(f"  topic: {topic_id}")
    print(f"  K-ball CSV: {args.kball_csv}")
    print(f"  max-bookless CSV: {args.maximal_bookless_csv}")
    print()
    print(f"  K-ball: {len(kball)} total gaps; took top {len(kball_topn)}")
    print(f"  max-bookless: {len(mb)} total gaps; took top {len(mb_topn)}")
    print()

    def _stats(label, sizes):
        if not sizes:
            return f"  {label}: no gaps"
        s = sorted(sizes)
        n = len(s)
        return (f"  {label}: n={n}, "
                f"min={s[0]}, q1={s[n//4]}, median={s[n//2]}, "
                f"q3={s[3*n//4]}, max={s[-1]}, mean={statistics.mean(s):.1f}")

    print(f"### Size distribution (top-{args.top_n})")
    print(_stats("K-ball  ", [r["size"] for r in kball_topn]))
    print(_stats("max-bookless", [r["size"] for r in mb_topn]))
    print()

    print(f"### Side-by-side table (rank, K-ball size, max-bookless size)")
    print()
    print(f"  {'rank':>4}  {'K-ball':>8}  {'max-bookless':>14}  ratio")
    n_rows = max(len(kball_topn), len(mb_topn))
    for i in range(n_rows):
        k = kball_topn[i]["size"] if i < len(kball_topn) else None
        m = mb_topn[i]["size"] if i < len(mb_topn) else None
        ratio = (k / m) if k is not None and m is not None and m > 0 else None
        print(f"  {i+1:>4}  {k if k is not None else '—':>8}  "
              f"{m if m is not None else '—':>14}  "
              f"{ratio:>5.2f}" if ratio is not None else
              f"  {i+1:>4}  {k if k is not None else '—':>8}  "
              f"{m if m is not None else '—':>14}  —")
    print()

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{topic_id}__top{args.top_n}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "topic_id",
            "kball_gap_id", "kball_size",
            "kball_percentile", "kball_zscore",
            "mb_gap_id", "mb_size",
            "mb_n_boundary_books", "mb_n_topics_spanned",
        ])
        for i in range(n_rows):
            k = kball_topn[i] if i < len(kball_topn) else None
            m = mb_topn[i] if i < len(mb_topn) else None
            w.writerow([
                i + 1, topic_id,
                k["gap_id"] if k else "",
                k["size"] if k else "",
                f"{k['percentile']:.4f}" if k else "",
                f"{k['zscore']:.4f}" if k else "",
                m["gap_id"] if m else "",
                m["size"] if m else "",
                m["n_boundary_books"] if m else "",
                m["n_topics_spanned"] if m else "",
            ])
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
