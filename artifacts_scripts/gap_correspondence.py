"""Gap correspondence: K-ball vs max-bookless on the same article space.

Both inputs must contain ``article_indices_json`` populated in the same
indexing (scope-global indices from ``load_scope``). The K-ball bypass
path (``--denoiser pca --drop-k 0``) and the max-bookless extractor both
satisfy this when run on the same scope.

For every pair (mb-gap A, kball-gap B) with non-empty intersection we
compute:

- ``intersection``: |A ∩ B|
- ``jaccard``: |A∩B| / |A∪B|
- ``overlap_coef``: |A∩B| / min(|A|,|B|) (Szymkiewicz–Simpson — primary)
- ``cont_mb_to_kball``: |A∩B| / |A| (what fraction of mb-gap is in kball-gap)
- ``cont_kball_to_mb``: |A∩B| / |B| (what fraction of kball-gap is in mb-gap)

Outputs:

- ``{topic}__pairs.csv``     long-format, one row per non-zero pair
- ``{topic}__mb_best.csv``   one row per mb gap with its best K-ball partner
- ``{topic}__kball_best.csv`` one row per K-ball gap with its best mb partner
- stdout markdown summary: per-side top-N, phantom counts, coverage,
  decomposition counts, optimal 1-1 matching score.

Usage::

    python -m artifacts_scripts.gap_correspondence \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0.csv \\
        --maximal-bookless-csv artifacts/maximal_bookless/T10181_no_denoise.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def _load_kball_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            article_idxs = json.loads(r["article_indices_json"])
            rows.append({
                "gap_id": int(r["gap_idx"]),
                "size": int(r["gap_size"]),
                "topic_id": r["topic_id"],
                "article_set": frozenset(article_idxs),
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
                "article_set": frozenset(article_idxs),
            })
    return rows


def _topic_id_from_path(path: Path) -> str:
    stem = path.stem
    parts = stem.split("__")[0].split("-")
    if len(parts) >= 3 and parts[-1].startswith("T"):
        return parts[-1]
    if stem.startswith("T") and "_" in stem:
        return stem.split("_")[0]
    return stem


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    s = sorted(xs)
    n = len(s)
    return s[n // 4], s[n // 2], s[3 * n // 4]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kball-csv", required=True)
    p.add_argument("--maximal-bookless-csv", required=True)
    p.add_argument("--top-n", type=int, default=20,
                   help="Show top-N gaps per side in stdout summary")
    p.add_argument("--filter-topic", default=None,
                   help="If K-ball CSV is multi-topic, restrict to this topic id")
    p.add_argument("--output-dir", default="artifacts/gap_correspondence")
    args = p.parse_args()

    kball = _load_kball_csv(Path(args.kball_csv))
    mb = _load_maximal_bookless_csv(Path(args.maximal_bookless_csv))

    topic_id = args.filter_topic or _topic_id_from_path(Path(args.maximal_bookless_csv))
    if args.filter_topic is not None:
        kball = [r for r in kball if r["topic_id"] == args.filter_topic]

    print(f"## Gap correspondence: K-ball vs max-bookless")
    print(f"  topic: {topic_id}")
    print(f"  K-ball: {len(kball)} gaps; sizes "
          f"min={min(k['size'] for k in kball)}, "
          f"max={max(k['size'] for k in kball)}, "
          f"median={statistics.median(k['size'] for k in kball)}")
    print(f"  max-bookless: {len(mb)} gaps; sizes "
          f"min={min(m['size'] for m in mb)}, "
          f"max={max(m['size'] for m in mb)}, "
          f"median={statistics.median(m['size'] for m in mb)}")
    print()

    # Pairwise overlap (sparse — emit only non-zero intersection)
    pairs: list[dict] = []
    for mi, m in enumerate(mb):
        a = m["article_set"]
        for ki, k in enumerate(kball):
            b = k["article_set"]
            inter = len(a & b)
            if inter == 0:
                continue
            union = len(a) + len(b) - inter
            pairs.append({
                "mb_idx": mi, "kball_idx": ki,
                "mb_gap_id": m["gap_id"], "mb_size": m["size"],
                "kball_gap_id": k["gap_id"], "kball_size": k["size"],
                "intersection": inter,
                "jaccard": inter / union,
                "overlap_coef": inter / min(len(a), len(b)),
                "cont_mb_to_kball": inter / len(a),
                "cont_kball_to_mb": inter / len(b),
            })

    print(f"### Pairwise overlap")
    print(f"  non-zero pairs: {len(pairs):,} of {len(mb)*len(kball):,} possible")
    print()

    # Best match per side (by overlap_coef)
    mb_best: dict[int, dict] = {}
    for pp in pairs:
        mi = pp["mb_idx"]
        if mi not in mb_best or pp["overlap_coef"] > mb_best[mi]["overlap_coef"]:
            mb_best[mi] = pp
    kball_best: dict[int, dict] = {}
    for pp in pairs:
        ki = pp["kball_idx"]
        if ki not in kball_best or pp["overlap_coef"] > kball_best[ki]["overlap_coef"]:
            kball_best[ki] = pp

    # Top-N summary tables
    mb_top = sorted(range(len(mb)), key=lambda i: -mb[i]["size"])[:args.top_n]
    kball_top = sorted(range(len(kball)), key=lambda i: -kball[i]["size"])[:args.top_n]

    print(f"### Top-{args.top_n} max-bookless gaps → best K-ball partner")
    print(f"  {'rank':>4}  {'mb_id':>6}  {'mb_sz':>5}     {'kb_id':>5}  {'kb_sz':>5}  "
          f"{'inter':>5}  {'mb→kb':>6}  {'kb→mb':>6}  {'overlap':>7}")
    for rank, mi in enumerate(mb_top, 1):
        if mi in mb_best:
            pp = mb_best[mi]
            print(f"  {rank:>4}  {mb[mi]['gap_id']:>6}  {mb[mi]['size']:>5}  →  "
                  f"{pp['kball_gap_id']:>5}  {pp['kball_size']:>5}  "
                  f"{pp['intersection']:>5}  {pp['cont_mb_to_kball']:>6.3f}  "
                  f"{pp['cont_kball_to_mb']:>6.3f}  {pp['overlap_coef']:>7.3f}")
        else:
            print(f"  {rank:>4}  {mb[mi]['gap_id']:>6}  {mb[mi]['size']:>5}  →   (no overlap)")
    print()

    print(f"### Top-{args.top_n} K-ball gaps → best max-bookless partner")
    print(f"  {'rank':>4}  {'kb_id':>5}  {'kb_sz':>5}     {'mb_id':>6}  {'mb_sz':>5}  "
          f"{'inter':>5}  {'mb→kb':>6}  {'kb→mb':>6}  {'overlap':>7}")
    for rank, ki in enumerate(kball_top, 1):
        if ki in kball_best:
            pp = kball_best[ki]
            print(f"  {rank:>4}  {kball[ki]['gap_id']:>5}  {kball[ki]['size']:>5}  →  "
                  f"{pp['mb_gap_id']:>6}  {pp['mb_size']:>5}  "
                  f"{pp['intersection']:>5}  {pp['cont_mb_to_kball']:>6.3f}  "
                  f"{pp['cont_kball_to_mb']:>6.3f}  {pp['overlap_coef']:>7.3f}")
        else:
            print(f"  {rank:>4}  {kball[ki]['gap_id']:>5}  {kball[ki]['size']:>5}  →   (no overlap)")
    print()

    # Aggregate / phantom stats
    mb_no_match = [mi for mi in range(len(mb)) if mi not in mb_best]
    kball_no_match = [ki for ki in range(len(kball)) if ki not in kball_best]
    print(f"### Aggregate")
    print(f"  max-bookless gaps with NO K-ball overlap: "
          f"{len(mb_no_match)} / {len(mb)} ({len(mb_no_match)/len(mb)*100:.1f}%)")
    print(f"  K-ball gaps with NO max-bookless overlap: "
          f"{len(kball_no_match)} / {len(kball)} ({len(kball_no_match)/len(kball)*100:.1f}%)")
    if mb_best:
        scores = [mb_best[mi]["overlap_coef"] for mi in mb_best]
        q1, med, q3 = _quartiles(scores)
        print(f"  mb→kball best overlap_coef:  median={med:.3f}, "
              f"q1={q1:.3f}, q3={q3:.3f}, mean={statistics.mean(scores):.3f}, "
              f"≥0.5: {sum(1 for s in scores if s >= 0.5)}/{len(scores)}")
    if kball_best:
        scores = [kball_best[ki]["overlap_coef"] for ki in kball_best]
        q1, med, q3 = _quartiles(scores)
        print(f"  kball→mb best overlap_coef:  median={med:.3f}, "
              f"q1={q1:.3f}, q3={q3:.3f}, mean={statistics.mean(scores):.3f}, "
              f"≥0.5: {sum(1 for s in scores if s >= 0.5)}/{len(scores)}")
    print()

    # Coverage of article space
    union_mb = set().union(*(m["article_set"] for m in mb)) if mb else set()
    union_kball = set().union(*(k["article_set"] for k in kball)) if kball else set()
    inter_union = len(union_mb & union_kball)
    union_union = len(union_mb | union_kball)
    print(f"### Article-space coverage (union over each side's gaps)")
    print(f"  union(max-bookless): {len(union_mb):,} articles")
    print(f"  union(K-ball):       {len(union_kball):,} articles")
    print(f"  intersection of unions: {inter_union:,}")
    print(f"  Jaccard of unions:    {inter_union/union_union:.3f}" if union_union else "  Jaccard of unions: n/a")
    print(f"  fraction of mb-union covered by K-ball: "
          f"{inter_union/len(union_mb):.3f}" if union_mb else "  mb-union: empty")
    print(f"  fraction of K-ball-union covered by mb: "
          f"{inter_union/len(union_kball):.3f}" if union_kball else "  K-ball-union: empty")
    print()

    # Decomposition: for each K-ball gap, which mb gaps are mostly inside?
    decomp: dict[int, list[dict]] = {}
    for pp in pairs:
        if pp["cont_mb_to_kball"] >= 0.5:
            decomp.setdefault(pp["kball_idx"], []).append(pp)
    decomp_counts = sorted([len(v) for v in decomp.values()], reverse=True)
    print(f"### Decomposition (mb gaps with ≥50% of self inside a K-ball gap)")
    if decomp_counts:
        n_kb_with_any = len(decomp_counts)
        print(f"  K-ball gaps absorbing ≥1 mb gap: {n_kb_with_any}/{len(kball)}")
        print(f"    1 mb gap absorbed:  {sum(1 for c in decomp_counts if c == 1)}")
        print(f"    2-4 mb gaps:        {sum(1 for c in decomp_counts if 2 <= c <= 4)}")
        print(f"    5-9 mb gaps:        {sum(1 for c in decomp_counts if 5 <= c <= 9)}")
        print(f"    10+ mb gaps:        {sum(1 for c in decomp_counts if c >= 10)}")
        print(f"  max mb gaps inside one K-ball gap: {max(decomp_counts)}")
    else:
        print(f"  (no pairs with cont_mb→kball ≥ 0.5)")
    print()

    # Optimal 1-1 matching (Hungarian, max-weight)
    if mb and kball:
        M = np.zeros((len(mb), len(kball)), dtype=np.float64)
        for pp in pairs:
            M[pp["mb_idx"], pp["kball_idx"]] = pp["overlap_coef"]
        row_ind, col_ind = linear_sum_assignment(-M)
        scores = M[row_ind, col_ind]
        n_pos = int((scores > 0).sum())
        n_strong = int((scores >= 0.5).sum())
        print(f"### Optimal 1-1 matching (Hungarian, maximize overlap_coef)")
        print(f"  matched pairs with overlap > 0:    {n_pos} / {min(len(mb), len(kball))}")
        print(f"  matched pairs with overlap ≥ 0.5:  {n_strong}")
        print(f"  total matched score: {scores.sum():.2f} (mean over matched: "
              f"{scores[scores > 0].mean():.3f})" if n_pos else "  total matched score: 0")
    print()

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = out_dir / f"{topic_id}__pairs.csv"
    cols = ["mb_gap_id", "mb_size", "kball_gap_id", "kball_size",
            "intersection", "jaccard", "overlap_coef",
            "cont_mb_to_kball", "cont_kball_to_mb"]
    with open(pairs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for pp in sorted(pairs, key=lambda r: -r["overlap_coef"]):
            row = {c: pp[c] for c in cols}
            for c in ("jaccard", "overlap_coef", "cont_mb_to_kball", "cont_kball_to_mb"):
                row[c] = f"{row[c]:.6f}"
            w.writerow(row)
    print(f"Wrote {pairs_csv} ({len(pairs)} pairs)")

    def _write_best_side(out_path: Path, items: list[dict], best: dict[int, dict],
                         id_key: str, size_key: str,
                         partner_id_key: str, partner_size_key: str) -> None:
        cols = [id_key, size_key,
                f"best_{partner_id_key}", f"best_{partner_size_key}",
                "best_intersection", "best_jaccard", "best_overlap_coef",
                "best_cont_mb_to_kball", "best_cont_kball_to_mb"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for ix, it in enumerate(items):
                if ix in best:
                    pp = best[ix]
                    w.writerow({
                        id_key: it["gap_id"], size_key: it["size"],
                        f"best_{partner_id_key}": pp[partner_id_key],
                        f"best_{partner_size_key}": pp[partner_size_key],
                        "best_intersection": pp["intersection"],
                        "best_jaccard": f"{pp['jaccard']:.6f}",
                        "best_overlap_coef": f"{pp['overlap_coef']:.6f}",
                        "best_cont_mb_to_kball": f"{pp['cont_mb_to_kball']:.6f}",
                        "best_cont_kball_to_mb": f"{pp['cont_kball_to_mb']:.6f}",
                    })
                else:
                    w.writerow({
                        id_key: it["gap_id"], size_key: it["size"],
                        f"best_{partner_id_key}": "",
                        f"best_{partner_size_key}": "",
                        "best_intersection": 0,
                        "best_jaccard": "0.000000",
                        "best_overlap_coef": "0.000000",
                        "best_cont_mb_to_kball": "0.000000",
                        "best_cont_kball_to_mb": "0.000000",
                    })

    _write_best_side(
        out_dir / f"{topic_id}__mb_best.csv",
        mb, mb_best,
        "mb_gap_id", "mb_size", "kball_gap_id", "kball_size",
    )
    print(f"Wrote {out_dir / f'{topic_id}__mb_best.csv'}")
    _write_best_side(
        out_dir / f"{topic_id}__kball_best.csv",
        kball, kball_best,
        "kball_gap_id", "kball_size", "mb_gap_id", "mb_size",
    )
    print(f"Wrote {out_dir / f'{topic_id}__kball_best.csv'}")


if __name__ == "__main__":
    main()
