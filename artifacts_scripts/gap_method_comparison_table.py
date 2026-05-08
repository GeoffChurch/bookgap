"""Cross-method gap comparison table generator.

REFERENCE IMPLEMENTATION for cross-method comparison artifacts. New
comparison scripts (loading two gap CSVs, computing keywords with the
production log-odds + lattice-bigram pipeline, looking up nearest-book
votes, emitting markdown) should copy from this file. See
``docs/gap-method-development.md`` for the comparison-table pattern.

Compares top-20 gaps from raw K-ball (size-ranked) vs cohesion-aware
article-as-seed (logit-ranked, with min_post >= 0.5 boundary), side by
side in markdown form. For each gap:

- 5 distinguishing keywords via log-odds-ratio (Fightin' Words style,
  marginal-background + freq-weighting), with bigram phrase mining as
  in the curriculum production pipeline.
- Top-3 nearest books (by article-vote frequency).
- Gap size and key cohesion metrics (article-as-seed only).
- 2 example article titles.
- Coverage label (real-gap / book-exists / book-anchored /
  book-exists+anchored / templated-leak) drawn from Phase-1 and Phase-3
  enrichment columns.

The CSVs should be the enriched outputs of ``gap_book_existence_filter``
and ``gap_article_as_seed --with-book-seeds``. Run those upstream first.
Un-enriched CSVs still load; missing Phase-1/3 columns default to 0/""
and every gap is labelled ``real-gap``.

Usage::

    python -m artifacts_scripts.gap_method_comparison_table \\
        --scope 17/1702/T10181 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0_enriched.csv \\
        --article-seed-csv /tmp/task10_t10181/T10181__article_seed_top100_enriched.csv \\
        --top-n 20
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from artifacts_scripts._keyword_helpers import compute_gap_keywords, compute_top_books
from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import build_index, FlatL2


def _coverage_label(row: dict) -> str:
    """Coverage label with tiered multi-scale inheritance.

    If the input CSV has a ``coverage_label_v2`` column (produced by
    ``gap_multi_scale_coverage``), use that directly. Otherwise fall back
    to the Phase-3-only classifier (templated-leak > book-anchored >
    book-exists > real-gap).
    """
    if row.get("coverage_label_v2"):
        return str(row["coverage_label_v2"])
    if row.get("templated_leak_flag"):
        return "templated-leak"
    if row.get("book_anchored"):
        if row.get("book_exists_flag"):
            return "book-exists+anchored"
        return "book-anchored"
    if row.get("book_exists_flag"):
        return "book-exists"
    return "real-gap"


# Labels considered "true research gaps" — basins not covered by any
# book-anchored basin at meaningful strength. inherited-weak survives
# because median_post < 0.65 means peripheral overlap, not real coverage.
_REAL_GAP_LABELS = frozenset({"real-gap", "inherited-weak"})


def _is_real(g: dict) -> bool:
    return _coverage_label(g) in _REAL_GAP_LABELS


def _load_kball(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_idx"]),
                "size": int(r["gap_size"]),
                "members": json.loads(r["article_indices_json"]),
                "book_exists_flag": int(r.get("book_exists_flag") or 0),
                "templated_leak_flag": int(r.get("templated_leak_flag") or 0),
                "book_anchored": int(r.get("book_anchored") or 0),
                "top_book_title": r.get("top_book_title", ""),
                "vote_concentration": float(r.get("vote_concentration") or 0),
            })
    return rows


def _load_article_seed(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "rank": int(r["rank"]),
                "seed": int(r.get("seed_article") or r.get("first_seed") or 0),
                "size": int(r.get("final_size") or r.get("total_size") or 0),
                "logit_sum": float(r["logit_sum"]),
                "mean_post": float(r["mean_post"]),
                "min_post": float(r["min_post"]),
                "members": json.loads(r["article_indices_json"]),
                "n_books": int(r.get("n_books") or 0),
                "book_exists_flag": int(r.get("book_exists_flag") or 0),
                "templated_leak_flag": int(r.get("templated_leak_flag") or 0),
                "book_anchored": int(r.get("book_anchored") or 0),
                "top_book_title": r.get("top_book_title", ""),
                "vote_concentration": float(r.get("vote_concentration") or 0),
                "coverage_label_v2": r.get("coverage_label_v2", ""),
                "top_keywords": r.get("top_keywords", ""),
                "inherited_coverage_median_post": (
                    float(r["inherited_coverage_median_post"])
                    if r.get("inherited_coverage_median_post") else None
                ),
                "inherited_coverage_max_basin_rank": (
                    int(r["inherited_coverage_max_basin_rank"])
                    if r.get("inherited_coverage_max_basin_rank") else None
                ),
            })
    return rows


def _load_articles_parquet(scope: str) -> tuple[list[str], list[str]]:
    """Return (titles, abstracts) for articles in the scope, indexed by
    the same global position as gap_article_as_seed/K-ball CSVs use."""
    parts = scope.split("/")
    parquet_dir = Path("cache/works_parquet").joinpath(*parts)
    article_pq = parquet_dir / "article.parquet"
    if not article_pq.exists():
        raise FileNotFoundError(f"No article.parquet at {article_pq}")
    tbl = pq.read_table(article_pq, columns=["display_name", "abstract"])
    titles = tbl.column("display_name").to_pylist()
    abstracts = tbl.column("abstract").to_pylist()
    return titles, abstracts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--kball-csv", required=True)
    p.add_argument("--article-seed-csv", required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--keywords-per-gap", type=int, default=5)
    p.add_argument("--titles-per-gap", type=int, default=2)
    p.add_argument("--bigram-iterations", type=int, default=2,
                   help="phrase-mining iterations; production default is 2 "
                        "(trigrams via bigram-of-bigram); higher surfaces longer phrases")
    p.add_argument("--output-md", default="docs/gap_method_comparison.md")
    args = p.parse_args()

    print(f"Loading article titles+abstracts for scope {args.scope}...")
    titles, abstracts = _load_articles_parquet(args.scope)
    print(f"  {len(titles):,} articles loaded")

    # Also load full scope for book lookups (need book embeddings + titles).
    print(f"Loading scope embeddings for nearest-book lookup...")
    emb, ids, types, all_titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    types_arr = np.array(types)
    is_book = types_arr == "book"
    book_global_idx = np.where(is_book)[0]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)
    all_titles_arr = np.array(all_titles, dtype=object)

    print(f"Loading K-ball gaps (sorted by size descending)...")
    kball_all = _load_kball(Path(args.kball_csv))
    kball_top = sorted(kball_all, key=lambda r: -r["size"])[:args.top_n]

    print(f"Loading article-as-seed gaps (already sorted by logit_sum)...")
    seed_all = _load_article_seed(Path(args.article_seed_csv))
    seed_top = sorted(seed_all, key=lambda r: -r["logit_sum"])[:args.top_n]

    print("Computing gap keywords...")
    all_gaps = [g["members"] for g in kball_top] + [g["members"] for g in seed_top]
    keywords = compute_gap_keywords(
        all_gaps, titles, abstracts,
        top_k=args.keywords_per_gap,
        bigram_iterations=args.bigram_iterations,
    )
    kball_kws = keywords[:len(kball_top)]
    seed_kws = keywords[len(kball_top):]

    print("Computing nearest books per gap...")
    def _top_books_str(book_list: list[tuple[int, str, int]]) -> str:
        return "<br>".join(
            f"  • [×{c}] {title.replace('|', chr(92) + '|')[:80]}"
            for _bi, title, c in book_list
        )

    kball_books = [compute_top_books(g["members"], emb, book_index, book_global_idx, all_titles_arr)
                   for g in kball_top]
    seed_books = [compute_top_books(g["members"], emb, book_index, book_global_idx, all_titles_arr)
                  for g in seed_top]

    def _kw_str(kw_list, k: int) -> str:
        return ", ".join(w for w, _ in kw_list[:k])

    def _title_lines(g, k: int) -> str:
        ts = []
        for m in g["members"][:k]:
            t = titles[m] if m < len(titles) else "(no title)"
            t = (t or "").replace("|", "\\|").replace("\n", " ").strip()
            ts.append(f"  • {t[:90]}")
        return "<br>".join(ts)

    out = []
    out.append(f"# Gap-detection method comparison: {args.scope}")
    out.append("")
    out.append("Top-20 gaps from each pipeline, side by side. K-ball is")
    out.append("the original size-ranked baseline; article-as-seed uses")
    out.append("the cohesion-aware HNSW pipeline (moving-centroid, strict")
    out.append("policy, min_post≥0.5 boundary, logit_sum ranking).")
    out.append("Keywords use log-odds-ratio (Fightin' Words) with marginal")
    out.append("background and freq-weighting, with bigram phrase mining,")
    out.append("matching the curriculum production pipeline. Top books are")
    out.append("the most-voted nearest book per gap article (1-NN).")
    out.append("")

    out.append("## K-ball (size-ranked)")
    out.append("")
    out.append("| rank | size | coverage | keywords | top books (×votes) | example titles |")
    out.append("|------|------|----------|----------|-------------------|----------------|")
    for i, g in enumerate(kball_top):
        cov = _coverage_label(g)
        out.append(
            f"| {i+1} | {g['size']} | {cov} | {_kw_str(kball_kws[i], args.keywords_per_gap)} | "
            f"{_top_books_str(kball_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
        )
    out.append("")

    out.append("### K-ball top-20 — real-gap only")
    out.append("")
    kball_real = [g for g in kball_top if _is_real(g)][:args.top_n]
    if not kball_real:
        out.append("_(no real-gap rows in this pipeline's top-20)_")
    else:
        out.append("| rank | size | keywords | top books (×votes) | example titles |")
        out.append("|------|------|----------|-------------------|----------------|")
        for g in kball_real:
            i = kball_top.index(g)
            out.append(
                f"| {i+1} | {g['size']} | {_kw_str(kball_kws[i], args.keywords_per_gap)} | "
                f"{_top_books_str(kball_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
            )
    out.append("")

    def _med_post_str(g: dict) -> str:
        v = g.get("inherited_coverage_median_post")
        return f"{v:.2f}" if v is not None else "—"

    out.append("## Article-as-seed (logit_sum-ranked, cohesion-aware)")
    out.append("")
    out.append("| rank | size | mean_p / min_p | logit | coverage | inh_med | keywords | top books (×votes) | example titles |")
    out.append("|------|------|----------------|-------|----------|---------|----------|-------------------|----------------|")
    for i, g in enumerate(seed_top):
        cov = _coverage_label(g)
        out.append(
            f"| {i+1} | {g['size']} | {g['mean_post']:.2f} / {g['min_post']:.2f} | "
            f"{g['logit_sum']:.0f} | {cov} | {_med_post_str(g)} | "
            f"{_kw_str(seed_kws[i], args.keywords_per_gap)} | "
            f"{_top_books_str(seed_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
        )
    out.append("")

    out.append("### Article-as-seed top-20 — real-gap (or inherited-weak) only")
    out.append("")
    out.append("_Filtered to basins with no direct or strong inherited book "
               "coverage. `inherited-weak` (peripheral overlap) included alongside "
               "`real-gap` because median posterior < 0.65 means the 'covering' "
               "basin only barely qualifies as a container._")
    out.append("")
    seed_real = [g for g in seed_top if _is_real(g)][:args.top_n]
    if not seed_real:
        out.append("_(no real-gap or inherited-weak rows in this pipeline's top-20)_")
    else:
        out.append("| rank | size | mean_p / min_p | logit | coverage | inh_med | keywords | top books (×votes) | example titles |")
        out.append("|------|------|----------------|-------|----------|---------|----------|-------------------|----------------|")
        for g in seed_real:
            i = seed_top.index(g)
            cov = _coverage_label(g)
            out.append(
                f"| {i+1} | {g['size']} | {g['mean_post']:.2f} / {g['min_post']:.2f} | "
                f"{g['logit_sum']:.0f} | {cov} | {_med_post_str(g)} | "
                f"{_kw_str(seed_kws[i], args.keywords_per_gap)} | "
                f"{_top_books_str(seed_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
            )
    out.append("")

    # Article-seed pool sorted by inherited_coverage_median_post ascending —
    # the "least-contained" leaderboard. These are the basins furthest from
    # any book-anchored basin's structural neighborhood, the strongest
    # candidates for genuine research gaps.
    seed_with_inh = [g for g in seed_all if g.get("inherited_coverage_median_post") is not None]
    if seed_with_inh:
        seed_least_covered = sorted(
            seed_with_inh,
            key=lambda r: r["inherited_coverage_median_post"],
        )[:args.top_n]
        out.append("### Article-as-seed top-20 — least-contained (sorted by inh_med ascending)")
        out.append("")
        out.append(f"_The basins furthest from any book-anchored basin's structural "
                   f"neighborhood — strongest candidates for genuine research gaps. "
                   f"Sorted by ``inherited_coverage_median_post`` ascending; only "
                   f"non-book-anchored basins shown (n_books=0)._")
        out.append("")
        out.append("| inh_med | size | logit | coverage | keywords (truncated) | top vote book |")
        out.append("|---------|------|-------|----------|----------------------|---------------|")
        for g in seed_least_covered:
            kw_short = (g.get("top_keywords") or "")[:60] if isinstance(g.get("top_keywords"), str) else ""
            top_book_short = g.get("top_book_title", "")[:50]
            out.append(
                f"| {g['inherited_coverage_median_post']:.2f} | {g['size']} | "
                f"{g['logit_sum']:.0f} | {_coverage_label(g)} | {kw_short} | {top_book_short} |"
            )
        out.append("")

    # The agreement matrix uses the full article-seed pool — not just top-N —
    # because rare disagreements are most informative across the whole population.
    out.append("## Phase-1 ↔ Phase-3 agreement matrix (article-seed pipeline)")
    out.append("")
    out.append("Agreement between the heuristic (`book_exists_flag` from "
               "vote-concentration + title overlap) and the structural "
               "(`book_anchored = (n_books > 0)`) signals on the full article-seed pool.")
    out.append("")
    matrix = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for g in seed_all:
        matrix[(g["book_exists_flag"], g["book_anchored"])] += 1
    out.append("| | book_anchored=0 | book_anchored=1 |")
    out.append("|---|---|---|")
    out.append(f"| **book_exists_flag=0** | {matrix[(0, 0)]} (real-gap) | {matrix[(0, 1)]} (Phase 3 catches) |")
    out.append(f"| **book_exists_flag=1** | {matrix[(1, 0)]} (heuristic-only) | {matrix[(1, 1)]} (both fire) |")
    out.append("")
    total = sum(matrix.values()) or 1
    agreement = (matrix[(0, 0)] + matrix[(1, 1)]) / total
    out.append(f"Agreement: {agreement*100:.1f}% ({matrix[(0, 0)] + matrix[(1, 1)]}/{total}).")
    out.append("")

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
