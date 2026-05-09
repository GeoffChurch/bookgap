"""Cross-method gap comparison table generator.

REFERENCE IMPLEMENTATION for cross-method comparison artifacts. New
comparison scripts (loading two gap CSVs, computing keywords with the
production log-odds + lattice-bigram pipeline, looking up nearest-book
votes, emitting markdown) should copy from this file. See
``docs/gap-method-development.md`` for the comparison-table pattern.

Compares top-20 gaps from raw K-ball (size-ranked), cohesion-aware
article-as-seed / soft-EM (logit-ranked, with min_post >= 0.5 boundary),
and HDBSCAN max-bookless (density-based, size-ranked) side by side in
markdown form. For each gap:

- 5 distinguishing keywords via log-odds-ratio (Fightin' Words style,
  marginal-background + freq-weighting), with bigram phrase mining as
  in the curriculum production pipeline.
- Top-3 nearest books (by article-vote frequency).
- Gap size and key cohesion metrics (article-as-seed only).
- 2 example article titles.
- Unanchored flag (top-CSLS book < 0.3) derived from the Phase 3
  ``top_books_csls_json`` column; replaces the old categorical
  coverage_label_v2 / templated_leak_flag scheme.

The CSVs should be the Phase 3 enriched outputs of ``gap_soft_em``
(which emits ``top_books_csls_json``, ``effective_size``, and
``n_articles_kNN``). Un-enriched CSVs still load; missing columns
default to "" and every gap is treated as unanchored.

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


def _is_unanchored(row: dict, weight_threshold: float = 0.3) -> bool:
    """Phase 3.5 replacement.

    A basin is 'unanchored' (gap candidate) if no book strongly belongs
    to it: the top book in top_books_membership_json[0] has weight
    < weight_threshold. Default 0.3 — softmax weights are in [0, 1] and
    a strongly-anchored basin's top book typically weights ≥0.5.
    """
    raw = row.get("top_books_membership_json", "")
    if not raw:
        return True
    try:
        top = json.loads(raw)
    except (ValueError, TypeError):
        return True
    if not top:
        return True
    return float(top[0].get("weight", 0.0)) < weight_threshold


def _load_kball(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_idx"]),
                "size": int(r["gap_size"]),
                "members": json.loads(r["article_indices_json"]),
                "book_exists_flag": int(r.get("book_exists_flag") or 0),
                "book_anchored": int(r.get("book_anchored") or 0),
                "top_book_title": r.get("top_book_title", ""),
                "vote_concentration": float(r.get("vote_concentration") or 0),
                "top_books_csls_json": r.get("top_books_csls_json", ""),
                "effective_size": r.get("effective_size", ""),
            })
    return rows


def _load_hdbscan(path: Path) -> list[dict]:
    """Load an HDBSCAN max-bookless CSV that was reformatted to the
    article-seed schema and run through the standard enrichment pipeline.

    HDBSCAN exposes no posterior — `mean_post`/`min_post` are 1.0 and
    `logit_sum` is 0 by construction, so basins are size-ranked.
    """
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "rank": int(r["rank"]),
                "size": int(r.get("final_size") or r.get("total_size") or r.get("n_articles") or 0),
                "members": json.loads(r["article_indices_json"]),
                "n_books": int(r.get("n_books") or 0),
                "book_exists_flag": int(r.get("book_exists_flag") or 0),
                "book_anchored": int(r.get("book_anchored") or 0),
                "top_book_title": r.get("top_book_title", ""),
                "vote_concentration": float(r.get("vote_concentration") or 0),
                "top_books_csls_json": r.get("top_books_csls_json", ""),
                "effective_size": r.get("effective_size", ""),
                "top_keywords": r.get("top_keywords", ""),
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
                "book_anchored": int(r.get("book_anchored") or 0),
                "top_book_title": r.get("top_book_title", ""),
                "vote_concentration": float(r.get("vote_concentration") or 0),
                "top_books_csls_json": r.get("top_books_csls_json", ""),
                "effective_size": r.get("effective_size", ""),
                "top_keywords": r.get("top_keywords", ""),
                "inherited_coverage_median_post": (
                    float(r["inherited_coverage_median_post"])
                    if r.get("inherited_coverage_median_post") else None
                ),
                "inherited_coverage_max_basin_rank": (
                    int(r["inherited_coverage_max_basin_rank"])
                    if r.get("inherited_coverage_max_basin_rank") else None
                ),
                "effective_uncovered_articles": (
                    float(r["effective_uncovered_articles"])
                    if r.get("effective_uncovered_articles") else None
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
    p.add_argument("--hdbscan-csv", default=None,
                   help="optional: enriched HDBSCAN max-bookless CSV "
                        "(article-seed schema, post gap_book_existence_filter "
                        "and gap_multi_scale_coverage). When provided, a third "
                        "comparison column is rendered.")
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

    hdbscan_all: list[dict] = []
    hdbscan_top: list[dict] = []
    if args.hdbscan_csv:
        print(f"Loading HDBSCAN max-bookless gaps (size-ranked)...")
        hdbscan_all = _load_hdbscan(Path(args.hdbscan_csv))
        hdbscan_top = sorted(hdbscan_all, key=lambda r: -r["size"])[:args.top_n]

    print("Computing gap keywords...")
    all_gaps = (
        [g["members"] for g in kball_top]
        + [g["members"] for g in seed_top]
        + [g["members"] for g in hdbscan_top]
    )
    keywords = compute_gap_keywords(
        all_gaps, titles, abstracts,
        top_k=args.keywords_per_gap,
        bigram_iterations=args.bigram_iterations,
    )
    kball_kws = keywords[:len(kball_top)]
    seed_kws = keywords[len(kball_top):len(kball_top) + len(seed_top)]
    hdbscan_kws = keywords[len(kball_top) + len(seed_top):]

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
    hdbscan_books = [compute_top_books(g["members"], emb, book_index, book_global_idx, all_titles_arr)
                     for g in hdbscan_top]

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
    pipelines_lead = (
        "Top-{n} gaps from each pipeline, side by side. Three methods are "
        "compared:\n\n"
        "- **K-ball** — the original size-ranked baseline (radius-based "
        "nearest-neighbor search around each book; basins are pure-article "
        "neighborhoods that exclude books).\n"
        "- **Soft-EM (article-as-seed)** — the cohesion-aware production "
        "pipeline (moving-centroid HNSW, soft-membership posteriors, strict "
        "policy, min_post≥0.5 boundary, logit_sum ranking).\n"
        "- **HDBSCAN max-bookless** — density-based; runs HDBSCAN's "
        "single-linkage tree over articles+books and extracts the *maximal "
        "bookless* cluster at each subtree (the largest pure-article cluster "
        "before any book joins). Size-ranked.\n\n"
        "Keywords use log-odds-ratio (Fightin' Words) with marginal "
        "background and freq-weighting, with bigram phrase mining, matching "
        "the curriculum production pipeline. Top books are the most-voted "
        "nearest book per gap article (1-NN)."
    ).format(n=args.top_n)
    out.append(pipelines_lead)
    out.append("")

    # ---- Basin summary (headline) ----------------------------------------
    out.append("## Basin summary")
    out.append("")
    out.append(
        "Counts and size ranges for all basins in each pipeline. "
        "Phase 3 drops the templated-leak tag; users can identify lexical "
        "artifacts directly from `top_keywords`. Unanchored basins are those "
        "where the top-CSLS book scores < 0.3 (see `_is_unanchored`)."
    )
    out.append("")

    def _summarize(rows: list[dict]) -> dict:
        sizes = [r["size"] for r in rows]
        unanchored = [r for r in rows if _is_unanchored(r)]
        return {
            "total": len(rows),
            "unanchored": len(unanchored),
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
        }

    kb_sum = _summarize(kball_all)
    se_sum = _summarize(seed_all)
    hb_sum = _summarize(hdbscan_all) if hdbscan_all else None

    out.append("| pipeline | total basins | unanchored (CSLS<0.3) | basin size range |")
    out.append("|----------|--------------|-----------------------|------------------|")
    out.append(f"| K-ball | {kb_sum['total']} | {kb_sum['unanchored']} | "
               f"{kb_sum['min_size']}–{kb_sum['max_size']} |")
    out.append(f"| Soft-EM (article-seed) | {se_sum['total']} | {se_sum['unanchored']} | "
               f"{se_sum['min_size']}–{se_sum['max_size']} |")
    if hb_sum:
        out.append(f"| HDBSCAN max-bookless | {hb_sum['total']} | {hb_sum['unanchored']} | "
                   f"{hb_sum['min_size']}–{hb_sum['max_size']} |")
    out.append("")

    out.append("## K-ball (size-ranked)")
    out.append("")
    out.append("| rank | size | unanchored | keywords | top books (×votes) | example titles |")
    out.append("|------|------|------------|----------|-------------------|----------------|")
    for i, g in enumerate(kball_top):
        unanchored = _is_unanchored(g)
        out.append(
            f"| {i+1} | {g['size']} | {'yes' if unanchored else 'no'} | "
            f"{_kw_str(kball_kws[i], args.keywords_per_gap)} | "
            f"{_top_books_str(kball_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
        )
    out.append("")

    out.append("### K-ball top-N — unanchored only")
    out.append("")
    kball_real = [g for g in kball_top if _is_unanchored(g)][:args.top_n]
    if not kball_real:
        out.append("_(no unanchored rows in this pipeline's top-N)_")
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
    out.append("| rank | size | mean_p / min_p | logit | unanchored | inh_med | keywords | top books (×votes) | example titles |")
    out.append("|------|------|----------------|-------|------------|---------|----------|-------------------|----------------|")
    for i, g in enumerate(seed_top):
        unanchored = _is_unanchored(g)
        out.append(
            f"| {i+1} | {g['size']} | {g['mean_post']:.2f} / {g['min_post']:.2f} | "
            f"{g['logit_sum']:.0f} | {'yes' if unanchored else 'no'} | {_med_post_str(g)} | "
            f"{_kw_str(seed_kws[i], args.keywords_per_gap)} | "
            f"{_top_books_str(seed_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
        )
    out.append("")

    out.append("### Article-as-seed top-N — unanchored only")
    out.append("")
    out.append("_Filtered to basins where no book strongly owns the space "
               "(top-CSLS book < 0.3). Phase 3 drops the inherited-coverage "
               "tier; use `top_books_csls_json` and `effective_size` directly "
               "to gauge coverage strength._")
    out.append("")
    seed_real = [g for g in seed_top if _is_unanchored(g)][:args.top_n]
    if not seed_real:
        out.append("_(no unanchored rows in this pipeline's top-N)_")
    else:
        out.append("| rank | size | mean_p / min_p | logit | unanchored | inh_med | keywords | top books (×votes) | example titles |")
        out.append("|------|------|----------------|-------|------------|---------|----------|-------------------|----------------|")
        for g in seed_real:
            i = seed_top.index(g)
            out.append(
                f"| {i+1} | {g['size']} | {g['mean_post']:.2f} / {g['min_post']:.2f} | "
                f"{g['logit_sum']:.0f} | yes | {_med_post_str(g)} | "
                f"{_kw_str(seed_kws[i], args.keywords_per_gap)} | "
                f"{_top_books_str(seed_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
            )
    out.append("")

    # Article-seed pool sorted by effective_uncovered_articles descending —
    # the "most-uncovered-research" leaderboard. Σ_a w(a)·(1-best_post) is
    # the membership-marginalized expectation of "uncovered article-equivalents"
    # in this basin. Volume-aware: bigger basins with low containment rank
    # higher than small basins with low containment.
    seed_with_inh = [g for g in seed_all if g.get("effective_uncovered_articles") is not None]
    if seed_with_inh:
        seed_least_covered = sorted(
            seed_with_inh,
            key=lambda r: -r["effective_uncovered_articles"],
        )[:args.top_n]
        out.append("### Article-as-seed top-20 — most uncovered research (sorted by effective_uncovered_articles descending)")
        out.append("")
        out.append(f"_The basins with the largest expected uncovered-article-equivalents — "
                   f"`Σ_a w(a)·(1 − best_post(a, H*))` — strongest candidates for genuine "
                   f"research gaps. Volume-aware: a 500-article basin at moderate containment "
                   f"outranks a 30-article basin at low containment. Only non-book-anchored "
                   f"basins shown (n_books=0)._")
        out.append("")
        out.append("| eff_unc | inh_med | size | logit | unanchored | keywords (truncated) | top vote book |")
        out.append("|---------|---------|------|-------|------------|----------------------|---------------|")
        for g in seed_least_covered:
            kw_short = (g.get("top_keywords") or "")[:60] if isinstance(g.get("top_keywords"), str) else ""
            top_book_short = g.get("top_book_title", "")[:50]
            inh_med = g.get("inherited_coverage_median_post")
            inh_med_s = f"{inh_med:.2f}" if inh_med is not None else "—"
            unanchored = _is_unanchored(g)
            out.append(
                f"| {g['effective_uncovered_articles']:.1f} | {inh_med_s} | {g['size']} | "
                f"{g['logit_sum']:.0f} | {'yes' if unanchored else 'no'} | {kw_short} | {top_book_short} |"
            )
        out.append("")

    # ---- HDBSCAN max-bookless (size-ranked) section --------------------
    if hdbscan_top:
        out.append("## HDBSCAN max-bookless (size-ranked)")
        out.append("")
        out.append("_Density-based extraction: HDBSCAN's single-linkage tree over "
                   "articles+books, then for each book-bearing subtree the maximal "
                   "pure-article cluster *before any book joins* is emitted. By "
                   "construction `n_books = 0`; basins are size-ranked since no "
                   "posterior is computed._")
        out.append("")
        out.append("| rank | size | unanchored | keywords | top books (×votes) | example titles |")
        out.append("|------|------|------------|----------|-------------------|----------------|")
        for i, g in enumerate(hdbscan_top):
            unanchored = _is_unanchored(g)
            out.append(
                f"| {i+1} | {g['size']} | {'yes' if unanchored else 'no'} | "
                f"{_kw_str(hdbscan_kws[i], args.keywords_per_gap)} | "
                f"{_top_books_str(hdbscan_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
            )
        out.append("")

        out.append("### HDBSCAN top-N — unanchored only")
        out.append("")
        hdbscan_real = [g for g in hdbscan_top if _is_unanchored(g)][:args.top_n]
        if not hdbscan_real:
            out.append("_(no unanchored rows in this pipeline's top-N)_")
        else:
            out.append("| rank | size | keywords | top books (×votes) | example titles |")
            out.append("|------|------|----------|-------------------|----------------|")
            for g in hdbscan_real:
                i = hdbscan_top.index(g)
                out.append(
                    f"| {i+1} | {g['size']} | {_kw_str(hdbscan_kws[i], args.keywords_per_gap)} | "
                    f"{_top_books_str(hdbscan_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
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
    out.append(f"| **book_exists_flag=0** | {matrix[(0, 0)]} (unanchored) | {matrix[(0, 1)]} (book_anchored only) |")
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
