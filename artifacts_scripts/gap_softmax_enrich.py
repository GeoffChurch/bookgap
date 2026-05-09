"""Multi-cluster softmax Phase 2 enrichment.

Replaces gap_book_existence_filter + gap_multi_scale_coverage with a
single one-shot multi-cluster softmax refinement layer over soft-EM's
basin set. Output is a drop-in enriched CSV with templated-leak
tagging, direct softmax effective_uncovered, and continuous
inherited_coverage_median_post (no coverage_label_v2 column —
downstream computes its own buckets from the raw flags + metric).

Spec: docs/superpowers/specs/2026-05-09-multicluster-softmax-phase2-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Soft-EM and Phase 2 CSVs both have member_weights_json fields > 131072 bytes
csv.field_size_limit(sys.maxsize)

from artifacts_scripts.gap_softmax_reassign import (
    compute_self_tuning_sigma,
    load_enriched_csv,
    softmax_e_step,
)


def load_soft_em_basins(
    csv_path: Path,
    is_article: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Read a soft-EM raw CSV → (M0, basin_meta).

    Thin alias around gap_softmax_reassign.load_enriched_csv; both raw
    soft-EM and Phase-1-enriched CSVs share the relevant columns
    (rank, seed_article, n_books, book_anchored,
    member_indices_json, member_weights_json), so the same loader works
    on either input.
    """
    return load_enriched_csv(csv_path, is_article)


def run_softmax_step(
    article_emb: np.ndarray,
    M0: np.ndarray,
    k_knn: int = 10,
    k_sigma: int = 7,
    perplexity: float = 2.0,
) -> np.ndarray:
    """One-shot softmax E-step over M0's basin set.

    Returns refined membership matrix M_new of shape
    (N_articles, N_basins). Only the k_knn basin columns per article
    are nonzero; the rest are 0 (kNN truncation). Rows sum to 1
    (Variant A — no ∅ class).
    """
    article_sigma = compute_self_tuning_sigma(article_emb, k_sigma=k_sigma)
    M_new, _ = softmax_e_step(
        article_emb=article_emb,
        article_sigma=article_sigma,
        M=M0,
        log_K_empty=None,
        k_knn=k_knn,
        P_target=perplexity,
    )
    return M_new


def crisp_filter(
    M_new: np.ndarray,
    tau_supp: float = 0.5,
    min_final_size: int = 5,
) -> dict[int, list[tuple[int, float]]]:
    """For each basin, return its supported members above tau_supp.

    Returns dict mapping basin_idx → list of (article_local_idx, weight)
    pairs, sorted by weight descending. Basins with fewer than
    min_final_size supported articles are dropped from the dict.
    """
    out: dict[int, list[tuple[int, float]]] = {}
    n_articles, n_basins = M_new.shape
    for b in range(n_basins):
        col = M_new[:, b]
        members = np.where(col > tau_supp)[0]
        if len(members) < min_final_size:
            continue
        weights = col[members]
        order = np.argsort(-weights)
        out[b] = [(int(members[i]), float(weights[i])) for i in order]
    return out


def inherited_coverage_per_article(
    M_new: np.ndarray,
    book_anchored_mask: np.ndarray,
) -> np.ndarray:
    """For each article a: max over book-anchored basins H of M_new[a, H].
    Returns shape (N_articles,). Zero everywhere if no basins anchored.
    """
    if not book_anchored_mask.any():
        return np.zeros(M_new.shape[0], dtype=np.float64)
    book_columns = M_new[:, book_anchored_mask]
    return book_columns.max(axis=1).astype(np.float64)


def per_basin_readouts(
    M_new: np.ndarray,
    surviving: dict[int, list[tuple[int, float]]],
    cov_per_article: np.ndarray,
    book_anchored_mask: np.ndarray,
) -> dict[int, dict]:
    """Compute per-basin direct softmax readouts.

    For each surviving non-book-anchored basin G:
      - inherited_coverage_median_post = median over G's tau_supp members
        a of cov_per_article[a]
      - effective_uncovered_articles = Σ_a M_new[a, G] · (1 − cov_per_article[a])
        summed over all articles (not just members)

    For book-anchored basins, both readouts are emitted as 0.0
    (they are themselves the source of coverage).
    """
    uncovered = 1.0 - cov_per_article
    out: dict[int, dict] = {}
    for b, members in surviving.items():
        if book_anchored_mask[b]:
            out[b] = {
                "inherited_coverage_median_post": 0.0,
                "effective_uncovered_articles": 0.0,
            }
            continue
        member_idx = np.array([i for i, _ in members], dtype=np.int64)
        median_post = float(np.median(cov_per_article[member_idx]))
        eff_uncovered = float((M_new[:, b] * uncovered).sum())
        out[b] = {
            "inherited_coverage_median_post": median_post,
            "effective_uncovered_articles": eff_uncovered,
        }
    return out


from artifacts_scripts._keyword_helpers import (
    compute_book_existence_signal,
    compute_gap_keywords,
    compute_top_books,
)
from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import FlatL2, build_index


def tag_basin_book_signals(
    surviving: dict[int, list[tuple[int, float]]],
    emb: np.ndarray,
    types_arr: np.ndarray,
    titles: list[str],
    abstracts: list[str],
    article_global_idx: np.ndarray,
    keywords_per_gap: int = 5,
    bigram_iterations: int = 2,
    vote_concentration_threshold: float = 0.25,
    title_overlap_threshold: float = 0.2,
    templated_concentration_threshold: float = 0.5,
) -> dict[int, dict]:
    """For each surviving basin, compute book-existence + templated-leak
    signals using the existing _keyword_helpers logic.

    Returns dict basin_idx → {top_book_id, top_book_title, top_book_votes,
    vote_concentration, top_keywords, title_keyword_overlap,
    book_exists_flag, templated_leak_flag}.
    """
    book_global_idx = np.where(types_arr == "book")[0]
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)
    all_titles_arr = np.array(titles, dtype=object)

    # Build per-basin global member-index lists (articles only)
    basin_ids = sorted(surviving.keys())
    gaps_global = [
        [int(article_global_idx[i]) for i, _ in surviving[b]]
        for b in basin_ids
    ]

    keywords_per_basin = compute_gap_keywords(
        gaps=gaps_global,
        titles=titles,
        abstracts=abstracts,
        top_k=keywords_per_gap,
        bigram_iterations=bigram_iterations,
        verbose=False,
    )

    out: dict[int, dict] = {}
    for b, members_global, kw_pairs in zip(basin_ids, gaps_global, keywords_per_basin):
        top_books = compute_top_books(
            members=members_global,
            emb=emb,
            book_index=book_index,
            book_global_idx=book_global_idx,
            all_titles=all_titles_arr,
            k=3,
        )
        if top_books:
            top_book_id, top_book_title, top_book_votes = top_books[0]
        else:
            top_book_id, top_book_title, top_book_votes = -1, "", 0
        top_keywords = [kw for kw, _ in kw_pairs]
        signal = compute_book_existence_signal(
            gap_size=len(members_global),
            top_book_votes=top_book_votes,
            top_book_title=top_book_title,
            top_keywords=top_keywords,
            vote_concentration_threshold=vote_concentration_threshold,
            title_overlap_threshold=title_overlap_threshold,
            templated_concentration_threshold=templated_concentration_threshold,
        )
        out[b] = {
            "top_book_id": top_book_id,
            "top_book_title": top_book_title,
            "top_book_votes": top_book_votes,
            "vote_concentration": signal["vote_concentration"],
            "top_keywords": " ".join(top_keywords),
            "title_keyword_overlap": signal["title_keyword_overlap"],
            "book_exists_flag": int(signal["book_exists_flag"]),
            "templated_leak_flag": int(signal["templated_leak_flag"]),
        }
    return out


def compute_book_csls_matrix(
    basin_centroids: np.ndarray,
    book_emb: np.ndarray,
    k_csls: int = 10,
) -> np.ndarray:
    """Per-(basin, book) CSLS reciprocity:
        csls(G, B) = 2·sim(G, B) − hub_basin(G) − hub_book(B)
    where hub_X is the mean cos-sim of X to its top-k_csls nearest
    counterparts. Distance-aware via raw cos-sim; bilateral against
    both basin- and book-side hubs.
    """
    sims = (basin_centroids @ book_emb.T).astype(np.float32)
    n_basins, n_books = sims.shape
    k_b = min(k_csls, n_books)
    k_g = min(k_csls, n_basins)
    top_b = np.partition(sims, -k_b, axis=1)[:, -k_b:]
    hub_basin = top_b.mean(axis=1)
    top_g = np.partition(sims, -k_g, axis=0)[-k_g:, :]
    hub_book = top_g.mean(axis=0)
    csls = 2.0 * sims - hub_basin[:, None] - hub_book[None, :]
    return csls.astype(np.float32)


def top_books_csls_per_basin(
    csls_matrix: np.ndarray,
    book_global_idx: np.ndarray,
    all_titles: np.ndarray,
    k_top_books: int = 10,
) -> list[list[dict]]:
    """Per basin, return top-k books by CSLS as a sorted-desc list of
    {"idx", "title", "csls"} dicts. Clamps k to N_books if smaller."""
    n_basins, n_books = csls_matrix.shape
    k = min(k_top_books, n_books)
    out: list[list[dict]] = []
    if k >= n_books:
        topk_idx = np.argsort(-csls_matrix, axis=1)[:, :k]
    else:
        part = np.argpartition(-csls_matrix, k - 1, axis=1)[:, :k]
        rows = np.arange(n_basins)[:, None]
        topk_idx = part[rows, np.argsort(-csls_matrix[rows, part], axis=1)]
    for g in range(n_basins):
        per_basin: list[dict] = []
        for local_b in topk_idx[g]:
            local_b_int = int(local_b)
            global_b = int(book_global_idx[local_b_int])
            title = str(all_titles[local_b_int]) if local_b_int < len(all_titles) else ""
            per_basin.append({
                "idx": global_b,
                "title": title,
                "csls": float(csls_matrix[g, local_b_int]),
            })
        out.append(per_basin)
    return out


PHASE3_FIELDNAMES = [
    "rank", "seed_article", "first_seed_type",
    "n_articles_kNN", "n_books", "book_members_json",
    "top_book_id", "top_book_title", "top_keywords",
    "top_books_csls_json", "effective_size",
]


def write_enriched_csv(
    out_path: Path,
    basin_meta: list[dict],
    book_members_per_basin: dict[int, list[dict]],
    n_articles_kNN_per_basin: dict[int, int],
    effective_size_per_basin: dict[int, float],
    top_keywords_per_basin: dict[int, str],
    top_books_csls_per_basin: dict[int, list[dict]],
) -> None:
    """Write the Phase 3 enriched CSV (summary schema; no per-article
    member lists — those live in the sidecar .npz).

    Output rows are sorted by effective_size descending; rank is
    re-assigned 1..N after sort.
    """
    rows: list[dict] = []
    n_basins = len(basin_meta)
    for b in range(n_basins):
        meta = basin_meta[b]
        if b not in effective_size_per_basin:
            continue
        top_books = top_books_csls_per_basin.get(b, [])
        top_book = top_books[0] if top_books else {"idx": -1, "title": "", "csls": 0.0}
        rows.append({
            "rank": meta["rank"],
            "seed_article": meta.get("first_seed", -1),
            "first_seed_type": meta.get("first_seed_type", ""),
            "n_articles_kNN": int(n_articles_kNN_per_basin.get(b, 0)),
            "n_books": int(meta.get("n_books", 0) or 0),
            "book_members_json": json.dumps(book_members_per_basin.get(b, [])),
            "top_book_id": top_book["idx"],
            "top_book_title": top_book["title"],
            "top_keywords": top_keywords_per_basin.get(b, ""),
            "top_books_csls_json": json.dumps(top_books),
            "effective_size": f"{effective_size_per_basin.get(b, 0.0):.3f}",
        })

    def sort_key(r):
        try:
            return -float(r["effective_size"])
        except (ValueError, TypeError):
            return 0.0
    rows.sort(key=sort_key)
    for new_rank, r in enumerate(rows, 1):
        r["rank"] = new_rank

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PHASE3_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def _carry_book_members(
    csv_path: Path,
    n_basins: int,
) -> dict[int, list[dict]]:
    """Re-read input CSV and carry through book members per basin
    (Phase 2 doesn't reassign books; they pass through unchanged)."""
    out: dict[int, list[dict]] = {b: [] for b in range(n_basins)}
    with csv_path.open() as f:
        for k, row in enumerate(csv.DictReader(f)):
            weights = json.loads(row["member_weights_json"])
            books = [w for w in weights if w.get("type") == "book"]
            out[k] = books
    return out


def run(
    scope: str,
    input_path: Path,
    output_path: Path,
    k_knn: int = 10,
    k_sigma: int = 7,
    k_marginal: int = 200,
    perplexity: float = 2.0,
    tau_supp: float = 0.5,
    min_final_size: int = 5,
    keywords_per_gap: int = 5,
    bigram_iterations: int = 2,
    vote_concentration_threshold: float = 0.25,
    title_overlap_threshold: float = 0.2,
    templated_concentration_threshold: float = 0.5,
) -> None:
    """Phase 2 enrichment pipeline. See module docstring for context."""
    print(f"Loading scope {scope}...", flush=True)
    emb, ids, types, titles, _topic_ids, _refs = load_scope(scope, with_refs=False)
    n = len(ids)
    types_arr = np.array(types)
    is_article = types_arr == "article"
    article_global_idx = np.where(is_article)[0]
    article_emb = emb[article_global_idx]
    norms = np.linalg.norm(article_emb, axis=1, keepdims=True)
    article_emb = (article_emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    print(f"  {n:,} works ({(~is_article).sum():,} books, {is_article.sum():,} articles)",
          flush=True)

    # The full-corpus emb (not article-only) is also needed for the
    # book-vote and keyword steps that index into the original embedding.
    full_emb = emb.copy()
    full_norms = np.linalg.norm(full_emb, axis=1, keepdims=True)
    full_emb = (full_emb / np.where(full_norms > 0, full_norms, 1.0)).astype(np.float32)
    full_emb = np.ascontiguousarray(full_emb)

    print("Loading soft-EM CSV...", flush=True)
    M0, basin_meta = load_soft_em_basins(input_path, is_article)
    n_basins = M0.shape[1]
    book_members_per_basin = _carry_book_members(input_path, n_basins)
    print(f"  {n_basins} basins; M0 shape {M0.shape}", flush=True)

    # We also need abstracts for keyword extraction. Read directly from
    # the per-scope parquet (matches gap_book_existence_filter:_load_articles_parquet).
    print("Loading article abstracts...", flush=True)
    import pyarrow.parquet as pq
    parts = scope.split("/")
    if Path("cache/works_parquet").joinpath(*parts, "article.parquet").exists():
        article_pq = Path("cache/works_parquet").joinpath(*parts, "article.parquet")
        tbl = pq.read_table(article_pq, columns=["display_name", "abstract"])
        article_titles = tbl.column("display_name").to_pylist()
        article_abstracts = tbl.column("abstract").to_pylist()
        # Map article-local titles/abstracts back to global indexing
        global_titles = list(titles)
        global_abstracts = [""] * n
        for local_i, g in enumerate(article_global_idx):
            global_titles[int(g)] = article_titles[local_i]
            global_abstracts[int(g)] = article_abstracts[local_i] or ""
    else:
        # Synthetic / no parquet: use load_scope titles + empty abstracts
        global_titles = list(titles)
        global_abstracts = [""] * n

    print(f"Running softmax (P={perplexity}, k_knn={k_knn})...", flush=True)
    M_new = run_softmax_step(
        article_emb=article_emb, M0=M0,
        k_knn=k_knn, k_sigma=k_sigma, perplexity=perplexity,
    )
    print("  done", flush=True)

    print(f"Crisp filter (tau_supp={tau_supp}, min_final_size={min_final_size})...",
          flush=True)
    surviving = crisp_filter(M_new, tau_supp=tau_supp, min_final_size=min_final_size)
    print(f"  {len(surviving)} basins survived", flush=True)

    book_anchored_mask = np.array(
        [int(m.get("book_anchored", 0)) == 1 for m in basin_meta], dtype=bool
    )
    cov = inherited_coverage_per_article(M_new, book_anchored_mask)
    readouts = per_basin_readouts(M_new, surviving, cov, book_anchored_mask)

    # Augment readouts with max_basin_rank / max_basin_books (which book-anchored
    # basin claims this basin's members most strongly)
    if book_anchored_mask.any():
        book_basin_indices = np.where(book_anchored_mask)[0]
        book_columns = M_new[:, book_anchored_mask]   # (N_articles, n_book_basins)
        argmax_book = book_columns.argmax(axis=1)     # (N_articles,)
        for b, members_local_w in surviving.items():
            if book_anchored_mask[b]:
                continue
            member_idx = np.array([i for i, _ in members_local_w], dtype=np.int64)
            if len(member_idx) == 0:
                continue
            chosen_book_basin_local = int(np.bincount(argmax_book[member_idx]).argmax())
            chosen_book_basin = int(book_basin_indices[chosen_book_basin_local])
            readouts[b]["inherited_coverage_max_basin_rank"] = basin_meta[chosen_book_basin]["rank"]
            readouts[b]["inherited_coverage_max_basin_books"] = basin_meta[chosen_book_basin].get("n_books", 0)
    else:
        for b in surviving:
            readouts[b].setdefault("inherited_coverage_max_basin_rank", "")
            readouts[b].setdefault("inherited_coverage_max_basin_books", "")

    print("Computing book signals + keywords...", flush=True)
    book_signals = tag_basin_book_signals(
        surviving=surviving,
        emb=full_emb,
        types_arr=types_arr,
        titles=global_titles,
        abstracts=global_abstracts,
        article_global_idx=article_global_idx,
        keywords_per_gap=keywords_per_gap,
        bigram_iterations=bigram_iterations,
        vote_concentration_threshold=vote_concentration_threshold,
        title_overlap_threshold=title_overlap_threshold,
        templated_concentration_threshold=templated_concentration_threshold,
    )

    print(f"Writing {output_path}...", flush=True)
    write_enriched_csv(
        out_path=output_path,
        M_new=M_new,
        surviving=surviving,
        basin_meta=basin_meta,
        book_members_per_basin=book_members_per_basin,
        article_global_idx=article_global_idx,
        readouts=readouts,
        book_signals=book_signals,
    )
    print("Done.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k-knn", type=int, default=10)
    p.add_argument("--k-sigma", type=int, default=7)
    p.add_argument("--k-marginal", type=int, default=200)
    p.add_argument("--perplexity", type=float, default=2.0)
    p.add_argument("--tau-supp", type=float, default=0.5)
    p.add_argument("--min-final-size", type=int, default=5)
    p.add_argument("--keywords-per-gap", type=int, default=5)
    p.add_argument("--bigram-iterations", type=int, default=2)
    p.add_argument("--vote-concentration-threshold", type=float, default=0.25)
    p.add_argument("--title-overlap-threshold", type=float, default=0.2)
    p.add_argument("--templated-concentration-threshold", type=float, default=0.5)
    args = p.parse_args()
    run(
        scope=args.scope,
        input_path=Path(args.input),
        output_path=Path(args.output),
        k_knn=args.k_knn,
        k_sigma=args.k_sigma,
        k_marginal=args.k_marginal,
        perplexity=args.perplexity,
        tau_supp=args.tau_supp,
        min_final_size=args.min_final_size,
        keywords_per_gap=args.keywords_per_gap,
        bigram_iterations=args.bigram_iterations,
        vote_concentration_threshold=args.vote_concentration_threshold,
        title_overlap_threshold=args.title_overlap_threshold,
        templated_concentration_threshold=args.templated_concentration_threshold,
    )


if __name__ == "__main__":
    main()
