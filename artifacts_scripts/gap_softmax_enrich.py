"""Multi-cluster softmax Phase 3 enrichment.

Replaces gap_book_existence_filter + gap_multi_scale_coverage with a
single one-shot multi-cluster softmax refinement layer over soft-EM's
basin set. Output is a minimal threshold-free 12-column summary CSV
plus a sidecar .npz containing the sparse softmax membership matrix.
No per-basin article lists in the CSV (load the npz, threshold M_new
yourself).

Spec: docs/superpowers/specs/2026-05-09-multicluster-softmax-phase3-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Soft-EM and Phase 3 CSVs both have member_weights_json fields > 131072 bytes
csv.field_size_limit(sys.maxsize)

from artifacts_scripts.gap_softmax_reassign import (
    compute_basin_centroids,
    compute_self_tuning_sigma,
    load_enriched_csv,
    softmax_e_step,
)

from artifacts_scripts._keyword_helpers import compute_gap_keywords
from artifacts_scripts.hdbscan_gap_probe import load_scope


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


def write_memberships_npz(
    out_path: Path,
    M_new: np.ndarray,
    article_global_idx: np.ndarray,
    basin_rank: np.ndarray,
) -> None:
    """Save the sparse softmax membership matrix + row/col labels as
    an .npz sidecar.

    Schema:
      - M_data, M_indices, M_indptr, M_shape: sparse CSR components
        of M_new (allow_pickle=False compatible).
      - article_global_idx: (N_articles,) int64, row labels.
      - basin_rank: (N_basins,) int64, column labels — each col's
        rank in the corresponding CSV row.

    To reload:
      data = np.load(path, allow_pickle=False)
      M = scipy.sparse.csr_matrix(
          (data['M_data'], data['M_indices'], data['M_indptr']),
          shape=tuple(data['M_shape']),
      )
    """
    import scipy.sparse as sp
    sparse = sp.csr_matrix(M_new)
    np.savez(
        out_path,
        M_data=sparse.data,
        M_indices=sparse.indices,
        M_indptr=sparse.indptr,
        M_shape=np.array(sparse.shape, dtype=np.int64),
        article_global_idx=article_global_idx.astype(np.int64),
        basin_rank=basin_rank.astype(np.int64),
    )


def _carry_book_members(
    csv_path: Path,
    n_basins: int,
) -> dict[int, list[dict]]:
    """Re-read input CSV and carry through book members per basin
    (Phase 3 doesn't reassign books; they pass through unchanged)."""
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
    keywords_per_gap: int = 5,
    bigram_iterations: int = 2,
    k_csls: int = 10,
    k_top_books: int = 10,
) -> None:
    """Phase 3 enrichment pipeline.

    1. Load scope embeddings + soft-EM CSV → M0
    2. Self-tuning σ + one-shot softmax E-step → M_new (full kNN-truncated)
    3. Compute basin centroids + per-(basin, book) CSLS matrix →
       top-k books per basin
    4. Compute weighted keywords per basin (input: full kNN-truncated
       softmax members + weights)
    5. Compute effective_size = Σ_a M_new[a, G] per basin
    6. Write 12-column CSV + sidecar .memberships.npz
    """
    print(f"Loading scope {scope}...", flush=True)
    emb, ids, types, titles, _topic_ids, _refs = load_scope(scope, with_refs=False)
    n = len(ids)
    types_arr = np.array(types)
    is_article = types_arr == "article"
    article_global_idx = np.where(is_article)[0]
    book_global_idx = np.where(~is_article)[0]

    article_emb = emb[article_global_idx]
    norms = np.linalg.norm(article_emb, axis=1, keepdims=True)
    article_emb = (article_emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)

    book_emb = emb[book_global_idx]
    book_norms = np.linalg.norm(book_emb, axis=1, keepdims=True)
    book_emb = (book_emb / np.where(book_norms > 0, book_norms, 1.0)).astype(np.float32)

    print(f"  {n:,} works ({len(book_global_idx):,} books, "
          f"{len(article_global_idx):,} articles)", flush=True)

    print("Loading soft-EM CSV...", flush=True)
    M0, basin_meta = load_soft_em_basins(input_path, is_article)
    n_basins = M0.shape[1]
    book_members_per_basin = _carry_book_members(input_path, n_basins)
    print(f"  {n_basins} basins; M0 shape {M0.shape}", flush=True)

    # Article abstracts for keyword extraction
    print("Loading article abstracts...", flush=True)
    import pyarrow.parquet as pq
    parts = scope.split("/")
    article_pq_path = Path("cache/works_parquet").joinpath(*parts, "article.parquet")
    if article_pq_path.exists():
        tbl = pq.read_table(article_pq_path, columns=["display_name", "abstract"])
        article_titles = tbl.column("display_name").to_pylist()
        article_abstracts = tbl.column("abstract").to_pylist()
        global_titles = list(titles)
        global_abstracts = [""] * n
        for local_i, g in enumerate(article_global_idx):
            global_titles[int(g)] = article_titles[local_i]
            global_abstracts[int(g)] = article_abstracts[local_i] or ""
    else:
        global_titles = list(titles)
        global_abstracts = [""] * n

    print(f"Running softmax (P={perplexity}, k_knn={k_knn})...", flush=True)
    M_new = run_softmax_step(
        article_emb=article_emb, M0=M0,
        k_knn=k_knn, k_sigma=k_sigma, perplexity=perplexity,
    )
    print("  done", flush=True)

    # CSLS book-anchoring (computed for ALL basins; basin centroids from M_new)
    print(f"Computing basin centroids + CSLS (k_csls={k_csls}, "
          f"k_top_books={k_top_books})...", flush=True)
    centroids_all = compute_basin_centroids(article_emb, M_new)
    csls_matrix = compute_book_csls_matrix(
        centroids_all, book_emb, k_csls=k_csls,
    )
    # IMPORTANT: top_books_csls_per_basin indexes all_titles by book-LOCAL
    # index (parallel to book_global_idx). Pass the book-subset titles
    # array, not the full corpus titles array.
    all_titles_corpus = np.array(titles, dtype=object)
    book_titles = all_titles_corpus[book_global_idx]
    top_books_per_basin_list = top_books_csls_per_basin(
        csls_matrix=csls_matrix,
        book_global_idx=book_global_idx,
        all_titles=book_titles,
        k_top_books=k_top_books,
    )
    top_books_csls_dict = {b: top_books_per_basin_list[b] for b in range(n_basins)}

    # Weighted keyword extraction over the full kNN-truncated membership
    print(f"Computing weighted keywords (top_k={keywords_per_gap})...", flush=True)
    gaps_global: list[list[int]] = []
    weights_per_basin: list[list[float]] = []
    n_articles_kNN_dict: dict[int, int] = {}
    for b in range(n_basins):
        col = M_new[:, b]
        nonzero_local = np.where(col > 0)[0]
        n_articles_kNN_dict[b] = int(len(nonzero_local))
        order = np.argsort(-col[nonzero_local])
        nonzero_local = nonzero_local[order]
        gaps_global.append(
            [int(article_global_idx[lidx]) for lidx in nonzero_local]
        )
        weights_per_basin.append(
            [float(col[lidx]) for lidx in nonzero_local]
        )

    keywords_per_basin = compute_gap_keywords(
        gaps=gaps_global,
        titles=global_titles,
        abstracts=global_abstracts,
        weights=weights_per_basin,
        top_k=keywords_per_gap,
        bigram_iterations=bigram_iterations,
        verbose=False,
    )
    top_keywords_dict = {
        b: " ".join(kw for kw, _ in keywords_per_basin[b])
        for b in range(n_basins)
    }

    # effective_size = basin softmax mass
    effective_size_dict = {
        b: float(M_new[:, b].sum()) for b in range(n_basins)
    }

    print(f"Writing {output_path}...", flush=True)
    write_enriched_csv(
        out_path=output_path,
        basin_meta=basin_meta,
        book_members_per_basin=book_members_per_basin,
        n_articles_kNN_per_basin=n_articles_kNN_dict,
        effective_size_per_basin=effective_size_dict,
        top_keywords_per_basin=top_keywords_dict,
        top_books_csls_per_basin=top_books_csls_dict,
    )

    npz_path = output_path.with_suffix(".csv.memberships.npz")
    print(f"Writing {npz_path}...", flush=True)
    basin_rank_arr = np.array([m["rank"] for m in basin_meta], dtype=np.int64)
    write_memberships_npz(
        out_path=npz_path,
        M_new=M_new,
        article_global_idx=article_global_idx,
        basin_rank=basin_rank_arr,
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
    p.add_argument("--keywords-per-gap", type=int, default=5)
    p.add_argument("--bigram-iterations", type=int, default=2)
    p.add_argument("--k-csls", type=int, default=10,
                   help="Hub-correction neighborhood for CSLS (mycooc convention).")
    p.add_argument("--k-top-books", type=int, default=10,
                   help="How many top-CSLS books to retain per basin.")
    args = p.parse_args()
    run(
        scope=args.scope,
        input_path=Path(args.input),
        output_path=Path(args.output),
        k_knn=args.k_knn,
        k_sigma=args.k_sigma,
        k_marginal=args.k_marginal,
        perplexity=args.perplexity,
        keywords_per_gap=args.keywords_per_gap,
        bigram_iterations=args.bigram_iterations,
        k_csls=args.k_csls,
        k_top_books=args.k_top_books,
    )


if __name__ == "__main__":
    main()
