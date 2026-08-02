"""Multi-cluster softmax Phase 3 enrichment.

Replaces gap_book_existence_filter + gap_multi_scale_coverage with a
single one-shot multi-cluster softmax refinement layer over soft-EM's
basin set. Output is a minimal threshold-free 12-column summary CSV
plus a sidecar .npz containing the sparse softmax membership matrix.
No per-basin article lists in the CSV (load the npz, threshold M_new
yourself).
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
    knn_basin_lookup,
    load_enriched_csv,
    fuzzy_deduplicate_basins,
    softmax_e_step,
    softmax_with_optional_empty,
    tune_tau_for_perplexity,
)

from artifacts_scripts.keyword_helpers import compute_gap_keywords
from artifacts_scripts.scope_loader import load_scope
from bookgap.index import build_index, FlatL2


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



def compute_book_basin_softmax(
    book_emb: np.ndarray,
    book_sigma: np.ndarray,
    article_emb: np.ndarray,
    article_sigma: np.ndarray,
    M_new: np.ndarray,
    k_knn: int = 10,
    perplexity: float = 2.0,
    k_marginal: int = 200,
) -> np.ndarray:
    """Symmetric to article→basin softmax: compute book→basin softmax
    membership.

    For each book B:
      1. Find B's top-k_knn nearest basin centroids (cos sim against
         centroids derived from M_new on article_emb).
      2. Per-basin density at B's location:
           K̃_G(B) = (1 / Σ_x M_new[x, G]) · Σ_x M_new[x, G] · κ(B, x)
         summed over B's top-k_marginal nearest articles
         (kernel-tail truncation, same as compute_log_K_basins).
      3. τ-tune for target perplexity (one global τ across all books).
      4. Softmax over B's k_knn candidate basins.

    Returns:
        M_books: (N_books, N_basins) float32. Each row is a book's
        softmax distribution over its top-k_knn basins; other entries
        are 0 (kNN truncation).
    """
    n_books = book_emb.shape[0]
    n_basins = M_new.shape[1]

    # 1. Basin centroids from articles + M_new
    centroids = compute_basin_centroids(article_emb, M_new)
    # 2. Each book's top-k_knn basins
    knn_book_basins = knn_basin_lookup(book_emb, centroids, k=k_knn)  # (N_books, k_knn)

    # 3. Book→article kNN for kernel truncation
    article_index = build_index(FlatL2(), article_emb.astype(np.float32))
    D, I = article_index.search(
        book_emb.astype(np.float32), k=min(k_marginal, article_emb.shape[0])
    )  # squared L2 distances, neighbor indices
    sigma_a_at = article_sigma.astype(np.float64)[I]      # (N_books, K_marg)
    book_sigma_f64 = book_sigma.astype(np.float64)
    d2 = D.astype(np.float64)
    kappa = np.exp(-d2 / (book_sigma_f64[:, None] * sigma_a_at + 1e-30))   # (N_books, K_marg)

    # 4. Per (book, kNN basin) density. Loop over k_knn slot to keep memory at (N_books, K_marg).
    denom_full = M_new.sum(axis=0)                          # (N_basins,)
    denom_at_kNN = denom_full[knn_book_basins]              # (N_books, k_knn)
    log_K = np.zeros((n_books, k_knn), dtype=np.float64)
    for k_col in range(k_knn):
        basin_id = knn_book_basins[:, k_col]                # (N_books,)
        # Gather M_new[I[b, j], basin_id[b]] for each (book b, neighbor j)
        M_at = M_new[I, basin_id[:, None]]                  # (N_books, K_marg)
        numerator = (kappa * M_at).sum(axis=1)              # (N_books,)
        K_tilde_col = numerator / np.maximum(denom_at_kNN[:, k_col], 1e-30)
        log_K[:, k_col] = np.log(K_tilde_col + 1e-30)

    # 5. τ tune (one global τ across all books, same as articles)
    tau = tune_tau_for_perplexity(
        log_K, log_K_empty=None, P_target=perplexity,
    )
    probs = softmax_with_optional_empty(log_K, None, tau)   # (N_books, k_knn)

    # 6. Scatter probs into full (N_books, N_basins) matrix
    M_books = np.zeros((n_books, n_basins), dtype=np.float32)
    rows = np.arange(n_books)[:, None]
    M_books[rows, knn_book_basins] = probs.astype(np.float32)
    return M_books


def top_books_membership_per_basin(
    M_books: np.ndarray,
    book_global_idx: np.ndarray,
    all_titles: np.ndarray,
    k_top_books: int = 10,
) -> list[list[dict]]:
    """Per basin, return the top-k books by softmax membership weight
    as a sorted-desc list of {"idx", "title", "weight"} dicts.

    Books with zero weight on a basin are excluded (most basins have
    at most a few books with nonzero weight due to kNN truncation
    on the book side).
    """
    n_books, n_basins = M_books.shape
    out: list[list[dict]] = []
    for g in range(n_basins):
        col = M_books[:, g]
        nonzero_local = np.where(col > 0)[0]
        if len(nonzero_local) == 0:
            out.append([])
            continue
        order = np.argsort(-col[nonzero_local])
        top_local = nonzero_local[order[:k_top_books]]
        per_basin: list[dict] = []
        for local_b in top_local:
            local_b_int = int(local_b)
            global_b = int(book_global_idx[local_b_int])
            title = (
                str(all_titles[local_b_int])
                if local_b_int < len(all_titles)
                else ""
            )
            per_basin.append({
                "idx": global_b,
                "title": title,
                "weight": float(col[local_b_int]),
            })
        out.append(per_basin)
    return out


PHASE3_FIELDNAMES = [
    "rank", "seed_article", "first_seed_type",
    "n_articles_kNN", "n_books", "book_members_json",
    "top_book_id", "top_book_title", "top_keywords",
    "top_books_membership_json", "effective_size",
]


def write_enriched_csv(
    out_path: Path,
    basin_meta: list[dict],
    book_members_per_basin: dict[int, list[dict]],
    n_articles_kNN_per_basin: dict[int, int],
    effective_size_per_basin: dict[int, float],
    top_keywords_per_basin: dict[int, str],
    top_books_membership_per_basin: dict[int, list[dict]],
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
        top_books = top_books_membership_per_basin.get(b, [])
        top_book = top_books[0] if top_books else {"idx": -1, "title": "", "weight": 0.0}
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
            "top_books_membership_json": json.dumps(top_books),
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
    M_books: np.ndarray | None = None,
    book_global_idx: np.ndarray | None = None,
) -> None:
    """Save the sparse softmax membership matrix + row/col labels as
    an .npz sidecar.

    Schema:
      - M_data, M_indices, M_indptr, M_shape: sparse CSR components
        of M_new (allow_pickle=False compatible).
      - article_global_idx: (N_articles,) int64, row labels.
      - basin_rank: (N_basins,) int64, column labels — each col's
        rank in the corresponding CSV row.
      - M_books_data, M_books_indices, M_books_indptr, M_books_shape:
        optional sparse CSR components of M_books (book→basin softmax).
      - book_global_idx: (N_books,) int64, book row labels (optional).

    To reload:
      data = np.load(path, allow_pickle=False)
      M = scipy.sparse.csr_matrix(
          (data['M_data'], data['M_indices'], data['M_indptr']),
          shape=tuple(data['M_shape']),
      )
    """
    import scipy.sparse as sp
    sparse_M = sp.csr_matrix(M_new)
    save_dict = dict(
        M_data=sparse_M.data,
        M_indices=sparse_M.indices,
        M_indptr=sparse_M.indptr,
        M_shape=np.array(sparse_M.shape, dtype=np.int64),
        article_global_idx=article_global_idx.astype(np.int64),
        basin_rank=basin_rank.astype(np.int64),
    )
    if M_books is not None:
        sparse_books = sp.csr_matrix(M_books)
        save_dict.update(
            M_books_data=sparse_books.data,
            M_books_indices=sparse_books.indices,
            M_books_indptr=sparse_books.indptr,
            M_books_shape=np.array(sparse_books.shape, dtype=np.int64),
        )
        if book_global_idx is not None:
            save_dict["book_global_idx"] = book_global_idx.astype(np.int64)
    np.savez(out_path, **save_dict)


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
    k_top_books: int = 10,
    fuzzy_dedup_threshold: float | None = None,
) -> None:
    """Phase 3 enrichment pipeline.

    1. Load scope embeddings + soft-EM CSV → M0
    2. (Optional) L1-generalized fuzzy deduplication on M0
    3. Self-tuning σ (articles) + one-shot softmax E-step → M_new (full kNN-truncated)
    4. Compute book σ + book→basin softmax → M_books → top-k books per basin
    5. Compute weighted keywords per basin (input: full kNN-truncated
       softmax members + weights)
    6. Compute effective_size = Σ_a M_new[a, G] per basin
    7. Write 12-column CSV + sidecar .memberships.npz (includes M_books)
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

    if fuzzy_dedup_threshold is not None:
        print(f"Fuzzy deduplicating basins (threshold={fuzzy_dedup_threshold})...", flush=True)
        M0, basin_meta, keep_indices = fuzzy_deduplicate_basins(
            M0, basin_meta, fuzzy_dedup_threshold
        )
        n_basins = M0.shape[1]
        book_members_per_basin = {k: book_members_per_basin[idx] for k, idx in enumerate(keep_indices)}
        print(f"  reduced to {n_basins} basins; M0 shape {M0.shape}", flush=True)

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
    article_sigma = compute_self_tuning_sigma(article_emb, k_sigma=k_sigma)
    M_new = run_softmax_step(
        article_emb=article_emb, M0=M0,
        k_knn=k_knn, k_sigma=k_sigma, perplexity=perplexity,
    )
    print("  done", flush=True)

    # Book→basin softmax (replaces CSLS). Basin centroids from M_new.
    print(f"Computing book→basin softmax (k_knn={k_knn}, k_top_books={k_top_books})...",
          flush=True)
    book_sigma = compute_self_tuning_sigma(book_emb, k_sigma=k_sigma)
    M_books = compute_book_basin_softmax(
        book_emb=book_emb,
        book_sigma=book_sigma,
        article_emb=article_emb,
        article_sigma=article_sigma,
        M_new=M_new,
        k_knn=k_knn,
        perplexity=perplexity,
        k_marginal=k_marginal,
    )
    # IMPORTANT: top_books_membership_per_basin indexes all_titles by book-LOCAL
    # index (parallel to book_global_idx). Pass the book-subset titles
    # array, not the full corpus titles array.
    all_titles_corpus = np.array(titles, dtype=object)
    book_titles = all_titles_corpus[book_global_idx]
    top_books_per_basin_list = top_books_membership_per_basin(
        M_books=M_books,
        book_global_idx=book_global_idx,
        all_titles=book_titles,
        k_top_books=k_top_books,
    )
    top_books_membership_dict = {b: top_books_per_basin_list[b] for b in range(n_basins)}

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
        top_books_membership_per_basin=top_books_membership_dict,
    )

    npz_path = output_path.with_suffix(".csv.memberships.npz")
    print(f"Writing {npz_path}...", flush=True)
    basin_rank_arr = np.array([m["rank"] for m in basin_meta], dtype=np.int64)
    write_memberships_npz(
        out_path=npz_path,
        M_new=M_new,
        article_global_idx=article_global_idx,
        basin_rank=basin_rank_arr,
        M_books=M_books,
        book_global_idx=book_global_idx,
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
    p.add_argument("--k-top-books", type=int, default=10,
                   help="How many top-membership books to retain per basin.")
    p.add_argument("--fuzzy-dedup-threshold", type=float, default=0.75,
                   help="Optional L1-generalized overlap threshold for greedy fold-in dedup before softmax. Default 0.75.")
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
        k_top_books=args.k_top_books,
        fuzzy_dedup_threshold=args.fuzzy_dedup_threshold,
    )


if __name__ == "__main__":
    main()
