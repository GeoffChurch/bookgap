import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Soft-EM CSV member_weights_json fields can exceed default 131072 bytes
csv.field_size_limit(sys.maxsize)


def write_soft_em_csv(path: Path, basins: list[dict]) -> None:
    """Helper: write a synthetic soft-EM raw CSV with the columns the
    Phase 2 driver reads."""
    fields = [
        "rank", "seed_article", "first_seed_type",
        "n_articles", "n_books", "book_anchored",
        "member_indices_json", "member_weights_json",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, b in enumerate(basins):
            w.writerow({
                "rank": k + 1,
                "seed_article": b.get("seed_article", -1),
                "first_seed_type": b.get("first_seed_type", "article"),
                "n_articles": b.get("n_articles", 0),
                "n_books": b.get("n_books", 0),
                "book_anchored": b.get("book_anchored", 0),
                "member_indices_json": json.dumps(b.get("member_indices", [])),
                "member_weights_json": json.dumps(b.get("member_weights", [])),
            })


def test_load_soft_em_csv_returns_M0_and_meta(tmp_path: Path):
    """The driver reuses gap_softmax_reassign.load_enriched_csv for parsing,
    so this test just confirms the integration: synthetic CSV roundtrips
    into a (N_articles, N_basins) M0 matrix."""
    from artifacts_scripts.gap_softmax_enrich import load_soft_em_basins

    csv_path = tmp_path / "soft_em.csv"
    write_soft_em_csv(csv_path, [
        {
            "seed_article": 100,
            "n_articles": 2, "n_books": 0, "book_anchored": 0,
            "member_indices": [{"idx": 100, "type": "article"},
                               {"idx": 101, "type": "article"}],
            "member_weights": [{"idx": 100, "type": "article", "weight": 1.0},
                               {"idx": 101, "type": "article", "weight": 0.7}],
        },
        {
            "seed_article": 200,
            "n_articles": 1, "n_books": 1, "book_anchored": 1,
            "member_indices": [{"idx": 200, "type": "article"},
                               {"idx": 300, "type": "book"}],
            "member_weights": [{"idx": 200, "type": "article", "weight": 0.9},
                               {"idx": 300, "type": "book", "weight": 1.0}],
        },
    ])

    is_article = np.zeros(400, dtype=bool)
    is_article[[100, 101, 200]] = True

    M0, basin_meta = load_soft_em_basins(csv_path, is_article)
    assert M0.shape == (3, 2)
    assert M0[0, 0] == pytest.approx(1.0)
    assert basin_meta[1]["book_anchored"] == 1
    assert basin_meta[1]["n_books"] == 1


def test_run_softmax_step_returns_membership_matrix(tmp_path: Path):
    """run_softmax_step composes σ, centroids, kNN, K̃, τ, softmax in one
    call. Returns M_new shape (N_articles, N_basins) with rows summing
    to 1 (per the per-article kNN truncation; sums are over kNN basins
    plus zeros elsewhere)."""
    from artifacts_scripts.gap_softmax_enrich import run_softmax_step

    rng = np.random.default_rng(11)
    N, d, K = 30, 16, 5
    article_emb = rng.normal(size=(N, d)).astype(np.float32)
    article_emb /= np.linalg.norm(article_emb, axis=1, keepdims=True)
    M0 = rng.uniform(0, 1, size=(N, K)).astype(np.float32)

    M_new = run_softmax_step(
        article_emb=article_emb, M0=M0,
        k_knn=4, k_sigma=7, perplexity=2.0,
    )
    assert M_new.shape == (N, K)
    np.testing.assert_allclose(M_new.sum(axis=1), 1.0, atol=1e-5)
    assert ((M_new > 0).sum(axis=1) <= 4).all()


def test_write_enriched_csv_phase3_schema(tmp_path: Path):
    """Phase 3 CSV: 12 summary columns, no per-article member lists.
    Books carry through as book_members_json (compact)."""
    from artifacts_scripts.gap_softmax_enrich import write_enriched_csv

    basin_meta = [
        {"rank": 1, "first_seed": 100, "first_seed_type": "article",
         "n_books": 0, "book_anchored": 0},
        {"rank": 2, "first_seed": 200, "first_seed_type": "article",
         "n_books": 1, "book_anchored": 1},
        {"rank": 3, "first_seed": 300, "first_seed_type": "article",
         "n_books": 0, "book_anchored": 0},
    ]
    book_members_per_basin = {
        0: [],
        1: [{"idx": 999, "type": "book", "weight": 1.0}],
        2: [],
    }
    n_articles_kNN_per_basin = {0: 200, 1: 50, 2: 1138}
    effective_size_per_basin = {0: 1.3, 1: 0.7, 2: 1129.6}
    top_keywords_per_basin = {0: "kw1 kw2", 1: "kw3", 2: "que para como"}
    top_books_csls_per_basin_dict = {
        0: [{"idx": 999, "title": "Some Book", "csls": 0.42}],
        1: [{"idx": 777, "title": "Anchor Book", "csls": 0.95}],
        2: [{"idx": 888, "title": "Other Book", "csls": 0.10}],
    }

    out = tmp_path / "phase3.csv"
    write_enriched_csv(
        out_path=out,
        basin_meta=basin_meta,
        book_members_per_basin=book_members_per_basin,
        n_articles_kNN_per_basin=n_articles_kNN_per_basin,
        effective_size_per_basin=effective_size_per_basin,
        top_keywords_per_basin=top_keywords_per_basin,
        top_books_csls_per_basin=top_books_csls_per_basin_dict,
    )

    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 3
    expected_cols = {
        "rank", "seed_article", "first_seed_type",
        "n_articles_kNN", "n_books", "book_members_json",
        "top_book_id", "top_book_title", "top_keywords",
        "top_books_csls_json", "effective_size",
    }
    assert set(rows[0].keys()) == expected_cols
    for legacy in ("coverage_label_v2", "templated_leak_flag",
                   "book_exists_flag", "inherited_coverage_median_post",
                   "effective_uncovered_articles", "vote_concentration",
                   "member_indices_json", "member_weights_json",
                   "total_size"):
        assert legacy not in rows[0]
    sizes = [float(r["effective_size"]) for r in rows]
    assert sizes == sorted(sizes, reverse=True)


def test_compute_book_csls_matrix_high_for_anchored_basin():
    """When a basin centroid is close to one specific book and far
    from all others, CSLS to that book should be substantially higher
    than CSLS to other books."""
    from artifacts_scripts.gap_softmax_enrich import compute_book_csls_matrix

    book_emb = np.eye(3, dtype=np.float32)
    basin_centroids = np.array([
        [0.95, 0.30, 0.10],
        [0.30, 0.95, 0.10],
    ], dtype=np.float32)
    basin_centroids /= np.linalg.norm(basin_centroids, axis=1, keepdims=True)

    csls = compute_book_csls_matrix(basin_centroids, book_emb, k_csls=2)
    assert csls.shape == (2, 3)
    assert csls[0, 0] > csls[0, 1]
    assert csls[0, 0] > csls[0, 2]
    assert csls[1, 1] > csls[1, 0]
    assert csls[1, 1] > csls[1, 2]


def test_compute_book_csls_matrix_handles_more_basins_than_books():
    from artifacts_scripts.gap_softmax_enrich import compute_book_csls_matrix
    rng = np.random.default_rng(3)
    basin_centroids = rng.normal(size=(5, 8)).astype(np.float32)
    basin_centroids /= np.linalg.norm(basin_centroids, axis=1, keepdims=True)
    book_emb = rng.normal(size=(2, 8)).astype(np.float32)
    book_emb /= np.linalg.norm(book_emb, axis=1, keepdims=True)
    csls = compute_book_csls_matrix(basin_centroids, book_emb, k_csls=5)
    assert csls.shape == (5, 2)


def test_compute_book_csls_matrix_low_when_book_is_hub():
    from artifacts_scripts.gap_softmax_enrich import compute_book_csls_matrix
    book_emb = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    basin_centroids = np.array([
        [1.0, 0.05],
        [0.7, 0.7],
        [1.0, 0.1],
    ], dtype=np.float32)
    basin_centroids /= np.linalg.norm(basin_centroids, axis=1, keepdims=True)
    csls = compute_book_csls_matrix(basin_centroids, book_emb, k_csls=2)
    assert csls[1, 1] >= csls[1, 0] - 1e-6


def test_top_books_csls_per_basin_returns_sorted_top_k():
    from artifacts_scripts.gap_softmax_enrich import top_books_csls_per_basin
    csls_matrix = np.array([
        [0.5, 0.2, 0.8, 0.1, 0.4],
        [0.1, 0.6, 0.3, 0.9, 0.2],
    ], dtype=np.float32)
    book_global_idx = np.array([100, 101, 102, 103, 104], dtype=np.int64)
    all_titles = np.array(
        ["Book A", "Book B", "Book C", "Book D", "Book E"], dtype=object
    )
    top = top_books_csls_per_basin(
        csls_matrix=csls_matrix,
        book_global_idx=book_global_idx,
        all_titles=all_titles,
        k_top_books=3,
    )
    assert len(top) == 2
    assert len(top[0]) == 3
    assert top[0][0]["idx"] == 102
    assert top[0][0]["title"] == "Book C"
    assert top[0][0]["csls"] == pytest.approx(0.8, abs=1e-5)
    assert top[1][0]["idx"] == 103


def test_top_books_csls_per_basin_clamps_when_k_exceeds_n_books():
    from artifacts_scripts.gap_softmax_enrich import top_books_csls_per_basin
    csls_matrix = np.array([[0.5, 0.2, 0.8]], dtype=np.float32)
    book_global_idx = np.array([10, 20, 30], dtype=np.int64)
    all_titles = np.array(["A", "B", "C"], dtype=object)
    top = top_books_csls_per_basin(
        csls_matrix=csls_matrix,
        book_global_idx=book_global_idx,
        all_titles=all_titles,
        k_top_books=10,
    )
    assert len(top[0]) == 3


def test_book_basin_softmax_book_assigned_to_nearer_basin():
    """User's scenario: book is geometrically close to a small tight
    basin (basin 0) and farther from a larger basin (basin 1).
    The book's softmax should concentrate on basin 0."""
    from artifacts_scripts.gap_softmax_enrich import compute_book_basin_softmax

    rng = np.random.default_rng(7)
    d = 8

    # Basin 0: 3 articles tightly clustered around (1, 0, 0, ...)
    # Basin 1: 30 articles loosely clustered around (0.7, 0.7, 0, ...)
    # Book is at (0.95, 0.05, 0, ...) — close to basin 0's center, farther from basin 1.
    base_a = np.array([1.0] + [0.0] * (d - 1), dtype=np.float32)
    base_b = np.array([0.7, 0.7] + [0.0] * (d - 2), dtype=np.float32)

    arts_a = base_a + 0.05 * rng.normal(size=(3, d)).astype(np.float32)
    arts_b = base_b + 0.30 * rng.normal(size=(30, d)).astype(np.float32)
    article_emb = np.concatenate([arts_a, arts_b], axis=0)
    article_emb /= np.linalg.norm(article_emb, axis=1, keepdims=True)

    M_new = np.zeros((33, 2), dtype=np.float32)
    M_new[:3, 0] = 1.0
    M_new[3:, 1] = 1.0

    book_emb = np.array([[0.95, 0.05] + [0.0] * (d - 2)], dtype=np.float32)
    book_emb /= np.linalg.norm(book_emb, axis=1, keepdims=True)

    article_sigma = np.full(33, 0.3, dtype=np.float32)
    book_sigma = np.array([0.3], dtype=np.float32)

    M_books = compute_book_basin_softmax(
        book_emb=book_emb,
        book_sigma=book_sigma,
        article_emb=article_emb,
        article_sigma=article_sigma,
        M_new=M_new,
        k_knn=2,
        perplexity=1.5,
        k_marginal=10,
    )
    assert M_books.shape == (1, 2)
    assert np.isclose(M_books.sum(axis=1)[0], 1.0, atol=1e-5)
    # Book's softmax should weight basin 0 (its tight neighbor) more than basin 1
    assert M_books[0, 0] > M_books[0, 1], (
        f"book should belong to tight nearby basin 0, not larger far basin 1; "
        f"got M_books={M_books}"
    )


def test_book_basin_softmax_returns_full_matrix_with_sparsity():
    """Sanity: M_books shape is (N_books, N_basins). Each book has
    nonzero weight on at most k_knn basins (kNN truncation)."""
    from artifacts_scripts.gap_softmax_enrich import compute_book_basin_softmax

    rng = np.random.default_rng(11)
    d = 16
    article_emb = rng.normal(size=(20, d)).astype(np.float32)
    article_emb /= np.linalg.norm(article_emb, axis=1, keepdims=True)
    book_emb = rng.normal(size=(5, d)).astype(np.float32)
    book_emb /= np.linalg.norm(book_emb, axis=1, keepdims=True)

    M_new = rng.uniform(0, 1, size=(20, 4)).astype(np.float32)
    M_new = M_new / M_new.sum(axis=1, keepdims=True)
    article_sigma = np.full(20, 0.3, dtype=np.float32)
    book_sigma = np.full(5, 0.3, dtype=np.float32)

    M_books = compute_book_basin_softmax(
        book_emb=book_emb,
        book_sigma=book_sigma,
        article_emb=article_emb,
        article_sigma=article_sigma,
        M_new=M_new,
        k_knn=2,
        perplexity=1.5,
        k_marginal=10,
    )
    assert M_books.shape == (5, 4)
    # Rows sum to 1 (full softmax across kNN basins, others zero)
    np.testing.assert_allclose(M_books.sum(axis=1), 1.0, atol=1e-5)
    # At most k_knn=2 nonzero entries per row
    nonzero_per_row = (M_books > 0).sum(axis=1)
    assert (nonzero_per_row <= 2).all(), f"got nonzero counts {nonzero_per_row}"


def test_top_books_membership_per_basin_returns_sorted_top_k():
    from artifacts_scripts.gap_softmax_enrich import top_books_membership_per_basin
    M_books = np.array([
        [0.7, 0.3, 0.0, 0.0],   # book 0 → basin 0 (high), basin 1 (some)
        [0.0, 0.6, 0.4, 0.0],   # book 1 → basin 1 (high), basin 2
        [0.5, 0.0, 0.0, 0.5],   # book 2 → basin 0, basin 3
    ], dtype=np.float32)
    book_global_idx = np.array([100, 200, 300], dtype=np.int64)
    all_titles = np.array(["Book A", "Book B", "Book C"], dtype=object)

    top = top_books_membership_per_basin(
        M_books=M_books,
        book_global_idx=book_global_idx,
        all_titles=all_titles,
        k_top_books=2,
    )
    assert len(top) == 4   # one per basin
    # Basin 0: book 0 (weight 0.7), book 2 (weight 0.5)
    assert len(top[0]) == 2
    assert top[0][0]["idx"] == 100
    assert top[0][0]["weight"] == pytest.approx(0.7, abs=1e-5)
    assert top[0][1]["idx"] == 300
    # Basin 3: only book 2 contributes (others have 0 weight)
    assert len(top[3]) == 1
    assert top[3][0]["idx"] == 300


def test_top_books_membership_per_basin_skips_zero_weight_entries():
    """Books with 0 softmax weight on a basin shouldn't appear in
    that basin's top list, even if k_top_books is large."""
    from artifacts_scripts.gap_softmax_enrich import top_books_membership_per_basin
    M_books = np.array([[0.7, 0.0]], dtype=np.float32)
    book_global_idx = np.array([100], dtype=np.int64)
    all_titles = np.array(["A"], dtype=object)
    top = top_books_membership_per_basin(M_books, book_global_idx, all_titles, k_top_books=10)
    assert len(top[0]) == 1   # only nonzero entry
    assert len(top[1]) == 0   # no books weight basin 1


def test_write_memberships_npz_roundtrip(tmp_path: Path):
    """write_memberships_npz emits a sparse softmax matrix + row/col
    labels; the npz round-trips correctly."""
    from artifacts_scripts.gap_softmax_enrich import write_memberships_npz
    import scipy.sparse as sp

    M_new = np.array([
        [0.7, 0.3, 0.0],
        [0.0, 0.6, 0.4],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    article_global_idx = np.array([100, 200, 300], dtype=np.int64)
    basin_rank = np.array([1, 2, 3], dtype=np.int64)

    out = tmp_path / "memberships.npz"
    write_memberships_npz(
        out_path=out,
        M_new=M_new,
        article_global_idx=article_global_idx,
        basin_rank=basin_rank,
    )
    assert out.exists()

    data = np.load(out, allow_pickle=False)
    assert "article_global_idx" in data.files
    assert "basin_rank" in data.files
    np.testing.assert_array_equal(data["article_global_idx"], article_global_idx)
    np.testing.assert_array_equal(data["basin_rank"], basin_rank)

    M_loaded = sp.csr_matrix(
        (data["M_data"], data["M_indices"], data["M_indptr"]),
        shape=tuple(data["M_shape"]),
    )
    np.testing.assert_allclose(M_loaded.toarray(), M_new, atol=1e-6)


def test_run_end_to_end_synthetic_phase3(tmp_path: Path, monkeypatch):
    """run() produces both the Phase 3 CSV (12 cols) and the sidecar
    .memberships.npz."""
    from artifacts_scripts import gap_softmax_enrich

    N_articles = 30
    N_books = 5
    N_total = N_articles + N_books
    d = 16

    rng = np.random.default_rng(42)
    emb = rng.normal(size=(N_total, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    types = ["article"] * N_articles + ["book"] * N_books
    titles = [f"Article {i}" for i in range(N_articles)] + [
        f"Book {i}" for i in range(N_books)
    ]
    ids = list(range(N_total))

    def fake_load_scope(scope, with_refs=False):
        return emb, ids, types, titles, [scope] * N_total, None

    monkeypatch.setattr(
        "artifacts_scripts.gap_softmax_enrich.load_scope", fake_load_scope,
        raising=False,
    )

    soft_em_csv = tmp_path / "soft_em.csv"
    write_soft_em_csv(soft_em_csv, [
        {"seed_article": 0, "n_articles": 5, "n_books": 0, "book_anchored": 0,
         "member_indices": [{"idx": i, "type": "article"} for i in range(5)],
         "member_weights": [{"idx": i, "type": "article", "weight": 0.9}
                            for i in range(5)]},
        {"seed_article": 5, "n_articles": 5, "n_books": 0, "book_anchored": 0,
         "member_indices": [{"idx": i, "type": "article"} for i in range(5, 10)],
         "member_weights": [{"idx": i, "type": "article", "weight": 0.9}
                            for i in range(5, 10)]},
        {"seed_article": 15, "n_articles": 5, "n_books": 1, "book_anchored": 1,
         "member_indices": [{"idx": i, "type": "article"} for i in range(15, 20)] +
                           [{"idx": N_articles, "type": "book"}],
         "member_weights": [{"idx": i, "type": "article", "weight": 0.9}
                            for i in range(15, 20)] +
                           [{"idx": N_articles, "type": "book", "weight": 1.0}]},
    ])

    out_csv = tmp_path / "phase3_enriched.csv"
    gap_softmax_enrich.run(
        scope="synthetic",
        input_path=soft_em_csv,
        output_path=out_csv,
        k_knn=3,
        k_sigma=5,
        k_marginal=20,
        perplexity=2.0,
        keywords_per_gap=3,
        bigram_iterations=1,
        k_csls=2,
        k_top_books=3,
    )
    assert out_csv.exists()
    npz_path = out_csv.with_suffix(".csv.memberships.npz")
    assert npz_path.exists(), f"expected sidecar at {npz_path}"

    rows = list(csv.DictReader(out_csv.open()))
    assert 1 <= len(rows) <= 3
    expected_cols = set(gap_softmax_enrich.PHASE3_FIELDNAMES)
    assert set(rows[0].keys()) == expected_cols

    data = np.load(npz_path, allow_pickle=False)
    assert "article_global_idx" in data.files
    assert "basin_rank" in data.files
    assert "M_shape" in data.files
