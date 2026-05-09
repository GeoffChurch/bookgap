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


def test_crisp_filter_returns_per_basin_member_lists():
    """For each basin, list of (article_local_idx, weight) pairs above
    tau_supp. Drop basins below min_final_size."""
    from artifacts_scripts.gap_softmax_enrich import crisp_filter

    M_new = np.array([
        [0.7, 0.3, 0.0, 0.0],   # article 0 → basin 0 (above 0.5)
        [0.6, 0.4, 0.0, 0.0],   # article 1 → basin 0
        [0.0, 0.8, 0.2, 0.0],   # article 2 → basin 1
        [0.0, 0.4, 0.6, 0.0],   # article 3 → basin 2
        [0.0, 0.4, 0.6, 0.0],   # article 4 → basin 2
        [0.0, 0.0, 0.0, 0.55],  # article 5 → basin 3 (single member; < min_size)
    ], dtype=np.float32)

    surviving = crisp_filter(M_new, tau_supp=0.5, min_final_size=2)
    # surviving is dict[basin_idx, list[(article_local_idx, weight)]]
    assert set(surviving) == {0, 2}        # basin 1 has only 1 member; basin 3 has 1
    assert sorted([i for i, _ in surviving[0]]) == [0, 1]
    assert sorted([i for i, _ in surviving[2]]) == [3, 4]


def test_inherited_coverage_per_article():
    """For each article: max softmax weight over book-anchored basins."""
    from artifacts_scripts.gap_softmax_enrich import inherited_coverage_per_article

    M_new = np.array([
        [0.6, 0.3, 0.1, 0.0],   # basins 0,2 are book-anchored; max(0.6, 0.1) = 0.6
        [0.0, 0.5, 0.5, 0.0],   # max(0.0, 0.5) = 0.5
        [0.0, 1.0, 0.0, 0.0],   # max(0.0, 0.0) = 0.0 (basin 1 not anchored)
    ], dtype=np.float32)
    book_anchored_mask = np.array([True, False, True, False])

    cov = inherited_coverage_per_article(M_new, book_anchored_mask)
    np.testing.assert_allclose(cov, [0.6, 0.5, 0.0], atol=1e-6)


def test_inherited_coverage_per_article_no_book_anchored():
    """If no basins are book-anchored, all-zeros."""
    from artifacts_scripts.gap_softmax_enrich import inherited_coverage_per_article

    M_new = np.eye(3, dtype=np.float32)
    book_anchored_mask = np.array([False, False, False])
    cov = inherited_coverage_per_article(M_new, book_anchored_mask)
    np.testing.assert_allclose(cov, [0.0, 0.0, 0.0], atol=1e-6)


def test_per_basin_readouts():
    """Per non-anchored basin: median_post over its tau_supp members,
    effective_uncovered_articles = Σ_a M_new[a, G] · (1 − cov[a])."""
    from artifacts_scripts.gap_softmax_enrich import (
        inherited_coverage_per_article,
        per_basin_readouts,
    )

    M_new = np.array([
        [0.6, 0.3, 0.1],
        [0.0, 0.7, 0.3],
        [0.0, 0.6, 0.4],
        [0.4, 0.0, 0.6],
    ], dtype=np.float32)
    book_anchored_mask = np.array([True, False, False])
    surviving = {1: [(1, 0.7), (2, 0.6)]}     # only basin 1 survived

    cov = inherited_coverage_per_article(M_new, book_anchored_mask)
    # cov = [0.6, 0.0, 0.0, 0.4]
    readouts = per_basin_readouts(M_new, surviving, cov, book_anchored_mask)
    # basin 1: members [1, 2]; cov[1]=0, cov[2]=0; median_post=0.0
    # effective_uncovered = Σ over all a of M[a,1]*(1-cov[a])
    #   = 0.3*(1-0.6) + 0.7*(1-0.0) + 0.6*(1-0.0) + 0.0*(1-0.4)
    #   = 0.12 + 0.7 + 0.6 + 0.0 = 1.42
    assert 1 in readouts
    assert readouts[1]["inherited_coverage_median_post"] == pytest.approx(0.0)
    assert readouts[1]["effective_uncovered_articles"] == pytest.approx(1.42, abs=1e-5)


def test_tag_basin_book_signals_passes_through_helpers():
    """tag_basin_book_signals delegates to the existing
    compute_top_books / compute_gap_keywords / compute_book_existence_signal
    helpers and returns the per-basin signal dict."""
    from artifacts_scripts.gap_softmax_enrich import tag_basin_book_signals

    rng = np.random.default_rng(7)
    N = 20
    d = 8
    emb = rng.normal(size=(N, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    types_arr = np.array(["article"] * 15 + ["book"] * 5)
    titles = [f"Article {i}" for i in range(15)] + [f"Book {i}" for i in range(5)]
    abstracts = [""] * N

    # Two basins with article-only members
    surviving = {
        0: [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)],
        1: [(5, 0.9), (6, 0.8), (7, 0.7), (8, 0.6), (9, 0.5)],
    }
    article_global_idx = np.where(types_arr == "article")[0]

    signals = tag_basin_book_signals(
        surviving=surviving,
        emb=emb,
        types_arr=types_arr,
        titles=titles,
        abstracts=abstracts,
        article_global_idx=article_global_idx,
    )
    assert set(signals) == {0, 1}
    for b in (0, 1):
        sig = signals[b]
        assert "top_book_id" in sig
        assert "top_book_title" in sig
        assert "top_book_votes" in sig
        assert "vote_concentration" in sig
        assert "top_keywords" in sig
        assert "title_keyword_overlap" in sig
        assert "book_exists_flag" in sig
        assert "templated_leak_flag" in sig


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


def _make_book_members_per_basin(basin_meta, book_global_idx_set):
    """Helper for the end-to-end test: every basin has zero book members."""
    return {b: [] for b in range(len(basin_meta))}


def test_run_end_to_end_synthetic(tmp_path: Path, monkeypatch):
    """run() composes load → softmax → crisp → tag → readouts → write
    against a synthetic in-memory scope. Verifies the output CSV is
    well-formed and has the right number of rows."""
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
    abstracts = [""] * N_total
    ids = list(range(N_total))

    def fake_load_scope(scope, with_refs=False):
        return emb, ids, types, titles, [scope] * N_total, None

    monkeypatch.setattr(
        "artifacts_scripts.gap_softmax_enrich.load_scope", fake_load_scope,
        raising=False,
    )

    # Synthetic soft-EM CSV: 4 basins (one book-anchored)
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
        {"seed_article": 10, "n_articles": 5, "n_books": 0, "book_anchored": 0,
         "member_indices": [{"idx": i, "type": "article"} for i in range(10, 15)],
         "member_weights": [{"idx": i, "type": "article", "weight": 0.9}
                            for i in range(10, 15)]},
        {"seed_article": 15, "n_articles": 5, "n_books": 1, "book_anchored": 1,
         "member_indices": [{"idx": i, "type": "article"} for i in range(15, 20)] +
                           [{"idx": N_articles, "type": "book"}],
         "member_weights": [{"idx": i, "type": "article", "weight": 0.9}
                            for i in range(15, 20)] +
                           [{"idx": N_articles, "type": "book", "weight": 1.0}]},
    ])

    out_csv = tmp_path / "phase2_enriched.csv"
    gap_softmax_enrich.run(
        scope="synthetic",
        input_path=soft_em_csv,
        output_path=out_csv,
        k_knn=3,
        k_sigma=5,
        k_marginal=20,
        perplexity=2.0,
        tau_supp=0.3,           # low, so synthetic basins survive
        min_final_size=2,
        keywords_per_gap=3,
        bigram_iterations=1,
        vote_concentration_threshold=0.25,
        title_overlap_threshold=0.2,
        templated_concentration_threshold=0.5,
    )
    assert out_csv.exists()
    rows = list(csv.DictReader(out_csv.open()))
    # Most synthetic basins should survive at tau_supp=0.3 / min_size=2
    assert 1 <= len(rows) <= 4
    # All emitted rows have the full schema
    expected_cols = set(gap_softmax_enrich.PHASE2_FIELDNAMES)
    assert set(rows[0].keys()) == expected_cols
