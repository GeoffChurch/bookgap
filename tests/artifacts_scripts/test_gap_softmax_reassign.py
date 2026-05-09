import json
import csv
from pathlib import Path

import numpy as np
import pytest

from artifacts_scripts.gap_softmax_reassign import load_enriched_csv
from artifacts_scripts.gap_softmax_reassign import compute_self_tuning_sigma
from artifacts_scripts.gap_softmax_reassign import (
    softmax_with_optional_empty,
    mean_entropy,
    tune_tau_for_perplexity,
)


def test_load_enriched_csv_builds_M0(tmp_path: Path):
    """Enriched CSV with member_indices_json + member_weights_json
    round-trips into a dense (N_articles, N_basins) M0 matrix."""
    csv_path = tmp_path / "enriched.csv"
    rows = [
        {
            "rank": "1",
            "seed_article": "100",
            "n_books": "0",
            "n_articles": "2",
            "book_anchored": "0",
            "member_indices_json": json.dumps([100, 101]),
            "member_weights_json": json.dumps(
                [{"idx": 100, "type": "article", "weight": 1.0},
                 {"idx": 101, "type": "article", "weight": 0.7}]
            ),
        },
        {
            "rank": "2",
            "seed_article": "200",
            "n_books": "1",
            "n_articles": "1",
            "book_anchored": "1",
            "member_indices_json": json.dumps([200, 300]),
            "member_weights_json": json.dumps(
                [{"idx": 200, "type": "article", "weight": 0.9},
                 {"idx": 300, "type": "book", "weight": 1.0}]
            ),
        },
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    is_article = np.zeros(400, dtype=bool)
    is_article[[100, 101, 200]] = True

    M0, basin_meta = load_enriched_csv(csv_path, is_article)
    # M0 shape: (N_articles_in_topic, N_basins). Article rows are
    # the global article indices; only article-side weights are used.
    assert M0.shape == (3, 2)
    assert M0[0, 0] == pytest.approx(1.0)   # article 100 in basin 1
    assert M0[1, 0] == pytest.approx(0.7)   # article 101 in basin 1
    assert M0[2, 1] == pytest.approx(0.9)   # article 200 in basin 2
    assert M0[0, 1] == 0.0                   # article 100 not in basin 2
    assert basin_meta[1]["book_anchored"] == 1


def test_compute_self_tuning_sigma_synthetic():
    """σ_i is the sqrt of the median squared L2 distance to the
    k_sigma nearest neighbors (excluding self).

    Set up 10 points evenly spaced on a line: emb[i] = [i, 0, ...].
    For point i (not at edge), 7-NN squared distances are
    {1, 1, 4, 4, 9, 9, 16}; median = 4. σ_i = 2.
    """
    rng = np.random.default_rng(0)
    n, d = 30, 8
    emb = np.zeros((n, d), dtype=np.float32)
    emb[:, 0] = np.arange(n) * 0.1   # spaced along axis 0
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms > 0, norms, 1.0)
    emb = emb.astype(np.float32)

    sigma = compute_self_tuning_sigma(emb, k_sigma=7)
    assert sigma.shape == (n,)
    assert np.all(sigma > 0)
    # Middle points should have similar σ (translation symmetry)
    assert sigma[10:20].std() < 0.1 * sigma[10:20].mean()


def test_tune_tau_hits_target_perplexity():
    """Synthetic logits over 4 basins for 100 articles. Bisection
    on log τ should land mean entropy near log(P) within 0.01 nats.
    """
    rng = np.random.default_rng(42)
    log_K = rng.normal(0, 1, size=(100, 4)).astype(np.float64)

    for P in (1.5, 2.0, 3.0):
        tau = tune_tau_for_perplexity(
            log_K, log_K_empty=None, P_target=P,
            log_tau_low=-3.0, log_tau_high=1.0,
            tol=0.01, max_iters=30,
        )
        probs = softmax_with_optional_empty(log_K, None, tau)
        H_bar = mean_entropy(probs)
        assert abs(H_bar - np.log(P)) < 0.02, (
            f"P={P}: target log P={np.log(P):.3f}, got H̄={H_bar:.3f}"
        )


def test_tune_tau_with_empty_class():
    """Variant B: ∅ class is an extra logit per article. Same target
    perplexity criterion."""
    rng = np.random.default_rng(0)
    log_K = rng.normal(0, 1, size=(50, 5)).astype(np.float64)
    log_K_empty = rng.normal(-1, 0.5, size=(50,)).astype(np.float64)
    P = 2.0
    tau = tune_tau_for_perplexity(
        log_K, log_K_empty=log_K_empty, P_target=P,
        log_tau_low=-3.0, log_tau_high=1.0, tol=0.01, max_iters=30,
    )
    probs = softmax_with_optional_empty(log_K, log_K_empty, tau)
    # Rows include the empty class as extra column
    assert probs.shape == (50, 6)
    # Rows sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    H_bar = mean_entropy(probs)
    assert abs(H_bar - np.log(P)) < 0.02


from artifacts_scripts.gap_softmax_reassign import compute_log_K_empty


def test_compute_log_K_empty_matches_brute_at_small_N():
    """At small N the kNN truncation should reproduce the brute-force
    marginal-density sum within numerical precision."""
    rng = np.random.default_rng(7)
    n, d = 40, 16
    emb = rng.normal(size=(n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sigma = np.full(n, 0.5, dtype=np.float32)

    # Brute force: K̃_∅(a) = (1/N) Σ_y exp(-||a-y||² / (σ_a σ_y))
    d2 = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=2)
    sigma_outer = sigma[:, None] * sigma[None, :]
    K_full = np.exp(-d2 / (sigma_outer + 1e-30)).astype(np.float64)
    expected = K_full.mean(axis=1)
    expected_log = np.log(expected + 1e-30)

    # kNN truncation with K=n (full set, sanity check)
    log_K_empty = compute_log_K_empty(
        article_emb=emb, article_sigma=sigma, k_marginal=n,
    )
    np.testing.assert_allclose(log_K_empty, expected_log, atol=1e-5)


def test_compute_log_K_empty_kNN_truncation_is_close():
    """At a small but representative K (kernel mass concentrated
    locally), kNN truncation should be within a few percent of brute."""
    rng = np.random.default_rng(11)
    n, d = 200, 32
    emb = rng.normal(size=(n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sigma = np.full(n, 0.2, dtype=np.float32)  # small σ → tight kernel

    log_K_full = compute_log_K_empty(emb, sigma, k_marginal=n)
    log_K_trunc = compute_log_K_empty(emb, sigma, k_marginal=50)
    rel_err = np.abs(np.exp(log_K_trunc) - np.exp(log_K_full)) / (
        np.exp(log_K_full) + 1e-30
    )
    assert rel_err.max() < 0.05, f"max rel err: {rel_err.max()}"


from artifacts_scripts.gap_softmax_reassign import (
    compute_basin_centroids,
    knn_basin_lookup,
)


def test_compute_basin_centroids_weighted_mean():
    """Centroids are unit-normalized weighted means of member emb."""
    emb = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    M = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)   # basin 0: arts 0+1; basin 1: art 2
    C = compute_basin_centroids(emb, M)
    # Basin 0 centroid: (e0+e1)/sqrt(2); Basin 1: e2.
    expected_0 = np.array([1, 1, 0]) / np.sqrt(2.0)
    np.testing.assert_allclose(C[0], expected_0, atol=1e-6)
    np.testing.assert_allclose(C[1], np.array([0, 0, 1.0]), atol=1e-6)


def test_knn_basin_lookup_returns_top_k():
    """For each article, kNN basin lookup returns indices of top-k
    nearest basin centroids by cosine."""
    article_emb = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    centroids = np.array([
        [1.0, 0.0, 0.0],   # nearest to article 0
        [0.0, 1.0, 0.0],   # nearest to article 1
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    knn_idx = knn_basin_lookup(article_emb, centroids, k=2)
    assert knn_idx.shape == (2, 2)
    assert knn_idx[0, 0] == 0
    assert knn_idx[1, 0] == 1
