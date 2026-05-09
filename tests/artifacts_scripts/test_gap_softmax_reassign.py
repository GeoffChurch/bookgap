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
