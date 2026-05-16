import numpy as np
import pytest


def test_setup_projection_signature_accepts_basis():
    """setup() must accept a `projection` kwarg of shape (d, n_keep)."""
    import inspect
    from artifacts_scripts.gap_static_seed_diagnostic import setup
    sig = inspect.signature(setup)
    assert "projection" in sig.parameters
    assert sig.parameters["projection"].default is None


def test_build_graph_toggles_match_production_and_probe():
    import numpy as np
    from artifacts_scripts.gap_denoiser_bakeoff import build_scatter_context

    ids = [f"W{i}" for i in range(5)]
    refs = [["W1"], ["W2"], [], ["W0"], []]
    emb = np.random.RandomState(0).randn(5, 4).astype(np.float32)
    labels = ["article", "article", "book", "article", "book"]

    ctx_prod = build_scatter_context(emb, ids, refs, labels,
                                     self_loops=True, row_normalize=True)
    A_prod = ctx_prod._A.toarray()
    assert np.all(np.diag(A_prod) > 0)
    rowsums = A_prod.sum(axis=1)
    assert np.allclose(rowsums[rowsums > 0], 1.0, atol=1e-6)

    ctx_probe = build_scatter_context(emb, ids, refs, labels,
                                      self_loops=False, row_normalize=False)
    A_probe = ctx_probe._A.toarray()
    assert np.all(np.diag(A_probe) == 0)
    nz = A_probe[A_probe > 0]
    assert np.all(nz == 1.0)

    assert ctx_probe.S_L().shape == (4, 4)
    assert ctx_probe.S_T().shape == (4, 4)


def test_solve_arm_shapes_and_auto_alpha():
    import numpy as np
    from artifacts_scripts.gap_denoiser_bakeoff import (
        build_scatter_context, solve_arm,
    )
    rs = np.random.RandomState(1)
    ids = [f"W{i}" for i in range(40)]
    refs = [[f"W{(i + 1) % 40}"] for i in range(40)]
    emb = rs.randn(40, 8).astype(np.float32)
    labels = (["article"] * 30) + (["book"] * 10)
    ctx = build_scatter_context(emb, ids, refs, labels,
                                self_loops=False, row_normalize=False)

    for arm in ("none", "pca", "lip", "cclip_auto",
                "cclip_norm_0.5", "cclip_norm_0"):
        Q, alpha_used = solve_arm(arm, ctx, n_keep=5)
        if arm == "none":
            assert Q is None and np.isnan(alpha_used)
        else:
            assert Q.shape == (8, 5)
            assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-6)

    S_B = ctx.S_B_multi(list(range(ctx.num_subsets())))
    S_L = ctx.S_L()
    tr_B, tr_L = float(np.trace(S_B)), float(np.trace(S_L))
    expected = tr_B / (tr_B + tr_L)
    _, alpha_auto = solve_arm("cclip_auto", ctx, n_keep=5)
    assert abs(alpha_auto - expected) < 1e-9


def test_kball_seeds_change_under_projection(monkeypatch):
    import numpy as np
    from artifacts_scripts.gap_denoiser_bakeoff import kball_seeds_from_state

    rs = np.random.RandomState(2)
    N = 200
    emb = rs.randn(N, 16).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    art = np.arange(0, 170)
    bk = np.arange(170, 200)
    state = {
        "emb": emb,
        "article_global_idx": art,
        "book_global_idx": bk,
        "topic_id": "T_SYN",
    }
    seeds_raw = kball_seeds_from_state(state, radius_fraction=0.75,
                                       num_gaps=5, min_gap_size=0)

    Q = np.linalg.qr(rs.randn(16, 8))[0]
    emb_p = (emb - emb.mean(0)) @ Q
    emb_p /= np.linalg.norm(emb_p, axis=1, keepdims=True)
    state_p = dict(state, emb=emb_p.astype(np.float32))
    seeds_p = kball_seeds_from_state(state_p, radius_fraction=0.75,
                                     num_gaps=5, min_gap_size=0)

    set_raw = {tuple(sorted(s["article_indices"])) for s in seeds_raw}
    set_p = {tuple(sorted(s["article_indices"])) for s in seeds_p}
    assert set_raw != set_p
    for s in seeds_raw:
        assert set(s) >= {"gap_idx", "gap_size", "topic_id",
                          "article_indices"}


def test_scorecard_metrics_basic():
    import numpy as np
    from artifacts_scripts.gap_denoiser_bakeoff import scorecard_row

    # Synthetic survivor dicts using real compute_gap_stats key "effective_size"
    # plus the dual-landmark keys (effective_size_books, effective_size_chapters,
    # books_below_rate_pval) that are planned but not yet emitted by
    # compute_gap_stats.  scorecard_row uses .get() with safe defaults for
    # those three keys so it works both now (missing keys) and after dual-
    # landmark is wired in (present keys).  The test exercises the forward-
    # compatible path by supplying the keys explicitly.
    survivors = [
        {"effective_size": 120.0, "effective_size_books": 120.0,
         "effective_size_chapters": 5.0,
         "refined_member_indices": list(range(0, 60)),
         "books_below_rate_pval": 1e-5},
        {"effective_size": 40.0, "effective_size_books": 40.0,
         "effective_size_chapters": 80.0,
         "refined_member_indices": list(range(50, 90)),
         "books_below_rate_pval": 1e-2},
        # Third survivor: both book and chapter dims BELOW split → lands in bc bucket.
        {"effective_size": 15.0, "effective_size_books": 10.0,
         "effective_size_chapters": 10.0,
         "refined_member_indices": list(range(100, 130)),
         "books_below_rate_pval": 0.5},
    ]
    ref_top = [set(range(0, 60)), set(range(200, 260))]
    row = scorecard_row(
        phase="p1", topic="T_SYN", cell_or_arm="lip",
        self_loops=True, row_normalize=True, n_keep=8,
        alpha_used=float("nan"), survivors=survivors,
        ref_top_member_sets=ref_top, runtime_s=12.3,
        eff_low_high_split=50.0,
    )
    assert row["phase"] == "p1"
    assert row["sig_count_p1e3"] == 1
    assert 0.0 <= row["top25_jaccard_vs_ref"] <= 1.0
    fc = (row["fourcorner_bc"] + row["fourcorner_bC"]
          + row["fourcorner_Bc"] + row["fourcorner_BC"])
    assert fc == 3
    assert row["fourcorner_bc"] == 1
    assert "runtime_s" in row


def test_corpus_book_rate_and_landmark_idx():
    import numpy as np
    from artifacts_scripts.gap_denoiser_bakeoff import (
        _corpus_book_rate, _landmark_global_idx,
    )
    types = np.array(["book", "article", "book-chapter", "article", "book"])
    state = {"types": types}
    assert abs(_corpus_book_rate(state) - 2 / 5) < 1e-9
    np.testing.assert_array_equal(
        _landmark_global_idx(state, "book"), np.array([0, 4]))
    np.testing.assert_array_equal(
        _landmark_global_idx(state, "book-chapter"), np.array([2]))


def test_depletion_pvalue_basic():
    import numpy as np
    from artifacts_scripts.gap_denoiser_bakeoff import _depletion_pvalue
    # 100-work region, 0 books, base rate 5% -> strongly depleted (tiny p)
    p_dep = _depletion_pvalue(size=100, target_count=0, base_rate=0.05)
    assert p_dep < 1e-2
    # region at base rate -> p ~ 0.5-ish, certainly not significant
    p_mid = _depletion_pvalue(size=100, target_count=5, base_rate=0.05)
    assert p_mid > 1e-2
    # guard: size 0 -> 1.0 (non-significant)
    assert _depletion_pvalue(size=0, target_count=0, base_rate=0.05) == 1.0
