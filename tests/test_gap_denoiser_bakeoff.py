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
