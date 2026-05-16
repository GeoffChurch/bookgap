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
