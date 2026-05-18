import numpy as np


def test_greedy_hard_byte_matches_oracle():
    """greedy_hard must reproduce _greedy_kball_gaps_inline exactly."""
    import numpy as np
    from artifacts_scripts.kball_gap_probe import _greedy_kball_gaps_inline
    from artifacts_scripts.gap_deflation_bakeoff import (
        build_kball_candidates, deflate,
    )
    rs = np.random.RandomState(3)
    art = rs.randn(300, 8).astype(np.float32)
    art /= np.linalg.norm(art, axis=1, keepdims=True)
    bk = rs.randn(20, 8).astype(np.float32)
    bk /= np.linalg.norm(bk, axis=1, keepdims=True)

    oracle = _greedy_kball_gaps_inline(
        book_emb=bk, article_emb=art, radius_fraction=0.75, k=1,
        num_gaps=50, min_gap_size=0)
    cands = build_kball_candidates(art, bk, radius_fraction=0.75, k=1)
    got = deflate(cands, "greedy_hard", art, num_gaps=50, min_gap_size=0)

    assert len(got) == len(oracle)
    for (gi_o, gs_o, ids_o), (gi_g, gs_g, ids_g) in zip(oracle, got):
        assert gi_o == gi_g and gs_o == gs_g
        np.testing.assert_array_equal(np.sort(ids_o), np.sort(ids_g))


def test_build_kball_candidates_shape():
    from artifacts_scripts.gap_deflation_bakeoff import build_kball_candidates
    rs = np.random.RandomState(0)
    art = rs.randn(200, 8).astype(np.float32)
    art /= np.linalg.norm(art, axis=1, keepdims=True)
    bk = rs.randn(15, 8).astype(np.float32)
    bk /= np.linalg.norm(bk, axis=1, keepdims=True)
    cands = build_kball_candidates(art, bk, radius_fraction=0.75, k=1)
    assert len(cands) == 200
    c = cands[0]
    assert c.idx == 0
    assert c.members.dtype == np.int64
    assert c.centroid.shape == (8,)
    assert all(0 <= m < 200 for m in c.members)
