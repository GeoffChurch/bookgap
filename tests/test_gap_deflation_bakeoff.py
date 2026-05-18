import numpy as np


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
