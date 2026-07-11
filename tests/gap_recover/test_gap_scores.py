import numpy as np

from artifacts_scripts.gap_recover.core.gap_scores import (
    method_count,
    raw_delta,
    size_controlled_delta,
)


def test_method_count_counts_direct_children():
    names = ["m.A", "m.A.f", "m.A.g", "m.B", "m.h"]
    is_F = np.array([False, True, True, False, True])
    mc = method_count(names, is_F)
    idx = {n: i for i, n in enumerate(names)}
    assert mc[idx["m.A"]] == 2     # f, g are direct F children of A
    assert mc[idx["m.B"]] == 0     # B has no method children
    assert mc[idx["m.A.f"]] == 0


def test_raw_delta_sign():
    # structural-but-under-exercised (high su, low ou) -> positive Δ
    d = raw_delta(np.array([10, 0]), np.array([0, 10]))
    assert d[0] > 0 and d[1] < 0


def test_size_controlled_delta_contract():
    names = ["m.A", "m.A.f", "m.B", "m.B.g", "m.B.h", "m.C", "m.x"]
    is_F = np.array([False, True, False, True, True, False, True])
    is_B = np.array([True, False, True, False, False, True, False])
    su = np.array([3, 0, 5, 0, 0, 1, 0])
    ou = np.array([1, 0, 2, 0, 0, 0, 0])
    scd = size_controlled_delta(names, is_F, is_B, su, ou)
    assert np.isnan(scd[is_F]).all()                 # non-B nodes -> NaN
    assert np.isfinite(scd[is_B]).all()              # B nodes -> finite residuals
    assert abs(float(np.nanmean(scd[is_B]))) < 1e-9  # OLS-with-intercept residuals are mean-zero
