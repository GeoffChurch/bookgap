import csv
import json
from pathlib import Path

import numpy as np
import pytest

from artifacts_scripts.gap_softmax_compare import (
    argmax_shift_rate,
    best_match_jaccard,
    eff_rank_summary,
    effective_rank,
    load_basin_supports,
    save_eff_rank_scatter,
)


def write_csv(path: Path, supports: list[list[int]]) -> None:
    """Write a minimal basin CSV with the columns load_basin_supports
    needs."""
    rows = []
    for k, members in enumerate(supports):
        rows.append({
            "rank": k + 1,
            "seed_article": members[0] if members else -1,
            "n_articles": len(members),
            "n_books": 0,
            "book_anchored": 0,
            "member_indices_json": json.dumps(members),
            "member_weights_json": json.dumps([
                {"idx": m, "type": "article", "weight": 1.0} for m in members
            ]),
        })
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_load_basin_supports_returns_frozensets(tmp_path: Path):
    path = tmp_path / "basins.csv"
    write_csv(path, [[1, 2, 3], [4, 5], []])
    supports = load_basin_supports(path)
    assert supports[0] == frozenset({1, 2, 3})
    assert supports[1] == frozenset({4, 5})
    assert supports[2] == frozenset()


def test_best_match_jaccard_perfect_match():
    old = [frozenset({1, 2, 3}), frozenset({4, 5})]
    new = [frozenset({1, 2, 3}), frozenset({4, 5})]
    stats = best_match_jaccard(old, new)
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["count_ge_0.95"] == 2
    assert stats["count_lt_0.80"] == 0


def test_best_match_jaccard_split():
    """One old basin split in two: best match sees a half-Jaccard."""
    old = [frozenset({1, 2, 3, 4})]
    new = [frozenset({1, 2}), frozenset({3, 4})]
    stats = best_match_jaccard(old, new)
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["count_lt_0.80"] == 1


def test_best_match_jaccard_skips_empty():
    """Empty basins (n_articles=0) don't count in the distribution."""
    old = [frozenset({1, 2}), frozenset()]
    new = [frozenset({1, 2})]
    stats = best_match_jaccard(old, new)
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["count"] == 1


# ---------------------------------------------------------------------------
# Task 13: argmax-shift + effective-rank + scatter
# ---------------------------------------------------------------------------


def test_effective_rank_pure():
    p = np.eye(3)
    eff = effective_rank(p)
    np.testing.assert_allclose(eff, [1.0, 1.0, 1.0], atol=1e-6)


def test_effective_rank_uniform_two_options():
    """[0.5, 0.5, 0.0] -> entropy = log 2; effective rank = 2."""
    p = np.array([[0.5, 0.5, 0.0]])
    eff = effective_rank(p)
    np.testing.assert_allclose(eff, [2.0], atol=1e-3)


def test_argmax_shift_rate():
    """Articles whose argmax basin column changed."""
    M_old = np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.7, 0.2],
        [0.3, 0.4, 0.3],
    ])
    M_new = np.array([
        [0.6, 0.3, 0.1],
        [0.4, 0.2, 0.4],   # argmax shifts (was 1; now 0 or 2 by tie-break)
        [0.5, 0.3, 0.2],   # argmax shifts (was 1; now 0)
    ])
    rate = argmax_shift_rate(M_old, M_new)
    # row 0 same, rows 1+2 changed -> 2/3
    assert rate == pytest.approx(2 / 3)


def test_eff_rank_summary_returns_quantiles():
    eff_old = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    eff_new = np.array([1.0, 1.0, 1.0, 1.5, 2.0])
    s = eff_rank_summary(eff_old, eff_new)
    assert s["median_delta"] == pytest.approx(np.median(eff_old - eff_new))
    assert "iqr_delta" in s
    assert "frac_delta_ge_0.5" in s


def test_save_eff_rank_scatter_creates_file(tmp_path: Path):
    """Smoke test: scatter PNG is written and is non-empty."""
    out = tmp_path / "scatter.png"
    rng = np.random.default_rng(42)
    eff_old = rng.uniform(1.0, 4.0, size=50)
    eff_new = rng.uniform(1.0, 4.0, size=50)
    save_eff_rank_scatter(out, eff_old, eff_new, None, title_suffix=" test")
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG
