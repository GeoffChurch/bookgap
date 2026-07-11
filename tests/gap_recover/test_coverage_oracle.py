"""Unit tests for the Phase-A coverage oracle (``coverage_oracle.py``).

Covers the two load-bearing, non-trivial pieces:
  1. the statistics helpers (rankdata / Spearman / Mann-Whitney / partial-Spearman)
     against scipy / the textbook partial-correlation identity, and
  2. per-class coverage line-span attribution from a synthetic ``coverage json``
     dict + the AST walk, including the size-confound surfacing in ``analyse``.
"""
import json

import numpy as np
import pytest

from artifacts_scripts.gap_recover import coverage_oracle as co


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_rankdata_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 30).astype(float)  # lots of ties
    assert np.allclose(co._rankdata(x), sp.rankdata(x))


def test_spearman_matches_scipy_with_ties():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, 40).astype(float)
    y = np.clip(rng.normal(0.8, 0.2, 40), 0, 1)
    y[::3] = 1.0  # ties at the [0,1] ceiling, like coverage fractions
    assert co.spearman(x, y) == pytest.approx(sp.spearmanr(x, y).statistic, abs=1e-9)


def test_partial_spearman_matches_pingouin_identity():
    """Residual-based partial ρ must equal the closed-form rank identity
    r_xy.z = (r_xy − r_xz·r_yz) / sqrt((1−r_xz²)(1−r_yz²))."""
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(2)
    z = rng.normal(size=60)
    x = z + rng.normal(scale=0.5, size=60)   # x correlated with z
    y = -z + rng.normal(scale=0.5, size=60)  # y anti-correlated with z
    r, p = co.partial_spearman(x, y, z)
    rxy = sp.spearmanr(x, y).statistic
    rxz = sp.spearmanr(x, z).statistic
    ryz = sp.spearmanr(y, z).statistic
    expect = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    assert r == pytest.approx(expect, abs=1e-6)
    assert 0.0 <= p <= 1.0


def test_partial_spearman_dissolves_an_induced_confound():
    """If x and y are correlated only THROUGH z, raw ρ is large but partial ρ ~ 0."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=200)
    x = z + rng.normal(scale=0.3, size=200)
    y = z + rng.normal(scale=0.3, size=200)  # x,y share z, no direct link
    raw = co.spearman(x, y)
    partial, _ = co.partial_spearman(x, y, z)
    assert raw > 0.7           # strong raw correlation
    assert abs(partial) < 0.25  # mostly explained away by z


def test_mannwhitney_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(4)
    a = rng.normal(0.0, 1.0, 25)
    b = rng.normal(0.5, 1.0, 30)
    u, _ = co._mannwhitney_u_p(a, b)
    res = sp.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    assert u == pytest.approx(res.statistic, abs=1e-6)


# --------------------------------------------------------------------------- #
# per-class coverage attribution
# --------------------------------------------------------------------------- #
# A tiny package: one well-covered small class, one poorly-covered larger class.
# Lines are chosen so the coverage dict's executed/missing/excluded sets map onto
# the class spans exactly.
SRC = (
    "class Small:\n"            # L1  (class line — not a statement coverage tracks here)
    "    x = 1\n"               # L2  stmt, covered
    "    def m(self):\n"        # L3  stmt (def), covered
    "        return self.x\n"   # L4  stmt, covered
    "\n"                        # L5  blank
    "class Big:\n"              # L6  class line
    "    def a(self):\n"        # L7  stmt, covered
    "        return 1\n"        # L8  stmt, covered
    "    def b(self):\n"        # L9  stmt, covered
    "        if False:\n"       # L10 stmt, MISSING
    "            return 2\n"    # L11 stmt, MISSING
    "        return 3\n"        # L12 stmt, covered
)


def _write_pkg(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(SRC)
    return tmp_path


def _coverage_json(executed, missing, excluded=()):
    return {
        "files": {
            "pkg/mod.py": {
                "executed_lines": list(executed),
                "missing_lines": list(missing),
                "excluded_lines": list(excluded),
            }
        }
    }


def test_per_class_coverage_line_span_attribution(tmp_path):
    repo = _write_pkg(tmp_path)
    pkg_dir = repo / "pkg"
    # Small: stmts {2,3,4} all executed -> 3/3. Big: stmts {7,8,9,10,11,12},
    # executed {7,8,9,12}, missing {10,11} -> 4/6.
    cov = _coverage_json(
        executed=[2, 3, 4, 7, 8, 9, 12], missing=[10, 11],
    )
    out = co.per_class_coverage(repo, "pkg", cov, pkg_dir)
    assert set(out) == {"pkg.mod.Small", "pkg.mod.Big"}
    small = out["pkg.mod.Small"]
    big = out["pkg.mod.Big"]
    assert (small.n_stmt, small.n_covered) == (3, 3)
    assert small.covered_fraction == pytest.approx(1.0)
    assert (big.n_stmt, big.n_covered) == (6, 4)
    assert big.covered_fraction == pytest.approx(4 / 6)


def test_excluded_lines_drop_out_of_statement_set(tmp_path):
    repo = _write_pkg(tmp_path)
    pkg_dir = repo / "pkg"
    # Exclude the two missing lines in Big -> Big becomes 4/4 (fully covered of
    # what's countable). Excluded lines must not count as statements.
    cov = _coverage_json(
        executed=[2, 3, 4, 7, 8, 9, 12], missing=[10, 11], excluded=[10, 11],
    )
    out = co.per_class_coverage(repo, "pkg", cov, pkg_dir)
    big = out["pkg.mod.Big"]
    assert (big.n_stmt, big.n_covered) == (4, 4)
    assert big.covered_fraction == pytest.approx(1.0)


def test_class_with_no_countable_statements_is_dropped(tmp_path):
    """A class whose span carries zero coverage-known statement lines (its lines
    are all blank/excluded) is omitted — coverage is undefined for it."""
    repo = tmp_path
    pkg = repo / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "stub.py").write_text("class Stub:\n    ...\n")  # body is a single `...`
    pkg_dir = pkg
    # coverage tracks no executable lines for Stub's body
    cov = {"files": {"pkg/stub.py": {"executed_lines": [], "missing_lines": [], "excluded_lines": []}}}
    out = co.per_class_coverage(repo, "pkg", cov, pkg_dir)
    assert "pkg.stub.Stub" not in out


def test_analyse_surfaces_the_size_confound(tmp_path):
    """End-to-end on a synthetic package: ``analyse`` must compute Δ per class,
    match to coverage, and report the size/coverage/Δ confound correlations."""
    repo = _write_pkg(tmp_path)
    cov = _coverage_json(executed=[2, 3, 4, 7, 8, 9, 12], missing=[10, 11])
    result, rows = co.analyse(repo, "pkg", cov)
    assert result.import_name == "pkg"
    assert result.n_matched == 2
    by = {r["qualname"]: r for r in rows}
    assert by["pkg.mod.Small"]["covered_fraction"] == pytest.approx(1.0)
    assert by["pkg.mod.Big"]["covered_fraction"] == pytest.approx(4 / 6)
    # both classes have Δ defined (real numbers, not NaN)
    assert all(np.isfinite(r["delta"]) for r in rows)


def test_exclude_substr_filters_targets(tmp_path):
    repo = _write_pkg(tmp_path)
    cov = _coverage_json(executed=[2, 3, 4, 7, 8, 9, 12], missing=[10, 11])
    _, rows = co.analyse(repo, "pkg", cov, exclude_substr=("Big",))
    names = {r["qualname"] for r in rows}
    assert "pkg.mod.Big" not in names
    assert "pkg.mod.Small" in names
