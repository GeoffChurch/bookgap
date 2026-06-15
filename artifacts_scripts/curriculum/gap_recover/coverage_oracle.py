"""Phase A — validate code-gap detection against a CLEAN external oracle: test coverage.

The code-gap detector (``docs/code-gap-triangulation.md``) ranks classes by

    Δ = log(structural_users + 1) − log(operational_users + 1)

A *high* Δ means a class is composed into many other classes (structurally central)
but exercised by few functions/tests (operationally under-used) — the detector's
"structural-but-under-tested gap." Every prior validation used a *planted* or
*temporal* proxy for "under-tested." This script swaps in the real thing: it runs a
package's own test suite under ``coverage.py`` and asks the clean question directly —

    **Do high-Δ (detected-gap) classes have LOWER actual test coverage than low-Δ ones?**

If the detector's "under-exercised" judgment lines up with the coverage oracle, then
high Δ ⇒ low covered-line fraction (negative Spearman ρ; top-Δ third covered less than
bottom-Δ third). This is the first time the code-gap detector is checked against an
*independent* signal of test exhaustiveness rather than a graph-internal perturbation.

Pipeline
--------
1.  Read ``coverage json`` output (``--coverage-json``): per source file, the
    ``executed_lines`` / ``missing_lines`` / ``excluded_lines`` that coverage.py
    tracked. (Generated beforehand by running the suite under coverage; see the
    REPRODUCE block in this module's docstring footer / the printed provenance.)
2.  AST-walk each click source file to find every ``ClassDef``, its qualified name
    (``import_name.module.Outer.Inner``) and its line span. A class's *statement*
    lines = the coverage-known lines (executed ∪ missing, minus excluded) inside its
    span. Covered fraction = |executed ∩ class-stmt-lines| / |class-stmt-lines|.
3.  Run the extractor (``adapters.python.extract.extract_repo``) and compute Δ per
    class node, matching to the AST classes by qualified name.
4.  Correlate Δ vs covered-fraction (Spearman) and compare mean coverage of the
    top-Δ third vs the bottom-Δ third.

The size confound (and why the PARTIAL correlation is the primary test)
-----------------------------------------------------------------------
There is a strong, mechanical confound that runs *opposite* to the hypothesis and
must be controlled for. Δ is built from log-counts of users, so a small utility class
(``su=1, ou=0``) scores a *positive* Δ, while large central classes (many users of
both kinds) sit near Δ≈0. But **larger classes have lower coverage** (more lines, more
edge cases): empirically ρ(class-size, coverage) ≈ −0.43. And **high-Δ classes are
smaller**: ρ(class-size, Δ) ≈ −0.45. These two cancel in the *raw* ρ(Δ, coverage),
which lands at ≈0. The honest test of "does the gap signal predict under-coverage *at
fixed class size*" is therefore the **partial Spearman ρ(Δ, coverage | size)**, which
is reported as the primary statistic alongside the raw one. (A platform-unreachable
file like ``_winconsole.py`` — Windows-only, 0% covered on Linux regardless of Δ — is
not "under-tested" but "untestable on this OS"; ``--exclude-substr`` drops it as a
robustness check.)

This is deterministic and uses NO LLM. (Phase B — gap-seeded test *generation* — is a
separate harness, ``gap_test_generation.py``.)

REPRODUCE
---------
Run the target's own suite under coverage scoped to its source, in an isolated venv,
then point this script at the resulting ``coverage.json``::

    python -m venv /tmp/covenv
    /tmp/covenv/bin/pip install -e cache/code_gap/python/repos/click 'pytest>=8,<9' coverage
    cd cache/code_gap/python/repos/click
    COVERAGE_FILE=/tmp/.coverage /tmp/covenv/bin/coverage run \
        --source=cache/code_gap/python/repos/click/src/click -m pytest tests -q -p no:randomly
    COVERAGE_FILE=/tmp/.coverage /tmp/covenv/bin/coverage json -o /tmp/coverage.json

    conda run -p .conda python -m artifacts_scripts.curriculum.gap_recover.coverage_oracle \
        --repo cache/code_gap/python/repos/click --import-name click \
        --coverage-json /tmp/coverage.json \
        --out artifacts/code_gap/python/coverage_oracle_click.json

(pytest is pinned <9 because click 8.5.0.dev's parametrize usage trips a hard error in
pytest 9; this is a tooling pin, not a result-affecting choice.)
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from artifacts_scripts.curriculum.gap_recover.adapters.python.extract import (
    extract_repo,
    module_name_for,
)

_MODULE = "artifacts_scripts.curriculum.gap_recover.coverage_oracle"


# --------------------------------------------------------------------------- #
# per-class coverage from coverage.json + ast
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassCoverage:
    qualname: str            # "click.module.Outer.Inner"
    rel_path: str            # coverage.json file key, e.g. "src/click/core.py"
    line_lo: int             # first line of the ClassDef (the `class` line)
    line_hi: int             # last line of the class body
    n_stmt: int              # # coverage-known statement lines in the span
    n_covered: int           # # of those that were executed
    covered_fraction: float  # n_covered / n_stmt


def _class_qualnames(tree: ast.Module, module: str) -> list[tuple[str, int, int]]:
    """Every class in ``tree`` as ``(qualname, first_line, last_line)``.

    ``qualname`` mirrors the extractor: ``module.Outer.Inner`` for nested classes.
    ``last_line`` uses ``end_lineno`` (py3.8+); the span is the full class body,
    nested defs included. Decorators are excluded from the span (we start at the
    ``class`` keyword line, ``node.lineno``) so the line set is the class's own
    statements, matching how coverage attributes lines.
    """
    out: list[tuple[str, int, int]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qual = f"{prefix}{child.name}"
                full = f"{module}.{qual}"
                lo = child.lineno
                hi = getattr(child, "end_lineno", None) or lo
                out.append((full, lo, hi))
                visit(child, f"{qual}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # methods can themselves contain nested classes; keep the
                # enclosing-class prefix so a class defined inside a method is
                # still attributed under that class (rare, but correct).
                visit(child, prefix)

    visit(tree, "")
    return out


def per_class_coverage(
    repo: Path, import_name: str, cov: dict, pkg_dir: Path,
) -> dict[str, ClassCoverage]:
    """Compute covered-line fraction for every class in the package source.

    For each coverage-tracked source file we AST-walk for classes, then for each
    class intersect its line span with the file's coverage-known statement lines
    (executed ∪ missing, minus excluded). The covered fraction is the share of
    those statement lines that were executed by the suite.

    Returns ``{qualname: ClassCoverage}``. Classes whose span contains zero
    coverage-known statement lines (e.g. a pure ``...`` Protocol/stub body, or a
    class entirely of excluded lines) are dropped — coverage is undefined for them.
    """
    files = cov["files"]
    by_qual: dict[str, ClassCoverage] = {}

    for rel_path, fd in files.items():
        abs_path = (repo / rel_path).resolve()
        if not abs_path.exists():
            # coverage keys can be repo-relative or absolute; try as-is too.
            abs_path = Path(rel_path)
            if not abs_path.exists():
                continue
        try:
            module = module_name_for(abs_path, pkg_dir, import_name)
        except ValueError:
            # file not under the package dir (shouldn't happen for --source scope)
            continue

        try:
            tree = ast.parse(abs_path.read_text(encoding="utf-8"), filename=str(abs_path))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue

        executed = set(fd.get("executed_lines", []))
        missing = set(fd.get("missing_lines", []))
        excluded = set(fd.get("excluded_lines", []))
        # coverage's universe of *statement* lines for this file: lines it knows
        # are code (executed or missed), minus anything explicitly excluded.
        stmt_lines = (executed | missing) - excluded

        for qual, lo, hi in _class_qualnames(tree, module):
            span = range(lo, hi + 1)
            cls_stmt = stmt_lines.intersection(span)
            if not cls_stmt:
                continue
            cls_cov = executed.intersection(cls_stmt)
            n_stmt = len(cls_stmt)
            n_cov = len(cls_cov)
            by_qual[qual] = ClassCoverage(
                qualname=qual,
                rel_path=rel_path,
                line_lo=lo,
                line_hi=hi,
                n_stmt=n_stmt,
                n_covered=n_cov,
                covered_fraction=n_cov / n_stmt,
            )
    return by_qual


# --------------------------------------------------------------------------- #
# Δ per class from the extractor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassDelta:
    qualname: str
    structural_users: int
    operational_users: int
    delta: float


def per_class_delta(repo: Path, import_name: str) -> dict[str, ClassDelta]:
    """Δ = log(su+1) − log(ou+1) for every class node the extractor finds."""
    r = extract_repo(str(repo), import_name)
    su = r.structural_users()
    ou = r.operational_users()
    out: dict[str, ClassDelta] = {}
    for i, name in enumerate(r.names):
        if not r.is_B[i]:
            continue
        s, o = int(su[i]), int(ou[i])
        out[name] = ClassDelta(
            qualname=name,
            structural_users=s,
            operational_users=o,
            delta=math.log(s + 1) - math.log(o + 1),
        )
    return out


# --------------------------------------------------------------------------- #
# statistics (no scipy dependency: spearman + a permutation p-value)
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged — matches scipy.stats.rankdata."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def permutation_pvalue(
    x: np.ndarray, y: np.ndarray, rho: float, *, n_perm: int = 20000, seed: int = 0,
) -> float:
    """Two-sided permutation p-value for Spearman ρ (shuffle y vs x)."""
    rng = np.random.default_rng(seed)
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry0 = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry0 * ry0).sum()))
    if denom == 0.0:
        return float("nan")
    obs = abs(rho)
    count = 0
    n = len(y)
    for _ in range(n_perm):
        perm = ry0[rng.permutation(n)]
        r = abs(float((rx * perm).sum()) / denom)
        if r >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def partial_spearman(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> tuple[float, float]:
    """Partial Spearman ρ(x, y | z): rank-transform all three, then correlate the
    OLS residuals of rank(x) and rank(y) on rank(z). Returns ``(rho, two_sided_p)``
    with the standard partial-correlation t-test (df = n − 3).

    This is the primary statistic for the coverage oracle: it asks whether Δ
    predicts coverage *after removing the linear effect of class size on each*,
    dissolving the opposite-signed size confound (see module docstring).
    """
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    rx, ry, rz = _rankdata(x), _rankdata(y), _rankdata(z)

    def _resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    ex, ey = _resid(rx, rz), _resid(ry, rz)
    sx, sy = float(ex.std()), float(ey.std())
    if sx == 0 or sy == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(ex, ey)[0, 1])
    r = max(-1.0, min(1.0, r))
    denom = 1.0 - r * r
    if denom <= 1e-12:
        return r, 0.0
    t = r * math.sqrt((n - 3) / denom)
    # two-sided p via the survival function of |t| under df = n-3, using a
    # normal approx is too crude here; use the incomplete-beta via math when
    # scipy is unavailable. scipy is a project dep, so prefer it.
    try:  # pragma: no cover - thin numerical wrapper
        from scipy.stats import t as _tdist

        p = float(2.0 * _tdist.sf(abs(t), n - 3))
    except Exception:  # pragma: no cover
        # normal approximation fallback
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return r, p


def _mannwhitney_u_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U (a vs b) with a normal approximation + tie correction.

    Returns ``(U_a, two_sided_p)``. Used to test whether top-third and
    bottom-third coverage distributions differ (rank-based, robust to the
    bounded-[0,1] coverage fractions).
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    ranks = _rankdata(allv)
    Ra = ranks[:na].sum()
    Ua = Ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    # tie-corrected variance
    _, counts = np.unique(allv, return_counts=True)
    n = na + nb
    tie = float(np.sum(counts**3 - counts))
    var = (na * nb / 12.0) * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return float(Ua), float("nan")
    z = (Ua - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided
    return float(Ua), float(p)


# --------------------------------------------------------------------------- #
# main analysis
# --------------------------------------------------------------------------- #
@dataclass
class OracleResult:
    repo: str
    import_name: str
    n_classes_total: int          # classes the extractor found
    n_classes_with_coverage: int  # matched to a coverage line-span
    n_matched: int                # in BOTH Δ and coverage (the test set)
    spearman_rho: float           # raw ρ(Δ, covered_fraction); confounded by size
    spearman_p: float
    partial_rho_delta_cov: float  # PRIMARY: ρ(Δ, coverage | size); hypothesis < 0
    partial_p_delta_cov: float
    mean_cov_topdelta_third: float
    mean_cov_botdelta_third: float
    median_cov_topdelta_third: float
    median_cov_botdelta_third: float
    third_size: int
    mannwhitney_p: float          # top-third vs bottom-third coverage
    # also report the size-vs-coverage and size-vs-Δ confounds honestly
    spearman_rho_size_cov: float
    spearman_rho_size_delta: float


def analyse(
    repo: Path, import_name: str, cov: dict, *,
    perm_seed: int = 0, exclude_substr: tuple[str, ...] = (),
) -> tuple[OracleResult, list[dict]]:
    from artifacts_scripts.curriculum.gap_recover.adapters.python.extract import (
        _locate_pkg_dir,
    )

    pkg_dir = _locate_pkg_dir(repo, import_name)
    if pkg_dir is None:
        raise FileNotFoundError(f"no package dir under {repo} for {import_name!r}")

    cov_by_qual = per_class_coverage(repo, import_name, cov, pkg_dir)
    delta_by_qual = per_class_delta(repo, import_name)

    matched_names = sorted(set(cov_by_qual) & set(delta_by_qual))
    if exclude_substr:
        matched_names = [
            q for q in matched_names
            if not any(s in cov_by_qual[q].rel_path or s in q for s in exclude_substr)
        ]
    rows: list[dict] = []
    for q in matched_names:
        c = cov_by_qual[q]
        d = delta_by_qual[q]
        rows.append(
            {
                "qualname": q,
                "delta": d.delta,
                "structural_users": d.structural_users,
                "operational_users": d.operational_users,
                "covered_fraction": c.covered_fraction,
                "n_stmt": c.n_stmt,
                "n_covered": c.n_covered,
                "rel_path": c.rel_path,
            }
        )

    delta = np.array([r["delta"] for r in rows], dtype=float)
    covf = np.array([r["covered_fraction"] for r in rows], dtype=float)
    size = np.array([r["n_stmt"] for r in rows], dtype=float)
    n = len(rows)

    rho = spearman(delta, covf) if n >= 3 else float("nan")
    pval = (
        permutation_pvalue(delta, covf, rho, seed=perm_seed)
        if n >= 3 and not math.isnan(rho)
        else float("nan")
    )
    prho, pp = partial_spearman(delta, covf, size) if n >= 4 else (float("nan"), float("nan"))

    # top/bottom Δ thirds (use the extreme thirds; middle third dropped)
    third = n // 3
    order = np.argsort(-delta, kind="mergesort")  # descending Δ
    top_idx = order[:third]
    bot_idx = order[n - third :]
    cov_top = covf[top_idx]
    cov_bot = covf[bot_idx]
    _, mw_p = _mannwhitney_u_p(cov_top, cov_bot) if third >= 1 else (float("nan"), float("nan"))

    result = OracleResult(
        repo=str(repo),
        import_name=import_name,
        n_classes_total=len(delta_by_qual),
        n_classes_with_coverage=len(cov_by_qual),
        n_matched=n,
        spearman_rho=rho,
        spearman_p=pval,
        partial_rho_delta_cov=prho,
        partial_p_delta_cov=pp,
        mean_cov_topdelta_third=float(cov_top.mean()) if third else float("nan"),
        mean_cov_botdelta_third=float(cov_bot.mean()) if third else float("nan"),
        median_cov_topdelta_third=float(np.median(cov_top)) if third else float("nan"),
        median_cov_botdelta_third=float(np.median(cov_bot)) if third else float("nan"),
        third_size=int(third),
        mannwhitney_p=mw_p,
        spearman_rho_size_cov=spearman(size, covf) if n >= 3 else float("nan"),
        spearman_rho_size_delta=spearman(size, delta) if n >= 3 else float("nan"),
    )
    return result, rows


def _provenance() -> str:
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    return (
        f"generated by: {cmd}  "
        "(regenerate via the command above + the REPRODUCE block in the module "
        "docstring; do not hand-edit)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="path to the cloned package repo")
    ap.add_argument("--import-name", required=True, help="top-level import name (e.g. click)")
    ap.add_argument("--coverage-json", required=True, help="path to coverage json output")
    ap.add_argument("--out", default=None, help="optional path to write the full result JSON")
    ap.add_argument("--perm-seed", type=int, default=0)
    ap.add_argument(
        "--exclude-substr", action="append", default=[],
        help="drop classes whose source path or qualname contains this substring "
        "(repeatable; e.g. _winconsole for platform-unreachable code). "
        "Reported as a robustness check.",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    cov = json.loads(Path(args.coverage_json).read_text())
    result, rows = analyse(
        repo, args.import_name, cov,
        perm_seed=args.perm_seed, exclude_substr=tuple(args.exclude_substr),
    )
    if args.exclude_substr:
        print(f"[excluding classes matching: {args.exclude_substr}]")

    print(f"\n=== coverage-oracle: {args.import_name} ({repo.name}) ===")
    print(f"classes (extractor): {result.n_classes_total}")
    print(f"classes with coverage line-span: {result.n_classes_with_coverage}")
    print(f"matched (the test set): n = {result.n_matched}")
    print()
    print("HYPOTHESIS: high-Δ classes have LOWER coverage  ⇒  ρ(Δ, coverage) < 0")
    print(f"  [PRIMARY] partial ρ(Δ, coverage | size) = {result.partial_rho_delta_cov:+.3f}  "
          f"(p = {result.partial_p_delta_cov:.4f})")
    print(f"  [raw, confounded] Spearman ρ(Δ, coverage) = {result.spearman_rho:+.3f}  "
          f"(perm p = {result.spearman_p:.4f})")
    print(f"  top-Δ third  mean coverage = {result.mean_cov_topdelta_third:.3f}  "
          f"(median {result.median_cov_topdelta_third:.3f})")
    print(f"  bot-Δ third  mean coverage = {result.mean_cov_botdelta_third:.3f}  "
          f"(median {result.median_cov_botdelta_third:.3f})")
    print(f"  third size = {result.third_size};  top-vs-bot Mann-Whitney p = {result.mannwhitney_p:.4f}")
    print()
    print("confound checks (Spearman) — these run OPPOSITE to the hypothesis and")
    print("cancel the raw ρ, which is why the partial correlation is primary:")
    print(f"  ρ(class size, coverage) = {result.spearman_rho_size_cov:+.3f}  (bigger classes covered less)")
    print(f"  ρ(class size, Δ)        = {result.spearman_rho_size_delta:+.3f}  (high-Δ classes are smaller)")
    print()
    # top-10 highest-Δ classes and their coverage
    rows_by_delta = sorted(rows, key=lambda r: -r["delta"])
    print("top-10 highest-Δ classes (the detected gaps) and their actual coverage:")
    print(f"  {'qualname':48s} {'su':>3s} {'ou':>3s} {'Δ':>6s} {'cov':>5s} {'n_stmt':>6s}")
    for r in rows_by_delta[:10]:
        print(f"  {r['qualname']:48s} {r['structural_users']:3d} {r['operational_users']:3d} "
              f"{r['delta']:+6.2f} {r['covered_fraction']:5.2f} {r['n_stmt']:6d}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provenance": _provenance(),
            "result": asdict(result),
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
