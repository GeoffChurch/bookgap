"""Multi-package driver for coverage-oracle + size-control validation.

Automates the per-package pipeline (venv → install → coverage run → coverage json →
coverage_oracle → gap_size_control) for a configurable list of ``(repo_dir, import_name)``
pairs, skipping packages whose suite fails at any step (logging the reason), and aggregating
the results into a single table.

Usage (run from the bookgap project root)::

    uv run python -m artifacts_scripts.gap_recover.coverage_oracle_multi \\
        [--packages "attrs:attr,jinja2:jinja2,rich:rich,flask:flask,typer:typer,black:black"] \\
        [--repos-root cache/code_gap/python/repos] \\
        [--out artifacts/code_gap/python/coverage_oracle_multi.json]

The click result (already computed) is automatically added as the reference row from
``artifacts/code_gap/python/coverage_oracle_click.json`` when it exists.

Output
------
- Prints an aggregate table (per package: n_classes, raw ρ(Δ,cov), partial ρ, graph
  size-controlled ρ, ρ(method_count, LOC)) to stdout.
- Writes the full table + per-package rows to ``--out`` JSON.

REPRODUCE
---------
    uv run python -m artifacts_scripts.gap_recover.coverage_oracle_multi

(regenerate rather than edit by hand; this file is the authoritative producer)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import venv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_MODULE = "artifacts_scripts.gap_recover.coverage_oracle_multi"

# Default packages to test: (repo_subdir, import_name, [extra_excludes...])
DEFAULT_PACKAGES = [
    ("attrs",  "attr",   []),
    ("jinja2", "jinja2", []),
    ("rich",   "rich",   []),
    ("flask",  "flask",  []),
    ("typer",  "typer",  []),
    ("black",  "black",  []),
]


@dataclass
class PackageResult:
    repo_name: str
    import_name: str
    status: str               # "ok" | "skipped:<reason>"
    n_classes: Optional[int]
    raw_rho: Optional[float]  # ρ(Δ, coverage) — raw, size-confounded
    partial_rho: Optional[float]  # ρ(Δ, coverage | n_stmt) — partial (primary)
    graph_scd_rho: Optional[float]  # ρ(Δ_resid|method_count, coverage) — graph score
    method_loc_rho: Optional[float]  # ρ(method_count, n_stmt) — size-proxy quality
    # full oracle result dict (None if skipped)
    oracle_result: Optional[dict]
    # size-control raw output text
    size_control_output: Optional[str]
    skip_reason: Optional[str]


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 600,
         env: dict | None = None) -> tuple[int, str, str]:
    """Run a subprocess; return (returncode, stdout, stderr)."""
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd,
        timeout=timeout, env=full_env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _locate_src_dir(repo: Path, import_name: str) -> Optional[Path]:
    """Find the package source directory: src/<import> or <repo>/<import>."""
    for candidate in [repo / "src" / import_name, repo / import_name]:
        if candidate.is_dir():
            return candidate
    return None


def _locate_test_dir(repo: Path) -> Optional[Path]:
    """Find the test directory: tests/ or test/."""
    for name in ("tests", "test"):
        d = repo / name
        if d.is_dir():
            return d
    return None


def _create_venv(venv_path: str, repo: Path, import_name: str) -> tuple[bool, str]:
    """Create a fresh venv and install the package + pytest + coverage.

    Returns (success, error_message).
    """
    # Create venv
    venv.create(venv_path, with_pip=True, clear=True)
    pip = os.path.join(venv_path, "bin", "pip")

    def pip_install(args: list[str]) -> tuple[int, str, str]:
        return _run([pip, "install", "--quiet"] + args)

    # Upgrade pip to avoid ancient-pip issues
    rc, _, err = pip_install(["--upgrade", "pip"])
    if rc != 0:
        return False, f"pip upgrade failed: {err[:300]}"

    # Try install variations in order of preference
    install_attempts = [
        # First try with all common extras
        ["-e", f"{repo}[tests,test,dev]", "pytest>=8,<9", "coverage"],
        # If that fails, try individual extras
        ["-e", f"{repo}[tests]", "pytest>=8,<9", "coverage"],
        ["-e", f"{repo}[test]",  "pytest>=8,<9", "coverage"],
        ["-e", f"{repo}[dev]",   "pytest>=8,<9", "coverage"],
        # Fall back to bare install
        ["-e", str(repo), "pytest>=8,<9", "coverage"],
    ]

    last_err = ""
    for attempt in install_attempts:
        rc, out, err = pip_install(attempt)
        if rc == 0:
            return True, ""
        last_err = err

    return False, f"all install attempts failed; last error: {last_err[:500]}"


def _run_coverage(
    venv_path: str, repo: Path, import_name: str, src_dir: Path, test_dir: Path,
    cov_file: str, cov_json: str,
) -> tuple[bool, str]:
    """Run pytest under coverage and export json.

    Returns (success, error_message).
    """
    cov_bin = os.path.join(venv_path, "bin", "coverage")
    env = {"COVERAGE_FILE": cov_file}

    rc, out, err = _run(
        [
            cov_bin, "run",
            f"--source={src_dir}",
            "-m", "pytest", str(test_dir),
            "-q", "-p", "no:randomly",
            "--tb=no",
            "-x",    # stop on first failure to save time
            "--ignore-glob=**/conftest*.py",
        ],
        cwd=str(repo), timeout=600, env=env,
    )
    # rc != 0 is OK if it's a test failure; coverage is still valid.
    # Only fail hard if coverage produced no data.
    rc2, out2, err2 = _run(
        [cov_bin, "json", "-o", cov_json],
        cwd=str(repo), timeout=60, env=env,
    )
    if rc2 != 0:
        return False, f"coverage json failed (rc={rc2}): {err2[:400]}"
    if not Path(cov_json).exists():
        return False, "coverage json output not produced"
    return True, ""


def _run_coverage_permissive(
    venv_path: str, repo: Path, import_name: str, src_dir: Path, test_dir: Path,
    cov_file: str, cov_json: str,
) -> tuple[bool, str]:
    """Run pytest under coverage WITHOUT -x (permissive: partial suite OK).

    Falls back to this when the strict run fails.
    """
    cov_bin = os.path.join(venv_path, "bin", "coverage")
    env = {"COVERAGE_FILE": cov_file}

    _run(
        [
            cov_bin, "run",
            f"--source={src_dir}",
            "-m", "pytest", str(test_dir),
            "-q", "-p", "no:randomly",
            "--tb=no",
        ],
        cwd=str(repo), timeout=600, env=env,
    )
    rc2, out2, err2 = _run(
        [cov_bin, "json", "-o", cov_json],
        cwd=str(repo), timeout=60, env=env,
    )
    if rc2 != 0:
        return False, f"coverage json failed (rc={rc2}): {err2[:400]}"
    if not Path(cov_json).exists():
        return False, "coverage json output not produced"
    return True, ""


def _run_oracle(
    repo: Path, import_name: str, cov_json: str, oracle_out: str,
    exclude_substrs: list[str],
) -> tuple[bool, str, Optional[dict]]:
    """Call coverage_oracle as a subprocess via uv run.

    Returns (success, error_message, result_dict_or_None).
    """
    cmd = [
        "uv", "run", "python", "-m",
        "artifacts_scripts.gap_recover.coverage_oracle",
        "--repo", str(repo),
        "--import-name", import_name,
        "--coverage-json", cov_json,
        "--out", oracle_out,
    ]
    for s in exclude_substrs:
        cmd += ["--exclude-substr", s]

    bookgap_root = Path(__file__).resolve().parents[2]
    rc, out, err = _run(cmd, cwd=str(bookgap_root), timeout=300)
    if rc != 0:
        return False, f"coverage_oracle failed (rc={rc}): {err[:500]}", None

    if not Path(oracle_out).exists():
        return False, "oracle --out JSON not written", None

    try:
        data = json.loads(Path(oracle_out).read_text())
    except Exception as e:
        return False, f"could not parse oracle JSON: {e}", None

    return True, "", data


def _run_size_control(
    repo: Path, import_name: str, oracle_out: str,
) -> tuple[bool, str, str]:
    """Call gap_size_control as a subprocess via uv run.

    Returns (success, error_message, stdout_text).
    """
    cmd = [
        "uv", "run", "python", "-m",
        "artifacts_scripts.gap_recover.gap_size_control",
        "--repo", str(repo),
        "--import-name", import_name,
        "--coverage-oracle-json", oracle_out,
    ]
    bookgap_root = Path(__file__).resolve().parents[2]
    rc, out, err = _run(cmd, cwd=str(bookgap_root), timeout=300)
    if rc != 0:
        return False, f"gap_size_control failed (rc={rc}): {err[:500]}", ""
    return True, "", out


def _parse_size_control_output(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract graph_scd_rho and method_loc_rho from gap_size_control stdout."""
    graph_rho = None
    method_loc_rho = None
    for line in text.splitlines():
        if "Δ_resid | method_count" in line and "ρ =" in line:
            try:
                part = line.split("ρ =")[1].strip().split()[0]
                graph_rho = float(part)
            except (IndexError, ValueError):
                pass
        if "ρ(method_count, LOC" in line and "ρ =" in line:
            try:
                part = line.split("ρ =")[1].strip().split()[0]
                method_loc_rho = float(part)
            except (IndexError, ValueError):
                pass
    return graph_rho, method_loc_rho


def process_package(
    repo_name: str,
    import_name: str,
    exclude_substrs: list[str],
    repos_root: Path,
    out_dir: Path,
) -> PackageResult:
    """Full pipeline for one package. Returns a PackageResult (may be skipped)."""
    repo = repos_root / repo_name
    if not repo.is_dir():
        return PackageResult(
            repo_name=repo_name, import_name=import_name,
            status=f"skipped:repo-not-found", n_classes=None,
            raw_rho=None, partial_rho=None, graph_scd_rho=None, method_loc_rho=None,
            oracle_result=None, size_control_output=None,
            skip_reason=f"repo directory not found: {repo}",
        )

    src_dir = _locate_src_dir(repo, import_name)
    if src_dir is None:
        return PackageResult(
            repo_name=repo_name, import_name=import_name,
            status="skipped:no-src-dir", n_classes=None,
            raw_rho=None, partial_rho=None, graph_scd_rho=None, method_loc_rho=None,
            oracle_result=None, size_control_output=None,
            skip_reason=f"could not find src dir for {import_name} under {repo}",
        )

    test_dir = _locate_test_dir(repo)
    if test_dir is None:
        return PackageResult(
            repo_name=repo_name, import_name=import_name,
            status="skipped:no-test-dir", n_classes=None,
            raw_rho=None, partial_rho=None, graph_scd_rho=None, method_loc_rho=None,
            oracle_result=None, size_control_output=None,
            skip_reason=f"could not find test directory under {repo}",
        )

    with tempfile.TemporaryDirectory(prefix=f"covenv_{repo_name}_") as tmpdir:
        venv_path = os.path.join(tmpdir, "venv")
        cov_file  = os.path.join(tmpdir, f".cov_{repo_name}")
        cov_json  = os.path.join(tmpdir, f"cov_{repo_name}.json")

        print(f"\n[{repo_name}] creating venv at {venv_path} ...", flush=True)
        ok, err = _create_venv(venv_path, repo, import_name)
        if not ok:
            return PackageResult(
                repo_name=repo_name, import_name=import_name,
                status="skipped:install-failed", n_classes=None,
                raw_rho=None, partial_rho=None, graph_scd_rho=None, method_loc_rho=None,
                oracle_result=None, size_control_output=None,
                skip_reason=err,
            )

        print(f"[{repo_name}] running coverage (strict -x) ...", flush=True)
        ok, err = _run_coverage(
            venv_path, repo, import_name, src_dir, test_dir, cov_file, cov_json,
        )
        if not ok:
            print(f"[{repo_name}] strict run failed ({err[:120]}); retrying without -x ...", flush=True)
            ok, err = _run_coverage_permissive(
                venv_path, repo, import_name, src_dir, test_dir, cov_file, cov_json,
            )
        if not ok:
            return PackageResult(
                repo_name=repo_name, import_name=import_name,
                status="skipped:coverage-failed", n_classes=None,
                raw_rho=None, partial_rho=None, graph_scd_rho=None, method_loc_rho=None,
                oracle_result=None, size_control_output=None,
                skip_reason=err,
            )

        oracle_out = str(out_dir / f"coverage_oracle_{repo_name}.json")
        print(f"[{repo_name}] running coverage_oracle ...", flush=True)
        ok, err, data = _run_oracle(
            repo, import_name, cov_json, oracle_out, exclude_substrs,
        )
        if not ok:
            return PackageResult(
                repo_name=repo_name, import_name=import_name,
                status="skipped:oracle-failed", n_classes=None,
                raw_rho=None, partial_rho=None, graph_scd_rho=None, method_loc_rho=None,
                oracle_result=None, size_control_output=None,
                skip_reason=err,
            )

        result = data["result"]
        print(f"[{repo_name}] running gap_size_control ...", flush=True)
        ok, err, sc_out = _run_size_control(repo, import_name, oracle_out)
        if not ok:
            print(f"[{repo_name}] WARNING: gap_size_control failed: {err[:120]}", flush=True)
            sc_out = ""

        graph_scd_rho, method_loc_rho = _parse_size_control_output(sc_out)

        return PackageResult(
            repo_name=repo_name,
            import_name=import_name,
            status="ok",
            n_classes=result["n_matched"],
            raw_rho=result["spearman_rho"],
            partial_rho=result["partial_rho_delta_cov"],
            graph_scd_rho=graph_scd_rho,
            method_loc_rho=method_loc_rho,
            oracle_result=result,
            size_control_output=sc_out,
            skip_reason=None,
        )


def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "    n/a"
    return f"{v:+.3f}"


def print_table(results: list[PackageResult], click_ref: dict | None) -> None:
    header = (
        f"{'package':<10}  {'n':>5}  {'raw ρ(Δ,cov)':>14}  "
        f"{'partial ρ':>11}  {'graph SCD ρ':>13}  {'method↔LOC ρ':>14}  status"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    all_rows = list(results)

    # Prepend click reference from existing JSON if available
    if click_ref is not None:
        cr = click_ref.get("result", click_ref)
        print(
            f"{'click (ref)':<10}  {cr['n_matched']:>5}  "
            f"{_fmt(cr['spearman_rho']):>14}  "
            f"{_fmt(cr['partial_rho_delta_cov']):>11}  "
            f"{'  n/a (use gap_size_control)':>13}  "
            f"  (reference — not re-run)"
        )

    ok_rows = []
    for r in all_rows:
        status_str = r.status if r.status != "ok" else ""
        print(
            f"{r.repo_name:<10}  "
            f"{r.n_classes if r.n_classes is not None else '':>5}  "
            f"{_fmt(r.raw_rho) if r.raw_rho is not None else '    n/a':>14}  "
            f"{_fmt(r.partial_rho) if r.partial_rho is not None else '    n/a':>11}  "
            f"{_fmt(r.graph_scd_rho) if r.graph_scd_rho is not None else '    n/a':>13}  "
            f"{_fmt(r.method_loc_rho) if r.method_loc_rho is not None else '    n/a':>14}  "
            f"{status_str}"
        )
        if r.status == "ok":
            ok_rows.append(r)

    print(sep)

    # Pooled summary
    if ok_rows:
        neg_partial = sum(1 for r in ok_rows if r.partial_rho is not None and r.partial_rho < 0)
        neg_graph   = sum(1 for r in ok_rows if r.graph_scd_rho is not None and r.graph_scd_rho < 0)
        partial_rhos = [r.partial_rho for r in ok_rows if r.partial_rho is not None and not math.isnan(r.partial_rho)]
        graph_rhos   = [r.graph_scd_rho for r in ok_rows if r.graph_scd_rho is not None and not math.isnan(r.graph_scd_rho)]
        n_ok = len(ok_rows)
        print(f"\nPOOLED SUMMARY (n={n_ok} successful packages, click NOT included in pool):")
        print(f"  partial ρ(Δ,cov|size) negative: {neg_partial}/{n_ok}  "
              f"mean = {np.mean(partial_rhos):+.3f}  (hypothesis: all negative)")
        print(f"  graph SCD ρ negative:            {neg_graph}/{len(graph_rhos)}  "
              f"mean = {np.mean(graph_rhos):+.3f}  (hypothesis: all negative)")
        print()

    skipped = [r for r in all_rows if r.status != "ok"]
    if skipped:
        print("SKIPPED packages:")
        for r in skipped:
            print(f"  {r.repo_name}: {r.skip_reason}")
        print()


def _provenance() -> str:
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    return (
        f"generated by: {cmd}  "
        "(regenerate rather than edit by hand; see module docstring for full REPRODUCE recipe)"
    )


def main() -> None:
    bookgap_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--packages",
        default=None,
        help="comma-separated list of repo:import_name pairs; default: attrs:attr,jinja2:jinja2,rich:rich,flask:flask,typer:typer,black:black",
    )
    ap.add_argument(
        "--repos-root",
        default=str(bookgap_root / "cache/code_gap/python/repos"),
        help="root directory containing cloned repos",
    )
    ap.add_argument(
        "--out",
        default=str(bookgap_root / "artifacts/code_gap/python/coverage_oracle_multi.json"),
        help="path to write the aggregate JSON result",
    )
    ap.add_argument(
        "--click-ref",
        default=str(bookgap_root / "artifacts/code_gap/python/coverage_oracle_click.json"),
        help="pre-computed click oracle JSON to include as reference row",
    )
    args = ap.parse_args()

    repos_root = Path(args.repos_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse package list
    if args.packages:
        pkg_list = []
        for item in args.packages.split(","):
            parts = item.strip().split(":")
            if len(parts) != 2:
                print(f"WARNING: ignoring malformed package spec {item!r} (expected repo:import_name)")
                continue
            pkg_list.append((parts[0], parts[1], []))
    else:
        pkg_list = DEFAULT_PACKAGES

    # Load click reference if available
    click_ref = None
    click_ref_path = Path(args.click_ref)
    if click_ref_path.exists():
        try:
            click_ref = json.loads(click_ref_path.read_text())
        except Exception:
            pass

    print(f"Processing {len(pkg_list)} packages from {repos_root}")

    results: list[PackageResult] = []
    for repo_name, import_name, excludes in pkg_list:
        print(f"\n{'='*60}")
        print(f"PACKAGE: {repo_name} (import: {import_name})")
        print(f"{'='*60}")
        try:
            r = process_package(
                repo_name=repo_name,
                import_name=import_name,
                exclude_substrs=excludes,
                repos_root=repos_root,
                out_dir=out_path.parent,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[{repo_name}] UNEXPECTED ERROR: {e}\n{tb}")
            r = PackageResult(
                repo_name=repo_name, import_name=import_name,
                status=f"skipped:unexpected-error", n_classes=None,
                raw_rho=None, partial_rho=None, graph_scd_rho=None, method_loc_rho=None,
                oracle_result=None, size_control_output=None,
                skip_reason=str(e),
            )
        results.append(r)
        print(f"\n[{repo_name}] → {r.status}", flush=True)
        if r.skip_reason:
            print(f"[{repo_name}] reason: {r.skip_reason}", flush=True)

    print_table(results, click_ref)

    # Serialize to JSON
    def _to_dict(r: PackageResult) -> dict:
        return {
            "repo_name": r.repo_name,
            "import_name": r.import_name,
            "status": r.status,
            "n_classes": r.n_classes,
            "raw_rho": r.raw_rho,
            "partial_rho": r.partial_rho,
            "graph_scd_rho": r.graph_scd_rho,
            "method_loc_rho": r.method_loc_rho,
            "oracle_result": r.oracle_result,
            "size_control_output": r.size_control_output,
            "skip_reason": r.skip_reason,
        }

    ok_rows = [r for r in results if r.status == "ok"]
    partial_rhos = [r.partial_rho for r in ok_rows if r.partial_rho is not None and not math.isnan(r.partial_rho)]
    graph_rhos   = [r.graph_scd_rho for r in ok_rows if r.graph_scd_rho is not None and not math.isnan(r.graph_scd_rho)]
    pooled = {
        "n_packages_ok": len(ok_rows),
        "partial_rho_negative_count": sum(1 for v in partial_rhos if v < 0),
        "partial_rho_mean": float(np.mean(partial_rhos)) if partial_rhos else None,
        "graph_scd_rho_negative_count": sum(1 for v in graph_rhos if v < 0),
        "graph_scd_rho_mean": float(np.mean(graph_rhos)) if graph_rhos else None,
    }

    payload = {
        "provenance": _provenance(),
        "click_reference": click_ref["result"] if click_ref else None,
        "packages": [_to_dict(r) for r in results],
        "pooled_summary": pooled,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote aggregate result to {out_path}")


if __name__ == "__main__":
    main()
