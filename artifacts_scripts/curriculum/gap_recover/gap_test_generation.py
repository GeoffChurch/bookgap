"""Phase B — the application: gap-seeded vs baseline LLM test generation.

Phase A (``coverage_oracle.py``) established that, controlling for class size, the
code-gap detector's high-Δ classes are genuinely *under-covered*. Phase B asks the
downstream question: **if you point an LLM test-generator at the detected gaps, does it
recover MORE new coverage than pointing it at a matched baseline of low-Δ classes?**

Design
------
* **Gap set**: the top-``k`` public classes by Δ (the detector's actual recommendations).
* **Baseline set**: ``k`` public classes with the *lowest* Δ, drawn to **match the gap
  set on class size** (greedy nearest-size match) so the comparison is not merely
  "gap classes are smaller" — the Phase-A size confound is neutralised by construction.

For each target class we ask a **local ollama** model (per project policy — NOT the
OpenAI Batch API) to write a focused ``pytest`` module that imports the class from
``click`` and constructs/exercises it. We provide the class's own source as context.
Each generated test is run *individually* in the isolated venv under ``coverage``, and
we record the **NEW source lines** it covers — statements in the click source that were
NOT already covered by Phase A's full-suite baseline run. Tests that fail to
import/collect/run are skipped and counted (the run-success rate is a headline number;
many LLM-written tests won't run, which is expected).

The honest confounds, reported alongside the result:
  * **Coverage headroom**: a mature library leaves little uncovered; if the gap classes
    are already ~100% covered there is nothing to recover regardless of test quality.
  * **Class size**: larger classes have more recoverable lines; the baseline is
    size-matched to control this, and per-class normalised deltas are also reported.

This is the *application* arm and depends on LLM output quality; Phase A is the clean
deterministic result.

REPRODUCE
---------
Requires the same isolated venv and baseline ``.coverage`` as Phase A (see
``coverage_oracle.py``'s REPRODUCE block) and a running ollama with the chosen model::

    conda run -p .conda python -m artifacts_scripts.curriculum.gap_recover.gap_test_generation \
        --repo cache/code_gap/python/repos/click --import-name click \
        --coverage-json /tmp/coverage.json \
        --baseline-coverage /tmp/.coverage \
        --venv /tmp/covenv --workdir /tmp/gaptests \
        --model gemma4:12b --k 8 --seed 0 \
        --out artifacts/code_gap/python/gap_test_generation_click.json

(Generated test files and per-test .coverage data live under ``--workdir`` in /tmp,
gitignored; only the result JSON is a durable artifact.)
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from artifacts_scripts.curriculum.gap_recover.adapters.python.extract import (
    _locate_pkg_dir,
    module_name_for,
)
from artifacts_scripts.curriculum.gap_recover.coverage_oracle import (
    per_class_coverage,
    per_class_delta,
)

_MODULE = "artifacts_scripts.curriculum.gap_recover.gap_test_generation"


# --------------------------------------------------------------------------- #
# target selection
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    qualname: str           # click.module.Class
    import_module: str      # click.module
    class_name: str         # Class
    delta: float
    structural_users: int
    operational_users: int
    n_stmt: int
    covered_fraction: float
    rel_path: str
    arm: str = ""           # "gap" | "baseline"


def _is_public_top_level(qual: str, rel_path: str) -> bool:
    parts = qual.split(".")
    if len(parts) != 3:  # import_name.module.Class only (no nested/methods)
        return False
    cls = parts[-1]
    return (
        not cls.startswith("_")
        and "_winconsole" not in qual
        and "InfoDict" not in cls  # TypedDicts aren't instantiable like classes
    )


def select_targets(
    repo: Path, import_name: str, cov: dict, *, k: int,
) -> tuple[list[Target], list[Target], list[dict]]:
    """Return ``(gap_targets, baseline_targets, all_rows)``.

    Gap = top-k public classes by Δ. Baseline = k lowest-Δ public classes,
    greedily matched to the gap set on ``n_stmt`` (class size) so the two arms
    have comparable recoverable-line budgets.
    """
    pkg_dir = _locate_pkg_dir(repo, import_name)
    if pkg_dir is None:
        raise FileNotFoundError(f"no package dir under {repo} for {import_name!r}")
    cov_by_qual = per_class_coverage(repo, import_name, cov, pkg_dir)
    delta_by_qual = per_class_delta(repo, import_name)

    rows: list[dict] = []
    cands: list[Target] = []
    for q in sorted(set(cov_by_qual) & set(delta_by_qual)):
        c, d = cov_by_qual[q], delta_by_qual[q]
        rows.append(
            {
                "qualname": q, "delta": d.delta,
                "structural_users": d.structural_users,
                "operational_users": d.operational_users,
                "n_stmt": c.n_stmt, "covered_fraction": c.covered_fraction,
                "rel_path": c.rel_path,
            }
        )
        if _is_public_top_level(q, c.rel_path):
            parts = q.split(".")
            cands.append(
                Target(
                    qualname=q, import_module=".".join(parts[:-1]), class_name=parts[-1],
                    delta=d.delta, structural_users=d.structural_users,
                    operational_users=d.operational_users, n_stmt=c.n_stmt,
                    covered_fraction=c.covered_fraction, rel_path=c.rel_path,
                )
            )

    cands.sort(key=lambda t: -t.delta)
    gap = cands[:k]
    for t in gap:
        t.arm = "gap"

    # baseline: lowest-Δ pool, size-matched greedily to the gap set.
    gap_names = {g.qualname for g in gap}
    low_pool: list[Target] = sorted(
        [t for t in cands if t.qualname not in gap_names],
        key=lambda t: t.delta,  # ascending Δ (most anti-gap first)
    )
    baseline: list[Target] = []
    used: set[str] = set()
    for g in sorted(gap, key=lambda t: -t.n_stmt):  # match big ones first
        best: Target | None = None
        best_d: int | None = None
        for pool_t in low_pool:
            if pool_t.qualname in used:
                continue
            dd = abs(pool_t.n_stmt - g.n_stmt)
            if best_d is None or dd < best_d:
                best_d, best = dd, pool_t
        if best is not None:
            best.arm = "baseline"
            baseline.append(best)
            used.add(best.qualname)
    return gap, baseline, rows


# --------------------------------------------------------------------------- #
# source extraction for the prompt
# --------------------------------------------------------------------------- #
def class_source(repo: Path, import_name: str, target: Target) -> str:
    """The class's own source text (truncated), as context for the generator."""
    pkg_dir = _locate_pkg_dir(repo, import_name)
    assert pkg_dir is not None
    abs_path = (repo / target.rel_path).resolve()
    if not abs_path.exists():
        return ""
    src = abs_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(abs_path))
    module = module_name_for(abs_path, pkg_dir, import_name)
    lines = src.splitlines()

    def find(node: ast.AST, prefix: str) -> tuple[int, int] | None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qual = f"{module}.{prefix}{child.name}"
                if qual == target.qualname:
                    lo = child.lineno
                    hi = getattr(child, "end_lineno", lo) or lo
                    return lo, hi
                r = find(child, f"{prefix}{child.name}.")
                if r:
                    return r
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                r = find(child, prefix)
                if r:
                    return r
        return None

    span = find(tree, "")
    if span is None:
        return ""
    lo, hi = span
    snippet = "\n".join(lines[lo - 1 : hi])
    return snippet[:6000]  # cap context length


# --------------------------------------------------------------------------- #
# ollama generation
# --------------------------------------------------------------------------- #
_PROMPT = """\
You are writing a focused pytest test module for the Python library `click`.

Target class: `{cls}` (import it with `from {mod} import {cls}`).

Here is the class's source code:
```python
{source}
```

Write a SINGLE self-contained pytest test module that:
- imports `{cls}` from `{mod}` (use exactly `from {mod} import {cls}`);
- constructs instances and/or calls methods of `{cls}` to exercise its behaviour;
- uses several `def test_*():` functions with plain `assert` statements;
- only uses the public `click` API and the standard library (no network, no extra deps);
- must be runnable as-is — do NOT use fixtures that require plugins, and do NOT reference
  undefined names.

Output ONLY the Python code for the test module, no prose, no markdown fences.
"""


def _extract_code(text: str) -> str:
    """Pull a python code block out of the model output, or take it verbatim."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code.strip() + "\n"


def generate_test(model: str, target: Target, source: str, *, host: str, timeout: int) -> str:
    prompt = _PROMPT.format(cls=target.class_name, mod=target.import_module, source=source)
    payload = json.dumps(
        {
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return _extract_code(data.get("response", ""))


# --------------------------------------------------------------------------- #
# run a generated test under coverage; measure NEW lines vs baseline
# --------------------------------------------------------------------------- #
@dataclass
class TestRun:
    qualname: str
    arm: str
    ok: bool                       # test process exited 0 (collected + passed/ran)
    error: str                     # short reason if not ok
    new_lines: int                 # source lines covered that baseline did NOT cover
    new_lines_in_target: int       # subset of new_lines inside the target class span
    n_stmt: int                    # target class size (for normalisation)
    covered_fraction: float        # target's baseline coverage fraction


def _baseline_covered(baseline_cov_json: dict, repo: Path) -> dict[str, set[int]]:
    """Per abs-source-path set of baseline-covered line numbers."""
    out: dict[str, set[int]] = {}
    for rel, fd in baseline_cov_json["files"].items():
        abs_path = str((repo / rel).resolve())
        out[abs_path] = set(fd.get("executed_lines", []))
    return out


def run_generated_test(
    venv: Path, workdir: Path, src_root: Path, repo: Path,
    target: Target, code: str,
    baseline_covered: dict[str, set[int]], target_span: tuple[int, int] | None,
    *, timeout: int,
) -> TestRun:
    """Write ``code`` to a temp test file, run it under coverage, diff vs baseline."""
    safe = target.qualname.replace(".", "_")
    test_path = workdir / f"test_gen_{target.arm}_{safe}.py"
    test_path.write_text(code)
    cov_file = workdir / f".cov_{target.arm}_{safe}"
    cov_json = workdir / f"cov_{target.arm}_{safe}.json"
    for p in (cov_file, cov_json):
        if p.exists():
            p.unlink()

    py = venv / "bin" / "python"
    env_run = [
        str(py), "-m", "coverage", "run", f"--source={src_root}", "-m", "pytest",
        str(test_path), "-q", "-p", "no:randomly", "-p", "no:cacheprovider",
    ]
    try:
        proc = subprocess.run(
            env_run, capture_output=True, text=True, timeout=timeout,
            cwd=str(repo), env={"COVERAGE_FILE": str(cov_file), "PATH": _path(), "HOME": "/tmp"},
        )
    except subprocess.TimeoutExpired:
        return TestRun(target.qualname, target.arm, False, "timeout", 0, 0,
                       target.n_stmt, target.covered_fraction)

    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        reason = _classify_failure(tail)
        return TestRun(target.qualname, target.arm, False, reason, 0, 0,
                       target.n_stmt, target.covered_fraction)

    # success: produce json and diff vs baseline
    try:
        subprocess.run(
            [str(py), "-m", "coverage", "json", "-o", str(cov_json)],
            capture_output=True, text=True, timeout=120,
            cwd=str(repo), env={"COVERAGE_FILE": str(cov_file), "PATH": _path(), "HOME": "/tmp"},
            check=True,
        )
        cj = json.loads(cov_json.read_text())
    except Exception as e:  # pragma: no cover
        return TestRun(target.qualname, target.arm, True, f"covjson:{e}", 0, 0,
                       target.n_stmt, target.covered_fraction)

    new_total = 0
    new_in_target = 0
    target_abs = str((repo / target.rel_path).resolve())
    for rel, fd in cj["files"].items():
        abs_path = str((repo / rel).resolve())
        execed = set(fd.get("executed_lines", []))
        base = baseline_covered.get(abs_path, set())
        new = execed - base
        new_total += len(new)
        if abs_path == target_abs and target_span is not None:
            lo, hi = target_span
            new_in_target += sum(1 for ln in new if lo <= ln <= hi)
    return TestRun(target.qualname, target.arm, True, "", new_total, new_in_target,
                   target.n_stmt, target.covered_fraction)


def _path() -> str:
    import os

    return os.environ.get("PATH", "/usr/bin:/bin")


def _classify_failure(tail: list[str]) -> str:
    joined = "\n".join(tail[-25:])
    for pat, label in [
        (r"ImportError|ModuleNotFoundError", "import_error"),
        (r"SyntaxError", "syntax_error"),
        (r"collected 0 items|no tests ran", "no_tests"),
        (r"errors during collection|ERROR", "collection_error"),
        (r"AttributeError", "attribute_error"),
        (r"TypeError", "type_error"),
        (r"failed", "assertion/runtime_failure"),
    ]:
        if re.search(pat, joined):
            return label
    return "nonzero_exit"


def _target_span(repo: Path, import_name: str, target: Target) -> tuple[int, int] | None:
    pkg_dir = _locate_pkg_dir(repo, import_name)
    assert pkg_dir is not None
    abs_path = (repo / target.rel_path).resolve()
    if not abs_path.exists():
        return None
    tree = ast.parse(abs_path.read_text(encoding="utf-8"))
    module = module_name_for(abs_path, pkg_dir, import_name)

    def find(node: ast.AST, prefix: str) -> tuple[int, int] | None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if f"{module}.{prefix}{child.name}" == target.qualname:
                    return child.lineno, getattr(child, "end_lineno", child.lineno) or child.lineno
                r = find(child, f"{prefix}{child.name}.")
                if r:
                    return r
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                r = find(child, prefix)
                if r:
                    return r
        return None

    return find(tree, "")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PhaseBResult:
    repo: str
    import_name: str
    model: str
    k: int
    gap_qualnames: list[str]
    baseline_qualnames: list[str]
    n_gap_ok: int
    n_baseline_ok: int
    gap_run_success_rate: float
    baseline_run_success_rate: float
    gap_new_lines_total: int
    baseline_new_lines_total: int
    gap_new_lines_in_target_total: int
    baseline_new_lines_in_target_total: int
    gap_mean_size: float
    baseline_mean_size: float
    gap_mean_baseline_cov: float
    baseline_mean_baseline_cov: float
    runs: list[dict] = field(default_factory=list)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--import-name", required=True)
    ap.add_argument("--coverage-json", required=True, help="Phase-A coverage json (for selection + baseline diff)")
    ap.add_argument("--baseline-coverage", default=None,
                    help="Phase-A .coverage (unused if --coverage-json carries executed_lines; kept for clarity)")
    ap.add_argument("--venv", required=True, help="isolated venv with click+pytest+coverage")
    ap.add_argument("--workdir", default="/tmp/gaptests", help="scratch dir for generated tests (gitignored)")
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-timeout", type=int, default=300)
    ap.add_argument("--run-timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    venv = Path(args.venv).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    src_root = (repo / "src" / args.import_name)
    if not src_root.exists():
        src_root = repo / args.import_name

    cov = json.loads(Path(args.coverage_json).read_text())
    gap, baseline, _rows = select_targets(repo, args.import_name, cov, k=args.k)
    baseline_covered = _baseline_covered(cov, repo)

    print(f"=== Phase B: gap-seeded vs baseline test generation ({args.model}) ===")
    print(f"src_root = {src_root}")
    print(f"\nGAP set (top-{args.k} Δ):")
    for t in gap:
        print(f"  {t.qualname:45s} Δ={t.delta:+.2f} n={t.n_stmt:3d} cov={t.covered_fraction:.2f}")
    print(f"BASELINE set (size-matched low-Δ):")
    for t in baseline:
        print(f"  {t.qualname:45s} Δ={t.delta:+.2f} n={t.n_stmt:3d} cov={t.covered_fraction:.2f}")
    print()

    runs: list[TestRun] = []
    for arm, targets in (("gap", gap), ("baseline", baseline)):
        for t in targets:
            src = class_source(repo, args.import_name, t)
            try:
                code = generate_test(args.model, t, src, host=args.host, timeout=args.gen_timeout)
            except Exception as e:
                print(f"  [{arm}] {t.class_name}: GEN FAILED ({e})")
                runs.append(TestRun(t.qualname, arm, False, f"gen_error:{e}", 0, 0,
                                    t.n_stmt, t.covered_fraction))
                continue
            span = _target_span(repo, args.import_name, t)
            tr = run_generated_test(
                venv, workdir, src_root, repo, t, code, baseline_covered, span,
                timeout=args.run_timeout,
            )
            runs.append(tr)
            status = "OK " if tr.ok else f"SKIP({tr.error})"
            print(f"  [{arm}] {t.class_name:30s} {status:28s} new_lines={tr.new_lines:3d} "
                  f"(in-target {tr.new_lines_in_target})")

    def agg(arm: str) -> dict:
        rs = [r for r in runs if r.arm == arm]
        ok = [r for r in rs if r.ok]
        return {
            "n": len(rs), "n_ok": len(ok),
            "rate": len(ok) / len(rs) if rs else 0.0,
            "new": sum(r.new_lines for r in ok),
            "new_in_target": sum(r.new_lines_in_target for r in ok),
            "mean_size": float(np.mean([r.n_stmt for r in rs])) if rs else 0.0,
            "mean_cov": float(np.mean([r.covered_fraction for r in rs])) if rs else 0.0,
        }

    g, b = agg("gap"), agg("baseline")
    result = PhaseBResult(
        repo=str(repo), import_name=args.import_name, model=args.model, k=args.k,
        gap_qualnames=[t.qualname for t in gap],
        baseline_qualnames=[t.qualname for t in baseline],
        n_gap_ok=g["n_ok"], n_baseline_ok=b["n_ok"],
        gap_run_success_rate=g["rate"], baseline_run_success_rate=b["rate"],
        gap_new_lines_total=g["new"], baseline_new_lines_total=b["new"],
        gap_new_lines_in_target_total=g["new_in_target"],
        baseline_new_lines_in_target_total=b["new_in_target"],
        gap_mean_size=g["mean_size"], baseline_mean_size=b["mean_size"],
        gap_mean_baseline_cov=g["mean_cov"], baseline_mean_baseline_cov=b["mean_cov"],
        runs=[asdict(r) for r in runs],
    )

    print("\n=== RESULT ===")
    print(f"run-success rate:  gap {g['n_ok']}/{g['n']} ({g['rate']:.0%})   "
          f"baseline {b['n_ok']}/{b['n']} ({b['rate']:.0%})")
    print(f"NEW lines covered (any source):  gap {g['new']}   baseline {b['new']}")
    print(f"NEW lines covered (in target class): gap {g['new_in_target']}   baseline {b['new_in_target']}")
    print(f"mean class size: gap {g['mean_size']:.0f}   baseline {b['mean_size']:.0f}  (size-matched)")
    print(f"mean baseline coverage: gap {g['mean_cov']:.2f}   baseline {b['mean_cov']:.2f}  "
          f"(headroom = 1 - this)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
        payload = {
            "provenance": f"generated by: {cmd} (regenerate; do not hand-edit)",
            "result": asdict(result),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
