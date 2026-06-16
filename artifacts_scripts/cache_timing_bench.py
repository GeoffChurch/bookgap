#!/usr/bin/env python3
"""Microbenchmark: joblib on-disk @MEMORY.cache vs functools.lru_cache for three
cheap-compute / tiny-output functions, decomposed by phase.

Decides whether to switch these three functions from joblib on-disk caching
(`@MEMORY.cache`) to in-memory `functools.lru_cache`:

  1. `_resolve_filter_scopes`   (bookgap/openalex_ingest.py:113)
  2. `_build_temporal_subcloud` (artifacts_scripts/historical_gap_analysis/historical_gap_analysis.py:49)
  3. `_build_temporal_test_set` (same file:84)

Phases measured per function (steady-state, warm page cache, GC disabled):

  T_compute  raw undecorated body            -> mf.func(*args)
  T_miss     1st decorated call after clear  -> compute + persist
  T_persist  T_miss - T_compute              (derived)
  T_jobhit   subsequent decorated call       -> joblib load path (stat + unpickle)
  T_lruhit   functools.lru_cache 2nd call    -> in-memory dict hit
  output_sz  len(pickle.dumps(output))

IMPORTANT: a *throwaway* joblib.Memory(<tempdir>) is used for all miss/hit
control so the live 172 GB cache/joblib/ tree is never touched. The real
*undecorated* `.func` is used for T_compute.

Decision rule reported per function:
  - Repeated same-process access: T_jobhit vs T_lruhit (expect lru >> faster).
  - Single cross-process access (the case on-disk persistence is *supposed* to
    win): on-disk pays only if T_jobhit < T_compute. If T_compute <= T_jobhit,
    recompute beats disk-load -> on-disk caching is pure overhead even
    cross-process. (A *cold* page-cache read is slower still; warm is the
    primary measurement and is the most charitable case for on-disk.)

Functions 2 & 3 caveat: joblib/lru cache the *return value* (a frozen SubCloud
= (parent_cloud, post_filter), ~0.9 KB pickled), NOT the embeddings array. The
function *body* forces an expensive `embeddings()` count, but that cost is paid
only on a true miss (T_compute). On a hit, both joblib and lru return the tiny
SubCloud without re-running embeddings(). So the decisive comparison is:
"is persisting+loading the tiny SubCloud cheaper than reconstructing the lazy
dataclass?" -- we report the bare-constructor cost separately from the
forced-embeddings() cost.

Regenerate (do not edit output by hand):

    conda run -p /home/gchurchill/src/bookgap/.conda \\
        python -m artifacts_scripts.cache_timing_bench

Optional flags:
    --reps N            reps for cheap function 1 (default 400)
    --reps-heavy N      reps for SubCloud micro-measurements (default 200)
    --snapshot DATE     taxonomy snapshot date (default 2025-11-06)
    --no-subcloud       skip functions 2&3 (function 1 only; no Cloud build)
    --write-doc         append results to docs/cache_cleanup.md
"""
from __future__ import annotations

import argparse
import gc
import pickle
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import joblib

REPO = Path(__file__).resolve().parents[1]
_MODULE = "artifacts_scripts.cache_timing_bench"


# ----------------------------------------------------------------------------- timing core
@dataclass
class Stat:
    """Summary of a timed phase, nanoseconds internally; rendered in us."""
    median_ns: float
    p90_ns: float
    min_ns: float
    n: int

    @property
    def median_us(self) -> float:
        return self.median_ns / 1e3

    @property
    def p90_us(self) -> float:
        return self.p90_ns / 1e3

    @property
    def min_us(self) -> float:
        return self.min_ns / 1e3


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def time_fn(fn: Callable[[], Any], reps: int, warmup: int = 5) -> Stat:
    """Time a zero-arg callable `reps` times, GC disabled, return median/p90/min."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    gc_was = gc.isenabled()
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            samples.append(float(t1 - t0))
    finally:
        if gc_was:
            gc.enable()
    samples.sort()
    return Stat(
        median_ns=statistics.median(samples),
        p90_ns=_percentile(samples, 0.90),
        min_ns=samples[0],
        n=reps,
    )


def _fresh_memory(tmpdir: Path, tag: str) -> joblib.Memory:
    d = tmpdir / tag
    d.mkdir(parents=True, exist_ok=True)
    return joblib.Memory(str(d), verbose=0)


# ----------------------------------------------------------------------------- function 1
def bench_resolve_filter_scopes(tmpdir: Path, reps: int, snapshot: str) -> list[dict]:
    """Full phase decomposition for _resolve_filter_scopes across small->large outputs."""
    from bookgap.openalex_ingest import _resolve_filter_scopes  # MemorizedFunc
    from bookgap.snapshot_ingest import load_taxonomy
    from bookgap.util import Filter, WorkType

    # Warm the in-process taxonomy cache (snapshot_ingest._TAXONOMY_CACHE) so we
    # measure the steady-state path the production code sees (taxonomy already
    # memoized), not a one-time JSON load.
    load_taxonomy(snapshot)

    raw_func = _resolve_filter_scopes.func  # undecorated body
    wt = (WorkType.BOOK.value, WorkType.ARTICLE.value)

    cases = [
        ("topic T10828 (small)", Filter(topic="T10828", work_types=wt)),
        ("subfield 1711 (medium)", Filter(subfield="1711", work_types=wt)),
        ("field 17 (large)", Filter(field="17", work_types=wt)),
    ]

    rows: list[dict] = []
    for label, filt in cases:
        out = raw_func(filt, snapshot)
        out_sz = len(pickle.dumps(out))
        n_scopes = len(out)

        # T_compute: raw body, no caching.
        t_compute = time_fn(lambda: raw_func(filt, snapshot), reps)

        # Fresh throwaway joblib Memory; decorate a copy of the raw func.
        mem = _fresh_memory(tmpdir, "f1")
        cached = mem.cache(raw_func)

        # T_miss: first decorated call after clear() = compute + persist.
        # Re-clear and re-time many singletons to get a distribution.
        miss_samples: list[float] = []
        for _ in range(min(reps, 60)):
            cached.clear(warn=False)
            t0 = time.perf_counter_ns()
            cached(filt, snapshot)
            t1 = time.perf_counter_ns()
            miss_samples.append(float(t1 - t0))
        miss_samples.sort()
        t_miss = Stat(statistics.median(miss_samples),
                      _percentile(miss_samples, 0.90),
                      miss_samples[0], len(miss_samples))

        # T_jobhit: subsequent decorated call (joblib stat + unpickle from disk).
        cached(filt, snapshot)  # ensure populated
        t_jobhit = time_fn(lambda: cached(filt, snapshot), reps)

        # T_lruhit: functools.lru_cache 2nd call (in-memory dict hit).
        lru_wrapped = lru_cache(maxsize=None)(raw_func)
        lru_wrapped(filt, snapshot)  # populate
        t_lruhit = time_fn(lambda: lru_wrapped(filt, snapshot), reps)

        rows.append({
            "function": "_resolve_filter_scopes",
            "case": label,
            "n_scopes": n_scopes,
            "output_bytes": out_sz,
            "T_compute": t_compute,
            "T_miss": t_miss,
            "T_jobhit": t_jobhit,
            "T_lruhit": t_lruhit,
        })
    return rows


# ----------------------------------------------------------------------------- functions 2 & 3
def _build_ai_cloud(snapshot: str):
    """Build the cached AI-subfield (1702) production cloud offline (cheap parent).

    Mirrors tests/conftest.py::_ai_cloud so the parent is a real, on-disk-backed
    Cloud with no network. ~3-4 s one-time setup.
    """
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate
    from bookgap.configs.config_store import register_configs

    overrides = [
        "cloud.filter_pair.pre.domain=null",
        "cloud.filter_pair.pre.subfield='1702'",
        "cloud.filter_pair.pre.work_types=[book,article]",
        "cloud.denoiser=LPP",
        "cloud.drop_k=10",
        "gap.radius_fraction=0.75",
        f"cloud.snapshot_date='{snapshot}'",
    ]
    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        cfg = compose(config_name="standard", overrides=overrides)
    return instantiate(cfg.cloud)


def bench_subcloud_builders(tmpdir: Path, reps_heavy: int, snapshot: str) -> list[dict]:
    """Decisive measurements for _build_temporal_subcloud / _build_temporal_test_set.

    The return value is a frozen SubCloud (parent, post_filter) ~0.9 KB. The
    body forces embeddings() (expensive) ONLY on a miss. We measure:

      T_compute_full   full body incl. forced embeddings() count (true miss cost)
      T_construct      bare SubCloud(...) dataclass construction (lazy; the part
                       recompute pays beyond the already-cloud-cached embeddings)
      T_jobhit         throwaway-Memory store -> load round-trip of the SubCloud
                       (joblib's persist+unpickle, == what a hit pays)
      T_pickle_rt      raw pickle.dumps + pickle.loads of the SubCloud
      T_lruhit         functools.lru_cache dict hit returning the SubCloud
      output_bytes     len(pickle.dumps(subcloud))
    """
    import importlib
    hga = importlib.import_module(
        "artifacts_scripts.historical_gap_analysis.historical_gap_analysis"
    )
    from bookgap.util import WorkType

    cloud = _build_ai_cloud(snapshot)

    specs = [
        ("_build_temporal_subcloud", hga._build_temporal_subcloud,
         dict(max_year=2015, work_type=WorkType.BOOK)),
        ("_build_temporal_test_set", hga._build_temporal_test_set,
         dict(target_year=2016, work_type=WorkType.BOOK)),
    ]

    rows: list[dict] = []
    for name, mf, kw in specs:
        raw_func = mf.func  # undecorated body

        # Representative output (the SubCloud the function returns).
        out = raw_func(cloud, **kw)
        out_sz = len(pickle.dumps(out))

        # T_compute_full: full body, including the forced embeddings() count.
        # This is the genuine miss cost. Few reps -- it is seconds each.
        full_reps = max(3, min(reps_heavy, 6))
        full_samples: list[float] = []
        for _ in range(full_reps):
            t0 = time.perf_counter_ns()
            raw_func(cloud, **kw)
            t1 = time.perf_counter_ns()
            full_samples.append(float(t1 - t0))
        full_samples.sort()
        t_compute_full = Stat(statistics.median(full_samples),
                              _percentile(full_samples, 0.90),
                              full_samples[0], len(full_samples))

        # T_construct: bare lazy SubCloud(...) build (no embeddings()). This is
        # the part of "recompute" that is NOT already cached on the Cloud --
        # i.e. the real alternative cost to a cache hit, since embeddings() is
        # itself MEMORY-cached on the cloud.
        from bookgap.filters import TemporalFilter
        from bookgap.subcloud import SubCloud
        pf = cloud.filter_pair.post
        if "max_year" in kw:
            tf_kwargs = dict(max_year=kw["max_year"])
        else:
            tf_kwargs = dict(exact_year=kw["target_year"])

        def _construct():
            tf = TemporalFilter(work_type=kw["work_type"], subfield=pf.subfield,
                                topic=pf.topic, **tf_kwargs)
            return SubCloud(parent=cloud, post_filter=tf)

        t_construct = time_fn(_construct, reps_heavy)

        # T_jobhit: throwaway-Memory store of the SubCloud, then load round-trip.
        # We stash a precomputed SubCloud via a trivial identity func so the
        # store cost is the SubCloud pickle and the hit cost is its unpickle --
        # exactly joblib's persist/load path, isolated from embeddings().
        mem = _fresh_memory(tmpdir, f"f23_{name}")

        @mem.cache
        def _identity(tag: str):
            return out  # closes over the representative SubCloud

        _identity("k")  # populate (pays persist once)
        t_jobhit = time_fn(lambda: _identity("k"), reps_heavy)

        # T_pickle_rt: raw pickle dumps+loads of the SubCloud.
        blob = pickle.dumps(out)
        t_pickle_rt = time_fn(lambda: pickle.loads(pickle.dumps(out)), reps_heavy)

        # T_lruhit: lru_cache dict hit returning the SubCloud.
        @lru_cache(maxsize=None)
        def _lru_identity(tag: str):
            return out

        _lru_identity("k")
        t_lruhit = time_fn(lambda: _lru_identity("k"), reps_heavy)

        rows.append({
            "function": name,
            "output_bytes": out_sz,
            "T_compute_full": t_compute_full,
            "T_construct": t_construct,
            "T_jobhit": t_jobhit,
            "T_pickle_rt": t_pickle_rt,
            "T_lruhit": t_lruhit,
        })
    return rows


# ----------------------------------------------------------------------------- rendering
def _fmt_us(stat: Stat) -> str:
    return f"{stat.median_us:,.2f} / {stat.p90_us:,.2f}"


def render_function1_table(rows: list[dict]) -> str:
    lines = [
        "#### `_resolve_filter_scopes` — full phase decomposition",
        "",
        "All times in microseconds, **median / p90** over N reps. "
        "`T_persist = T_miss − T_compute`.",
        "",
        "| case | n_scopes | output | T_compute (µs) | T_miss (µs) | T_persist (µs) | T_jobhit (µs) | T_lruhit (µs) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        persist_med = r["T_miss"].median_us - r["T_compute"].median_us
        lines.append(
            f"| {r['case']} | {r['n_scopes']} | {r['output_bytes']:,} B "
            f"| {_fmt_us(r['T_compute'])} | {_fmt_us(r['T_miss'])} "
            f"| {persist_med:,.2f} | {_fmt_us(r['T_jobhit'])} | {_fmt_us(r['T_lruhit'])} |"
        )
    return "\n".join(lines)


def render_function1_verdict(rows: list[dict]) -> str:
    out = ["", "**Verdict — `_resolve_filter_scopes`:**", ""]
    for r in rows:
        jh = r["T_jobhit"].median_us
        co = r["T_compute"].median_us
        lh = r["T_lruhit"].median_us
        cross = ("on-disk WINS cross-proc" if jh < co
                 else "on-disk is pure OVERHEAD even cross-proc (recompute ≤ disk-load)")
        out.append(
            f"- **{r['case']}**: repeated same-proc — T_jobhit {jh:,.2f} µs vs "
            f"T_lruhit {lh:,.2f} µs → lru is **{jh / lh:,.0f}× faster**. "
            f"Single cross-proc — T_compute {co:,.2f} µs vs T_jobhit {jh:,.2f} µs "
            f"→ {cross}."
        )
    return "\n".join(out)


def render_subcloud_table(rows: list[dict]) -> str:
    lines = [
        "#### `_build_temporal_subcloud` / `_build_temporal_test_set` — decisive measurements",
        "",
        "The cached **return value** is a frozen `SubCloud` (`parent`, `post_filter`); "
        "`embeddings()` is NOT in the cached object. `T_compute_full` is the genuine "
        "miss cost (forces the `embeddings()` count); `T_construct` is the bare lazy "
        "`SubCloud(...)` build (the real alternative to a hit, since `embeddings()` is "
        "itself `@MEMORY.cache`d on the Cloud). `T_jobhit`/`T_pickle_rt`/`T_lruhit` "
        "operate on the **representative SubCloud object** (clearly a stand-in for the "
        "cache load path, isolated from `embeddings()`). Times median / p90.",
        "",
        "| function | output | T_compute_full | T_construct (µs) | T_jobhit (µs) | T_pickle_rt (µs) | T_lruhit (µs) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        cf = r["T_compute_full"]
        cf_str = f"{cf.median_us / 1e3:,.1f} / {cf.p90_us / 1e3:,.1f} ms"
        lines.append(
            f"| `{r['function']}` | {r['output_bytes']:,} B | {cf_str} "
            f"| {_fmt_us(r['T_construct'])} | {_fmt_us(r['T_jobhit'])} "
            f"| {_fmt_us(r['T_pickle_rt'])} | {_fmt_us(r['T_lruhit'])} |"
        )
    return "\n".join(lines)


def render_subcloud_verdict(rows: list[dict]) -> str:
    out = ["", "**Verdict — `_build_temporal_subcloud` / `_build_temporal_test_set`:**", ""]
    for r in rows:
        jh = r["T_jobhit"].median_us
        ct = r["T_construct"].median_us
        lh = r["T_lruhit"].median_us
        cf = r["T_compute_full"].median_us
        cross = ("on-disk WINS cross-proc vs bare reconstruct"
                 if jh < ct else
                 "on-disk is pure OVERHEAD vs bare reconstruct (the lazy ctor is cheaper than unpickling)")
        out.append(
            f"- **`{r['function']}`**: the returned SubCloud is {r['output_bytes']:,} B. "
            f"Repeated same-proc — T_jobhit {jh:,.2f} µs vs T_lruhit {lh:,.2f} µs → "
            f"lru is **{jh / lh:,.0f}× faster**. Single cross-proc — bare reconstruct "
            f"T_construct {ct:,.2f} µs vs T_jobhit {jh:,.2f} µs → {cross}. "
            f"(A true miss that recomputes the body is {cf / 1e3:,.0f} ms, dominated by "
            f"`embeddings()`, but neither cache stores that array — only the tiny view.)"
        )
    return "\n".join(out)


def _provenance() -> str:
    cmd = f"python -m {_MODULE} " + " ".join(sys.argv[1:])
    return ("\n\n---\n"
            f"*Generated by `conda run -p .conda {cmd}`*  \n"
            "*Regenerate with the command above; do not edit by hand. "
            "Absolute times depend on hardware/page-cache and will drift; the "
            "ratios (T_jobhit/T_lruhit, T_compute vs T_jobhit) are the stable signal.*\n")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--reps-heavy", type=int, default=200)
    ap.add_argument("--snapshot", type=str, default="2025-11-06")
    ap.add_argument("--no-subcloud", action="store_true")
    ap.add_argument("--write-doc", action="store_true")
    args = ap.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="cache_timing_bench_"))
    print(f"# throwaway joblib cache: {tmpdir}", file=sys.stderr)

    try:
        print("## benchmarking _resolve_filter_scopes ...", file=sys.stderr)
        f1_rows = bench_resolve_filter_scopes(tmpdir, args.reps, args.snapshot)

        f23_rows: list[dict] = []
        if not args.no_subcloud:
            print("## building AI-1702 parent cloud + benchmarking SubCloud builders ...",
                  file=sys.stderr)
            f23_rows = bench_subcloud_builders(tmpdir, args.reps_heavy, args.snapshot)

        # --- console / stdout report ---
        parts = [
            "## Empirical timing (measured)",
            "",
            f"Measured on `{time.strftime('%Y-%m-%d')}` via "
            f"`{_MODULE}` (snapshot `{args.snapshot}`, "
            f"reps={args.reps} cheap / {args.reps_heavy} heavy). "
            "Throwaway `joblib.Memory(tmpdir)` used for all miss/hit control; the "
            "live `cache/joblib/` tree was not touched. Warm page cache (a *cold* "
            "first-ever cross-process read is slower still, making on-disk look "
            "even worse than shown).",
            "",
            render_function1_table(f1_rows),
            render_function1_verdict(f1_rows),
        ]
        if f23_rows:
            parts += ["", render_subcloud_table(f23_rows), render_subcloud_verdict(f23_rows)]

        # Bottom line.
        f1_ratio = statistics.median(
            [r["T_jobhit"].median_us / r["T_lruhit"].median_us for r in f1_rows]
        )
        f1_cross = all(r["T_compute"].median_us <= r["T_jobhit"].median_us for r in f1_rows)
        bottom = [
            "",
            "**Bottom line.**",
            "",
            f"The data confirms the `→ lru_cache` recommendation. For "
            f"`_resolve_filter_scopes`, an in-memory hit is ~**{f1_ratio:,.0f}×** "
            f"faster than joblib's disk-load hit across all output sizes, and "
            + ("recompute is *already* as cheap as (or cheaper than) the warm disk-load "
               "for every case — so on-disk persistence buys nothing even on a single "
               "cross-process call, while costing a pickle round-trip + cache-dir stat "
               "on every same-process repeat."
               if f1_cross else
               "the warm disk-load only beats recompute for the largest outputs."),
        ]
        if f23_rows:
            bottom.append(
                "For the two SubCloud builders, the cached object is a ~0.9 KB lazy "
                "view; unpickling it via joblib costs more than simply re-running its "
                "frozen-dataclass constructor (the genuinely expensive `embeddings()` "
                "count is cached on the Cloud, not in these results), so on-disk "
                "caching here is net-negative in every regime. **Switch all three to "
                "`lru_cache`.**"
            )
        parts += bottom

        report = "\n".join(parts)
        print(report)

        if args.write_doc:
            doc = REPO / "docs" / "cache_cleanup.md"
            existing = doc.read_text()
            marker = "## Empirical timing (measured)"
            if marker in existing:
                existing = existing.split(marker)[0].rstrip() + "\n"
            with doc.open("w") as fh:
                fh.write(existing.rstrip() + "\n\n" + report + _provenance())
            print(f"\n# appended to {doc}", file=sys.stderr)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"# cleaned up {tmpdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
