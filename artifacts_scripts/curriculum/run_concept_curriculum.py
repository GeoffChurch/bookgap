"""One-command curriculum-from-a-book-ToC pipeline — the de-incanted best recipe.

Chains the validated steps so the recipe is ONE command, not a flag-incantation:
  keyword-concept seeding  ->  GCCA multi-view consensus clustering (+ citation ordering)
  ->  judge-parseable summary

Default clusterer = GCCA (branchless multi-view consensus over keyword + embedding + bibliographic-
coupling; validated 2026-06-04 as a near-best AUTO-SELECTOR — 15/16 dimension-comparisons tie-or-win vs
each book's best single-view CCA oracle, so it retires the per-book embedding-vs-coupling view choice).
``--cluster cca`` = the single-view CCA-hybrid oracle; ``--cluster embedding`` = the decoupled-selection
+ embedding-dendrogram path.

Prereq: the cached field work×term index (`field_term_index --field <F> --build`, ~19 min once); the
sub-steps build it on first use if missing.

PROTOTYPE orchestrator (subprocess-chains the existing scripts; reloads the cloud per step — fine since
the index/citation-graph are joblib-cached). Productionizing into the main curriculum pipeline (shared
in-process state, the embedding-path output-dir quirk) is backlogged.

Example:
  python -m artifacts_scripts.curriculum.run_concept_curriculum \\
      --toc-json artifacts/book_gaps/jurafsky_toc.json --field 17 --n 25 \\
      --out-dir artifacts/curriculum/jurafsky_concept_pipeline
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bookgap.util import LOGGER, setup_logging

_M = "artifacts_scripts.curriculum"


def _run(mod: str, *cli: str) -> None:
    LOGGER.info("STEP %s %s", mod, " ".join(cli[:8]))
    subprocess.run([sys.executable, "-m", f"{_M}.{mod}", *cli], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toc-json", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--n", type=int, default=25, help="number of chapters (match the book's ToC)")
    ap.add_argument("--cluster", choices=["gcca", "cca", "embedding"], default="gcca",
                    help="gcca = branchless multi-view consensus (DEFAULT: near-best auto-selector, "
                         "retires the per-book embedding-vs-coupling view choice); cca = single-view "
                         "CCA-hybrid oracle; embedding = decoupled-selection + dendrogram")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="concept_pipeline", log_level="INFO", include_time=False)

    concept_dir = args.out_dir / "concept"
    track = args.out_dir / "trackA"

    # 1. keyword-concept seeding from the ToC -> concept_gap.json (soft target-works vector)
    _run("gap_keyword_concept", "--toc-json", str(args.toc_json), "--field", args.field,
         "--backend", "full", "--per-chapter", "--iters", str(args.iters), "--out", str(concept_dir))
    gap = str(concept_dir / "concept_gap.json")

    if args.cluster in ("gcca", "cca"):
        # 2a. GCCA (default) or single-view CCA clustering + citation ordering.
        extra = (["--method", "gcca", "--gcca-variant", "maxvar", "--citation-source", "full_coupling"]
                 if args.cluster == "gcca" else [])
        _run("cluster_concept_cca", "--concept-gap", gap, "--field", args.field,
             "--n", str(args.n), "--out", str(track), *extra)
        # 3. render the judge-parseable summary (cluster_concept_cca writes a terser one)
        _run("render_judge_summary", "--run-dir", str(track), "--field", args.field)
        out = track
    else:
        # 2b. embedding path: decoupled selection (recency) + dendrogram clustering.
        # NOTE: book_curriculum prepends "artifacts/curriculum/" to --output-dir, so the result
        # lands at artifacts/curriculum/<name> (not inside --out-dir). Pass a bare name.
        name = f"{args.out_dir.name}_embedding"
        _run("book_curriculum", "--gap-json", gap, "--field", args.field,
             "--decouple-selection", "--dynamic-topk-floor", "0.05", "--keywords-top-k", "30",
             "--desired-n-topics", str(args.n), "--no-plot", "--output-dir", name)
        out = Path("artifacts/curriculum") / name
        # book_curriculum already writes a judge-parseable summary.

    LOGGER.info("DONE: curriculum -> %s (cluster=%s)", out, args.cluster)
    print(f"curriculum -> {out} (cluster={args.cluster})")


if __name__ == "__main__":
    main()
