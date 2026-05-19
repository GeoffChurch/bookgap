# Held-out book recovery as a production-ready eval (paper-table artifact)

## Idea

Promote the held-out-real-books rediscovery eval from the bake-off
harness into a **production-grade, standalone eval** in `bookgap/`
that runs against a configured production pipeline run and emits a
clean LaTeX/CSV table artifact for the paper.

Recommended shape: a **separate eval entrypoint**, not a metric
computed inside every batch run (the eval needs a 20%-book mask
applied *before* detection, so it's a distinct run mode, not a
post-hoc score on a normal run). Think
`bookgap.eval.heldout_recovery` (or an `artifacts_scripts`-style
table generator but production-grade and importable from `bookgap/`):
input = an `ExperimentConfig` (incl. `gap.deflation_strategy`);
it masks 20% of books (fixed seed), runs `compute_gaps_for_config`,
computes the metric over the ranked gaps, writes a topic × method
table.

## Hard requirement (carry the lesson)

**Genuine recall@K only.** The bake-off's `heldout_book_recall_at_20`
was a full-survivor-set misnomer (rank-insensitive, rewarded gap
quantity) until `303ae02`; it inverted an entire investigation's
conclusions (`docs/gap_deflation_results.md`,
`docs/dead_ends/cohesion-metric-design/
deflation-demand-engineering-metric-artifact.md`). The production
eval MUST slice `ranked[:K]` and report it AS recall@K, and SHOULD
also emit the full-set value as a clearly-named separate column (the
`fullset − @K` delta is itself a publishable "coverage breadth vs
top-ranked quality" signal). Reuse the validated logic
(`heldout_book_split` / `heldout_recall_spread` + the `[:K]` slice),
not the pre-fix path.

## Why it matters

It's the paper's one objective external axis ("does this rediscover
where humans actually wrote books, non-redundantly"). A clean,
reproducible table generator means the paper can report it directly
instead of hand-extracting from bake-off logs. Triangulation, not
oracle — absolute recall@K is low (0.10–0.25); pair with
qualitative + the deferred ollama LLM-coverage check. See the living
tracker `book-rediscovery-eval-candidates.md` for the candidate
methods/topics and the standing caveats.
