# Wire keyword detection into `batch_generator` so every gap carries its keywords

## Idea

Make per-gap keyword extraction a first-class part of the official
gap-detection pipeline: every `GapWithContext` produced by
`_compute_gaps_for_topic` should carry a `keywords` field
(`list[tuple[str, float]]`) populated by `compute_gap_keywords` from
`artifacts_scripts/_keyword_helpers.py`. Today keywords are computed
ad-hoc in research scripts (e.g.
`artifacts_scripts/per_topic_leaderboard.py`) but the LLM proposal
generator never sees them.

## Why

- The LLM proposal context could use the keyword set directly
  (cheap, structured hint about what the gap is *about*) instead of
  rederiving from titles/abstracts every prompt.
- Downstream analysis (leaderboards, dashboards, evaluation) keeps
  reinventing the per-gap keyword pipeline. Lifting it once into the
  Cloud-native pipeline removes that duplication.
- Keywords are the cleanest signal for filtering templated-leak
  clusters (cf. `gap_book_existence_filter.py`'s reliance on keyword
  Jaccard) — making them part of the gap object enables the filter
  step to be a tiny consumer rather than a re-extraction.

## Sketch

- Add `keywords: list[tuple[str, float]] | None = None` to
  `GapWithContext` (alongside the LLR-refined optional fields if
  that integration has landed by then).
- In `_compute_gaps_for_topic`, after the algorithm yields its gap
  list, batch-call `compute_gap_keywords` once over all per-topic
  gaps (joint phrase mining gives them a shared vocab, per
  `_keyword_helpers.compute_gap_keywords` docstring) and attach
  results to each `GapWithContext`. This is one extra step inside
  the existing cache boundary, so re-runs reuse cached keyword
  output.
- The CLI knob for `--top-keywords` becomes a `KeywordConfig`
  dataclass field referenced from `GapConfig` (or its sibling
  `RefinementConfig`).

## Cross-pollination: what else in `artifacts_scripts/curriculum/`
should be upstreamed?

`artifacts_scripts/curriculum/` has accumulated a fair amount of
infrastructure that started as research scripts but now duplicates
or extends what the Cloud-native pipeline does. Worth a sweep:

- `keyword_extraction.py` — likely the canonical home of the
  phrase-mining + log-odds machinery that `_keyword_helpers.py` calls
  into. If so, the official-pipeline integration above probably
  shouldn't import `artifacts_scripts._keyword_helpers` directly —
  the `_helpers` shim should be pulled inside `bookgap/`.
- `multi_view_spectral.py`, `dimensionality_reduction.py`,
  `pagerank.py`, `circuitrank.py`, `similarity_graph.py` — these
  are graph / spectral methods written for curriculum building.
  Some primitives (e.g. similarity-graph construction over a
  SubCloud, PageRank-on-citation-graph) may have natural homes
  alongside `bookgap.cloud_index` / `bookgap.subcloud`.
- `mass_statistics.py`, `sampling.py`, `search_cache.py` — utility
  shapes that may belong in `bookgap.util` or a new
  `bookgap.statistics` module if the curriculum pipeline ends up
  sharing them with the main batch generator.
- `summarize_topics.py`, `output_summary.py` — display / artifact-
  formatting helpers. If we standardize on a unified leaderboard
  output format across `per_topic_leaderboard.py` and curriculum
  summaries, there's likely shared code to factor out.

Scope of the cross-pollination question: do **not** unilaterally
refactor everything. The right thing is a one-shot audit pass that
identifies which functions are (a) used only by curriculum, (b) used
by both curriculum and at least one main-pipeline script, and (c)
plausibly useful to the main pipeline but currently only used by
curriculum. Items in (b) and (c) are upstream candidates; items in
(a) stay where they are.

## Not in scope

- Replacing `compute_gap_keywords` with a different algorithm —
  this entry is purely about wiring up the existing one.
- LLM-side prompt redesign — once keywords are available on
  `GapWithContext`, the prompt template can be updated separately.

## Related

- `_keyword_helpers.compute_gap_keywords` — current entry point.
- `artifacts_scripts/curriculum/keyword_extraction.py` — likely
  canonical phrase-mining home.
- `artifacts_scripts/per_topic_leaderboard.py` — research consumer
  that proves the integration shape works.
- `bookgap.llm.data_structures.GapWithContext` — destination type.
