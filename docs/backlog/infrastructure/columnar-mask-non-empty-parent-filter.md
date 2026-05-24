# Columnar fast path for non-empty parent Filters

## Context

`TemporalFilter.compute_columnar_mask` (in `artifacts_scripts/historical_gap_analysis/historical_gap_analysis.py`) currently returns `None` — fall back to dict iteration — whenever the inherited base `Filter` has any of `domain` / `field` / `subfield` / `topic` / `work_types` set. This was the safe choice for the historical-gap-analysis sweep where the parent post-filter is empty (`FilterConfig()` with all fields `None`).

For other callers (anyone constructing a `TemporalFilter` with a non-trivial parent `Filter`), the fast path silently degrades and we pay the full ~6-10 GB `cloud.works()` materialisation.

## TODO

Extend the columnar fast path to also handle the base `Filter` fields:

- `subfield` → read `primary_subfield_id` column and compare to `f"https://openalex.org/subfields/{self.subfield}"`
- `topic` → read `primary_topic_id` column and compare to `f"https://openalex.org/{self.topic}"`
- `work_types` (the parent's, distinct from TemporalFilter's `work_type`) → read `type` column and check membership against `self.work_types`
- `domain` / `field` — not present as flat columns in the parquet schema; would need taxonomy resolution. Defer.

## Implementation sketch

1. Add `Cloud.work_primary_topic_ids()` and `Cloud.work_primary_subfield_ids()` accessors in `bookgap/cloud.py` (analogous to `get_cloud_work_types`).
2. Move the columnar implementation from `TemporalFilter` into base `Filter.compute_columnar_mask`, vectorising whatever subset of (domain, field, subfield, topic, work_types) is non-None.
3. Have `TemporalFilter.compute_columnar_mask` first call `super().compute_columnar_mask(cloud)` (which may itself return a mask or `None`), then AND in the temporal-specific constraints.
4. Keep the `domain`/`field` fall-back: if either is set and we don't have a vectorised path for it, return `None`.

## Why this matters

`subcloud.get_subclouds_for_topic` (in `bookgap/subcloud.py:74-78`) builds two SubClouds per topic with `Filter(topic=topic_id, work_types=("book",))` and `Filter(topic=topic_id, work_types=("article",))`. These are called per-topic in every gap-detection run. Today each invocation hits the dict-iteration slow path because the base `Filter` doesn't override `compute_columnar_mask`. Wiring this up would speed up production gap detection too, not just the historical analysis.

## Tests

The existing `tests/test_columnar_mask.py` already has scaffolding for `_MockCloud` with `work_publication_dates()` and `work_types()`. Extend it with `work_primary_topic_ids()` / `work_primary_subfield_ids()` mocks and add equivalence tests for the new vectorised paths.

## Related

- `docs/backlog/infrastructure/columnar-works-audit.md` — parent backlog entry
- `bookgap/util.py:Filter.compute_columnar_mask` — the hook to extend
- `tests/test_columnar_mask.py` — existing test scaffolding
