# Audit `get_subclouds_for_topic` + other SubCloud users for columnar opportunities

## Context

The first concrete win from the `Filter.compute_columnar_mask` hook (added 2026-05) landed in `TemporalFilter` for the historical-gap-analysis sweep. The hook itself is general, but every other SubCloud construction in the codebase still takes the dict-iteration slow path — `subcloud.parent.works()` materialises ~6-10 GB of work dicts for filters that only inspect 2-3 columns.

## Highest-impact known site

`bookgap/subcloud.py:74-78`:

```python
def get_subclouds_for_topic(cloud: Cloud, topic_id: Optional[str]) -> tuple[SubCloud, SubCloud]:
    books = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=("book",)))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=topic_id, work_types=("article",)))
    return books, articles
```

This is called once per topic per gap-detection run. For the 77-topic AI subfield, that's 154 SubCloud constructions per run; each currently iterates 1.6M Python dicts before retaining only the topic's ~10k works. The base `Filter` doesn't override `compute_columnar_mask`, so today everything hits the slow path.

Blocked on `columnar-mask-non-empty-parent-filter.md` (vectorising base `Filter` fields).

## Audit checklist

Walk the ~50 `cloud.works()` and `parent.works()` callers across the repo (see `docs/backlog/infrastructure/columnar-works-audit.md` for the file-count breakdown) and tag each:

- **Needs full dicts** (titles, abstracts, references, authors) — leave alone, columnar can't help.
- **Inspects only structured fields** (topic, subfield, type, publication_date, id) — wire `compute_columnar_mask` once the base `Filter` supports it.
- **Could be refactored to columnar accessors directly** (e.g. just needs the id list — use `cloud.work_ids()`).

Concentration: `bookgap/data_sources.py` has 39 of the 50 call sites — start there. Many are likely doing iteration that could be vectorised.

## Expected speedup

For gap-detection runs on the AI subfield (1.6M corpus, 77 topics), the per-topic SubCloud construction currently takes ~5-10s each via dict iteration → 154 × 7s ≈ 18 min of overhead per run. Columnar reads should drop this to <1 min total.

For the full-domain curriculum corpus (40M retained works), the dict materialisation is a documented OOM trigger (~40 GB Python strings + 30 GB pyarrow buffer; see `bookgap/cloud_denoiser.py:91-100`). Columnar reads would dodge this entirely.

## Related

- `docs/backlog/infrastructure/columnar-works-audit.md` — parent backlog entry, full audit scope
- `docs/backlog/infrastructure/columnar-mask-non-empty-parent-filter.md` — prerequisite
- `tests/test_columnar_mask.py` — test scaffolding to extend
