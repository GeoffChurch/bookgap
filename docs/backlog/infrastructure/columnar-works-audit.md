# Columnar audit of `cloud.works()` callers

`cloud.works()` loads **all** columns of every retained work into a Python list of dicts (`List[Dict[str, Any]]`). For the AI subfield (~1.6M works), that's ~6-10 GB of Python objects (titles, abstracts, author lists, referenced_works, …) held in memory. It's `@lru_cache(maxsize=1)` so the cost is paid once per process, but it's still the dominant memory hit at startup and the eager I/O blocks process spin-up.

Many callers only need 1-3 columns. `cloud.py` already exposes columnar accessors that read just the needed slice:

- `cloud.work_ids()` → only `"id"`
- `cloud.work_publication_dates()` → only `"publication_date"`
- `cloud.work_titles()`, `cloud.work_abstracts()`, `cloud.work_references()` (each reads only its column)

## TODO

Walk every `cloud.works()` (and `parent.works()`) call site and check whether it actually needs the full dicts, or just a handful of columns. There are ~50 call sites across the repo; concentrated in `bookgap/data_sources.py` (39), `artifacts_scripts/historical_gap_analysis/historical_gap_analysis.py` (4), `bookgap/subcloud.py` (2), `bookgap/llm/generators/workflow_functions.py` (2), and a few others.

### Highest-impact known site

`bookgap/subcloud.py:60-66` — `get_subcloud_retained_idxs`:

```python
@MEMORY.cache
def get_subcloud_retained_idxs(subcloud: SubCloud) -> NDArray[np.uint64]:
    return np.array(
        [i for i, work in enumerate(subcloud.parent.works())
         if subcloud.post_filter.check_work(work)],
        dtype=np.uint64,
    )
```

This iterates 1.6M Python dicts even when `post_filter` only inspects 2-3 fields (typical: `topic`, `type`, `publication_date`). For known-structured filters (`Filter`, `TemporalFilter`), a vectorized columnar path could:
1. Read only the columns the filter inspects via `_read_corpus_column`.
2. Build the boolean mask with numpy/pandas vectorization.
3. Convert to `retained_idxs` in O(n) ndarray ops.

Per-year filter cost in the historical_gap_analysis sweep drops from ~5-10s to <100ms with this change.

### Design sketch

Either:
- **(a)** Add a `Filter.required_columns() -> Optional[FrozenSet[str]]` method (default `None` = "I need everything"). Subclasses like `TemporalFilter` declare `frozenset({"publication_date", "type", "id"})`. `get_subcloud_retained_idxs` checks for `required_columns` and takes a fast columnar path when it's a finite set. Cleanest, generalizes to any structured filter.
- **(b)** Special-case `TemporalFilter` (and any other known filter classes) with explicit fast paths. Simpler but doesn't generalize.

Option (a) parallels the existing `required_corpus_columns` pattern on denoisers (see `NoneDenoiser.required_corpus_columns: ClassVar[Optional[FrozenSet[str]]] = frozenset()` in `bookgap/cloud_denoiser.py:100`), so prior art exists.

## Why this matters now

- The historical_gap_analysis in-process sweep is the visible immediate motivator, but the over-fetch affects any startup that goes through SubCloud filtering.
- On the 40 M-retained-work domain=3 curriculum corpus, the full `cloud.works()` load is ~40 GB of Python strings plus a transient ~30 GB pyarrow buffer — already documented in `NoneDenoiser` as an OOM trigger. The same trap probably exists for several callers that don't really need the full dicts.

## Related

- `bookgap/cloud_denoiser.py:91-100` — `NoneDenoiser.required_corpus_columns = frozenset()` is the existing pattern this would extend to filters.
- `bookgap/cloud.py:540-580` — `_read_corpus_column` is the per-column reader to reuse.
