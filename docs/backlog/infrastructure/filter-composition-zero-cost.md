# filter-composition-zero-cost

Refactor `bookgap.filters` into composable atoms (`MaxYearFilter`,
`ExactYearFilter`, `WorkTypeFilter`, …) plus boolean composition
operators (`AndFilter`, `OrFilter`, `NotFilter`). The architecture
already supports this; the refactor is mostly mechanical.

## Why

`TemporalFilter` today bundles "filter by year" and "filter by type"
into one class — the name understates its scope, and reuse across
either dimension requires copy-pasting. Composition would let callers
write `AndFilter([MaxYearFilter(2010), WorkTypeFilter(BOOK)])` and get
the same retention with cleaner semantics. The pattern generalises to
any future predicates we want to layer on.

## When

Pays off once we have ≥3 concrete `Filter` subclasses that would
benefit. With just `TemporalFilter` today, the immediate ROI is small.
Revisit when a second cloud-aware filter shows up.

## Sketch

```python
@dataclass(frozen=True)
class MaxYearFilter(Filter):
    max_year: int
    def compute_columnar_mask(self, cloud):
        years = cloud.work_publication_years()  # cached, see prereq
        return years <= self.max_year

@dataclass(frozen=True)
class WorkTypeFilter(Filter):
    work_type: WorkType
    def compute_columnar_mask(self, cloud):
        types = cloud.work_types()  # already cached
        return types == self.work_type.value  # or with article/preprint conflation logic if kept

@dataclass(frozen=True)
class AndFilter(Filter):
    filters: tuple[Filter, ...]
    def check_work(self, work):
        return all(f.check_work(work) for f in self.filters)
    def compute_columnar_mask(self, cloud):
        masks = [f.compute_columnar_mask(cloud) for f in self.filters]
        if any(m is None for m in masks):
            return None  # fall back to per-work iteration
        return np.logical_and.reduce(masks)
```

## Prerequisite

Atomic filters must read from **cached** column accessors on Cloud, or
composition pays the derivation cost N times. Today
`cloud.work_publication_dates()` is `@lru_cache + @MEMORY.cache`d, but
the year-extraction list comprehension inside
`TemporalFilter.compute_columnar_mask` is not. Promote it:

```python
@lru_cache(maxsize=1)
@MEMORY.cache
def get_cloud_work_publication_years(cloud: Cloud) -> NDArray[np.int32]:
    return np.array([
        -1 if not d else (_extract_year(d) or -1)
        for d in get_cloud_work_publication_dates(cloud)
    ], dtype=np.int32)
```

Then `MaxYearFilter` and `ExactYearFilter` can both call it for free
after the first invocation.

## Runtime cost analysis (once derivations cached)

- `check_work` composition: 2 method calls + 1 bool op per work — same
  order of magnitude as a single filter's check.
- `compute_columnar_mask` composition: one numpy bitwise AND per pair
  of masks. O(n_works), vectorised. Essentially free at our scale
  (~1.5M works → microseconds).

## Open questions to settle when picking this up

1. Should `AndFilter`/`OrFilter` short-circuit on the fast path? (If
   any child returns `None`, we already fall back. Could we instead
   take a hybrid: skip mask-returning children entirely and only run
   `check_work` on the dict-iterating ones? Probably more complex than
   it's worth.)
2. Keep `TemporalFilter` as a back-compat alias that constructs the
   right `AndFilter`, or deprecate cleanly?
3. Where does the article/preprint conflation belong (currently
   hardcoded in `TemporalFilter`)? Probably its own
   `ArticleOrPreprintFilter`, or a `WorkTypeFilter` flag.
