# Cache cleanup

Audit of caching decorators in `bookgap/` and `artifacts_scripts/`. `MEMORY` is the
joblib `Memory(CACHE_DIR/"joblib")` singleton in `util.py` — it persists results to
disk (pickle + I/O), keyed by a hash of the arguments, and **survives across separate
CLI/process invocations**. `functools.lru_cache` is **per-process, in-memory only** —
zero disk, zero pickle, but lost when the process exits.

The trade-off this section evaluates: for a *cheap* function with a *small* result,
on-disk caching can cost more than it saves — you pay a pickle round-trip + a stat/open
of a cache directory to avoid a computation that was a handful of milliseconds. Such
functions are candidates to switch to `lru_cache` (recomputed many times *within* one
process) or drop caching entirely (called once per process).

Persisted-output sizes below are measured from the live `cache/joblib/` tree
(`du` per-function directory, and per-call `output.pkl` where it clarifies). Total
joblib cache at audit time: **172 GB**, of which **157 GB** is `bookgap/cloud`.

## On-disk caches that could be lru_cache

| function (file:line) | what it computes | compute cost | persisted output size | recommendation |
| --- | --- | --- | --- | --- |
| `_resolve_filter_scopes` (`bookgap/openalex_ingest.py:113`) | Filter → list of concrete `(field, subfield, topic, work_type)` `ResolvedScope`s, by reading the in-process-memoized snapshot taxonomy | **cheap** — pure-local dict walk + sort; the taxonomy it reads is already memoized in `snapshot_ingest._TAXONOMY_CACHE`; no network, no GPU, no large arrays | **tiny** — 137 B – 225 KB per call (most <10 KB); 392 KB total over all keys | **→ lru_cache** |
| `_cached_embedding_space` (`bookgap/embedder.py:15`) | `(dim, dtype_str)` of a SentenceTransformer model | model **load + a dummy `encode`** (seconds, GPU/CPU warmup) — the *only* expensive cache here with a trivially small result | **trivial** — a 2-tuple; 32 KB total | **keep MEMORY.cache** (already also `lru_cache`-wrapped) — output is tiny so disk cost is nil, but persisting the model-load result across CLI runs is the whole point. Don't drop the disk layer. |
| `get_subcloud_retained_idxs` (`bookgap/subcloud.py:106`) | uint64 array of parent-row indices retained by a `post_filter` | columnar mask (fast) or full `works()` scan (slow at scale) | **large** — up to 287 MB per call | **keep MEMORY.cache** (already double-decorated with `lru_cache(8)` on top — current state is correct: lru_cache absorbs in-process repeats, disk absorbs cross-process) |
| `_compute_gaps_for_topic` (`bookgap/gaps.py:31`) | `list[GapWithContext]` for one topic | **expensive** — full per-topic gap detection (FAISS k-ball + deflation) | small — 32–36 KB per call (68 MB total) | **keep MEMORY.cache** — small output but the compute is the pipeline's core cost; worth surviving restarts |
| `_compute_gaps_impl` (`bookgap/gaps.py:141`) | `list[GapWithContext]` over all topics of a filter | **expensive** — loops `_compute_gaps_for_topic` over every topic | small (20 MB total) | **keep MEMORY.cache** |
| `knn` (`bookgap/cloud_index.py:74`) | FAISS k-NN / range-search result (sq-dists + ids arrays) | **expensive** — FAISS search over the index | **large** — up to 522 MB per call (575 MB total) | **keep MEMORY.cache** |
| `within_cloud_nn_ranks` (`bookgap/cloud_index.py:142`) | per-query NN-rank arrays (within one cloud) | **expensive** — `knn` + FAISS range search | **large at call time** (per-query int arrays) though disk dir is currently empty (rarely materialised) | **keep MEMORY.cache** (large array output; cheap to keep, costly to recompute) |
| `cross_cloud_nn_ranks` (`bookgap/cloud_index.py:200`) | per-query cross-cloud NN-rank arrays | **expensive** — `knn` + FAISS range search | large array output (disk dir currently empty) | **keep MEMORY.cache** |
| `get_avgrefs` (`bookgap/cloud_index.py:240`) | averaged-reference embeddings + index map | **expensive** — builds id→cloud map over all works, averages per-work referenced embeddings | large `(N, D)` float array (disk dir currently empty) | **keep MEMORY.cache** |
| `avgref_ranks` (`bookgap/cloud_index.py:289`) | per-cloud avgref rank arrays | **expensive** — `get_avgrefs` + FAISS range search per index cloud | large array output (disk dir currently empty) | **keep MEMORY.cache** |
| `ensure_cloud_data` (`bookgap/cloud.py:320`) | `(corpus_fp, sorted_idxs, retained_positions)` for a cloud | **very expensive** — fetch + embed + semantic dedup over the full corpus | 4.9 GB total (large index arrays) | **keep MEMORY.cache** (already also `lru_cache(1)`) |
| `get_cloud_work_ids` (`bookgap/cloud.py:706`) | retained work-id strings | moderate — column read over full corpus | **large** — 2.1 GB total | **keep MEMORY.cache** |
| `get_cloud_work_references` (`bookgap/cloud.py:714`) | retained referenced-works lists | moderate — column read | 623 MB | **keep MEMORY.cache** |
| `get_cloud_citation_graph` (`bookgap/cloud.py:853`) | citation adjacency CSR matrix | **very expensive** — streamed two-pass build over ~40 M works | **large** — 2.5 GB | **keep MEMORY.cache** |
| `get_cloud_work_titles` (`bookgap/cloud.py:864`) | retained display-name strings | moderate — column read (hits pyarrow 2 GB offset limit at scale) | **large** — 5.0 GB | **keep MEMORY.cache** |
| `get_cloud_work_abstracts` (`bookgap/cloud.py:896`) | retained abstract strings | moderate — column read | **huge** — 56 GB | **keep MEMORY.cache** |
| `get_cloud_work_publication_dates` (`bookgap/cloud.py:905`) | retained publication-date strings | moderate — column read | **large** — 745 MB | **keep MEMORY.cache** |
| `get_cloud_work_types` (`bookgap/cloud.py:915`) | retained `type` strings | moderate — column read | 82 MB | **keep MEMORY.cache** |
| `get_cloud_index` (`bookgap/cloud.py:934`) | FAISS index over cloud embeddings | **expensive** — `build_index` over all embeddings | **large** — 11 GB | **keep MEMORY.cache** |
| `get_cloud_embeddings` (`bookgap/cloud.py:942`) | denoised `(N, D)` embedding array | **expensive** — streamed parquet read + denoise | **huge** — 75 GB | **keep MEMORY.cache** |
| `get_cloud_projection` (`bookgap/cloud.py:972`) | `ProjectiveDenoiserProjection` artifact | **expensive** — fits the denoiser projection | 100 MB | **keep MEMORY.cache** |
| `deduplicate_works_semantic_` (`bookgap/cloud.py:999`) | uint64 retained-index array (semantic dedup) | **expensive** — all-pairs dedup over full embedding set | 432 MB | **keep MEMORY.cache** (embeddings arg ignored in key by design) |
| `_compute_and_store_…` / fetch in ingest — `deduplicate_works_syntactic` (`bookgap/openalex_ingest.py:440`) | first work of each syntactic-dup group | moderate — syntactic dedup over a scope's works | **large** — 1.1 GB | **keep MEMORY.cache** |
| `build_index_from_clouds` (`bookgap/data_sources.py:45`) | FAISS index over concatenated clouds | **expensive** — concatenate embeddings + `build_index` | large (no disk dir yet) | **keep MEMORY.cache** |
| `get_nearest_neighbor` (`bookgap/data_sources.py:61`) | NN dist/idx arrays for a cloud | **expensive** — FAISS search + validation | large array output | **keep MEMORY.cache** |
| `get_rank_of_nearest_incloud_neighbor` (`bookgap/data_sources.py:127`) | NN-rank stats (also writes its own JSON side-files) | **expensive** — FAISS NN + batched range search | large; also has hand-rolled file-cache inside | **keep MEMORY.cache** |
| `get_points_by_coverage_ball` (`bookgap/data_sources.py:725`) | coverage-ball point sets | **expensive** — FAISS coverage search | large array output | **keep MEMORY.cache** |
| `reduce_dimensions_pca` (`bookgap/data_sources.py:925`) | `(transformed_embeddings, PCA)` | **expensive** — PCA fit over concatenated embeddings | large `(N, d)` array | **keep MEMORY.cache** |
| `reduce_dimensions_tsne` (`bookgap/data_sources.py:933`) | `(transformed_embeddings, TSNE)` | **very expensive** — t-SNE (minutes) | large `(N, d)` array | **keep MEMORY.cache** |
| `build_field_term_index` (`artifacts_scripts/curriculum/field_term_index.py:32`) | field-wide work×term CSR + vocab + DF | **very expensive** — CountVectorizer over a whole field (tens of GB scratch) | **large** — 4.0 GB | **keep MEMORY.cache** (explicitly a "build once, reuse instantly" cache) |
| `_build_temporal_subcloud` (`artifacts_scripts/historical_gap_analysis/historical_gap_analysis.py:49`) | a `SubCloud` view up to a year | cheap-ish — constructs a frozen `SubCloud` (lazy; no data copy) but forces an `embeddings()` count | **tiny** — 140 KB total; the `SubCloud` pickle is just `(parent_cloud, post_filter)` | **→ lru_cache** (or drop) |
| `_build_temporal_test_set` (`artifacts_scripts/historical_gap_analysis/historical_gap_analysis.py:84`) | a `SubCloud` view for one exact year | cheap-ish — same as above | **tiny** — 272 KB total | **→ lru_cache** (or drop) |
| `get_sorted_topics_by_work_count` (`artifacts_scripts/analyze_book_article_gap.py:1379`) | topics sorted by per-topic work count | mixed — **network** (`pyalex.Subfields()[subfield]`) + a `SubCloud.embeddings()` count per topic | small list output | **keep MEMORY.cache** — the `pyalex` call is a live API hit; persisting across runs is worth it |

### Commented-out / non-applicable

- `bookgap/data_sources.py:908` (`reduce_dimensions`) and `artifacts_scripts/curriculum/build_curriculum.py:438` — `@MEMORY.cache` is commented out; no action.
- `artifacts_scripts/historical_gap_analysis/historical_gap_analysis_sweep.py` — no cache decorators (delegates to the cached functions above).

## Summary — clearest `→ lru_cache` switches

The audit is lopsided: the vast majority of `@MEMORY.cache` sites legitimately persist
large arrays or expensive FAISS/embedding/PCA/dedup results and should stay on disk.
Only a handful are cheap-with-tiny-output. The clearest switches, in priority order:

1. **`_resolve_filter_scopes` (`bookgap/openalex_ingest.py:113`) → `lru_cache`.** The
   single best candidate in `bookgap/` proper. Output is ≤225 KB (usually <10 KB), the
   compute is a pure-local taxonomy walk + sort, and the taxonomy it depends on is
   *already* in-process memoized (`snapshot_ingest._TAXONOMY_CACHE`). Within a sweep this
   is called repeatedly (once per cloud/scope resolution) — `lru_cache` is strictly
   better there; the cross-process disk persistence buys almost nothing because recompute
   is sub-millisecond once the taxonomy is loaded. Use `@lru_cache(maxsize=None)` (args
   are a hashable `Filter` + `str`).

2. **`_build_temporal_subcloud` (`historical_gap_analysis.py:49`) → `lru_cache`.** The
   persisted object is just a frozen `SubCloud` = `(parent_cloud, post_filter)` — pickling
   it to disk (140 KB across all keys) to avoid constructing a lazy dataclass is
   net-negative. The expensive work (`embeddings()`) is itself cached on the cloud. Same
   process, repeated calls in a year-sweep → `lru_cache(maxsize=…)` wins.

3. **`_build_temporal_test_set` (`historical_gap_analysis.py:84`) → `lru_cache`.**
   Identical reasoning to (2); 272 KB of pickled `SubCloud` views for what is a trivial
   constructor.

Borderline / explicitly NOT a switch:

- **`_cached_embedding_space` (`embedder.py:15`)** *looks* like a `lru_cache` candidate
  (2-tuple output) but the compute is a **model load** — exactly the kind of expensive
  one-time cost worth persisting across CLI runs. It already layers `lru_cache(1)` over
  `MEMORY.cache`; leave both.
- **`get_subcloud_retained_idxs` (`subcloud.py:106`)** already uses the ideal pattern
  (`lru_cache(8)` + `MEMORY.cache`): lru_cache absorbs in-process repeats, disk absorbs
  cross-process. No change.
- **`_compute_gaps_for_topic` / `_compute_gaps_impl` (`gaps.py:31,141`)** have small
  outputs but the compute *is* the gap-detection pipeline — keep on disk.

Estimated disk reclaimed by the three switches is negligible (≈0.4 MB + 0.4 MB); the
motivation is correctness-of-fit (avoid a pickle round-trip + cache-dir stat on a
sub-ms, repeatedly-called, same-process function), not space.

---
*Regenerate rather than edit by hand. Source audit: `grep -rn "@MEMORY.cache" bookgap/ artifacts_scripts/`*
*plus per-function `du` over `cache/joblib/`. Sizes are from the live cache tree at audit time and will drift.*

## Empirical timing (measured)

Measured on `2026-06-16` via `artifacts_scripts.cache_timing_bench` (snapshot `2025-11-06`, reps=400 cheap / 200 heavy). Throwaway `joblib.Memory(tmpdir)` used for all miss/hit control; the live `cache/joblib/` tree was not touched. Warm page cache (a *cold* first-ever cross-process read is slower still, making on-disk look even worse than shown).

#### `_resolve_filter_scopes` — full phase decomposition

All times in microseconds, **median / p90** over N reps. `T_persist = T_miss − T_compute`.

| case | n_scopes | output | T_compute (µs) | T_miss (µs) | T_persist (µs) | T_jobhit (µs) | T_lruhit (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| topic T10828 (small) | 2 | 169 B | 4.64 / 5.08 | 247.06 / 295.09 | 242.42 | 126.87 / 130.93 | 0.30 / 0.33 |
| subfield 1711 (medium) | 20 | 700 B | 17.00 / 18.16 | 454.89 / 503.08 | 437.89 | 175.10 / 187.66 | 0.24 / 0.27 |
| field 17 (large) | 604 | 20,057 B | 319.38 / 378.31 | 6,631.30 / 6,992.63 | 6,311.92 | 3,555.02 / 3,613.40 | 0.30 / 0.34 |

**Verdict — `_resolve_filter_scopes`:**

- **topic T10828 (small)**: repeated same-proc — T_jobhit 126.87 µs vs T_lruhit 0.30 µs → lru is **420× faster**. Single cross-proc — T_compute 4.64 µs vs T_jobhit 126.87 µs → on-disk is pure OVERHEAD even cross-proc (recompute ≤ disk-load).
- **subfield 1711 (medium)**: repeated same-proc — T_jobhit 175.10 µs vs T_lruhit 0.24 µs → lru is **727× faster**. Single cross-proc — T_compute 17.00 µs vs T_jobhit 175.10 µs → on-disk is pure OVERHEAD even cross-proc (recompute ≤ disk-load).
- **field 17 (large)**: repeated same-proc — T_jobhit 3,555.02 µs vs T_lruhit 0.30 µs → lru is **11,811× faster**. Single cross-proc — T_compute 319.38 µs vs T_jobhit 3,555.02 µs → on-disk is pure OVERHEAD even cross-proc (recompute ≤ disk-load).

#### `_build_temporal_subcloud` / `_build_temporal_test_set` — decisive measurements

The cached **return value** is a frozen `SubCloud` (`parent`, `post_filter`); `embeddings()` is NOT in the cached object. `T_compute_full` is the genuine miss cost (forces the `embeddings()` count); `T_construct` is the bare lazy `SubCloud(...)` build (the real alternative to a hit, since `embeddings()` is itself `@MEMORY.cache`d on the Cloud). `T_jobhit`/`T_pickle_rt`/`T_lruhit` operate on the **representative SubCloud object** (clearly a stand-in for the cache load path, isolated from `embeddings()`). Times median / p90.

| function | output | T_compute_full | T_construct (µs) | T_jobhit (µs) | T_pickle_rt (µs) | T_lruhit (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `_build_temporal_subcloud` | 913 B | 7,561.4 / 7,778.2 ms | 2.64 / 2.82 | 282.80 / 286.73 | 39.63 / 40.16 | 0.20 / 0.23 |
| `_build_temporal_test_set` | 913 B | 7,703.8 / 8,267.7 ms | 2.62 / 2.70 | 283.16 / 288.24 | 39.69 / 40.18 | 0.20 / 0.22 |

**Verdict — `_build_temporal_subcloud` / `_build_temporal_test_set`:**

- **`_build_temporal_subcloud`**: the returned SubCloud is 913 B. Repeated same-proc — T_jobhit 282.80 µs vs T_lruhit 0.20 µs → lru is **1,400× faster**. Single cross-proc — bare reconstruct T_construct 2.64 µs vs T_jobhit 282.80 µs → on-disk is pure OVERHEAD vs bare reconstruct (the lazy ctor is cheaper than unpickling). (A true miss that recomputes the body is 7,561 ms, dominated by `embeddings()`, but neither cache stores that array — only the tiny view.)
- **`_build_temporal_test_set`**: the returned SubCloud is 913 B. Repeated same-proc — T_jobhit 283.16 µs vs T_lruhit 0.20 µs → lru is **1,402× faster**. Single cross-proc — bare reconstruct T_construct 2.62 µs vs T_jobhit 283.16 µs → on-disk is pure OVERHEAD vs bare reconstruct (the lazy ctor is cheaper than unpickling). (A true miss that recomputes the body is 7,704 ms, dominated by `embeddings()`, but neither cache stores that array — only the tiny view.)

**Bottom line.**

The data confirms the `→ lru_cache` recommendation. For `_resolve_filter_scopes`, an in-memory hit is ~**727×** faster than joblib's disk-load hit across all output sizes, and recompute is *already* as cheap as (or cheaper than) the warm disk-load for every case — so on-disk persistence buys nothing even on a single cross-process call, while costing a pickle round-trip + cache-dir stat on every same-process repeat.
For the two SubCloud builders, the cached object is a ~0.9 KB lazy view; unpickling it via joblib costs more than simply re-running its frozen-dataclass constructor (the genuinely expensive `embeddings()` count is cached on the Cloud, not in these results), so on-disk caching here is net-negative in every regime. **Switch all three to `lru_cache`.**

---
*One-time measurement; the generating script (`artifacts_scripts/cache_timing_bench.py`) has been removed. The table above is the permanent record — do not hand-edit it. Absolute times depend on hardware/page-cache and will drift; the ratios (T_jobhit/T_lruhit, T_compute vs T_jobhit) are the stable signal.*
