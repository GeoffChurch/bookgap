# Refactor K-ball to GPU brute-force kNN + manual radius filter

(Migrated from `artifacts_scripts/curriculum/TODO.md`.)

faiss-gpu's `GpuIndexFlatL2` is 965× faster than CPU FlatL2 on 200K × 384 (smoke test 2026-04-29: 6.4 ms vs 6173 ms for 1000 queries × k=10).

Replace `kballs_with`'s HNSW range-search with:
1. GPU brute-force kNN with K ≥ max-expected-neighbors
2. Post-filter by squared-distance ≤ threshold

At K-ball scope scale this would be substantially faster than CPU HNSW.

**Note**: this is the path we'd have to take with USearch too — USearch and faiss-gpu both lack native GPU range_search at the python-API level, so the kNN+filter emulation is unavoidable.

**Open questions**:
- K_max sizing — fixed conservative value vs adaptive binary expansion
- Per-topic GPU memory management on 48GB GPU (77 topics × ~150MB = 11GB if all loaded; manageable)
- Graceful fallback to CPU HNSW for small topics where GPU overhead doesn't pay off

**Cost**: refactor `bookgap/index.py::kballs_with` and adjust callers in `bookgap/llm/gap_algorithms/kball_algorithm.py`.

**When**: if probe-sweep wall-clock becomes a real obstacle for iteration. Given that HNSW already gives 5-9× speedup over FlatL2 for `gap_article_as_seed.py`, the urgency is reduced.

**Connection**: USearch evaluation as a faiss-gpu replacement was already done (`artifacts_scripts/usearch_smoke_test.py`, 2026-04-29) — verdict: not a path to GPU range_search; CPU performance ~25% slower than FAISS HNSW at our scale.
