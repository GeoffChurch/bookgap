# Gap-detection method development guide

How to prototype, validate, and compare a new gap-detection method. Captures the conventions and feedback loop developed during the cohesion-aware gap-detection investigation.

## Reference implementations

When writing a new gap-detection script, copy from one of these and adapt:

- **`artifacts_scripts/gap_article_as_seed.py`** — article-as-seed cohesion expansion with HNSW K-NN. End-to-end pipeline: load scope → build indices → compute σ → iterate seeds → exact-match dedup → output ranked CSV. Reference for any *hard-membership seeded-expansion* method (`--batch-addition` flag = ordered cumulative ADD/REMOVE; `--with-book-seeds` = Phase 3 books-as-seeds-and-members).
- **`artifacts_scripts/gap_soft_em.py`** — soft-EM expansion preserving the local CSLS-balanced boundary. Replaces the Viterbi-style hard-threshold inner loop with continuous posterior weight updates; books participate as members but are not seeded. Reference for any *fuzzy-membership* method. Drop-in compatible with `gap_multi_scale_coverage.py` and `gap_method_comparison_table.py`. Slower per-seed than the hard-membership pipeline (vectorized matmul over candidate set); validated on T11303, T10181 still needs scale work. See `docs/coverage_design_tradeoffs.md` for the empirical tradeoffs.
- **`artifacts_scripts/gap_multi_scale_coverage.py`** — post-hoc multi-scale coverage labeling. Reads a gap CSV (any seeded-expansion output), evaluates each non-book-anchored basin's articles against each book-anchored basin's `compute_state`, assigns 7-tier coverage labels (`book-anchored / book-exists / inherited-{strong,moderate,weak} / real-gap / templated-leak`) plus the continuous `inherited_coverage_median_post` score. Reference for any *coverage-detection labeling pass*.
- **`artifacts_scripts/gap_book_existence_filter.py`** — vote-concentration + title-overlap heuristic for "is there a textbook that covers this basin?" Reference for any *heuristic flag* on basin CSVs.
- **`artifacts_scripts/gap_books_inspector.py`** — qualitative inspection: prints sample article titles + nearest-book lookup per gap. Reference for any *inspection / sanity-check* tool. Takes a gap CSV and prints human-readable output.
- **`artifacts_scripts/gap_method_comparison_table.py`** — cross-method comparison. Loads two gap CSVs, runs log-odds keyword extraction with iterative bigram phrase mining, computes top-vote books per gap, emits a side-by-side markdown table. Reference for any *cross-method comparison* artifact.

Older / probe-only scripts (not reference implementations): `gap_effective_size*.py`, `gap_kball_cohesion_curve.py`, `gap_topn_comparison.py`, `gap_correspondence.py`. These were diagnostic during the cohesion investigation; future variants should NOT copy from them.

## Conventions

### CSV input/output schemas

**Gap CSVs**: each row is one gap. Required columns:

| column | description |
|--------|-------------|
| `rank` | 1-indexed position in sorted output (by `logit_sum` typically) |
| `<method>_gap_id` (or `seed_article`) | identifier within the method (K-ball gap_idx; max-bookless gap_id; article-seed seed article index) |
| `final_size` (or `gap_size`) | number of articles in the gap |
| `logit_sum` | Σ log(post / (1−post)) over gap members; canonical Bayes-evidence score |
| `mean_post` | mean posterior; gap "purity" |
| `min_post` | minimum posterior; the cluster's weakest member's confidence |
| `article_indices_json` | JSON list of scope-global indices of gap members |

For K-ball / max-bookless legacy CSVs, `gap_size` and `gap_idx` instead of `final_size` and `seed_article`. Reference scripts handle both.

### Naming conventions

```
artifacts/<method>/<topic-id>__<config>.csv
```

- `<method>`: `kball_gap_probe`, `maximal_bookless`, `gap_article_as_seed`, `gap_kball_cohesion`, etc.
- `<topic-id>`: e.g., `17-1702-T10181` (K-ball uses `field-subfield-topic`); `T11303` (single-topic shorthand).
- `<config>`: brief encoding of method-specific knobs, e.g., `pca__drop0` (denoiser+drop_k for K-ball), `kball__rf0.75__num200` (K-ball with radius_fraction and num_gaps).

State directories use the same scheme without the `.csv` suffix: `artifacts/hdbscan_state/T10181_no_denoise/`.

### Indexing convention

All gap article indices are **scope-global** — positions in the array returned by `load_scope(scope)`. NOT per-topic local indices. This makes article membership directly comparable across K-ball, max-bookless, article-as-seed, and any new method that consumes `load_scope` output.

When implementing a new K-ball-style method that operates on per-topic article-only embeddings, convert to scope-global before writing CSV: `scope_idx = article_global_idx[local_idx]`.

## Validation order

Validate new methods on progressively larger scopes to catch issues early:

| Topic | Articles | Books | Approx wall (HNSW backend) | Use for |
|-------|----------|-------|------------------------------|---------|
| T11303 (Bayesian Networks) | 14K | 145 | ~40 sec | Quick correctness check; iterate quickly here |
| T10181 (NLP) | 88K | 1,252 | ~10 min | Realistic scale; matches paper's primary scope |
| T13734 (Applied AI) | 142K | 353 | ~65 min | Stress test; book-poor regime where natural cluster sizes are large |

Run on T11303 first; only proceed to T10181 once correctness and timing look right. T13734 is optional and only needed when the method needs cross-topic generalization validation.

## HNSW K-NN backend

Default for full-corpus index in any new method:

```python
from bookgap.index import build_index, FlatL2, HNSW

# full-corpus index (used in get_knn() callbacks)
efs = max(args.k_query + 50, args.k_sigma + 20)  # efSearch ≥ K_query
full_index = build_index(HNSW(hnsw_m=32, hnsw_efc=400, hnsw_efs=efs), emb)

# book and article sub-indices stay Flat (smaller, batched calls fine)
book_index = build_index(FlatL2(), emb[book_global_idx])
article_index = build_index(FlatL2(), emb[article_global_idx])
```

**Critical**: `efs ≥ K_query` so HNSW returns enough neighbors to match Flat. With `efs < K_query`, HNSW silently truncates and your scores diverge from Flat's. Empirical: 5-9× speedup over FlatL2 on T10181/T13734 with identical gap counts when `efs` is set correctly.

## Progress reporting pattern

Long-running expansions need observable progress. Standard pattern:

```python
import time
t_start = time.time()
for seed_idx, seed in enumerate(seeds):
    # Progress print at every 1000-seed boundary, BEFORE any short-circuits
    if (seed_idx + 1) % 1000 == 0 or seed_idx == len(seeds) - 1:
        elapsed = time.time() - t_start
        rate = (seed_idx + 1) / max(elapsed, 0.01)
        eta = (len(seeds) - seed_idx - 1) / max(rate, 0.01)
        print(f"  processed {seed_idx + 1}/{len(seeds)} seeds in {elapsed:.0f}s "
              f"({rate:.0f}/s, eta {eta:.0f}s); "
              f"discovered {n_unique} gaps; "
              f"<other counters>",
              flush=True)
    # ... rest of loop including continues / short-circuits
```

**Critical**: the progress print must be at the TOP of the loop, before any `continue` short-circuits. Otherwise once the early-skip filter starts firing reliably (e.g., after most seeds become claimed), the progress print stops triggering and you lose visibility.

### Buffering pitfalls

- `python -u` makes Python's stdout unbuffered — required for long-running scripts.
- `> /tmp/file.log 2>&1` redirects stdout/stderr cleanly. Conda run buffers ~4-8KB but flushes on EOL when output is unbuffered.
- `| tail -N` is **block-buffered** and won't show output until process exit. Don't use during long runs; redirect to a file instead.
- For real-time monitoring: launch with `run_in_background: true` redirecting to a file, then use `Monitor` with a `tail -F | grep --line-buffered` filter to receive notifications on key events.

## Inspector pattern

For any new gap method, validate qualitatively by reading article titles and nearest books per gap. Use `gap_books_inspector.py` directly if your output schema matches; otherwise adapt it.

Pattern:
1. Load scope (titles, embeddings)
2. Build a small FlatL2 index over books
3. For each top-N gap: print 5-10 sample article titles; for each gap article, find its 1-NN book and tally; print top-3 books with their vote counts.

Vote concentration on a topically-aligned book = "topic exists, has a book." Vote concentration on a random book = corpus-QC artifact (templated data leak). Low concentration = diffuse subfield. See [vote-concentration backlog](backlog/cohesion-metric-design/vote-concentration-as-book-exists-filter.md).

## Cross-method comparison

For comparing a new method to an existing baseline, use `gap_method_comparison_table.py` as a template:
- Loads two gap CSVs (with `article_indices_json`)
- Runs log-odds-ratio keyword extraction with iterative bigram phrase mining (production default: `bigram_iterations=2`, `parse_with_phrases` for longest-match merging)
- Computes top-3 nearest-book votes per gap
- Emits markdown table side-by-side

The output is a self-contained markdown file under `docs/` that captures the comparison snapshot. See `docs/gap_method_comparison.md` for an example.

## Iteration discipline

**Background jobs**: any expansion that takes more than a minute should be launched with `run_in_background: true` redirecting to `/tmp/<run-name>.log`. Don't pipe through `tee` or `tail` — the buffering bites.

**Monitor for events**: arm a `Monitor` with `tail -F /tmp/<run>.log | grep --line-buffered "<key markers>"` to get notifications when key events fire (e.g., `processed`, `Discovery complete`, `Wrote`, `Error`).

**Kill if pace is wrong**: estimate per-1000-seed time from the first 2-3 progress lines. If observed rate is ≥10× slower than expected, kill and optimize. The HNSW switch came from exactly this pattern: T13734 was running at ~5 seeds/sec with FlatL2 (vs ~140 seeds/sec for HNSW); after observing the gap, killed + restarted with HNSW.

**Validate timing per topic**: before running a sweep, time a single representative scope and extrapolate. Don't assume linear scaling — early-abort cost grows super-linearly with N because per-seed cost includes O(N)-cost FAISS Flat queries.

## Where related ideas live

- `docs/backlog/` — forward-looking ideas. New gap-detection-method ideas go in `gap-detection-methods/`. New cohesion-metric ideas go in `cohesion-metric-design/`.
- `docs/dead_ends/` — refuted approaches with diagnoses. Consult before implementing something that "feels right" — it might already have been tried (e.g., centroid-Bayesian likelihood, monotone-score-as-boundary, inverse-square kernel).
- `docs/cohesion_design_axes.md` (when present) — taxonomy of design axes (initialization, frontier, acceptance policy, boundary criterion, ranking score, kernel, bandwidth, etc.) and which option each method picks.
- `docs/curriculum-method-comparison.md` — verdict tables for curriculum-side clustering methods (different problem, but same kind of cross-method comparison).
