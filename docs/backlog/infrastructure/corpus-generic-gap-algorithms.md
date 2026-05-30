# Make gap algorithms `Corpus`-generic at runtime (finish the SubCloud→Corpus relaxation)

## Why

SP1 Task 5 relaxed the *public* gap-algorithm seams
(`KballAlgorithm.find_gaps` / `compute_gap_sizes_for_embeddings` and the
Voronoi equivalents) from `SubCloud` to the `Corpus` protocol, so
`detect_gaps(ambient, landmark, query)` is source-agnostic. But the
*internals* still assume `SubCloud`:

- `bookgap/cloud_index.py::get_kball_coverage` asserts
  `isinstance(ambient, SubCloud)` and `ambient.parent == landmark.parent
  == query.parent` — a genuine runtime contract, not just a hint. A
  `UnionCorpus` (which has no `.parent`) would fail this assert.
- `KballGap` / `VoronoiGap` data-carrier fields and their
  `get_ambient`/`get_landmark` accessors are still typed `SubCloud`.
- `kball_gaps` / `_resid_gaps` internal helpers are still typed `SubCloud`.

Net effect: passing the now-`Corpus`-typed public params into these
`SubCloud`-typed sinks produces ~6 mypy `arg-type` notes inside
`find_gaps`' body (and at `VoronoiGap(...)` construction). They are
*honest* — the OpenAlex path only ever passes `SubCloud`s — so the
relaxation was deliberately kept partial in Task 5 (mypy is not the
Task-5 gate; import-smoke + corpus pytest are). The errors will resolve
once the algorithms are made `Corpus`-generic at runtime.

## What to do (when SP3 / curriculum-on-Corpus needs it)

1. Replace `get_kball_coverage`'s `isinstance(SubCloud)` + shared-`parent`
   assertions with a shared-embedding-space check expressed in `Corpus`
   terms (e.g. equal `embedder_fingerprint`, which `detect_gaps` already
   enforces) — then widen its param hints to `Corpus`.
2. Widen `kball_gaps` / `_resid_gaps` params and the `KballGap` /
   `VoronoiGap` dataclass fields + accessors to `Corpus`.
3. Confirm the `knn(index=corpus, query=corpus, ...)` paths (cloud_index)
   only use `.embeddings()` / `.index()` — they do today — and widen their
   `Union[Cloud, SubCloud]` hints to include `Corpus`.
4. Re-run mypy on the gap-algorithm files; the residual `arg-type` notes
   from SP1-T5 should drop to zero.

Annotation + assertion-only; no algorithm logic changes. The byte-for-byte
parity gate (SP1-T6) must still pass.
