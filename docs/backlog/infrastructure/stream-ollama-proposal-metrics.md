# Stream Ollama proposal metrics per generation, not in a batch at the end

## Idea

Compute coverage / gap-stats incrementally as each Ollama proposal is
generated, instead of deferring the entire embedding + metrics pass
until all proposals are done.

Today's flow in `_run_ollama_local_path` (`workflow_functions.py:921`):

1. Loop over all 77 contexts, calling `OllamaProposalGenerator.generate()`
   per topic; append each record to `ollama_results.jsonl` (crash-resumable). ✓
2. **After the loop returns**, call `_compute_proposal_embeddings`
   (line 1058) → `_generate_downstream_artifacts` (line 1080), which
   does ONE bulk Gemma/GIST embedding pass and computes outrank counts
   + gap stats for every proposal.

For a 77-topic, ~1.3-proposal/minute generation rate (gemma4:31b on a
single RTX 6000 Ada), that's ~55 min of generation before any of the
headline numbers (target gap coverage %, proposal gap percentile) become
visible. If a configuration tweak is bad — wrong denoiser, wrong drop_k,
prompt regression — the operator doesn't know for an hour.

## Work: parallel path, not replacement

Add streaming as an OBSERVABILITY path that runs alongside the
existing vectorized post-loop pipeline. The canonical
`gaps_analysis_results.csv` still comes from the post-loop
`_generate_downstream_artifacts` call (which batches embeddings — a
real efficiency win); the streaming path produces a separate
`coverage_progress.csv` for live monitoring.

Two paths, one canonical, one preview:

- **Canonical (existing, unchanged):** post-loop batched embedding +
  per-gap outrank-counts + per-gap gap-stats → `gaps_analysis_results.csv`.
- **Streaming (new):** per-iteration single-proposal embedding +
  single-gap outrank counts → row appended to `coverage_progress.csv`.

Per-iteration streaming cost: ~1-2 seconds (vs ~30s for the ollama
generation it follows). Across 77 topics, total streaming overhead is
~150 seconds — trivial vs. the ~50min generation budget. **Async /
threading is unnecessary**: the cheap part is already noise relative
to the expensive ollama generation.

Implementation in `_run_ollama_local_path` (`workflow_functions.py:921`):

1. Open a `SentenceTransformerEmbedder(...)` context manager spanning
   the entire ollama for-loop (model loaded once, reused).
2. Open `coverage_progress.csv` and write a header.
3. After each successful proposal generation (proposal is not None):
   - Embed title+abstract via the hot embedder (single-row batch).
   - Apply `cloud.denoiser_projection()` + re-normalise.
   - Compute outrank counts against THIS gap's book index (the
     existing per-gap logic, just run inside the loop).
   - Append one row to `coverage_progress.csv`.
4. After the loop, the existing post-loop pipeline runs unchanged,
   producing `gaps_analysis_results.csv` as canonical.

Cost: ~50-80 lines. The hot embedder costs ~1.5 GB of GPU (negligible
next to ollama's ~24 GB).

## Why it matters

- **Faster iteration on the pre-paper LPP-vs-CCLIP verification.** A
  bad config shows its coverage trend after 5–10 proposals (~5 min),
  not 55 min.
- **Crash recovery preserves metrics**, not just raw proposals. Today
  a SIGINT'd run leaves you with JSONL but no coverage numbers; a
  rerun replays the metrics computation. With streaming, the CSV
  survives.
- **Cheap to do.** No new dependencies. Code already exists, just in
  the wrong place in the pipeline.

Not relevant to the OpenAI Batch path (that one is genuinely batched
end-to-end by the OpenAI API; metrics-after-the-fact is forced there).
This change is Ollama-specific.
