# Proposal: Reusing Cloud Denoiser Projections for Proposal Embeddings

## Background
- `cloud.Cloud` builds topic-specific subclouds and optionally applies a projective denoiser that can reduce embedding dimensionality (for example, 768 → 384) before downstream gap analysis.
- `_compute_proposal_embeddings` in `bookgap/llm/generators/workflow_functions.py` currently derives embeddings directly from the base `Embedder` model without applying the same denoiser, leading to inconsistent embedding spaces between cloud-derived artifacts and proposal embeddings.
- To maintain continuity between gap discovery and proposal generation, proposal embeddings should be projected using the identical denoiser transformation that shaped the cloud representations.

## Goals & Success Criteria
- Persist and reuse the exact projection (e.g., PCA matrix) produced by the configured denoiser so that proposal embeddings live in the same vector space as cloud embeddings.
- Avoid recomputing the denoiser transformation for each batch; instead, cache it once per (cloud, embedder, filter) combination.
- Ensure the proposal embedding workflow degrades gracefully when no denoiser is configured.
- Preserve compatibility with Hydra-managed configs and existing cache/memoization layers.

## Investigation Tasks
1. **Audit denoiser APIs**: Confirm how `ProjectiveDenoiser` and its wrapped implementations expose projection matrices or transformation metadata (e.g., via attributes like `W`, `components_`, etc.). Determine whether a standard interface exists or needs to be added.
2. **Trace current caching**: Review existing memoization decorators (`@MEMORY.cache`, `@lru_cache`) around denoiser calls in `cloud.py` to understand where the projection currently lives in memory and how to persist it.
3. **Identify serialization needs**: Decide on a stable format (NumPy `.npy`, `.npz`, or joblib) that captures the projection matrix and any auxiliary statistics (means, scaling factors) the denoiser requires during inference.
4. **Locate hydra output hooks**: Determine the most appropriate location to store/retrieve the serialized projection within Hydra's runtime output directory or reusable cache directories.

## Implementation Plan
1. **Leverage `joblib.Memory` for projection caching**
   - Extend the denoising workflow (likely within `ProjectiveDenoiser.denoise` or the surrounding cloud orchestration) to register the projection matrix and supporting parameters with the existing `MEMORY` cache, using a deterministic key derived from the cloud configuration (filters, embedder model, deduplicator, denoiser name, normalization flag, etc.).
   - Reuse the same cache directory already managed by `MEMORY` instead of writing dedicated artifacts to `ARTIFACTS_DIR`.
   - Ensure the cached payload includes everything `_compute_proposal_embeddings` needs (projection matrix, mean vector, normalization hints, etc.) so it can be reloaded without recomputation.

2. **Expose projection metadata on `Cloud`**
   - Augment the `Cloud` dataclass to retain the cache key and `joblib` accessor needed to retrieve the cached projection payload.
   - Provide helper methods (e.g., `cloud.get_projection()` or `cloud.denoiser_projection()`) that call into the `MEMORY` cache and return the projection matrix, raising a clear error if unavailable when a denoiser is configured.

3. **Plumb projection details through batch workflow metadata**
   - When `run_batch_workflow` instantiates the cloud and builds contexts, capture the cache key or callable that retrieves the projection and record it in the batch run's metadata (e.g., extend `ArtifactPaths`, `BatchResults.metadata`, or attach it to the Hydra output directory).
   - Ensure that `_compute_proposal_embeddings` can access the `MEMORY` cache entry either through direct invocation or via data passed into `BatchResults`.

4. **Apply projection during proposal embedding**
   - Modify `_compute_proposal_embeddings` to:
     1. Load the cached projection associated with the current batch configuration.
     2. Apply any preprocessing required by the projection (e.g., mean centering) prior to multiplying by the projection matrix.
     3. Fall back to raw embeddings if no denoiser is configured.
   - Validate that the resulting embedding dimensionality matches the one used in the cloud pipeline.

5. **Testing & validation**
   - Add unit/integration tests that run a minimal configuration with a simple denoiser (e.g., PCA) to verify that:
     - The projection artifact is created and persisted.
     - Proposal embeddings are projected to the expected dimension.
     - Workflows without a denoiser remain unchanged.
   - Update documentation (README/TODO) to explain the new artifact and reusability pattern.

## Open Questions & Considerations
- Do all supported denoisers expose a consistent projection interface, or do we need adapter methods per implementation?
- Should projections be invalidated and recomputed if upstream cached embeddings change (e.g., new works fetched)? How to detect staleness efficiently?
- Is in-memory caching via `@MEMORY.cache` sufficient for repeated calls within a single process, or do we also need on-disk persistence for cross-process reuse?
- How will concurrency be handled if multiple batch jobs attempt to compute the projection simultaneously? Consider file locks or atomic writes.
- Are there existing Hydra config fields where we should surface the projection artifact path for transparency or downstream tooling?

## Next Steps
- Complete the investigation tasks to confirm assumptions about denoiser interfaces and caching.
- Design the serialization schema and artifact directory structure.
- Prototype the caching mechanism and update `_compute_proposal_embeddings` to consume it.
- Iterate with tests to ensure reproducibility and compatibility across denoiser variants.
