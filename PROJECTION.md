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

## Current Status
- ✅ Introduced a `ProjectiveDenoiserProjection` payload that standardizes what gets cached (projection matrix + optional mean) and added `Cloud.denoiser_projection()` / `get_cloud_projection` helpers backed by `MEMORY`.
- ✅ Required every `WorkDenoiser` to expose `compute_projection_artifact`, including the `NONE` denoiser which now returns an identity transformation and respects normalization.
- ✅ Simplified generator error handling to use standard exceptions and updated tests accordingly.
- 🚧 The proposal/gap workflow still needs to load and apply the cached projection when generating proposal embeddings.

## Updated Implementation Plan
1. **Projection artifact plumbing *(done)***
   - Cache projection metadata via `compute_projection_artifact` and `MEMORY`.
   - Expose projections directly on `Cloud` for downstream consumers.

2. **Batch workflow integration *(done)***
   - Re-instantiate the `Cloud` from Hydra config wherever needed and access `cloud.denoiser_projection()` directly (construction is cheap and all heavy work stays cached).

3. **Apply projection during proposal embedding *(done)***
   - `_compute_proposal_embeddings` now rehydrates the `Cloud`, applies the cached projection, and mirrors normalization safeguards.

4. **Testing & validation *(in progress)***
   - Added a unit test covering projection application and normalization; expand with higher-level workflow coverage as needed.

## Open Questions & Considerations
- How should we surface projection availability to the LLM workflow—via configuration, metadata objects, or direct `Cloud` access?
- What invalidation rules do we need if upstream caches (works, embeddings) are refreshed? Do we need explicit versioning beyond `MEMORY` cache keys?
- Is additional synchronization required if multiple batch jobs request the projection simultaneously, or is `MEMORY` sufficient?

## Next Steps
- Expand workflow-level tests to ensure end-to-end reuse of projections.
- Document the new reuse path in README/TESTING guidance once integration coverage is complete.
