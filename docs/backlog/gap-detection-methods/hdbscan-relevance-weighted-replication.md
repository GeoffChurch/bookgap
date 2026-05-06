# HDBSCAN with relevance-weighted point replication

(Migrated from `artifacts_scripts/curriculum/TODO.md`.)

Density-based clustering with naturally variable cluster sizes and no forced cluster count. HDBSCAN's core algorithm is unweighted; to fold per-work relevance into it, replicate each work proportional to its relevance score (or use a discrete weight via repetition).

For the curriculum case: replication weights = PageRank/CircuitRank mass per work. For the gap-detection case: weights could be per-article cohesion proxy (e.g., 1-NN-book distance, with farther = more weight).

**Why interesting**: HDBSCAN handles variable density; replication is a clean unweighted-to-weighted conversion that doesn't require modifying the algorithm.

**Open questions**:
- Replication factor: integer rounding loses precision; large factors blow up runtime
- Whether `cuml.HDBSCAN` (GPU) or `fast_hdbscan` handles replicated points efficiently
