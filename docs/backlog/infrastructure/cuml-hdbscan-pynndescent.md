# Evaluate cuml.HDBSCAN + PyNNDescent for HDBSCAN speedup

(Migrated from `artifacts_scripts/curriculum/TODO.md`.)

Currently `hdbscan_gap_probe.py` uses `fast_hdbscan` (CPU-only); the BallTree mutual-reachability phase dominates wall-clock on subfield-scale clustering.

Two GPU/faster-CPU options:

- **`cuml.HDBSCAN`** (RAPIDS, GPU): claims 10-100× speedup. Install is large (CUDA version dependencies often clash with existing envs); main risk is dependency conflicts with the project's conda env. Worth testing in an isolated env first.
- **PyNNDescent** (CPU): builds approximate NN graph faster than exact BallTree; ~5× speedup on large datasets without GPU. `fast_hdbscan` may already use it under the hood — worth checking.

**Investigation plan**:
1. Profile a representative HDBSCAN probe run; identify whether tree-build or hierarchy-extraction is the bottleneck.
2. Pick the right intervention based on the bottleneck.

**When**: if probe-sweep wall-clock becomes a real obstacle for iteration. With current pipeline producing per-topic state in 30-90s for T11303/T10181/T13734, this is not urgent.
