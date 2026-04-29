"""Smoke test for faiss-gpu installation.

Verifies that faiss-gpu is installed, the GPU is detected, and the
operations we care about (index build, kNN search, range search) work
on both CPU and GPU paths and produce equivalent results.

This is gated by a deliberate env separation: the project's main conda
env has ``faiss-cpu``. To avoid breaking the production pipeline mid-
sweep, install ``faiss-gpu-cu12`` into an isolated venv first and run
this script there to confirm functionality before swapping the package
in the main env.

Usage::

    python -m venv /tmp/faiss-gpu-test
    /tmp/faiss-gpu-test/bin/pip install faiss-gpu-cu12==1.12.0 numpy
    /tmp/faiss-gpu-test/bin/python artifacts_scripts/faiss_gpu_smoke_test.py

The script reports:
  - faiss version + GPU count
  - CPU/GPU IndexFlatL2 build + kNN search timing + correctness match
  - CPU/GPU range_search timing + correctness match (critical for K-ball)
  - HNSW behaviour on GPU (informational; HNSW typically stays on CPU
    in FAISS, so the question is whether `index_cpu_to_gpu` succeeds
    or falls back gracefully)
  - End-to-end timing on a 200K × 384 dataset (representative subfield
    scale) for FlatL2 CPU vs GPU
"""
from __future__ import annotations

import time

import numpy as np


def _bench(label: str, fn) -> tuple[float, object]:
    t0 = time.time()
    out = fn()
    elapsed = time.time() - t0
    print(f"  {label:30s} {elapsed * 1000:>8.1f} ms")
    return elapsed, out


def main() -> None:
    import faiss

    print(f"faiss version:        {faiss.__version__}")
    n_gpus = faiss.get_num_gpus()
    print(f"faiss GPU count:      {n_gpus}")
    if n_gpus == 0:
        raise RuntimeError(
            "No GPUs detected by faiss. Either CUDA is unavailable or "
            "you're running with faiss-cpu instead of faiss-gpu."
        )

    rng = np.random.default_rng(42)
    d = 384

    # ------------------------------------------------------------------
    # Small index correctness (CPU vs GPU should produce identical k-NN)
    # ------------------------------------------------------------------
    print("\n=== Correctness: CPU FlatL2 vs GPU FlatL2 (n=10000) ===")
    n = 10_000
    data = rng.standard_normal((n, d), dtype=np.float32)
    queries = rng.standard_normal((100, d), dtype=np.float32)

    cpu_idx = faiss.IndexFlatL2(d)
    cpu_idx.add(data)
    _, (cpu_dists, cpu_ids) = _bench(
        "CPU FlatL2 search (k=5)", lambda: cpu_idx.search(queries, 5)
    )

    res = faiss.StandardGpuResources()
    gpu_idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    gpu_idx.add(data)
    _, (gpu_dists, gpu_ids) = _bench(
        "GPU FlatL2 search (k=5)", lambda: gpu_idx.search(queries, 5)
    )

    matches = int((cpu_ids == gpu_ids).sum())
    total = cpu_ids.size
    print(f"  k-NN match rate:               {matches}/{total} = {100 * matches / total:.1f}%")
    if matches < total - 5:  # allow a few mismatches due to FP rounding
        print("  WARNING: substantial CPU/GPU mismatch on k-NN — investigate")

    # ------------------------------------------------------------------
    # Range search (what K-ball uses) on CPU vs GPU
    # ------------------------------------------------------------------
    print("\n=== Range search: CPU FlatL2 vs GPU FlatL2 ===")
    radius = 380.0  # squared L2; tuned so ~few hundred results per query
    cpu_lims, cpu_d, cpu_i = cpu_idx.range_search(queries[:10], radius)
    print(f"  CPU range_search:              total neighbours = {len(cpu_i)}")
    try:
        gpu_lims, gpu_d, gpu_i = gpu_idx.range_search(queries[:10], radius)
        print(f"  GPU range_search:              total neighbours = {len(gpu_i)}")
        if len(cpu_i) == len(gpu_i):
            print(f"  range_search count match:      OK")
        else:
            print(f"  WARNING: count mismatch (CPU={len(cpu_i)}, GPU={len(gpu_i)})")
    except (RuntimeError, AttributeError) as e:
        print(f"  GPU range_search: FAILED — {type(e).__name__}: {e}")
        print("  (Note: K-ball requires range_search; if GPU range_search isn't")
        print("   available, K-ball can only get GPU benefit via index_gpu_to_cpu")
        print("   round-trips, which usually negates the speedup.)")

    # ------------------------------------------------------------------
    # HNSW on GPU (informational — typically stays on CPU in FAISS)
    # ------------------------------------------------------------------
    print("\n=== HNSW: CPU and (attempted) GPU ===")
    hnsw_cpu = faiss.IndexHNSWFlat(d, 32)
    hnsw_cpu.hnsw.efConstruction = 64
    _bench("CPU HNSW build", lambda: hnsw_cpu.add(data))
    _bench("CPU HNSW search (k=5)", lambda: hnsw_cpu.search(queries, 5))
    try:
        hnsw_gpu = faiss.index_cpu_to_gpu(res, 0, hnsw_cpu)
        _bench("GPU HNSW search (k=5)", lambda: hnsw_gpu.search(queries, 5))
    except (RuntimeError, AttributeError) as e:
        print(f"  GPU HNSW: not supported — {type(e).__name__}: {str(e)[:100]}")
        print("  Expected: FAISS HNSW is CPU-only; for GPU benefit on K-ball-")
        print("  scale data we'd switch to GpuIndexFlatL2 (brute-force on GPU)")
        print("  or GpuIndexIVFFlat (clustered).")

    # ------------------------------------------------------------------
    # Subfield-scale timing comparison (representative of K-ball workload)
    # ------------------------------------------------------------------
    print("\n=== Subfield-scale timing (n=200000, d=384, q=1000) ===")
    n_large = 200_000
    data_large = rng.standard_normal((n_large, d), dtype=np.float32)
    query_large = rng.standard_normal((1000, d), dtype=np.float32)

    cpu_flat = faiss.IndexFlatL2(d)
    _bench("CPU FlatL2 add 200K", lambda: cpu_flat.add(data_large))
    _bench("CPU FlatL2 search 1000q × k=10", lambda: cpu_flat.search(query_large, 10))

    gpu_flat = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
    _bench("GPU FlatL2 add 200K", lambda: gpu_flat.add(data_large))
    _bench("GPU FlatL2 search 1000q × k=10", lambda: gpu_flat.search(query_large, 10))

    print("\n=== faiss-gpu smoke test complete ===")


if __name__ == "__main__":
    main()
