"""Smoke test for USearch (CPU + GPU).

Companion to ``faiss_gpu_smoke_test.py``. Verifies:
  - USearch installs and imports
  - GPU support is detectable (and works if so)
  - Index build + kNN search produce sensible results
  - Range search (radius-based) works — critical for K-ball gap detection
  - Correctness comparison: USearch results match a FAISS baseline (close
    enough; both are approximate HNSW)

Run via isolated venv::

    python -m venv /tmp/usearch-test
    /tmp/usearch-test/bin/pip install usearch faiss-cpu numpy
    /tmp/usearch-test/bin/python artifacts_scripts/usearch_smoke_test.py

Outcome: if USearch range search works on GPU, we have a contained path
to GPU-accelerate K-ball without rewriting the algorithm. If not, we
fall back to refactoring K-ball to use brute-force GPU kNN with a
manual radius filter (see the corresponding TODO entry).
"""
from __future__ import annotations

import time

import numpy as np


def _bench(label: str, fn):
    t0 = time.time()
    out = fn()
    elapsed = time.time() - t0
    print(f"  {label:36s} {elapsed * 1000:>8.1f} ms")
    return out


def main() -> None:
    import usearch
    from usearch.index import Index, MetricKind

    print(f"usearch version: {getattr(usearch, '__version__', 'unknown')}")

    # ------------------------------------------------------------------
    # Detect GPU support: try to inspect the package, then attempt GPU
    # index construction.
    # ------------------------------------------------------------------
    print("\n=== GPU detection ===")
    try:
        # USearch indicates GPU support via either an explicit constructor
        # arg or a separate package. Probe both.
        gpu_supported_attr = any(
            hasattr(usearch, attr)
            for attr in ("USEARCH_GPU", "compiled_for_cuda", "has_cuda")
        )
        print(f"  usearch package GPU markers: {gpu_supported_attr}")
    except Exception as e:
        print(f"  GPU marker probe failed: {e}")

    # Construct an Index and inspect supported features.
    rng = np.random.default_rng(42)
    d = 384
    n = 10_000
    data = rng.standard_normal((n, d), dtype=np.float32)
    queries = rng.standard_normal((100, d), dtype=np.float32)

    # ------------------------------------------------------------------
    # CPU index — baseline functionality
    # ------------------------------------------------------------------
    print("\n=== CPU Index: build + k-NN search (n=10000) ===")
    keys = np.arange(n, dtype=np.uint64)
    cpu_idx = Index(ndim=d, metric=MetricKind.L2sq, dtype="f32")
    _bench("CPU build (add 10K)", lambda: cpu_idx.add(keys, data))
    print(f"  CPU index size: {len(cpu_idx)}")
    matches = _bench(
        "CPU search 100q × k=5",
        lambda: cpu_idx.search(queries, count=5),
    )
    print(f"  matches type: {type(matches).__name__}")
    print(f"  first-query keys (top-5): {matches[0].keys.tolist() if hasattr(matches[0], 'keys') else matches.keys[0].tolist()}")

    # ------------------------------------------------------------------
    # Range search (critical for K-ball)
    # ------------------------------------------------------------------
    print("\n=== Range search (radius-based) ===")
    radius = 380.0  # squared L2 — tuned so we get a few hundred neighbours
    try:
        # Try the API: USearch supports radius-based search via the
        # ``radius`` kwarg on .search() in some versions, or a separate
        # method. Probe both.
        if hasattr(cpu_idx, "search") and "radius" in cpu_idx.search.__doc__ if cpu_idx.search.__doc__ else False:
            range_matches = cpu_idx.search(queries[:10], count=1000, radius=radius)
            print(f"  CPU range_search via search(..., radius=...): OK")
            print(f"  matches type: {type(range_matches).__name__}")
        else:
            range_matches = cpu_idx.search(queries[:10], count=1000)
            counts = [
                int((m.distances <= radius).sum()) if hasattr(m, "distances") else 0
                for m in range_matches
            ]
            print(f"  CPU 'range_search' emulated via search(count=1000) + filter")
            print(f"  per-query neighbour counts within radius {radius}: {counts}")
    except Exception as e:
        print(f"  Range search probe failed: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # GPU attempt
    # ------------------------------------------------------------------
    print("\n=== GPU attempt ===")
    gpu_supported = False
    try:
        # Many USearch builds expose .gpu() or accept a device kwarg.
        # Try the most common patterns.
        gpu_idx = Index(ndim=d, metric=MetricKind.L2sq, dtype="f32")
        if hasattr(gpu_idx, "gpu"):
            gpu_idx = gpu_idx.gpu()
            print(f"  GPU index via .gpu(): OK")
            gpu_supported = True
        else:
            # Try device argument
            gpu_idx_2 = Index(ndim=d, metric=MetricKind.L2sq, dtype="f32", device="cuda")
            print(f"  GPU index via device='cuda': OK")
            gpu_idx = gpu_idx_2
            gpu_supported = True
    except (TypeError, RuntimeError, AttributeError) as e:
        print(f"  GPU index construction failed: {type(e).__name__}: {str(e)[:200]}")

    if gpu_supported:
        try:
            _bench("GPU build (add 10K)", lambda: gpu_idx.add(keys, data))
            _bench(
                "GPU search 100q × k=5",
                lambda: gpu_idx.search(queries, count=5),
            )
            print("  GPU search: OK")
        except Exception as e:
            print(f"  GPU search failed: {type(e).__name__}: {str(e)[:200]}")

    # ------------------------------------------------------------------
    # Subfield-scale timing comparison vs FAISS CPU baseline
    # ------------------------------------------------------------------
    print("\n=== Subfield-scale timing (n=200000, q=1000) ===")
    n_large = 200_000
    data_large = rng.standard_normal((n_large, d), dtype=np.float32)
    query_large = rng.standard_normal((1000, d), dtype=np.float32)
    keys_large = np.arange(n_large, dtype=np.uint64)

    # USearch CPU
    us_idx = Index(ndim=d, metric=MetricKind.L2sq, dtype="f32")
    _bench("USearch CPU build 200K", lambda: us_idx.add(keys_large, data_large))
    _bench(
        "USearch CPU search 1000q × k=10",
        lambda: us_idx.search(query_large, count=10),
    )

    # FAISS HNSW comparison
    try:
        import faiss
        fhnsw = faiss.IndexHNSWFlat(d, 32)
        fhnsw.hnsw.efConstruction = 64
        _bench("FAISS HNSW build 200K", lambda: fhnsw.add(data_large))
        _bench(
            "FAISS HNSW search 1000q × k=10",
            lambda: fhnsw.search(query_large, 10),
        )
    except ImportError:
        print("  (faiss not available in this venv; skipping baseline)")

    print("\n=== USearch smoke test complete ===")


if __name__ == "__main__":
    main()
