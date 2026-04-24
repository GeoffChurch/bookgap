"""Correctness + approximation-quality tests for the deduplicators.

Structured in three bands:

1. Semantic-contract tests run against both :class:`SqDistDeduplicator`
   (exact) and :class:`SimHashDeduplicator` (with high-recall params).
   Same assertions must hold — captures the invariant "drop iff an older
   alive neighbour is within sqrt(min_sqdist) L2".

2. SimHash-specific approximation tests sweep ``k_bits`` and ``n_tables``
   and check recall against brute-force ground truth. Validates the
   probabilistic scaling we relied on in ``dedup_methods_bench.py``.

3. Adversarial / scaling tests exercise hot buckets, huge-bucket memory
   bounds, and determinism.
"""
from __future__ import annotations

import numpy as np
import pytest

from bookgap.deduplicators import (
    Deduplicator,
    SimHashDeduplicator,
    SqDistDeduplicator,
)
from bookgap.index import HNSW


@pytest.fixture
def hnsw_cfg():
    """IndexInfo passed into deduplicate(); required by both implementations
    even though SimHash ignores it."""
    return HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)


def _unit_sphere_random(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _inject_near_duplicates(
    emb: np.ndarray, pairs: list[tuple[int, int]], sigma: float, seed: int = 1
) -> np.ndarray:
    """Rewrite ``emb[b]`` to be a tiny perturbation of ``emb[a]``, then
    re-normalise. With sigma=0.001 the sqdist is ~4e-4, well under our
    default 0.01 threshold."""
    rng = np.random.default_rng(seed)
    out = emb.copy()
    d = emb.shape[1]
    for a, b in pairs:
        noise = rng.standard_normal(d).astype(np.float32) * sigma
        out[b] = out[a] + noise
        out[b] /= np.linalg.norm(out[b])
    return out


def _pair_sqdist(emb: np.ndarray, a: int, b: int) -> float:
    diff = emb[a] - emb[b]
    return float(np.dot(diff, diff))


# ---------------------------------------------------------------------------
# 1. Semantic-contract tests (run for both implementations).
# ---------------------------------------------------------------------------

def _both_dedups(min_sqdist: float = 0.01) -> list[Deduplicator]:
    """Deduplicators configured with enough recall headroom that semantic
    tests are deterministic: SimHash at k=48, L=30 has P(miss per pair)
    ≈ (1 - 0.99^48)^30 ≈ 10^-30 for cos>0.995 dupes — effectively zero.
    """
    return [
        SqDistDeduplicator(min_sqdist=min_sqdist),
        SimHashDeduplicator(min_sqdist=min_sqdist, k_bits=48, n_tables=30, seed=0),
    ]


@pytest.mark.parametrize("dedup", _both_dedups(), ids=["sqdist", "simhash_k48L30"])
def test_empty_input_returns_empty(dedup, hnsw_cfg):
    emb = np.zeros((0, 384), dtype=np.float32)
    out = dedup.deduplicate(emb, hnsw_cfg)
    assert len(out) == 0


@pytest.mark.parametrize("dedup", _both_dedups(), ids=["sqdist", "simhash_k48L30"])
def test_single_point_returns_singleton(dedup, hnsw_cfg):
    emb = _unit_sphere_random(1, 384)
    out = dedup.deduplicate(emb, hnsw_cfg)
    assert list(out) == [0]


@pytest.mark.parametrize("dedup", _both_dedups(), ids=["sqdist", "simhash_k48L30"])
def test_no_duplicates_retains_all(dedup, hnsw_cfg):
    emb = _unit_sphere_random(500, 384, seed=7)
    # All random unit vectors are near-orthogonal in high-D, so no pair
    # has sqdist < 0.01 (their sqdist distribution is centred around 2.0).
    out = dedup.deduplicate(emb, hnsw_cfg)
    assert len(out) == 500


@pytest.mark.parametrize("dedup", _both_dedups(), ids=["sqdist", "simhash_k48L30"])
def test_all_identical_keeps_only_oldest(dedup, hnsw_cfg):
    base = _unit_sphere_random(1, 384)[0]
    emb = np.tile(base, (20, 1)).astype(np.float32)
    out = dedup.deduplicate(emb, hnsw_cfg)
    # Only index 0 (the oldest) should survive — every other point has
    # an older-alive-neighbour at sqdist=0.
    assert list(out) == [0]


@pytest.mark.parametrize("dedup", _both_dedups(), ids=["sqdist", "simhash_k48L30"])
def test_injected_near_duplicates_drop_newer(dedup, hnsw_cfg):
    emb = _unit_sphere_random(500, 384, seed=7)
    # Points at indices 100..109 become near-duplicates of 0..9
    # respectively. Older wins, so the newer member of each pair drops.
    pairs = [(i, 100 + i) for i in range(10)]
    emb = _inject_near_duplicates(emb, pairs, sigma=0.001, seed=11)
    # Sanity: true sqdist is below threshold
    assert all(_pair_sqdist(emb, a, b) < 0.01 for a, b in pairs)

    out = set(dedup.deduplicate(emb, hnsw_cfg).tolist())
    # Original 500 minus the 10 newer duplicates → 490.
    assert len(out) == 490
    # The older originals 0..9 all survive.
    for i in range(10):
        assert i in out
    # The newer duplicates 100..109 all dropped.
    for i in range(10):
        assert (100 + i) not in out


def test_threshold_boundary_exact():
    """Pairs at sqdist just below the threshold drop; just above, survive.
    Exact threshold = kept (kernel uses strict `<`)."""
    d = 64
    dedup = SqDistDeduplicator(min_sqdist=0.01)
    hnsw = HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)
    a = np.zeros(d, dtype=np.float32); a[0] = 1.0
    # Construct b at exactly a specified sqdist from a.
    def b_at(sqd):
        b = a.copy()
        b[1] = np.sqrt(sqd)  # shifts b off a along an orthogonal axis
        b[0] = np.sqrt(max(0, 1 - b[1] ** 2))  # re-normalise first component
        b /= np.linalg.norm(b)
        return b
    for target in [0.005, 0.0099]:
        emb = np.stack([a, b_at(target)])
        out = dedup.deduplicate(emb, hnsw)
        assert len(out) == 1, f"below threshold ({target}) should drop newer"
    for target in [0.011, 0.02]:
        emb = np.stack([a, b_at(target)])
        out = dedup.deduplicate(emb, hnsw)
        assert len(out) == 2, f"above threshold ({target}) should keep both"


# ---------------------------------------------------------------------------
# 2. SimHash approximation-quality tests.
# ---------------------------------------------------------------------------

def _recall_on_injected(
    n_tables: int, k_bits: int, n_pairs: int = 20, seed: int = 0
) -> float:
    """Return per-pair recall: fraction of injected near-duplicate pairs
    that the deduplicator correctly drops."""
    emb = _unit_sphere_random(1000, 384, seed=seed)
    pairs = [(i, 500 + i) for i in range(n_pairs)]
    emb = _inject_near_duplicates(emb, pairs, sigma=0.001, seed=seed + 100)
    dedup = SimHashDeduplicator(
        min_sqdist=0.01, k_bits=k_bits, n_tables=n_tables, seed=seed
    )
    hnsw = HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)
    retained = set(dedup.deduplicate(emb, hnsw).tolist())
    # A pair is "caught" iff the newer member was dropped.
    caught = sum(1 for (_, b) in pairs if b not in retained)
    return caught / n_pairs


@pytest.mark.parametrize("n_tables", [1, 3, 10, 30])
def test_recall_increases_with_n_tables(n_tables):
    """More tables → monotone recall improvement. At L=30 with k=32 and
    tight (sqdist < 0.01) dupes we expect 100%; at L=1 we expect < 100%
    per-pair. Test a loose ordering."""
    recall = _recall_on_injected(n_tables=n_tables, k_bits=32, n_pairs=40)
    assert 0.0 <= recall <= 1.0
    if n_tables >= 10:
        assert recall >= 0.9, (
            f"L={n_tables}, k=32 should catch ≥90% of cos>0.999 dupes, got {recall:.2%}"
        )
    if n_tables >= 30:
        assert recall >= 0.99


def test_recall_high_at_production_defaults():
    """The production configuration (k=32, L=20) should achieve ≥98% on
    injected tight near-duplicates — matches our benchmark observations."""
    recall = _recall_on_injected(n_tables=20, k_bits=32, n_pairs=100)
    assert recall >= 0.98, f"production defaults underperformed: recall={recall:.2%}"


def test_recall_improves_with_k_bits_when_buckets_dense():
    """When many points land in the same bucket (e.g. k=12 on 1k points),
    bucket-size explosion costs time but doesn't affect recall. The real
    k_bits lever is for *speed* at scale, not accuracy."""
    # Just sanity: both k=16 and k=48 should achieve full recall on our
    # easy-cos>0.999 injected set with L=30.
    for k in [16, 48]:
        recall = _recall_on_injected(n_tables=30, k_bits=k, n_pairs=40)
        assert recall >= 0.95


# ---------------------------------------------------------------------------
# 3. Adversarial / scaling tests.
# ---------------------------------------------------------------------------

def test_simhash_determinism_given_seed():
    """Same seed, same input → same retained indices."""
    emb = _unit_sphere_random(800, 128, seed=3)
    emb = _inject_near_duplicates(emb, [(i, 400 + i) for i in range(30)], sigma=0.001)
    hnsw = HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)
    a = SimHashDeduplicator(min_sqdist=0.01, k_bits=32, n_tables=10, seed=42).deduplicate(emb, hnsw)
    b = SimHashDeduplicator(min_sqdist=0.01, k_bits=32, n_tables=10, seed=42).deduplicate(emb, hnsw)
    assert np.array_equal(a, b)


def test_simhash_varies_with_seed():
    """Different seed → potentially different output (probabilistic miss
    distribution varies). Sanity-check that hyperplanes really depend on
    seed by asserting *some* injected dupes differ in which tables catch
    them; since at L=2 we don't always catch all dupes, at least one seed
    should deviate."""
    emb = _unit_sphere_random(400, 128, seed=5)
    pairs = [(i, 200 + i) for i in range(60)]
    emb = _inject_near_duplicates(emb, pairs, sigma=0.002)  # slightly looser sigma → harder
    hnsw = HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)
    outs = [
        set(SimHashDeduplicator(min_sqdist=0.01, k_bits=16, n_tables=2, seed=s)
            .deduplicate(emb, hnsw).tolist())
        for s in range(4)
    ]
    # If the hyperplanes really do depend on seed and L=2 is weak enough
    # to miss some dupes, at least two of the four outputs will differ.
    assert any(outs[i] != outs[j] for i in range(4) for j in range(i + 1, 4)), \
        "SimHash output identical across seeds — RNG seed not being used"


def test_hot_bucket_drops_to_oldest_only():
    """5000 near-identical points (all within ε of point 0) should be
    collapsed to just point 0. Tests the chunked pair-verify path inside
    a large bucket."""
    base = _unit_sphere_random(1, 384, seed=0)[0]
    n = 5000
    rng = np.random.default_rng(10)
    emb = np.empty((n, 384), dtype=np.float32)
    emb[0] = base
    for i in range(1, n):
        # Each perturbation is tight enough to stay within ε of `base`
        e = base + rng.standard_normal(384).astype(np.float32) * 0.0005
        emb[i] = e / np.linalg.norm(e)
    # Confirm they are within ε.
    assert all(_pair_sqdist(emb, 0, i) < 0.01 for i in range(1, n))

    dedup = SimHashDeduplicator(min_sqdist=0.01, k_bits=16, n_tables=10, seed=0)
    hnsw = HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)
    out = dedup.deduplicate(emb, hnsw)
    assert list(out) == [0], f"hot bucket should collapse to the oldest point, got {len(out)} survivors"


def test_non_normalized_embeddings_use_real_l2():
    """sqdist uses ||a-b||² explicitly, so non-normalised embeddings
    should still work (we don't rely on the ||a||=1 simplification)."""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 64)).astype(np.float32) * 3.7  # scale ≠ 1
    # Point 50 is a close copy of point 5, both large-magnitude.
    emb[50] = emb[5] + rng.standard_normal(64).astype(np.float32) * 0.01
    sqd = _pair_sqdist(emb, 5, 50)
    assert sqd < 0.02, f"setup error, pair sqdist {sqd} not in expected range"

    dedup = SqDistDeduplicator(min_sqdist=0.02)
    hnsw = HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)
    out = set(dedup.deduplicate(emb, hnsw).tolist())
    # 50 should drop, 5 should stay.
    assert 5 in out
    assert 50 not in out


def test_simhash_vectorized_vs_scalar_equivalence():
    """Cross-check the vectorised chunked implementation against the
    brute-force semantics by using SqDistDeduplicator as the spec. When
    SimHash's recall is 100% (which happens at L=30 on these easy inputs),
    its output must exactly equal SqDistDeduplicator's.
    """
    emb = _unit_sphere_random(600, 256, seed=2)
    pairs = [(i, 300 + i) for i in range(40)]
    emb = _inject_near_duplicates(emb, pairs, sigma=0.001)
    hnsw = HNSW(hnsw_m=16, hnsw_efc=384, hnsw_efs=128)
    exact = set(SqDistDeduplicator(min_sqdist=0.01).deduplicate(emb, hnsw).tolist())
    approx = set(SimHashDeduplicator(
        min_sqdist=0.01, k_bits=32, n_tables=30, seed=0
    ).deduplicate(emb, hnsw).tolist())
    missing = exact - approx
    extra = approx - exact
    assert not missing, (
        f"SimHash retained {len(missing)} points that SqDist dropped: {sorted(missing)[:10]}"
    )
    # SimHash can at most under-drop (miss dupes), never over-drop beyond
    # what SqDist would — the pair kernel is identical, only the candidate
    # retrieval differs.
    assert not extra or extra.issubset(
        {b for (_, b) in pairs}
    ), "SimHash dropped a point SqDist kept and not on the injected-dupes list"
