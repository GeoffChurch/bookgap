"""Benchmark candidate near-duplicate-detection methods on the real corpus.

Compares HNSW (baseline), FAISS IVFFlat, FAISS IndexLSH + verify,
custom SimHash + verify, and custom sorted-projection sweep against
brute-force IndexFlatL2 ground truth. Reports build time, query time,
recall of true duplicate pairs, and projected 42 M wall time.

All candidate methods terminate with an "exact verify" step: the
candidates returned by the approximate index are double-checked with
real L2 distances, so false positives don't matter for recall — only
false negatives (missed true duplicates).

Usage::

    python -m artifacts_scripts.dedup_methods_bench --sample 200000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from bookgap.cloud import read_embeddings_rows, read_embeddings_store


def load_sample(store_path: Path, n: int, seed: int = 0) -> np.ndarray:
    """Random-row sample from the full-corpus embeddings store.

    Random rows from a 64 GB memory-mapped store are scattered page reads, so
    the sample is cached under /tmp for repeat runs. Prefer
    :func:`load_scope_sample` when approximate coverage is OK -- it gives
    denser duplicate clusters (same-scope works are more often near-duplicates).
    """
    cache = Path(f"/tmp/dedup_bench_sample_{n}_{seed}.npy")
    if cache.exists():
        return np.load(cache)
    total = read_embeddings_store(store_path).shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=min(n, total), replace=False))
    arr = read_embeddings_rows(store_path, idx)
    np.save(cache, arr)
    return arr


def _scope_sizes(emb_root: Path) -> list[tuple[Path, int]]:
    """Return [(scope_dir, total_works), ...] for every scope leaf under emb_root."""
    scope_dirs = sorted({p.parent for p in emb_root.rglob("batch_0_embeddings.npy")})
    sizes = []
    for scope_dir in scope_dirs:
        total = 0
        for f in scope_dir.glob("batch_*_embeddings.npy"):
            total += np.load(f, mmap_mode="r").shape[0]
        sizes.append((scope_dir, total))
    return sizes


def load_scope_sample(
    target_sizes: list[int],
    max_per_scope: int = 200_000,
    min_per_scope: int = 1_000,
    seed: int = 0,
    exclude: Optional[set] = None,
    emb_root: Path = Path("cache/embeddings/avsolatorio/GIST-small-Embedding-v0"),
) -> tuple[np.ndarray, list]:
    """Sample whole scopes (taxonomy leaves) with explicit target sizes.

    For each target size in ``target_sizes``, picks the *closest-sized*
    unpicked scope that's within [min_per_scope, max_per_scope]. This
    guarantees size diversity — without it, a pure random pick over the
    heavy-tailed scope-size distribution tends to yield a glut of tiny
    scopes and no large ones. Sampling is without replacement so
    duplicate target sizes still pull distinct scopes.

    Why this is faster than the random-row path: each scope already lives
    as a handful of sequential ``batch_*_embeddings.npy`` files. Loading
    them is a sequential file scan vs. the scattered page reads that
    dominate ``load_sample``.

    Why this is statistically useful: same-scope works are more often
    near-duplicates than cross-scope works, so scope-based sampling
    yields denser duplicate clusters and tighter recall estimates.
    """
    # No on-disk cache: scope loads are sequential small-file reads
    # (~2-5 s) and caching would either mis-ignore cross-seed exclude
    # state or need a costly hash of it in the key.
    sizes = _scope_sizes(emb_root)
    exclude = exclude or set()
    candidates = [(d, n) for (d, n) in sizes
                  if min_per_scope <= n <= max_per_scope and d not in exclude]
    print(f"  scope pool: {len(candidates):,} scopes within "
          f"[{min_per_scope:,}, {max_per_scope:,}] works "
          f"(of {len(sizes):,} total, {len(exclude)} globally excluded)", flush=True)

    # Shuffle candidates once with the seed so repeated targets don't
    # always pick the same closest scope — ties broken randomly.
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    picked: list[tuple[Path, int]] = []
    available = list(candidates)  # mutable, we'll remove picks
    k_closest = 5  # pick uniformly from the K closest unpicked scopes per target
    for target in target_sizes:
        if not available:
            break
        # Sort by |size - target|, take top-K closest, uniform pick.
        # This gives real per-seed data variance while still maintaining
        # size diversity — a strict argmin would pick the same scope
        # every seed (no data variance at all).
        sorted_by_dist = sorted(
            range(len(available)),
            key=lambda i: abs(available[i][1] - target),
        )
        candidates_top_k = sorted_by_dist[:min(k_closest, len(sorted_by_dist))]
        chosen_idx = int(rng.choice(candidates_top_k))
        picked.append(available.pop(chosen_idx))

    acc: list[np.ndarray] = []
    total = 0
    for scope_dir, n_scope in picked:
        for f in sorted(scope_dir.glob("batch_*_embeddings.npy"),
                        key=lambda p: int(p.name.split("_")[1])):
            acc.append(np.asarray(np.load(f), dtype=np.float32))
        total += n_scope
        # Show which scopes we got
        rel = scope_dir.relative_to(emb_root)
        print(f"    picked {str(rel):<40} {n_scope:>7,} works", flush=True)

    print(f"  total: {total:,} works from {len(picked)} scopes", flush=True)
    arr = np.concatenate(acc, axis=0)
    return arr, [d for d, _ in picked]


def pairs_from_range_search(lims, ids, n_q: int) -> set:
    """Turn FAISS range_search output into a set of (query_idx, neighbour_idx) pairs, self-matches stripped."""
    pairs = set()
    for q in range(n_q):
        for k in range(lims[q], lims[q + 1]):
            nb = int(ids[k])
            if nb != q:
                pairs.add((q, nb))
    return pairs


def fmt_s(s: float) -> str:
    if s < 60: return f"{s:.2f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def m_brute(embeddings: np.ndarray, epsilon: float) -> tuple[float, float, set, int]:
    d = embeddings.shape[1]
    t0 = time.perf_counter()
    idx = faiss.IndexFlatL2(d)
    idx.add(embeddings)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lims, dists, ids = idx.range_search(embeddings, epsilon)
    query_s = time.perf_counter() - t0
    pairs = pairs_from_range_search(lims, ids, len(embeddings))
    return build_s, query_s, pairs, 0


def m_hnsw(embeddings: np.ndarray, epsilon: float, M=16, efC=384, efS=128, batch_size=96) -> tuple:
    """HNSW with batched range_search — single big range_search(emb, eps)
    SIGABRT-crashes for non-trivial sample sizes (FAISS internal buffer
    issue), so we batch like production ``range_search_with_`` does."""
    d = embeddings.shape[1]
    n = len(embeddings)
    t0 = time.perf_counter()
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    idx.hnsw.efConstruction = efC
    idx.hnsw.efSearch = efS
    idx.add(embeddings)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pairs = set()
    for i in range(0, n, batch_size):
        batch = embeddings[i:i + batch_size]
        lims_, dists_, ids_ = idx.range_search(batch, epsilon)
        for j in range(len(batch)):
            q = i + j
            for k in range(lims_[j], lims_[j + 1]):
                nb = int(ids_[k])
                if nb != q:
                    pairs.add((q, nb))
    query_s = time.perf_counter() - t0
    return build_s, query_s, pairs, 0


def m_ivfflat(embeddings: np.ndarray, epsilon: float, nlist_factor=1.0, nprobe=8) -> tuple:
    n, d = embeddings.shape
    nlist = max(int(np.sqrt(n) * nlist_factor), 16)
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(quantizer, d, nlist)
    # Train on a random subset for speed; full-sample training rarely helps
    # once nlist * ~40 training points is reached.
    train_n = min(n, nlist * 40)
    idx.train(embeddings[:train_n])
    idx.add(embeddings)
    idx.nprobe = nprobe
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lims, dists, ids = idx.range_search(embeddings, epsilon)
    query_s = time.perf_counter() - t0
    pairs = pairs_from_range_search(lims, ids, len(embeddings))
    return build_s, query_s, pairs, nlist


def m_lsh_verify(
    embeddings: np.ndarray,
    epsilon: float,
    n_bits: int = 256,
    initial_k: int = 200,
    max_k: int = 8192,
    saturation_threshold: float = 0.05,
) -> tuple:
    """FAISS IndexLSH retrieval with exact-L2 verification and adaptive k expansion.

    Fixed-k LSH drops anything beyond rank k; we add a batched expansion
    loop to recover range-search semantics: after each round at current k,
    any query whose new candidates (positions prev_k..current_k in the
    Hamming-nearest list) still contain at least ``saturation_threshold``
    fraction of in-ball points is re-queried at 2k. Expansion stops when
    the marginal in-ball fraction drops below the threshold (~ball
    exhausted).

    Default ``saturation_threshold=0.05`` (keep expanding while any
    meaningful fraction of the next window is in-ball, not the initial
    0.5 which terminated too aggressively — single-projection IndexLSH's
    Hamming order has enough noise that true dupes can live at high
    Hamming ranks). ``initial_k=200`` starts high enough that dense
    hot points usually converge in one or two rounds.
    """
    n, d = embeddings.shape
    t0 = time.perf_counter()
    lsh = faiss.IndexLSH(d, n_bits)
    lsh.train(embeddings[:min(n, 50000)])
    lsh.add(embeddings)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairs: set = set()
    # Per-query set of already-verified candidate indices. Sparse — most
    # queries see only their first-round candidates. Stored as dict so
    # we only pay the memory cost for queries that actually expand.
    verified: dict[int, set[int]] = {}

    pending = np.arange(n, dtype=np.int64)
    current_k = initial_k

    while len(pending) > 0 and current_k <= max_k:
        _, cand_I = lsh.search(embeddings[pending], current_k)

        new_pending: list[int] = []
        for qi in range(len(pending)):
            q = int(pending[qi])
            row_q = embeddings[q]
            seen = verified.setdefault(q, set())
            new_cands = [int(c) for c in cand_I[qi]
                         if int(c) != -1 and int(c) != q and int(c) not in seen]
            in_ball = 0
            for cand in new_cands:
                diff = row_q - embeddings[cand]
                sq = float(np.dot(diff, diff))
                if sq < epsilon:
                    pairs.add((q, cand))
                    in_ball += 1
                seen.add(cand)
            # Expand iff the marginal in-ball fraction is still a majority,
            # signaling the ε-ball likely extends beyond the current k.
            if new_cands and in_ball >= saturation_threshold * len(new_cands):
                new_pending.append(q)

        pending = np.asarray(new_pending, dtype=np.int64)
        current_k *= 2

    query_s = time.perf_counter() - t0
    # Return n_bits and final expansion depth info for diagnostics.
    return build_s, query_s, pairs, n_bits


def m_simhash_verify(embeddings: np.ndarray, epsilon: float, n_bits=24, n_tables=5) -> tuple:
    """Custom SimHash with L tables. Candidates = union of co-bucket-mates
    across tables, then L2 verify."""
    n, d = embeddings.shape
    t0 = time.perf_counter()
    rng = np.random.default_rng(42)
    # n_tables × n_bits × d: one Gaussian direction per bit per table
    hyperplanes = rng.standard_normal((n_tables, n_bits, d)).astype(np.float32)
    # Normalize each hyperplane for numerical stability
    hyperplanes /= np.linalg.norm(hyperplanes, axis=2, keepdims=True)
    # Compute signatures: n × n_tables × n_bits → bits
    # Shape (n, n_tables, n_bits) = embeddings @ hyperplanes.T per table
    sigs = np.zeros((n, n_tables), dtype=np.uint32)
    for t in range(n_tables):
        proj = embeddings @ hyperplanes[t].T  # (n, n_bits)
        bits = (proj > 0).astype(np.uint32)
        # Pack n_bits bits into a uint (up to 32)
        packed = np.zeros(n, dtype=np.uint32)
        for b in range(n_bits):
            packed |= bits[:, b] << b
        sigs[:, t] = packed
    # Build per-table buckets: signature → list of point indices
    tables: list[dict[int, list[int]]] = []
    for t in range(n_tables):
        bucket: dict[int, list[int]] = {}
        for i in range(n):
            bucket.setdefault(int(sigs[i, t]), []).append(i)
        tables.append(bucket)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairs = set()
    for t in range(n_tables):
        for sig, indices in tables[t].items():
            if len(indices) < 2:
                continue
            # All pairs in bucket are candidates
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    diff = embeddings[a] - embeddings[b]
                    sq = float(np.dot(diff, diff))
                    if sq < epsilon:
                        pairs.add((a, b))
                        pairs.add((b, a))
    query_s = time.perf_counter() - t0
    return build_s, query_s, pairs, n_bits * n_tables


def m_sorted_projection(embeddings: np.ndarray, epsilon: float, n_projs=8, window=64) -> tuple:
    """Project onto random 1-D directions, sort, sliding window verify."""
    n, d = embeddings.shape
    rng = np.random.default_rng(7)
    directions = rng.standard_normal((n_projs, d)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    t0 = time.perf_counter()
    # Project & sort per direction
    sorted_orderings = []
    for p in range(n_projs):
        proj = embeddings @ directions[p]
        order = np.argsort(proj)
        sorted_orderings.append(order)
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairs = set()
    for order in sorted_orderings:
        for i in range(n):
            a = int(order[i])
            for off in range(1, window + 1):
                j = i + off
                if j >= n:
                    break
                b = int(order[j])
                diff = embeddings[a] - embeddings[b]
                sq = float(np.dot(diff, diff))
                if sq < epsilon:
                    pairs.add((a, b))
                    pairs.add((b, a))
    query_s = time.perf_counter() - t0
    return build_s, query_s, pairs, n_projs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path,
                    default=Path("cache/embeddings_npy/2f77728932c4b159/6bcf5b5e02946b7e/embeddings.npy"))
    ap.add_argument("--sample", type=int, default=200_000,
                    help="Target sample size for random-row mode (ignored with --scope-sample).")
    ap.add_argument("--scope-sample", action="store_true",
                    help="Sample whole scopes (per-scope batch cache) instead of random "
                         "rows. Much faster load + denser duplicate clusters.")
    ap.add_argument("--target-sizes", type=str,
                    default="10000,50000,100000,100000,200000,200000",
                    help="For --scope-sample: comma-separated target scope sizes. Each "
                         "target pulls the closest-sized unpicked scope, so repeated "
                         "targets still get distinct scopes and the final sample is "
                         "size-diverse rather than dominated by tiny scopes.")
    ap.add_argument("--max-per-scope", type=int, default=200_000,
                    help="For --scope-sample: ignore scopes larger than this.")
    ap.add_argument("--min-per-scope", type=int, default=1_000,
                    help="For --scope-sample: ignore scopes smaller than this.")
    ap.add_argument("--seeds", type=str, default="0",
                    help="Comma-separated sample-seeds to run. Multiple seeds give "
                         "variance estimates and more total pairs to evaluate against, "
                         "without the O(N²) cost of scaling up the sample itself.")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--full-n", type=int, default=42_137_307)
    ap.add_argument("--skip-hnsw", action="store_true")
    ap.add_argument("--include-sortedproj", action="store_true",
                    help="Include the known-bad SortedProj method (slow, low recall).")
    ap.add_argument("--with-gt", action="store_true",
                    help="Also compute exact brute-force GT for one sanity-check seed. "
                         "Otherwise we skip GT entirely and use the union of all methods' "
                         "pair sets as the working ground truth (valid because each method's "
                         "pairs are exact-distance-verified, so the union has no false positives).")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    method_pair_sets: dict[str, list[set]] = {}  # method_name -> per-seed pair sets
    method_times: dict[str, list[tuple[float, float]]] = {}  # method_name -> [(build_s, query_s), ...]

    # Global exclude set for scope-sample mode — ensures each seed picks
    # strictly disjoint scopes, which is what we want for statistical
    # independence across seeds. Without-replacement across all seeds, not
    # just within one.
    globally_excluded_scopes: set = set()

    for seed_i, seed in enumerate(seeds):
        print(f"\n{'='*80}\nSeed {seed} ({seed_i + 1}/{len(seeds)})\n{'='*80}")
        mode = "scope" if args.scope_sample else "random-row"
        print(f"Loading ({mode}) sample embeddings (seed={seed}) ...")
        t0 = time.perf_counter()
        if args.scope_sample:
            target_sizes = [int(s) for s in args.target_sizes.split(",")]
            emb, picked_dirs = load_scope_sample(
                target_sizes,
                max_per_scope=args.max_per_scope,
                min_per_scope=args.min_per_scope,
                seed=seed,
                exclude=globally_excluded_scopes,
            )
            globally_excluded_scopes.update(picked_dirs)
        else:
            emb = load_sample(args.store, args.sample, seed=seed)
        print(f"  loaded in {fmt_s(time.perf_counter() - t0)}, shape={emb.shape}")

        run_bench_seed(args, emb, seed, seed_i, method_pair_sets, method_times)

    print_aggregate_report(args, method_pair_sets, method_times, len(seeds))


def run_bench_seed(args, emb, seed, seed_i, method_pair_sets, method_times):
    # Optional GT for sanity at one seed only
    gt_pairs: set = set()
    if args.with_gt and seed_i == 0:
        gt_cache = Path(f"/tmp/dedup_bench_gt_{args.sample}_{args.epsilon}_{seed}.npz")
        if gt_cache.exists():
            print(f"\nLoading cached GT from {gt_cache}")
            gt_pairs = set(map(tuple, np.load(gt_cache)["pairs"]))
            print(f"  {len(gt_pairs):,} pairs loaded")
        else:
            print(f"\nGround truth (IndexFlatL2 range_search) on {len(emb):,} points × {len(emb):,} queries ...")
            _, gt_query, gt_pairs, _ = m_brute(emb, args.epsilon)
            print(f"  query {fmt_s(gt_query)}, true pairs (ordered): {len(gt_pairs):,}")
            np.savez(gt_cache, pairs=np.array(list(gt_pairs), dtype=np.int64))

    n = len(emb)
    full_n = args.full_n
    lin_ratio = full_n / n
    log_ratio = np.log(full_n) / np.log(n)

    def safe_run(name, fn, *fn_args, extra="", **fn_kwargs):
        try:
            build_s, query_s, pairs, _ = fn(emb, args.epsilon, *fn_args, **fn_kwargs)
        except Exception as e:
            print(f"{name:<28} | ERROR: {type(e).__name__}: {e}", flush=True)
            return
        except BaseException as e:
            print(f"{name:<28} | FATAL: {type(e).__name__}: {e}", flush=True)
            raise
        method_pair_sets.setdefault(name, []).append(pairs)
        method_times.setdefault(name, []).append((build_s, query_s))
        proj = (build_s + query_s) * lin_ratio * log_ratio
        if gt_pairs:
            recall = len(pairs & gt_pairs) / max(len(gt_pairs), 1)
            print(f"{name:<28} | {fmt_s(build_s):>8} {fmt_s(query_s):>8} | "
                  f"pairs={len(pairs):<7} gt_recall={recall*100:>6.2f}% | "
                  f"{fmt_s(proj):>10}  {extra}", flush=True)
        else:
            print(f"{name:<28} | {fmt_s(build_s):>8} {fmt_s(query_s):>8} | "
                  f"pairs={len(pairs):<7} | "
                  f"{fmt_s(proj):>10}  {extra}", flush=True)

    print()
    print(f"{'method':<28} | {'build':>8} {'query':>8} | "
          f"{'result':>22} | "
          f"{'proj 42M':>10}")
    print("-" * 95)

    # Cheap-first ordering. Each method family is swept across a spread of
    # quality/time settings so we can trace a proper Pareto frontier
    # (time vs recall) and answer "at time budget X, which method is
    # best?". HNSW comes last — it's the slowest — and can be skipped
    # entirely via --skip-hnsw for scaling experiments.
    for nf, np_ in [(1.0, 4), (1.0, 8), (1.0, 16), (2.0, 8)]:
        safe_run(f"IVFFlat np={np_} nf={nf}", m_ivfflat, nf, np_)

    for nb in [128, 256, 512]:
        safe_run(f"IndexLSH {nb}b + verify", m_lsh_verify, nb)

    # Higher k = fewer, tinier buckets = much less per-pair verification. Low k
    # (20-24) is dominated by O(bucket_size²) Python pair iteration; sweep
    # higher-k settings which should trace out the true Pareto frontier.
    for nb, nt in [(24, 5), (28, 5), (28, 10), (32, 10), (32, 20)]:
        safe_run(f"SimHash k={nb} L={nt}", m_simhash_verify, nb, nt)

    # SortedProj with random Gaussian 1-D projections was confirmed as a
    # dead end for this workload (cos=0.995 pairs land within a narrow 1-D
    # offset that's a tiny fraction of the sorted array's span — we saw
    # 15-48% recall at high query cost). Call it out by reference only.
    if args.include_sortedproj:
        safe_run("SortedProj P=16 W=256 (ref)", m_sorted_projection, 16, 256)

    if not args.skip_hnsw:
        for M, efC, efS in [(16, 100, 32), (16, 100, 64), (16, 200, 64),
                            (16, 200, 128), (16, 384, 128)]:
            tag = f"HNSW M={M} efC={efC} efS={efS}"
            safe_run(tag, m_hnsw, M=M, efC=efC, efS=efS)


def print_aggregate_report(args, method_pair_sets, method_times, n_seeds):
    """Aggregate per-method recall across seeds, using the per-seed union
    of all methods' pair sets as the working ground truth.

    Each method's pair set is exact-distance-verified (candidate methods
    verify inline, IndexFlat-based ones return only in-radius pairs), so
    the union has no false positives. The only way a true pair escapes the
    union is if *every* method misses it, which becomes vanishingly likely
    given the algorithmic diversity (IVF centroid buckets, random-hyperplane
    SimHash, FAISS IndexLSH bit-signatures, HNSW graph walks). Pairs
    missed-by-all only surface when another method joins the ensemble —
    at which point earlier methods' scores retroactively drop.
    """
    per_seed_union = [set() for _ in range(n_seeds)]
    for _, seed_pairsets in method_pair_sets.items():
        for si, ps in enumerate(seed_pairsets):
            per_seed_union[si] |= ps

    print("\n" + "=" * 98)
    print(f"AGGREGATE across {n_seeds} seed(s); truth = per-seed union of all methods' pair sets")
    print(f"  per-seed union sizes: {[len(u) for u in per_seed_union]}  "
          f"(total {sum(len(u) for u in per_seed_union)})")
    print("=" * 98)

    rows = []
    for name, seed_pairsets in method_pair_sets.items():
        recalls, missed_counts, unique_counts = [], [], []
        for si, ps in enumerate(seed_pairsets):
            u = per_seed_union[si]
            recalls.append(len(ps & u) / max(len(u), 1))
            missed_counts.append(len(u - ps))
            other_union = set()
            for n2, sps2 in method_pair_sets.items():
                if n2 != name:
                    other_union |= sps2[si]
            unique_counts.append(len(ps - other_union))
        times = method_times[name]
        avg_build = float(np.mean([t[0] for t in times]))
        avg_query = float(np.mean([t[1] for t in times]))
        avg_recall = float(np.mean(recalls))
        std_recall = float(np.std(recalls)) if len(recalls) > 1 else 0.0
        avg_missed = float(np.mean(missed_counts))
        avg_unique = float(np.mean(unique_counts))
        lin_ratio = args.full_n / args.sample
        log_ratio = np.log(args.full_n) / np.log(args.sample)
        proj = (avg_build + avg_query) * lin_ratio * log_ratio
        rows.append((name, avg_recall, std_recall, avg_missed, avg_unique, avg_build, avg_query, proj))

    rows.sort(key=lambda r: (-r[1], r[7]))

    print(f"{'method':<28} | {'recall':>8} {'± std':>7} | "
          f"{'missed':>7} {'unique':>7} | "
          f"{'build':>8} {'query':>8} {'proj 42M':>10}")
    print("-" * 98)
    for name, rec, std, missed, unique, build, query, proj in rows:
        print(f"{name:<28} | {rec*100:>7.2f}% {std*100:>6.2f}% | "
              f"{missed:>7.1f} {unique:>7.1f} | "
              f"{fmt_s(build):>8} {fmt_s(query):>8} "
              f"{fmt_s(proj):>10}")


if __name__ == "__main__":
    main()
