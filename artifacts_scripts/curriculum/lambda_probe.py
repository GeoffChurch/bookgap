"""λ-sweep probe for multi-view spectral clustering co-regularisation strength.

Loads cached artifacts from an existing curriculum run directory (citation
graph, filtered work IDs, metadata), fetches embeddings from the Cloud, then
sweeps co_reg_lambda over an explicit grid PLUS heuristic-predicted values.
Each λ is scored via modularity Q, conductance, citation-flow capture, Gini,
max-fraction, and fragment count.  The script reports which heuristic's
predicted λ best tracks the empirical Q-optimal λ.

Joint-embedding spectral-gap logging (SVD gap of [U1|U2] at each λ) is
intentionally omitted in v1 — the current ``cluster_with_multi_view_spectral``
API only returns labels, not the intermediate U1/U2 matrices.  Re-exposing
those would require an API change; deferred until the heuristic analysis
motivates it.

Usage::

    python -m artifacts_scripts.curriculum.lambda_probe \\
        --run-dir artifacts/curriculum/<dir> \\
        --output artifacts/lambda_probe/<name>.json \\
        [--csls-k 10] \\
        [--citation-mode weighted_unified] \\
        [--max-iters 30] \\
        [--tol 1e-5] \\
        [--lambda-grid 0.01,0.03,0.1,0.3,0.5,1.0,2.0,5.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from bookgap.util import LOGGER

from artifacts_scripts.curriculum.multi_view_spectral import (
    build_citation_similarity_view,
    build_embedding_similarity_view,
    cluster_with_multi_view_spectral,
    _normalized_laplacian,
)
from artifacts_scripts.curriculum.compare_clusterings import (
    modularity_q,
    average_conductance,
    citation_flow_capture,
    cluster_size_gini,
    max_cluster_fraction,
    fragment_cluster_count,
)


# ---------------------------------------------------------------------------
# Cloud construction helpers
# ---------------------------------------------------------------------------

def _build_cloud_from_metadata(metadata: Dict[str, Any]):
    """Construct a Cloud matching the metadata's cloud_name / embedder / denoiser.

    Both QC and Robotics runs use domain=3, work_types=[book,article,book-chapter,preprint],
    GIST-small, denoiser=NONE.  We hard-code this derivation but log a warning
    if the metadata indicates something different.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from bookgap.configs.cloud.cloud_config import CloudConfig

    cloud_name = metadata.get("cloud_name", "")
    embedder_str = metadata.get("embedder", "")
    denoiser = metadata.get("denoiser", "NONE")

    # Warn if anything unexpected
    if "domain=3" not in cloud_name:
        LOGGER.warning(
            f"cloud_name '{cloud_name}' doesn't contain 'domain=3' — using default CloudConfig anyway"
        )
    if "GIST-small" not in embedder_str and "GIST_small" not in embedder_str:
        LOGGER.warning(
            f"embedder '{embedder_str}' doesn't look like GIST-small — using default CloudConfig anyway"
        )
    if denoiser != "NONE":
        LOGGER.warning(
            f"metadata denoiser='{denoiser}' != 'NONE'; probe does not apply denoiser "
            f"— embeddings may differ slightly from the original clustering run"
        )

    # Default CloudConfig already encodes domain=3, all work types, GIST-small, NONE denoiser.
    cfg = OmegaConf.structured(CloudConfig())
    cloud = instantiate(cfg)
    LOGGER.info(f"Cloud constructed: {cloud.name()}")
    return cloud


def _fetch_embeddings_for_ids(
    cloud,
    filtered_work_ids: List[str],
    all_work_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """Return (n_filtered, D) embedding matrix aligned to filtered_work_ids order.

    Uses :func:`bookgap.cloud.read_embeddings_rows` to read only the needed
    rows from the embeddings Parquet file. ``cloud.embeddings()`` would load
    the full ~62 GB embeddings array into memory (40 M rows × 384 dims × 4 B)
    just to slice 5 k of them out — that OOM-kills the box. The targeted
    read here allocates only ``len(filtered_work_ids) × D`` (a few MB).

    If ``all_work_ids`` is pre-loaded (e.g. from a prior ``cloud.work_ids()``
    call), pass it in to avoid a second lookup.

    Note: applies the cloud's denoiser if non-trivial. NoneDenoiser is a
    no-op so QC/Robotics (denoiser=NONE) runs go through this path
    unchanged. For a non-trivial denoiser, we'd need the denoiser's
    projection (cloud.denoiser_projection()) to apply consistently with
    the original clustering run; we currently warn rather than implement
    this since both target corpora use NONE.
    """
    from bookgap.cloud import (
        ensure_cloud_data,
        read_embeddings_rows,
        _embeddings_store_path,
    )

    if all_work_ids is None:
        LOGGER.info("Loading all cloud work IDs (cached)…")
        all_work_ids = cloud.work_ids()
        LOGGER.info(f"  {len(all_work_ids)} total works in cloud")

    id_to_idx = {wid: i for i, wid in enumerate(all_work_ids)}
    # work_ids_position[k] = position of filtered_work_ids[k] in all_work_ids
    # (which is the same order as cloud.embeddings()).
    work_ids_position: List[int] = []
    missing_rows: List[int] = []
    for k, wid in enumerate(filtered_work_ids):
        idx = id_to_idx.get(wid)
        if idx is None:
            missing_rows.append(k)
            work_ids_position.append(-1)
        else:
            work_ids_position.append(idx)
    if missing_rows:
        LOGGER.warning(
            f"{len(missing_rows)}/{len(filtered_work_ids)} filtered IDs not found in cloud"
        )

    # Map from work-ids position → physical embeddings-parquet row position.
    # cloud.embeddings() effectively returns embeddings_parquet[retained_positions];
    # so to read row k of cloud.embeddings(), we need parquet row retained_positions[k].
    LOGGER.info("Resolving embeddings parquet path + retained positions…")
    corpus_fp, sorted_idxs, retained_positions = ensure_cloud_data(cloud)
    emb_path = _embeddings_store_path(corpus_fp, cloud.embedder)
    LOGGER.info(f"  emb_path={emb_path}")
    LOGGER.info(f"  retained_positions: {len(retained_positions)} (matches len(work_ids))")

    # Build the parquet-row positions for the works we want, dropping misses.
    # read_embeddings_rows requires ascending order; we'll sort, read, then
    # invert the permutation to recover filtered_work_ids order.
    valid_indices = [k for k, p in enumerate(work_ids_position) if p >= 0]
    parquet_positions = np.array(
        [retained_positions[work_ids_position[k]] for k in valid_indices],
        dtype=np.uint64,
    )
    # Sort ascending and remember the permutation.
    sort_perm = np.argsort(parquet_positions, kind="stable")
    sorted_parquet_positions = parquet_positions[sort_perm]

    LOGGER.info(
        f"Reading {len(sorted_parquet_positions)} embeddings rows from parquet "
        f"(sorted)…"
    )
    sorted_embeddings = read_embeddings_rows(emb_path, sorted_parquet_positions)
    D = sorted_embeddings.shape[1]
    LOGGER.info(f"  read embeddings shape: {sorted_embeddings.shape}")

    # Apply denoiser if non-trivial; NoneDenoiser is a no-op (handled below
    # via the metadata.denoiser=='NONE' check at probe entry).
    # We don't apply the denoiser here for non-NONE configs — the warning
    # in _build_cloud_from_metadata covers that case.

    # Invert the sort to put rows back in `filtered_work_ids[valid_indices]` order.
    inv_perm = np.empty_like(sort_perm)
    inv_perm[sort_perm] = np.arange(len(sort_perm))
    valid_embeddings = sorted_embeddings[inv_perm]

    # Allocate output and fill in valid rows; missing rows stay zero.
    out = np.zeros((len(filtered_work_ids), D), dtype=np.float32)
    for slot, k in enumerate(valid_indices):
        out[k] = valid_embeddings[slot]
    return out


# ---------------------------------------------------------------------------
# Core probe logic
# ---------------------------------------------------------------------------

def _compute_heuristics(
    eigvals_emb: np.ndarray,
    eigvals_cite: np.ndarray,
    K: int,
) -> Dict[str, float]:
    """Compute λ-heuristic predictions from the two views' Laplacian spectra.

    Each heuristic is computed independently per view then averaged.
    """
    def _per_view(ev: np.ndarray) -> Dict[str, float]:
        n = len(ev)
        return {
            "mean_bottom_K": float(ev[:K].mean()),
            "Kth": float(ev[K - 1]),
            "trace_per_n": float(ev.sum() / n),
            "op_norm_per_K": float(ev[-1] / K),
            "gap_K": float(ev[K] - ev[K - 1]) if K < n else float(ev[K - 1]),
            "sqrt_gap_product": float(math.sqrt(abs(ev[K] * ev[K - 1]))) if K < n else float(ev[K - 1]),
        }

    h_emb = _per_view(eigvals_emb)
    h_cite = _per_view(eigvals_cite)
    return {
        name: (h_emb[name] + h_cite[name]) / 2.0
        for name in h_emb
    }


def _score_lambda(
    W_emb: np.ndarray,
    W_cite: np.ndarray,
    sub_adj: sp.spmatrix,
    K: int,
    lam: float,
    max_iters: int,
    tol: float,
) -> Dict[str, Any]:
    labels = cluster_with_multi_view_spectral(
        W_emb, W_cite,
        n_clusters=K,
        co_reg_lambda=lam,
        max_iters=max_iters,
        tol=tol,
        seed=42,
    )
    sizes = np.bincount(labels, minlength=K).tolist()
    total = int(sum(sizes))
    return {
        "lambda": float(lam),
        "Q": modularity_q(sub_adj, labels),
        "conductance": average_conductance(sub_adj, labels),
        "flow_capture": citation_flow_capture(sub_adj, labels),
        "gini": cluster_size_gini(sizes),
        "max_fraction": max_cluster_fraction(sizes, total=total),
        "fragments": fragment_cluster_count(sizes, threshold=20),
    }


def run_probe(
    run_dir: Path,
    output_path: Path,
    *,
    csls_k: int = 10,
    citation_mode: str = "weighted_unified",
    max_iters: int = 30,
    tol: float = 1e-5,
    lambda_grid: List[float],
) -> None:
    LOGGER.info(f"=== λ probe: {run_dir} ===")

    # ------------------------------------------------------------------
    # 1. Load artifacts
    # ------------------------------------------------------------------
    LOGGER.info("Loading citation_graph.npz …")
    sub_adj = sp.load_npz(run_dir / "citation_graph.npz")
    LOGGER.info(f"  sub_adj shape={sub_adj.shape}, nnz={sub_adj.nnz}")

    with open(run_dir / "filtered_work_ids.json") as f:
        filtered_work_ids: List[str] = json.load(f)
    LOGGER.info(f"  filtered_work_ids: {len(filtered_work_ids)}")

    with open(run_dir / "metadata.json") as f:
        metadata: Dict[str, Any] = json.load(f)

    K = int(metadata["desired_n_topics"])
    LOGGER.info(
        f"  K={K}, cloud_name={metadata['cloud_name']}, "
        f"embedder={metadata['embedder']}, denoiser={metadata['denoiser']}"
    )
    LOGGER.info(
        f"  target_work_ids={metadata.get('target_work_ids', [])}, "
        f"source_work_ids={metadata.get('source_work_ids', [])}"
    )

    n_samples = len(filtered_work_ids)

    # ------------------------------------------------------------------
    # 2. Load embeddings via Cloud (also used to extract subgraph)
    # ------------------------------------------------------------------
    cloud = _build_cloud_from_metadata(metadata)

    # Map filtered_work_ids → corpus indices so we can slice the full adj.
    LOGGER.info("Mapping filtered work IDs to corpus indices…")
    all_work_ids = cloud.work_ids()
    id_to_idx = {wid: i for i, wid in enumerate(all_work_ids)}
    top_k_indices = np.array(
        [id_to_idx[wid] for wid in filtered_work_ids if wid in id_to_idx],
        dtype=np.int64,
    )
    n_found = len(top_k_indices)
    if n_found < n_samples:
        LOGGER.warning(
            f"{n_samples - n_found} filtered IDs not found in cloud; "
            f"subgraph will have {n_found} nodes"
        )
        n_samples = n_found

    # Check if citation_graph.npz is already the subgraph (5k×5k) or the full graph.
    if sub_adj.shape == (n_samples, n_samples):
        LOGGER.info("citation_graph.npz is already the filtered subgraph — using as-is")
    else:
        LOGGER.info(
            f"citation_graph.npz is full corpus ({sub_adj.shape}); "
            f"extracting {n_samples}×{n_samples} subgraph…"
        )
        sub_adj = sub_adj[top_k_indices][:, top_k_indices]
        LOGGER.info(f"  subgraph: shape={sub_adj.shape}, nnz={sub_adj.nnz}")

    embeddings = _fetch_embeddings_for_ids(cloud, filtered_work_ids, all_work_ids=all_work_ids)
    LOGGER.info(f"Embeddings for filtered works: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 3. Build views
    # ------------------------------------------------------------------
    LOGGER.info(f"Building embedding similarity view (csls_k={csls_k}) …")
    W_emb = build_embedding_similarity_view(embeddings, csls_k=csls_k)
    LOGGER.info(f"Building citation similarity view (mode={citation_mode}, csls_k={csls_k}) …")
    W_cite = build_citation_similarity_view(sub_adj, csls_k=csls_k, mode=citation_mode)

    # ------------------------------------------------------------------
    # 4. Compute Laplacian eigenvalues (full spectrum via eigvalsh)
    # ------------------------------------------------------------------
    LOGGER.info("Computing Laplacian spectra …")
    L_emb = _normalized_laplacian(W_emb)
    L_cite = _normalized_laplacian(W_cite)
    eigvals_emb = np.linalg.eigvalsh(L_emb)  # ascending
    eigvals_cite = np.linalg.eigvalsh(L_cite)
    LOGGER.info(
        f"  emb eigvals[:5]={eigvals_emb[:5].tolist()}, cite eigvals[:5]={eigvals_cite[:5].tolist()}"
    )

    # ------------------------------------------------------------------
    # 5. Heuristics
    # ------------------------------------------------------------------
    heuristics = _compute_heuristics(eigvals_emb, eigvals_cite, K)
    LOGGER.info("Heuristic λ predictions:")
    for name, val in heuristics.items():
        LOGGER.info(f"  {name:22s} {val:.4f}")

    # ------------------------------------------------------------------
    # 6. Build combined lambda list (grid + heuristics, sorted, deduped)
    # ------------------------------------------------------------------
    all_lambdas_raw = list(lambda_grid) + list(heuristics.values())
    # Keep λ=0 from the explicit grid (useful baseline); filter out negative
    # heuristic predictions (which arise when eigenvalues are near-zero noise).
    # Grid values pass through as-is; heuristic values are rounded to 6 sig figs.
    grid_set = set(round(v, 6) for v in lambda_grid)
    heuristic_lams = set(round(v, 6) for v in heuristics.values() if v > 0)
    all_lambdas: List[float] = sorted(grid_set | heuristic_lams)
    LOGGER.info(f"Total λ values to sweep: {len(all_lambdas)} — {all_lambdas}")

    # ------------------------------------------------------------------
    # 7. Sweep
    # ------------------------------------------------------------------
    grid_results: List[Dict[str, Any]] = []
    for lam in all_lambdas:
        LOGGER.info(f"  λ={lam:.5f} …")
        result = _score_lambda(W_emb, W_cite, sub_adj, K, lam, max_iters, tol)
        grid_results.append(result)
        LOGGER.info(
            f"    Q={result['Q']:.4f}  conductance={result['conductance']:.4f}  "
            f"gini={result['gini']:.4f}  max_frac={result['max_fraction']:.4f}  "
            f"fragments={result['fragments']}"
        )

    # ------------------------------------------------------------------
    # 8. Best λ by Q
    # ------------------------------------------------------------------
    best_result = max(grid_results, key=lambda r: r["Q"])
    best_lambda = best_result["lambda"]
    best_Q = best_result["Q"]
    LOGGER.info(f"Empirical optimum: λ={best_lambda}, Q={best_Q:.4f}")

    # ------------------------------------------------------------------
    # 9. Heuristic performance (find Q at closest swept λ)
    # ------------------------------------------------------------------
    lambda_to_result = {r["lambda"]: r for r in grid_results}

    def _find_Q_at(predicted_lam: float) -> Optional[float]:
        # Find the swept λ closest to predicted_lam
        if predicted_lam <= 0:
            return None
        best_swept = min(all_lambdas, key=lambda v: abs(v - predicted_lam))
        return lambda_to_result[best_swept]["Q"]

    heuristic_performance: Dict[str, Dict[str, Any]] = {}
    for name, predicted_lam in heuristics.items():
        q_at = _find_Q_at(predicted_lam)
        heuristic_performance[name] = {
            "predicted_lambda": float(predicted_lam),
            "Q_at_predicted": float(q_at) if q_at is not None else None,
            "delta_to_optimum": float(q_at - best_Q) if q_at is not None else None,
        }

    # ------------------------------------------------------------------
    # 10. Corpus name
    # ------------------------------------------------------------------
    run_name = run_dir.name
    target_ids = [t.split("/")[-1] for t in metadata.get("target_work_ids", [])]
    qc_ids = {"W2020725861", "W2132764286", "W3038067977"}
    robotics_ids = {"W2476752140", "W2766808627", "W3021768497"}
    if set(target_ids) & qc_ids:
        corpus_name = "QC"
    elif set(target_ids) & robotics_ids:
        corpus_name = "Robotics"
    else:
        corpus_name = "Other"

    # ------------------------------------------------------------------
    # 11. Assemble JSON output
    # ------------------------------------------------------------------
    output = {
        "run_dir": str(run_dir),
        "corpus_name": corpus_name,
        "n_samples": n_samples,
        "K": K,
        "csls_k": csls_k,
        "citation_mode": citation_mode,
        "spectra_summary": {
            "eigvals_emb_first10": eigvals_emb[:10].tolist(),
            "eigvals_emb_last10": eigvals_emb[-10:].tolist(),
            "eigvals_cite_first10": eigvals_cite[:10].tolist(),
            "eigvals_cite_last10": eigvals_cite[-10:].tolist(),
        },
        "heuristics": heuristics,
        "grid_results": grid_results,
        "best_lambda_by_Q": float(best_lambda),
        "best_Q": float(best_Q),
        "heuristic_performance": heuristic_performance,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    LOGGER.info(f"Saved JSON output to {output_path}")

    # ------------------------------------------------------------------
    # 12. Print summary table
    # ------------------------------------------------------------------
    _print_summary(output, lambda_grid=lambda_grid)


def _print_summary(output: Dict[str, Any], lambda_grid: List[float]) -> None:
    corpus = output["corpus_name"]
    n = output["n_samples"]
    K = output["K"]
    mode = output["citation_mode"]
    csls_k = output["csls_k"]

    print(f"\n=== λ probe results: {corpus} ===")
    print(f"n_samples={n}, K={K}, citation_mode={mode}, csls_k={csls_k}")

    ev_emb = output["spectra_summary"]["eigvals_emb_first10"]
    ev_cite = output["spectra_summary"]["eigvals_cite_first10"]
    print(f"\nSpectra (bottom 5 eigvals):")
    print(f"  emb:  {[round(v, 4) for v in ev_emb[:5]]}")
    print(f"  cite: {[round(v, 4) for v in ev_cite[:5]]}")

    print(f"\nHeuristics' λ predictions:")
    for name, val in output["heuristics"].items():
        print(f"  {name:<22s} {val:.4f}")

    grid_lambda_set = set(round(v, 6) for v in lambda_grid)
    best_lam = output["best_lambda_by_Q"]
    print(f"\nGrid sweep (λ, Q, conductance, max_fraction):")
    for r in output["grid_results"]:
        marker = "   <-- best Q" if abs(r["lambda"] - best_lam) < 1e-9 else ""
        in_grid = "*" if round(r["lambda"], 6) in grid_lambda_set else "h"
        print(
            f"  [{in_grid}] {r['lambda']:8.4f}   Q={r['Q']:.4f}   "
            f"c={r['conductance']:.4f}   max={r['max_fraction']:.4f}{marker}"
        )

    print(f"\nEmpirical optimum: λ={best_lam:.4f}, Q={output['best_Q']:.4f}")

    # Best heuristic = smallest |delta_to_optimum|
    hperf = output["heuristic_performance"]
    ranked = sorted(
        [(n, v) for n, v in hperf.items() if v["delta_to_optimum"] is not None],
        key=lambda kv: abs(kv[1]["delta_to_optimum"]),
    )
    if ranked:
        best_h, best_v = ranked[0]
        print(
            f"Heuristic that best matches: {best_h} "
            f"(predicted λ={best_v['predicted_lambda']:.4f}, "
            f"Q={best_v['Q_at_predicted']:.4f}, "
            f"delta={best_v['delta_to_optimum']:+.4f})"
        )
        print("\nAll heuristics ranked by |delta|:")
        for hname, hval in ranked:
            print(
                f"  {hname:<22s} pred={hval['predicted_lambda']:.4f}  "
                f"Q={hval['Q_at_predicted']:.4f}  "
                f"delta={hval['delta_to_optimum']:+.4f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)-8s | %(funcName)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Sweep co_reg_lambda for multi-view spectral clustering and score each λ."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to an existing curriculum run directory containing "
             "citation_graph.npz, filtered_work_ids.json, metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for JSON results (parent dirs created automatically).",
    )
    parser.add_argument(
        "--csls-k",
        type=int,
        default=10,
        help="CSLS neighbourhood size (default: 10).",
    )
    parser.add_argument(
        "--citation-mode",
        type=str,
        default="weighted_unified",
        choices=["binary", "weighted_unified", "weighted_per_direction"],
        help="Citation Jaccard mode (default: weighted_unified).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="Max alternation iterations for the co-regularized solver (default: 30).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Subspace-distance convergence tolerance (default: 1e-5).",
    )
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default="0.01,0.03,0.1,0.3,0.5,1.0,2.0,5.0",
        help="Comma-separated explicit λ grid values (default: 0.01,0.03,0.1,0.3,0.5,1.0,2.0,5.0).",
    )
    args = parser.parse_args()

    lambda_grid = [float(v) for v in args.lambda_grid.split(",")]

    run_probe(
        run_dir=args.run_dir,
        output_path=args.output,
        csls_k=args.csls_k,
        citation_mode=args.citation_mode,
        max_iters=args.max_iters,
        tol=args.tol,
        lambda_grid=lambda_grid,
    )


if __name__ == "__main__":
    main()
