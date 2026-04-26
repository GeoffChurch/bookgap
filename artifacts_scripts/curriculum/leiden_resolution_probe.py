"""Find a Leiden resolution that yields a target community count.

The default RBConfigurationVertexPartition resolution=1.0 produces
hundreds of communities on the citation subgraph of the top-k
relevance-filtered works, far more than the curriculum's
``desired_n_topics``. The frontier-adjacency PageRank step then OOMs
trying to allocate ``(n_corpus, n_communities)``.

This probe sweeps resolution values, runs Leiden on the citation
subgraph extracted via the curriculum's standard rank-before-clustering
pipeline, and reports the resulting community count for each resolution.
Pick the value closest to ``desired_n_topics`` for the production run.

Reuses the cached weighted_bisecting __24 forward PageRank vector to
avoid re-running PR on the full corpus (PR is the expensive part of
the curriculum; clustering is fast).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.curriculum.build_curriculum import _cluster_with_leiden


def run(
    run_dir: Path,
    *,
    resolutions: List[float],
    seed: int = 42,
) -> None:
    print(f"=== Leiden resolution probe on {run_dir} ===")
    print("  loading citation graph and filtered work IDs...")
    adj_full = sp.load_npz(run_dir / "citation_graph.npz")
    with open(run_dir / "filtered_work_ids.json") as f:
        filtered_ids = json.load(f)
    print(f"  full citation graph: {adj_full.shape}")
    print(f"  filtered top-k: {len(filtered_ids)}")

    # Map filtered work IDs back to corpus indices via cloud.
    print("  loading cloud to map work_ids -> corpus indices...")
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    from bookgap.configs.cloud.cloud_config import CloudConfig

    cfg = OmegaConf.structured(CloudConfig())
    cloud = instantiate(cfg)
    work_ids = cloud.work_ids()
    id_to_idx = {wid: i for i, wid in enumerate(work_ids)}
    top_k_indices = np.array(
        [id_to_idx[wid] for wid in filtered_ids],
        dtype=np.int64,
    )
    print(f"  mapped {len(top_k_indices)} filtered IDs to corpus indices")

    print("  extracting subgraph...")
    sub_adj = adj_full[top_k_indices][:, top_k_indices]
    print(f"  subgraph: {sub_adj.shape}, nnz={sub_adj.nnz}")

    print()
    print(f"  {'resolution':>11}  {'n_communities':>14}  {'top10_sizes'}")
    for res in resolutions:
        membership, n_comm = _cluster_with_leiden(
            sub_adj, resolution=res, seed=seed,
        )
        sizes = np.bincount(membership)
        top10 = sorted(sizes.tolist(), reverse=True)[:10]
        print(f"  {res:>11.4f}  {n_comm:>14}  {top10}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.run_dir, resolutions=args.resolutions, seed=args.seed)


if __name__ == "__main__":
    main()
