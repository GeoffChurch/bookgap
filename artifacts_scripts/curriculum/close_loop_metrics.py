#!/usr/bin/env python3
"""Close-the-loop ablation metrics: does gap-targeting + good-edge ordering make
OUR curriculum reconstruct a real book's ToC?

Three arms share a held-out book's ToC (Track B, the ground-truth partition):
  A0  curriculum at a count-matched RANDOM region (density floor; >=1 run, averaged)
  A1  curriculum at the book's DETECTED own-gap
  A2  A1's chapters RE-ORDERED by good edges (topical adjacency) — same works as A1

Co-primaries (no LLM judge), each on the comparison it discriminates:
  * reconstruction = size-weighted best-match ruzicka WORK-overlap coverage of the
    book's chapters by our chapters (placement). A1 vs A0: did we land on the book's
    material? Order-invariant, so identical for A1/A2.
  * prereq-violation = forward citation-flow share (toc_metrics.forward_violations)
    over A1's clusters, scored under A1's frontier order vs A2's good-edge order
    (ordering). Same F, two orders, plus a random-order reference.
Support: Kendall-tau of each order vs the book's true chapter order (shared matching).

Cloud is loaded ONCE at the gap's scope; citation_graph.npz rows map to the scoped
cloud's work_ids (compare_clusterings convention). CPU-only.

Usage:
  .conda/bin/python -m artifacts_scripts.curriculum.close_loop_metrics \
      --gap-json artifacts/book_gaps/mcs_gap.json \
      --toc-json artifacts/book_gaps/W4235013513_toc.json \
      --run-a1 artifacts/curriculum/cl_A1_mcs --run-b artifacts/curriculum/cl_B_mcs \
      --runs-a0 artifacts/curriculum/cl_A0_mcs_0 artifacts/curriculum/cl_A0_mcs_1 \
      --out artifacts/book_gaps/cl_mcs_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import kendalltau

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.gap_detection.toc_metrics import citation_flow, forward_violations
from bookgap.util import LOGGER, setup_logging

N_RAND_ORDER = 2000
RAND_SEED = 0


def _chapters(run_dir: Path) -> list[list[str]]:
    twi = json.loads((run_dir / "topic_work_ids.json").read_text())
    return [twi[k] for k in sorted(twi, key=lambda x: int(x))]


def _ruz_cover_book(A: list[list[str]], B: list[list[str]]) -> float:
    """Size-weighted best-match ruzicka WORK-overlap coverage of book chapters B by
    our chapters A (= share of the book's works our curriculum recovers; 1 - this is
    the work-level comprehensiveness gap). Pure set ops, order-invariant."""
    As = [set(c) for c in A]
    Bs = [set(c) for c in B]
    w = np.array([len(s) for s in Bs], dtype=np.float64)
    w = w / w.sum() if w.sum() else w
    best = np.zeros(len(Bs))
    for j, bj in enumerate(Bs):
        for ai in As:
            u = len(ai | bj)
            if u:
                best[j] = max(best[j], len(ai & bj) / u)
    return float((best * w).sum())


def _good_edge_order(centroids: np.ndarray) -> list[int]:
    """Greedy good-edge chapter order over L2-normed cluster centroids: start at the
    BROAD node (max mean cosine to all centroids = the natural intro), then greedily
    append the nearest unused centroid. Maximizes consecutive-chapter topical
    adjacency (the generative-ordering recipe; ordering_genre_firmup / toc_dendrogram)."""
    n = len(centroids)
    if n <= 2:
        return list(range(n))
    S = centroids @ centroids.T
    start = int(np.argmax((S.sum(1) - 1.0) / (n - 1)))   # broadest node
    order = [start]
    used = {start}
    while len(order) < n:
        last = order[-1]
        cand = [(S[last, j], j) for j in range(n) if j not in used]
        order.append(max(cand)[1])
        used.add(max(cand)[1])
    return order


def _build_F(run_dir: Path, id2idx: dict, n_clusters: int) -> np.ndarray:
    """Cluster x cluster citation-flow F[a,b] = #edges (a cites b) among the run's
    clustered works, from citation_graph.npz (G[i,j]=i cites j, build_curriculum:540)."""
    filtered = json.loads((run_dir / "filtered_work_ids.json").read_text())
    p2c = np.load(run_dir / "point_to_cluster.npy")
    G = sp.load_npz(run_dir / "citation_graph.npz").tocsr()
    # citation_graph is the build's EXPANDED node set (base cloud works occupy rows
    # 0..n_cloud-1 in cloud.work_ids() order, linked external nodes appended after;
    # the compare_clusterings convention). Filtered works are all base-cloud works,
    # so id2idx (from the scoped cloud) gives their rows, all < n_cloud <= G.shape[0].
    rows = np.array([id2idx[w] for w in filtered], dtype=np.int64)   # KeyError = scope mismatch
    assert int(rows.max()) < G.shape[0], "filtered row exceeds citation_graph (ordering mismatch)"
    sub = G[rows][:, rows].tocoo()
    r, c = sub.row, sub.col
    cr, cc = p2c[r], p2c[c]
    keep = (cr >= 0) & (cc >= 0) & (cr < n_clusters) & (cc < n_clusters)
    return citation_flow(cr[keep], cc[keep], n_clusters)


def _violation_share(F: np.ndarray, order: list[int]) -> float:
    return forward_violations(F, order)["share"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-json", required=True, type=Path)
    ap.add_argument("--toc-json", required=True, type=Path)
    ap.add_argument("--run-a1", required=True, type=Path)
    ap.add_argument("--run-b", required=True, type=Path)
    ap.add_argument("--runs-a0", nargs="*", type=Path, default=[])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="close_loop_metrics", log_level="INFO", include_time=False,
                  log_path=args.out.with_suffix(".log"))

    gap = json.loads(args.gap_json.read_text())
    subfield = gap.get("scope_subfield")
    field = gap.get("scope_field")
    book_id = gap.get("book", gap).get("book_id")

    cfg = OmegaConf.structured(CurriculumConfig())
    if subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
    if field:
        cfg.experiment.cloud.filter_pair.pre.field = str(field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids()
    id2idx = {w: i for i, w in enumerate(wid)}
    E = np.asarray(cloud.embeddings(), dtype=np.float32)
    LOGGER.info("cloud: %s (%d works); book=%s scope subfield=%s field=%s",
                cloud.name(), len(wid), book_id, subfield, field)

    B = _chapters(args.run_b)
    A1 = _chapters(args.run_a1)
    K = len(A1)
    LOGGER.info("Track-B chapters=%d; A1 clusters=%d", len(B), K)

    # ---- reconstruction (co-primary 1): work-overlap coverage of the book by each arm
    cov_a1 = _ruz_cover_book(A1, B)
    cov_a0 = []
    for rd in args.runs_a0:
        A0 = _chapters(rd)
        cov_a0.append(_ruz_cover_book(A0, B))
        LOGGER.info("  A0 %-28s reconstruction coverage = %.3f", rd.name, cov_a0[-1])
    LOGGER.info("RECON coverage  A1=%.3f  A0 mean=%.3f (n=%d)  -> A1-A0 = %+.3f",
                cov_a1, float(np.mean(cov_a0)) if cov_a0 else float("nan"), len(cov_a0),
                cov_a1 - (float(np.mean(cov_a0)) if cov_a0 else float("nan")))

    # ---- A2 good-edge reorder of A1's clusters (same works as A1)
    def _centroid(c: list[str]) -> np.ndarray:
        rows = [id2idx[w] for w in c if w in id2idx]
        v = E[rows].mean(0) if rows else np.zeros(E.shape[1], dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)
    cents = np.stack([_centroid(c) for c in A1])
    order_frontier = list(range(K))               # A1: twi key order = frontier order
    order_goodedge = _good_edge_order(cents)      # A2
    LOGGER.info("A1 frontier order=%s", order_frontier)
    LOGGER.info("A2 good-edge order=%s", order_goodedge)

    # ---- prereq-violation (co-primary 2): forward citation-flow share, A1 F, 3 orders
    F = _build_F(args.run_a1, id2idx, K)
    rng = np.random.default_rng(RAND_SEED)
    rand_shares = [_violation_share(F, list(rng.permutation(K))) for _ in range(N_RAND_ORDER)]
    pv = {"A1_frontier": _violation_share(F, order_frontier),
          "A2_goodedge": _violation_share(F, order_goodedge),
          "random_mean": float(np.mean(rand_shares)),
          "random_p05": float(np.percentile(rand_shares, 5)),
          "cross_edges": float(F.sum() - np.trace(F))}
    LOGGER.info("PREREQ-violation share  A1=%.3f  A2=%.3f  random=%.3f  (cross-edges=%.0f)",
                pv["A1_frontier"], pv["A2_goodedge"], pv["random_mean"], pv["cross_edges"])

    # ---- support: Kendall-tau to the book's true chapter order (shared A1 matching)
    # Match each book chapter j -> the A1 cluster with the largest work overlap, then
    # tau between book order (j) and that cluster's position under each arm's order.
    As = [set(c) for c in A1]
    fp = {c: i for i, c in enumerate(order_frontier)}
    gp = {c: i for i, c in enumerate(order_goodedge)}
    pairs = []
    for j, bj in enumerate(set(c) for c in B):
        ov = [len(ai & bj) for ai in As]
        if max(ov) > 0:
            pairs.append((j, int(np.argmax(ov))))
    tau = {"n_pairs": len(pairs), "A1_frontier": float("nan"), "A2_goodedge": float("nan")}
    if len(pairs) >= 3:
        bk_order = [p[0] for p in pairs]
        tau["A1_frontier"] = float(kendalltau(bk_order, [fp[p[1]] for p in pairs])[0])
        tau["A2_goodedge"] = float(kendalltau(bk_order, [gp[p[1]] for p in pairs])[0])
    LOGGER.info("ORDER tau-to-human  A1=%.3f  A2=%.3f  (n_pairs=%d)",
                tau["A1_frontier"], tau["A2_goodedge"], tau["n_pairs"])

    result = {
        "book_id": book_id, "scope_subfield": subfield, "scope_field": field,
        "n_book_chapters": len(B), "n_a1_clusters": K,
        "reconstruction_coverage": {"A1": cov_a1, "A0_runs": cov_a0,
                                    "A0_mean": float(np.mean(cov_a0)) if cov_a0 else None,
                                    "A1_minus_A0": cov_a1 - float(np.mean(cov_a0)) if cov_a0 else None},
        "prereq_violation_share": pv,
        "order_frontier": order_frontier, "order_goodedge": order_goodedge,
        "kendall_tau_to_human": tau,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    LOGGER.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
