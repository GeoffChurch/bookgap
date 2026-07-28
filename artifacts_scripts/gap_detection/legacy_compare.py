"""Bounded comparison against the repo's legacy k-ball gap detector.

Maps our plant-and-recover setting onto the legacy roles: landmark = foreground
(supply/occupancy, the role books played), ambient = query = background (demand/
habitability, the role articles played). The greedy k-ball detector then yields
gaps = balls of bg points far from fg, exactly its production semantics.

The legacy joblib cache is bypassed in-memory (MemorizedFunc.func) so repeated
trials don't churn cache/joblib with throwaway corpora pickles. Nothing tracked
is modified.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # legacy chain imports torch; keep it off-GPU

import time
from dataclasses import dataclass, field

import numpy as np

from artifacts_scripts.gap_detection.detectors import iou


@dataclass
class ArrayCorpus:
    """Minimal in-memory implementation of the bookgap Corpus protocol."""

    emb: np.ndarray
    label: str

    _index: object = field(default=None, repr=False, compare=False)

    def embeddings(self) -> np.ndarray:
        return np.ascontiguousarray(self.emb, dtype=np.float32)

    def index(self):
        if self._index is None:
            import faiss

            ix = faiss.IndexFlatL2(self.emb.shape[1])
            ix.add(self.embeddings())
            self._index = ix
        return self._index

    def work_ids(self) -> list[str]:
        return [f"{self.label}:{i}" for i in range(self.emb.shape[0])]

    def work_titles(self) -> list[str]:
        return self.work_ids()

    def work_abstracts(self) -> list[str]:
        return [""] * self.emb.shape[0]

    def name(self) -> str:
        return self.label

    @property
    def embedder_fingerprint(self) -> str:
        return "gapdeteval-lsa"

    @property
    def identity(self):
        return (self.label, int(self.emb.shape[0]), int(self.emb.shape[1]))

    # joblib/pickle must never see the swig faiss index
    def __getstate__(self):
        st = self.__dict__.copy()
        st["_index"] = None
        return st


def _bypass_joblib_cache() -> None:
    """Point kball_algorithm at the raw (uncached) coverage function, in memory only."""
    import bookgap.cloud_index as ci
    import bookgap.llm.gap_algorithms.kball_algorithm as ka

    raw = getattr(ci.get_kball_coverage, "func", None)
    if raw is not None:
        ka.get_kball_coverage = raw


def run_legacy_trial(
    fg_pruned: np.ndarray, bg: np.ndarray, region_bg: np.ndarray,
    radius_fraction: float = 0.75, k: int = 1, num_gaps: int = 5,
    budget_s: float = 180.0,
) -> dict:
    """Run the legacy greedy k-ball detector on one planted trial.

    Returns hit/precision/IoU of the best of the first `num_gaps` gaps against the
    planted region (gap members are ambient = bg-local indices).
    """
    from bookgap.llm.gap_algorithms.kball_algorithm import KballAlgorithm

    _bypass_joblib_cache()
    fgC = ArrayCorpus(fg_pruned, "fg")
    bgC = ArrayCorpus(bg, "bg")
    region = set(map(int, region_bg))
    t0 = time.monotonic()
    gaps = []
    gen = KballAlgorithm.kball_gaps(
        ambient=bgC, landmark=fgC, query=bgC,
        radius_fraction=radius_fraction, k=k, is_intra_cloud=False,
    )
    for gap in gen:
        members = np.asarray(gap.dists_idxs.idxs, dtype=np.int64)
        gaps.append(members)
        if len(gaps) >= num_gaps or (time.monotonic() - t0) > budget_s:
            break
    best_prec, best_iou, first_hit_rank = 0.0, 0.0, None
    for rank, members in enumerate(gaps, start=1):
        prec = float(np.mean([int(m) in region for m in members])) if members.size else 0.0
        ji = iou(members, region_bg)
        if prec > best_prec:
            best_prec = prec
        if ji > best_iou:
            best_iou = ji
        if first_hit_rank is None and prec >= 0.5:
            first_hit_rank = rank
    return {
        "n_gaps_seen": len(gaps),
        "gap_sizes": [int(m.size) for m in gaps],
        "best_precision": round(best_prec, 4),
        "best_iou": round(best_iou, 4),
        "first_hit_rank": first_hit_rank,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
