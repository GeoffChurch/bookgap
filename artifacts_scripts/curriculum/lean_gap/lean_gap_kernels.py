"""Embedding-space kNN kernels for the Lean gap plant-and-recover.

The expensive object is the per-candidate sorted list of nearest FOREGROUND
(theorem) neighbours; once cached, re-scoring under ANY theorem-deletion is a masked
read of the k-th surviving neighbour (milliseconds), so ≥30 plants + controls + nulls
are cheap. Background (def) neighbours never change, so r_B,k is computed once.

All CPU, float32, normalized embeddings (cosine == euclidean monotone).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a gemma build; CPU only.

import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-9


def knn_dists(query: np.ndarray, ref: np.ndarray, m: int, *, exclude_self: bool = False,
              return_idx: bool = False):
    """Distances (and optionally indices) to the m nearest ``ref`` rows for each query."""
    mm = m + 1 if exclude_self else m
    mm = min(mm, ref.shape[0])
    nn = NearestNeighbors(n_neighbors=mm, algorithm="brute", metric="euclidean").fit(ref)
    d, idx = nn.kneighbors(query)
    if exclude_self:
        d, idx = d[:, 1:], idx[:, 1:]
    return (d, idx) if return_idx else d


class FgNeighborTable:
    """Per-candidate sorted nearest-foreground table supporting cheap deletion re-scores.

    For each candidate we cache the indices + distances of its ``m`` nearest foreground
    (theorem) points in the FULL (intact) foreground. ``r_F_k(deleted)`` masks the
    deleted foreground rows and returns the k-th surviving distance; if a candidate's
    cached window is exhausted by the deletion (rare — only near a large local
    deletion) the row falls back to an exact recompute against surviving F.
    """

    def __init__(self, cand: np.ndarray, fg: np.ndarray, m: int = 256, k: int = 10):
        self.cand = cand
        self.fg = fg
        self.m = min(m, fg.shape[0])
        self.k = k
        d, idx = knn_dists(cand, fg, self.m, return_idx=True)
        self.dist = d.astype(np.float32)        # (n_cand, m) ascending
        self.idx = idx.astype(np.int64)         # (n_cand, m) fg indices
        self.r_F_intact = self.dist[:, k - 1].copy()

    def r_F_k(self, deleted_mask: np.ndarray | None) -> np.ndarray:
        """k-th nearest SURVIVING foreground distance per candidate (deleted_mask over fg)."""
        if deleted_mask is None or not deleted_mask.any():
            return self.r_F_intact.copy()
        n = self.cand.shape[0]
        out = np.empty(n, dtype=np.float32)
        survive = ~deleted_mask[self.idx]                       # (n_cand, m)
        csum = np.cumsum(survive, axis=1)
        # first column where the k-th survivor appears
        hit = csum >= self.k
        has = hit.any(axis=1)
        col = np.argmax(hit, axis=1)                            # first True per row
        rows_ok = np.where(has)[0]
        out[rows_ok] = self.dist[rows_ok, col[rows_ok]]
        # fallback: window exhausted -> exact recompute against surviving fg
        bad = np.where(~has)[0]
        if bad.size:
            surv_fg = self.fg[~deleted_mask]
            if surv_fg.shape[0] >= self.k:
                dd = knn_dists(self.cand[bad], surv_fg, self.k)
                out[bad] = dd[:, self.k - 1]
            else:
                out[bad] = self.dist[bad, -1] * 4.0  # degenerate; push far
        return out


def mann_whitney_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(planted region) > score(control)); tie-aware. NaN if either empty."""
    from scipy.stats import mannwhitneyu

    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (pos.size * neg.size))
