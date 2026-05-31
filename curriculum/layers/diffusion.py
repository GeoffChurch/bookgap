"""Graph Diffusion Convolution (GDC) densification — optional graph rewiring.

Lifted from ``artifacts_scripts/wiki/joint_graph_curriculum.py`` (``gdc_diffuse``
+ the sparse helpers). A smooth, weighted, multi-hop heat/PPR re-wiring of a
sparse adjacency, re-sparsified to top-``top_m``/row and renormalized. Used to
densify sparse backbones (e.g. the OA citation block in the joint graph) before
ranking/clustering. Pure numpy/scipy.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from curriculum._logging import get_logger

LOGGER = get_logger(__name__)


def row_topm(M: sp.csr_matrix, m: int) -> sp.csr_matrix:
    """Keep only the top-``m`` entries (by magnitude) of each row of ``M``."""
    M = M.tocsr()
    if M.nnz == 0:
        return M
    ptr, data, ind = M.indptr, M.data, M.indices
    keep = np.ones(M.nnz, dtype=bool)
    for i in range(M.shape[0]):
        s, e = int(ptr[i]), int(ptr[i + 1])
        if e - s > m:
            drop = np.argpartition(data[s:e], e - s - m - 1)[: e - s - m]
            keep[s + drop] = False
    rows = np.repeat(np.arange(M.shape[0]), np.diff(ptr))
    return sp.csr_matrix((data[keep], (rows[keep], ind[keep])), shape=M.shape)


def row_l1_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    rs = np.asarray(M.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return (sp.diags(1.0 / rs) @ M).tocsr()


def gdc_diffuse(
    A_sym: sp.csr_matrix,
    kernel: str,
    t: float,
    ppr_alpha: float,
    top_m: int,
    series_terms: int,
    cap_m: int,
) -> sp.csr_matrix:
    """GDC on a SYMMETRIC adjacency ``A_sym``.

    Returns the re-sparsified (top-``top_m``/row) + row-L1-renormalized
    diffusion affinity ``S``. ``heat`` uses Poisson weights
    ``e^{-t} t^k/k!``; ``ppr`` uses the Neumann series ``α(1-α)^k`` for the
    Personalized-PageRank kernel. Each intermediate power is capped to
    top-``cap_m``/row so the sparse series never densifies.
    """
    n = A_sym.shape[0]
    deg = np.asarray(A_sym.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    T = (sp.diags(1.0 / deg) @ A_sym).tocsr()

    if kernel == "heat":
        coef = [math.exp(-t) * t ** k / math.factorial(k) for k in range(series_terms + 1)]
    elif kernel == "ppr":
        coef = [ppr_alpha * (1.0 - ppr_alpha) ** k for k in range(series_terms + 1)]
    else:
        raise ValueError(f"unknown diffusion kernel {kernel!r}")

    Tk = sp.identity(n, format="csr", dtype=np.float64)
    S = coef[0] * Tk
    for k in range(1, series_terms + 1):
        Tk = (Tk @ T).tocsr()
        Tk = row_topm(Tk, cap_m)
        S = (S + coef[k] * Tk).tocsr()
    S = row_topm(S, top_m)
    return row_l1_normalize(S)


def diffuse_adjacency(
    adj_dir: sp.csr_matrix,
    *,
    kernel: str = "heat",
    t: float = 8.0,
    ppr_alpha: float = 0.15,
    top_m: int = 32,
    series_terms: int = 12,
    cap_m: int = 64,
) -> sp.csr_matrix:
    """Densify a directed adjacency: symmetrize -> GDC -> overlay the directed
    skeleton (unit weight) so forward PageRank keeps citation direction.

    Returns a directed CSR block ready to drop back into the engine.
    """
    n = adj_dir.shape[0]
    A_sym = (adj_dir + adj_dir.T).astype(np.float64)
    A_sym.data[:] = 1.0
    A_sym.setdiag(0.0)
    A_sym.eliminate_zeros()
    S = gdc_diffuse(A_sym, kernel, t, ppr_alpha, top_m, series_terms, cap_m)
    S.setdiag(0.0)
    S.eliminate_zeros()
    skel = adj_dir.copy().astype(np.float64)
    skel.data[:] = 1.0
    block = (S + skel).tocsr()
    block.sum_duplicates()
    LOGGER.info(
        "GDC diffuse (%s, t=%.3g): %d diffused edges; block %d edges",
        kernel, t, int(S.nnz), int(block.nnz - n),
    )
    return block
