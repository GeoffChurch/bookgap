"""Task C: bipartite-Gramian (random-walk) geometry over doc x keyword.

Concept gaps are geometrically invisible in LSA-128 (AUC 0.47): removing
keyword-defined fg docs leaves no local density dent because LSA distances are
dominated by common-term structure. Hypothesis (the stochastic-Gramian idea):
in the geometry of the two-step random walk doc -> keyword -> doc, docs sharing
RARE keywords are strongly connected (each rare keyword splits its unit of walk
mass among few docs), so a keyword-defined removal DOES leave a radial dent.

Operator (never materialized): with B the binary doc x keyword matrix,
D_d = row sums (keywords per doc), D_k = column sums (document frequency),

    P = (D_d^-1 B) (D_k^-1 B^T)        # two-step doc->doc stochastic Gramian

Embedding via truncated SVD of the walk-normalized bipartite matrix
S = D_d^-1/2 B D_k^-1/2:  P = D_d^-1/2 (S S^T) D_d^1/2, so P's eigenvalues are
sigma_i^2 with eigenvectors D_d^-1/2 u_i (u_i = left singular vectors of S).
Diffusion-map coordinates after t walk steps:

    Psi_j(doc i) = sigma_j^(2t) * u_ij / sqrt(d_i),    j = 2 .. dim+1

(the trivial top pair u_1 ~ sqrt(d), psi_1 = const is dropped; Euclidean distance
in Psi = diffusion distance between docs' t-step walk distributions).

Scaling choices (documented + sensitivity rows in the report):
  * t = 1 (one full doc->doc step) is the main configuration; t = 0.5 (single
    bipartite hop) is the flatter-spectrum sensitivity row.
  * degree_scale=True applies the D_d^-1/2 eigenvector scaling (true diffusion
    coordinates); False keeps plain sigma^(2t)-weighted singular vectors
    (spectral coordinates) as a robustness row, since 1/sqrt(d) inflates
    low-degree docs.
Isolated docs (no top-2000 keywords) embed at the origin and are counted in meta.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

WALK_DIM = 128


def walk_embedding(
    K: sp.spmatrix,
    dim: int = WALK_DIM,
    t: float = 1.0,
    degree_scale: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Diffusion-map embedding of the doc x keyword walk; returns (W, meta).

    W is (n_docs, dim) float32. Deterministic: ARPACK started from a seeded v0.
    """
    K = sp.csr_matrix(K, dtype=np.float64)
    d_doc = np.asarray(K.sum(axis=1)).ravel()
    d_kw = np.asarray(K.sum(axis=0)).ravel()
    iso = d_doc == 0
    dd = np.where(iso, 1.0, d_doc)
    dk = np.where(d_kw == 0, 1.0, d_kw)
    S = sp.diags(1.0 / np.sqrt(dd)) @ K @ sp.diags(1.0 / np.sqrt(dk))
    k = min(dim + 1, min(S.shape) - 1)
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=min(S.shape))
    u, s, _ = svds(S, k=k, v0=v0)
    order = np.argsort(s)[::-1]
    u, s = u[:, order], s[order]
    u, s = u[:, 1:], s[1:]  # drop the trivial top pair (sigma ~ 1, psi ~ const)
    W = u * (s ** (2.0 * t))[None, :]
    if degree_scale:
        W = W / np.sqrt(dd)[:, None]
    W[iso] = 0.0
    meta = {
        "dim": int(W.shape[1]),
        "t": float(t),
        "degree_scale": bool(degree_scale),
        "n_isolated_docs": int(iso.sum()),
        "top_singular_values": [round(float(x), 4) for x in s[:5]],
        "sv_last": round(float(s[-1]), 4),
    }
    return W.astype(np.float32), meta
