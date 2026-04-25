"""HDBSCAN gap-detection probe.

Loads a single scope's works + embeddings straight from the per-scope
caches (bypassing the full Cloud machinery), runs fast_hdbscan, and
reports the signals that matter for the gap-detection idea: cluster
size distribution, per-cluster book-fraction histogram, and sample
titles from the top-N populous book-free clusters.

Usage::

    conda run -p .conda python -m artifacts_scripts.hdbscan_gap_probe \\
        --scope 17/1702/T10181 \\
        --min-cluster-size 50 --min-samples 15

Scope key is either "field/subfield" (aggregates all topics × work
types under it) or "field/subfield/topic" (single topic, all work
types). Embeddings are assumed to exist under
``cache/embeddings/<model>/<scope>/<work_type>/batch_*_embeddings.npy``;
if not, run the fetch/embed pipeline for that scope first.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq

from bookgap.embedder import SentenceTransformerEmbedder as Embedder
from bookgap.util import CACHE_DIR


def _load_scope_shard(
    parquet_path: Path, emb_dir: Path, with_refs: bool = False,
) -> Tuple[np.ndarray, List[str], List[str], Optional[List[Optional[List[str]]]]]:
    """Return (embeddings, ids, titles, refs_or_None) for one shard.

    Aligns embeddings to parquet row order via the per-batch id manifests.
    The primary_topic_id column is present in the parquet schema but was
    not populated by snapshot_ingest; the caller derives topic identity
    from the parquet path instead (``<field>/<subfield>/<topic>/<type>``).
    """
    columns = ["id", "display_name"]
    if with_refs:
        columns.append("referenced_works")
    t = pq.read_table(parquet_path, columns=columns)
    ids = t.column("id").to_pylist()
    titles = t.column("display_name").to_pylist()
    refs = t.column("referenced_works").to_pylist() if with_refs else None

    batch_embs: List[np.ndarray] = []
    batch_ids: List[str] = []
    for i in range(100_000):
        eb = emb_dir / f"batch_{i}_embeddings.npy"
        ib = emb_dir / f"batch_{i}_ids.json"
        if not eb.exists():
            break
        batch_embs.append(np.load(eb))
        batch_ids.extend(json.loads(ib.read_text()))
    if not batch_embs:
        raise FileNotFoundError(f"no embedding batches under {emb_dir}")

    emb = np.concatenate(batch_embs, axis=0).astype(np.float32)
    if batch_ids != ids:
        id_to_row = {wid: i for i, wid in enumerate(batch_ids)}
        reorder = np.array([id_to_row[i] for i in ids], dtype=np.int64)
        emb = emb[reorder]
    return emb, ids, titles, refs


def load_scope(scope_key: str, with_refs: bool = False):
    parts = scope_key.split("/")
    model_name = Embedder.ModelName.GIST_SMALL.value
    parquet_root = CACHE_DIR / "works_parquet"
    emb_root = CACHE_DIR / "embeddings" / model_name
    base = parquet_root.joinpath(*parts)
    emb_base = emb_root.joinpath(*parts)

    if len(parts) == 2:
        topic_dirs = sorted(d for d in base.iterdir() if d.is_dir())
        emb_dirs = [emb_base / d.name for d in topic_dirs]
    elif len(parts) == 3:
        topic_dirs = [base]
        emb_dirs = [emb_base]
    else:
        raise ValueError(f"scope_key must be F/SF or F/SF/T, got {scope_key!r}")

    all_emb: List[np.ndarray] = []
    all_ids: List[str] = []
    all_types: List[str] = []
    all_titles: List[str] = []
    all_topic_ids: List[str] = []
    all_refs: List[Optional[List[str]]] = []
    for topic_dir, emb_topic_dir in zip(topic_dirs, emb_dirs):
        topic_label = topic_dir.name  # e.g. "T10181"; path is authoritative
        for wt_parquet in sorted(topic_dir.glob("*.parquet")):
            wt = wt_parquet.stem
            if pq.ParquetFile(wt_parquet).metadata.num_rows == 0:
                continue
            emb_dir = emb_topic_dir / wt
            if not emb_dir.exists() or not any(emb_dir.glob("batch_*_embeddings.npy")):
                print(f"  WARN: no embeddings under {emb_dir}; skipping shard")
                continue
            emb, ids, titles, refs = _load_scope_shard(wt_parquet, emb_dir, with_refs=with_refs)
            all_emb.append(emb)
            all_ids.extend(ids)
            all_types.extend([wt] * len(ids))
            all_titles.extend(titles)
            all_topic_ids.extend([topic_label] * len(ids))
            if refs is not None:
                all_refs.extend(refs)

    embeddings = np.concatenate(all_emb, axis=0)
    return embeddings, all_ids, all_types, all_titles, all_topic_ids, (all_refs if with_refs else None)


# ---------------------------------------------------------------------------
# Citation graph + lightweight DenoiserContext (for graph-based denoisers)
# ---------------------------------------------------------------------------

def _build_in_scope_citation_csr(
    ids: List[str], refs: List[Optional[List[str]]],
) -> "scipy.sparse.csr_matrix":
    """Build an N×N sparse adjacency where A[i,j]=1 iff work i cites work j
    AND j is in scope. Out-of-scope refs are dropped."""
    import scipy.sparse as sp
    n = len(ids)
    id_to_idx = {wid: i for i, wid in enumerate(ids)}
    rows: List[int] = []
    cols: List[int] = []
    for i, ref_list in enumerate(refs):
        if not ref_list:
            continue
        for r in ref_list:
            j = id_to_idx.get(r)
            if j is not None and j != i:  # drop self-loops
                rows.append(i)
                cols.append(j)
    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix(
        (data, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(n, n),
    )
    A.sum_duplicates()
    return A


class _ProbeDenoiserContext:
    """Minimal DenoiserContext implementation for the probe.

    Implements the abstract interface (X, X_n, num_subsets, A) plus the
    derived helpers the existing project denoisers need (X_mean,
    X_centered, A_sym, D, L, S_T, S_L, S_B_multi). Avoids a hard
    dependency on bookgap.denoisers.DenoiserContext at import time, but
    duck-types its public surface.
    """

    def __init__(self, embeddings: np.ndarray, class_labels: List[str],
                 A_sparse) -> None:
        self._X = embeddings.astype(np.float64, copy=False)
        labels_arr = np.asarray(class_labels)
        unique, inverse = np.unique(labels_arr, return_inverse=True)
        self._unique_labels = unique
        self._inverse = inverse
        self._A = A_sparse
        self._mean = self._X.mean(axis=0, keepdims=True)
        self._X_centered: Optional[np.ndarray] = None
        # Cached scatters
        self._S_T: Optional[np.ndarray] = None
        self._S_L: Optional[np.ndarray] = None

    def X(self) -> np.ndarray:
        return self._X

    def X_n(self, n: int) -> np.ndarray:
        return self._X[self._inverse == n]

    def num_subsets(self) -> int:
        return len(self._unique_labels)

    def A(self):
        return self._A

    def X_mean(self) -> np.ndarray:
        return self._mean

    def X_centered(self) -> np.ndarray:
        if self._X_centered is None:
            self._X_centered = self._X - self._mean
        return self._X_centered

    def A_sym(self):
        import scipy.sparse as sp
        A = self._A
        return A + A.T

    def D(self):
        import scipy.sparse as sp
        deg = np.asarray(self.A_sym().sum(axis=1)).ravel()
        return sp.diags(deg)

    def L(self):
        return self.D() - self.A_sym()

    def S_T(self) -> np.ndarray:
        if self._S_T is None:
            X_c = self.X_centered()
            self._S_T = X_c.T @ X_c
        return self._S_T

    def S_L(self) -> np.ndarray:
        if self._S_L is None:
            X_c = self.X_centered()
            self._S_L = X_c.T @ (self.L() @ X_c)
        return self._S_L

    def S_B_multi(self, subset_indices: List[int]) -> np.ndarray:
        d = self._X.shape[1]
        if len(subset_indices) < 2:
            return np.zeros((d, d))
        overall = self._mean.squeeze()
        S_B = np.zeros((d, d), dtype=np.float64)
        total = 0
        for i in subset_indices:
            sub = self.X_n(i)
            n_i = len(sub)
            if n_i == 0:
                continue
            mu = sub.mean(axis=0) - overall
            S_B += n_i * np.outer(mu, mu)
            total += n_i
        return S_B / max(total, 1)


# ---------------------------------------------------------------------------
# Graph-based denoiser drivers (work directly with the matrices we care about)
# ---------------------------------------------------------------------------

def _project_via_basis(emb: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Apply a (d × n_components) basis: X_centered @ basis. Returns float32."""
    centered = (emb.astype(np.float64) - emb.astype(np.float64).mean(axis=0))
    out = centered @ basis
    return out.astype(np.float32)


def _solve_keep_smallest_gep(A: np.ndarray, B: np.ndarray, n_keep: int) -> np.ndarray:
    """Solve A v = λ B v, return the n_keep smallest-eigenvalue eigvecs as columns.

    Uses an adaptive Tikhonov ridge on B: if the unregularised call fails
    (B too rank-deficient), retries with progressively larger ridges.
    """
    import scipy.linalg
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    trace_B = float(np.trace(B))
    base_ridge = max(trace_B, 1.0) / B.shape[0]
    for ridge_scale in (1e-6, 1e-3, 1e-1, 1.0):
        try:
            B_reg = B + ridge_scale * base_ridge * np.eye(B.shape[0])
            eigvals, eigvecs = scipy.linalg.eigh(A, B_reg)
            return eigvecs[:, :n_keep]
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        f"GEP failed at all ridge scales up to {1.0 * base_ridge:.3e}"
    )


def _denoise_lpp(ctx: _ProbeDenoiserContext, n_keep: int) -> np.ndarray:
    """Locality-Preserving Projections — keep n_keep smallest eigvecs of
    Xc^T L Xc v = λ Xc^T D Xc v. Citation-graph-driven; needs no class labels."""
    X_c = ctx.X_centered()
    A_num = X_c.T @ (ctx.L() @ X_c)
    A_den = X_c.T @ (ctx.D() @ X_c)
    return _solve_keep_smallest_gep(A_num, A_den, n_keep)


def _denoise_lip(ctx: _ProbeDenoiserContext, n_keep: int) -> np.ndarray:
    """Laplacian-Inducing Projection — keep n_keep smallest eigvecs of
    S_L v = λ S_T v. Differs from LPP by the denominator (total scatter vs
    degree-weighted scatter). Mass-uniform rather than degree-weighted."""
    return _solve_keep_smallest_gep(ctx.S_L(), ctx.S_T(), n_keep)


def _denoise_lip_eigen_only(ctx: _ProbeDenoiserContext, n_keep: int) -> np.ndarray:
    """Minimise v^T S_L v without a GEP denominator: plain eigendecomp of
    S_L, keep smallest eigvecs. Unlike LIP/LPP this has no variance-
    preservation constraint — low-variance directions with tiny Laplacian
    scatter can dominate, which may or may not be desirable."""
    import scipy.linalg
    S_L = ctx.S_L()
    S_L = 0.5 * (S_L + S_L.T)
    _, eigvecs = scipy.linalg.eigh(S_L)
    return eigvecs[:, :n_keep]


def _denoise_lip_diff(ctx: _ProbeDenoiserContext, n_keep: int) -> np.ndarray:
    """S_L_norm v = λ (S_T − S_L_norm) v, using the *normalized* Laplacian
    (eigenvalues in [0, 2]) to keep the denominator positive-definite.

    The unnormalized-Laplacian version doesn't make sense: L can have
    eigenvalues up to 2·max_degree (often >> 2), so S_L can exceed S_T
    along most directions and S_T − S_L is negative definite. With the
    symmetric normalized Laplacian L_sym = I − D^(−1/2) A D^(−1/2),
    S_L_norm = Xc^T L_sym Xc is bounded, and
        S_T − S_L_norm = Xc^T (I − L_sym) Xc = Xc^T (D^(−1/2) A D^(−1/2)) Xc
    is positive semi-definite — it's a *similarity-weighted scatter*
    rewarding directions where citation-linked works covary.

    Minimising S_L_norm / (S_T − S_L_norm) is then: "find directions where
    graph-incoherence is low relative to graph-coherence." Equivalent to
    LPP with a specifically-chosen denominator, but stated in a way that
    makes the signal/noise interpretation direct.
    """
    import scipy.sparse as sp
    X_c = ctx.X_centered()
    B = ctx.A_sym()
    degrees = np.asarray(B.sum(axis=1)).ravel()
    safe = degrees > 0
    inv_sqrt = np.zeros_like(degrees, dtype=float)
    inv_sqrt[safe] = np.power(degrees[safe], -0.5)
    D_inv_sqrt = sp.diags(inv_sqrt)
    # Symmetric normalised Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
    sim = D_inv_sqrt @ B @ D_inv_sqrt  # = I - L_sym
    S_L_norm = X_c.T @ ((sp.eye(B.shape[0]) - sim) @ X_c)
    S_coherent = X_c.T @ (sim @ X_c)  # = S_T - S_L_norm (PSD by construction)
    return _solve_keep_smallest_gep(S_L_norm, S_coherent, n_keep)


def _denoise_cclip(
    ctx: _ProbeDenoiserContext, n_keep: int, alpha: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """CCLIP: minimise ((1-α) S_B + α S_L) / S_T.

    S_B and S_L have incommensurate magnitudes (tr(S_L) can be 10^7× tr(S_B)
    at scale), so we pre-normalise both to trace=d before the convex
    combination. Then α has its natural interpretation:
      α=0 → pure class-conflation (S_B only)
      α=1 → pure locality-inducing (S_L only)
      α=0.5 → balanced contribution
    """
    S_B = ctx.S_B_multi(list(range(ctx.num_subsets())))
    S_L = ctx.S_L()
    S_T = ctx.S_T()
    d = S_T.shape[0]
    tr_B = float(np.trace(S_B))
    tr_L = float(np.trace(S_L))
    print(f"  raw: trace(S_B)={tr_B:.3e}, trace(S_L)={tr_L:.3e}")
    S_B_norm = S_B * (d / tr_B) if tr_B > 0 else S_B
    S_L_norm = S_L * (d / tr_L) if tr_L > 0 else S_L
    A = (1 - alpha) * S_B_norm + alpha * S_L_norm
    return _solve_keep_smallest_gep(A, S_T, n_keep), alpha


def _denoise_hierarchical(
    ctx: _ProbeDenoiserContext, drop_k_class: int, drop_k_graph: int,
) -> np.ndarray:
    """Two-stage projection: project out top class-discriminating directions,
    then within the residual project out top citation-incoherent directions.
    Returns a basis (d × n_keep) for the final residual subspace."""
    d = ctx.X().shape[1]
    S_B = ctx.S_B_multi(list(range(ctx.num_subsets())))
    S_T = ctx.S_T()
    # Step 1: keep the d - drop_k_class smallest eigvecs of S_B v = λ S_T v
    basis_1 = _solve_keep_smallest_gep(S_B, S_T, d - drop_k_class)
    # Step 2: in the residual subspace, recompute S_L and S_T using basis_1
    # transform: S_L_residual = basis_1.T @ S_L @ basis_1, similarly S_T
    S_L = ctx.S_L()
    S_L_res = basis_1.T @ S_L @ basis_1
    S_T_res = basis_1.T @ S_T @ basis_1
    keep_2 = max(1, basis_1.shape[1] - drop_k_graph)
    basis_2 = _solve_keep_smallest_gep(S_L_res, S_T_res, keep_2)
    # Compose: full d-dim → residual → final
    return basis_1 @ basis_2


def _denoise_criteria_union(
    ctx: _ProbeDenoiserContext, drop_k_class: int, drop_k_graph: int,
) -> np.ndarray:
    """Solve S_B v = λ S_T v and S_L v = λ S_T v separately; take the top-k
    eigvecs of each; project out the union subspace.

    Differs from hierarchical (sequential, residual-dependent) and from
    rnojd (joint diagonalisation, approximate per-matrix diagonals): each
    criterion gets its EXACT top-k directions. Overlap between the two
    eigenbases is handled by QR: if S_B's top-k and S_L's top-k span
    identical directions, the union is just k dims; if fully disjoint, 2k.
    Returns a (d × n_keep) orthonormal basis for the complement of the
    union subspace.
    """
    d = ctx.X().shape[1]
    S_T = ctx.S_T()
    ridge = 1e-6 * np.trace(S_T) / d
    S_T_reg = S_T + ridge * np.eye(d)

    import scipy.linalg
    from scipy.linalg import null_space

    cols: List[np.ndarray] = []
    if drop_k_class > 0:
        S_B = ctx.S_B_multi(list(range(ctx.num_subsets())))
        _, V_B = scipy.linalg.eigh(S_B, S_T_reg)
        cols.append(V_B[:, -drop_k_class:])
    if drop_k_graph > 0:
        S_L = ctx.S_L()
        _, V_L = scipy.linalg.eigh(S_L, S_T_reg)
        cols.append(V_L[:, -drop_k_graph:])
    if not cols:
        return np.eye(d, dtype=np.float64)
    combined = np.hstack(cols)
    Q, _ = np.linalg.qr(combined)
    # Rank of the union: if the two top-k subspaces overlap, Q has fewer
    # effective columns; null_space produces the correctly-sized complement.
    complement = null_space(Q.T)
    union_rank = d - complement.shape[1]
    overlap = (drop_k_class + drop_k_graph) - union_rank
    print(f"  criteria_union: projected out {union_rank}-dim union subspace "
          f"({drop_k_class} + {drop_k_graph} requested, {overlap} shared between bases)")
    return complement


def _denoise_rnojd(
    ctx: _ProbeDenoiserContext, n_drop: int,
) -> np.ndarray:
    """Joint diagonalisation of {S_B, S_L, S_T} via manopt_rnojd.
    Each axis gets per-matrix scores; project out the axes that maximise the
    combined penalty (S_B normalised by S_T + S_L normalised by S_T).
    Returns a (d × n_keep) basis where n_keep = d - n_drop.
    """
    from bookgap.rsdc import manopt_rnojd
    S_B = ctx.S_B_multi(list(range(ctx.num_subsets())))
    S_L = ctx.S_L()
    S_T = ctx.S_T()
    # Stack as (3, d, d). manopt_rnojd expects (k, d, d).
    C = np.stack([S_B, S_L, S_T], axis=0).astype(np.float64)
    # symmetrise each
    for i in range(C.shape[0]):
        C[i] = 0.5 * (C[i] + C[i].T)
    print(f"  rnojd: jointly diagonalising {C.shape[0]} matrices of size "
          f"{C.shape[1]}×{C.shape[2]}...")
    V_raw = manopt_rnojd(C, trials=3, max_iter=80, pd=False)
    # manopt_rnojd returns columns on the Oblique manifold (unit-norm,
    # not orthogonal). Orthonormalise via QR so downstream projection
    # doesn't distort distances.
    V, _ = np.linalg.qr(V_raw)
    diag_B = np.einsum("ij,jk,ki->i", V.T, S_B, V)
    diag_L = np.einsum("ij,jk,ki->i", V.T, S_L, V)
    diag_T = np.einsum("ij,jk,ki->i", V.T, S_T, V)
    # Normalise each matrix's diagonal to a probability distribution over
    # axes so magnitudes are comparable. Drop score = relative noise minus
    # relative signal; positive means axis is disproportionately noise.
    def _norm(x):
        x_pos = np.maximum(x, 0)
        s = x_pos.sum()
        return x_pos / s if s > 0 else np.zeros_like(x_pos)
    p_B, p_L, p_T = _norm(diag_B), _norm(diag_L), _norm(diag_T)
    score = p_B + p_L - 2 * p_T
    drop_idx = np.argsort(-score)[:n_drop]
    keep_idx = np.array([i for i in range(V.shape[1]) if i not in set(drop_idx.tolist())])
    print(f"  rnojd: dropped-axis score range: "
          f"[{score[drop_idx].min():.2e}, {score[drop_idx].max():.2e}]")
    print(f"  rnojd: kept-axis score median: {float(np.median(score[keep_idx])):.2e}")
    print(f"  rnojd: variance retained: {p_T[keep_idx].sum():.1%}")
    return V[:, keep_idx]


def _drop_top_k_pca(emb: np.ndarray, drop_k: int) -> np.ndarray:
    """Center and project out the top-k right-singular vectors.

    Blunt but cheap: just subtracts the directions of max variance. Works
    at any scope — no class labels needed. For an embedding cloud with one
    or two dominant "I belong to scope X" directions, removing them
    concentrates what's left onto within-scope variation.
    """
    if drop_k <= 0:
        return emb
    mean = emb.mean(axis=0)
    centered = emb - mean
    gram = centered.T @ centered
    gram = 0.5 * (gram + gram.T)
    import scipy.linalg
    _, eigvecs = scipy.linalg.eigh(gram)
    top_k = eigvecs[:, -drop_k:]
    projector = np.eye(emb.shape[1], dtype=emb.dtype) - top_k @ top_k.T
    return (centered @ projector).astype(emb.dtype)


def _drop_top_k_multi_lda(
    emb: np.ndarray, class_labels: List[str], drop_k: int,
) -> np.ndarray:
    """Project out the top-k directions of Fisher's multi-class LDA.

    Solves the GEP ``S_B v = λ S_W v`` with classes = ``class_labels`` and:
      * ``S_B = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ``   (between-class scatter)
      * ``S_W = Σ_{x} (x − μ_{c(x)})(x − μ_{c(x)})ᵀ``  (within-class scatter)

    Top-k eigenvectors are the directions most discriminating of class.
    Projecting them out surgically removes *between-class variance* while
    preserving within-class structure — exactly what we need when the
    "undesirable" signal is "I'm in class c" (scope/topic identity) and the
    "desirable" signal is within-class sub-structure.

    Degenerates to no-op when the scope has only one class (e.g., a single-
    topic scope passed with ``class_labels = primary_topic_id``). Callers
    that want a single-topic-viable denoiser should use PCA instead (or
    select classes along a different axis, e.g., publication year bucket).

    A small ridge is added to S_W for numerical stability.
    """
    if drop_k <= 0:
        return emb
    # Works with no OpenAlex topic assignment come through as None; treat
    # them as a single "__unknown__" class rather than letting np.unique
    # crash on mixed None/str.
    normalised = ["__unknown__" if x is None else x for x in class_labels]
    labels_arr = np.asarray(normalised)
    unique_labels, inverse = np.unique(labels_arr, return_inverse=True)
    n_classes = len(unique_labels)
    if n_classes < 2:
        print(f"  multi_lda: only {n_classes} class; skipping projection")
        return emb

    n, d = emb.shape
    # Per-class means and counts (accumulate in a single pass for speed)
    class_counts = np.bincount(inverse, minlength=n_classes).astype(np.float64)
    class_sums = np.zeros((n_classes, d), dtype=np.float64)
    np.add.at(class_sums, inverse, emb.astype(np.float64))
    class_means = class_sums / class_counts[:, None]
    global_mean = emb.astype(np.float64).mean(axis=0)

    diffs = class_means - global_mean  # (n_classes, d)
    S_B = (diffs.T * class_counts) @ diffs  # (d, d)

    # S_W via the class-centered view of X
    emb_class_centered = emb.astype(np.float64) - class_means[inverse]
    S_W = emb_class_centered.T @ emb_class_centered
    # Ridge for numerical stability (covers low-rank S_W from classes
    # smaller than d; also cheap insurance).
    ridge = 1e-4 * np.trace(S_W) / d
    S_W = S_W + ridge * np.eye(d, dtype=np.float64)

    import scipy.linalg
    eigvals, eigvecs = scipy.linalg.eigh(S_B, S_W)
    # Ascending; the most class-discriminating directions are the last k.
    top_k = eigvecs[:, -drop_k:].astype(emb.dtype)
    # S_B-S_W GEP eigvecs aren't orthonormal under the standard inner product;
    # orthonormalise via QR before building the projector.
    q, _ = np.linalg.qr(top_k)
    projector = np.eye(d, dtype=emb.dtype) - q @ q.T
    centered = emb - emb.mean(axis=0).astype(emb.dtype)
    return (centered @ projector).astype(emb.dtype)


def _class_labels(
    topic_ids: List[str], types: List[str], axis: str,
) -> List[str]:
    """Build per-work class labels for LDA projection.

    ``topic``: primary_topic_id (None → "__unknown__").
    ``type``: work_type.
    ``topic_x_type``: cartesian product. Using this projects out both
    between-topic and between-work-type variance together, which matters
    for gap detection: if books and articles separate along a strong
    embedding axis, clusters landing on the article side are
    artificially "book-free" without reflecting any topic-level gap.
    """
    if axis == "topic":
        return ["__unknown__" if t is None else t for t in topic_ids]
    if axis == "type":
        return list(types)
    if axis == "topic_x_type":
        return [f"{t or '__unknown__'}|{wt}" for t, wt in zip(topic_ids, types)]
    raise ValueError(f"unknown class axis {axis!r}")


def run(scope: str, min_cluster_size: int, min_samples: int,
        top_clusters: int, size_floor: int,
        cluster_selection_method: str = "eom",
        cluster_selection_epsilon: float = 0.0,
        drop_k: int = 0,
        drop_k_graph: int = 0,
        denoiser: str = "pca",
        class_axis: str = "topic_x_type",
        cclip_alpha: Optional[float] = None) -> None:
    from fast_hdbscan import HDBSCAN

    needs_graph = denoiser in {"lpp", "lip", "lip_eigen", "lip_diff",
                               "cclip", "hierarchical", "criteria_union", "rnojd"}
    print(f"Loading scope {scope}...")
    emb, ids, types, titles, topic_ids, refs = load_scope(scope, with_refs=needs_graph)
    n, d = emb.shape
    print(f"  {n:,} works, {d}-d embeddings")
    print(f"  type counts: {dict(Counter(types))}")
    print(f"  topic count (path-derived): {len(set(topic_ids))}")

    if denoiser == "pca":
        if drop_k > 0:
            print(f"Denoiser pca: projecting out top {drop_k} PCA components...")
            emb = _drop_top_k_pca(emb, drop_k)
    elif denoiser == "multi_lda":
        if drop_k > 0:
            labels = _class_labels(topic_ids, types, class_axis)
            n_classes = len(set(labels))
            print(f"Denoiser multi_lda[{class_axis}]: projecting out top "
                  f"{drop_k} between-class directions ({n_classes} classes)...")
            emb = _drop_top_k_multi_lda(emb, labels, drop_k)
    elif denoiser in {"lpp", "lip", "lip_eigen", "lip_diff",
                      "cclip", "hierarchical", "criteria_union", "rnojd"}:
        labels = _class_labels(topic_ids, types, class_axis)
        n_classes = len(set(labels))
        print(f"Building in-scope citation graph for {n:,} works...")
        assert refs is not None
        A = _build_in_scope_citation_csr(ids, refs)
        print(f"  citation graph: nnz={A.nnz:,}, density={A.nnz/(n*n):.2e}")
        ctx = _ProbeDenoiserContext(emb, labels, A)
        n_keep = max(1, d - drop_k)
        if denoiser == "lpp":
            print(f"Denoiser lpp: keeping {n_keep} smallest eigvecs of Xc^T L Xc / Xc^T D Xc...")
            basis = _denoise_lpp(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "lip":
            print(f"Denoiser lip: keeping {n_keep} smallest eigvecs of S_L / S_T...")
            basis = _denoise_lip(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "lip_eigen":
            print(f"Denoiser lip_eigen: keeping {n_keep} smallest eigvecs of S_L alone (no GEP)...")
            basis = _denoise_lip_eigen_only(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "lip_diff":
            print(f"Denoiser lip_diff: keeping {n_keep} smallest eigvecs of S_L / (S_T − S_L)...")
            basis = _denoise_lip_diff(ctx, n_keep)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "cclip":
            alpha_used = 0.5 if cclip_alpha is None else cclip_alpha
            print(f"Denoiser cclip[{class_axis}]: α={alpha_used:.3f}, "
                  f"keeping {n_keep} smallest eigvecs of "
                  f"((1-α) Ŝ_B + α Ŝ_L) / S_T, pre-trace-normalised "
                  f"({n_classes} classes)...")
            basis, _ = _denoise_cclip(ctx, n_keep, alpha_used)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "hierarchical":
            print(f"Denoiser hierarchical[{class_axis}]: project out top {drop_k} class-disc, "
                  f"then top {drop_k_graph} citation-incoherent in residual...")
            basis = _denoise_hierarchical(ctx, drop_k_class=drop_k, drop_k_graph=drop_k_graph)
            print(f"  final basis: {basis.shape[1]} kept dims out of {d}")
            emb = _project_via_basis(emb, basis)
        elif denoiser == "criteria_union":
            print(f"Denoiser criteria_union[{class_axis}]: project out union of top "
                  f"{drop_k} eigvecs of (S_B, S_T) and top {drop_k_graph} of (S_L, S_T)...")
            basis = _denoise_criteria_union(ctx, drop_k_class=drop_k, drop_k_graph=drop_k_graph)
            emb = _project_via_basis(emb, basis)
        elif denoiser == "rnojd":
            print(f"Denoiser rnojd[{class_axis}]: jointly diagonalise {{S_B, S_L, S_T}}, "
                  f"drop top {drop_k} scoring axes...")
            basis = _denoise_rnojd(ctx, n_drop=drop_k)
            emb = _project_via_basis(emb, basis)
    else:
        raise ValueError(f"unknown denoiser {denoiser!r}")

    # Normalise so Euclidean on unit sphere ≈ cosine-distance ranking.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.where(norms > 0, norms, 1.0)

    print(f"Running fast_hdbscan (min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}, method={cluster_selection_method}, "
          f"epsilon={cluster_selection_epsilon})...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    labels = clusterer.fit_predict(emb_norm)

    n_noise = int((labels == -1).sum())
    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0

    print(f"\n=== Clustering results ===")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise: {n_noise:,} ({100 * n_noise / n:.1f}%)")

    if n_clusters == 0:
        return

    unique_labels, sizes = np.unique(labels[labels >= 0], return_counts=True)

    print(f"\n=== Cluster size histogram ===")
    for lo, hi in [(2, 10), (10, 50), (50, 100), (100, 500),
                   (500, 1000), (1000, 5000), (5000, 20000),
                   (20000, 1_000_000)]:
        h = int(((sizes >= lo) & (sizes < hi)).sum())
        print(f"  [{lo:>6}, {hi:>6}): {h}")
    print(f"  max cluster size: {int(sizes.max()):,}")

    types_arr = np.array(types)
    is_book = types_arr == "book"
    n_books_total = int(is_book.sum())
    corpus_book_rate = n_books_total / n

    print(f"\n=== Cluster book-fraction analysis (size >= {size_floor}, "
          f"corpus book rate = {corpus_book_rate:.3%}) ===")

    # Per-cluster (id, size, books, expected_books, p_value, effect_size).
    # p_value = P(X <= k | n, p) under H0 (cluster uniform-random from corpus).
    # effect_size = (expected - observed) / expected ∈ (-∞, 1]
    #   = 1.0 means perfectly depleted (observed = 0)
    #   = 0.0 means observed at base-rate
    #   < 0  means cluster is book-ENRICHED, not a gap
    from scipy.stats import binom
    rows = []
    for c in unique_labels:
        mask = labels == c
        sz = int(mask.sum())
        if sz < size_floor:
            continue
        books = int(is_book[mask].sum())
        expected = sz * corpus_book_rate
        p_value = float(binom.cdf(books, sz, corpus_book_rate))
        effect = (expected - books) / expected if expected > 0 else 0.0
        rows.append((int(c), sz, books, expected, p_value, effect))
    rows.sort(key=lambda r: (r[4], -r[1]))  # ascending p, ties → larger first

    n_strictly = sum(1 for _, _, b, _, _, _ in rows if b == 0)
    n_sig_001 = sum(1 for _, _, _, _, p, _ in rows if p < 1e-3)
    n_sig_0001 = sum(1 for _, _, _, _, p, _ in rows if p < 1e-4)
    n_sig_000001 = sum(1 for _, _, _, _, p, _ in rows if p < 1e-6)

    # Benjamini-Hochberg FDR: rank p-values ascending; find largest i where
    # p_(i) <= (i / N_tests) * alpha. Declare ranks 1..i discoveries.
    # Return number of discoveries at each FDR threshold.
    def _bh_count(alpha: float) -> int:
        N_tests = len(rows)
        if N_tests == 0:
            return 0
        ps = np.array([r[4] for r in rows])  # already sorted ascending
        thresholds = (np.arange(1, N_tests + 1) / N_tests) * alpha
        passes = ps <= thresholds
        if not passes.any():
            return 0
        # largest i where passes[i] is True; everything up to i is discovery
        return int(np.where(passes)[0].max() + 1)

    print(f"  qualifying clusters:              {len(rows)}")
    print(f"  strictly book-free (books==0):    {n_strictly}  [raw count]")
    print(f"  significantly depleted (p<1e-3):  {n_sig_001}")
    print(f"  significantly depleted (p<1e-4):  {n_sig_0001}")
    print(f"  very significantly dep (p<1e-6):  {n_sig_000001}")
    print(f"  FDR-discoveries at α=0.05:        {_bh_count(0.05)}")
    print(f"  FDR-discoveries at α=0.10:        {_bh_count(0.10)}")
    print(f"  FDR-discoveries at α=0.20:        {_bh_count(0.20)}")

    titles_arr = np.array(titles, dtype=object)

    print(f"\n=== Top {top_clusters} clusters by binomial significance "
          f"(lowest p, size-tiebreak) ===")
    if not rows:
        print("  (none found — no qualifying clusters at this threshold)")
    for c, sz, books, expected, p, effect in rows[:top_clusters]:
        mask = labels == c
        print(f"\n  Cluster {c} (size {sz:,}, books={books}, expected={expected:.1f}, "
              f"p={p:.2e}, effect={effect:.3f}):")
        for t in titles_arr[mask][:5]:
            print(f"    - {(t or '(no title)')[:90]}")

    # Also show top by effect size, restricted to those with non-trivial
    # significance — avoids elevating small clusters where effect=1.0 is
    # cheap (size 100, expected 0.76, observed 0 → effect=1, p=0.46).
    sig_rows = [r for r in rows if r[4] < 0.05]
    sig_rows.sort(key=lambda r: -r[5])  # descending effect size
    print(f"\n=== Top {top_clusters} clusters by effect size "
          f"among p<0.05 (largest depletion) ===")
    if not sig_rows:
        print("  (none found — no clusters with p<0.05)")
    for c, sz, books, expected, p, effect in sig_rows[:top_clusters]:
        mask = labels == c
        print(f"\n  Cluster {c} (size {sz:,}, books={books}, expected={expected:.1f}, "
              f"p={p:.2e}, effect={effect:.3f}):")
        for t in titles_arr[mask][:5]:
            print(f"    - {(t or '(no title)')[:90]}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True, help="F/SF or F/SF/T")
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--top-clusters", type=int, default=5)
    p.add_argument("--size-floor", type=int, default=100,
                   help="smallest cluster counted in book-fraction report")
    p.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="eom")
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    p.add_argument("--drop-k", type=int, default=0,
                   help="project out top-k components before clustering")
    p.add_argument("--denoiser",
                   choices=["pca", "multi_lda", "lpp", "lip", "lip_eigen",
                            "lip_diff", "cclip", "hierarchical",
                            "criteria_union", "rnojd"],
                   default="pca",
                   help="pca=variance-greedy; multi_lda=between-class GEP; "
                        "lpp/cclip/hierarchical/rnojd=citation-graph based")
    p.add_argument("--class-axis", choices=["topic", "type", "topic_x_type"],
                   default="topic_x_type",
                   help="class partition (used by multi_lda, cclip, hierarchical, rnojd)")
    p.add_argument("--cclip-alpha", type=float, default=None,
                   help="α for CCLIP; default auto-picks via trace normalisation")
    p.add_argument("--drop-k-graph", type=int, default=2,
                   help="for hierarchical: drop_k for the graph-Laplacian step")
    args = p.parse_args()
    run(args.scope, args.min_cluster_size, args.min_samples,
        args.top_clusters, args.size_floor,
        args.cluster_selection_method, args.cluster_selection_epsilon,
        args.drop_k, args.drop_k_graph, args.denoiser, args.class_axis,
        args.cclip_alpha)


if __name__ == "__main__":
    main()
