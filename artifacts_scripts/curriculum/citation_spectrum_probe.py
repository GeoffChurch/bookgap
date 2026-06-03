"""Probe the citation graph's spectrum + its overlap with the keyword view.

Answers the 2026-06-03 questions:
  1. Effective rank of the citation graph under BOTH soft-rank notions:
       - entropy effective rank (Roy & Vetterli 2007): erank = exp(H), H = -Σ p_i ln p_i,
         p_i = σ_i / Σσ_j  (L1-normalized singular values).
       - participation ratio (Rényi-2 / "trace-normalize then sum-of-squares"):
         PR = (Σλ_i)² / Σλ_i² = 1/Σ q_i²,  q_i = λ_i/Σλ_j,  λ = σ²  (trace-normalized eigenvalues).
       - stable rank ‖A‖_F²/‖A‖_2² = Σλ/λ_max, reported alongside as a third common soft rank.
  2. Directed vs symmetrized (G+Gᵀ) vs concat [out‖in] — which representation, and how does
     rank change.
  3. THE PUZZLE: is the keyword overlap R²≈0.04 an artifact of the internal-subgraph restriction
     (G[rows][:,rows] drops every reference leaving the 9886-work concept — most of each paper's
     refs) or genuine citation⊥keyword independence? Test by re-measuring overlap with the FULL
     out-citation profile (bibliographic coupling over all field targets) and co-citation.

Each variant's overlap with the keyword LSA T is the SAME lstsq R² used to get 0.04 in
cluster_concept_cca's discard-clash diagnostic, plus the canonical-correlation spectrum.

Regenerate (do not edit by hand):
  python -m artifacts_scripts.curriculum.citation_spectrum_probe \
    --concept-gap artifacts/book_gaps/jurafsky_concept_full/concept_gap.json --field 17
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.field_term_index import build_field_term_index
from artifacts_scripts.curriculum.multi_view_spectral import build_citation_similarity_view
from bookgap.util import LOGGER, setup_logging


def soft_ranks(lam: np.ndarray) -> tuple[float, float, float, int]:
    """Return (entropy effective rank, participation ratio, stable rank, hard rank) from
    eigenvalues lam (= squared singular values) of a Gram matrix."""
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, None)
    lam = lam[lam > lam.max() * 1e-10]                 # drop numerical zeros
    sig = np.sqrt(lam)
    p = sig / sig.sum()                                # L1-normalized singular values
    erank = float(np.exp(-(p * np.log(p)).sum()))      # Roy-Vetterli entropy effective rank
    q = lam / lam.sum()                                # trace-normalized eigenvalues
    pr = float(1.0 / (q * q).sum())                    # participation ratio = (Σλ)²/Σλ²
    stable = float(lam.sum() / lam.max())              # ‖A‖_F²/‖A‖_2²
    return erank, pr, stable, int(lam.size)


def eig_features(X: sp.spmatrix, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Full eigenspectrum of the row-space Gram X Xᵀ (≤ n_rows nonzero) AND the top-dim
    left-singular scores (= TruncatedSVD transform, for the overlap test) in one shot."""
    G = (X @ X.T)
    G = G.toarray() if sp.issparse(G) else np.asarray(G)
    G = 0.5 * (G + G.T)
    lam, U = np.linalg.eigh(G)
    lam = lam[::-1]; U = U[:, ::-1]
    k = min(dim, (lam > lam.max() * 1e-10).sum())
    scores = U[:, :k] * np.sqrt(np.clip(lam[:k], 0, None))
    return lam, scores


def overlap(T: np.ndarray, C: np.ndarray) -> tuple[float, np.ndarray]:
    """(lstsq R² of T on C, canonical-correlation spectrum)."""
    Tc = T - T.mean(0); Cc = C - C.mean(0)
    coef, *_ = np.linalg.lstsq(Cc, Tc, rcond=None)
    r2 = 1.0 - float(((Tc - Cc @ coef) ** 2).sum() / (Tc ** 2).sum())
    Ut, _, _ = np.linalg.svd(Tc, full_matrices=False)
    Uc, _, _ = np.linalg.svd(Cc, full_matrices=False)
    rho = np.linalg.svd(Ut.T @ Uc, compute_uv=False)
    return r2, rho


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept-gap", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--lsa-dim", type=int, default=200)
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=1_000_000)
    ap.add_argument("--snapshot-date", default=None)
    args = ap.parse_args()
    setup_logging(log_name="cit_probe", log_level="INFO", include_time=False)

    X, _vocab, _fdf = build_field_term_index(args.field, args.snapshot_date, args.min_df, args.max_features)
    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}
    ids = json.loads(args.concept_gap.read_text())["book"]["gap_article_ids"]
    rows = [id2idx[w] for w in ids if w in id2idx]
    N = len(rows)
    dim = min(args.lsa_dim, N - 1)
    LOGGER.info("concept: %d works in field %s", N, args.field)

    # keyword view T (identical construction to cluster_concept_cca)
    Xc = X[rows].astype(np.float64)
    idf_c = np.log(N / (np.asarray((Xc > 0).sum(0)).ravel() + 1.0))
    T = TruncatedSVD(n_components=min(dim, Xc.shape[1] - 1), random_state=0).fit_transform(
        normalize(Xc.multiply(idf_c).tocsr()))
    del X, Xc                               # free the ~multi-GB field term index before the graph work

    # citation graph: directed G[i,j] = i cites j; strip the PageRank self-loops.
    G = cloud.citation_graph(False).tocsr().astype(np.float64)
    G.setdiag(0.0); G.eliminate_zeros()
    Gint = G[rows][:, rows]                 # internal concept subgraph (what we use today)
    Xout = G[rows]                          # full out-citations  -> bibliographic coupling features
    Xin = G.tocsc()[:, rows].T.tocsr()      # full in-citations   -> co-citation features
    rs = np.diff(Xout.indptr); LOGGER.info("internal nnz=%d; mean out-refs(in-field)=%.1f",
                                           Gint.nnz, rs.mean())

    variants = {
        "internal_directed": Gint,
        "internal_symmetrized": Gint + Gint.T,
        "full_coupling(out)": Xout,
        "full_cocitation(in)": Xin,
        "full_concat[out|in]": sp.hstack([Xout, Xin]).tocsr(),
    }

    print(f"\n{'variant':<26} {'erank':>8} {'partic':>8} {'stable':>8} {'hard':>6} "
          f"{'kw R²':>7} {'ρ1':>6} {'ρ̄(10)':>7}")
    print("-" * 86)
    # the EXACT current feature (weighted_unified Ruzicka similarity on the internal subgraph)
    Csim = build_citation_similarity_view(Gint, csls_k=10)
    lam_cur = np.linalg.eigvalsh(0.5 * (Csim + Csim.T))[::-1]
    Ccur = TruncatedSVD(n_components=dim, random_state=0).fit_transform(Csim)
    er, pr, st, hd = soft_ranks(np.clip(lam_cur, 0, None))
    r2, rho = overlap(T, Ccur)
    print(f"{'CURRENT (Ruzicka int.)':<26} {er:>8.1f} {pr:>8.1f} {st:>8.1f} {hd:>6} "
          f"{r2:>7.3f} {rho[0]:>6.3f} {rho[:10].mean():>7.3f}")

    for name, Xv in variants.items():
        lam, C = eig_features(Xv, dim)
        er, pr, st, hd = soft_ranks(lam)
        r2, rho = overlap(T, C)
        print(f"{name:<26} {er:>8.1f} {pr:>8.1f} {st:>8.1f} {hd:>6} "
              f"{r2:>7.3f} {rho[0]:>6.3f} {rho[:10].mean():>7.3f}")
    print("\nerank=entropy effective rank (exp Shannon of L1-normed σ); partic=participation ratio "
          "(Σλ)²/Σλ²; stable=‖·‖_F²/‖·‖_2²; kw R²=keyword variance explained by the citation view.")


if __name__ == "__main__":
    main()
