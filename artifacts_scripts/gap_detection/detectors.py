"""Gap detectors + metrics. Pure numpy/scipy/sklearn, CPU only.

Core statistic (docs/backlog/curriculum-applications.md, "Gap detection, revisited"):

    gap(x) = g(x)^a * z( log g(x)/p(x) ),   x in candidate set (bg points + probes)

Effect size = the kNN-radius density log-ratio (scale-adaptive, fringe-robust):

    Delta(x) = log r_F,k(x) - log r_B,k(x)

Variance normalization = a lower-confidence-bound shrinkage ("emptiness must be
statistically earned"), with the count variance measured in the adaptive max-ball
R(x) = max(r_F,k, r_B,k) — the smallest scale at which both densities are estimable:

    sigma(x) = sqrt( 1/(n_F,R + a0) + 1/(n_B,R + a0) ),   z(x) = Delta(x) - lam*sigma(x)

Two rejected z-forms, kept on record (both measured to invert rankings here):
  * Delta_knn / sigma_count: incoherent units — 1/n_F,loc explodes exactly where fg
    is absent (the signal), so fg-dense controls outrank planted holes.
  * Delta_count / sigma_count (textbook Fightin' Words ratio): pure significance —
    amplifies tiny well-evidenced imbalances (fg-dense regions, bg fringes) over
    large low-count effects, at any counting scale. LCB keeps the earned-emptiness
    intent without the significance != effect-size pathology.

Habitability factor: g(x)^a ~ exp(a * standardized(-log r_B,k(x))).

Variants: 'full' (hab * z), 'no_hab' (distance-to-fg only), 'no_z' (hab * Delta),
plus purified 'z_only' / 'delta_only' columns.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

EPS = 1e-9
VARIANTS = ("full", "no_hab", "no_z", "z_only", "delta_only")


# --------------------------------------------------------------------------- kNN plumbing
def knn_radii(query: np.ndarray, ref: np.ndarray, k: int, *, ref_is_query: bool = False) -> np.ndarray:
    """Distance to the k-th nearest reference point (self excluded when ref==query)."""
    kk = k + 1 if ref_is_query else k
    kk = min(kk, ref.shape[0])
    nn = NearestNeighbors(n_neighbors=kk, algorithm="brute").fit(ref)
    d, _ = nn.kneighbors(query)
    col = kk - 1
    return np.maximum(d[:, col], EPS)


def count_within_radius(
    query: np.ndarray, ref: np.ndarray, radius: float | np.ndarray, *, chunk: int = 2048,
    ref_is_query: bool = False,
) -> np.ndarray:
    """# reference points within `radius` (scalar or per-query) of each query point."""
    ref32 = ref.astype(np.float32)
    ref_sq = np.einsum("ij,ij->i", ref32, ref32)
    out = np.empty(query.shape[0], dtype=np.int64)
    r2 = np.square(np.broadcast_to(np.asarray(radius, dtype=np.float32), (query.shape[0],)))
    for s in range(0, query.shape[0], chunk):
        q = query[s : s + chunk].astype(np.float32)
        d2 = np.maximum(
            np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ ref32.T) + ref_sq[None, :], 0.0
        )
        out[s : s + chunk] = (d2 <= r2[s : s + chunk, None] * (1 + 1e-6)).sum(axis=1)
    if ref_is_query:
        out = out - 1  # remove self-count
    return out


def make_probes(Z_all: np.ndarray, n: int, rng: np.random.Generator,
                scale_range: tuple[float, float] = (1.3, 2.5)) -> np.ndarray:
    """Off-cloud probe candidates: data points pushed radially out from the centroid.

    These are 'uninhabitable' locations (no bg mass lives there) used to expose the
    everything-outside-the-cloud pathology of habitability-free scoring.
    """
    centroid = Z_all.mean(axis=0)
    radii = np.linalg.norm(Z_all - centroid, axis=1)
    r95 = float(np.quantile(radii, 0.95))
    base = Z_all[rng.choice(Z_all.shape[0], size=n, replace=False)]
    d = base - centroid
    dirs = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), EPS)
    s = rng.uniform(*scale_range, size=(n, 1)).astype(Z_all.dtype)
    return (centroid + dirs * (r95 * s)).astype(Z_all.dtype)


# --------------------------------------------------------------------------- the statistic
@dataclass
class GapScorer:
    """Scores a fixed candidate set (bg points + optional probes) against any fg cloud.

    Background-side quantities are computed once at construction; per-trial cost is
    fg-side only (fg changes when holes are planted).
    """

    bg: np.ndarray                       # (n_b, d) background points
    probes: np.ndarray | None = None     # (n_p, d) synthetic off-cloud candidates
    k: int = 10
    alpha_prior: float = 1.0             # Dirichlet prior on local counts
    hab_exp: float = 0.5                 # 'a' in g^a
    lcb_lambda: float = 0.5              # confidence shrinkage 'lam' in z = Delta - lam*sigma
    r0: float | None = None              # diagnostics-count radius; set from first fg

    def __post_init__(self) -> None:
        self.cand = (
            np.vstack([self.bg, self.probes]) if self.probes is not None else self.bg.copy()
        )
        self.real_mask = np.zeros(self.cand.shape[0], dtype=bool)
        self.real_mask[: self.bg.shape[0]] = True
        # r_B,k for every candidate (self excluded for real bg rows).
        nb = self.bg.shape[0]
        r_real = knn_radii(self.bg, self.bg, self.k, ref_is_query=True)
        if self.probes is not None:
            r_probe = knn_radii(self.probes, self.bg, self.k)
            self.r_Bk = np.concatenate([r_real, r_probe])
        else:
            self.r_Bk = r_real
        self._log_rBk = np.log(self.r_Bk)
        mu, sd = self._log_rBk[: nb].mean(), self._log_rBk[: nb].std() + EPS
        self._hab = np.exp(self.hab_exp * (-(self._log_rBk - mu) / sd))
        self._n_Bloc: np.ndarray | None = None  # lazy: needs r0

    # -- bg-side local counts (fixed once r0 is known)
    def _ensure_bg_counts(self) -> None:
        if self._n_Bloc is None:
            assert self.r0 is not None, "r0 must be set before scoring"
            nb = self.bg.shape[0]
            cnt = count_within_radius(self.cand, self.bg, self.r0)
            cnt[:nb] -= 1  # self
            self._n_Bloc = np.maximum(cnt, 0)

    def set_r0_from_fg(self, fg: np.ndarray) -> float:
        """Default neighborhood scale: median r_F,k over real bg candidates."""
        if self.r0 is None:
            r = knn_radii(self.bg, fg, self.k)
            self.r0 = float(np.median(r))
        return self.r0

    @property
    def n_Bloc(self) -> np.ndarray:
        self._ensure_bg_counts()
        assert self._n_Bloc is not None
        return self._n_Bloc

    def score(self, fg: np.ndarray) -> dict[str, np.ndarray]:
        """Score all candidates against the given foreground; returns all variants."""
        if self.r0 is None:
            self.set_r0_from_fg(fg)
        a0 = self.alpha_prior
        nb = self.bg.shape[0]
        r_Fk = knn_radii(self.cand, fg, self.k)
        # effect size: kNN-radius density log-ratio (scale-adaptive)
        delta = np.log(r_Fk) - self._log_rBk
        # evidence at the adaptive max-ball scale (both clouds estimable there)
        R = np.maximum(r_Fk, self.r_Bk)
        n_FR = count_within_radius(self.cand, fg, R)
        n_BR = count_within_radius(self.cand, self.bg, R)
        n_BR[:nb] -= 1  # self
        sigma = np.sqrt(1.0 / (n_FR + a0) + 1.0 / (np.maximum(n_BR, 0) + a0))
        z = delta - self.lcb_lambda * sigma  # lower-confidence-bound shrinkage
        return {
            "full": self._hab * z,
            "no_hab": r_Fk,             # ablation (i): distance-to-fg only
            "no_z": self._hab * delta,  # ablation (ii): variance normalization off
            "z_only": z,
            "delta_only": delta,
            "_r_Fk": r_Fk,
            "_n_Floc": n_FR.astype(np.float64),
        }

    def habitable_mask(self, q: float = 0.90) -> np.ndarray:
        """Candidates inside the bg-dense hull: r_B,k below the q-quantile of real bg."""
        thr = np.quantile(self.r_Bk[self.real_mask], q)
        return self.r_Bk <= thr

    def bg_density_stratum(self, n_strata: int = 3) -> np.ndarray:
        """Stratum per candidate by local bg count (0 = sparsest)."""
        ranks = np.searchsorted(
            np.quantile(self.n_Bloc[self.real_mask], np.linspace(0, 1, n_strata + 1)[1:-1]),
            self.n_Bloc, side="right",
        )
        return ranks


# --------------------------------------------------------------------------- level sets
def mutual_knn_graph(X: np.ndarray, k: int = 10) -> sp.csr_matrix:
    """Symmetric mutual-kNN connectivity graph."""
    k = min(k, max(X.shape[0] - 1, 1))
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    return A.minimum(A.T).tocsr()


def effective_number(weights: np.ndarray) -> float:
    """Perplexity of a (sub)distribution: exp(entropy of normalized weights)."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0) + EPS
    w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def level_set_components(
    scores: np.ndarray, X: np.ndarray, real_mask: np.ndarray,
    top_frac: float = 0.05, k_graph: int = 10, min_size: int = 3,
) -> list[dict]:
    """Top-`top_frac` scored real candidates -> mutual-kNN graph -> components.

    Components are sized by the effective number (perplexity) of their score mass and
    returned sorted by n_eff descending. Member indices are global candidate indices.
    """
    real_idx = np.where(real_mask)[0]
    order = real_idx[np.argsort(scores[real_idx])[::-1]]
    m = max(int(np.ceil(top_frac * real_idx.size)), min_size)
    top = order[:m]
    A = mutual_knn_graph(X[top], k=k_graph)
    _, labels = connected_components(A, directed=False)
    comps = []
    for lab in np.unique(labels):
        members = top[labels == lab]
        if members.size < min_size:
            continue
        w = np.maximum(scores[members], 0)
        comps.append(
            {"members": members, "n_eff": effective_number(w),
             "size": int(members.size), "mean_score": float(scores[members].mean())}
        )
    comps.sort(key=lambda c: c["n_eff"], reverse=True)
    return comps


# --------------------------------------------------------------------------- pattern miner
def mine_frequent_itemsets(
    Kb: np.ndarray, min_support: int, max_len: int = 3, max_pairs_expand: int = 30_000,
) -> dict[frozenset, int]:
    """Apriori over a dense boolean doc x keyword matrix; returns {itemset: support}."""
    Kb = np.asarray(Kb, dtype=bool)
    sup1 = Kb.sum(axis=0)
    f1 = np.where(sup1 >= min_support)[0]
    family: dict[frozenset, int] = {frozenset([int(i)]): int(sup1[i]) for i in f1}
    if max_len < 2 or f1.size < 2:
        return family
    Kf = Kb[:, f1].astype(np.float32)
    C = Kf.T @ Kf  # pair co-occurrence counts
    iu, ju = np.triu_indices(f1.size, k=1)
    ok = C[iu, ju] >= min_support
    pi, pj = iu[ok], ju[ok]
    pair_sup = C[pi, pj].astype(np.int64)
    for a, b, s in zip(f1[pi], f1[pj], pair_sup):
        family[frozenset([int(a), int(b)])] = int(s)
    if max_len < 3 or pi.size == 0:
        return family
    # Triples: extend each frequent pair (i<j in f1-order) by keywords l>j (dedup).
    if pi.size > max_pairs_expand:  # keep compute bounded; prefer high-support pairs
        keep = np.argsort(pair_sup)[::-1][:max_pairs_expand]
        pi, pj = pi[keep], pj[keep]
    Kb_f = Kb[:, f1]
    for s0 in range(0, pi.size, 512):
        sl = slice(s0, min(s0 + 512, pi.size))
        Bpair = (Kb_f[:, pi[sl]] & Kb_f[:, pj[sl]]).astype(np.float32)  # (n, chunk)
        sup3 = Bpair.T @ Kf  # (chunk, |f1|)
        rows, cols = np.where(sup3 >= min_support)
        for r, c in zip(rows, cols):
            j = pj[sl][r]
            if c <= j:  # only extend upward to avoid duplicates
                continue
            key = frozenset([int(f1[pi[sl][r]]), int(f1[j]), int(f1[c])])
            family[key] = int(sup3[r, c])
    return family


def closed_filter(family: dict[frozenset, int]) -> dict[frozenset, int]:
    """Keep only closed itemsets (no superset in the family with equal support)."""
    closed = dict.fromkeys(family, True)
    for T, sup_t in family.items():
        if len(T) < 2:
            continue
        for item in T:
            S = T - {item}
            if family.get(S) == sup_t:
                closed[S] = False
    return {s: v for s, v in family.items() if closed[s]}


def fightin_words_rank(
    family: dict[frozenset, int], Kf: np.ndarray, n_bg: int,
    alpha: float = 1.0, min_len: int = 2,
) -> list[dict]:
    """Rank itemsets by z(log-odds bg - log-odds fg): bg-frequent & fg-empty first."""
    Kf = np.asarray(Kf, dtype=bool)
    n_fg = Kf.shape[0]
    out = []
    for S, n_b in family.items():
        if len(S) < min_len:
            continue
        cols = sorted(S)
        v = Kf[:, cols[0]].copy()
        for c in cols[1:]:
            v &= Kf[:, c]
        n_f = int(v.sum())
        d_b = np.log((n_b + alpha) / (n_bg - n_b + alpha))
        d_f = np.log((n_f + alpha) / (n_fg - n_f + alpha))
        sigma = np.sqrt(1.0 / (n_b + alpha) + 1.0 / (n_f + alpha))
        out.append(
            {"itemset": S, "z": float((d_b - d_f) / sigma),
             "bg_support": int(n_b), "fg_support": n_f}
        )
    out.sort(key=lambda r: r["z"], reverse=True)
    return out


# --------------------------------------------------------------------------- voltage bridge
def voltage_current_scores(
    X: np.ndarray, clamp0: np.ndarray, clamp1: np.ndarray,
    k_graph: int = 10, leakage: float = 1e-8, mid_weight: bool = True,
) -> np.ndarray:
    """Clamp two terminal sets at 0V/1V on a bg kNN graph; return per-node bridge score.

    Reuses artifacts_scripts.gap_detection.circuitrank (Dirichlet solve + current flow). The graph is
    a symmetrized (union) kNN graph with Gaussian conductances exp(-(d/sigma)^2).
    With mid_weight=True the raw node current is weighted by 4*phi*(1-phi): a bridging
    corridor carries high through-current at mid potential, whereas the (uninformative)
    current maxima hugging the terminals sit at phi ~ 0/1 and are suppressed.
    """
    from artifacts_scripts.gap_detection.circuitrank import compute_current_flow, compute_graph_potential

    k_graph = min(k_graph, max(X.shape[0] - 1, 1))
    D = kneighbors_graph(X, n_neighbors=k_graph, mode="distance", include_self=False).tocoo()
    sigma = np.median(D.data) + EPS
    W = sp.coo_matrix((np.exp(-((D.data / sigma) ** 2)), (D.row, D.col)), shape=D.shape).tocsr()
    W = W.maximum(W.T)
    fixed: dict[int, float] = {int(i): 0.0 for i in clamp0}
    fixed.update({int(i): 1.0 for i in clamp1})
    phi = np.asarray(
        compute_graph_potential(
            W, fixed_potentials=fixed, normalize=False, edge_weighting="pagerank",
            leakage=leakage, gmres_tol=1e-8,
        )
    )
    cur = compute_current_flow(W, phi, normalize=False, edge_weighting="pagerank")
    if mid_weight:
        p = np.clip(phi, 0.0, 1.0)
        cur = cur * (4.0 * p * (1.0 - p))
    return cur


# --------------------------------------------------------------------------- metrics
def rank_of(scores: np.ndarray, member_mask: np.ndarray, eligible: np.ndarray) -> int:
    """Best (1-based) rank of any member among eligible candidates, by descending score."""
    idx = np.where(eligible)[0]
    order = idx[np.argsort(scores[idx])[::-1]]
    hits = np.where(member_mask[order])[0]
    return int(hits[0]) + 1 if hits.size else int(idx.size) + 1


def precision_at(scores: np.ndarray, member_mask: np.ndarray, eligible: np.ndarray, n: int = 50) -> float:
    idx = np.where(eligible)[0]
    order = idx[np.argsort(scores[idx])[::-1]][:n]
    return float(member_mask[order].mean()) if order.size else 0.0


def topk_indices(scores: np.ndarray, eligible: np.ndarray, n: int = 50) -> np.ndarray:
    idx = np.where(eligible)[0]
    return idx[np.argsort(scores[idx])[::-1]][:n]


def auc_vs_controls(scores: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score(planted-region bg) > score(matched-control bg))."""
    pos = scores[pos_idx]
    neg = scores[neg_idx]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (pos.size * neg.size))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
