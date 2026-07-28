"""Track 1 finish: type-blind composite machinery (task_t1_comp, 2026-06-11 dispatch).

The composite detector receives the AFTER state only (pruned fg coordinates +
pruned fg keyword rows) and NO gap-type label. Three channels, per the banked
verdicts (A: fused r_F+veto; P1: miner-led arbitration rejected, miner is one
voice/corroboration, never the router; P3a/P2: change fields closed for
single-snapshot Track 1):

  geo   scoped fusion r_F,k + veto (fusion.py verbatim)        — radial home
  pat   miner doc layer: per-doc max Fightin'-Words z over the
        scoped closed-itemset family (polish.py P1 machinery)  — concept's
        doc-space projection
  volt  single-pair voltage with a SELF-SELECTED pair: KMeans the after-fg,
        rank cluster pairs by band-EMPTINESS (fg pushed out of the
        equidistance band relative to bg), voltage the top pair with the
        banked extended-electrode + 4*phi*(1-phi) recipe        — corridor home,
        minus the pair oracle

Arbitration = symmetric calibrated rank fusion: each channel's scores are
converted to in-trial percentiles over the real candidate universe (abstain /
veto = 0, finite scores ranked among themselves so a sparse channel's top claim
is comparable to a dense channel's), then combined by MAX (noisy-OR: any
channel's confident claim surfaces; no channel leads). Ablation rules: MEAN
percentile, reciprocal-rank fusion, and geo+volt (miner dropped).

All functions are pure CPU numpy/scipy/sklearn; randomness through seeds only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.

import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from sklearn.cluster import KMeans

from artifacts_scripts.gap_detection.synth import _axis_t

EPS = 1e-12


# ---------------------------------------------------------------- percentile fusion
def percentile_scores(sc: np.ndarray) -> np.ndarray:
    """In-trial percentile in (0, 1]: finite scores ranked ascending among the
    FINITE entries only (top finite claim -> 1.0); non-finite (abstain/veto) -> 0.

    Ranking among finite entries keeps a sparse channel's top claim comparable
    to a dense channel's top claim; an abstention is never evidence.
    """
    sc = np.asarray(sc, dtype=np.float64)
    p = np.zeros(sc.shape[0], dtype=np.float64)
    fin = np.isfinite(sc)
    n = int(fin.sum())
    if n:
        p[fin] = rankdata(sc[fin], method="average") / n
    return p


def fuse_max(percentiles: list[np.ndarray]) -> np.ndarray:
    """Main rule: noisy-OR — max of channel percentiles (no channel leads)."""
    return np.max(np.stack(percentiles), axis=0)


def fuse_mean(percentiles: list[np.ndarray]) -> np.ndarray:
    """Ablation: mean of channel percentiles (consensus; dilution expected)."""
    return np.mean(np.stack(percentiles), axis=0)


def fuse_rrf(scores: list[np.ndarray], k: float = 60.0) -> np.ndarray:
    """Ablation: reciprocal-rank fusion sum_i 1/(k + rank_i); abstain contributes 0.

    rank_i is the 1-based descending rank among the channel's FINITE entries.
    """
    out = np.zeros(scores[0].shape[0], dtype=np.float64)
    for sc in scores:
        sc = np.asarray(sc, dtype=np.float64)
        fin = np.isfinite(sc)
        n = int(fin.sum())
        if not n:
            continue
        # descending rank: highest score -> rank 1
        r = n + 1 - rankdata(sc[fin], method="average")
        out[fin] += 1.0 / (k + r)
    return out


# ---------------------------------------------------------------- pair self-selection
def after_fg_pairs(
    Zf_after: np.ndarray,
    Zb_s: np.ndarray,
    n_clusters: int = 12,
    seed: int = 7,
    min_members: int = 60,
    min_fg_lens: int = 20,
    min_bg_lens: int = 15,
    lens_quantile: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Cluster the AFTER foreground; score every cluster pair by band-emptiness.

    Emptiness(a, b) = Mann-Whitney AUC  P(|t - 0.5|_fg > |t - 0.5|_bg)  within the
    pair's locality lens (dA + dB <= lens_quantile of member dA+dB), where t is the
    projection onto the centroid axis (t=0.5 = the dA=dB bisector). A removed
    corridor empties the fg band while bg keeps living there, pushing the fg
    |t-0.5| distribution above the bg one (AUC -> high); a natural well-bridged
    pair interleaves (AUC ~ 0.5). fg side = ALL after-fg docs in the lens (robust
    to cluster-assignment differences), bg side = scoped bg in the lens.

    Returns (labels, centroids, pairs sorted by emptiness desc). Each pair dict:
    {a, b, emptiness, n_members, n_fg_lens, n_bg_lens, lens_r, centroid_dist}.
    """
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4).fit(Zf_after)
    cents = km.cluster_centers_
    labels = km.labels_
    pairs: list[dict] = []
    for a in range(n_clusters):
        for b in range(a + 1, n_clusters):
            mem = np.where((labels == a) | (labels == b))[0]
            if mem.size < min_members:
                continue
            cA, cB = cents[a], cents[b]
            dA_m = np.linalg.norm(Zf_after[mem] - cA, axis=1)
            dB_m = np.linalg.norm(Zf_after[mem] - cB, axis=1)
            lens_r = float(np.quantile(dA_m + dB_m, lens_quantile))
            dA_f = np.linalg.norm(Zf_after - cA, axis=1)
            dB_f = np.linalg.norm(Zf_after - cB, axis=1)
            fg_lens = np.where(dA_f + dB_f <= lens_r)[0]
            dA_b = np.linalg.norm(Zb_s - cA, axis=1)
            dB_b = np.linalg.norm(Zb_s - cB, axis=1)
            bg_lens = np.where(dA_b + dB_b <= lens_r)[0]
            if fg_lens.size < min_fg_lens or bg_lens.size < min_bg_lens:
                continue
            s_fg = np.abs(_axis_t(Zf_after[fg_lens], cA, cB) - 0.5)
            s_bg = np.abs(_axis_t(Zb_s[bg_lens], cA, cB) - 0.5)
            u = mannwhitneyu(s_fg, s_bg, alternative="greater").statistic
            pairs.append(
                {"a": int(a), "b": int(b),
                 "emptiness": float(u / (s_fg.size * s_bg.size)),
                 "n_members": int(mem.size), "n_fg_lens": int(fg_lens.size),
                 "n_bg_lens": int(bg_lens.size), "lens_r": lens_r,
                 "centroid_dist": float(np.linalg.norm(cA - cB))}
            )
    pairs.sort(key=lambda p: p["emptiness"], reverse=True)
    return labels, cents, pairs


def voltage_pair_field(
    Zf_after: np.ndarray,
    Zb_s: np.ndarray,
    labels: np.ndarray,
    cents: np.ndarray,
    pair: dict,
    k_graph: int = 10,
    max_nodes: int = 6000,
) -> tuple[np.ndarray, dict] | None:
    """Banked single-pair voltage recipe, emitted over the full scoped-bg universe.

    Subgraph = scoped bg with dA + dB <= 1.1 * lens_r (capped at max_nodes);
    extended electrodes = bg within the median flank-member distance of each
    centroid; per-node score = mid-potential-weighted current as a SHARE of the
    eligible total (comparable across pairs), -inf outside the subgraph and on
    the clamps. Returns None when the electrodes degenerate.
    """
    from artifacts_scripts.gap_detection.detectors import voltage_current_scores

    a, b = pair["a"], pair["b"]
    cA, cB = cents[a], cents[b]
    dA = np.linalg.norm(Zb_s - cA, axis=1)
    dB = np.linalg.norm(Zb_s - cB, axis=1)
    nodes = np.where(dA + dB <= pair["lens_r"] * 1.1)[0]
    if nodes.size > max_nodes:
        nodes = nodes[np.argsort((dA + dB)[nodes])[:max_nodes]]
    fg_a = Zf_after[labels == a]
    fg_b = Zf_after[labels == b]
    if fg_a.shape[0] < 5 or fg_b.shape[0] < 5:
        return None
    r_a = float(np.median(np.linalg.norm(fg_a - cA, axis=1)))
    r_b = float(np.median(np.linalg.norm(fg_b - cB, axis=1)))
    clamp0 = np.where(dA[nodes] <= r_a)[0]
    clamp1 = np.where(dB[nodes] <= r_b)[0]
    if clamp0.size < 5:
        clamp0 = np.argsort(dA[nodes])[:20]
    if clamp1.size < 5:
        clamp1 = np.argsort(dB[nodes])[:20]
    clamp1 = np.setdiff1d(clamp1, np.intersect1d(clamp0, clamp1))
    if clamp1.size < 5:
        return None
    cur = voltage_current_scores(Zb_s[nodes], clamp0, clamp1, k_graph=k_graph)
    eligible = np.ones(nodes.size, dtype=bool)
    eligible[clamp0] = eligible[clamp1] = False
    tot = float(cur[eligible].sum())
    share = cur / (tot + EPS)
    out = np.full(Zb_s.shape[0], -np.inf)
    out[nodes[eligible]] = share[eligible]
    diag = {"n_nodes": int(nodes.size), "n_clamped": int(clamp0.size + clamp1.size),
            "pair": (int(a), int(b)), "emptiness": pair.get("emptiness")}
    return out, diag
