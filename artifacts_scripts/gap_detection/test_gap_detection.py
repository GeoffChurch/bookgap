"""Unit/integration tests on 2D toy Gaussian fixtures.

Every detector must recover a hand-made removal; the ablation contracts
(habitability term, variance normalization) are pinned down exactly.

Run:  uv run pytest artifacts_scripts/gap_detection/test_gap_detection.py -q
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import scipy.sparse as sp

from artifacts_scripts.gap_detection.detectors import (
    GapScorer,
    auc_vs_controls,
    closed_filter,
    effective_number,
    fightin_words_rank,
    iou,
    knn_radii,
    level_set_components,
    make_probes,
    mine_frequent_itemsets,
    mutual_knn_graph,
    precision_at,
    rank_of,
    topk_indices,
    voltage_current_scores,
)
from artifacts_scripts.gap_detection.synth import (
    corridor_pairs,
    plant_concept,
    plant_corridor,
    plant_radial,
    concept_candidates,
)
from artifacts_scripts.gap_detection.calibration import (
    empirical_p,
    stratified_p,
)
from artifacts_scripts.gap_detection.fusion import fused_scores, veto_mask
from artifacts_scripts.gap_detection.polish import (
    attributable_best_rank,
    attribution_stats,
    bipartite_graph,
    centroid_adjacency,
    cluster_vocab,
    commute_embedding,
    component_recovery,
    cross_conjunctions,
    doc_pattern_scores,
    flow_share,
    gaussian_knn_graph,
    itemset_support_matrix,
    low_df_tail,
    mine_tail_family,
    order_components_by_mass,
    pocket_status,
    rank_family_fast,
    residual_scores,
    voltage_on_graph,
)
from artifacts_scripts.gap_detection.walk_geometry import walk_embedding


# ------------------------------------------------------------------ fixtures
def toy_clouds(seed: int = 0, n_fg: int = 1500, n_bg: int = 5000):
    """Foreground gaussian inside a broader background gaussian (2D)."""
    rng = np.random.default_rng(seed)
    Zf = rng.normal(0.0, 1.0, size=(n_fg, 2)).astype(np.float32)
    Zb = rng.normal(0.0, 1.8, size=(n_bg, 2)).astype(np.float32)
    return Zf, Zb, rng


def planted_radial_setup(seed: int = 0):
    Zf, Zb, rng = toy_clouds(seed)
    gap = plant_radial(Zf, Zb, rng)
    Zf_pruned = np.delete(Zf, gap.removed_fg, axis=0)
    return Zf, Zb, Zf_pruned, gap


# ------------------------------------------------------------------ core statistic
def test_knn_ratio_recovers_radial_hole():
    _, Zb, Zf_pruned, gap = planted_radial_setup(0)
    scorer = GapScorer(bg=Zb, k=10)
    scorer.set_r0_from_fg(Zf_pruned)
    sc = scorer.score(Zf_pruned)["full"]
    member = np.zeros(Zb.shape[0], dtype=bool)
    member[gap.region_bg] = True
    eligible = np.ones(Zb.shape[0], dtype=bool)
    k_eval = min(20, gap.region_bg.size)
    assert rank_of(sc, member, eligible) <= 10
    assert precision_at(sc, member, eligible, k_eval) >= 0.3


def test_radial_auc_vs_matched_controls():
    _, Zb, Zf_pruned, gap = planted_radial_setup(1)
    scorer = GapScorer(bg=Zb, k=10)
    sc = scorer.score(Zf_pruned)["full"]
    neg = np.concatenate(gap.controls_bg)
    assert auc_vs_controls(sc, gap.region_bg, neg) >= 0.8


def test_habitability_suppresses_offcloud_probes():
    _, Zb, Zf_pruned, gap = planted_radial_setup(2)
    rng = np.random.default_rng(7)
    probes = make_probes(np.vstack([Zf_pruned, Zb]), 50, rng, scale_range=(2.0, 3.0))
    scorer = GapScorer(bg=Zb, probes=probes, k=10)
    out = scorer.score(Zf_pruned)
    probe_mask = ~scorer.real_mask
    eligible = np.ones(scorer.cand.shape[0], dtype=bool)
    rank_full = rank_of(out["full"], probe_mask, eligible)
    rank_dist = rank_of(out["no_hab"], probe_mask, eligible)
    # distance-to-fg-only puts an off-cloud probe near the very top;
    # the habitability factor pushes all probes far down the ranking.
    assert rank_dist <= 5
    assert rank_full > 20 * rank_dist
    # and probes sit outside the habitable hull
    assert not scorer.habitable_mask(0.9)[probe_mask].any()


def test_variance_normalization_exact_contract():
    """Equal density ratio, different local evidence => z prefers the evidenced one."""
    ring = np.stack(
        [0.4 * np.cos(np.linspace(0, 2 * np.pi, 30, endpoint=False)),
         0.4 * np.sin(np.linspace(0, 2 * np.pi, 30, endpoint=False))], axis=1
    )
    bg = np.vstack(
        [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], ring,          # candidate A: 30 extra nbrs
         [100.0, 0.0], [100.1, 0.0], [99.9, 0.0]]            # candidate B: bare
    ).astype(np.float64)
    fg = np.array([[0.0, 1.0], [0.0, -1.0], [100.0, 1.0], [100.0, -1.0]])
    scorer = GapScorer(bg=bg, k=2, r0=0.5)
    out = scorer.score(fg)
    iA, iB = 0, 33
    assert abs(out["delta_only"][iA] - out["delta_only"][iB]) < 1e-5
    assert out["z_only"][iA] > out["z_only"][iB] + 0.05
    assert out["full"][iA] > out["full"][iB]


def test_variance_normalization_prefers_evidenced_hole():
    """Two symmetric holes; the one in dense bg outranks the one in sparse bg under z."""
    rng = np.random.default_rng(3)
    fg = np.vstack(
        [rng.normal([-3, 0], 0.9, size=(900, 2)), rng.normal([3, 0], 0.9, size=(900, 2))]
    ).astype(np.float32)
    bg = np.vstack(
        [rng.normal([-3, 0], 1.1, size=(2400, 2)), rng.normal([3, 0], 1.1, size=(120, 2))]
    ).astype(np.float32)
    holes = {"dense": np.array([-3.0, 0.0]), "sparse": np.array([3.0, 0.0])}
    r = 0.45
    keep = np.ones(fg.shape[0], dtype=bool)
    for c in holes.values():
        keep &= np.linalg.norm(fg - c, axis=1) > r
    scorer = GapScorer(bg=bg, k=10)
    out = scorer.score(fg[keep])
    in_dense = np.linalg.norm(bg - holes["dense"], axis=1) <= r
    in_sparse = np.linalg.norm(bg - holes["sparse"], axis=1) <= r
    assert in_dense.sum() >= 10 and in_sparse.sum() >= 3
    adv_z = np.mean(out["z_only"][in_dense]) - np.mean(out["z_only"][in_sparse])
    adv_d = np.mean(out["delta_only"][in_dense]) - np.mean(out["delta_only"][in_sparse])
    # variance normalization shifts preference toward the well-evidenced hole
    assert adv_z > adv_d


# ------------------------------------------------------------------ level sets
def test_level_set_components_recover_hole():
    _, Zb, Zf_pruned, gap = planted_radial_setup(4)
    scorer = GapScorer(bg=Zb, k=10)
    sc = scorer.score(Zf_pruned)["full"]
    comps = level_set_components(sc, scorer.cand, scorer.real_mask, top_frac=0.05)
    assert comps, "no components found"
    # recovery criterion (matches run_eval): a top-3 component overlaps the planted
    # region with IoU >= 0.1 (IoU is structurally capped by the top-frac budget vs
    # region size; components landing elsewhere score 0.0)
    best = max(iou(c["members"], gap.region_bg) for c in comps[:3])
    assert best >= 0.1


def test_effective_number_bounds():
    assert abs(effective_number(np.ones(10)) - 10.0) < 1e-6
    w = np.zeros(10)
    w[0] = 5.0
    assert effective_number(w) < 1.5
    assert effective_number(np.random.default_rng(0).uniform(size=50)) <= 50.0


def test_mutual_knn_graph_symmetric():
    X = np.random.default_rng(0).normal(size=(200, 2))
    A = mutual_knn_graph(X, k=5)
    assert (A != A.T).nnz == 0


# ------------------------------------------------------------------ pattern mining
def test_apriori_matches_bruteforce():
    rng = np.random.default_rng(5)
    K = rng.uniform(size=(40, 7)) < 0.4
    fam = mine_frequent_itemsets(K, min_support=4, max_len=3)
    from itertools import combinations

    for L in (1, 2, 3):
        for cols in combinations(range(7), L):
            sup = int(K[:, list(cols)].all(axis=1).sum())
            if sup >= 4:
                assert fam[frozenset(cols)] == sup, cols
            else:
                assert frozenset(cols) not in fam


def test_closed_filter_drops_nonclosed():
    fam = {
        frozenset({1, 2}): 10,
        frozenset({1, 2, 3}): 10,   # same support => {1,2} not closed
        frozenset({4, 5}): 8,
        frozenset({4, 5, 6}): 5,    # smaller support => {4,5} stays
    }
    out = closed_filter(fam)
    assert frozenset({1, 2}) not in out
    assert frozenset({1, 2, 3}) in out
    assert frozenset({4, 5}) in out


def test_emerging_pattern_finds_removed_pair():
    rng = np.random.default_rng(6)
    n_bg, n_fg, m = 400, 250, 12
    Kb = rng.uniform(size=(n_bg, m)) < 0.25
    Kf = rng.uniform(size=(n_fg, m)) < 0.25
    Kb[:70, 2] = True
    Kb[:70, 5] = True            # planted: frequent in bg
    both = Kf[:, 2] & Kf[:, 5]
    Kf[both, 5] = False          # removed from fg entirely
    fam = closed_filter(mine_frequent_itemsets(Kb, min_support=15, max_len=3))
    ranked = fightin_words_rank(fam, Kf, n_bg=n_bg)
    pos = next(i for i, r in enumerate(ranked) if r["itemset"] == frozenset({2, 5}))
    assert pos == 0, f"planted pair ranked {pos + 1}"
    assert ranked[pos]["fg_support"] == 0


# ------------------------------------------------------------------ voltage bridging
def test_voltage_corridor_concentrates_current():
    rng = np.random.default_rng(8)
    bg = rng.uniform([-4, -2.2], [4, 2.2], size=(1800, 2)).astype(np.float32)
    cA, cB = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    # extended electrodes: clamp every bg node inside each cluster ball; point-like
    # terminals create a constriction halo that drowns the corridor signal.
    clamp0 = np.where(np.linalg.norm(bg - cA, axis=1) <= 0.8)[0]
    clamp1 = np.where(np.linalg.norm(bg - cB, axis=1) <= 0.8)[0]
    cur = voltage_current_scores(bg, clamp0, clamp1, k_graph=8)
    eligible = np.ones(bg.shape[0], dtype=bool)
    eligible[clamp0] = eligible[clamp1] = False
    corridor = (np.abs(bg[:, 0]) <= 1.2) & (np.abs(bg[:, 1]) <= 0.9)
    top = topk_indices(cur, eligible, 30)
    assert corridor[top].mean() >= 0.6


# ------------------------------------------------------------------ planting invariants
def test_plant_radial_bounds():
    Zf, Zb, rng = toy_clouds(9)
    gap = plant_radial(Zf, Zb, rng)
    assert 30 <= gap.removed_fg.size <= 150
    assert gap.region_bg.size >= 8          # orphaned core
    assert gap.region_bg.size <= gap.meta["n_bg_in_ball"]
    assert len(gap.controls_bg) >= 2
    # removed fg really are inside the ball; core bg really are inside the ball
    assert (np.linalg.norm(Zf[gap.removed_fg] - gap.center, axis=1) <= gap.radius + 1e-6).all()
    assert (np.linalg.norm(Zb[gap.region_bg] - gap.center, axis=1) <= gap.radius + 1e-6).all()


def test_plant_concept_bounds():
    rng = np.random.default_rng(10)
    Kf = sp.csr_matrix((rng.uniform(size=(800, 30)) < 0.3).astype(np.uint8))
    Kb = sp.csr_matrix((rng.uniform(size=(2000, 30)) < 0.3).astype(np.uint8))
    scoped_idx = np.arange(2000)
    vocab = np.asarray([f"kw{i}" for i in range(30)], dtype=object)
    cands = concept_candidates(Kf, Kb, fg_range=(30, 150), min_bg_scoped=30, rng=rng)
    assert cands, "no candidate combos in toy data"
    gap = plant_concept(Kf, Kb, scoped_idx, vocab, rng, cands)
    assert 30 <= gap.meta["fg_support"] <= 150
    assert gap.meta["scoped_bg_support"] >= 30
    assert gap.removed_fg.size == gap.meta["fg_support"]
    cols = list(gap.meta["combo_cols"])
    assert np.asarray(Kf[gap.removed_fg][:, cols].todense()).all()


def test_plant_corridor_bounds():
    rng = np.random.default_rng(11)
    Zf = np.vstack(
        [rng.normal([-2, 0], 0.85, size=(1000, 2)), rng.normal([2, 0], 0.85, size=(1000, 2)),
         rng.normal([0, 2.5], 0.85, size=(1000, 2))]
    ).astype(np.float32)
    Zb = rng.uniform([-4.5, -2.5], [4.5, 5.0], size=(6000, 2)).astype(np.float32)
    labels, cents, pairs = corridor_pairs(Zf, Zb, n_clusters=6, seed=3)
    assert pairs, "no corridor pairs found"
    gap = None
    for pair in pairs:
        gap = plant_corridor(Zf, Zb, labels, cents, pair, pairs, rng)
        if gap is not None:
            break
    assert gap is not None, "no pair produced a detectable corridor core"
    assert 30 <= gap.removed_fg.size <= 150
    assert gap.region_bg.size >= 8          # orphaned core
    # removed fg really sit in the equidistance band (dA ~ dB)
    cA, cB = cents[gap.meta["a"]], cents[gap.meta["b"]]
    dA = np.linalg.norm(Zf[gap.removed_fg] - cA, axis=1)
    dB = np.linalg.norm(Zf[gap.removed_fg] - cB, axis=1)
    rho = dA / (dA + dB)
    assert (np.abs(rho - 0.5) <= 0.45).all()
    assert np.median(np.abs(rho - 0.5)) <= 0.15


# ------------------------------------------------------------------ metric helpers
def test_rank_and_precision_helpers():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    member = np.array([False, False, True, False])
    eligible = np.ones(4, dtype=bool)
    assert rank_of(scores, member, eligible) == 3
    assert precision_at(scores, member, eligible, 2) == 0.0
    assert precision_at(scores, member, eligible, 3) == pytest.approx(1 / 3)
    eligible[1] = False  # excluding the top changes ranks
    assert rank_of(scores, member, eligible) == 2


def test_auc_extremes():
    sc = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    assert auc_vs_controls(sc, np.array([3, 4, 5]), np.array([0, 1, 2])) == 1.0
    assert auc_vs_controls(sc, np.array([0, 1, 2]), np.array([3, 4, 5])) == 0.0
    sc2 = np.ones(6)
    assert abs(auc_vs_controls(sc2, np.array([0, 1, 2]), np.array([3, 4, 5])) - 0.5) < 1e-9


def test_knn_radii_self_exclusion():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    r = knn_radii(X, X, 1, ref_is_query=True)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(1.0)
    assert r[2] == pytest.approx(2.0)
    r2 = knn_radii(np.array([[0.5, 0.0]]), X, 1)
    assert r2[0] == pytest.approx(0.5)


# ------------------------------------------------------------------ task A: fusion veto
def fusion_fixture():
    """fg + field bg + a dense FAR cluster (habitable, off-field) + off-hull probes."""
    rng = np.random.default_rng(0)
    Zf = rng.normal(0.0, 1.0, size=(1500, 2)).astype(np.float32)
    bg_field = rng.normal(0.0, 1.8, size=(4000, 2)).astype(np.float32)
    bg_far = (rng.normal(0.0, 0.5, size=(600, 2)) + np.array([40.0, 0.0])).astype(np.float32)
    Zb = np.vstack([bg_field, bg_far])
    gap = plant_radial(Zf, Zb, rng)
    fgp = np.delete(Zf, gap.removed_fg, axis=0)
    probes = make_probes(np.vstack([Zf, Zb]), 50, rng, scale_range=(2.0, 3.0))
    scorer = GapScorer(bg=Zb, probes=probes, k=10)
    scorer.set_r0_from_fg(Zf)
    out = scorer.score(fgp)
    return scorer, out, gap, np.arange(4000, 4600)


def test_veto_contract_offhull_and_offfield():
    scorer, out, gap, far_rows = fusion_fixture()
    veto = veto_mask(scorer, out)
    # off-hull probes: all vetoed (the g gate)
    assert veto[~scorer.real_mask].all()
    # dense far cluster: HABITABLE (passes the hull) yet vetoed by the Delta ceiling
    assert scorer.habitable_mask(0.9)[far_rows].mean() > 0.9
    assert veto[far_rows].all()
    # in-cloud planted gap core: overwhelmingly NOT vetoed. (Only the deepest dent
    # points of a 2D hole can exceed the ceiling — documented failure mode; the
    # 128-d field cores sit at Delta q99 ~ 0.80 << log 3.)
    assert veto[gap.region_bg].mean() <= 0.15
    assert float(np.median(out["delta_only"][gap.region_bg])) < float(np.log(3.0))


def test_fused_scores_preserve_raw_order_and_kill_probes():
    scorer, out, gap, _ = fusion_fixture()
    veto = veto_mask(scorer, out)
    fz = fused_scores(out, veto)
    keep = ~veto
    # the veto must never REWEIGHT: ordering among survivors == raw r_F order
    assert (np.argsort(fz[keep]) == np.argsort(out["no_hab"][keep].astype(np.float64))).all()
    every = np.ones(scorer.cand.shape[0], dtype=bool)
    top = topk_indices(fz, every, 50)
    assert scorer.real_mask[top].all()  # no probe in the fused top-50
    assert not veto[top].any()          # vetoed candidates cannot surface


# ------------------------------------------------------------------ task B: calibration
def test_empirical_p_uniform_and_extremes():
    rng = np.random.default_rng(0)
    null = rng.normal(size=1000)
    cands = rng.normal(size=500)  # drawn from the null distribution
    p = empirical_p(cands, null)
    assert abs(float(p.mean()) - 0.5) < 0.05
    grid = np.sort(p)
    assert float(np.max(np.abs(grid - np.linspace(0, 1, grid.size)))) < 0.08  # ~KS
    # extreme candidate gets the minimal p; hopeless candidate gets p ~ 1
    assert empirical_p(np.array([null.max() + 1.0]), null)[0] == pytest.approx(1 / 1001)
    assert empirical_p(np.array([null.min() - 1.0]), null)[0] == pytest.approx(1.0)


def test_stratified_p_respects_strata():
    """Same raw value: significant in the tight stratum, ordinary in the wide one."""
    rng = np.random.default_rng(1)
    n = 400
    tight = rng.normal(0.0, 0.1, size=n)
    wide = rng.normal(0.0, 10.0, size=n)
    stats = np.concatenate([tight, wide, [1.0, 1.0]])
    strata = np.concatenate([np.zeros(n, int), np.ones(n, int), [0, 1]])
    null_idx = np.arange(2 * n)
    p = stratified_p(stats, strata, null_idx, n_strata=2)
    assert p[-2] < 0.02
    assert p[-1] > 0.2
    assert p[-2] < p[-1]


def test_stratified_p_pooled_fallback():
    rng = np.random.default_rng(2)
    stats = rng.normal(size=100)
    strata = np.zeros(100, dtype=int)
    strata[50:] = 1
    null_idx = np.arange(30)  # all nulls live in stratum 0
    p = stratified_p(stats, strata, null_idx, n_strata=2, min_null=15)
    pooled = empirical_p(stats, stats[null_idx])
    assert np.allclose(p[50:], pooled[50:])  # empty stratum falls back to pooled


# ------------------------------------------------------------------ task C: walk geometry
def test_walk_embedding_shapes_determinism_isolated():
    rng = np.random.default_rng(3)
    K = (rng.uniform(size=(300, 50)) < 0.2).astype(np.uint8)
    K[5] = 0  # isolated doc (no keywords)
    W1, m1 = walk_embedding(sp.csr_matrix(K), dim=16, seed=0)
    W2, _ = walk_embedding(sp.csr_matrix(K), dim=16, seed=0)
    assert W1.shape == (300, 16)
    assert np.array_equal(W1, W2)  # ARPACK seeded => deterministic
    assert m1["n_isolated_docs"] == 1
    assert np.allclose(W1[5], 0.0)
    assert m1["top_singular_values"][0] <= 1.0 + 1e-6  # stochastic operator bound
    assert (W1.std(axis=0) > 0).all()


def _concept_bipartite_fixture():
    """Equal-degree docs; concept docs swap 2 common keywords for 2 of 3 special ones.

    Any two concept docs share >= 1 special keyword (2-subsets of a 3-set intersect),
    and special keywords are rare (df ~ 80 vs ~ 460), so the walk geometry binds the
    concept group tightly while raw count geometry barely sees the 2-bit difference.
    """
    rng = np.random.default_rng(1)
    n_common, n_special, deg = 40, 3, 13

    def make_docs(n, special):
        K = np.zeros((n, n_common + n_special), dtype=np.uint8)
        for i in range(n):
            k = deg - 2 if special else deg
            K[i, rng.choice(n_common, size=k, replace=False)] = 1
            if special:
                K[i, n_common + rng.choice(n_special, size=2, replace=False)] = 1
        return K

    Kf = np.vstack([make_docs(700, False), make_docs(60, True)])
    Kb = np.vstack([make_docs(700, False), make_docs(60, True)])
    return Kf, Kb, np.arange(700, 760), np.arange(700, 760)


# ------------------------------------------------------------------ P1: miner cross-species
def _miner_attrib_fixture():
    """bg lattice with a frequent pair {2,5}; fg supports it via a doc subset."""
    rng = np.random.default_rng(6)
    n_bg, n_fg, m = 400, 250, 12
    Kb = rng.uniform(size=(n_bg, m)) < 0.25
    Kf = rng.uniform(size=(n_fg, m)) < 0.25
    Kb[:70, 2] = True
    Kb[:70, 5] = True
    Kf[:30, 2] = True
    Kf[:30, 5] = True            # fg supporters in the front block...
    both = Kf[:, 2] & Kf[:, 5]
    both[:30] = False
    Kf[both, 5] = False          # ...and nowhere else (exact wipe possible)
    fam = closed_filter(mine_frequent_itemsets(Kb, min_support=15, max_len=3))
    keys = list(fam.keys())
    key_index = {S: i for i, S in enumerate(keys)}
    sup_bg = np.array([fam[S] for S in keys])
    M_fg = itemset_support_matrix(keys, Kf)
    M_bg = itemset_support_matrix(keys, Kb).tocsc()
    return Kb, Kf, fam, keys, key_index, sup_bg, M_fg, M_bg


def test_rank_family_fast_matches_fightin_words():
    Kb, Kf, fam, keys, key_index, sup_bg, M_fg, _ = _miner_attrib_fixture()
    keep = np.ones(Kf.shape[0], dtype=bool)
    keep[:30] = False  # remove the supporters
    ref = fightin_words_rank(fam, Kf[keep], n_bg=Kb.shape[0])
    fast = rank_family_fast(keys, sup_bg, M_fg, keep, n_bg=Kb.shape[0])
    assert len(ref) == len(fast)
    for a, b in zip(ref, fast):
        assert a["itemset"] == b["itemset"]
        assert a["z"] == pytest.approx(b["z"])
        assert (a["bg_support"], a["fg_support"]) == (b["bg_support"], b["fg_support"])


def test_attribution_finds_wiped_pattern_not_random_removal():
    Kb, Kf, fam, keys, key_index, sup_bg, M_fg, _ = _miner_attrib_fixture()
    removed = np.zeros(Kf.shape[0], dtype=bool)
    removed[:30] = True  # exactly the {2,5} fg supporters
    n_pre, n_rem = attribution_stats(M_fg, removed)
    ranked = rank_family_fast(keys, sup_bg, M_fg, ~removed, n_bg=Kb.shape[0])
    pos, row = attributable_best_rank(ranked, key_index, n_pre, n_rem,
                                      min_pre=10, ratio=0.8)
    assert pos == 1 and frozenset({2, 5}) <= row["itemset"]
    # a random removal of the same size leaves no >=80%-wiped frequent pattern
    rng = np.random.default_rng(0)
    rand = np.zeros(Kf.shape[0], dtype=bool)
    rand[rng.choice(Kf.shape[0], size=30, replace=False)] = True
    n_pre2, n_rem2 = attribution_stats(M_fg, rand)
    ranked2 = rank_family_fast(keys, sup_bg, M_fg, ~rand, n_bg=Kb.shape[0])
    pos2, _ = attributable_best_rank(ranked2, key_index, n_pre2, n_rem2,
                                     min_pre=10, ratio=0.8)
    assert pos2 is None


def test_doc_pattern_scores_max_and_sum():
    Kb, Kf, fam, keys, key_index, sup_bg, M_fg, M_bg = _miner_attrib_fixture()
    removed = np.zeros(Kf.shape[0], dtype=bool)
    removed[:30] = True
    ranked = rank_family_fast(keys, sup_bg, M_fg, ~removed, n_bg=Kb.shape[0])
    sc = doc_pattern_scores(ranked, key_index, M_bg, agg="max")
    top_z = ranked[0]["z"]
    supporters = Kb[:, 2] & Kb[:, 5]
    # every supporter of the top (wiped) pattern carries at least its z
    assert (sc[supporters] >= top_z - 1e-9).all()
    assert sc[supporters].mean() > sc[~supporters].mean()
    sc_sum = doc_pattern_scores(ranked, key_index, M_bg, agg="sum")
    assert sc_sum[supporters].mean() > sc_sum[~supporters].mean()


def test_cluster_vocab_and_cross_conjunctions():
    rng = np.random.default_rng(7)
    n, m = 600, 20
    Kf = rng.uniform(size=(n, m)) < 0.15
    members = np.zeros(n, dtype=bool)
    members[:200] = True
    Kf[:200, 3] = rng.uniform(size=200) < 0.8     # cluster-distinctive keyword
    Kf[200:, 3] = rng.uniform(size=400) < 0.05
    voc = cluster_vocab(Kf, members, n_top=5)
    assert 3 in voc.tolist()
    Kb = rng.uniform(size=(800, m)) < 0.15
    Kb[:120, 3] = True
    Kb[:120, 7] = True  # strong cross conjunction {3,7}
    conj = cross_conjunctions(np.array([3, 1]), np.array([7, 2]), Kb,
                              min_support=50, cap=10)
    assert frozenset({3, 7}) in conj
    assert conj[frozenset({3, 7})] >= 120
    for S, s in conj.items():
        assert s >= 50 and len(S) == 2


# ------------------------------------------------------------------ P2: residualization
def test_residual_recovers_plant_and_drops_standing_pocket():
    """Absolute scores surface the permanent fg-free pocket; differential scores
    surface the planted removal and drop the pocket."""
    rng = np.random.default_rng(12)
    Zf = rng.normal(0.0, 1.0, size=(1500, 2)).astype(np.float32)
    bg_field = rng.normal(0.0, 1.8, size=(4000, 2)).astype(np.float32)
    pocket_pts = (rng.normal(0.0, 0.35, size=(250, 2)) + np.array([6.0, 0.0])).astype(
        np.float32
    )
    Zb = np.vstack([bg_field, pocket_pts])
    pocket = np.arange(4000, 4250)
    gap = plant_radial(Zf, Zb, rng)
    fgp = np.delete(Zf, gap.removed_fg, axis=0)
    scorer = GapScorer(bg=Zb, k=10)
    scorer.set_r0_from_fg(Zf)
    before = scorer.score(Zf)
    after = scorer.score(fgp)
    d = residual_scores(before, after, "full")
    comps_abs = level_set_components(after["full"], scorer.cand, scorer.real_mask)
    comps_d = level_set_components(d, scorer.cand, scorer.real_mask)
    # absolute: the standing pocket dominates the top components
    st_abs = pocket_status(comps_abs, after["full"], pocket, scorer.real_mask)
    assert st_abs["rank"] == 1
    # differential: plant recovered, pocket gone from the selection
    rec = component_recovery(comps_d, gap.region_bg)
    assert rec["recall_top3"]
    st_d = pocket_status(comps_d, d, pocket, scorer.real_mask)
    assert st_d["rank"] is None
    assert st_d["members_in_topfrac"] <= 5
    # mass ordering is a valid alternative ordering of the same components
    by_mass = order_components_by_mass(comps_d, d)
    assert component_recovery(by_mass, gap.region_bg)["rank"] is not None


# ------------------------------------------------------------------ P3a: voltage field
def test_voltage_on_graph_matches_wrapped_detector():
    rng = np.random.default_rng(8)
    bg = rng.uniform([-4, -2.2], [4, 2.2], size=(900, 2)).astype(np.float32)
    cA, cB = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    clamp0 = np.where(np.linalg.norm(bg - cA, axis=1) <= 0.8)[0]
    clamp1 = np.where(np.linalg.norm(bg - cB, axis=1) <= 0.8)[0]
    ref = voltage_current_scores(bg, clamp0, clamp1, k_graph=8)
    got = voltage_on_graph(gaussian_knn_graph(bg, k=8), clamp0, clamp1)
    assert np.allclose(ref, got, rtol=1e-6, atol=1e-12)


def test_centroid_adjacency_chain():
    cents = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    pairs = centroid_adjacency(cents, n_neighbors=1)
    assert (0, 1) in pairs and (1, 2) in pairs
    assert all(a < b for a, b in pairs)
    # mutual-union with k=1: the far cluster attaches to its nearest neighbor only
    assert (2, 3) in pairs and (0, 3) not in pairs


def test_allpairs_change_field_localizes_removal():
    """Removing fg in the inter-cluster band changes the bg flow-share field
    most strongly near the band (|delta share|; the sign inverts — see
    polish.flow_share docstring — so magnitude is the contract)."""
    rng = np.random.default_rng(9)
    fgA = rng.normal([-2.0, 0.0], 0.5, size=(250, 2))
    fgB = rng.normal([2.0, 0.0], 0.5, size=(250, 2))
    band = rng.uniform([-0.7, -0.5], [0.7, 0.5], size=(120, 2))
    Zf = np.vstack([fgA, fgB, band]).astype(np.float32)
    Zb = rng.uniform([-3.5, -1.5], [3.5, 1.5], size=(1500, 2)).astype(np.float32)
    joint = np.vstack([Zf, Zb])
    n_fg = Zf.shape[0]
    cA, cB = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    c0 = np.where(np.linalg.norm(joint - cA, axis=1) <= 0.6)[0]
    c1 = np.where(np.linalg.norm(joint - cB, axis=1) <= 0.6)[0]
    cur0 = voltage_on_graph(gaussian_knn_graph(joint, k=8), c0, c1)
    keep = np.ones(joint.shape[0], dtype=bool)
    keep[500:620] = False  # remove the fg band
    remap = np.cumsum(keep) - 1
    cur1 = voltage_on_graph(
        gaussian_knn_graph(joint[keep], k=8), remap[c0[keep[c0]]], remap[c1[keep[c1]]]
    )
    bg_keep = np.where(keep)[0] >= n_fg
    s0 = flow_share(cur0[keep], bg_keep)[bg_keep]
    s1 = flow_share(cur1, bg_keep)[bg_keep]
    dmag = np.abs(s1 - s0)
    bgpts = joint[keep][bg_keep]
    in_band = (np.abs(bgpts[:, 0]) <= 0.9) & (np.abs(bgpts[:, 1]) <= 0.7)
    assert dmag[in_band].mean() > 2 * dmag[~in_band].mean()
    top = np.argsort(dmag)[::-1][:30]
    assert in_band[top].mean() >= 0.6


# ------------------------------------------------------------------ P3b: commute embedding
def test_commute_embedding_path_graph_resistance():
    """Path graph: R_eff(i,j) = |i-j| exactly; near-full rank must reproduce it."""
    n = 30
    rows = np.arange(n - 1)
    W = sp.coo_matrix(
        (np.ones(n - 1), (rows, rows + 1)), shape=(n, n)
    ).tocsr()
    W = W + W.T
    X, meta = commute_embedding(W, dim=n - 2, seed=0)
    assert meta["n_lcc"] == n and meta["n_components"] == 1
    d2 = np.sum((X[0] - X[n - 1]) ** 2)
    assert d2 == pytest.approx(n - 1, rel=0.02)
    d2_mid = np.sum((X[3] - X[10]) ** 2)
    assert d2_mid == pytest.approx(7.0, rel=0.05)
    # determinism
    X2, _ = commute_embedding(W, dim=n - 2, seed=0)
    assert np.allclose(X, X2)


def test_commute_embedding_disconnected_nodes_nan():
    n = 25
    rows = np.arange(n - 1)
    W = sp.lil_matrix((n + 2, n + 2))
    W[rows, rows + 1] = 1.0
    W[rows + 1, rows] = 1.0
    W[n, n + 1] = 1.0  # tiny separate component
    W[n + 1, n] = 1.0
    X, meta = commute_embedding(W.tocsr(), dim=10, seed=0)
    assert meta["n_lcc"] == n and meta["n_components"] == 2
    assert np.isnan(X[n]).all() and np.isnan(X[n + 1]).all()
    assert np.isfinite(X[:n]).all()


def test_bipartite_graph_shape_and_symmetry():
    K = sp.csr_matrix((np.random.default_rng(0).uniform(size=(40, 7)) < 0.3).astype(float))
    W = bipartite_graph(K)
    assert W.shape == (47, 47)
    assert (W != W.T).nnz == 0
    assert W[:40][:, :40].nnz == 0  # no doc-doc edges


# ------------------------------------------------------------------ P4: low-df tail
def test_low_df_tail_window_and_cap():
    rng = np.random.default_rng(13)
    n = 500
    df_targets = np.concatenate([np.full(30, 5), np.full(40, 20), np.full(30, 80)])
    cols = []
    for d in df_targets:
        v = np.zeros(n)
        v[rng.choice(n, size=d, replace=False)] = 1.0
        cols.append(v)
    X = sp.csr_matrix(np.stack(cols, axis=1))
    tail_cols, K_tail = low_df_tail(X, df_range=(3, 50), cap=100)
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    assert ((df[tail_cols] >= 3) & (df[tail_cols] <= 50)).all()
    assert tail_cols.size == 70
    # cap keeps the rarest
    tail_capped, _ = low_df_tail(X, df_range=(3, 50), cap=30)
    assert tail_capped.size == 30
    assert (df[tail_capped] == 5).all()
    assert K_tail.shape == (n, 70)


def test_mine_tail_family_supports_and_offsets():
    rng = np.random.default_rng(14)
    n = 300
    K_top = rng.uniform(size=(n, 6)) < 0.4
    K_tail = sp.csr_matrix((rng.uniform(size=(n, 4)) < 0.06).astype(np.uint8))
    Kt = np.asarray(K_tail.todense(), dtype=bool)
    Kt[:25, 0] = True
    Kt[:25, 1] = True            # tail-tail pair
    K_top[:25, 2] = True         # tail0 x top2 pair
    fam = mine_tail_family(K_top, sp.csr_matrix(Kt.astype(np.uint8)),
                           min_support=12, top_offset=100)
    assert fam[frozenset({100, 101})] == int((Kt[:, 0] & Kt[:, 1]).sum())
    assert frozenset({100, 2}) in fam
    assert fam[frozenset({100})] == int(Kt[:, 0].sum())
    for S, s in fam.items():
        assert s >= 12
        assert all(c >= 100 or c < 6 for c in S)
        assert any(c >= 100 for c in S)  # every itemset touches the tail


def test_concept_removal_radial_in_walk_not_raw_geometry():
    Kf, Kb, concept_fg, concept_bg = _concept_bipartite_fixture()
    n_fg = Kf.shape[0]
    W, _ = walk_embedding(sp.csr_matrix(np.vstack([Kf, Kb])), dim=16, seed=0)
    keep = np.ones(n_fg, dtype=bool)
    keep[concept_fg] = False
    neg = np.setdiff1d(np.arange(Kb.shape[0]), concept_bg)
    aucs = {}
    for name, (F, B) in {"walk": (W[:n_fg], W[n_fg:]),
                         "raw": (Kf.astype(np.float32), Kb.astype(np.float32))}.items():
        sc = GapScorer(bg=B, k=10)
        sc.set_r0_from_fg(F)
        o = sc.score(F[keep])
        aucs[name] = {v: auc_vs_controls(o[v], concept_bg, neg) for v in ("full", "no_hab")}
    # the concept removal IS a radial hole in walk geometry...
    assert aucs["walk"]["full"] >= 0.9
    assert aucs["walk"]["no_hab"] >= 0.9
    # ...and is NOT a density dent in raw count geometry
    assert aucs["raw"]["full"] <= 0.65
    assert aucs["walk"]["no_hab"] - aucs["raw"]["no_hab"] >= 0.15


# ------------------------------------------------------------------ Track 2 (wiki temporal)
def test_id_date_proxy_monotone_and_inversion():
    """Isotonic id->date proxy: monotone predictions, correct date->id inversion."""
    from artifacts_scripts.gap_detection.wiki_temporal import (
        fit_id_date_proxy,
        invert_date_to_id,
    )

    rng = np.random.default_rng(7)
    ids = np.sort(rng.choice(100_000, size=400, replace=False)).astype(np.int64)
    true_days = 0.05 * ids  # linear growth
    noisy = true_days + rng.normal(0, 30, size=ids.size)  # wrinkles
    predict = fit_id_date_proxy(ids, noisy)
    grid = np.linspace(0, 100_000, 2_000)
    pred = predict(grid)
    assert (np.diff(pred) >= -1e-9).all(), "proxy must be monotone in id"
    # inversion: smallest id whose predicted date >= target
    target = float(predict(np.array([50_000]))[0])
    t_id = invert_date_to_id(predict, target, 0, 100_001)
    assert float(predict(np.array([t_id]))[0]) >= target
    assert t_id == 0 or float(predict(np.array([t_id - 1]))[0]) < target
    # accuracy on clean data: held-out error small relative to noise scale
    err = np.abs(predict(ids) - true_days)
    assert np.median(err) < 60.0


def test_t_corpus_leakage_guard_and_frozen_transform():
    """Post-T docs can neither enter the pool nor leak tokens into the vocab."""
    from artifacts_scripts.gap_detection.t_corpus import fit_t_corpus

    rng = np.random.default_rng(11)
    # token_pattern is alphabetic-only: build letter-only terms
    base_terms = [f"term{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(200)]
    n = 400

    def make_text(r):
        words = list(r.choice(base_terms, size=15, replace=False)) * 2
        return " ".join(words)

    texts = [make_text(rng) for _ in range(n)]
    days = np.linspace(0, 1000, n)
    t_days = 600.0
    pre = days <= t_days
    kw = dict(
        seed=3, n_docs=200, fg_size_range=(5, 200), min_interleave=0.0,
        lsa_dim=8, kmeans_k=4, n_keywords=20, tfidf_min_df=2,
    )
    # (a) leakage guard trips on a post-T doc in the pool
    with pytest.raises(ValueError, match="leakage guard"):
        fit_t_corpus(
            texts, days, t_days,
            titles=np.array([f"t{i}" for i in range(n)], dtype=object),
            page_ids=np.arange(n), store_rows=np.arange(n),
            part_a_rows=np.array([], dtype=np.int64), **kw,
        )
    # (b) clean pre-T pool fits; a post-T-only token never enters the vocabulary
    pre_idx = np.where(pre)[0]
    post_idx = np.where(~pre)[0]
    post_texts = [texts[i] + " zzleakytoken zzleakytoken zzleakytoken" for i in post_idx]
    tc = fit_t_corpus(
        [texts[i] for i in pre_idx], days[pre_idx], t_days,
        titles=np.array([f"t{i}" for i in pre_idx], dtype=object),
        page_ids=pre_idx.astype(np.int64), store_rows=pre_idx.astype(np.int64),
        part_a_rows=pre_idx[:50].astype(np.int64), **kw,
    )
    assert (tc.days <= t_days).all()
    vocab_all = set(tc.vectorizer.get_feature_names_out().tolist())
    assert "zzleakytoken" not in vocab_all
    # frozen transform embeds post-T docs without refitting
    Z_arr, K_arr, nnz = tc.transform_texts(post_texts)
    assert Z_arr.shape == (len(post_texts), tc.Z.shape[1])
    assert K_arr.shape[1] == tc.K.shape[1]
    assert "zzleakytoken" not in set(tc.vectorizer.get_feature_names_out().tolist())
    norms = np.linalg.norm(Z_arr, axis=1)
    assert np.allclose(norms[nnz >= 5], 1.0, atol=1e-4)


def test_matched_controls_contract():
    """Controls: never flagged, spatially disjoint, stratum-matched when possible."""
    from artifacts_scripts.gap_detection.t2_outcomes import (
        covariate_strata,
        matched_controls,
    )

    rng = np.random.default_rng(5)
    n = 800
    X = rng.normal(0, 1, size=(n, 2)).astype(np.float32)
    radii = np.full(n, 0.05, dtype=np.float64)
    C = np.column_stack([
        np.linalg.norm(X, axis=1),            # density-ish covariate
        rng.normal(0, 1, size=n),             # generic covariate
        rng.poisson(3, size=n).astype(float), # growth-ish covariate
    ])
    flagged = rng.choice(n, size=12, replace=False)
    # global exclusion: flagged docs + anything inside a flagged neighborhood
    excluded = np.zeros(n, dtype=bool)
    excluded[flagged] = True
    for f in flagged:
        excluded |= np.linalg.norm(X - X[f], axis=1) <= radii[f]
    controls, info = matched_controls(
        X[flagged], radii[flagged], C[flagged], X, radii, C,
        excluded=excluded, m_controls=6, seed=2,
    )
    assert controls.shape == (12, 6)
    assert (controls >= 0).all()
    mu, sd = C.mean(axis=0), C.std(axis=0) + 1e-12
    cal = np.asarray(info["calipers_sigma"])
    flag_set = set(flagged.tolist())
    for i, f in enumerate(flagged):
        row = controls[i]
        assert len(set(row.tolist())) == 6, "controls unique within a location"
        assert not (set(row.tolist()) & flag_set), "controls never flagged"
        assert not excluded[row].any(), "controls outside every flagged neighborhood"
        d = np.linalg.norm(X[row] - X[f], axis=1)
        assert (d > np.maximum(radii[f], radii[row])).all(), \
            "control centers outside the flagged neighborhood and vice versa"
    # caliper containment for unrelaxed locations (all on this easy fixture)
    if info["n_relaxed"] == 0:
        for i, f in enumerate(flagged):
            diff = np.abs((C[controls[i]] - mu) / sd - (C[f] - mu) / sd)
            assert (diff <= cal + 1e-9).all(), "caliper violated at relax level 0"
    # synthetic (non-pool) locations work through the same interface
    cent = X[flagged[:3]].mean(axis=0, keepdims=True)
    c_cent = np.array([[float(np.linalg.norm(cent)), 0.0, 3.0]])
    ctr2, _ = matched_controls(cent, np.array([0.05]), c_cent, X, radii, C,
                               excluded=excluded, m_controls=6, seed=3)
    assert ctr2.shape == (1, 6) and not excluded[ctr2[0]].any()


def test_fill_rate_ratio_and_fill_counts():
    """Fill counting + ratio CI: planted enrichment is detected, null straddles 1."""
    from artifacts_scripts.gap_detection.t2_outcomes import (
        fill_counts,
        fill_rate_ratio,
        knn_scale,
    )

    rng = np.random.default_rng(9)
    locs = rng.normal(0, 1, size=(10, 2)).astype(np.float32)
    ref = rng.normal(0, 1, size=(500, 2)).astype(np.float32)
    radii = knn_scale(locs, ref, k=10)
    assert (radii > 0).all()
    # arrivals concentrated around the first 5 locations
    arr = np.vstack([
        locs[:5] + rng.normal(0, 0.01, size=(5, 2)) for _ in range(20)
    ]).astype(np.float32)
    fills = fill_counts(locs, radii, arr)
    assert fills[:5].min() >= 1
    flagged_fill = fills[:5].astype(float)
    control_fill = np.tile(fills[5:].astype(float), (5, 1))  # (5, 5) far controls
    res = fill_rate_ratio(flagged_fill, control_fill + 0.05, n_boot=200, seed=1)
    assert res["ratio"] > 1.0
    # exchangeable null: CI straddles 1
    null_f = rng.poisson(3, size=12).astype(float)
    null_c = rng.poisson(3, size=(12, 8)).astype(float)
    res0 = fill_rate_ratio(null_f, null_c, n_boot=400, seed=2)
    assert res0["ci_lo"] <= 1.0 <= res0["ci_hi"]


def test_knn_scale_self_exclusion():
    """Locations that ARE reference points use the (k+1)-th neighbor."""
    from artifacts_scripts.gap_detection.t2_outcomes import knn_scale

    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, size=(200, 2)).astype(np.float32)
    locs = ref[:20]
    self_rows = np.ones(20, dtype=bool)
    r_self = knn_scale(locs, ref, k=10, self_rows=self_rows)
    r_naive = knn_scale(locs, ref, k=10, self_rows=np.zeros(20, dtype=bool))
    # excluding self can only push the radius out (k-th non-self >= k-th incl self)
    assert (r_self >= r_naive - 1e-9).all()
    assert (r_self > r_naive).any()


def test_clustered_bootstrap_widens_for_duplicate_locations():
    """Near-duplicate locations must not tighten the CI: clustered >= naive width."""
    from artifacts_scripts.gap_detection.t2_outcomes import (
        fill_rate_ratio,
        greedy_location_groups,
    )

    rng = np.random.default_rng(4)
    # 5 well-separated spots, each repeated 10x (50 'locations', 5 groups)
    base = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0], [6.0, 0.0]])
    locs = np.repeat(base, 10, axis=0)
    radii = np.full(50, 0.3)
    groups = greedy_location_groups(locs, radii)
    assert len(np.unique(groups)) == 5
    assert (groups == np.repeat(np.arange(5), 10)).all()
    f = np.repeat(rng.poisson(8, size=5).astype(float), 10)
    c = rng.poisson(5, size=(50, 6)).astype(float)
    res = fill_rate_ratio(f, c, n_boot=500, seed=3, groups=groups)
    naive_w = res["ci_hi"] - res["ci_lo"]
    clust_w = res["ci_hi_clustered"] - res["ci_lo_clustered"]
    assert clust_w >= naive_w * 0.9  # clustered never materially tighter
    # all-singleton groups: clustered CI ~ naive CI in distribution (same machinery)
    g1 = np.arange(50)
    res1 = fill_rate_ratio(f, c, n_boot=500, seed=3, groups=g1)
    assert abs((res1["ci_hi_clustered"] - res1["ci_lo_clustered"]) - naive_w) < naive_w


# ------------------------------------------------------------------ Track 2 (OA phase)
def test_oa_short_id_and_dates():
    """W-id normalization across URL forms; date->days->iso round-trips."""
    from artifacts_scripts.gap_detection.oa_corpus import (
        date_to_days,
        days_to_iso,
        short_id,
        year_to_days,
    )

    assert short_id("https://openalex.org/W42") == "W42"
    assert short_id("openalex.org/W42") == "W42"
    assert short_id("W42") == "W42"
    assert short_id(None) is None
    assert date_to_days("2008-01-01") == 0.0
    assert date_to_days("2015-01-01") == year_to_days(2015)
    # round-trip: days -> iso -> days is the identity on whole days
    for iso in ("2008-01-01", "2012-06-15", "2019-12-31"):
        assert days_to_iso(date_to_days(iso)) == iso


def test_oa_is_english_filter():
    """Language filter: English prose passes; a French abstract is rejected (the
    ~5% non-English minority that forms a degenerate language fg cluster)."""
    from artifacts_scripts.gap_detection.oa_corpus import is_english

    en = ("we present a neural model for machine translation that uses attention "
          "over the source sentence and outperforms the baseline on the test set")
    fr = ("nous presentons un modele neuronal pour la traduction automatique qui "
          "utilise une attention sur la phrase source et ameliore les resultats")
    assert is_english(en) is True
    assert is_english(fr) is False
    assert is_english("") is True  # empty handled by the char filter, not here
    # a technical English abstract with no stopwords at all still passes (en>=non=0)
    assert is_english("transformer bert embeddings tokenizer corpus") is True


def test_oa_decode_abstract_inverted_index():
    """Inverted-index decode reconstructs token order; gaps -> spaces; robust."""
    from artifacts_scripts.gap_detection.oa_corpus import decode_abstract

    # "neural machine translation models" with a gap at position 4 (word missing)
    inv = {"neural": [0], "machine": [1], "translation": [2], "models": [3], "rare": [6]}
    out = decode_abstract(inv)
    toks = out.split(" ")
    assert toks[0:4] == ["neural", "machine", "translation", "models"]
    assert toks[6] == "rare"
    assert out != ""  # non-empty
    # empty / None -> "" (the missing-abstract case the missingness model handles)
    assert decode_abstract(None) == ""
    assert decode_abstract({}) == ""
    # duplicate position: later write wins, no crash (unlike the strict ingest resolver)
    assert decode_abstract({"a": [0], "b": [0, 1]}) in ("b b", "b b")


def test_oa_dated_graph_no_future_leakage():
    """The dated citation graph at T excludes every citing act with date > T.

    Toy universe: 4 works dated 2010/2012/2014/2016. Edges: W2->W1 (2012),
    W3->W2 (2014), W4->W3 (2016). At T=2015-01-01 only the first two edges may
    appear; the 2016 citing act (W4->W3) must NOT be present — a leak would put
    it in the T-graph."""
    from artifacts_scripts.gap_detection.oa_corpus import (
        build_dated_graph,
        graph_stats,
        year_to_days,
    )

    wid = np.array(["W1", "W2", "W3", "W4"], dtype=object)
    days = np.array([year_to_days(2010), year_to_days(2012),
                     year_to_days(2014), year_to_days(2016)], dtype=np.float64)
    row_of = {w: i for i, w in enumerate(wid)}

    # monkeypatch iter_works to stream our toy works (with referenced_works as URLs)
    import artifacts_scripts.gap_detection.oa_corpus as oc

    works = [
        {"id": "https://openalex.org/W1", "referenced_works": []},
        {"id": "https://openalex.org/W2", "referenced_works": ["https://openalex.org/W1"]},
        {"id": "https://openalex.org/W3", "referenced_works": ["https://openalex.org/W2"]},
        {"id": "https://openalex.org/W4", "referenced_works": ["https://openalex.org/W3"]},
    ]
    orig = oc.iter_works
    oc.iter_works = lambda key: iter(works)
    try:
        g = build_dated_graph("toy", row_of, wid, days, log=lambda *a, **k: None)
    finally:
        oc.iter_works = orig

    assert g["src"].size == 3  # three intra-universe edges total
    T = year_to_days(2015)
    adjT = g["at"](T)
    # present at T: W2->W1 (2012) and W3->W2 (2014); NOT W4->W3 (2016)
    assert adjT[row_of["W2"], row_of["W1"]] == 1
    assert adjT[row_of["W3"], row_of["W2"]] == 1
    assert adjT[row_of["W4"], row_of["W3"]] == 0, "future citing act leaked into the T-graph"
    # the T+Δ graph (T=2019) contains all three
    adjE = g["at"](year_to_days(2019))
    assert adjE.nnz == 3
    # graph_stats restricted to dated-at-T nodes (W1..W3 exist at T; W4 does not)
    dated_T = days <= T
    st = graph_stats(adjT, dated_T)
    assert st["n_nodes_dated"] == 3 and st["n_edges"] == 2


def test_oa_frozen_transform_identity_on_decoded_text():
    """The frozen T-vectorizer/LSA embeds OA-style decoded arrivals without refit
    and never admits a post-T-only token — the OA leakage-guard end to end."""
    from artifacts_scripts.gap_detection.oa_corpus import decode_abstract
    from artifacts_scripts.gap_detection.t_corpus import fit_t_corpus

    rng = np.random.default_rng(21)
    vocab_terms = [f"topic{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(120)]
    n = 300

    def inv_index(r):
        words = list(r.choice(vocab_terms, size=12, replace=False)) * 2
        return {w: [j for j, x in enumerate(words) if x == w] for w in set(words)}

    texts = [decode_abstract(inv_index(rng)) for _ in range(n)]
    assert all(isinstance(t, str) and t for t in texts)
    days = np.linspace(0, 1000, n)
    t_days = 600.0
    pre = np.where(days <= t_days)[0]
    post = np.where(days > t_days)[0]
    kw = dict(seed=5, n_docs=150, fg_size_range=(5, 150), min_interleave=0.0,
              lsa_dim=8, kmeans_k=4, n_keywords=20, tfidf_min_df=2)
    tc = fit_t_corpus(
        [texts[i] for i in pre], days[pre], t_days,
        titles=np.array([f"W{i}" for i in pre], dtype=object),
        page_ids=pre.astype(np.int64), store_rows=pre.astype(np.int64),
        part_a_rows=pre[:40].astype(np.int64), **kw,
    )
    assert (tc.days <= t_days).all()
    # arrivals carry a post-T-only token; it must never enter the frozen vocab
    post_texts = [texts[i] + " zzfutureonly zzfutureonly zzfutureonly" for i in post]
    Z_arr, K_arr, nnz = tc.transform_texts(post_texts)
    vocab = set(tc.vectorizer.get_feature_names_out().tolist())
    assert "zzfutureonly" not in vocab
    assert Z_arr.shape == (len(post_texts), tc.Z.shape[1])
    assert np.allclose(np.linalg.norm(Z_arr[nnz >= 5], axis=1), 1.0, atol=1e-4)


def test_oa_matched_null_calibrates_to_one():
    """Matched-null sanity: on RANDOM flagged locations drawn from the same pool,
    the fill-rate ratio must hug 1.0 with a CI spanning 1 (the wiki v1 bug
    inflated this to 1.27 — calibration must catch a biased matcher)."""
    from artifacts_scripts.gap_detection.t2_outcomes import (
        fill_counts,
        fill_rate_ratio,
        greedy_location_groups,
        knn_scale,
        matched_controls,
    )

    rng = np.random.default_rng(31)
    n = 1200
    pool = rng.normal(0, 1, size=(n, 16)).astype(np.float32)
    pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-9)  # unit sphere
    # arrivals: an exchangeable cloud (no detector structure) so any honest
    # matcher must score ~1.0; covariates = local density + a generic axis + growth
    arr = rng.normal(0, 1, size=(900, 16)).astype(np.float32)
    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
    radii = knn_scale(pool, pool, k=10, self_rows=np.ones(n, dtype=bool))
    fill_pool = fill_counts(pool, radii, arr).astype(np.float64)
    grow = fill_counts(pool, radii, pool[: n // 2]).astype(np.float64)
    C = np.column_stack([np.log(radii), pool[:, 0], np.log1p(grow)])

    flagged = rng.choice(n, size=50, replace=False)  # RANDOM locations
    excluded = np.zeros(n, dtype=bool)
    excluded[flagged] = True
    for f in flagged:
        excluded |= np.linalg.norm(pool - pool[f], axis=1) <= radii[f]
    ctr, info = matched_controls(
        pool[flagged], radii[flagged], C[flagged], pool, radii, C,
        excluded=excluded, m_controls=20, seed=7,
    )
    groups = greedy_location_groups(pool[flagged], radii[flagged])
    rr = fill_rate_ratio(fill_pool[flagged], fill_pool[ctr], n_boot=800, seed=9,
                         groups=groups)
    # the calibrated machinery null straddles 1 and sits near it
    assert rr["ci_lo_clustered"] <= 1.0 <= rr["ci_hi_clustered"], \
        f"random-location null does not span 1: {rr}"
    assert 0.8 <= rr["ratio"] <= 1.25, f"random-location ratio off 1.0: {rr['ratio']}"


def test_oa_dissoc_window_split_leakage_guard():
    """Q2 window-split contract: the W1 = (t0, t1] flow-change inputs use ONLY
    citing acts dated ≤ t1 and ONLY conduits dated inside (t0, t1]. A citing act
    or an arrival dated after t1 (i.e. in W2) must change NOTHING."""
    from artifacts_scripts.gap_detection.oa_corpus import year_to_days
    from artifacts_scripts.gap_detection.oa_dissoc import window_conduit_mask
    from artifacts_scripts.gap_detection.run_t2_oa_phase3 import _cite_adjacency

    t0, t1 = year_to_days(2015), year_to_days(2017)
    # universe: rows 0..3 pre-T works, row 4 a W1 arrival, row 5 a W2 arrival
    graph = {
        "src": np.array([1, 2, 4, 5], dtype=np.int64),
        "dst": np.array([0, 0, 1, 2], dtype=np.int64),
        "citing_days": np.array([
            year_to_days(2012), year_to_days(2014),   # pre-T citing acts
            year_to_days(2016),                        # W1 citing act (arrival 4 -> 1)
            year_to_days(2018),                        # W2 citing act (arrival 5 -> 2)
        ], dtype=np.float64),
        "n_works": np.int64(6),
    }
    arr_rows = np.array([4, 5], dtype=np.int64)
    arr_days = np.array([year_to_days(2016), year_to_days(2018)], dtype=np.float64)
    in_scope = np.array([True, True])

    # conduit mask: only the W1-dated arrival qualifies for the W1 field
    m = window_conduit_mask(arr_days, in_scope, t0, t1)
    assert m.tolist() == [True, False], "W2-dated arrival leaked into W1 conduits"

    # graph thresholds: nodes = scoped (0..3) + W1 conduits (4)
    nodes = np.concatenate([np.arange(4, dtype=np.int64), arr_rows[m]])
    W0 = _cite_adjacency(graph, nodes, t0)
    W1 = _cite_adjacency(graph, nodes, t1)
    assert W0.nnz == 4   # two pre-T edges, symmetrized
    assert W1.nnz == 6   # + the W1 citing act
    # the W2 citing act (5 -> 2, 2018) is absent from BOTH (and node 5 is not present)
    local_of = {int(g): j for j, g in enumerate(nodes)}
    assert 5 not in local_of
    assert W1[local_of[2], :].nnz == 1  # only its own pre-T act (2 -> 0)

    # invariance: adding another post-t1 citing act changes neither W0 nor W1
    graph2 = {
        "src": np.concatenate([graph["src"], [3]]),
        "dst": np.concatenate([graph["dst"], [0]]),
        "citing_days": np.concatenate([graph["citing_days"], [year_to_days(2018)]]),
        "n_works": graph["n_works"],
    }
    W0b = _cite_adjacency(graph2, nodes, t0)
    W1b = _cite_adjacency(graph2, nodes, t1)
    assert (W0 != W0b).nnz == 0 and (W1 != W1b).nnz == 0, \
        "a W2 citing act altered the W1-window graphs"


def test_oa_dissoc_infrastructure_operationalization():
    """Q1 contract: INFRA := g ≥ q75(g | citation-active pool) AND a ≤ median(a |
    pool); IR = log1p(g) − log1p(a). Pool-quantile thresholds, boundary inclusive."""
    from artifacts_scripts.gap_detection.oa_dissoc import infrastructure_operationalization

    pool_g_active = np.arange(101, dtype=np.float64)   # q75 = 75
    pool_a = np.arange(101, dtype=np.float64)          # median = 50
    g = np.array([90.0, 90.0, 5.0, 75.0, 76.0])
    a = np.array([10.0, 80.0, 10.0, 50.0, 51.0])
    op = infrastructure_operationalization(g, a, pool_g_active, pool_a)
    assert op["g_thr"] == 75.0 and op["a_thr"] == 50.0
    # cited-by-everyone + landed-on-by-no-one -> INFRA; either side broken -> not
    assert op["infra"].tolist() == [True, False, False, True, False]
    # IR orders the infrastructure-likeness within fixed g
    assert op["IR"][0] > op["IR"][1]
    # boundary semantics: g == thr counts (≥), a == thr counts (≤)
    assert bool(op["infra"][3])


# ------------------------------------------------------------------ T1 composite (Part A/B)
from artifacts_scripts.gap_detection.composite import (  # noqa: E402
    after_fg_pairs,
    fuse_max,
    fuse_mean,
    fuse_rrf,
    percentile_scores,
    voltage_pair_field,
)


def test_percentile_scores_abstain_and_sparse_top():
    """Abstentions (-inf) score 0; a sparse channel's top claim == a dense one's (1.0)."""
    dense = np.array([0.1, 0.5, 0.9, 0.3])
    p = percentile_scores(dense)
    assert p[2] == 1.0 and p.min() > 0.0
    sparse = np.array([-np.inf, 7.0, -np.inf, 1.0])
    q = percentile_scores(sparse)
    assert q[1] == 1.0 and q[0] == 0.0 and q[2] == 0.0 and 0 < q[3] < 1.0


def test_fusion_rules_contract():
    """max preserves a single-channel-only signal (the concept analog: only the
    miner channel sees it); mean dilutes it below two-channel consensus; an
    all-abstain candidate (probe analog) is bottom under every rule."""
    rng = np.random.default_rng(0)
    n = 200
    ch_a = rng.normal(size=n)
    ch_b = rng.normal(size=n)
    ch_c = np.full(n, -np.inf)
    ch_c[7] = 5.0                      # single-channel-only signal at 7
    ch_a[7] = ch_b[7] = -np.inf        # other channels abstain there
    ch_a[3] = ch_b[3] = 10.0           # two-channel consensus candidate at 3
    probe = 11
    ch_a[probe] = ch_b[probe] = ch_c[probe] = -np.inf
    ps = [percentile_scores(c) for c in (ch_a, ch_b, ch_c)]
    f_max = fuse_max(ps)
    f_mean = fuse_mean(ps)
    f_rrf = fuse_rrf([ch_a, ch_b, ch_c])
    # max: the single-channel top ties the global top stratum
    assert (f_max > f_max[7]).sum() == 0
    # mean: diluted below the consensus candidate
    assert f_mean[7] < f_mean[3]
    # all rules: the all-abstain probe is strictly bottom
    for f in (f_max, f_mean, f_rrf):
        assert (f <= f[probe]).sum() == 1


def test_after_fg_pairs_selects_emptied_band_and_voltage_ranks_it():
    """2D fixture: removing the fg bridge between two blobs raises that pair's
    band-emptiness well above the intact value, and the self-selected pair's
    voltage field ranks the band bg on top."""
    rng = np.random.default_rng(3)
    fgA = rng.normal([-3.0, 0.0], 0.6, size=(500, 2))
    fgB = rng.normal([3.0, 0.0], 0.6, size=(500, 2))
    bridge = np.column_stack([rng.uniform(-1.5, 1.5, 250), rng.normal(0, 0.3, 250)])
    Zf = np.vstack([fgA, fgB, bridge]).astype(np.float32)
    Zb = rng.uniform([-4.5, -2.0], [4.5, 2.0], size=(2500, 2)).astype(np.float32)
    _, _, pairs_before = after_fg_pairs(Zf, Zb, n_clusters=2, seed=7)
    Zf_after = Zf[:1000]  # bridge removed
    labels, cents, pairs_after = after_fg_pairs(Zf_after, Zb, n_clusters=2, seed=7)
    assert pairs_before and pairs_after
    assert pairs_after[0]["emptiness"] > pairs_before[0]["emptiness"] + 0.1
    assert pairs_after[0]["emptiness"] > 0.75
    r = voltage_pair_field(Zf_after, Zb, labels, cents, pairs_after[0], k_graph=8)
    assert r is not None
    sc, diag = r
    # the 2D equidistance band is the vertical strip around the bisector x=0
    member = np.abs(Zb[:, 0]) <= 1.2
    top = np.argsort(np.where(np.isfinite(sc), sc, -np.inf))[::-1][:40]
    assert member[top].mean() >= 0.6
    assert rank_of(sc, member, np.isfinite(sc)) <= 5
    # the field abstains (-inf) outside the pair's subgraph
    assert not np.isfinite(sc).all()


def test_radial_frontier_monotonicity_fixture():
    """Frontier contract on a 2D toy: the orphan core (the gap's footprint) is
    non-decreasing in the removed-subset size, and every CONSTRUCTED cell is
    recoverable (AUC >= 0.75). AUC itself is NOT monotone in size — a tiny
    removal that does leave a core leaves a sharply orphaned one (measured here:
    AUC 0.97 at m=6 vs 0.88 at m=80); the size-dependence of detectability is
    carried by footprint existence (construct rate), which is what the real
    frontier's spec rule quotes."""
    from artifacts_scripts.gap_detection.synth import orphaned_core

    rng = np.random.default_rng(5)
    Zf = rng.normal(0.0, 1.0, size=(1500, 2)).astype(np.float32)
    Zb = rng.normal(0.0, 1.8, size=(5000, 2)).astype(np.float32)
    center = np.array([0.45, 0.0], dtype=np.float32)
    d_fg = np.linalg.norm(Zf - center, axis=1)
    ctrl_centers = [np.array([-0.45, 0.0]), np.array([0.0, -0.45])]
    cores, aucs = [], []
    for m in (6, 24, 80):
        removed = np.argsort(d_fg)[:m]
        r_ball = float(d_fg[removed[-1]])
        fgp = np.delete(Zf, removed, axis=0)
        bg_in = np.where(np.linalg.norm(Zb - center, axis=1) <= r_ball)[0]
        core = bg_in[orphaned_core(Zb[bg_in], Zf, fgp, k=10, ratio=1.15)]
        cores.append(core.size)
        controls = []
        for cc in ctrl_centers:
            dc = np.linalg.norm(Zf - cc, axis=1)
            c_removed = np.argsort(dc)[:m]
            c_ball = float(dc[c_removed[-1]])
            c_bg = np.where(np.linalg.norm(Zb - cc, axis=1) <= c_ball)[0]
            c_after = np.delete(Zf, c_removed, axis=0)
            c_core = c_bg[orphaned_core(Zb[c_bg], Zf, c_after, k=10, ratio=1.15)]
            controls.append(np.setdiff1d(c_core, bg_in))
        neg = np.concatenate(controls)
        if core.size and neg.size:
            scorer = GapScorer(bg=Zb, k=10)
            scorer.set_r0_from_fg(Zf)
            out = scorer.score(fgp)
            sc = fused_scores(out, veto_mask(scorer, out))
            aucs.append(auc_vs_controls(sc, core, neg))
        else:
            aucs.append(None)
    assert cores[0] <= cores[1] <= cores[2], cores
    assert cores[2] > cores[0]
    assert all(a >= 0.75 for a in aucs if a is not None), aucs


# --------------------------------------------------------------------------- track 3: book audit
from artifacts_scripts.gap_detection.book_audit import (  # noqa: E402
    embed_joint,
    split_units,
    strip_latex,
)


def test_t3_strip_latex_rule():
    """The documented LaTeX->text contract: comments, code, math, refs dropped;
    text commands unwrapped; captions survive; escapes/structural chars cleaned."""
    tex = (
        "Intro text % a comment to drop\n"
        "with \\textbf{bold} and \\emph{emphasis} kept. "
        "Math $x^2 + y$ goes, display \\[ \\sum_i a_i \\] goes, "
        "$$ \\int f $$ goes, and \\begin{equation} e = mc^2 \\end{equation} goes. "
        "\\begin{lstlisting}[language=Python]\nprint('secret code')\n\\end{lstlisting} "
        "A citation \\cite{foo2020} and \\ref{sec:x} and \\label{here} vanish. "
        "\\begin{figure}\\centering\\includegraphics[width=0.5\\textwidth]{img.png}"
        "\\caption{A caption survives}\\label{fig:1}\\end{figure} "
        "Escapes: A \\& B, 100\\%. \\begin{itemize}\\item first point\\end{itemize}"
    )
    out = strip_latex(tex)
    for kept in ("Intro text", "bold", "emphasis", "A caption survives",
                 "A & B", "100%", "first point"):
        assert kept in out, kept
    for gone in ("comment to drop", "secret code", "mc^2", "foo2020", "sec:x",
                 "img.png", "lstlisting", "x^2", "sum_i", "int f",
                 "\\", "{", "}", "$", "%a"):
        assert gone not in out, gone
    assert "  " not in out  # whitespace collapsed


def test_t3_split_units_rule():
    """Unit rule: chapter intro + one unit per \\section/\\subsection; titles
    prepended; tiny units merge into the previous unit (coverage kept)."""
    body = (
        "\\documentclass{book}\\usepackage{amsmath}\n"
        "\\begin{document}\n"
        "\\chapter{Toy Chapter}\n"
        "This chapter intro has plenty of text. " * 8 +
        "\\section{Alpha Topic}\n" + "Alpha section body text here. " * 10 +
        "\\subsection{Alpha Sub}\n" + "Subsection body about alpha details. " * 10 +
        "\\section{Tiny}\nToo short.\n"
        "\\section{Beta Topic}\n" + "Beta section body text continues. " * 10 +
        "\\end{document}\n"
    )
    units = split_units(body, "ch1")
    labels = [(u.section_title, u.subsection_title) for u in units]
    assert labels[0] == ("(chapter intro)", None)
    assert labels[1] == ("Alpha Topic", None)
    assert labels[2] == ("Alpha Topic", "Alpha Sub")  # inherits parent section
    assert all(u.chapter_title == "Toy Chapter" for u in units)
    # the tiny "Tiny" section merged into the PREVIOUS unit (the subsection)
    assert ("Tiny", None) not in labels
    assert "Too short" in units[2].text and "Tiny" in units[2].merged_titles
    assert labels[3] == ("Beta Topic", None) and len(units) == 4
    # titles are part of the unit text; intro carries the chapter title
    assert units[1].text.startswith("Alpha Topic")
    assert units[0].text.startswith("Toy Chapter")
    assert all(len(u.text) >= 200 for u in units)
    assert units[1].label == "ch1 Toy Chapter § Alpha Topic"


def test_t3_embed_joint_identity():
    """Joint-embed contract: fg rows first and aligned (marker tokens land in
    the right rows of K), unit-norm Z, binary K, deterministic for fixed seed."""
    rng = np.random.default_rng(7)
    base = ["data analysis statistics inference model learning theory"] * 3
    marks = ["fgmarkeralpha", "fgmarkerbeta", "fgmarkergamma"]
    fg = [f"{t} {m} {m}" for t, m in zip(base, marks)]
    bg = [
        " ".join(rng.choice(
            ["regression", "bayesian", "matrix", "kernel", "entropy", "risk",
             "estimator", "variance", "spectral", "sparse", "data", "model"],
            size=12, replace=True)) + f" doc{j}"
        for j in range(40)
    ]
    e1 = embed_joint(fg, bg, min_df=1, max_df=1.0, lsa_dim=8, n_keywords=30, seed=11)
    e2 = embed_joint(fg, bg, min_df=1, max_df=1.0, lsa_dim=8, n_keywords=30, seed=11)
    assert e1["fg_mask"].sum() == 3 and e1["fg_mask"][:3].all()
    assert e1["Z"].shape == (43, 8) and e1["K"].shape[0] == 43
    np.testing.assert_allclose(np.linalg.norm(e1["Z"], axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(e1["Z"], e2["Z"])  # determinism
    assert set(np.unique(e1["K"].toarray())) <= {0, 1}
    voc = list(e1["tfidf_vocab"])
    for i in range(3):  # fg row i carries exactly its own marker
        col = voc.index(marks[i])
        hit = np.where(np.asarray((e1["X"][:, col] > 0).todense()).ravel())[0]
        np.testing.assert_array_equal(hit, [i])


def test_t3_full_statistic_identity_with_gapscorer():
    """t3's split-k statistic == GapScorer 'full' when k_fg == k_bg (the only
    deviation from the validated implementation is the k decoupling)."""
    from artifacts_scripts.gap_detection.book_audit import full_statistic

    rng = np.random.default_rng(3)
    F = rng.normal(size=(60, 8)).astype(np.float32)
    B = rng.normal(size=(300, 8)).astype(np.float32)
    for k in (3, 10):
        scorer = GapScorer(bg=B, probes=None, k=k)
        out = scorer.score(F)
        mine = full_statistic(F, B, k_fg=k, k_bg=k)
        np.testing.assert_allclose(mine["full"], out["full"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(mine["r_F"], out["_r_Fk"], rtol=1e-6)
    # decoupled form: r_F side changes, r_B/hab side does not
    m35 = full_statistic(F, B, k_fg=3, k_bg=10)
    m10 = full_statistic(F, B, k_fg=10, k_bg=10)
    np.testing.assert_array_equal(m35["r_B"], m10["r_B"])
    assert (m35["r_F"] <= m10["r_F"] + 1e-9).all()  # 3-NN radius <= 10-NN radius


# --------------------------------------------------------------------------- curriculum showdown
from artifacts_scripts.gap_detection.toc_metrics import (  # noqa: E402
    citation_flow,
    coverage_at_radius,
    forward_violations,
    n_violations,
    nearest_rep_dist,
    order_min_forward_flow,
    region_closure,
    select_k,
)


def test_showdown_select_k_rule():
    """Pre-committed k rule: max silhouette among mass-feasible k; documented
    fallback (fewest violating clusters, then higher silhouette) when none is."""
    rows = {
        8: {"silhouette": 0.30, "shares": [1 / 8] * 8},                # feasible
        9: {"silhouette": 0.35, "shares": [0.26] + [0.74 / 8] * 8},    # >25% cluster
        10: {"silhouette": 0.32, "shares": [1 / 10] * 10},             # feasible
        11: {"silhouette": 0.20, "shares": [0.02] + [0.98 / 10] * 10}, # <3% cluster
    }
    k, why = select_k(rows)
    assert (k, why) == (10, "feasible-max-silhouette")   # 0.35 and 0.20 disqualified
    assert n_violations(rows[9]["shares"]) == 1
    assert n_violations(rows[11]["shares"]) == 1
    # fallback: nothing feasible -> fewest violations, then higher silhouette
    bad = {
        5: {"silhouette": 0.50, "shares": [0.30, 0.28, 0.28, 0.07, 0.07]},  # 3 violations
        6: {"silhouette": 0.10, "shares": [0.26] + [0.74 / 5] * 5},         # 1 violation
        7: {"silhouette": 0.40, "shares": [0.27] + [0.73 / 6] * 6},         # 1 violation
    }
    k, why = select_k(bad)
    assert (k, why) == (7, "fallback-fewest-violations")  # ties on 1 violation -> higher sil


def test_showdown_kmeans_sweep_determinism():
    """Same seed, same data -> identical silhouettes, shares, and chosen k
    (the generator's chapter rule is deterministic end to end)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    rng = np.random.default_rng(5)
    blobs = np.vstack([rng.normal(loc=c, scale=0.08, size=(60, 6))
                       for c in np.eye(6)[:4] * 1.5]).astype(np.float32)
    Z = normalize(blobs)

    def sweep():
        rows = {}
        for k in (3, 4, 5):
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
            sil = float(silhouette_score(Z, km.labels_, metric="euclidean",
                                         sample_size=min(len(Z), 200), random_state=42))
            shares = (np.bincount(km.labels_, minlength=k) / len(Z)).tolist()
            rows[k] = {"silhouette": sil, "shares": shares}
        return rows, select_k(rows)

    r1, pick1 = sweep()
    r2, pick2 = sweep()
    assert pick1 == pick2
    for k in r1:
        assert r1[k]["silhouette"] == r2[k]["silhouette"]
        assert r1[k]["shares"] == r2[k]["shares"]
    # 4 true blobs of 60/240 = 25% each: k=4 is feasible and silhouette-max
    assert pick1[0] == 4 and pick1[1] == "feasible-max-silhouette"


def test_showdown_ordering_three_cluster_fixture():
    """Synthetic 3-cluster citation fixture: C cites B, B cites A => order
    A, B, C with zero forward flow; DP optimum == brute force; reversed order
    counts every edge as a violation."""
    from itertools import permutations

    # edges (citing_cluster, cited_cluster): 10x C->B, 10x B->A, 2x C->A
    src = np.array([2] * 10 + [1] * 10 + [2] * 2)
    dst = np.array([1] * 10 + [0] * 10 + [0] * 2)
    F = citation_flow(src, dst, 3)
    assert F[2, 1] == 10 and F[1, 0] == 10 and F[2, 0] == 2 and F.sum() == 22
    years = np.array([2005.0, 2010.0, 2015.0])
    order, cost = order_min_forward_flow(F, years)
    assert order == [0, 1, 2] and cost == 0.0

    def brute(F):
        return min(sum(F[a, b] for i, a in enumerate(p) for b in p[i + 1:])
                   for p in permutations(range(F.shape[0])))

    assert cost == brute(F)
    # a noisy fixture where some forward flow is unavoidable
    F2 = F.copy()
    F2[0, 2] = 3.0   # A also cites C: any order pays something
    order2, cost2 = order_min_forward_flow(F2, years)
    assert cost2 == brute(F2) == 3.0   # keep A,B,C: pay only the A->C edge
    v = forward_violations(F2, order2)
    assert v["violations"] == cost2 and v["cross_edges"] == 25.0
    rev = forward_violations(F, [2, 1, 0])
    assert rev["violations"] == 22.0 and rev["share"] == 1.0
    good = forward_violations(F, [0, 1, 2])
    assert good["violations"] == 0.0 and good["share"] == 0.0


def test_showdown_ordering_tie_break_median_year():
    """Zero citation flow degenerates to ascending median-year order (the
    pre-committed tie-break), deterministically."""
    F = np.zeros((4, 4))
    years = np.array([2018.0, 2003.0, 2011.0, 2007.0])
    order, cost = order_min_forward_flow(F, years)
    assert cost == 0.0
    assert order == [1, 3, 2, 0]   # ascending year
    assert order == order_min_forward_flow(F, years)[0]  # deterministic


def test_showdown_coverage_and_closure_metrics():
    """Metric 1/2 contracts on hand-placed points: coverage counts docs within
    the radius of the nearest rep; a region is closed iff its MEDIAN member
    distance is inside the radius."""
    # 6 docs on a line, reps at 0 and 10
    Z = np.array([[0.0], [0.5], [1.1], [9.6], [10.0], [5.0]])
    reps = np.array([[0.0], [10.0]])
    d = nearest_rep_dist(Z, reps)
    np.testing.assert_allclose(d, [0.0, 0.5, 1.1, 0.4, 0.0, 5.0], atol=1e-4)
    assert coverage_at_radius(Z, reps, radius=1.0) == pytest.approx(4 / 6)
    # region A members {0,1,2}: median d = 0.5 -> closed at r=1.0;
    # region B members {2,5}: median d = (1.1+5.0)/2 = 3.05 -> open
    closed = region_closure(Z, [np.array([0, 1, 2]), np.array([2, 5])], reps, radius=1.0)
    assert closed == [True, False]
    # one rep covering everything closes both
    assert region_closure(Z, [np.array([0, 1, 2]), np.array([2, 5])],
                          np.array([[5.0]]), radius=6.0) == [True, True]


def test_showdown_prereq_violation_counts():
    """Metric 3: violations = forward-pointing citation mass under the given
    order; shares use cross-chapter mass only (diagonal ignored)."""
    F = np.array([[5.0, 4.0, 0.0],
                  [1.0, 7.0, 2.0],
                  [3.0, 0.0, 9.0]])
    # order [0,1,2]: forward = F[0,1]+F[0,2]+F[1,2] = 6; cross = 10
    v = forward_violations(F, [0, 1, 2])
    assert v["violations"] == 6.0 and v["cross_edges"] == 10.0
    assert v["share"] == pytest.approx(0.6)
    # best order found by DP must not exceed any manual order
    order, cost = order_min_forward_flow(F, np.array([2000.0, 2001.0, 2002.0]))
    assert cost <= 6.0
    assert forward_violations(F, order)["violations"] == cost


def test_showdown_gap_bar_and_fullstat_closure():
    """Revised closure rule: the bar is the lowest score inside the top-frac
    level set; the incumbent (whose scores defined the selection) closes 0 by
    construction; a competitor closes exactly the regions it drives below bar."""
    from artifacts_scripts.gap_detection.toc_metrics import (
        gap_bar,
        region_closure_fullstat,
    )

    full_inc = np.array([9.0, 8.0, 7.0, 6.0, 1.0, 0.5, 0.2, 0.1, 0.0, -1.0])
    bar = gap_bar(full_inc, top_frac=0.4)        # top-4 -> lowest selected = 6.0
    assert bar == 6.0
    regions = [np.array([0, 1]), np.array([2, 3])]   # both inside the selection
    # incumbent: members' own scores are >= bar -> nothing closes
    assert region_closure_fullstat(full_inc, regions, bar) == [False, False]
    # competitor whose sections sit on region 2: those members drop below bar
    full_comp = full_inc.copy()
    full_comp[[2, 3]] = [0.5, 0.3]
    assert region_closure_fullstat(full_comp, regions, bar) == [False, True]
    # mean rule: one member below, one far above -> mean stays above -> open
    full_half = full_inc.copy()
    full_half[2] = -5.0
    full_half[3] = 20.0
    assert region_closure_fullstat(full_half, regions, bar) == [False, False]


def test_showdown_fullstat_closure_with_real_statistic():
    """End-to-end on toy geometry: a fg cloud leaves a dense bg pocket
    uncovered -> the pocket clears the gap bar under fg; adding a rep INSIDE
    the pocket closes it under the re-run statistic (the revised rule's
    actual mechanics, GapScorer-backed)."""
    from artifacts_scripts.gap_detection.book_audit import full_statistic
    from artifacts_scripts.gap_detection.toc_metrics import (
        gap_bar,
        region_closure_fullstat,
    )

    rng = np.random.default_rng(11)
    bg_main = rng.normal(size=(400, 6)).astype(np.float32)
    pocket = (rng.normal(size=(40, 6)) * 0.15 + 4.0).astype(np.float32)
    B = np.vstack([bg_main, pocket])
    fg = rng.normal(size=(80, 6)).astype(np.float32)      # covers main only
    s_inc = full_statistic(fg, B, k_fg=3, k_bg=10)["full"]
    bar = gap_bar(s_inc, top_frac=0.1)
    region = [np.arange(400, 440)]                        # the pocket
    assert region_closure_fullstat(s_inc, region, bar) == [False]  # open under fg
    fg_plus = np.vstack([fg, pocket[:5]])                 # a "section" in the pocket
    s_new = full_statistic(fg_plus, B, k_fg=3, k_bg=10)["full"]
    assert region_closure_fullstat(s_new, region, bar) == [True]   # closed


# ===================================================================== showdown p1
def test_p1_fit_embed_matches_embed_joint():
    """fit_embed REPLICATES book_audit.embed_joint (book_audit stays read-only):
    identical Z / vocab / evr on a toy corpus, plus the fitted objects."""
    from artifacts_scripts.gap_detection.book_audit import embed_joint
    from artifacts_scripts.gap_detection.showdown_p1 import fit_embed

    rng = np.random.default_rng(3)
    words = ["alpha", "beta", "gamma", "delta", "epsil", "zeta", "etaaa", "theta"]
    texts = []
    for i in range(100):
        w = [words[int(j)] for j in rng.choice(8, size=2, replace=False)]
        texts.append(" ".join(w * 6))
    a = embed_joint([], texts, seed=42, lsa_dim=64)
    b = fit_embed([], texts, seed=42, lsa_dim=64)
    assert list(a["tfidf_vocab"]) == list(b["tfidf_vocab"])
    assert np.allclose(a["Z"], b["Z"], atol=1e-6)
    assert abs(a["evr"] - b["evr"]) < 1e-9
    assert (a["K"] != b["K"]).nnz == 0
    assert hasattr(b["vec"], "transform") and hasattr(b["svd"], "transform")


def test_p1_transform_texts_roundtrip_and_zero_guard():
    """transform_texts maps TRAINING texts back onto their fitted embedding
    (same unit vectors) and zero-vocabulary texts are masked out, not faked."""
    from artifacts_scripts.gap_detection.showdown_p1 import fit_embed, transform_texts

    rng = np.random.default_rng(4)
    words = ["alpha", "beta", "gamma", "delta", "epsil", "zeta", "etaaa",
             "theta", "iotaa", "kappa", "lambd", "muuuu"]
    texts = []
    for i in range(100):
        w = [words[int(j)] for j in rng.choice(12, size=3, replace=False)]
        texts.append(" ".join(w * 4))
    emb = fit_embed([], texts)
    Z2, ok = transform_texts(emb["vec"], emb["svd"], texts[:10])
    assert ok.all()
    assert np.allclose(Z2, emb["Z"][:10], atol=1e-4)
    Z3, ok3 = transform_texts(emb["vec"], emb["svd"],
                              ["zzzz qqqq wwww", texts[0]])
    assert not ok3[0] and ok3[1]
    assert np.allclose(Z3[0], 0.0)
    assert np.allclose(Z3[1], emb["Z"][0], atol=1e-4)


def test_p1_parse_toc_md_fixture():
    """Tolerant toc.md parser: bold/plain bullets, ':' and em-dash splits,
    plain Intent/Dependencies lines, sectionless-chapter fallback."""
    from artifacts_scripts.gap_detection.showdown_p1 import (
        parse_toc_md,
        section_rep_texts,
    )

    text = """# Table of Contents: Toy

## Chapter 1: Foundations
*Intent: basics first.*
- **1.1 Vectors**: linear algebra intro.
- **1.2 Calculus:** derivatives and integrals.
**Dependencies:** none.

## Chapter 2 — Methods
Intent: the core methods.
* 2.1 Regression — fitting lines.
* 2.2 Classification: drawing boundaries.
Dependencies: Chapter 1.

## Chapter 3: Outlook
*Intent: chapter with no sections.*
**Dependencies:** Chapter 2.
"""
    p = parse_toc_md(text)
    assert p["h1"] == "Table of Contents: Toy"
    assert [c["num"] for c in p["chapters"]] == [1, 2, 3]
    c1, c2, c3 = p["chapters"]
    assert c1["title"] == "Foundations" and c1["intent"] == "basics first"
    assert [s["title"] for s in c1["sections"]] == ["Vectors", "Calculus"]
    assert c1["sections"][1]["intent"] == "derivatives and integrals"
    assert c1["deps"] == "none"
    assert c2["title"] == "Methods" and c2["intent"] == "the core methods"
    assert [s["label"] for s in c2["sections"]] == ["2.1", "2.2"]
    assert c2["sections"][0]["intent"] == "fitting lines"
    assert c2["deps"] == "Chapter 1"
    assert c3["fallback_section"] and len(c3["sections"]) == 1
    assert p["n_fallback_chapters"] == 1
    texts, chap_idx = section_rep_texts(p)
    assert len(texts) == 5 and chap_idx == [0, 0, 1, 1, 2]
    assert texts[0] == "Vectors. linear algebra intro"


def test_p1_ab_assignment_deterministic_and_swapped():
    """Blinding plan: seeded, deterministic, 9 cells; order-2 is exactly the
    A/B swap of order-1; each arm of a pair appears as A exactly once."""
    from artifacts_scripts.gap_detection.showdown_p1 import (
        PAIRS,
        TOPIC_ORDER,
        ab_assignments,
    )

    a1, a2 = ab_assignments(), ab_assignments()
    assert a1 == a2
    assert len(a1) == 9
    for t in TOPIC_ORDER:
        for pair in PAIRS:
            o1, o2 = a1[(t, pair)]["order1"], a1[(t, pair)]["order2"]
            assert {o1["A"], o1["B"]} == set(pair)
            assert o2["A"] == o1["B"] and o2["B"] == o1["A"]


def test_p1_majority_and_winner_extraction():
    from artifacts_scripts.gap_detection.showdown_p1 import (
        extract_winner,
        majority_of,
    )

    assert majority_of(["algo", "algo", "algo", "agent"]) == "algo"
    assert majority_of(["algo", "agent", "algo", "agent"]) == "tie"
    assert majority_of(["algo", "algo", None, None]) == "tie"      # no >2 majority
    assert majority_of(["algo", "algo", "algo", None]) == "algo"
    assert majority_of([None, None, None, None]) == "tie"
    assert extract_winner("WINNER: A\nx\ny\nz") == "A"
    assert extract_winner("**Winner: B**\nrationale") == "B"
    assert extract_winner("The winner - (a) because...") == "A"
    assert extract_winner("no verdict here") is None


def test_p1_assign_by_nearest_toy():
    """Doc->chapter mapping: nearest kept section rep's chapter, exact on a
    hand-made fixture (two reps per chapter, docs planted at the reps)."""
    from artifacts_scripts.gap_detection.showdown_p1 import assign_by_nearest

    reps = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [1.0, 5.0]],
                    dtype=np.float32)
    sec_chap = np.array([0, 0, 1, 1])
    Z = np.array([[0.1, 0.1], [0.9, -0.1], [0.1, 4.9], [2.0, 5.5],
                  [0.5, 2.4]], dtype=np.float32)   # last: nearer y=0 band
    chap = assign_by_nearest(Z, reps, sec_chap)
    assert chap.tolist() == [0, 0, 1, 1, 0]


# --------------------------------------------------------------------------- R0 round-trip
def _r0_block_context():
    """Clean two-chapter formal context (worklog R0 pre-commits): 16 docs x 12
    terms. Cols 0-2 = A core (all 8 A docs), col 3 = A straddler (exactly 2/8 =
    the minsup boundary), cols 4-7 = B core, cols 8-9 = register (every doc),
    col 10 = junk (every doc), col 11 = A rare (1/8, below minsup)."""
    K = np.zeros((16, 12), dtype=np.uint8)
    K[:8, [0, 1, 2]] = 1          # A core
    K[[0, 1], 3] = 1              # A straddler: support 2 of 8 == minsup
    K[8:, [4, 5, 6, 7]] = 1       # B core
    K[:, [8, 9]] = 1              # register both sides
    K[:, 10] = 1                  # junk col
    K[0, 11] = 1                  # rare: support 1 < minsup
    junk = np.zeros(12, dtype=bool)
    junk[10] = True
    return sp.csr_matrix(K), junk


def test_r0_galois_extent_intent_toy():
    """Galois toy: on the clean block context the pre-committed operators
    behave as a Galois connection — extent(intent(D_i)) = D_i and
    intent(extent(K_i)) = K_i; register/junk excluded; the straddler term at
    exactly 25% support is IN, the sub-minsup term OUT."""
    from artifacts_scripts.gap_detection.roundtrip import (
        extent_respect,
        extract_closure,
    )

    K, junk = _r0_block_context()
    D = [np.arange(8), np.arange(8, 16)]
    intents = [extract_closure(K, m, junk) for m in D]
    assert intents[0]["cols"] == [0, 1, 2, 3]      # core (sup 8) then straddler (sup 2)
    assert intents[0]["minsup"] == 2
    assert intents[1]["cols"] == [4, 5, 6, 7]
    assert intents[0]["n_register"] == 2           # cols 8, 9
    assert 11 not in intents[0]["cols"]            # below minsup
    # extent(intent(D_i)) = D_i
    labels = extent_respect(K, [i["cols"] for i in intents])
    assert np.array_equal(np.where(labels == 0)[0], D[0])
    assert np.array_equal(np.where(labels == 1)[0], D[1])
    # intent(extent(K_i)) = K_i
    re0 = extract_closure(K, np.where(labels == 0)[0], junk)
    re1 = extract_closure(K, np.where(labels == 1)[0], junk)
    assert re0["cols"] == intents[0]["cols"]
    assert re1["cols"] == intents[1]["cols"]


def test_r0_closure_idempotence_toy():
    """The owner's contract: on mechanically derived chapters, re-extracting
    after re-deriving reproduces the keyword set exactly (chapter-level
    idempotence, E o D o E = E) — both extractors on the clean fixture."""
    from artifacts_scripts.gap_detection.roundtrip import (
        extent_respect,
        extract_sets,
    )

    K, junk = _r0_block_context()
    D = [np.arange(8), np.arange(8, 16)]
    for ext in ("closure", "ctfidf"):
        sets1 = [s["cols"] for s in extract_sets(K, D, junk, ext)]
        labels = extent_respect(K, sets1)
        D2 = [np.where(labels == i)[0] for i in range(2)]
        sets2 = [s["cols"] for s in extract_sets(K, D2, junk, ext)]
        assert sets2 == sets1, ext
        # and the derived extents are exactly the original chapters
        assert all(np.array_equal(a, b) for a, b in zip(D2, D))


def test_r0_extent_assignment_determinism():
    """Pre-committed extent rules: best-match share, ties -> EARLIEST published
    chapter; zero-match docs -1; repeat calls identical; free extent = docs
    with >=1 pooled term."""
    from artifacts_scripts.gap_detection.roundtrip import (
        extent_free,
        extent_respect,
    )

    K = sp.csr_matrix(np.array([
        [1, 1, 0, 0, 0],     # full match K0
        [0, 0, 1, 1, 0],     # full match K1
        [1, 0, 1, 0, 0],     # tie (1/2 vs 1/2) -> chapter 0 (earliest)
        [0, 0, 0, 0, 1],     # matches nothing -> -1
    ], dtype=np.uint8))
    sets = [[0, 1], [2, 3]]
    lab1 = extent_respect(K, sets)
    lab2 = extent_respect(K, sets)
    assert lab1.tolist() == [0, 1, 0, -1]
    assert np.array_equal(lab1, lab2)
    # empty keyword set never wins; pool extent is the >=1-hit union
    assert extent_respect(K, [[], [2, 3]]).tolist() == [-1, 1, 1, -1]
    assert extent_free(K, sets).tolist() == [0, 1, 2]
    assert extent_free(K, [[], []]).size == 0


def test_r0_respect_vs_free_mode_contract():
    """Mode contract: RESPECT keeps the keyword-set chapter identities (extents
    = the original blocks) and re-derives ONLY ordering/sections — the citation
    flow reorders the published order; FREE re-runs the frozen generator, so k
    comes from the pre-committed sweep rule, not from the input chapter count."""
    from artifacts_scripts.gap_detection.roundtrip import (
        derive_free,
        derive_respect,
        extent_free,
        extent_respect,
    )
    from artifacts_scripts.gap_detection.toc_metrics import select_k

    # ---- respect: 2 chapters, published order [B, A], 6 edges B cites A
    K, junk = _r0_block_context()
    rng = np.random.default_rng(42)
    Z = np.vstack([rng.normal(0, 0.01, (8, 4)) + np.eye(4)[0],
                   rng.normal(0, 0.01, (8, 4)) + np.eye(4)[1]]).astype(np.float32)
    td = {"Z": Z, "K": K, "vocab": [f"t{i}" for i in range(12)],
          "titles": [f"doc {i}" for i in range(16)],
          "year": np.array([2001] * 8 + [2005] * 8, dtype=np.int64),
          "src": np.array([8, 9, 10, 11, 12, 13], dtype=np.int64),   # B docs...
          "dst": np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),       # ...cite A docs
          "junk": junk}
    sets_pub = [[4, 5, 6, 7], [0, 1, 2, 3]]      # position 0 = B, position 1 = A
    labels = extent_respect(td["K"], sets_pub)
    d = derive_respect(td, labels, 2)
    assert d["mode"] == "respect" and d["n_unassigned"] == 0
    assert sorted(d["chap_order"]) == [0, 1]                  # identities kept
    assert d["chap_order"] == [1, 0]                          # A before B (flow)
    assert d["forward_flow"]["violations"] == 0.0
    assert np.array_equal(np.where(d["labels"] == 1)[0], np.arange(8))
    assert {c["cluster_id"] for c in d["chapters"]} == {0, 1}

    # ---- free: 5 tight blobs; the generator re-picks k by the frozen rule
    nb, per = 5, 20
    Zf = np.vstack([rng.normal(0, 0.01, (per, 8)) + 2.0 * np.eye(8)[i]
                    for i in range(nb)]).astype(np.float32)
    Kf = np.zeros((nb * per, 10), dtype=np.uint8)
    for i in range(nb):
        Kf[i * per:(i + 1) * per, [2 * i, 2 * i + 1]] = 1
    tdf = {"Z": Zf, "K": sp.csr_matrix(Kf), "vocab": [f"w{i}" for i in range(10)],
           "titles": [f"work {i}" for i in range(nb * per)],
           "year": np.repeat(np.arange(2000, 2000 + nb), per).astype(np.int64),
           "src": np.empty(0, dtype=np.int64), "dst": np.empty(0, dtype=np.int64),
           "junk": np.zeros(10, dtype=bool)}
    pool_sets = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    rows = extent_free(tdf["K"], pool_sets)
    assert rows.size == nb * per                      # every doc has >=1 pooled term
    df = derive_free(tdf, rows)
    assert df["mode"] == "free"
    assert df["k"] == select_k(df["sweep"])[0]        # rule-picked, k range {5..10}
    assert df["k"] == nb                              # the 5 true blobs
    # zero citation flow -> pre-committed median-year ascending tie-break
    years = [c["median_year"] for c in df["chapters"]]
    assert years == sorted(years)


def test_r0_ctfidf_toy_top_terms():
    """c-TF-IDF comparator: block terms rank top for their class, register and
    junk columns are excluded, weights are finite and descending."""
    from artifacts_scripts.gap_detection.roundtrip import extract_ctfidf

    K, junk = _r0_block_context()
    D = [np.arange(8), np.arange(8, 16)]
    out = extract_ctfidf(K, D, junk)
    assert set(out[0]["cols"][:3]) == {0, 1, 2}       # A core leads
    assert 3 in out[0]["cols"] and 11 in out[0]["cols"]   # no minsup in ctfidf
    assert not ({8, 9, 10} & set(out[0]["cols"]))     # register + junk excluded
    assert set(out[1]["cols"]) == {4, 5, 6, 7}
    w = out[0]["weight"]
    assert all(np.isfinite(w)) and w == sorted(w, reverse=True)
    assert extract_ctfidf(K, [np.empty(0, dtype=np.int64)], junk)[0]["cols"] == []


def test_r0_fidelity_and_energy_forms():
    """Pre-committed forms: J(empty,empty)=1, J(A,empty)=0; respect fidelity is
    identity-matched with dropped chapters scoring 0; free fidelity is
    per-original-chapter best match; energy = 1 - mean best-match Jaccard."""
    from artifacts_scripts.gap_detection.roundtrip import (
        fidelity_free,
        fidelity_respect,
        jaccard,
    )

    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, set()) == 0.0
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    mean_r, per_r = fidelity_respect([[1, 2], [3, 4]], {0: [1, 2]})   # ch 1 dropped
    assert per_r == [1.0, 0.0] and mean_r == pytest.approx(0.5)
    mean_f, per_f = fidelity_free([[1, 2], [3, 4]], [[2, 3], [3, 4]])
    assert per_f == [pytest.approx(1 / 3, abs=2e-4), 1.0]   # best match per chapter
    assert mean_f == pytest.approx((1 / 3 + 1.0) / 2, abs=2e-4)
    assert 1.0 - mean_f == pytest.approx(1.0 - (1 / 3 + 1.0) / 2, abs=2e-4)
