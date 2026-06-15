"""Plant-and-recover validation of gap detection on Mathlib theorems (CPU only).

Question (the project's one open pipeline cell): does an EMBEDDING LEVEL-SET detector
recover planted theorem-deletions where a single-decl citation-count density-ratio (M1)
does not? Mirrors the article-domain plant-and-recover (artifacts/gap_detection_eval),
whose one temporally-validated method was level-set components.

Setup (math-core gated; see adapters/lean/build_data.py):
  F = theorems/lemmas  (foreground = dense literature)
  B = defs/inductives  (background = objects/scaffolding) -> the CANDIDATE set

Detectors scored on the SAME candidates:
  M1            single-decl citation Δ = log(structural_users+1) − log(operational_users+1)   [baseline]
  M_emb         embedding kNN Δ = log r_F,k − log r_B,k  (dist to k-th theorem − to k-th def)
  M_emb-levelset  threshold top-frac high-M_emb candidates -> mutual-kNN components (n_eff)

Plant: pick a coherent concept C (a Mathlib object/def/inductive, .Defs-preferred) and
its theorem cluster T(C) = theorems in F citing C; DELETE T(C) from F; re-score.
Ground-truth region = orphaned core (B-candidates near C whose r_F,k grew ≥ ratio).
Controls = un-planted concepts (hypothetical-removal cores). AUC = MWU(region vs control)
under the planted state; recall@K = level-set component overlaps region (IoU ≥ thr).
Null = delete a RANDOM theorem set of matched size (must NOT be flagged -> AUC≈0.5).
Clustered bootstrap over the concept's module for CIs.

Transcript hygiene: only decl NAMES / kinds / counts / scores are emitted — never proof
bodies or statement prose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a gemma build; CPU only.

import numpy as np
import scipy.sparse as sp

from .kernels import EPS, FgNeighborTable, knn_dists, mann_whitney_auc
from .data import Data
from artifacts_scripts.gap_detection.detectors import (
    effective_number, iou, level_set_components, mutual_knn_graph)

K_NN = 10                 # kNN scale for densities and orphan cores
CORE_RATIO = 1.15         # orphan-core growth factor (article radial default)
TOPFRAC = 0.05            # level-set selection budget
IOU_THR = 0.10            # recall@K IoU threshold (article default)
CLUSTER_LO, CLUSTER_HI = 20, 400   # planted theorem-cluster size budget
GRAPH_CORE_MIN_LOST = 3            # graph-region: min #operational citers a candidate must lose
GRAPH_CORE_FRAC = 0.5             # graph-region: min fraction of operational support lost


# --------------------------------------------------------------------------- M1
def m1_scores(operational_users: np.ndarray, structural_users: np.ndarray) -> np.ndarray:
    """Single-decl citation Δ = log(structural_users+1) − log(operational_users+1) over candidates."""
    return np.log(structural_users + 1.0) - np.log(operational_users + 1.0)


# --------------------------------------------------------------------------- concept mining
def concept_theorem_cluster(data: Data, c_gated: int) -> np.ndarray:
    """Theorem-citers of concept c (positions within F). T(C) = gated theorems citing C."""
    users = data.subT.getrow(c_gated).indices          # gated rows that cite c
    users = users[data.is_F[users]]                    # keep theorem citers
    return data.f_pos[users]                           # -> positions in F


def coherent_concepts(data: Data, rng: np.random.Generator, n_want: int,
                      cluster_lo: int = CLUSTER_LO, cluster_hi: int = CLUSTER_HI) -> list[dict]:
    """Pick concepts: gated B-decls with a theorem cluster in the size budget.

    Preference: ``.Defs``-module objects / inductives (the 'genuine objects' the brief
    contrasts with templated leaks). Returns dicts with the concept's gated row, F-cluster
    positions, module, kind, name.
    """
    cand_rows = data.B_idx
    # vectorized theorem-user count per B-decl: subT_B rows hold all gated users of each
    # B-decl; restrict columns to theorem rows and row-sum.
    thm_col_mask = data.is_F.astype(np.int8)
    n_thm_per_B = np.asarray(data.subT_B.dot(thm_col_mask)).ravel()  # aligned to B_idx order
    rng_order = rng.permutation(cand_rows.size)
    out: list[dict] = []
    seen_mod = defaultdict(int)
    pool_defs, pool_other = [], []
    for bpos in rng_order:
        j = cand_rows[bpos]
        n_thm = int(n_thm_per_B[bpos])
        if not (cluster_lo <= n_thm <= cluster_hi):
            continue
        mod = str(data.mods[j])
        rec = {"c_gated": int(j), "n_thm": n_thm, "mod": mod,
               "kind": str(data.kinds[j]), "name": str(data.names[j])}
        if data.concept_pref[j]:
            pool_defs.append(rec)
        else:
            pool_other.append(rec)
    # take from the genuine-object pool first, cap 2/module for diversity
    for rec in pool_defs + pool_other:
        if seen_mod[rec["mod"]] >= 2:
            continue
        seen_mod[rec["mod"]] += 1
        out.append(rec)
        if len(out) >= n_want:
            break
    return out


# --------------------------------------------------------------------------- planting
def orphan_core(data: Data, near_mask: np.ndarray, r_before: np.ndarray,
                r_after: np.ndarray, ratio: float = CORE_RATIO) -> np.ndarray:
    """Candidate positions (in B) flagged as orphaned: r_F,k grew >= ratio, within near_mask."""
    grew = (r_after / np.maximum(r_before, EPS)) >= ratio
    return np.where(grew & near_mask)[0]


def region_for_deletion(data: Data, tbl: FgNeighborTable, del_pos_F: np.ndarray,
                        center_row: int | None, ratio: float = CORE_RATIO,
                        mode: str = "embedding") -> np.ndarray:
    """Orphaned-region (positions in B) produced by deleting foreground set del_pos_F.

    mode='embedding' (Lean default): the EMBEDDING orphan core — candidates whose k-th
        nearest-foreground distance grew >= ratio. Requires citation/embedding COUPLING
        (a theorem about a concept is embedding-near it).
    mode='graph': the GRAPH orphan core — candidates that lost a substantial share of their
        operational (foreground) citers. Needed where the call graph and the text-embedding
        space are DECOUPLED (e.g. code: a function that calls a class is not textually similar
        to it), so deleting a concept's callers does not move its embedding neighbourhood and
        the embedding core never forms.
    """
    dmask = np.zeros(data.F_idx.size, dtype=bool)
    dmask[del_pos_F] = True
    if mode == "graph":
        del_rows = data.F_idx[del_pos_F]
        del_mask_full = np.zeros(data.emb.shape[0], dtype=np.int8)
        del_mask_full[del_rows] = 1
        lost_b = np.asarray(data.subT_B.dot(del_mask_full)).ravel()      # #deleted oper-citers / B
        oper_b = data.operational_users[data.B_idx].astype(np.float64)
        frac_lost = lost_b / np.maximum(oper_b, 1.0)
        region = np.where((lost_b >= GRAPH_CORE_MIN_LOST) & (frac_lost >= GRAPH_CORE_FRAC))[0]
        return region, dmask
    r_before = tbl.r_F_intact
    r_after = tbl.r_F_k(dmask)
    # 'near' guard: restrict to candidates whose intact nearest-foreground set actually
    # contained a deleted foreground node (the deletion can only orphan those).
    touched = np.isin(tbl.idx, del_pos_F).any(axis=1)
    return orphan_core(data, touched, r_before, r_after, ratio), dmask


# --------------------------------------------------------------------------- detector scoring
def score_all(data: Data, tbl: FgNeighborTable, r_B: np.ndarray,
              dmask: np.ndarray) -> dict[str, np.ndarray]:
    """All three detector score vectors over the B candidate set, given a deletion mask."""
    r_F = tbl.r_F_k(dmask)
    delta_emb = np.log(np.maximum(r_F, EPS)) - np.log(np.maximum(r_B, EPS))
    # M1 with operational_users reduced by the deletion (citers that were deleted no longer count)
    del_rows = data.F_idx[dmask]
    thm_after = data.operational_users[data.B_idx].astype(np.float64).copy()
    if del_rows.size:
        # subgraph users(dst)=subT; columns are citers. Count deleted citers per B row.
        del_mask_full = np.zeros(data.emb.shape[0], dtype=np.int8)
        del_mask_full[del_rows] = 1
        # subT_B[b, :] dot del_mask_full = #deleted citers of b
        lost_b = np.asarray(data.subT_B.dot(del_mask_full)).ravel()
        thm_after = np.maximum(thm_after - lost_b, 0.0)
    m1 = np.log(data.structural_users[data.B_idx] + 1.0) - np.log(thm_after + 1.0)
    return {"M1": m1, "M_emb": delta_emb, "_r_F": r_F}


def levelset_recall(scores: np.ndarray, Zb: np.ndarray, region_pos: np.ndarray,
                    top_frac: float = TOPFRAC, iou_thr: float = IOU_THR,
                    topk: int = 3) -> dict:
    """Run level-set components on M_emb; recall@topk via IoU with the region; + AUC support."""
    n = scores.shape[0]
    real_mask = np.ones(n, dtype=bool)
    comps = level_set_components(scores, Zb, real_mask, top_frac=top_frac, k_graph=K_NN)
    region_set = set(map(int, region_pos))
    top = comps[:topk]
    best_iou_top = max((iou(c["members"], region_pos) for c in top), default=0.0)
    best_iou_any = max((iou(c["members"], region_pos) for c in comps), default=0.0)
    # level-set membership score per candidate: n_eff of the component it belongs to
    # (0 if not in any surfaced component) — the region-detection score for AUC.
    ls_score = np.zeros(n, dtype=np.float64)
    for c in comps:
        ls_score[c["members"]] = c["n_eff"]
    # which component (if any) overlaps the region, and how much of region it covers
    cover = 0.0
    if comps and region_set:
        best = max(comps, key=lambda c: len(set(map(int, c["members"])) & region_set))
        cover = len(set(map(int, best["members"])) & region_set) / max(len(region_set), 1)
    return {
        "recall_topk": bool(best_iou_top >= iou_thr),
        "best_iou_topk": float(best_iou_top),
        "best_iou_any": float(best_iou_any),
        "n_components": len(comps),
        "region_cover_best_comp": float(cover),
        "ls_score": ls_score,
    }


# --------------------------------------------------------------------------- one trial
def run_trial(data: Data, tbl: FgNeighborTable, r_B: np.ndarray, concept: dict,
              control_regions: list[np.ndarray], control_dmask_none: bool = True,
              region_mode: str = "embedding", min_region: int = 3) -> dict:
    """Plant concept's cluster, score all detectors, compute region-vs-control AUC + recall."""
    del_pos = concept["cluster"]
    region, dmask = region_for_deletion(data, tbl, del_pos, concept["c_gated"], mode=region_mode)
    if region.size < min_region:
        return {"ok": False, "reason": "region<min", "region_size": int(region.size)}
    scores = score_all(data, tbl, r_B, dmask)
    ls = levelset_recall(scores["M_emb"], data.Zb, region)
    ls01 = levelset_recall(scores["M_emb"], data.Zb, region, top_frac=0.01)  # matched-budget

    # controls: matched negatives = OTHER concepts' orphan cores under their OWN
    # hypothetical deletion, scored in the CURRENT (this-concept-planted) state.
    neg = np.concatenate([c for c in control_regions if c.size]) if control_regions else np.array([], int)
    neg = np.setdiff1d(neg, region)

    res = {"ok": True, "region_size": int(region.size), "n_deleted": int(del_pos.size),
           "mod": concept["mod"], "kind": concept["kind"], "name": concept["name"],
           "auc": {}, "levelset": {k: v for k, v in ls.items() if k != "ls_score"},
           "levelset_tf01": {"recall_topk": ls01["recall_topk"],
                             "best_iou_topk": ls01["best_iou_topk"],
                             "region_cover_best_comp": ls01["region_cover_best_comp"]}}
    for det in ("M1", "M_emb"):
        res["auc"][det] = mann_whitney_auc(scores[det][region], scores[det][neg])
    res["auc"]["M_emb_levelset"] = mann_whitney_auc(ls["ls_score"][region], ls["ls_score"][neg])
    return res


# --------------------------------------------------------------------------- null trial
def _local_random_cluster(data: Data, size: int, rng: np.random.Generator) -> np.ndarray:
    """A spatially-localized but ARBITRARY theorem set: a random theorem + its `size`-1
    nearest theorem neighbours (no concept coherence). The article's hypothetical-removal-
    at-an-untreated-location null — tests whether the signal is concept-coherence or just
    'any local theorem mass was deleted'."""
    seed = int(rng.integers(0, data.F_idx.size))
    d = knn_dists(data.Zf[seed:seed + 1], data.Zf, min(size, data.F_idx.size),
                  return_idx=True)[1][0]
    return d.astype(np.int64)


def run_null(data: Data, tbl: FgNeighborTable, r_B: np.ndarray, size: int,
             control_regions: list[np.ndarray], rng: np.random.Generator,
             mode: str = "scatter", region_mode: str = "embedding", min_region: int = 3) -> dict:
    """Delete a null theorem set of `size`; its orphan core should NOT beat controls.

    mode='scatter'  : random theorems anywhere (machinery null — does the pipeline
                      spuriously flag a non-localized deletion?).
    mode='local'    : a random localized theorem ball, NO concept coherence (the
                      discriminating null — is the signal concept-coherence or just
                      any-local-deletion? if 'local' AUC ≈ plant AUC, M_emb is detecting
                      deletion-of-local-mass, not under-theorization of a real concept).
    """
    if mode == "local":
        del_pos = _local_random_cluster(data, size, rng)
    else:
        del_pos = rng.choice(data.F_idx.size, size=min(size, data.F_idx.size), replace=False)
    region, dmask = region_for_deletion(data, tbl, del_pos, None, mode=region_mode)
    if region.size < min_region:
        return {"ok": False, "region_size": int(region.size), "mode": mode}
    scores = score_all(data, tbl, r_B, dmask)
    ls = levelset_recall(scores["M_emb"], data.Zb, region)
    neg = np.concatenate([c for c in control_regions if c.size]) if control_regions else np.array([], int)
    neg = np.setdiff1d(neg, region)
    out = {"ok": True, "region_size": int(region.size), "n_deleted": int(del_pos.size),
           "mode": mode, "auc": {}}
    for det in ("M1", "M_emb"):
        out["auc"][det] = mann_whitney_auc(scores[det][region], scores[det][neg])
    out["auc"]["M_emb_levelset"] = mann_whitney_auc(ls["ls_score"][region], ls["ls_score"][neg])
    return out


# --------------------------------------------------------------------------- bootstrap
def clustered_ci(values: list[float], groups: list[str], rng: np.random.Generator,
                 B: int = 2000) -> tuple[float, float, float]:
    """Clustered bootstrap over `groups`: resample groups, mean of pooled member values."""
    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    v, groups = v[ok], [g for g, m in zip(groups, ok) if m]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    by = defaultdict(list)
    for val, g in zip(v, groups):
        by[g].append(val)
    keys = list(by.keys())
    point = float(np.mean(v))
    boots = np.empty(B)
    for b in range(B):
        gs = rng.choice(len(keys), size=len(keys), replace=True)
        pool = np.concatenate([by[keys[i]] for i in gs])
        boots[b] = pool.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def lift_ci(plant_vals: list[float], plant_groups: list[str], null_vals: list[float],
            rng: np.random.Generator, B: int = 2000) -> tuple[float, float, float]:
    """Bootstrap CI for (mean plant AUC − mean null AUC): clustered over plant modules,
    iid over null trials (independent samples)."""
    pv = np.asarray(plant_vals, float)
    pok = ~np.isnan(pv)
    pv, pg = pv[pok], [g for g, m in zip(plant_groups, pok) if m]
    nv = np.asarray(null_vals, float)
    nv = nv[~np.isnan(nv)]
    if pv.size == 0 or nv.size == 0:
        return float("nan"), float("nan"), float("nan")
    by = defaultdict(list)
    for val, g in zip(pv, pg):
        by[g].append(val)
    keys = list(by.keys())
    point = float(pv.mean() - nv.mean())
    boots = np.empty(B)
    for b in range(B):
        gs = rng.choice(len(keys), size=len(keys), replace=True)
        pmean = np.concatenate([by[keys[i]] for i in gs]).mean()
        nmean = nv[rng.integers(0, nv.size, nv.size)].mean()
        boots[b] = pmean - nmean
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/lean_gap")
    ap.add_argument("--n-plants", type=int, default=40)
    ap.add_argument("--n-null", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--m-window", type=int, default=256)
    ap.add_argument("--out", default="/tmp/lean_gap/results.json")
    ap.add_argument("--region", choices=["embedding", "graph"], default="embedding",
                    help="orphan-region definition: embedding core (Lean) or graph oper-loss (code)")
    ap.add_argument("--min-region", type=int, default=3, dest="min_region",
                    help="minimum orphaned-region size for a usable plant/null")
    ap.add_argument("--cluster-lo", type=int, default=CLUSTER_LO, dest="cluster_lo",
                    help="min operational-cluster size for a plantable concept (lower for coarse graphs)")
    ap.add_argument("--cluster-hi", type=int, default=CLUSTER_HI, dest="cluster_hi",
                    help="max operational-cluster size for a plantable concept")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    data = Data(args.data)
    print(f"[load] F={data.F_idx.size} B={data.B_idx.size} "
          f"emb={data.emb.shape} in {time.time()-t0:.1f}s", flush=True)

    # background-side fixed quantity: r_B,k for every candidate (k-th nearest def)
    tb = time.time()
    r_B = knn_dists(data.Zb, data.Zb, K_NN, exclude_self=True)[:, K_NN - 1].astype(np.float32)
    print(f"[r_B] computed in {time.time()-tb:.1f}s "
          f"(median {np.median(r_B):.3f})", flush=True)

    # foreground neighbour table (the cached object enabling cheap deletion re-scores)
    tf = time.time()
    tbl = FgNeighborTable(data.Zb, data.Zf, m=args.m_window, k=K_NN)
    print(f"[tbl] FgNeighborTable m={tbl.m} built in {time.time()-tf:.1f}s", flush=True)

    # baseline (intact) detector sanity: report what M1 / M_emb rank at the top
    base = score_all(data, tbl, r_B, np.zeros(data.F_idx.size, dtype=bool))
    for det in ("M1", "M_emb"):
        top = np.argsort(base[det])[::-1][:8]
        print(f"[baseline] top-{det}: "
              + ", ".join(f"{data.names[data.B_idx[i]]}({data.kinds[data.B_idx[i]]})"
                          for i in top), flush=True)

    # ---- select concepts and precompute each concept's cluster + hypothetical core
    concepts = coherent_concepts(data, rng, n_want=args.n_plants * 3 + 40,
                                 cluster_lo=args.cluster_lo, cluster_hi=args.cluster_hi)
    for c in concepts:
        c["cluster"] = concept_theorem_cluster(data, c["c_gated"])
    # filter to concepts whose deletion actually leaves a >=3 orphan core
    usable = []
    for c in concepts:
        reg, _ = region_for_deletion(data, tbl, c["cluster"], c["c_gated"], mode=args.region)
        c["hyp_region"] = reg
        if reg.size >= args.min_region:
            usable.append(c)
        if len(usable) >= args.n_plants:
            break
    print(f"[concepts] {len(usable)} usable plants "
          f"(of {len(concepts)} candidates)", flush=True)

    # all hypothetical regions = control pool (keyed by concept index)
    all_hyp = [c["hyp_region"] for c in usable]

    plant_rows: list[dict] = []
    for ci, c in enumerate(usable):
        ctrls = [r for k, r in enumerate(all_hyp) if k != ci]
        res = run_trial(data, tbl, r_B, c, ctrls, region_mode=args.region, min_region=args.min_region)
        res["concept_idx"] = ci
        plant_rows.append(res)
        if res.get("ok"):
            a = res["auc"]
            print(f"[plant {ci:2d}] {c['name'][:48]:48s} kind={c['kind'][:9]:9s} "
                  f"|T|={res['n_deleted']:3d} reg={res['region_size']:3d} "
                  f"AUC M1={a['M1']:.2f} emb={a['M_emb']:.2f} ls={a['M_emb_levelset']:.2f} "
                  f"recall@3={int(res['levelset']['recall_topk'])} "
                  f"iou={res['levelset']['best_iou_topk']:.2f}", flush=True)

    # ---- null trials: TWO nulls of matched deletion size.
    #   scatter = random theorems anywhere (machinery null);
    #   local   = random localized theorem ball, no concept coherence (discriminating null).
    sizes = [r["n_deleted"] for r in plant_rows if r.get("ok")]
    median_size = int(np.median(sizes)) if sizes else 60
    ctrl_all = [c["hyp_region"] for c in usable]
    null_rows: dict[str, list[dict]] = {"scatter": [], "local": []}
    for mode in ("scatter", "local"):
        for j in range(args.n_null):
            nr = run_null(data, tbl, r_B, median_size, ctrl_all, rng, mode=mode,
                          region_mode=args.region, min_region=args.min_region)
            nr["null_idx"] = j
            null_rows[mode].append(nr)
        nok = [r for r in null_rows[mode] if r.get("ok")]
        print(f"[null:{mode}] {len(nok)}/{args.n_null} deletions yielded a core "
              f"(size~{median_size})", flush=True)
    nulls_scatter = [r for r in null_rows["scatter"] if r.get("ok")]
    nulls_local = [r for r in null_rows["local"] if r.get("ok")]
    nulls_ok = nulls_scatter  # back-compat name; 'local' reported alongside

    # ---- aggregate: AUC point + clustered CI over module, per detector
    ok = [r for r in plant_rows if r.get("ok")]
    groups = [r["mod"] for r in ok]
    summary = {"n_plants_ok": len(ok), "n_null_scatter_ok": len(nulls_scatter),
               "n_null_local_ok": len(nulls_local),
               "median_cluster_size": median_size, "detectors": {}}
    for det in ("M1", "M_emb", "M_emb_levelset"):
        vals = [r["auc"][det] for r in ok]
        pt, lo, hi = clustered_ci(vals, groups, rng)
        rec = None
        rec01 = None
        if det == "M_emb_levelset":
            rec = float(np.mean([r["levelset"]["recall_topk"] for r in ok]))
            rec01 = float(np.mean([r["levelset_tf01"]["recall_topk"] for r in ok]))
        nsc = [r["auc"][det] for r in nulls_scatter]
        nlo = [r["auc"][det] for r in nulls_local]
        null_sc = float(np.nanmean(nsc)) if nsc else float("nan")
        null_lc = float(np.nanmean(nlo)) if nlo else float("nan")
        # de-confounded signal: lift of the planted (coherent) deletion over the
        # LOCAL null floor — isolates concept coherence above the region-definition
        # coupling (any local theorem deletion produces an r_F,k-grown region).
        lift = pt - null_lc
        lpt, llo_, lhi = lift_ci(vals, groups, nlo, rng)
        summary["detectors"][det] = {
            "auc_mean": pt, "auc_ci": [lo, hi], "auc_median": float(np.nanmedian(vals)),
            "null_scatter_auc": null_sc, "null_local_auc": null_lc,
            "lift_over_local_null": lift, "lift_ci": [llo_, lhi],
            "recall_top3_tf05": rec, "recall_top3_tf01": rec01,
            "n_modules": len(set(groups)),
        }
        msg = (f"[AUC] {det:16s} plant={pt:.3f} CI[{lo:.3f},{hi:.3f}] "
               f"median={np.nanmedian(vals):.3f}  null_scatter={null_sc:.3f} "
               f"null_local={null_lc:.3f}  lift={lift:+.3f} CI[{llo_:+.3f},{lhi:+.3f}]")
        if rec is not None:
            msg += f"  recall@3 tf05={rec:.2f}/tf01={rec01:.2f}"
        print(msg, flush=True)

    # example planted concepts (NAMES ONLY) with who-flagged-them
    examples = []
    for r in sorted(ok, key=lambda r: r["auc"]["M_emb_levelset"], reverse=True)[:6]:
        flagged = [d for d in ("M1", "M_emb", "M_emb_levelset")
                   if not np.isnan(r["auc"][d]) and r["auc"][d] >= 0.65]
        examples.append({"name": r["name"], "kind": r["kind"], "mod": r["mod"],
                         "n_deleted": r["n_deleted"], "region_size": r["region_size"],
                         "auc": r["auc"], "flagged_by": flagged,
                         "recall@3": r["levelset"]["recall_topk"]})
    summary["examples"] = examples

    out = {"summary": summary, "plants": plant_rows,
           "nulls_scatter": null_rows["scatter"], "nulls_local": null_rows["local"],
           "config": vars(args), "elapsed_s": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] wrote {args.out} in {time.time()-t0:.1f}s", flush=True)
    print("[examples] top level-set-recovered concepts (names only):", flush=True)
    for e in examples:
        print(f"   {e['name'][:50]:50s} {e['kind'][:9]:9s} flagged_by={e['flagged_by']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
