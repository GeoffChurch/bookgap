"""Mathlib git-tag TEMPORAL HOLDOUT for the validated Lean gap detector (CPU + git only).

Question (the open pipeline cell): does the graph citation-ratio detector
    M1 = log(def_users+1) - log(thm_users+1)   [high = under-theorized]
PREDICT WHERE FUTURE THEOREMS LAND, or does it inherit the article domain's
snapshot->arrival NEGATIVE (artifacts/gap_detection_eval Track-2: single-snapshot
gaps did NOT forecast arrivals above a growth-matched null)? Lean's dep graph is a
clean oracle -- so we ask whether M1 forecasts where noisy article citations could not.

DESIGN
  Birth tags: for each gated graph node, the earliest version tag (v4.10.0..v4.20.0)
    at which its fully-qualified name appears as a SOURCE decl line (namespace-stack
    reconstruction; see fqn_reconstruct.collect_tag). The decl GRAPH is pinned at
    v4.20.0; "present at T" / "born in window" are read off birth tags. Decls never
    seen as a source line (~18%: auto-generated/derived projections) are UNRESOLVED ->
    treated as never-present (documented limitation; resolution rate reported up front).

  At T:  candidate defs = B-nodes (def/inductive) present at T. M1_at_T uses the gated
    citation subgraph restricted to edges whose BOTH endpoints are present at T (the
    honest approximation the brief flagged: a TRUE earlier graph needs recompile+network,
    which we forbid -- we intersect the v4.20 graph's edges with the vT name set).
    Rank defs by M1_at_T (high = under-theorized at T).

  Outcome (T, v4.20]:  ARRIVAL-ATTACHMENT per candidate def d =
    (a) citation: # NEW theorems (born in window) that cite d in the v4.20 graph;
    (b) namespace: # NEW theorems sharing d's namespace (FQN minus leaf).
    Primary = (a) citation (the clean oracle). (b) reported alongside.

  Score:  do high-M1 defs receive MORE arrival-attachment than a GROWTH-MATCHED null of
    defs matched on (existing-theorem-count at T, total in-degree at T) -- so we are not
    rediscovering "big popular defs grow"? Report fill-ratio (top-M1 vs matched null) and
    AUC(M1 ranks attachment>0), with a CLUSTERED BOOTSTRAP over modules.

  Calibrate FIRST:  a RANDOM-def ranking must give fill-ratio ~ 1.0 / AUC ~ 0.5 before
    trusting any M1 number (the banked lesson: a biased matcher invalidates everything).

Transcript hygiene: only decl NAMES / kinds / counts / scores emitted.
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

DATA = Path(os.environ.get("LEAN_GAP_DATA", "/tmp/lean_gap"))

TAGS = ["v4.10.0", "v4.11.0", "v4.12.0", "v4.13.0", "v4.14.0", "v4.15.0",
        "v4.16.0", "v4.17.0", "v4.18.0", "v4.19.0", "v4.20.0"]
SENTINEL = len(TAGS)  # birth value for unresolved (never seen as source decl)


# --------------------------------------------------------------------------- data
class Data:
    def __init__(self) -> None:
        d = np.load(DATA / "data.npz", allow_pickle=True)
        self.is_F = d["is_F"]
        self.is_B = d["is_B"]
        self.kinds = d["kinds"]
        self.mods = d["mods"]
        self.names = np.array([str(n) for n in d["names"]])
        self.N = self.names.shape[0]
        # gated citation subgraph: CSR src(depends)->dst. column j = users of decl j.
        self.sub = sp.load_npz(DATA / "subgraph.npz").tocsr()
        self.subT = self.sub.T.tocsr()  # subT[j].indices = gated rows that cite j
        if not (DATA / "births.npz").exists():
            raise SystemExit(
                "births.npz missing — run build_lean_gap_births.py first "
                "(reconstructs FQN-by-tag from the local Mathlib git; CPU + git only)."
            )
        b = np.load(DATA / "births.npz", allow_pickle=True)
        self.birth = b["birth"].astype(np.int64)
        # namespace (FQN minus leaf) per node
        self.ns = np.array([n.rsplit(".", 1)[0] if "." in n else "" for n in self.names],
                           dtype=object)


# --------------------------------------------------------------------------- M1 at T
def m1_at_T(data: Data, Ti: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M1, def_users_T, thm_users_T for ALL gated nodes, counting only citers present at T.

    Uses the gated subgraph restricted to edges whose source (citer) is present at T.
    (Target presence is enforced separately when we restrict to candidate defs.)
    """
    present = data.birth <= Ti          # bool over all gated nodes
    citer_present = present
    citer_is_def = (data.kinds == "def") | (data.kinds == "inductive")
    citer_is_thm = (data.kinds == "theorem")
    # subT in COO: row=dst(used), col=src(citer)
    subT_coo = data.subT.tocoo()
    dst = subT_coo.row
    src = subT_coo.col
    keep = citer_present[src]
    dst_k = dst[keep]
    src_k = src[keep]
    def_users = np.zeros(data.N, dtype=np.int64)
    thm_users = np.zeros(data.N, dtype=np.int64)
    np.add.at(def_users, dst_k, citer_is_def[src_k].astype(np.int64))
    np.add.at(thm_users, dst_k, citer_is_thm[src_k].astype(np.int64))
    m1 = np.log(def_users + 1.0) - np.log(thm_users + 1.0)
    return m1, def_users, thm_users


# --------------------------------------------------------------------------- arrivals
def arrival_attachment(data: Data, Ti: int) -> tuple[np.ndarray, np.ndarray]:
    """Per gated node: (citation_attach, namespace_attach) from NEW theorems in (T, v4.20].

    citation_attach[d] = # new theorems (born in window) that cite d in the v4.20 graph.
    namespace_attach[d] = # new theorems whose namespace equals d's namespace.
    """
    new_thm = (data.birth > Ti) & (data.birth < SENTINEL) & (data.is_F)
    new_thm_idx = np.where(new_thm)[0]
    # citation: for each dst, count citers that are new theorems.
    new_thm_mask = new_thm.astype(np.int64)
    subT_coo = data.subT.tocoo()
    dst = subT_coo.row
    src = subT_coo.col
    cit = np.zeros(data.N, dtype=np.int64)
    np.add.at(cit, dst, new_thm_mask[src])
    # namespace: count new theorems per namespace string, then map to each node's ns.
    ns_counts: dict[str, int] = defaultdict(int)
    for i in new_thm_idx:
        ns_counts[data.ns[i]] += 1
    nsa = np.array([ns_counts.get(data.ns[i], 0) for i in range(data.N)], dtype=np.int64)
    return cit, nsa


# --------------------------------------------------------------------------- matching
def decile_bins(x: np.ndarray, nb: int = 10) -> np.ndarray:
    """Bin x into nb quantile bins (ties -> same bin via rank)."""
    r = np.argsort(np.argsort(x))
    return np.minimum((r * nb) // max(len(x), 1), nb - 1)


def growth_matched_null(rng: np.random.Generator, thm_T: np.ndarray, deg_T: np.ndarray,
                        top_idx: np.ndarray, pool_idx: np.ndarray, nb: int = 10) -> np.ndarray:
    """Draw a null set, matched to ``top_idx`` on (thm_users_T, total-in-degree_T) joint
    deciles, sampling WITHOUT replacement from ``pool_idx`` (which excludes top_idx).

    Returns null indices of the same size as top_idx (per-bin best-effort; shortfalls in a
    bin are back-filled from neighbouring bins so |null| == |top| when possible)."""
    tb = decile_bins(thm_T, nb)
    db = decile_bins(deg_T, nb)
    joint = tb * nb + db
    pool_set = set(pool_idx.tolist())
    by_bin: dict[int, list[int]] = defaultdict(list)
    for j in pool_idx:
        by_bin[int(joint[j])].append(int(j))
    for k in by_bin:
        rng.shuffle(by_bin[k])
    want = defaultdict(int)
    for j in top_idx:
        want[int(joint[j])] += 1
    chosen: list[int] = []
    used: set[int] = set()
    for b, cnt in want.items():
        avail = by_bin.get(b, [])
        take = [j for j in avail if j not in used][:cnt]
        for j in take:
            used.add(j)
        chosen.extend(take)
    # back-fill any shortfall from the global remaining pool
    need = len(top_idx) - len(chosen)
    if need > 0:
        rest = [j for j in pool_idx.tolist() if j not in used]
        rng.shuffle(rest)
        for j in rest[:need]:
            used.add(j)
            chosen.append(j)
    return np.array(chosen, dtype=np.int64)


# --------------------------------------------------------------------------- AUC + boot
def mwu_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (pos.size * neg.size))


def auc_rank_vs_label(score: np.ndarray, label: np.ndarray) -> float:
    """AUC that ``score`` ranks ``label`` (bool). = MWU(score[label], score[~label])."""
    return mwu_auc(score[label.astype(bool)], score[~label.astype(bool)])


def clustered_bootstrap_ratio(top_attach: np.ndarray, top_mod: np.ndarray,
                              null_attach: np.ndarray, null_mod: np.ndarray,
                              rng: np.random.Generator, B: int = 2000) -> tuple:
    """Bootstrap CI for fill-ratio = mean(top_attach)/mean(null_attach), resampling MODULES
    (clusters) independently within the top set and the null set each replicate."""
    def by_module(att, mod):
        bm = defaultdict(list)
        for a, m in zip(att, mod):
            bm[m].append(a)
        return bm, list(bm.keys())
    tb, tk = by_module(top_attach, top_mod)
    nb, nk = by_module(null_attach, null_mod)
    eps = 1e-9
    point = float(np.mean(top_attach) / (np.mean(null_attach) + eps))
    boots = np.empty(B)
    for b in range(B):
        tg = rng.choice(len(tk), size=len(tk), replace=True)
        ng = rng.choice(len(nk), size=len(nk), replace=True)
        tm = np.concatenate([tb[tk[i]] for i in tg]).mean()
        nm = np.concatenate([nb[nk[i]] for i in ng]).mean()
        boots[b] = tm / (nm + eps)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- one T
def run_for_T(data: Data, Tname: str, args, rng: np.random.Generator) -> dict:
    Ti = TAGS.index(Tname)
    present = data.birth <= Ti
    new_resolved = (data.birth > Ti) & (data.birth < SENTINEL)

    m1, du_T, tu_T = m1_at_T(data, Ti)
    deg_T = du_T + tu_T  # total in-degree at T (centrality proxy)
    cit_attach, ns_attach = arrival_attachment(data, Ti)

    # candidate defs: B-nodes present at T
    cand = np.where(present & data.is_B)[0]
    # optional mild floor: candidate must already be used by >=1 present decl (drop dead
    # leaves so M1's log-ratio is meaningful); report both.
    used_floor = deg_T[cand] >= args.min_deg
    cand_f = cand[used_floor]

    out: dict = {"T": Tname, "Ti": Ti,
                 "n_candidates_all": int(cand.size),
                 "n_candidates_floor": int(cand_f.size),
                 "min_deg": args.min_deg,
                 "n_new_theorems_window": int((new_resolved & data.is_F).sum()),
                 "n_new_defs_window": int((new_resolved & data.is_B).sum())}

    def eval_block(cset: np.ndarray, attach: np.ndarray, label_name: str) -> dict:
        thm_T_c = tu_T[cset].astype(float)
        deg_T_c = deg_T[cset].astype(float)
        att_c = attach[cset].astype(float)
        m1_c = m1[cset]
        mods_c = data.mods[cset]
        names_c = data.names[cset]
        topfrac = args.topfrac
        n_top = max(int(round(cset.size * topfrac)), 1)
        order = np.argsort(m1_c)[::-1]  # high M1 first
        top_local = order[:n_top]
        top_idx = cset[top_local]
        # ---------- CALIBRATION: random-def "ranking" -> matched null, fill-ratio & AUC
        rand_local = rng.permutation(cset.size)[:n_top]
        rand_idx = cset[rand_local]
        pool_for_rand = np.setdiff1d(cset, rand_idx)
        rand_null = growth_matched_null(rng, tu_T, deg_T, rand_idx, pool_for_rand)
        cal_ratio, cal_lo, cal_hi = clustered_bootstrap_ratio(
            attach[rand_idx].astype(float), data.mods[rand_idx],
            attach[rand_null].astype(float), data.mods[rand_null], rng, B=args.boot)
        # calibration AUC: random scores vs label(attach>0) -> ~0.5
        rand_scores = rng.standard_normal(cset.size)
        cal_auc = auc_rank_vs_label(rand_scores, att_c > 0)
        # ---------- M1: matched null on (thm_T, deg_T); fill-ratio + clustered boot
        pool = np.setdiff1d(cset, top_idx)
        null_idx = growth_matched_null(rng, tu_T, deg_T, top_idx, pool)
        m1_ratio, m1_lo, m1_hi = clustered_bootstrap_ratio(
            attach[top_idx].astype(float), data.mods[top_idx],
            attach[null_idx].astype(float), data.mods[null_idx], rng, B=args.boot)
        # ---------- M1 AUC vs growth-matched-null AUC (centrality-stratified discrimination)
        # AUC that M1 separates attach>0 from attach==0, RESTRICTED within matched strata so
        # it is not just centrality. Implement as: stratify by (thm_T,deg_T) joint decile,
        # compute within-stratum MWU(m1 | attach>0  vs  m1 | attach==0), pool.
        m1_auc = stratified_auc(m1_c, att_c > 0, thm_T_c, deg_T_c, args.nb)
        # raw (unstratified) AUC for reference
        m1_auc_raw = auc_rank_vs_label(m1_c, att_c > 0)
        # example top-M1 defs that DID vs did NOT attract arrivals (names only)
        did = [(names_c[i], int(att_c[i])) for i in top_local if att_c[i] > 0]
        didnot = [(names_c[i], int(att_c[i])) for i in top_local if att_c[i] == 0]
        did.sort(key=lambda t: -t[1])
        return {
            "label": label_name,
            "n_set": int(cset.size), "n_top": int(n_top), "topfrac": topfrac,
            "calibration": {"fill_ratio": cal_ratio, "ci": [cal_lo, cal_hi], "auc": cal_auc},
            "m1": {"fill_ratio": m1_ratio, "ci": [m1_lo, m1_hi],
                   "auc_stratified": m1_auc, "auc_raw": m1_auc_raw,
                   "n_modules_top": int(len(set(data.mods[top_idx].tolist()))),
                   "n_modules_null": int(len(set(data.mods[null_idx].tolist())))},
            "attach_frac_top": float((att_c[top_local] > 0).mean()),
            "attach_frac_all": float((att_c > 0).mean()),
            "mean_attach_top": float(att_c[top_local].mean()),
            "mean_attach_all": float(att_c.mean()),
            "examples_did": [n for n, _ in did[:8]],
            "examples_did_attach": [c for _, c in did[:8]],
            "examples_didnot": [n for n, _ in didnot[:8]],
        }

    out["citation_floor"] = eval_block(cand_f, cit_attach, "citation_floor")
    out["citation_all"] = eval_block(cand, cit_attach, "citation_all")
    out["namespace_floor"] = eval_block(cand_f, ns_attach, "namespace_floor")

    # ---- diagnostics: the mechanism. M1 is a (negated) proxy for existing theorem-count,
    #      which itself drives arrivals (preferential attachment). Record the confound.
    m1c = m1[cand_f]; ac = cit_attach[cand_f].astype(float)
    tuc = tu_T[cand_f].astype(float); dgc = deg_T[cand_f].astype(float)
    def srank(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    mb = decile_bins(m1c, 10)
    out["diagnostics"] = {
        "spearman_m1_attach": srank(m1c, ac),
        "spearman_thmT_attach": srank(tuc, ac),
        "spearman_degT_attach": srank(dgc, ac),
        "spearman_m1_thmT": srank(m1c, tuc),
        "m1_stratAUC_high_to_attach": stratified_auc(m1c, ac > 0, tuc, dgc, args.nb),
        "m1_stratAUC_low_to_attach": stratified_auc(-m1c, ac > 0, tuc, dgc, args.nb),
        "m1_decile_mean_attach": [float(ac[mb == b].mean()) for b in range(10)],
        "m1_decile_frac_attach": [float((ac[mb == b] > 0).mean()) for b in range(10)],
        "m1_decile_mean_thmT": [float(tuc[mb == b].mean()) for b in range(10)],
    }
    return out


def stratified_auc(score: np.ndarray, label: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                   nb: int) -> float:
    """Pooled within-stratum AUC: stratify by joint deciles of (c1,c2); within each stratum
    accumulate MWU U-statistic of score for label vs ~label; return pooled U / pooled pairs.

    This asks whether M1 separates 'will attract arrivals' from 'will not' AMONG defs of
    matched centrality -- the growth-matched discrimination question."""
    b1 = decile_bins(c1, nb); b2 = decile_bins(c2, nb)
    joint = b1 * nb + b2
    lab = label.astype(bool)
    U = 0.0; P = 0.0
    for b in np.unique(joint):
        m = joint == b
        s = score[m]; y = lab[m]
        pos = s[y]; neg = s[~y]
        if pos.size == 0 or neg.size == 0:
            continue
        # tie-aware U via rank-sum
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(pos, neg, alternative="greater").statistic
        U += u; P += pos.size * neg.size
    return float(U / P) if P > 0 else float("nan")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", nargs="+", default=["v4.15.0", "v4.13.0"])
    ap.add_argument("--topfrac", type=float, default=0.10)
    ap.add_argument("--min-deg", type=int, default=1, dest="min_deg")
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="/tmp/lean_gap/temporal_holdout.json")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    data = Data()
    print(f"[load] N={data.N} F={int(data.is_F.sum())} B={int(data.is_B.sum())} "
          f"in {time.time()-t0:.1f}s", flush=True)
    resolved = (data.birth < SENTINEL)
    print(f"[resolve] gated resolved={resolved.mean()*100:.1f}%  "
          f"F={resolved[data.is_F].mean()*100:.1f}%  B={resolved[data.is_B].mean()*100:.1f}%  "
          f"(unresolved -> never-present, documented)", flush=True)

    results = {"resolution": {"all": float(resolved.mean()),
                              "F": float(resolved[data.is_F].mean()),
                              "B": float(resolved[data.is_B].mean())},
               "config": vars(args), "tags": TAGS, "per_T": {}}
    for Tname in args.Ts:
        tT = time.time()
        r = run_for_T(data, Tname, args, rng)
        results["per_T"][Tname] = r
        print(f"\n===== T={Tname}  (cand_floor={r['n_candidates_floor']}  "
              f"new_thm={r['n_new_theorems_window']})  {time.time()-tT:.1f}s =====", flush=True)
        for blk in ("citation_floor", "namespace_floor", "citation_all"):
            e = r[blk]
            cal = e["calibration"]; m1 = e["m1"]
            print(f"[{blk:16s}] CAL fill={cal['fill_ratio']:.2f} "
                  f"CI[{cal['ci'][0]:.2f},{cal['ci'][1]:.2f}] AUC={cal['auc']:.3f}   "
                  f"||  M1 fill={m1['fill_ratio']:.2f} CI[{m1['ci'][0]:.2f},{m1['ci'][1]:.2f}] "
                  f"AUCstrat={m1['auc_stratified']:.3f} AUCraw={m1['auc_raw']:.3f}  "
                  f"top_attach_frac={e['attach_frac_top']:.2f} vs all={e['attach_frac_all']:.2f}",
                  flush=True)
        ce = r["citation_floor"]
        print(f"  [examples DID attract]    {ce['examples_did'][:6]}", flush=True)
        print(f"  [examples did NOT]        {ce['examples_didnot'][:6]}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] wrote {args.out} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
