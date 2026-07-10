#!/usr/bin/env python3
"""Can a generality/centrality ANCHOR recover the 1-bit ORIENTATION of a curriculum?

Background. The greedy good-edge chain recovers the human chapter SEQUENCE as an
UNDIRECTED chain (|tau| ~0.23-0.62; beats citation-frontier 4/4 in the close-loop;
22/25 textbooks >=90th pctile). The only thing missing is the 1-bit orientation:
which end is Chapter 1. Citations cannot recover it (humans order against the
prereq DAG ~half the time). This script tests whether a GENERALITY signal --
computed from chapter/cluster EMBEDDINGS ONLY (NO publication year, NO cluster
size) -- can orient the chain general->specific (Chapter 1 = most general/overview).

The key risk we measure rather than assume away: intros are general AND first, but
CONCLUSIONS are also general (summary) AND last. So generality may be U-SHAPED over
human chapter order (high at both ends), which would defeat a linear "general->
specific" orientation. Experiment A's shape diagnostic on the TRUE human order is
the crux -- it separates the orientation question from the chain-quality confound.

Two generality measures (embeddings L2-normalized; S = chapter x chapter cosine,
diag 0, negatives clipped to 0):
  1. eigen-centrality: |principal eigenvector of S| (np.linalg.eigh, largest eval).
  2. transport-entropy (UOT proxy): row-normalize S -> P; H_i = -sum_j P_ij log P_ij.
Higher = more central / connects more broadly = more general.

Two orientation rules for the (direction-arbitrary) good-edge chain:
  * gradient : choose forward-vs-reversed so corr(position, generality) is more
               negative (generality decreasing along the chain).
  * endpoint : consider ONLY the two chain endpoints; start = the more-general end.
               (Robust to a U-shaped interior: only compares the two ends.)

Experiment A (n~25 harvested ToCs, decisive/well-powered): per-ToC shape diagnostic
+ orientation sign test, pooled.
Experiment B (4 close-loop books, confirmatory): generality on OUR generated
clusters' centroids, oriented, directed tau to the human order. NOTE the close-loop
A1 runs on disk were built with the weighted_bisecting_kmeans tree (metadata.json),
NOT the new linkage_method='average' default; we reuse them and say so.

    CUDA_VISIBLE_DEVICES="" uv run python -m artifacts_scripts.curriculum.intro_orientation
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

BASE = Path(__file__).resolve().parents[2]
TOC_GLOB = str(BASE / "artifacts/book_gaps/*_toc.json")
OUT_JSON = BASE / "artifacts/book_gaps/intro_orientation.json"
MIN_CHAPTERS = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
RAND_SEED = 0
N_RAND_ORIENT = 2000  # random-orientation baseline draws for Experiment B

# Close-loop books -> (gap_json basename, human ToC id). Tree = weighted_bisecting_kmeans
# (the on-disk cl_A1_* runs; see metadata.json), reused as-is. See close_loop_metrics.py.
# (gap-json filename differs for cyberbullying; all carry scope_subfield/scope_field.)
CLOSE_LOOP = {
    "cyberbullying": ("W2898513995_cyberbullying_ai", "W2898513995"),
    "mcs": ("mcs_gap", "W4235013513"),
    "neuromorphic": ("neuromorphic_gap", "W4247813589"),
    "color_image": ("color_image_gap", "W4211017872"),
}


# --------------------------------------------------------------------------- #
# Generality measures (both operate on L2-normalized embeddings)
# --------------------------------------------------------------------------- #
def _affinity(E: np.ndarray) -> np.ndarray:
    """S = chapter x chapter cosine, diagonal 0, negatives clipped to 0."""
    S = E @ E.T
    np.fill_diagonal(S, 0.0)
    return np.clip(S, 0.0, None)


def eigen_centrality(E: np.ndarray) -> np.ndarray:
    """|principal eigenvector of S| (largest-eigenvalue vector). Higher = more central."""
    S = _affinity(E)
    w, V = np.linalg.eigh(S)            # ascending eigenvalues
    v = np.abs(V[:, -1])               # principal eigenvector, made non-negative
    return v


def transport_entropy(E: np.ndarray) -> np.ndarray:
    """Row-normalize S -> P; H_i = -sum_j P_ij log P_ij. Higher = connects broadly."""
    S = _affinity(E)
    rs = S.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    P = S / rs
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, P * np.log(P), 0.0)
    return -terms.sum(1)


GEN_MEASURES = {"eigen": eigen_centrality, "entropy": transport_entropy}


# --------------------------------------------------------------------------- #
# Good-edge chain (reuse of close_loop_metrics._good_edge_order; the chain
# direction is arbitrary -- we deliberately do NOT use its broad-node seed for
# orientation, only the undirected PATH it produces).
# --------------------------------------------------------------------------- #
def good_edge_order(cents: np.ndarray) -> list[int]:
    """Greedy good-edge chain over L2-normed centroids (max consecutive cosine).
    Seeds at the broadest node then appends the nearest unused node. The resulting
    PATH is what we use; its direction is treated as arbitrary and re-oriented."""
    n = len(cents)
    if n <= 2:
        return list(range(n))
    S = cents @ cents.T
    start = int(np.argmax((S.sum(1) - 1.0) / (n - 1)))
    order, used = [start], {start}
    while len(order) < n:
        last = order[-1]
        nxt = max((S[last, j], j) for j in range(n) if j not in used)[1]
        order.append(nxt)
        used.add(nxt)
    return order


# --------------------------------------------------------------------------- #
# Orientation rules. Each returns the chain oriented so index 0 = predicted
# Chapter 1 (most general / overview).
# --------------------------------------------------------------------------- #
def orient_gradient(chain: list[int], gen: np.ndarray) -> list[int]:
    """Pick forward vs reversed so corr(chain position, generality) is more negative
    (generality decreasing along the chain => front is most general)."""
    pos = np.arange(len(chain))
    g_fwd = gen[chain]
    # correlation of position vs generality for forward order
    c_fwd = _safe_corr(pos, g_fwd)
    c_rev = _safe_corr(pos, g_fwd[::-1])
    return chain if c_fwd <= c_rev else chain[::-1]


def orient_endpoint(chain: list[int], gen: np.ndarray) -> list[int]:
    """Only compare the two chain ENDPOINTS; start at the more-general end.
    Robust to a U-shaped interior."""
    a, b = chain[0], chain[-1]
    return chain if gen[a] >= gen[b] else chain[::-1]


ORIENT_RULES = {"gradient": orient_gradient, "endpoint": orient_endpoint}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(y) < 1e-12 or np.std(x) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------- #
# Shape diagnostic on the TRUE human chapter order (the crux).
# --------------------------------------------------------------------------- #
def shape_diagnostic(gen: np.ndarray) -> dict:
    """gen is in TRUE human chapter order (index 0 = Chapter 1). Quantify whether
    generality decreases monotonically (anchor works), is U-shaped (conclusion
    confounds it), or is flat."""
    k = len(gen)
    pos = np.arange(k, dtype=float)
    sp = spearmanr(pos, gen)[0]
    pe = pearsonr(pos, gen)[0] if np.std(gen) > 1e-12 else 0.0

    # top-quartile membership (by generality) of Chapter 1 and the LAST chapter
    q75 = np.quantile(gen, 0.75)
    ch1_top = bool(gen[0] >= q75)
    last_top = bool(gen[-1] >= q75)

    # linear vs quadratic fit on standardized position; compare adjusted R^2.
    x = (pos - pos.mean()) / (pos.std() + 1e-12)
    y = gen
    lin_r2 = _adj_r2(np.vstack([np.ones(k), x]).T, y, p=1)
    quad_r2 = _adj_r2(np.vstack([np.ones(k), x, x * x]).T, y, p=2)
    # sign of the quadratic coefficient: >0 = convex = U-shape (high at both ends).
    coef = np.linalg.lstsq(np.vstack([np.ones(k), x, x * x]).T, y, rcond=None)[0]
    quad_sign = float(coef[2])

    # classify shape. U-shape: quadratic clearly wins AND is convex AND both ends high.
    # monotone: |spearman| reasonably large and negative. else flat.
    if quad_r2 > lin_r2 + 0.05 and quad_sign > 0 and (ch1_top and last_top):
        shape = "U"
    elif sp <= -0.30:
        shape = "monotone_decreasing"
    elif sp >= 0.30:
        shape = "monotone_increasing"
    else:
        shape = "flat"

    return {
        "spearman": _round(sp), "pearson": _round(pe),
        "ch1_top_quartile": ch1_top, "last_top_quartile": last_top,
        "lin_adj_r2": _round(lin_r2), "quad_adj_r2": _round(quad_r2),
        "quad_coef_sign": _round(quad_sign), "shape": shape,
    }


def _adj_r2(X: np.ndarray, y: np.ndarray, p: int) -> float:
    n = len(y)
    if n <= p + 1 or np.std(y) < 1e-12:
        return 0.0
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-18:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def _round(x, nd=3):
    try:
        return round(float(x), nd)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Experiment A
# --------------------------------------------------------------------------- #
def directed_tau_for_oriented(oriented: list[int], k: int) -> float:
    """oriented is a permutation of chapter indices, index 0 = predicted Ch1.
    Build the predicted RANK of each chapter and tau it against the true order [0..k-1]."""
    rank = np.empty(k, dtype=int)
    for pos, ch in enumerate(oriented):
        rank[ch] = pos
    # true order is chapters 0,1,...,k-1 at positions 0,1,...,k-1
    return float(kendalltau(np.arange(k), rank)[0])


def run_experiment_a(model) -> dict:
    rows = []
    for f in sorted(glob.glob(TOC_GLOB)):
        slug = Path(f).stem.replace("_toc", "")
        toc = json.loads(Path(f).read_text())
        chs = toc.get("chapters") or []
        k = len(chs)
        if k < MIN_CHAPTERS:
            continue
        texts = [f"{c.get('title','')}. {c.get('description','')}" for c in chs]
        E = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        E = np.asarray(E, dtype=np.float64)

        # good-edge chain over the chapters (undirected path; direction arbitrary).
        chain = good_edge_order(E)
        # undirected ceiling: |tau| of the chain regardless of orientation.
        tau_chain = directed_tau_for_oriented(chain, k)
        undirected_ceiling = abs(tau_chain)

        row = {"slug": slug, "n": k, "undirected_ceiling": _round(undirected_ceiling),
               "measures": {}}
        for mname, mfn in GEN_MEASURES.items():
            gen = mfn(E)  # generality per chapter, in TRUE human order (index=chapter#)
            shape = shape_diagnostic(gen)
            # chain-free orientation bit: is Ch1 (index 0) more general than the LAST
            # chapter? This is the purest 1-bit test -- no good-edge chain involved, so
            # it is uncontaminated by chain quality. (A U-shape makes both ends high and
            # pushes this toward a coin flip.)
            ch1_vs_last = bool(gen[0] > gen[-1])
            orient = {}
            for rname, rfn in ORIENT_RULES.items():
                oriented = rfn(chain, gen)
                dtau = directed_tau_for_oriented(oriented, k)
                orient[rname] = {"directed_tau": _round(dtau),
                                 "correct_direction": bool(dtau > 0)}
            row["measures"][mname] = {"shape": shape, "orient": orient,
                                      "ch1_more_general_than_last": ch1_vs_last,
                                      "corr_pos_gen_spearman": shape["spearman"]}
        rows.append(row)

    # ---- pooled summary: per (measure, rule) ----
    summary = {}
    for mname in GEN_MEASURES:
        shapes = [r["measures"][mname]["shape"]["shape"] for r in rows]
        spear = [r["measures"][mname]["shape"]["spearman"] for r in rows]
        ch1_top = [r["measures"][mname]["shape"]["ch1_top_quartile"] for r in rows]
        last_top = [r["measures"][mname]["shape"]["last_top_quartile"] for r in rows]
        per_rule = {}
        for rname in ORIENT_RULES:
            dtaus = [r["measures"][mname]["orient"][rname]["directed_tau"] for r in rows]
            correct = [r["measures"][mname]["orient"][rname]["correct_direction"] for r in rows]
            # binomial sign test vs 0.5
            n = len(correct)
            n_correct = int(sum(correct))
            per_rule[rname] = {
                "n": n, "n_correct_direction": n_correct,
                "frac_correct_direction": _round(n_correct / n if n else 0.0),
                "mean_directed_tau": _round(np.mean(dtaus)),
                "binom_p_two_sided": _round(_binom_two_sided(n_correct, n)),
            }
        # chain-free 1-bit test pooled
        cf = [r["measures"][mname]["ch1_more_general_than_last"] for r in rows]
        n_cf = len(cf); k_cf = int(sum(cf))
        # orientation correctness stratified by chain quality (gradient rule) -- separates
        # the orientation question from the chain-quality confound.
        strat = {}
        for lbl, lo, hi in [("chain_good_|t|>=0.25", 0.25, 1.01),
                            ("chain_mid_0.10-0.25", 0.10, 0.25),
                            ("chain_noise_|t|<0.10", 0.0, 0.10)]:
            sub = [r for r in rows if lo <= r["undirected_ceiling"] < hi]
            if not sub:
                continue
            c = sum(r["measures"][mname]["orient"]["gradient"]["correct_direction"] for r in sub)
            dt = [r["measures"][mname]["orient"]["gradient"]["directed_tau"] for r in sub]
            strat[lbl] = {"n": len(sub), "n_correct": c,
                          "frac_correct": _round(c / len(sub)),
                          "mean_directed_tau": _round(np.mean(dt))}
        # orientation correctness stratified by shape (gradient rule)
        strat_shape = {}
        for shp in sorted(set(shapes)):
            sub = [r for r in rows if r["measures"][mname]["shape"]["shape"] == shp]
            c = sum(r["measures"][mname]["orient"]["gradient"]["correct_direction"] for r in sub)
            strat_shape[shp] = {"n": len(sub), "n_correct": c, "frac_correct": _round(c / len(sub))}
        summary[mname] = {
            "mean_corr_pos_gen_spearman": _round(np.mean(spear)),
            "frac_ch1_top_quartile": _round(np.mean(ch1_top)),
            "frac_last_top_quartile": _round(np.mean(last_top)),
            "shape_breakdown": {s: shapes.count(s) for s in sorted(set(shapes))},
            "n_tocs": len(rows),
            "chain_free_ch1_gt_last": {"n": n_cf, "n_correct": k_cf,
                                       "frac": _round(k_cf / n_cf),
                                       "binom_p_two_sided": _round(_binom_two_sided(k_cf, n_cf))},
            "orient_by_chain_quality_gradient": strat,
            "orient_by_shape_gradient": strat_shape,
            "per_rule": per_rule,
        }
    return {"per_toc": rows, "summary": summary,
            "mean_undirected_ceiling": _round(np.mean([r["undirected_ceiling"] for r in rows])),
            "n_tocs": len(rows)}


def _binom_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial p-value vs p=0.5 (no scipy.stats.binomtest dependency)."""
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) for i in range(n + 1)]
    total = float(sum(probs))
    obs = probs[k]
    p = sum(pi for pi in probs if pi <= obs + 1e-9) / total
    return min(1.0, p)


# --------------------------------------------------------------------------- #
# Experiment B (close-loop; reuses on-disk cl_A1_* runs + cl_*_metrics.json)
# --------------------------------------------------------------------------- #
def _chapters(run_dir: Path) -> list[list[str]]:
    twi = json.loads((run_dir / "topic_work_ids.json").read_text())
    return [twi[k] for k in sorted(twi, key=lambda x: int(x))]


def run_experiment_b() -> dict:
    """For each close-loop book: load the scoped cloud, build A1 cluster centroids,
    rebuild the good-edge chain + book-chapter->cluster matching (close_loop_metrics
    convention), compute generality on the centroids, orient general->specific, and
    report directed tau to the human chapter order. Compares to the undirected |tau|
    ceiling and a random-orientation baseline.

    NOTE: the cl_A1_* runs use the weighted_bisecting_kmeans tree, not the new
    average-linkage default. We reuse them and flag the tree."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from artifacts_scripts.curriculum.config import CurriculumConfig

    rng = np.random.default_rng(RAND_SEED)
    out = {}
    for book, (gap_name, toc_id) in CLOSE_LOOP.items():
        gap = json.loads((BASE / f"artifacts/book_gaps/{gap_name}.json").read_text())
        run_a1 = BASE / f"artifacts/curriculum/cl_A1_{book}"
        run_b = BASE / f"artifacts/curriculum/cl_B_{book}"
        tree = json.loads((run_a1 / "metadata.json").read_text()).get("linkage_method")

        subfield, field = gap.get("scope_subfield"), gap.get("scope_field")
        cfg = OmegaConf.structured(CurriculumConfig())
        if subfield:
            cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
        if field:
            cfg.experiment.cloud.filter_pair.pre.field = str(field)
        cloud = instantiate(cfg.experiment.cloud)
        id2idx = {w: i for i, w in enumerate(cloud.work_ids())}
        Ecloud = np.asarray(cloud.embeddings(), dtype=np.float32)

        A1, Bk = _chapters(run_a1), _chapters(run_b)
        K = len(A1)

        def centroid(c):
            r = [id2idx[w] for w in c if w in id2idx]
            v = Ecloud[r].mean(0) if r else np.zeros(Ecloud.shape[1], np.float32)
            return v / (np.linalg.norm(v) + 1e-12)
        cents = np.stack([centroid(c) for c in A1]).astype(np.float64)

        chain = good_edge_order(cents)

        # book chapter j -> best-overlap A1 cluster (close_loop_metrics matching)
        As = [set(c) for c in A1]
        pairs = []
        for j, bj in enumerate(set(c) for c in Bk):
            ov = [len(ai & bj) for ai in As]
            if max(ov) > 0:
                pairs.append((j, int(np.argmax(ov))))
        bk_order = [p[0] for p in pairs]

        def tau_for_order(order: list[int]) -> float:
            pos = {c: i for i, c in enumerate(order)}
            if len(pairs) < 3:
                return float("nan")
            return float(kendalltau(bk_order, [pos[p[1]] for p in pairs])[0])

        tau_chain = tau_for_order(chain)
        undirected_ceiling = abs(tau_chain)

        # random-orientation baseline: |E[tau]| of randomly flipped chains is ~the same
        # |tau|, so the meaningful random baseline is the expected SIGNED tau of a coin-
        # flip orientation = 0 (mean of +tau_chain / -tau_chain). Report directly.
        rand_directed = float(np.mean(
            [tau_chain if rng.random() < 0.5 else -tau_chain for _ in range(N_RAND_ORIENT)]))

        per_measure = {}
        for mname, mfn in GEN_MEASURES.items():
            gen = mfn(cents)
            rule_out = {}
            for rname, rfn in ORIENT_RULES.items():
                oriented = rfn(chain, gen)
                dtau = tau_for_order(oriented)
                rule_out[rname] = {"directed_tau": _round(dtau),
                                   "correct_direction": bool(dtau > 0) if dtau == dtau else None}
            per_measure[mname] = rule_out

        out[book] = {
            "toc_id": toc_id, "tree": tree, "n_a1_clusters": K,
            "n_book_chapters": len(Bk), "n_pairs": len(pairs),
            "good_edge_chain": chain,
            "undirected_ceiling": _round(undirected_ceiling),
            "signed_tau_chain_as_built": _round(tau_chain),
            "random_orientation_directed_tau": _round(rand_directed),
            "per_measure": per_measure,
        }
    return out


# --------------------------------------------------------------------------- #
def _print_experiment_a(A: dict) -> None:
    print("\n" + "=" * 78)
    print(f"EXPERIMENT A -- generality anchor over {A['n_tocs']} harvested ToCs "
          f"(k>={MIN_CHAPTERS})")
    print(f"mean undirected |tau| ceiling (good-edge chain) = {A['mean_undirected_ceiling']}")
    print("=" * 78)
    for mname, s in A["summary"].items():
        print(f"\n--- generality = {mname} ---")
        print(f"  mean Spearman(chapter#, generality) = {s['mean_corr_pos_gen_spearman']}  "
              f"(negative => decreasing => anchor would work)")
        print(f"  frac Ch1 in top generality quartile  = {s['frac_ch1_top_quartile']}")
        print(f"  frac LAST in top generality quartile = {s['frac_last_top_quartile']}  "
              f"(high => conclusion confound)")
        print(f"  shape breakdown (n={s['n_tocs']}): {s['shape_breakdown']}")
        cf = s["chain_free_ch1_gt_last"]
        print(f"  chain-FREE 1-bit (Ch1 more general than LAST chapter): "
              f"{cf['n_correct']}/{cf['n']} = {cf['frac']}  (binom p={cf['binom_p_two_sided']})")
        for rname, r in s["per_rule"].items():
            print(f"    orient={rname:8}: correct-direction {r['n_correct_direction']}/{r['n']} "
                  f"= {r['frac_correct_direction']}  mean directed tau={r['mean_directed_tau']}  "
                  f"binom p(2-sided vs 0.5)={r['binom_p_two_sided']}")
        print(f"  [gradient] orientation correctness by CHAIN QUALITY (the key stratification):")
        for lbl, st in s["orient_by_chain_quality_gradient"].items():
            print(f"      {lbl:22} n={st['n']:2}  correct {st['n_correct']}/{st['n']} "
                  f"= {st['frac_correct']}  mean dir-tau={st['mean_directed_tau']}")
        print(f"  [gradient] orientation correctness by SHAPE (on true human order):")
        for shp, st in s["orient_by_shape_gradient"].items():
            print(f"      {shp:22} n={st['n']:2}  correct {st['n_correct']}/{st['n']} = {st['frac_correct']}")


def _print_experiment_b(B: dict) -> None:
    print("\n" + "=" * 78)
    print("EXPERIMENT B -- close-loop (OUR generated clusters; tree per book noted)")
    print("=" * 78)
    hdr = (f"{'book':14}{'tree':24}{'undir|t|':>9}{'rand':>7}"
           f"{'eig.grad':>9}{'eig.end':>9}{'ent.grad':>9}{'ent.end':>9}")
    print(hdr)
    for book, d in B.items():
        pm = d["per_measure"]
        print(f"{book:14}{str(d['tree'])[:23]:24}{d['undirected_ceiling']:>9}"
              f"{d['random_orientation_directed_tau']:>7}"
              f"{pm['eigen']['gradient']['directed_tau']:>9}"
              f"{pm['eigen']['endpoint']['directed_tau']:>9}"
              f"{pm['entropy']['gradient']['directed_tau']:>9}"
              f"{pm['entropy']['endpoint']['directed_tau']:>9}")
    print("\n(directed tau > 0 = correct human direction; undir|t| = ceiling; "
          "rand orientation ~ 0)")


def _provenance() -> str:
    return ("Generated by artifacts_scripts/curriculum/intro_orientation.py -- "
            "regenerate rather than edit by hand.\n"
            f"  CUDA_VISIBLE_DEVICES=\"\" uv run python -m "
            f"artifacts_scripts.curriculum.intro_orientation {' '.join(sys.argv[1:])}".rstrip())


def main() -> None:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    A = run_experiment_a(model)
    _print_experiment_a(A)

    try:
        B = run_experiment_b()
        _print_experiment_b(B)
    except Exception as e:  # close-loop needs the cloud; A is the decisive arm.
        B = {"error": repr(e)}
        print(f"\n[Experiment B skipped: {e!r}]")

    result = {"_provenance": _provenance(),
              "config": {"embed_model": EMBED_MODEL, "min_chapters": MIN_CHAPTERS,
                         "generality_measures": list(GEN_MEASURES),
                         "orientation_rules": list(ORIENT_RULES)},
              "experiment_a": A, "experiment_b": B}
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
