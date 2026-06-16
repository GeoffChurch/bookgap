#!/usr/bin/env python3
"""Does the average-linkage curriculum dendrogram lift CHAIN QUALITY and ORIENTATION?

Average-linkage is now the curriculum default (config.py linkage_method='average';
+0.137 ARI over weighted_bisecting_kmeans on the close-loop recon metric). The
committed orientation prototype (intro_orientation.py, Experiment B) showed the
good-edge chain's 1-bit orientation is recoverable by an eigen-centrality generality
anchor ~89% CONDITIONAL on the undirected chain being good (|tau|>=0.25), but its
Experiment B reused the OLD weighted_bisecting_kmeans clusters. This script regenerates
the 4 close-loop A1 curricula with average-linkage and runs the SAME generality+
orientation pipeline, A/B against the bisecting arm, end to end.

Two effects, kept distinct:
  * CHAIN QUALITY -- undirected |tau| of the good-edge chain to the human order. Does
    the cleaner average-linkage tree raise it, and how many books cross |tau|>=0.25
    (the regime where the eigen anchor works ~89%)?
  * ORIENTATION -- directed tau via the eigen/entropy generality anchor (general->
    specific), and orientation-correct (sign). Does it follow the chain quality?

Procedure (per book, per arm in {bisecting, average}):
  1. Load the scoped cloud ONCE (shared across arms; field-17 is ~6M works ~6min).
  2. Build A1 cluster centroids from the arm's topic_work_ids.json.
  3. good-edge chain (intro_orientation.good_edge_order); match each Track-B book
     chapter -> best-overlap A1 cluster (close_loop_metrics convention); undirected
     |tau|; eigen/entropy generality on centroids; gradient/endpoint orientation;
     directed tau; orientation-correct.
Track B (cl_B_<slug>) is the human ToC populated with gap articles -- it is INDEPENDENT
of the curriculum linkage, so it is shared across both arms (only A1 changes).

  CUDA_VISIBLE_DEVICES="" .conda/bin/python -m artifacts_scripts.curriculum.intro_orientation_avglink
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from artifacts_scripts.curriculum.intro_orientation import (
    GEN_MEASURES,
    ORIENT_RULES,
    good_edge_order,
)

BASE = Path(__file__).resolve().parents[2]
OUT_JSON = BASE / "artifacts/book_gaps/intro_orientation_avglink.json"
BASELINE_JSON = BASE / "artifacts/book_gaps/intro_orientation.json"
RAND_SEED = 0
N_RAND_ORIENT = 2000
TAU_GOOD = 0.25  # the chain-quality threshold above which the eigen anchor works ~89%

# Close-loop books -> (gap_json basename, human ToC id). Same panel as Experiment B.
CLOSE_LOOP = {
    "cyberbullying": ("W2898513995_cyberbullying_ai", "W2898513995"),
    "mcs": ("mcs_gap", "W4235013513"),
    "neuromorphic": ("neuromorphic_gap", "W4247813589"),
    "color_image": ("color_image_gap", "W4211017872"),
}

# Per book, the A1 run dir for each arm. Track B (cl_B_<slug>) is shared.
ARM_DIRS = {
    "bisecting": "cl_A1_{slug}",       # the existing on-disk weighted_bisecting_kmeans runs
    "average": "cl_A1_avg_{slug}",     # regenerated with linkage_method='average' (the default)
}


def _round(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, nd)  # NaN -> None


def _chapters(run_dir: Path) -> list[list[str]]:
    twi = json.loads((run_dir / "topic_work_ids.json").read_text())
    return [twi[k] for k in sorted(twi, key=lambda x: int(x))]


def _eval_arm(A1: list[list[str]], Bk: list[list[str]], Ecloud: np.ndarray,
              id2idx: dict, rng: np.random.Generator) -> dict:
    """Full generality+orientation pipeline for ONE arm's A1 clusters against the
    shared Track-B book chapters. Mirrors intro_orientation.run_experiment_b exactly."""
    K = len(A1)

    def centroid(c: list[str]) -> np.ndarray:
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

    # random-orientation baseline: expected SIGNED tau of a coin-flip orientation ~ 0.
    rand_directed = float(np.mean(
        [tau_chain if rng.random() < 0.5 else -tau_chain for _ in range(N_RAND_ORIENT)]))

    per_measure = {}
    for mname, mfn in GEN_MEASURES.items():
        gen = mfn(cents)
        rule_out = {}
        for rname, rfn in ORIENT_RULES.items():
            oriented = rfn(chain, gen)
            dtau = tau_for_order(oriented)
            rule_out[rname] = {
                "directed_tau": _round(dtau),
                "correct_direction": bool(dtau > 0) if dtau == dtau else None,
            }
        per_measure[mname] = rule_out

    return {
        "n_a1_clusters": K,
        "n_book_chapters": len(Bk),
        "n_pairs": len(pairs),
        "good_edge_chain": chain,
        "undirected_ceiling": _round(undirected_ceiling),
        "signed_tau_chain_as_built": _round(tau_chain),
        "random_orientation_directed_tau": _round(rand_directed),
        "per_measure": per_measure,
    }


def run() -> dict:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from artifacts_scripts.curriculum.config import CurriculumConfig

    rng = np.random.default_rng(RAND_SEED)
    per_book = {}
    skipped = {}

    for book, (gap_name, toc_id) in CLOSE_LOOP.items():
        # Confirm both arms' A1 dirs and the shared Track B exist before loading cloud.
        run_b = BASE / f"artifacts/curriculum/cl_B_{book}"
        arm_paths = {arm: BASE / f"artifacts/curriculum/{tpl.format(slug=book)}"
                     for arm, tpl in ARM_DIRS.items()}
        missing = [str(p) for p in [run_b, *arm_paths.values()]
                   if not (p / "topic_work_ids.json").exists()]
        if missing:
            skipped[book] = f"missing topic_work_ids.json in: {missing}"
            print(f"[skip {book}] {skipped[book]}")
            continue

        gap = json.loads((BASE / f"artifacts/book_gaps/{gap_name}.json").read_text())
        subfield, field = gap.get("scope_subfield"), gap.get("scope_field")
        cfg = OmegaConf.structured(CurriculumConfig())
        if subfield:
            cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
        if field:
            cfg.experiment.cloud.filter_pair.pre.field = str(field)
        print(f"[{book}] loading cloud (subfield={subfield} field={field}) ...", flush=True)
        cloud = instantiate(cfg.experiment.cloud)
        id2idx = {w: i for i, w in enumerate(cloud.work_ids())}
        Ecloud = np.asarray(cloud.embeddings(), dtype=np.float32)

        Bk = _chapters(run_b)
        arms = {}
        for arm, p in arm_paths.items():
            tree = json.loads((p / "metadata.json").read_text()).get("linkage_method")
            res = _eval_arm(_chapters(p), Bk, Ecloud, id2idx, rng)
            res["tree"] = tree
            res["run_dir"] = str(p.relative_to(BASE))
            arms[arm] = res
            print(f"  [{book}/{arm}] tree={tree} K={res['n_a1_clusters']} "
                  f"undir|t|={res['undirected_ceiling']} "
                  f"eig.grad_dtau={res['per_measure']['eigen']['gradient']['directed_tau']}",
                  flush=True)
        per_book[book] = {"toc_id": toc_id, "arms": arms}

    return {"per_book": per_book, "skipped": skipped}


def _eig_grad(arm: dict) -> dict:
    """The default-of-record orientation cell: eigen generality, gradient rule."""
    return arm["per_measure"]["eigen"]["gradient"]


def _summarize(per_book: dict) -> dict:
    """Pooled A/B: mean Delta|tau|, #books crossing TAU_GOOD each arm, #orientation-
    correct each arm (eigen+gradient = the default-of-record cell)."""
    books = list(per_book)
    bis_tau = [per_book[b]["arms"]["bisecting"]["undirected_ceiling"] for b in books]
    avg_tau = [per_book[b]["arms"]["average"]["undirected_ceiling"] for b in books]
    d_tau = [a - b for a, b in zip(avg_tau, bis_tau)]

    def n_cross(taus):
        return int(sum(1 for t in taus if t is not None and t >= TAU_GOOD))

    def n_correct(arm_name):
        return int(sum(1 for b in books
                       if _eig_grad(per_book[b]["arms"][arm_name])["correct_direction"]))

    def n_positive(arm_name):
        return int(sum(1 for b in books
                       if (per_book[b]["arms"][arm_name]["per_measure"]["eigen"]
                           ["gradient"]["directed_tau"] or 0) > 0))

    return {
        "n_books": len(books),
        "tau_good_threshold": TAU_GOOD,
        "mean_undirected_tau": {"bisecting": _round(np.mean(bis_tau)),
                                "average": _round(np.mean(avg_tau))},
        "mean_delta_undirected_tau_avg_minus_bis": _round(np.mean(d_tau)),
        "per_book_delta_undirected_tau": {b: _round(d) for b, d in zip(books, d_tau)},
        "n_books_chain_good": {"bisecting": n_cross(bis_tau), "average": n_cross(avg_tau)},
        "n_orientation_correct_eigen_gradient": {
            "bisecting": n_correct("bisecting"), "average": n_correct("average")},
        "n_directed_tau_positive_eigen_gradient": {
            "bisecting": n_positive("bisecting"), "average": n_positive("average")},
    }


def _ab_table(per_book: dict) -> str:
    """Per-book A/B table: rows = {bisecting, average}, cols = [undir|t|, dir-tau(eigen),
    orient-correct, #chapters]."""
    lines = []
    hdr = (f"{'book':14}{'arm':11}{'tree':26}{'undir|t|':>9}"
           f"{'dir-tau':>9}{'orient':>8}{'#cl':>5}{'#pairs':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for book, d in per_book.items():
        for arm in ("bisecting", "average"):
            a = d["arms"][arm]
            eg = _eig_grad(a)
            oc = eg["correct_direction"]
            mark = "OK" if oc else ("--" if oc is False else "??")
            lines.append(
                f"{book:14}{arm:11}{str(a['tree'])[:25]:26}"
                f"{a['undirected_ceiling']:>9}{eg['directed_tau']:>9}"
                f"{mark:>8}{a['n_a1_clusters']:>5}{a['n_pairs']:>7}")
    return "\n".join(lines)


def _provenance() -> str:
    return ("Generated by artifacts_scripts/curriculum/intro_orientation_avglink.py -- "
            "regenerate rather than edit by hand.\n"
            f"  CUDA_VISIBLE_DEVICES=\"\" .conda/bin/python -m "
            f"artifacts_scripts.curriculum.intro_orientation_avglink "
            f"{' '.join(sys.argv[1:])}".rstrip())


def main() -> None:
    out = run()
    per_book = out["per_book"]

    print("\n" + "=" * 88)
    print("AVERAGE-LINKAGE vs BISECTING -- chain quality & orientation (eigen, gradient rule)")
    print("=" * 88)
    if per_book:
        print(_ab_table(per_book))
        summ = _summarize(per_book)
        print("\n--- pooled ---")
        print(f"  mean undirected |tau|:  bisecting={summ['mean_undirected_tau']['bisecting']}  "
              f"average={summ['mean_undirected_tau']['average']}  "
              f"(mean Delta = {summ['mean_delta_undirected_tau_avg_minus_bis']:+})")
        print(f"  per-book Delta|tau| (avg-bis): {summ['per_book_delta_undirected_tau']}")
        print(f"  #books chain-good (|tau|>={TAU_GOOD}):  "
              f"bisecting={summ['n_books_chain_good']['bisecting']}/{summ['n_books']}  "
              f"average={summ['n_books_chain_good']['average']}/{summ['n_books']}")
        print(f"  #orientation-correct (eigen+gradient): "
              f"bisecting={summ['n_orientation_correct_eigen_gradient']['bisecting']}/{summ['n_books']}  "
              f"average={summ['n_orientation_correct_eigen_gradient']['average']}/{summ['n_books']}")
    else:
        summ = {}
        print("[no books evaluated]")
    if out["skipped"]:
        print(f"\n[skipped books: {out['skipped']}]")

    result = {
        "_provenance": _provenance(),
        "config": {
            "generality_measures": list(GEN_MEASURES),
            "orientation_rules": list(ORIENT_RULES),
            "tau_good_threshold": TAU_GOOD,
            "arm_dirs": ARM_DIRS,
            "note": ("Track B (cl_B_<slug>) is the human ToC populated with gap articles, "
                     "independent of curriculum linkage; shared across arms. Only A1 changes."),
        },
        "per_book": per_book,
        "summary": summ,
        "skipped": out["skipped"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
