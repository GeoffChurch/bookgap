#!/usr/bin/env python3
"""Diagnose ONE book's ordering arm: is the good-edge tau-to-human negative because
the traversal is valid-but-REVERSED (tau flips positive when reversed) or genuinely
scrambled? Prints frontier / good-edge / reversed orders, the matched
book-chapter -> cluster pairs with each order's position, tau for each, and the
consecutive-cosine smoothness of each order (good-edge should be smoothest).

  uv run python -m artifacts_scripts.curriculum.close_loop_order_diag \
      --gap-json artifacts/book_gaps/neuromorphic_gap.json \
      --toc-json artifacts/book_gaps/W4247813589_toc.json \
      --run-a1 artifacts/curriculum/cl_A1_neuromorphic --run-b artifacts/curriculum/cl_B_neuromorphic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.stats import kendalltau

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.close_loop_metrics import _build_F, _chapters, _good_edge_order
from artifacts_scripts.gap_detection.toc_metrics import forward_violations
from bookgap.util import LOGGER, setup_logging


def _consec_cos(order, cents):
    return float(np.mean([cents[order[k]] @ cents[order[k + 1]] for k in range(len(order) - 1)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-json", required=True, type=Path)
    ap.add_argument("--toc-json", required=True, type=Path)
    ap.add_argument("--run-a1", required=True, type=Path)
    ap.add_argument("--run-b", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="order_diag", log_level="INFO", include_time=False)

    gap = json.loads(args.gap_json.read_text())
    toc = json.loads(args.toc_json.read_text())
    subfield, field = gap.get("scope_subfield"), gap.get("scope_field")
    cfg = OmegaConf.structured(CurriculumConfig())
    if subfield:
        cfg.experiment.cloud.filter_pair.pre.subfield = str(subfield)
    if field:
        cfg.experiment.cloud.filter_pair.pre.field = str(field)
    cloud = instantiate(cfg.experiment.cloud)
    id2idx = {w: i for i, w in enumerate(cloud.work_ids())}
    E = np.asarray(cloud.embeddings(), dtype=np.float32)

    A1, B = _chapters(args.run_a1), _chapters(args.run_b)
    K = len(A1)
    kw = json.loads((args.run_a1 / "topic_keywords.json").read_text()) \
        if (args.run_a1 / "topic_keywords.json").exists() else {}

    def cen(c):
        r = [id2idx[w] for w in c if w in id2idx]
        v = E[r].mean(0) if r else np.zeros(E.shape[1], np.float32)
        return v / (np.linalg.norm(v) + 1e-12)
    cents = np.stack([cen(c) for c in A1])

    of = list(range(K))
    og = _good_edge_order(cents)
    ogr = og[::-1]
    fp, gp, grp = ({c: i for i, c in enumerate(o)} for o in (of, og, ogr))

    # matched book chapter -> best-overlap A1 cluster
    As = [set(c) for c in A1]
    chap_titles = [c.get("title", f"ch{c.get('n','?')}") for c in toc["chapters"]]
    pairs = []
    for j, bj in enumerate(set(c) for c in B):
        ov = [len(ai & bj) for ai in As]
        if max(ov) > 0:
            pairs.append((j, int(np.argmax(ov)), max(ov)))
    bk = [p[0] for p in pairs]

    def tau(posmap):
        return float(kendalltau(bk, [posmap[p[1]] for p in pairs])[0]) if len(pairs) >= 3 else float("nan")

    LOGGER.info("book=%s  K=%d clusters  matched pairs=%d", toc.get("title"), K, len(pairs))
    LOGGER.info("frontier order        : %s", of)
    LOGGER.info("good-edge order        : %s", og)
    LOGGER.info("good-edge REVERSED     : %s", ogr)
    LOGGER.info("consec-cos  frontier=%.3f  good-edge=%.3f  (good-edge should be >= frontier)",
                _consec_cos(of, cents), _consec_cos(og, cents))
    # Ground-truth-FREE orientation: the good-edge walk fixes the PATH but not the
    # direction; orient it to agree with the frontier/citation prereq direction.
    # tau(good-edge cluster positions vs frontier positions) < 0 => the chain runs
    # against the prereq gradient => flip. (Uses only frontier, never the human.)
    agree = kendalltau([gp[c] for c in range(K)], [fp[c] for c in range(K)])[0]
    oriented, otag = (gp, "keep") if agree >= 0 else (grp, "FLIP")
    LOGGER.info("good-edge vs frontier direction: tau=%.3f -> orient=%s", agree, otag)
    # Citation-DAG orientation: of the two directions of the good-edge chain, pick the
    # one with LOWER prereq-violation (forward citation-flow) share. Good-edge gives the
    # PATH; the citation DAG picks the DIRECTION. Self-contained (reuses F, no years).
    F = _build_F(args.run_a1, id2idx, K)
    sh_g, sh_r = forward_violations(F, og)["share"], forward_violations(F, ogr)["share"]
    dag_oriented, dtag = (og, "keep") if sh_g <= sh_r else (ogr, "FLIP")
    dgp = {c: i for i, c in enumerate(dag_oriented)}
    LOGGER.info("good-edge prereq share: fwd=%.3f rev=%.3f -> DAG-orient=%s", sh_g, sh_r, dtag)
    LOGGER.info("tau-to-human  frontier=%.3f  good-edge=%.3f  reversed=%.3f  "
                "frontier-orient(%s)=%.3f  DAG-orient(%s)=%.3f",
                tau(fp), tau(gp), tau(grp), otag, tau(oriented), dtag, tau(dgp))
    LOGGER.info("--- matched book chapter -> cluster (pos under each order) ---")
    for j, ci, ov in pairs:
        kws = ", ".join(w for w, _ in (kw.get(str(ci), [])[:5])) if kw else ""
        LOGGER.info("  bk ch%-2d %-42s -> cl%-2d (ov%3d) | frontier#%-2d good#%-2d rev#%-2d | %s",
                    j, chap_titles[j][:42], ci, ov, fp[ci], gp[ci], grp[ci], kws[:60])


if __name__ == "__main__":
    main()
