"""Where does a method's curriculum DEPART from the book's ToC — and is each departure better or worse?

The objective metrics (AMI=fidelity, redund=distinctness) are scalars; this surfaces the actual
chapter-level deviations for human judgement (the 2026-06-03 "Jurafsky isn't a ceiling" question).

Matches each method chapter (centroid of its works in GIST space) to each ToC chapter (GIST embedding of
title+description), then reports: the optimal 1:1 backbone (weakest pairs = strained matches), method
chapters with NO strong ToC home (novel topics — improvements or junk), ToC chapters with no strong method
home (dropped/merged), and many-method-to-one-ToC collisions (a Jurafsky chapter the method split).

  python -m artifacts_scripts.curriculum.deviation_report \
    --run jurafsky_cca_coupling --toc-json artifacts/book_gaps/jurafsky_toc.json \
    --concept-gap artifacts/book_gaps/jurafsky_concept_full/concept_gap.json --field 17
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.util import LOGGER, setup_logging

AD = "artifacts/curriculum"
WEAK = 0.55          # cosine below this = no strong match (deviation worth reading)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--toc-json", required=True, type=Path)
    ap.add_argument("--concept-gap", required=True, type=Path)
    ap.add_argument("--field", default="17")
    args = ap.parse_args()
    setup_logging(log_name="deviation", log_level="INFO", include_time=False)

    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}
    EM = normalize(np.asarray(cloud.embeddings(), dtype=np.float64))

    twi = json.loads((Path(AD) / args.run / "topic_work_ids.json").read_text())
    kw = json.loads((Path(AD) / args.run / "topic_keywords.json").read_text())
    chap_ids = sorted(twi, key=lambda x: int(x))
    cents = np.vstack([EM[[id2idx[w] for w in twi[c] if w in id2idx]].mean(0) for c in chap_ids])
    cents = normalize(cents)
    labels = [", ".join(k for k, _ in kw[c][:6]) for c in chap_ids]

    toc = json.loads(args.toc_json.read_text())["chapters"]
    ttitles = [c.get("title", f"ch{i}") for i, c in enumerate(toc)]
    ttext = [f"{c.get('title','')}. {c.get('description','')}" for c in toc]
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0", device="cpu")
    Te = normalize(np.asarray(st.encode(ttext), dtype=np.float64))

    S = cents @ Te.T                                       # (n_method × n_toc) cosine
    best_t = S.argmax(1); best_t_s = S.max(1)              # each method chapter -> nearest ToC
    best_m = S.argmax(0); best_m_s = S.max(0)              # each ToC chapter -> nearest method chapter
    ri, ci = linear_sum_assignment(-S)                     # optimal 1:1 backbone
    assign = {int(r): int(c) for r, c in zip(ri, ci)}

    print(f"\n===== {args.run}: 1:1 backbone vs Jurafsky ToC (weakest matches first = strained) =====")
    for r in sorted(assign, key=lambda r: S[r, assign[r]]):
        c = assign[r]
        flag = "  <-- STRAINED" if S[r, c] < WEAK else ""
        print(f"  [{S[r,c]:.2f}] M{chap_ids[r]} ({labels[r]})\n         ~ Jurafsky '{ttitles[c]}'{flag}")

    print(f"\n===== METHOD chapters with NO strong ToC home (cos<{WEAK}) — novel topics (better or junk?) =====")
    novel = [r for r in range(len(chap_ids)) if best_t_s[r] < WEAK]
    for r in novel or [None]:
        if r is None:
            print("  (none)"); break
        print(f"  [{best_t_s[r]:.2f}] M{chap_ids[r]} ({labels[r]})  | nearest ToC: '{ttitles[best_t[r]]}'")

    print(f"\n===== Jurafsky chapters with NO strong method home (cos<{WEAK}) — dropped / merged =====")
    dropped = [j for j in range(len(toc)) if best_m_s[j] < WEAK]
    for j in dropped or [None]:
        if j is None:
            print("  (none)"); break
        print(f"  [{best_m_s[j]:.2f}] Jurafsky '{ttitles[j]}'  | nearest method: M{chap_ids[best_m[j]]} ({labels[best_m[j]]})")

    print("\n===== Jurafsky chapters the method SPLIT (>=2 method chapters map here) =====")
    coll = defaultdict(list)
    for r in range(len(chap_ids)):
        coll[best_t[r]].append(r)
    for j, rs in sorted(coll.items()):
        if len(rs) >= 2:
            print(f"  Jurafsky '{ttitles[j]}' <- " + " ; ".join(f"M{chap_ids[r]}({labels[r].split(',')[0]})" for r in rs))


if __name__ == "__main__":
    main()
