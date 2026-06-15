"""Within-corpus modality conflict — the decisive test that the gap's MODALITY (set by how
the gap is defined), not the domain, decides the detector. Same Python corpus, same two
detectors, two different perturbations of the SAME concept C:

  STRUCTURAL plant: delete C's graph-citers (functions that CALL C)        -> graph perturbation
  SEMANTIC   plant: delete C's embedding-nearest foreground (functions     -> embedding perturbation
                    textually NEAR C, whether or not they call it)

Because the call graph and the text-embedding are decoupled in code, these are genuinely
different deletions. Metric: how much each detector's score AT the planted concept moves,
Δscore = score_after − score_before (a direct sensitivity measure — no region/control
confound). Expect a clean DIAGONAL: the structural plant moves M1, the semantic plant moves
M_emb; each detector is BLIND to the other modality's perturbation.

    python -m artifacts_scripts.curriculum.gap_recover.modality_conflict \\
        --data cache/code_gap/python/data --label Python
"""
from __future__ import annotations

import argparse

import numpy as np

from artifacts_scripts.curriculum.gap_recover.core.kernels import (
    FgNeighborTable,
    knn_dists,
)
from artifacts_scripts.curriculum.gap_recover.core.plant_recover import (
    K_NN,
    Data,
    coherent_concepts,
    concept_theorem_cluster,
    score_all,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-plants", type=int, default=60)
    ap.add_argument("--cluster-lo", type=int, default=10)
    ap.add_argument("--cluster-hi", type=int, default=400)
    ap.add_argument("--m-window", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    data = Data(a.data)
    r_B = knn_dists(data.Zb, data.Zb, K_NN, exclude_self=True)[:, K_NN - 1].astype(np.float32)
    tbl = FgNeighborTable(data.Zb, data.Zf, m=a.m_window, k=K_NN)
    nF = data.F_idx.size
    base = score_all(data, tbl, r_B, np.zeros(nF, dtype=bool))

    concepts = coherent_concepts(data, rng, a.n_plants * 3,
                                 cluster_lo=a.cluster_lo, cluster_hi=a.cluster_hi)
    rows: dict[tuple[str, str], list[float]] = {
        ("M1", "struct"): [], ("M_emb", "struct"): [], ("M1", "sem"): [], ("M_emb", "sem"): []}
    used = 0
    for c in concepts:
        C = c["c_gated"]
        b = int(data.b_pos[C])
        del_s = concept_theorem_cluster(data, C)              # F positions that CALL C
        if del_s.size == 0:
            continue
        dm_s = np.zeros(nF, dtype=bool); dm_s[del_s] = True
        size = min(int(del_s.size), nF)
        idxF = knn_dists(data.emb[C:C + 1], data.Zf, size, return_idx=True)[1][0].astype(np.int64)
        dm_e = np.zeros(nF, dtype=bool); dm_e[idxF] = True    # F positions NEAR C in embedding
        sc_s = score_all(data, tbl, r_B, dm_s)
        sc_e = score_all(data, tbl, r_B, dm_e)
        rows[("M1", "struct")].append(float(sc_s["M1"][b] - base["M1"][b]))
        rows[("M_emb", "struct")].append(float(sc_s["M_emb"][b] - base["M_emb"][b]))
        rows[("M1", "sem")].append(float(sc_e["M1"][b] - base["M1"][b]))
        rows[("M_emb", "sem")].append(float(sc_e["M_emb"][b] - base["M_emb"][b]))
        used += 1
        if used >= a.n_plants:
            break

    def stat(k: tuple[str, str]) -> tuple[float, float]:
        v = np.array(rows[k]); return float(v.mean()), float(np.median(v))

    print(f"[{a.label or a.data}] modality conflict over {used} concepts "
          f"(mean Δscore at the planted concept; higher = detector SAW the perturbation):")
    print("                       STRUCTURAL plant     SEMANTIC plant")
    for det in ("M1", "M_emb"):
        ms, _ = stat((det, "struct")); me, _ = stat((det, "sem"))
        print(f"   {det:18s}  Δ={ms:+.3f}            Δ={me:+.3f}")
    # diagonal dominance: on-modality move vs off-modality move
    on = stat(("M1", "struct"))[0] + stat(("M_emb", "sem"))[0]
    off = stat(("M1", "sem"))[0] + stat(("M_emb", "struct"))[0]
    print(f"   -> on-modality Δ sum={on:+.3f}  vs off-modality Δ sum={off:+.3f}  "
          f"(expect on >> off)")


if __name__ == "__main__":
    main()
