"""Two-panel method map: lab (synthetic recovery) vs wild (real-corpus fill).

Panel A — detector x gap-type heatmap on the 16-trial synthetic benchmarks.
  Cell = median best rank (annotated; found-rate where the method can miss),
  color = log10 percentile-of-its-own-pool (pools differ: doc detectors ~7.1-7.2k
  candidates, miner 36,199 itemsets — percentile makes them commensurable).
  Hatched grey = never run on that type; ✗ = measured refusal.
Panel B — forest plot of real-gap fill ratios (flagged / growth-matched nulls,
  clustered 95% CI) on wiki-2003 and OA-NLP-2015. Line at 1.0; the lab/wild
  inversion is the story: the lab radial champion (fused) straddles 1.0 in the
  wild; level-sets (lab retrieval ✗) are the only repeated wild positive.

Reads banked rows from artifacts/gap_detection_eval/results.json (read-only).
Output: artifacts/gap_detection_eval/figs/method_map.png
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts" / "gap_detection_eval" / "results.json"
OUT = ROOT / "artifacts" / "gap_detection_eval" / "figs" / "method_map.png"

DOC_POOL = 7_139          # scoped bg candidates (handoff)
MINER_POOL = 36_199       # itemset lattice

d = json.loads(RES.read_text())

# ---------------------------------------------------------------- Panel A data
def med(vals):
    vals = [v for v in vals if v is not None]
    return median(vals) if vals else None

def first_key(row, *names):
    for n in names:
        if isinstance(row, dict) and n in row and row[n] is not None:
            return row[n]
    return None

# fused (task_a, radial only): find the rank field defensively
fused_ranks = []
for t in d["task_a"]["trials"]:
    sc = t.get("scoped", {}).get("fused", {})
    r = first_key(sc, "best_rank_real", "best_rank_all", "best_rank", "rank")
    if r is not None:
        fused_ranks.append(r)

# legacy k-ball (radial only)
leg_hits = [r["legacy"]["first_hit_rank"] for r in d["radial"]]
leg_found = sum(1 for h in leg_hits if h is not None) / len(leg_hits)
leg_med = med(leg_hits)  # among hits

rows = [
    # (label, {type: (median_rank, pool, note)}), None entries -> n/a hatch
    ("fused  scope→r_F→veto", {
        "radial": (med(fused_ranks), DOC_POOL, ""),
        "concept": None, "corridor": None}),
    ("full statistic (z/LCB)", {
        "radial": (med([r["ratio"]["full"]["best_rank"] for r in d["radial"]]), DOC_POOL, ""),
        "concept": (med([r["ratio"]["full"]["best_rank"] for r in d["concept"]]), DOC_POOL, ""),
        "corridor": (med([r["ratio"]["full"]["best_rank"] for r in d["corridor"]]), DOC_POOL, "")}),
    ("pattern miner (scoped)", {
        "radial": (med([r.get("attrib_rank_r08") for r in d["task_p1"]["radial"]]), MINER_POOL,
                   "attrib@80%"),
        "concept": (med([r["miner"]["rank"] for r in d["concept"]]), MINER_POOL, ""),
        "corridor": (med([r.get("attrib_rank_r08") for r in d["task_p1"]["corridor"]]), MINER_POOL,
                     "attrib@80%")}),
    ("voltage (single pair)", {
        "radial": None, "concept": None,
        "corridor": (med([r["voltage"]["best_rank"] for r in d["corridor"]]), DOC_POOL, "")}),
    ("flow-change field (P3a)*", {
        "radial": (1.5, DOC_POOL, ""), "concept": (20.5, DOC_POOL, ""),
        "corridor": (3.5, DOC_POOL, "")}),
    ("level-sets (retrieval)", {
        "radial": ("REF", None, "recall@3=0"), "concept": ("REF", None, "recall@3=0"),
        "corridor": ("REF", None, "recall@3=0")}),
    ("legacy greedy k-ball", {
        "radial": (leg_med, DOC_POOL, f"found {leg_found:.0%}"),
        "concept": None, "corridor": None}),
]

# composite (type-blind, task_t1_comp Part A): median best composite rank per true type
if "task_t1_comp" in d and isinstance(d["task_t1_comp"].get("part_a"), dict):
    trials = d["task_t1_comp"]["part_a"].get("trials", [])
    by_kind: dict[str, list] = {}
    for t in trials:
        # main arbitration rule = "max" (noisy-OR of channel percentiles)
        r = first_key(t.get("rules", {}).get("max", {}), "rank_real", "rank_all")
        if r is not None:
            by_kind.setdefault(t.get("kind", "?"), []).append(r)
    if by_kind:
        rows.insert(1, ("composite (TYPE-BLIND)", {
            ty: ((med(by_kind.get(ty, [])), DOC_POOL, "no oracle")
                 if by_kind.get(ty) else None)
            for ty in ("radial", "concept", "corridor")}))
types = ["radial", "concept", "corridor"]

# ---------------------------------------------------------------- Panel B data
forest = []  # (label, ratio, lo, hi, group)
for corpus, key in (("wiki 2003", "task_t2_wiki"), ("OA NLP 2015", "task_t2_oa")):
    per = d[key]["phase3"]["per_detector"]
    for det in ("levelset", "fused", "miner", "voltage", "voltage_emb", "voltage_cite",
                "random50"):
        if det not in per:
            continue
        f = per[det]["fill"]
        lo = f.get("ci_lo_clustered", f.get("ci_lo"))
        hi = f.get("ci_hi_clustered", f.get("ci_hi"))
        forest.append((f"{det}  ({corpus})", f["ratio"], lo, hi, corpus))

# ---------------------------------------------------------------- draw
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

cmap = plt.get_cmap("RdYlGn_r")
for i, (label, cells) in enumerate(rows):
    for j, ty in enumerate(types):
        cell = cells[ty]
        if cell is None:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="0.92",
                                       hatch="///", edgecolor="0.7"))
            ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center",
                    fontsize=8, color="0.45")
            continue
        val, pool, note = cell
        if val == "REF":
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#f3d6d6",
                                       edgecolor="0.7"))
            ax.text(j + 0.5, i + 0.5, "✗\n" + note, ha="center", va="center", fontsize=7.5)
            continue
        pct = max(val / pool, 1e-4)
        # color scale: log10 pct in [-4, 0] -> [good, bad]
        c = cmap((np.log10(pct) + 4) / 4)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor="0.7"))
        txt = f"{val:.0f}\n({pct:.2%})" + (f"\n{note}" if note else "")
        ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7.5)

ax.set_xlim(0, 3); ax.set_ylim(len(rows), 0)
ax.set_xticks([j + 0.5 for j in range(3)]); ax.set_xticklabels([t.upper() for t in types])
ax.set_yticks([i + 0.5 for i in range(len(rows))])
ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_title("LAB — synthetic recovery (16 trials/type)\nmedian best rank (percentile of own pool)",
             fontsize=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.text(0, len(rows) + 0.35,
        "*flow-change field needs before/after snapshots (change mode, not single-snapshot)",
        fontsize=7, color="0.35")

# forest
ypos = np.arange(len(forest))[::-1]
for y, (label, r, lo, hi, corpus) in zip(ypos, forest):
    col = ("#1a7a3a" if "levelset" in label else
           "#b03030" if "cite" in label else
           "0.45" if "random" in label else "#2a4a8a")
    sig = lo is not None and hi is not None and (lo > 1.0 or hi < 1.0)
    bx.plot([lo, hi], [y, y], color=col, lw=2.2 if sig else 1.2,
            alpha=1.0 if sig else 0.65)
    bx.plot(r, y, "o", color=col, ms=7 if sig else 5)
    bx.text(max(hi, r) + 0.06, y, f"{r:.2f}", va="center", fontsize=8, color=col)
bx.axvline(1.0, color="0.3", lw=0.8, ls="--")
bx.set_yticks(ypos); bx.set_yticklabels([f[0] for f in forest], fontsize=8)
bx.set_xlabel("fill ratio vs growth-matched nulls (clustered 95% CI)")
bx.set_title("WILD — real-gap filling (temporal holdouts)\nbold = CI excludes 1.0", fontsize=10)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)

fig.suptitle("Gap detection — method map: lab champions vs wild champions invert", y=1.0,
             fontsize=12)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
