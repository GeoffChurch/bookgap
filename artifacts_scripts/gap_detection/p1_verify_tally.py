"""P1 scratch: INDEPENDENT re-tally of the 36 blind verdicts from the raw
verdict.md files + banked assignment table; compare to the banked judging row.
Also: independent recompute of coverage/median-distance for the 6 agent/hybrid
candidates (no bank writes anywhere)."""
import numpy as np

from artifacts_scripts.gap_detection.run_followups import load_banked
from artifacts_scripts.gap_detection.showdown_p1 import (
    ARMS,
    JUDGES_TSV,
    PAIRS,
    TOPIC_ORDER,
    _candidate_reps,
    extract_winner,
    majority_of,
    read_manifest,
)
from artifacts_scripts.gap_detection.toc_metrics import nearest_rep_dist

banked = load_banked()["task_showdown_p1"]
asn = banked["judge_assignments"]["table"]
judging = banked["judging"]
judges = read_manifest(JUDGES_TSV)

mismatches = []
for t in TOPIC_ORDER:
    for pair in PAIRS:
        key0 = f"{t}_{pair[0]}-v-{pair[1]}"
        winners = []
        for o in (1, 2):
            for s2 in (1, 2):
                key = f"judge_{key0}_o{o}_s{s2}"
                rd = judges.get(key)
                arm = None
                if rd is not None and (rd / "verdict.md").is_file():
                    letter = extract_winner((rd / "verdict.md").read_text(encoding="utf-8"))
                    if letter:
                        arm = asn[key][letter]
                winners.append(arm)
                cell_banked = judging["fourway"][key0]["cells"][f"o{o}_s{s2}"]
                if (arm or "INVALID") != cell_banked:
                    mismatches.append(f"{key}: raw={arm} banked={cell_banked}")
        maj = majority_of(winners)
        if maj != judging["fourway"][key0]["majority"]:
            mismatches.append(f"{key0}: majority raw={maj} banked differs")
        if judging["win_matrix"][t][f"{pair[0]}-v-{pair[1]}"] != maj:
            mismatches.append(f"{key0}: win_matrix mismatch")
print("re-tally:", "NO MISMATCHES (36 cells + 9 majorities)" if not mismatches
      else f"{len(mismatches)} MISMATCHES")
for m in mismatches:
    print(" ", m)

# independent coverage/median recompute for the 6 LLM-arm candidates + algo DS
print("\nmetric spot-recompute (coverage@1.0181, median d):")
for t in TOPIC_ORDER:
    for arm in ARMS:
        c = _candidate_reps(t, arm)
        d = nearest_rep_dist(c["Z"], c["reps"])
        cov, med = float((d <= 1.0181).mean()), float(np.median(d))
        b = banked["metrics"][t][arm]
        ok = abs(cov - b["coverage"]) < 5e-4 and abs(med - b["median_nearest_dist"]) < 5e-4
        print(f"  {t}/{arm}: cov {cov:.4f} vs banked {b['coverage']} | "
          f"med {med:.4f} vs {b['median_nearest_dist']} | {'OK' if ok else 'MISMATCH'}")
