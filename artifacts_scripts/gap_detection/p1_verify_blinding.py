"""P1 scratch: verify banked judge assignments == recomputed seeded plan, and
that every judge brief embeds exactly the right candidates as A and B (by text
identity against the collected p1_<topic>_<arm>.md). Read-only; no bank writes."""
from pathlib import Path

from artifacts_scripts.gap_detection.run_followups import load_banked
from artifacts_scripts.gap_detection.showdown_p1 import (
    ARMS,
    BRIEFS_DIR,
    PAIRS,
    SHOW_DIR,
    TOPIC_ORDER,
    ab_assignments,
)

banked = load_banked()["task_showdown_p1"]
asn_banked = banked["judge_assignments"]["table"]
asn_re = ab_assignments()

cand = {
    (t, a): (SHOW_DIR / f"p1_{t}_{a}.md").read_text(encoding="utf-8").strip()
    for t in TOPIC_ORDER
    for a in ARMS
}

n_ok = 0
problems: list[str] = []
for t in TOPIC_ORDER:
    for pair in PAIRS:
        for o in (1, 2):
            amap = asn_re[(t, pair)][f"order{o}"]
            for s2 in (1, 2):
                key = f"judge_{t}_{pair[0]}-v-{pair[1]}_o{o}_s{s2}"
                row = asn_banked.get(key)
                if row is None:
                    problems.append(f"{key}: MISSING from banked table")
                    continue
                if row["A"] != amap["A"] or row["B"] != amap["B"]:
                    problems.append(f"{key}: banked {row['A']}/{row['B']} != "
                                    f"recomputed {amap['A']}/{amap['B']}")
                    continue
                brief = (BRIEFS_DIR / f"{key}.md").read_text(encoding="utf-8")
                a_body = brief.split("## Candidate A", 1)[1].split("---", 1)[0]
                b_body = brief.split("## Candidate B", 1)[1]
                # identity check: the A/B slots contain the mapped arm's text
                ok_a = cand[(t, amap["A"])][:2000] in a_body
                ok_b = cand[(t, amap["B"])][:2000] in b_body
                if not (ok_a and ok_b):
                    problems.append(f"{key}: brief text mismatch A={ok_a} B={ok_b}")
                    continue
                n_ok += 1

print(f"verified {n_ok}/36 judge briefs against the seeded plan + candidate text")
print(f"banked n_judgments={banked['judge_assignments']['n_judgments']}, "
      f"seed={banked['judge_assignments']['seed']}")
for p in problems:
    print("PROBLEM:", p)
if not problems:
    print("NO PROBLEMS")
# also: the A/B draw per (topic, pair) for the record
for t in TOPIC_ORDER:
    row = []
    for pair in PAIRS:
        amap = asn_re[(t, pair)]["order1"]
        row.append(f"{pair[0]}-v-{pair[1]}: o1 A={amap['A']}")
    print(t, "|", "; ".join(row))
