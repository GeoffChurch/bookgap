"""Spot-check a derived pickle's structural invariants (counts only)."""
import pickle
import sys

import numpy as np

from artifacts_scripts.gap_detection.roundtrip import RT_DIR, derived_reps

name = sys.argv[1]
with (RT_DIR / name).open("rb") as f:
    d = pickle.load(f)
reps, sec_chap = derived_reps(d)
print(name, "| mode", d["mode"], "| chap_order", d["chap_order"],
      "| n_chapters", len(d["chapters"]),
      "| sections", reps.shape[0],
      "| unassigned", d.get("n_unassigned"),
      "| dropped", d.get("n_dropped_chapters"),
      "| n_extent", d.get("n_extent"))
if d["mode"] == "respect":
    labels = d["labels"]
    sizes = [int((labels == c).sum()) for c in d["chap_order"]]
    print("  extent sizes (derived order):", sizes, "| sum+unassigned =",
          sum(sizes) + int((labels == -1).sum()), "=", labels.size)
else:
    sizes = [int((d["labels"] == c).sum()) for c in d["chap_order"]]
    print("  cluster sizes:", sizes, "| sum =", sum(sizes), "=", d["rows"].size)
norms = np.linalg.norm(reps, axis=1)
print("  rep norms min/max: %.4f/%.4f" % (norms.min(), norms.max()))
