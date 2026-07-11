"""Precompute birth-tag index for every gated graph node by reconstructing the
FQN set at each version tag <= v4.20.0 (the graph snapshot). Cache to /tmp/lean_gap.

CPU + git only. Each tag's FQN set is reconstructed by namespace-stack scanning of
Mathlib/*.lean at that tag (fqn_reconstruct.collect_tag); birth[node] = earliest tag
whose source contains the node's fully-qualified name. Nodes never matched (~18%:
auto-generated/derived projections) keep the SENTINEL value (== never-present)."""
import os, sys, time
from pathlib import Path
from .fqn_reconstruct import collect_tag
import numpy as np

OUT = Path(os.environ.get("LEAN_GAP_DATA", "/tmp/lean_gap"))
TAGS = ["v4.10.0","v4.11.0","v4.12.0","v4.13.0","v4.14.0","v4.15.0",
        "v4.16.0","v4.17.0","v4.18.0","v4.19.0","v4.20.0"]

d = np.load(OUT / "data.npz", allow_pickle=True)
names = np.array([str(n) for n in d["names"]])
name_to_idx = {n:i for i,n in enumerate(names)}
N = len(names)
birth = np.full(N, len(TAGS), dtype=np.int16)   # default = "after v4.20" sentinel == unresolved
for ti, tag in enumerate(TAGS):
    t0=time.time()
    fq = collect_tag(tag)
    cnt = 0
    for nm in fq:
        j = name_to_idx.get(nm)
        if j is not None and birth[j] == len(TAGS):
            birth[j] = ti
            cnt += 1
    resolved_so_far = int((birth < len(TAGS)).sum())
    print(f"[{tag}] reconstructed={len(fq)} newly-born-in-graph={cnt} "
          f"cumulative-resolved={resolved_so_far}/{N} ({100*resolved_so_far/N:.1f}%) "
          f"in {time.time()-t0:.1f}s", flush=True)

np.savez(OUT / "births.npz", birth=birth, tags=np.array(TAGS, dtype=object))
unresolved = int((birth==len(TAGS)).sum())
print(f"[done] unresolved (never seen as source decl thru v4.20): {unresolved} "
      f"({100*unresolved/N:.1f}%)", flush=True)
