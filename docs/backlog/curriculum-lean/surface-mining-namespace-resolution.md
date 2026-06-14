# Surface mining: resolve within-namespace bare citations

(Index entry pending — `docs/backlog/index.md` was tracked and this session ran under a
no-tracked-file-modifications constraint; add
"surface-mining-namespace-resolution — recover bare within-namespace lemma citations for
the surface usage distribution" on next index touch.)

**Context.** `curriculum_lean/usage_difficulty.py` mines capitalized/dotted identifiers
from traced proof text (`cache/lean/surface_idents_v4.20.0.json`). Known recall hole,
documented in `artifacts/lean_frame_v2/usage_difficulty.md` and the 2026-06-10
usage-difficulty handoff: within `namespace Real`, proofs cite `exp_log` BARE (lowercase,
undotted) — lexically indistinguishable from locals, so dropped. Result: median 1.7 mined
idents/proof and per-family surface support of only 5–47% (`s_n%`), thinnest exactly for
self-contained families (`Mathlib.Data.Matroid`: 7%), which are the v2 frame favorites.

**Idea.** The difficulty/usage walk already tracks the `namespace` stack
(`difficulty.extract_module_records`). For each bare lowercase token in a proof, try
resolving `<ns_prefix>.<token>` against the decl-graph name set for every suffix of the
ambient namespace stack (plus `open` declarations if we parse them — the trace's
`commandASTs` include `open` commands), longest prefix first; accept on unique hit. Locals
almost never collide with namespaced decl names, so precision should stay high; recall on
within-namespace API citations goes from ~0% to near-complete. Re-run
`usage_difficulty --rank` and compare: does `Data.Matroid` keep its low surface S_mean once
its bare self-citations are visible, and does `s_n%` even out across families?

**Cheap validation:** the term-level premises of the same theorems give per-theorem ground
truth on which library lemmas the proof used; measure resolution precision/recall against
the term sets (restricted to lemmas that appear textually in any form).
