# Modality fusion (M1 + M_emb) — REFUTED: strict substitutes, fusion HURTS

**Tested 2026-06-15** (code-gap triangulation follow-on; full result in
`docs/code-gap-triangulation.md` §Follow-on). Whether *fusing* the graph detector (M1, citation-ratio)
and the embedding detector (M_emb, density-ratio) beats either alone — i.e. whether the two modalities
carry complementary signal. Added a z-scored `M1 + M_emb` fused detector to `core/plant_recover.py`
and ran the graph-region plant on all five structural substrates (Lean, Python, Go, Rust, Bazel).

**Expected:** if complementary, `fused AUC > max(M1, M_emb)`.

**Actual:** fused ≤ M1 in *every* substrate, never exceeding; where M_emb is at/below chance (Go)
fusion **drags M1 down** (0.898 → 0.788).

**Diagnosis:** for a given gap modality the *off*-modality detector is **anti-signal**, not merely
weaker — blending only dilutes the modality-matched detector. The modalities are **strict
substitutes**; the actionable form is detector *selection* by gap modality, not combination. (An
equal-weight fusion; a learned weight would simply collapse to the matched detector, M_emb→0.)

**Salvage:** none for fusion itself. The positive form of this finding is the modality-selection rule
(`docs/code-gap-triangulation.md`).
