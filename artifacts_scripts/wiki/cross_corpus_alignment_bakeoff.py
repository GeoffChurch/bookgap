#!/usr/bin/env python
# artifacts_scripts/wiki/cross_corpus_alignment_bakeoff.py
"""Cross-corpus (wiki<->OpenAlex) alignment bake-off: raw vs CSLS dictionary, and
whether CORAL / CCA second-moment alignment beats it.

Context. ``docs/wikipedia-openalex-shared-space.md`` established (topic-matched
re-test) that after per-corpus mean-matching the wiki<->OA mean separation is
gone (centered logistic AUC ~0.53, no LDA deflation direction recovers it) but a
GENUINE NONLINEAR RESIDUAL persists: covariance mean principal angle ~41 deg
(k=50), cross MMD^2 ~37 sigma above the within-corpus null, and local kNN
same-corpus purity ~0.92 (the two clouds occupy largely *disjoint local regions*
despite a shared mean). The coocmap correspondence re-test, separately, found a
strong shared-encoder cross-corpus dictionary (~0.90-ish mutual-NN-ish quality)
but its baseline was CSLS'd, leaving ambiguous whether the strong dictionary came
from RAW cosine or from the CSLS correction. This script answers two questions:

  (A) Is the strong cross-corpus dictionary RAW or CSLS-corrected cosine?
      Disentangle by reporting both, in isolation, on the SAME mean-matched
      slices: mutual-NN rate + top-20-by-confidence correspondence precision.

  (B) Does CORAL or CCA (second-moment / covariance-capable alignment) beat the
      raw/CSLS mean-matched dictionary? Anchored on the raw/CSLS mutual-NN pairs
      as pseudo-anchors:
        - CORAL (Sun et al.): whiten OA by its covariance, recolor by wiki's
          (Cov_OA^{-1/2} then Cov_wiki^{1/2}; +epsI regularized). Apply to OA.
        - CCA: fit on paired pseudo-anchors, project both corpora into the top-d
          canonical components (d in {32,64,128}).
      Evaluate each vs the raw mean-matched baseline:
        1. cross-corpus NN correspondence precision (top-20 confidence, eyeball),
        2. shift-battery residual: PC-subspace principal angle (k=50), MMD^2 vs
           the within-corpus permutation null, kNN same-corpus purity (k=10).

Interpretation guard. CORAL aligns covariance BY CONSTRUCTION, so the principal
angle dropping is expected/trivial and is NOT the success signal. The real
questions: does kNN same-corpus purity fall TOWARD 0.5 (local support actually
less disjoint), and does NN precision improve? The residual is nonlinear, so
honestly expect partial-at-best.

Setup (matches the coocmap re-test for comparability): NLP-wiki top-3000 (cosine
to the OA-NLP centroid) vs OA-NLP 3000. PER-SLICE mean-match (subtract each
slice's OWN mean, then unit-normalize) -- NOT the full-store centroid (the NLP
slice is non-representative; full-store centering leaves a residual mean ~0.39).
Seed 0. CPU-only, reuses cached embeddings.

Usage:
    uv run python -m artifacts_scripts.wiki.cross_corpus_alignment_bakeoff \
        --wiki-store cache/wiki/store/wiki_400k --n 3000 --seed 0 --append
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from artifacts_scripts.wiki.shared_space_analysis import (
    _normalize,
    load_openalex_nlp,
    load_wikipedia,
)
from artifacts_scripts.wiki.shift_characterization import (
    covariance_comparison,
    knn_purity,
    two_sample_battery,
)


# ---------------------------------------------------------------------------
# slice selection + per-slice mean-match
# ---------------------------------------------------------------------------

def select_slices(
    oa: NDArray, oa_titles: list[str], wiki: NDArray, wiki_titles: list[str],
    n: int,
) -> tuple[NDArray, list[str], NDArray, list[str]]:
    """NLP-wiki top-n (cosine to OA-NLP centroid) and OA-NLP n (closest to own
    centroid for stability) -- identical selection to the coocmap re-test.

    Inputs are raw L2-normalized GIST. The selection uses the *raw* (un-mean-
    matched) cosine-to-centroid, exactly as ``coocmap_correspondence_retest``
    does; the per-slice mean-match is applied AFTER selection (see
    :func:`mean_match`).
    """
    oa_centroid = _normalize(oa.mean(axis=0, keepdims=True))[0]
    wiki_to_oa = wiki @ oa_centroid
    wiki_sel = np.argsort(-wiki_to_oa)[:n]
    wiki_s = np.ascontiguousarray(wiki[wiki_sel], dtype=np.float32)
    wiki_s_titles = [wiki_titles[i] for i in wiki_sel]

    if len(oa) <= n:
        oa_sel = np.arange(len(oa))
    else:
        oa_sel = np.argsort(-(oa @ oa_centroid))[:n]
    oa_s = np.ascontiguousarray(oa[oa_sel], dtype=np.float32)
    oa_s_titles = [oa_titles[i] for i in oa_sel]
    return wiki_s, wiki_s_titles, oa_s, oa_s_titles


def mean_match(X: NDArray) -> NDArray:
    """PER-SLICE mean-match: subtract the slice's OWN mean, then unit-normalize.

    This is the validated within-topic alignment (docs topic-matched re-test):
    the NLP slice is non-representative of the full store, so full-store centering
    leaves a residual mean ~0.39 and keeps the clouds trivially separable. Own-mean
    centering removes the matched-mean offset and is the apples-to-apples baseline
    on top of which CORAL/CCA must demonstrate any improvement.
    """
    return _normalize(X - X.mean(axis=0, keepdims=True))


# ---------------------------------------------------------------------------
# dictionary metrics (Part A) -- torch on CPU, small 3000x3000 matrices
# ---------------------------------------------------------------------------

def csls(sim: torch.Tensor, k: int) -> torch.Tensor:
    """CSLS correction: sim - 0.5*(topk-col-mean + topk-row-mean), k neighbors."""
    mean0 = sim.topk(k, dim=0).values.mean(dim=0, keepdim=True)  # [1, V_tgt]
    mean1 = sim.topk(k, dim=1).values.mean(dim=1, keepdim=True)  # [V_src, 1]
    return sim - 0.5 * (mean0 + mean1)


def mutual_nn_pairs(sim: torch.Tensor) -> tuple[NDArray, NDArray]:
    """Return the (src_idx, tgt_idx) arrays of mutual nearest neighbors under sim
    (src=rows, tgt=cols): src i and tgt j are mutual NN iff argmax_j sim[i,:]=j
    AND argmax_i sim[:,j]=i."""
    fwd = sim.argmax(dim=1)  # [V_src] -> tgt
    bwd = sim.argmax(dim=0)  # [V_tgt] -> src
    V_src = sim.shape[0]
    src_ids = torch.arange(V_src, device=sim.device)
    is_mutual = bwd[fwd] == src_ids
    src = src_ids[is_mutual].cpu().numpy()
    tgt = fwd[is_mutual].cpu().numpy()
    return src, tgt


def mutual_nn_rate(sim: torch.Tensor) -> float:
    src, _ = mutual_nn_pairs(sim)
    return len(src) / sim.shape[0]


def top_confidence_matches(
    sim: torch.Tensor, src_titles: list[str], tgt_titles: list[str], top: int,
) -> list[tuple[float, str, str]]:
    """Top-`top` src->tgt matches ranked by the match's sim value (confidence)."""
    fwd = sim.argmax(dim=1)
    conf = sim[torch.arange(sim.shape[0]), fwd].cpu().numpy()
    fwd_np = fwd.cpu().numpy()
    order = np.argsort(-conf)[:top]
    return [(float(conf[r]), src_titles[r], tgt_titles[fwd_np[r]]) for r in order]


# ---------------------------------------------------------------------------
# Part B alignments
# ---------------------------------------------------------------------------

def _sym_inv_sqrt(C: NDArray, eps: float) -> NDArray:
    """C^{-1/2} for a symmetric PSD covariance C (regularized by +eps*I)."""
    d = C.shape[0]
    C = C + eps * np.eye(d, dtype=C.dtype)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-12, None)
    return (evecs * (1.0 / np.sqrt(evals))) @ evecs.T


def _sym_sqrt(C: NDArray, eps: float) -> NDArray:
    """C^{1/2} for a symmetric PSD covariance C (regularized by +eps*I)."""
    d = C.shape[0]
    C = C + eps * np.eye(d, dtype=C.dtype)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 1e-12, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


def coral_align(
    oa: NDArray, wiki: NDArray, anchor_src_wiki: NDArray, anchor_tgt_oa: NDArray,
    eps: float,
) -> tuple[NDArray, NDArray]:
    """CORAL (Sun et al.): align OA second moments to wiki's, estimated on the
    pseudo-anchor pairs. Whiten OA by its (anchor) covariance, recolor by wiki's
    (anchor) covariance; apply the recoloring transform to ALL OA rows, then
    renormalize. Wiki is left as-is (only the source/OA cloud is moved).

    Returns (wiki_out, oa_out): wiki_out == wiki (unchanged), oa_out is the
    recolored+renormalized OA cloud, so downstream metrics compare the same
    wiki cloud against a covariance-matched OA cloud.

    anchor_src_wiki / anchor_tgt_oa index the mean-matched wiki / OA slices and
    define the matched subsets used to ESTIMATE both covariances (so the
    second-moment alignment is anchored on the discovered correspondence).
    """
    Wa = wiki[anchor_src_wiki]      # matched wiki anchors
    Oa = oa[anchor_tgt_oa]          # matched OA anchors
    # Covariances of the matched subsets (mean-centered within the anchor set).
    Wc = Wa - Wa.mean(axis=0, keepdims=True)
    Oc = Oa - Oa.mean(axis=0, keepdims=True)
    cov_w = (Wc.T @ Wc) / max(len(Wc) - 1, 1)
    cov_o = (Oc.T @ Oc) / max(len(Oc) - 1, 1)
    # Transform: OA -> OA @ cov_o^{-1/2} @ cov_w^{1/2}. Center OA on the OA-anchor
    # mean first, recolor, then add back the wiki-anchor mean so the recolored OA
    # sits where wiki sits (CORAL aligns 2nd moments; mean alignment is the 1st).
    T = _sym_inv_sqrt(cov_o, eps) @ _sym_sqrt(cov_w, eps)
    oa_out = (oa - Oa.mean(axis=0, keepdims=True)) @ T + Wa.mean(axis=0, keepdims=True)
    oa_out = _normalize(oa_out.astype(np.float32))
    return wiki, oa_out


def cca_align(
    oa: NDArray, wiki: NDArray, anchor_src_wiki: NDArray, anchor_tgt_oa: NDArray,
    d: int,
) -> tuple[NDArray, NDArray]:
    """CCA on the paired pseudo-anchors (X_wiki_anchor, X_OA_anchor); project BOTH
    full corpora into the top-d canonical directions, then renormalize.

    Returns (wiki_proj, oa_proj), both (N, d) then L2-normalized so the same NN /
    battery metrics apply. (sklearn's CCA carries its own internal ridge, so no
    explicit regularizer is threaded here.)
    """
    Wa = wiki[anchor_src_wiki]
    Oa = oa[anchor_tgt_oa]
    from sklearn.cross_decomposition import CCA

    n_comp = min(d, Wa.shape[1], len(Wa) - 1)
    cca = CCA(n_components=n_comp, max_iter=1000)
    cca.fit(Wa, Oa)
    wiki_proj = cca.transform(wiki)
    oa_proj = cca.transform(oa)
    wiki_proj = _normalize(np.ascontiguousarray(wiki_proj, dtype=np.float32))
    oa_proj = _normalize(np.ascontiguousarray(oa_proj, dtype=np.float32))
    return wiki_proj, oa_proj


# ---------------------------------------------------------------------------
# shift-battery residual on an aligned (wiki, oa) pair
# ---------------------------------------------------------------------------

def _mmd_z(t: dict) -> float:
    return (
        (t["cross_mmd2"] - t["perm_mmd_mean"]) / t["perm_mmd_std"]
        if t["perm_mmd_std"] > 0 else float("inf")
    )


def shift_residual(
    wiki: NDArray, oa: NDArray, pc_k: int, mmd_n_sub: int, n_perm: int,
    knn_k: int, knn_n_query: int, seed: int,
) -> dict:
    """Principal angle (k=pc_k), MMD^2 vs within null, kNN same-corpus purity.

    Inputs are already aligned + unit-normalized. ``oa`` is treated as group 0,
    ``wiki`` as group 1 (matching the doc's battery convention)."""
    cov = covariance_comparison(oa, wiki, [pc_k])[pc_k]
    ts = two_sample_battery(oa, wiki, mmd_n_sub, n_perm, seed)
    kp = knn_purity(oa, wiki, knn_k, knn_n_query, seed)
    return {
        "angle_mean": cov["angle_mean"],
        "spec_rel_l1": cov["spec_rel_l1"],
        "mmd_z": _mmd_z(ts),
        "mmd_p": ts["p_mmd"],
        "knn_purity": kp["mean_purity"],
        "knn_excess": kp["excess"],
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def fmt_matches(matches: list[tuple[float, str, str]], judged: list[bool] | None,
                cap: int = 70) -> list[str]:
    lines = []
    for i, (conf, s, t) in enumerate(matches):
        mark = ""
        if judged is not None:
            mark = " ✓" if judged[i] else " ✗"
        lines.append(f"  [{conf:+.3f}]{mark} WIKI: {s[:cap]}")
        lines.append(f"            ->  OA: {t[:cap]}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/wiki_400k"))
    ap.add_argument("--n", type=int, default=3000, help="items per corpus")
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--top", type=int, default=20, help="top-confidence matches to list")
    ap.add_argument("--coral-eps", type=float, default=1e-3, help="covariance ridge for CORAL")
    ap.add_argument("--cca-d", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--pc-k", type=int, default=50, help="PC subspace dim for principal angle")
    ap.add_argument("--mmd-n-sub", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--knn-n-query", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("docs/wikipedia-openalex-shared-space.md"))
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    # CPU-only, deterministic.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    print(f"device={device}")

    # --- load + slice ------------------------------------------------------
    print("Loading OpenAlex NLP slice (raw GIST) ...")
    oa, oa_titles, _oa_ids, oa_fp = load_openalex_nlp()
    oa = _normalize(np.ascontiguousarray(oa, dtype=np.float32))
    print(f"  OA NLP full: {oa.shape}, fp={oa_fp}")
    print("Loading Wikipedia corpus ...")
    wiki, wiki_titles, _wiki_ids, wiki_fp = load_wikipedia(args.wiki_store)
    wiki = _normalize(np.ascontiguousarray(wiki, dtype=np.float32))
    print(f"  Wiki full: {wiki.shape}, fp={wiki_fp}")
    assert oa_fp == wiki_fp, f"fingerprint mismatch OA={oa_fp} wiki={wiki_fp}"

    wiki_s, wiki_s_titles, oa_s, oa_s_titles = select_slices(
        oa, oa_titles, wiki, wiki_titles, args.n
    )
    print(f"SLICE SHAPES: wiki_s={wiki_s.shape}  oa_s={oa_s.shape}")

    # PER-SLICE mean-match (subtract own mean, renormalize).
    Xw = mean_match(wiki_s)
    Xo = mean_match(oa_s)
    # Residual mean norms (should be ~0 for own-mean centering on the sphere).
    print(f"  post-mean-match residual mean norms: wiki={np.linalg.norm(Xw.mean(0)):.4f} "
          f"oa={np.linalg.norm(Xo.mean(0)):.4f}")

    Xw_t = torch.tensor(Xw, device=device)
    Xo_t = torch.tensor(Xo, device=device)

    # =====================================================================
    # PART A -- raw cosine vs CSLS dictionary (in isolation)
    # =====================================================================
    print("\n" + "=" * 72)
    print("PART A: raw cosine vs CSLS dictionary (per-slice mean-matched)")
    sim_raw = Xw_t @ Xo_t.T  # [V_W, V_OA]
    sim_csls = csls(sim_raw, args.csls_k)

    raw_mnn = mutual_nn_rate(sim_raw)
    csls_mnn = mutual_nn_rate(sim_csls)
    print(f"  raw  mutual-NN rate = {raw_mnn:.4f}")
    print(f"  csls mutual-NN rate = {csls_mnn:.4f}")

    raw_top = top_confidence_matches(sim_raw, wiki_s_titles, oa_s_titles, args.top)
    csls_top = top_confidence_matches(sim_csls, wiki_s_titles, oa_s_titles, args.top)

    print("\n--- RAW top-20 by confidence ---")
    for ln in fmt_matches(raw_top, None):
        print(ln)
    print("\n--- CSLS top-20 by confidence ---")
    for ln in fmt_matches(csls_top, None):
        print(ln)

    # Pseudo-anchors from each dictionary (for Part B).
    raw_src, raw_tgt = mutual_nn_pairs(sim_raw)
    csls_src, csls_tgt = mutual_nn_pairs(sim_csls)
    print(f"\n  raw  pseudo-anchors (mutual-NN pairs): {len(raw_src)}")
    print(f"  csls pseudo-anchors (mutual-NN pairs): {len(csls_src)}")

    # =====================================================================
    # PART B -- CORAL & CCA, anchored on the raw mutual-NN pairs
    # =====================================================================
    print("\n" + "=" * 72)
    print("PART B: CORAL & CCA (anchored on raw mutual-NN pseudo-anchors)")

    # Baseline residual (raw mean-matched, no alignment).
    print("  [baseline] raw mean-matched shift residual ...")
    base_res = shift_residual(
        Xw, Xo, args.pc_k, args.mmd_n_sub, args.n_perm, args.knn_k,
        args.knn_n_query, args.seed,
    )
    print(f"    angle(k={args.pc_k})={base_res['angle_mean']:.1f}  "
          f"mmd_z={base_res['mmd_z']:.1f}  knn_purity={base_res['knn_purity']:.4f}")

    # CORAL. Arg order: coral_align(oa, wiki, anchor_src_wiki, anchor_tgt_oa).
    print("  [CORAL] aligning OA second moments to wiki ...")
    wiki_coral, oa_coral = coral_align(Xo, Xw, raw_src, raw_tgt, args.coral_eps)
    sim_coral = torch.tensor(wiki_coral, device=device) @ torch.tensor(oa_coral, device=device).T
    coral_mnn = mutual_nn_rate(sim_coral)
    coral_top = top_confidence_matches(sim_coral, wiki_s_titles, oa_s_titles, args.top)
    coral_res = shift_residual(
        wiki_coral, oa_coral, args.pc_k, args.mmd_n_sub, args.n_perm, args.knn_k,
        args.knn_n_query, args.seed,
    )
    print(f"    mutual-NN={coral_mnn:.4f}  angle(k={args.pc_k})={coral_res['angle_mean']:.1f}  "
          f"mmd_z={coral_res['mmd_z']:.1f}  knn_purity={coral_res['knn_purity']:.4f}")

    # CCA sweep.
    cca_results: dict[int, dict] = {}
    for d in args.cca_d:
        print(f"  [CCA d={d}] fitting on raw pseudo-anchors ...")
        wiki_cca, oa_cca = cca_align(Xo, Xw, raw_src, raw_tgt, d)
        sim_cca = torch.tensor(wiki_cca, device=device) @ torch.tensor(oa_cca, device=device).T
        cca_mnn = mutual_nn_rate(sim_cca)
        cca_top = top_confidence_matches(sim_cca, wiki_s_titles, oa_s_titles, args.top)
        cca_res = shift_residual(
            wiki_cca, oa_cca, min(args.pc_k, d), args.mmd_n_sub, args.n_perm,
            args.knn_k, args.knn_n_query, args.seed,
        )
        cca_results[d] = {
            "mnn": cca_mnn, "top": cca_top, "res": cca_res,
            "pc_k": min(args.pc_k, d),
        }
        print(f"    mutual-NN={cca_mnn:.4f}  angle(k={min(args.pc_k, d)})="
              f"{cca_res['angle_mean']:.1f}  mmd_z={cca_res['mmd_z']:.1f}  "
              f"knn_purity={cca_res['knn_purity']:.4f}")

    # =====================================================================
    # summary tables to stdout
    # =====================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  Part A dictionary mutual-NN: raw={raw_mnn:.4f}  csls={csls_mnn:.4f}")
    print(f"  baseline residual: angle={base_res['angle_mean']:.1f}  "
          f"mmd_z={base_res['mmd_z']:.1f}  knn_purity={base_res['knn_purity']:.4f}")
    print(f"  CORAL   residual: angle={coral_res['angle_mean']:.1f}  "
          f"mmd_z={coral_res['mmd_z']:.1f}  knn_purity={coral_res['knn_purity']:.4f}  "
          f"mnn={coral_mnn:.4f}")
    for d, r in cca_results.items():
        print(f"  CCA d={d:3d} residual: angle={r['res']['angle_mean']:.1f}  "
              f"mmd_z={r['res']['mmd_z']:.1f}  knn_purity={r['res']['knn_purity']:.4f}  "
              f"mnn={r['mnn']:.4f}")

    R = {
        "n": args.n, "csls_k": args.csls_k, "seed": args.seed,
        "wiki_store": str(args.wiki_store),
        "wiki_resid_mean": float(np.linalg.norm(Xw.mean(0))),
        "oa_resid_mean": float(np.linalg.norm(Xo.mean(0))),
        "raw_mnn": raw_mnn, "csls_mnn": csls_mnn,
        "raw_top": raw_top, "csls_top": csls_top,
        "n_raw_anchor": len(raw_src), "n_csls_anchor": len(csls_src),
        "base_res": base_res,
        "coral_eps": args.coral_eps, "coral_mnn": coral_mnn,
        "coral_top": coral_top, "coral_res": coral_res,
        "cca": cca_results, "pc_k": args.pc_k,
    }
    report = render_report(R, args)
    if args.append:
        with open(args.out, "a") as f:
            f.write("\n\n" + report)
        print(f"\nAppended section to {args.out}")
    else:
        print("\n--- report (not written; pass --append) ---\n")
        print(report)


def render_report(R: dict, args) -> str:
    """Render the markdown section. Precision estimates are left as TODO markers
    the operator fills in after eyeballing the printed top-20 lists (same-concept
    judgement is not automatable here); the match lists themselves are embedded."""
    L: list[str] = []
    L.append("## Cross-corpus alignment bake-off: raw vs CSLS dictionary, and CORAL / CCA\n")
    L.append(
        "Two prior threads left a gap. (i) The §\"Topic-matched re-test\" battery "
        "established that after **per-slice mean-match** the wiki↔OA mean offset is "
        "gone but a **genuine nonlinear residual** persists (covariance angle ~41°, "
        "MMD² ~37σ, kNN purity ~0.92 — the clouds occupy largely disjoint local "
        "support), and explicitly flagged CORAL/CCA as the next test. (ii) The "
        "coocmap correspondence re-test found a strong shared-encoder cross-corpus "
        "**dictionary**, but its baseline was *CSLS'd*, leaving ambiguous whether the "
        "strong dictionary came from RAW cosine or the CSLS correction. This section "
        "resolves both, on the SAME per-slice-mean-matched slices.\n"
    )
    L.append(
        f"- **Slices.** NLP-wiki top-{R['n']} (cosine to the OA-NLP centroid) vs "
        f"OA-NLP {R['n']} (closest to own centroid), `{R['wiki_store']}`, seed "
        f"{R['seed']}. **Per-slice mean-match**: subtract each slice's OWN mean, "
        f"then unit-normalize (NOT full-store centering — the NLP slice is "
        f"non-representative). Post-mean-match residual mean norms: "
        f"wiki={R['wiki_resid_mean']:.4f}, OA={R['oa_resid_mean']:.4f} (~0 ✓).\n"
    )

    # --- Part A ---
    L.append("### Part A — dictionary: raw cosine vs CSLS, in isolation\n")
    L.append(
        "Cross-corpus NN = argmax over the per-slice-mean-matched cosine "
        "`sim = Xw @ Xo.T`. CSLS subtracts "
        "`0.5·(topk-col-mean + topk-row-mean)`, k="
        f"{R['csls_k']}.\n"
    )
    L.append("| scorer | mutual-NN rate | top-20 precision (same-concept) |")
    L.append("|---|---|---|")
    L.append(f"| raw cosine | {R['raw_mnn']:.4f} | __RAW_PREC__ |")
    L.append(f"| CSLS (k={R['csls_k']}) | {R['csls_mnn']:.4f} | __CSLS_PREC__ |")
    L.append("")
    L.append("#### Top-20 RAW matches (by confidence)\n")
    L.append("```")
    L += fmt_matches(R["raw_top"], None)
    L.append("```")
    L.append("\n#### Top-20 CSLS matches (by confidence)\n")
    L.append("```")
    L += fmt_matches(R["csls_top"], None)
    L.append("```")
    L.append("")
    L.append("__PART_A_VERDICT__\n")

    # --- Part B ---
    L.append("### Part B — CORAL & CCA (anchored on raw mutual-NN pseudo-anchors)\n")
    L.append(
        f"Pseudo-anchors = the {R['n_raw_anchor']} raw mutual-NN pairs (the ~strong "
        f"dictionary). CORAL whitens OA by its anchor covariance and recolors by "
        f"wiki's (`Cov_OA^{{-1/2}}` then `Cov_wiki^{{1/2}}`, ε={R['coral_eps']:g}); "
        f"CCA fits on the paired anchors and projects both corpora into the top-d "
        f"canonical components. **Residual** = principal angle (k={R['pc_k']}), "
        f"cross MMD² σ-gap vs the within-corpus permutation null, and kNN "
        f"same-corpus purity (k={args.knn_k}). Baseline = raw mean-matched (no "
        f"alignment).\n"
    )
    b, c = R["base_res"], R["coral_res"]
    L.append("| alignment | mutual-NN | NN top-20 prec | angle (deg) | MMD² σ-gap | kNN purity (k=10) |")
    L.append("|---|---|---|---|---|---|")
    L.append(
        f"| raw mean-matched (baseline) | {R['raw_mnn']:.4f} | __RAW_PREC__ | "
        f"{b['angle_mean']:.1f} | {b['mmd_z']:.1f} | {b['knn_purity']:.4f} |"
    )
    L.append(
        f"| CORAL | {R['coral_mnn']:.4f} | __CORAL_PREC__ | "
        f"{c['angle_mean']:.1f} | {c['mmd_z']:.1f} | {c['knn_purity']:.4f} |"
    )
    for d, r in R["cca"].items():
        rr = r["res"]
        L.append(
            f"| CCA d={d} | {r['mnn']:.4f} | __CCA{d}_PREC__ | "
            f"{rr['angle_mean']:.1f} (k={r['pc_k']}) | {rr['mmd_z']:.1f} | "
            f"{rr['knn_purity']:.4f} |"
        )
    L.append("")
    L.append(
        f"_Interpretation guard:_ CORAL aligns covariance by construction, so its "
        f"angle dropping below the baseline {b['angle_mean']:.1f}° is **expected and "
        f"not the success signal**. The real questions are whether **kNN purity "
        f"falls toward 0.5** (baseline {b['knn_purity']:.3f}) and whether **NN "
        f"precision improves**.\n"
    )
    L.append("#### Top-20 CORAL matches (by confidence)\n")
    L.append("```")
    L += fmt_matches(R["coral_top"], None)
    L.append("```")
    # Show the best CCA d's matches (largest d by default).
    best_d = max(R["cca"].keys())
    L.append(f"\n#### Top-20 CCA (d={best_d}) matches (by confidence)\n")
    L.append("```")
    L += fmt_matches(R["cca"][best_d]["top"], None)
    L.append("```")
    L.append("")
    L.append("__PART_B_VERDICT__\n")
    L.append("### Verdict\n")
    L.append("__OVERALL_VERDICT__\n")
    L.append(
        f"*(Reproduce: `uv run python -m "
        f"artifacts_scripts.wiki.cross_corpus_alignment_bakeoff --wiki-store "
        f"{R['wiki_store']} --n {R['n']} --csls-k {R['csls_k']} --seed {R['seed']}`.)*\n"
    )
    return "\n".join(L)


if __name__ == "__main__":
    main()
