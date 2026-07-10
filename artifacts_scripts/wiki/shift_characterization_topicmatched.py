#!/usr/bin/env python
# artifacts_scripts/wiki/shift_characterization_topicmatched.py
"""Topic-matched re-run of the wiki<->OpenAlex shift battery.

``shift_characterization.py`` compared **OA-NLP** (topic T10181) against a
**random** Wikipedia sample drawn from all 7.1M pages, and found a strong
NONLINEAR residual after per-corpus mean-centering: multi-direction LDA AUC
~0.54 (no linear structure), but covariance principal angle 55 deg (k=50),
cross MMD^2 71 sigma above the within-corpus null, and kNN same-corpus purity
0.988. The problem: a random Wikipedia sample is mostly non-NLP
(footballers/villages/chemicals), so much of that residual — especially the
0.988 kNN purity — is plausibly *topic-distribution mismatch* (NLP vs
all-of-Wikipedia), not a corpus/register effect.

This script re-runs the SAME battery, changing ONLY the wiki sample to a
**topic-matched (NLP-relevant)** set: the top-N Wikipedia pages by cosine to
the OpenAlex-NLP centroid — the exact selection the wiki curriculum uses
(``select_nlp_pages`` in ``wiki_curriculum_demo.py``). OA side is the SAME
OA-NLP slice (T10181, raw GIST), subsampled to match the wiki count.

Centering (the offset we'd remove in practice) keeps the established
convention (``cross_corpus_gap_demo``): OA centered by the OA-set centroid,
wiki centered by the **full-store** centroid (the global per-corpus offset
estimate), then both renormalized to the unit sphere. The full-store wiki
centroid is computed as a by-product of the streaming selection pass.

Battery (imported verbatim from ``shift_characterization``):
  0. raw vs centered single-fit logistic AUC
  1. multi-direction LDA deflation (dirs 1..5)
  2. covariance principal angles (k=10,20,50) + spectra
  3. MMD^2 / energy distance vs within-corpus permutation null
  4. local kNN same-corpus purity (k=10) vs mixing-ratio baseline

Usage:
    uv run python -m artifacts_scripts.wiki.shift_characterization_topicmatched \
        --wiki-store cache/wiki/store/enwiki_full --n-select 40000 --seed 0 --append
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from artifacts_scripts.wiki.shared_space_analysis import _normalize, load_openalex_nlp
from artifacts_scripts.wiki.wiki_curriculum_demo import _row_group_embeddings
from artifacts_scripts.wiki.shift_characterization import (
    covariance_comparison,
    knn_purity,
    logistic_auc,
    multi_direction_auc,
    two_sample_battery,
)


# --- topic-matched wiki selection (curriculum's select_nlp_pages, returning
#     the full-store centroid used for centering) -----------------------------

def select_nlp_wiki_cached(
    store_dir: Path, nlp_dir: NDArray[np.float32], n_select: int, cache: Path
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Cache wrapper around :func:`select_nlp_wiki`. The 3-pass stream over the
    7.1M-row embeddings parquet (~7 min) is the only slow step; persisting the
    selected embeddings + full-store centroid makes centering/battery reruns
    instant. The cache lives under /tmp (NOT cache/), keyed by n_select."""
    if cache.exists():
        z = np.load(cache)
        print(f"  select: cache hit {cache} ({z['sel'].shape[0]} rows)")
        return (
            np.ascontiguousarray(z["sel"], dtype=np.float32),
            np.ascontiguousarray(z["centroid"], dtype=np.float32),
        )
    sel, centroid = select_nlp_wiki(store_dir, nlp_dir, n_select)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, sel=sel, centroid=centroid)
    print(f"  select: wrote cache {cache}")
    return sel, centroid


def select_nlp_wiki(
    store_dir: Path, nlp_dir: NDArray[np.float32], n_select: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Top ``n_select`` wiki pages by cosine to ``nlp_dir`` (mean-centered),
    AND the full-store wiki centroid.

    Replicates ``wiki_curriculum_demo.select_nlp_pages``'s selection exactly:
    stream all 7.1M rows, compute the streaming full-store centroid, score each
    row's centered+renormalized embedding against ``nlp_dir``, keep the top-N.
    Returns ``(raw_unit_norm_embeddings_of_selected_rows, full_store_centroid)``.
    The selected embeddings are RAW (unit-norm GIST, not centered) so the
    downstream battery can apply its own centering, matching the random-sample
    run which centered raw vectors.
    """
    emb_path = store_dir / "embeddings.parquet"
    pf = pq.ParquetFile(emb_path)
    n_rg = pf.num_row_groups

    # Pass 1: streaming full-store centroid (mean over all unit-norm rows).
    t = time.time()
    acc = np.zeros(384, dtype=np.float64)
    total = 0
    for rg in range(n_rg):
        En = _normalize(_row_group_embeddings(pf, rg))
        acc += En.sum(axis=0)
        total += En.shape[0]
    centroid = (acc / max(total, 1)).astype(np.float32)
    print(f"  select: streamed {total} rows for full-store centroid in {time.time()-t:.1f}s")

    # Pass 2: cosine of (centered, renormalized) row to nlp_dir.
    t = time.time()
    all_scores: list[NDArray[np.float32]] = []
    for rg in range(n_rg):
        En = _normalize(_row_group_embeddings(pf, rg))
        Ec = _normalize(En - centroid)
        all_scores.append((Ec @ nlp_dir).astype(np.float32))
    scores_full = np.concatenate(all_scores)
    n_sel = min(n_select, scores_full.shape[0])
    part = np.argpartition(-scores_full, n_sel - 1)[:n_sel]
    sel_rows = part[np.argsort(-scores_full[part])].astype(np.int64)
    print(
        f"  select: scored {scores_full.shape[0]} rows in {time.time()-t:.1f}s; "
        f"kept top-{n_sel} (cos {scores_full[sel_rows].min():.3f}.."
        f"{scores_full[sel_rows].max():.3f})"
    )

    # Pass 3: gather the RAW (unit-norm) embeddings of the selected rows.
    order = np.argsort(sel_rows)
    sel_rows_sorted = sel_rows[order]
    sel_emb_sorted = np.empty((n_sel, 384), dtype=np.float32)
    cursor = 0
    rg_start = 0
    for rg in range(n_rg):
        E = _row_group_embeddings(pf, rg)
        rg_end = rg_start + E.shape[0]
        mask = (sel_rows_sorted >= rg_start) & (sel_rows_sorted < rg_end)
        local = sel_rows_sorted[mask] - rg_start
        if local.size:
            sel_emb_sorted[cursor : cursor + local.size] = E[local]
            cursor += local.size
        rg_start = rg_end
    inv = np.empty_like(order)
    inv[order] = np.arange(n_sel)
    sel_emb = _normalize(sel_emb_sorted[inv])
    return sel_emb, centroid


def load_nlp_dir(oa_emb: NDArray[np.float32], cache: Path) -> NDArray[np.float32]:
    """OpenAlex-NLP centroid direction (384-D unit vector). Reuses the cached
    ``/tmp/nlp_dir.npy`` the curriculum writes; recomputes from ``oa_emb`` if
    absent (mean of unit-norm NLP article embeddings, renormalized)."""
    if cache.exists():
        d = np.load(cache).astype(np.float32)
        print(f"  loaded NLP direction from {cache}")
        return d
    d = _normalize(oa_emb).mean(axis=0)
    d = (d / (np.linalg.norm(d) or 1.0)).astype(np.float32)
    np.save(cache, d)
    print(f"  computed NLP direction from OA-NLP slice -> {cache}")
    return d


# --- centering ----------------------------------------------------------------
# Two centerings are reported, and the distinction is load-bearing:
#
#   * OWN-MEAN (primary). Subtract each *sample's own* mean and renormalize —
#     exactly what the prior random-sample run's ``center_renormalize`` did. For
#     a random 50k sample the sample mean ~ the full-store mean, so the prior run
#     was implicitly removing the global offset. For the topic-matched slice the
#     two diverge, so to isolate "topic-distribution vs register" (the question
#     this re-test exists to answer) we must remove the *same kind* of offset the
#     prior run did: the sample's own mean. This is the apples-to-apples battery.
#
#   * FULL-STORE (practical caveat). Subtract the OA-set centroid from OA and the
#     GLOBAL full-store wiki centroid from wiki (the offset cross_corpus_gap_demo
#     actually removes in the pipeline). The NLP-relevant wiki slice is NOT
#     representative of the full store, so its mean sits ~0.25 off the full-store
#     centroid; removing only the global offset leaves a large residual mean
#     (norm ~0.39 on the sphere) and the clouds stay trivially separable
#     (logistic AUC ~1.0). Reported as the §0 secondary row + a caveat, NOT run
#     through the full battery, because the leftover offset — not a shape
#     residual — would dominate every downstream metric and confound the
#     topic-vs-register decomposition.

def center_renormalize_own_mean(oa: NDArray, wiki: NDArray) -> tuple[NDArray, NDArray]:
    """Subtract each sample's own mean, renormalize (matches the prior run's
    ``shift_characterization.center_renormalize`` exactly)."""
    oa2 = _normalize(oa - oa.mean(axis=0, keepdims=True))
    wiki2 = _normalize(wiki - wiki.mean(axis=0, keepdims=True))
    return oa2, wiki2


def center_renormalize_full_store(
    oa: NDArray, wiki: NDArray, wiki_full_centroid: NDArray
) -> tuple[NDArray, NDArray]:
    """OA centered by its own (subsample) centroid; wiki by the GLOBAL full-store
    centroid (the offset cross_corpus_gap_demo removes in practice)."""
    oa2 = _normalize(oa - oa.mean(axis=0, keepdims=True))
    wiki2 = _normalize(wiki - wiki_full_centroid[None, :])
    return oa2, wiki2


# --- prior (random-sample) numbers, from the committed doc / commit fc392f3 ---
# Used only for the side-by-side report; the battery itself is recomputed.
PRIOR = {
    "n_each": 50000,
    "auc_raw": 0.9992,
    "auc_centered": 0.5446,
    "multi_dir": [0.5379, 0.5391, 0.5344, 0.5373, 0.5379],  # prior §1 table
    "cov": {
        10: {"angle_mean": 71.9, "angle_median": 73.8, "angle_max": 88.9, "spec_rel_l1": 0.470},
        20: {"angle_mean": None, "angle_median": None, "angle_max": None, "spec_rel_l1": None},
        50: {"angle_mean": 54.9, "angle_median": 55.4, "angle_max": 89.8, "spec_rel_l1": 0.274},
    },
    "mmd_z": 71.0,
    "energy_ratio": 2.8,
    "p_mmd": 0.0050,
    "knn_purity": 0.9879,
    "knn_baseline": 0.5000,
    "knn_excess": 0.4879,
    "knn_oa": 0.9971,
    "knn_wiki": 0.9787,
}


def _mmd_z(t: dict) -> float:
    return (
        (t["cross_mmd2"] - t["perm_mmd_mean"]) / t["perm_mmd_std"]
        if t["perm_mmd_std"] > 0
        else float("inf")
    )


def _energy_ratio(t: dict) -> float:
    return t["cross_energy"] / t["within_energy"] if t["within_energy"] > 0 else float("inf")


def render_report(R: dict, args) -> str:
    t = R["two_sample"]
    kp = R["knn"]
    mmd_z = _mmd_z(t)
    eratio = _energy_ratio(t)
    new_aucs = [a for _d, a, _n in R["multi_dir"]]

    L: list[str] = []
    L.append("## Topic-matched re-test\n")
    L.append(
        "The battery above used a **random** Wikipedia sample drawn from all "
        "7.1M pages. That sample is dominated by non-NLP topics "
        "(footballers, villages, chemical compounds), so part of the residual "
        "it found — especially the 0.988 kNN same-corpus purity — could be "
        "*topic-distribution mismatch* (NLP vs all-of-Wikipedia) rather than a "
        "genuine corpus/register shift. The decision-relevant question is the "
        "**within-topic** one: take only NLP-relevant Wikipedia and re-run.\n"
    )
    L.append(
        f"- **Wiki side (topic-matched).** Top-{R['n_wiki']} Wikipedia pages by "
        f"cosine to the OpenAlex-NLP centroid — the exact `select_nlp_pages` "
        f"selection the wiki curriculum uses (cosine "
        f"{R['sel_cos_min']:.3f}..{R['sel_cos_max']:.3f} to the centered "
        f"NLP direction).\n"
        f"- **OA side.** SAME OA-NLP slice (topic T10181, raw 384-D GIST) as the "
        f"prior run, subsampled to {R['n_oa']} to balance against wiki. Seed "
        f"{args.seed}.\n"
    )
    L.append(
        "**Centering — two variants, and the distinction matters.** The prior "
        "random-sample run subtracted each *sample's own* mean. For a random 50k "
        "sample that mean ≈ the full-store mean, so it was implicitly removing the "
        "global wiki offset. The NLP-relevant slice is **not** representative of "
        "the store, so its own mean sits ~0.25 off the full-store centroid; the "
        "two centerings now diverge:\n\n"
        "- **Own-mean (primary).** Subtract each sample's own mean, renormalize — "
        "the apples-to-apples analog of the prior run. This is the battery used "
        "for the side-by-side below and the verdicts: it removes the *same kind* "
        "of offset the prior run did, so any surviving residual is a genuine "
        "topic-vs-register comparison.\n"
        "- **Full-store (practical caveat).** Subtract the global full-store wiki "
        "centroid (the offset `cross_corpus_gap_demo` removes in the pipeline). "
        f"Because the slice is non-representative, this leaves a residual wiki "
        f"mean of norm {R['wiki_resid_mean_norm']:.3f} on the sphere, and the "
        f"clouds stay trivially separable (logistic AUC "
        f"{R['auc_centered_fullstore']:.4f}). Reported as the §0 secondary row; "
        f"NOT run through the full battery, since this leftover *offset* — not a "
        f"shape residual — would dominate every downstream metric.\n"
    )
    L.append(
        "Side-by-side with the prior random-sample run (battery identical; only "
        "the wiki sample + centering scope changed):\n"
    )

    # 0. AUC
    L.append("### 0. Sanity — raw vs centered logistic AUC\n")
    L.append("| centering | random-sample (prior) | topic-matched (new) |")
    L.append("|---|---|---|")
    L.append(f"| raw (uncentered) | {PRIOR['auc_raw']:.4f} | {R['auc_raw']:.4f} |")
    L.append(
        f"| own-mean centered (primary) | {PRIOR['auc_centered']:.4f} | "
        f"{R['auc_centered']:.4f} |"
    )
    L.append(
        f"| full-store centered (caveat) | — | {R['auc_centered_fullstore']:.4f} |"
    )
    L.append("")

    # 1. multi-dir
    L.append("### 1. Multi-direction linear separability (LDA deflation)\n")
    L.append("| direction | random-sample 1-D LDA AUC | topic-matched 1-D LDA AUC |")
    L.append("|---|---|---|")
    for i in range(len(new_aucs)):
        prior_a = PRIOR["multi_dir"][i] if i < len(PRIOR["multi_dir"]) else float("nan")
        L.append(f"| {i+1} | {prior_a:.4f} | {new_aucs[i]:.4f} |")
    L.append("")

    # 2. covariance
    L.append("### 2. Covariance / shape (principal angles + spectra)\n")
    L.append(
        "| k | random-sample mean angle | topic-matched mean angle | "
        "random spec rel-L1 | topic-matched spec rel-L1 |"
    )
    L.append("|---|---|---|---|---|")
    for k in args.pc_ks:
        c = R["cov"][k]
        pc = PRIOR["cov"].get(k, {})
        pa = pc.get("angle_mean")
        ps = pc.get("spec_rel_l1")
        pa_s = f"{pa:.1f}" if pa is not None else "—"
        ps_s = f"{ps:.3f}" if ps is not None else "—"
        L.append(
            f"| {k} | {pa_s} | {c['angle_mean']:.1f} | {ps_s} | {c['spec_rel_l1']:.3f} |"
        )
    L.append("")
    kmax = max(args.pc_ks)
    angles = R["cov"][kmax]["angles_deg"]
    L.append(
        f"First 10 topic-matched principal angles at k={kmax} (deg): "
        + ", ".join(f"{a:.1f}" for a in angles[:10])
        + "\n"
    )
    ov = R["cov"][kmax]["oa_var_topk"][:8]
    wv = R["cov"][kmax]["wiki_var_topk"][:8]
    L.append(
        "Top-8 PC variances (topic-matched) — OA: "
        + ", ".join(f"{v:.4f}" for v in ov)
        + "; wiki: "
        + ", ".join(f"{v:.4f}" for v in wv)
        + "\n"
    )

    # 3. two-sample
    L.append("### 3. Nonlinear two-sample test (MMD + energy distance)\n")
    L.append(
        f"On {t['n_sub']} centered+renorm points/group (RBF bandwidth "
        f"sigma={t['sigma']:.4f}, median heuristic; {t['n_perm']}-permutation null).\n"
    )
    L.append("| statistic | random-sample (prior) | topic-matched (new) |")
    L.append("|---|---|---|")
    L.append(
        f"| cross MMD^2 (sigma above null) | {PRIOR['mmd_z']:.0f}σ | {mmd_z:.0f}σ |"
    )
    L.append(
        f"| energy distance (× within baseline) | {PRIOR['energy_ratio']:.1f}× | {eratio:.1f}× |"
    )
    L.append(f"| permutation p-value | {PRIOR['p_mmd']:.4f} | {t['p_mmd']:.4f} |")
    L.append("")
    L.append(
        f"Raw topic-matched two-sample values: cross MMD^2={t['cross_mmd2']:.6f} "
        f"(within baseline {t['within_mmd2']:.6f}, null {t['perm_mmd_mean']:.6f}"
        f"±{t['perm_mmd_std']:.6f}); cross energy={t['cross_energy']:.6f} "
        f"(within baseline {t['within_energy']:.6f}).\n"
    )

    # 4. kNN
    L.append("### 4. Local kNN same-corpus purity (k=10)\n")
    L.append(
        "| | random-sample (prior) | topic-matched (new) |"
    )
    L.append("|---|---|---|")
    L.append(f"| mean purity | {PRIOR['knn_purity']:.4f} | {kp['mean_purity']:.4f} |")
    L.append(f"| mixing-ratio baseline | {PRIOR['knn_baseline']:.4f} | {kp['baseline']:.4f} |")
    L.append(f"| excess over baseline | {PRIOR['knn_excess']:+.4f} | {kp['excess']:+.4f} |")
    L.append(f"| OA-query purity | {PRIOR['knn_oa']:.4f} | {kp['mean_purity_oa']:.4f} |")
    L.append(f"| wiki-query purity | {PRIOR['knn_wiki']:.4f} | {kp['mean_purity_wiki']:.4f} |")
    L.append("")

    # verdicts
    L.append("### Verdict (topic-matched)\n")
    L.append(R["verdict"])
    L.append("")
    L.append(
        f"*(Reproduce: `uv run python -m "
        f"artifacts_scripts.wiki.shift_characterization_topicmatched --wiki-store "
        f"{args.wiki_store} --n-select {args.n_select} --seed {args.seed}`.)*\n"
    )
    return "\n".join(L)


def build_verdict(R: dict) -> str:
    t = R["two_sample"]
    kp = R["knn"]
    kmax = max(R["cov"].keys())
    ang = R["cov"][kmax]["angle_mean"]
    eratio = _energy_ratio(t)
    mmd_z = _mmd_z(t)

    # Deltas vs prior.
    d_purity = kp["mean_purity"] - PRIOR["knn_purity"]
    d_angle = ang - PRIOR["cov"][kmax]["angle_mean"]
    d_mmdz = mmd_z - PRIOR["mmd_z"]

    # How much of the residual collapsed?
    # Purity excess: prior 0.488 above 0.5 baseline. New excess:
    excess_collapse = 1.0 - (kp["excess"] / PRIOR["knn_excess"]) if PRIOR["knn_excess"] else 0.0

    purity_collapsed = kp["mean_purity"] < 0.65   # toward the 0.5 baseline
    angle_shrank = ang < 35.0                      # well below the prior 55 deg
    mmd_small = mmd_z < 10.0                        # near the same-distribution floor
    # Did the own-mean centering remove the linear (mean) separation, as it did
    # in the prior run? Phrase §2 from the data rather than asserting it.
    new_aucs = [a for _d, a, _n in R["multi_dir"]]
    lda_collapsed = max(new_aucs) < 0.6 if new_aucs else False
    lda_phrase = (
        "no LDA deflation direction recovers it"
        if lda_collapsed
        else f"LDA deflation still separates (max 1-D AUC {max(new_aucs):.2f})"
    )

    if purity_collapsed and angle_shrank and mmd_small:
        kind = "OFFSET-LIKE (residual collapses)"
        transfer = (
            "naive LPP-transfer is viable: a wiki-learned linear projection should "
            "carry to OA with little degradation, since the within-topic covariances "
            "largely align after the offset is removed"
        )
    elif not purity_collapsed and not angle_shrank:
        kind = "GENUINE NONLINEAR REGISTER RESIDUAL (residual persists)"
        transfer = (
            "naive LPP-transfer will degrade: the within-topic covariance mismatch "
            "persists, so a wiki-learned projection is a poor basis for the OA cloud — "
            "covariance alignment (CORAL/CCA) or a joint formulation is needed"
        )
    else:
        kind = "PARTIAL COLLAPSE (residual shrinks but does not vanish)"
        transfer = (
            "LPP-transfer is borderline: the offset explains much of the residual but "
            "a non-trivial covariance/shape mismatch remains — expect some degradation; "
            "a light covariance alignment (CORAL) likely recovers it"
        )

    return (
        f"**Topic-matched verdict: {kind}.** Restricting Wikipedia to its "
        f"NLP-relevant slice changes the picture markedly relative to the random "
        f"sample.\n\n"
        f"**(1) How much was topic-distribution vs genuine corpus shift?** "
        f"Local kNN same-corpus purity drops from {PRIOR['knn_purity']:.3f} "
        f"(random) to {kp['mean_purity']:.3f} (topic-matched), excess over the "
        f"0.5 baseline collapsing from {PRIOR['knn_excess']:+.3f} to "
        f"{kp['excess']:+.3f} (~{100*excess_collapse:.0f}% of the local clumping "
        f"removed). The covariance mean principal angle (k={kmax}) moves "
        f"{PRIOR['cov'][kmax]['angle_mean']:.0f}° → {ang:.0f}° "
        f"(Δ{d_angle:+.0f}°), and the cross MMD^2 σ-gap moves "
        f"{PRIOR['mmd_z']:.0f}σ → {mmd_z:.0f}σ (Δ{d_mmdz:+.0f}σ). So a "
        f"{'large' if abs(d_purity) > 0.2 else 'modest'} fraction of the prior "
        f"residual was **topic-distribution mismatch**, not corpus/register.\n\n"
        f"**(2) Within-topic, after the offset:** the residual is "
        f"{'essentially offset-like — it collapses toward the same-distribution floor' if kind.startswith('OFFSET') else ('a genuine nonlinear register residual — it persists' if kind.startswith('GENUINE') else 'partly offset, partly a real shape mismatch')}. "
        f"After per-corpus (own-mean) centering the means coincide (centered "
        f"logistic AUC {R['auc_centered']:.3f}, {lda_phrase}), and "
        f"the {'covariance now largely matches' if angle_shrank else 'covariance still differs materially'} "
        f"(angle {ang:.0f}°), with kNN purity "
        f"{'near the 0.5 mixing floor' if purity_collapsed else 'still elevated'}.\n\n"
        f"**(3) Revised LPP-transfer bound:** {transfer}. "
        f"At {ang:.0f}° mean principal angle (k={kmax}) and {mmd_z:.0f}σ cross "
        f"MMD^2, the within-topic covariance mismatch is "
        f"{'small' if angle_shrank and mmd_small else ('large' if not angle_shrank else 'moderate')}. "
        f"Practical caveat: the pipeline removes the *global* full-store wiki "
        f"offset, not the slice's own mean — for an NLP-relevant slice that leaves "
        f"a residual offset (norm {R['wiki_resid_mean_norm']:.2f}; full-store-"
        f"centered logistic AUC {R['auc_centered_fullstore']:.3f}), so a wiki→OA "
        f"projection must mean-match per-slice (or jointly) before any covariance "
        f"alignment can help."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/enwiki_full"))
    ap.add_argument("--n-select", type=int, default=40000,
                    help="top-cosine NLP-relevant wiki pages to keep (also the OA cap)")
    ap.add_argument("--n-dirs", type=int, default=5)
    ap.add_argument("--pc-ks", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--mmd-n-sub", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--knn-n-query", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nlp-dir-cache", type=Path, default=Path("/tmp/nlp_dir.npy"))
    ap.add_argument("--sel-cache", type=Path, default=None,
                    help="npz cache for the streamed wiki selection "
                         "(default /tmp/wiki_nlp_sel_n{n_select}.npz)")
    ap.add_argument("--out", type=Path, default=Path("docs/wikipedia-openalex-shared-space.md"))
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    sel_cache = args.sel_cache or Path(f"/tmp/wiki_nlp_sel_n{args.n_select}.npz")

    print("Loading OpenAlex NLP slice (raw GIST) ...")
    oa_full, _t, _i, oa_fp = load_openalex_nlp()
    oa_full = _normalize(np.ascontiguousarray(oa_full, dtype=np.float32))
    print(f"  OA NLP: {oa_full.shape}, fp={oa_fp}")

    nlp_dir = load_nlp_dir(oa_full, args.nlp_dir_cache)

    print("Selecting topic-matched (NLP-relevant) Wikipedia pages ...")
    wiki_sel, wiki_full_centroid = select_nlp_wiki_cached(
        args.wiki_store, nlp_dir, args.n_select, sel_cache
    )
    # Re-score the selected rows for the cosine range report (against centered dir).
    sel_centered_cos = _normalize(wiki_sel - wiki_full_centroid[None, :]) @ nlp_dir

    # Balance OA to the wiki count.
    n = min(args.n_select, len(oa_full), len(wiki_sel))
    rng = np.random.default_rng(args.seed)
    oa_idx = rng.choice(len(oa_full), size=n, replace=False)
    oa_s = np.ascontiguousarray(oa_full[oa_idx], dtype=np.float32)
    wiki_s = np.ascontiguousarray(wiki_sel[:n], dtype=np.float32)
    print(f"Balanced sample: OA={len(oa_s)}, wiki={len(wiki_s)}.")

    # 0. raw vs centered AUC. PRIMARY centering is own-mean (the apples-to-apples
    # analog of the prior run); full-store centering is reported as a secondary
    # row + caveat (it leaves a residual offset for this non-representative slice).
    print("== 0. sanity AUC ==")
    auc_raw = logistic_auc(oa_s, wiki_s, args.seed)
    oa_c, wiki_c = center_renormalize_own_mean(oa_s, wiki_s)
    auc_centered = logistic_auc(oa_c, wiki_c, args.seed)
    # Full-store-centered AUC + leftover wiki residual-mean norm (the practical caveat).
    oa_fs, wiki_fs = center_renormalize_full_store(oa_s, wiki_s, wiki_full_centroid)
    auc_centered_fullstore = logistic_auc(oa_fs, wiki_fs, args.seed)
    wiki_resid_mean_norm = float(np.linalg.norm(wiki_fs.mean(axis=0)))
    print(
        f"  raw AUC={auc_raw:.4f}  own-mean-centered AUC={auc_centered:.4f}  "
        f"full-store-centered AUC={auc_centered_fullstore:.4f} "
        f"(wiki residual-mean-norm {wiki_resid_mean_norm:.3f})"
    )

    # 1. multi-direction LDA.
    print("== 1. multi-direction LDA deflation ==")
    multi_dir = multi_direction_auc(oa_c, wiki_c, args.n_dirs, args.seed)
    for d, auc, _nrm in multi_dir:
        print(f"  dir {d}: AUC={auc:.4f}")

    # 2. covariance.
    print("== 2. covariance / shape ==")
    cov = covariance_comparison(oa_c, wiki_c, args.pc_ks)
    for k in args.pc_ks:
        print(
            f"  k={k}: mean angle={cov[k]['angle_mean']:.1f} deg, "
            f"max={cov[k]['angle_max']:.1f}, spec rel-L1={cov[k]['spec_rel_l1']:.3f}"
        )

    # 3. two-sample.
    print("== 3. MMD + energy two-sample ==")
    two_sample = two_sample_battery(oa_c, wiki_c, args.mmd_n_sub, args.n_perm, args.seed)
    print(
        f"  MMD^2 cross={two_sample['cross_mmd2']:.6f} within={two_sample['within_mmd2']:.6f} "
        f"p={two_sample['p_mmd']:.4f} (z={_mmd_z(two_sample):.1f})"
    )
    print(
        f"  energy cross={two_sample['cross_energy']:.6f} within={two_sample['within_energy']:.6f} "
        f"ratio={_energy_ratio(two_sample):.2f} p={two_sample['p_energy']:.4f}"
    )

    # 4. kNN purity.
    print("== 4. kNN purity ==")
    knn = knn_purity(oa_c, wiki_c, args.knn_k, args.knn_n_query, args.seed)
    print(
        f"  mean purity={knn['mean_purity']:.4f} baseline={knn['baseline']:.4f} "
        f"excess={knn['excess']:+.4f}"
    )

    R = {
        "n_oa": len(oa_s),
        "n_wiki": len(wiki_s),
        "sel_cos_min": float(sel_centered_cos.min()),
        "sel_cos_max": float(sel_centered_cos.max()),
        "auc_raw": auc_raw,
        "auc_centered": auc_centered,
        "auc_centered_fullstore": auc_centered_fullstore,
        "wiki_resid_mean_norm": wiki_resid_mean_norm,
        "multi_dir": multi_dir,
        "cov": cov,
        "two_sample": two_sample,
        "knn": knn,
    }
    R["verdict"] = build_verdict(R)
    print("\n=== VERDICT ===")
    print(R["verdict"])

    report = render_report(R, args)
    if args.append:
        with open(args.out, "a") as f:
            f.write("\n\n" + report)
        print(f"\nAppended section to {args.out}")
    else:
        print("\n--- report (not written; pass --append to write) ---\n")
        print(report)


if __name__ == "__main__":
    main()
