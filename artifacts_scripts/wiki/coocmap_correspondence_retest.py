#!/usr/bin/env python
# artifacts_scripts/wiki/coocmap_correspondence_retest.py
"""CORRECTED re-test of coocmap for wiki<->OA CORRESPONDENCE (not mixing).

Prior runs measured kNN same-corpus purity ("mixing"), which is NOT coocmap's
objective. coocmap is an unsupervised cross-corpus DICTIONARY/CORRESPONDENCE
method: two clouds can stay locally disjoint while the induced matches are
correct (English/French embedding clouds never merge yet yield a translation
dictionary). This script measures CORRESPONDENCE QUALITY instead.

Algorithm (faithful to ~/src/mycooc/coocmap_baseline.py), src=wiki, tgt=OA:

  INTRA-cloud association matrices (the crux):
    M_W  = V_W x V_W : [i,j] = cos(wiki_i, wiki_j) if j in i's top-k WIKI
           neighbors else 0. Built over WIKI embeddings only.
    M_OA = V_OA x V_OA, over OA embeddings only.
    Each normalized rows-wise via _normalize(["unit","center","unit"]).

  Dictionary priming (we SHARE an encoder, so skip the sorted-signature
  _match_sim bootstrap): init  sim = X_W @ X_OA.T  where X = the GIST
  embeddings, per-corpus mean-centered + unit-normed.

  Loop per step:
    sc = _csls(sim, csls_k)
    inds1, inds2 = _fbm(sc)
    Xf = unit(M_W[:, inds1]);  Zf = unit(M_OA[:, inds2])
    sim = Xf @ Zf.T

Note Xf indexes M_W's OWN columns (each wiki item's intra-wiki associations to
the matched wiki-anchors); Zf likewise for OA. The pair (inds1[a], inds2[a]) is
a matched cross-corpus anchor.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from artifacts_scripts.wiki.shared_space_analysis import (
    load_openalex_nlp,
    load_wikipedia,
)

# ----------------------------------------------------------------------------
# coocmap primitives — verbatim from ~/src/mycooc/coocmap_baseline.py
# ----------------------------------------------------------------------------


def _unit_normalize(X: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row (coocmap length_normalize)."""
    norms = X.norm(p=2, dim=1, keepdim=True).clamp(min=1e-30)
    return X / norms


def _normalize(X: torch.Tensor, actions: list[str]) -> torch.Tensor:
    """coocmap _normalize: a sequence of unit / center row-wise ops."""
    for action in actions:
        if action == "unit":
            X = _unit_normalize(X)
        elif action == "center":
            X = X - X.mean(dim=0, keepdim=True)
    return X


def _csls(sim: torch.Tensor, k: int) -> torch.Tensor:
    """CSLS correction (coocmap most_diff_match)."""
    mean0 = sim.topk(k, dim=0).values.mean(dim=0, keepdim=True)  # [1, V_tgt]
    mean1 = sim.topk(k, dim=1).values.mean(dim=1, keepdim=True)  # [V_src, 1]
    return sim - 0.5 * (mean0 + mean1)


def _fbm(sim: torch.Tensor):
    """Forward/backward match (coocmap forward_backward_match)."""
    V_src, V_tgt = sim.shape
    indsf2 = sim.argmax(dim=1)  # [V_src] best tgt for each src
    indsb1 = sim.argmax(dim=0)  # [V_tgt] best src for each tgt
    inds1 = torch.cat([torch.arange(V_src, device=sim.device), indsb1])
    inds2 = torch.cat([indsf2, torch.arange(V_tgt, device=sim.device)])
    return inds1, inds2


# ----------------------------------------------------------------------------
# intra-cloud kNN-gram (the M_W / M_OA association matrices)
# ----------------------------------------------------------------------------


def knn_gram(emb: np.ndarray, k: int, device: torch.device) -> torch.Tensor:
    """V x V intra-cloud kNN-gram: [i,j] = cos(i,j) if j in i's top-k (incl self)
    else 0. Built over THIS corpus's embeddings only. Returns normalized
    (unit->center->unit) row-wise per coocmap.
    """
    X = torch.tensor(emb, device=device, dtype=torch.float32)
    Xn = _unit_normalize(X)              # cosine == dot
    S = Xn @ Xn.T                        # V x V cosine
    V = S.shape[0]
    # keep top-k per row (each row i: its own neighbors), zero the rest.
    topv, topi = S.topk(k, dim=1)        # [V, k]
    G = torch.zeros_like(S)
    G.scatter_(1, topi, topv)
    G = _normalize(G, ["unit", "center", "unit"])
    return G


def prime_sim(wiki_emb: np.ndarray, oa_emb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Shared-encoder priming: sim = X_W @ X_OA.T, per-corpus mean-centered
    then unit-normed (the identity-space cross-corpus cosine NN)."""
    Xw = torch.tensor(wiki_emb, device=device, dtype=torch.float32)
    Xo = torch.tensor(oa_emb, device=device, dtype=torch.float32)
    Xw = _unit_normalize(Xw - Xw.mean(dim=0, keepdim=True))
    Xo = _unit_normalize(Xo - Xo.mean(dim=0, keepdim=True))
    return Xw @ Xo.T  # V_W x V_OA


# ----------------------------------------------------------------------------
# correspondence-quality metrics
# ----------------------------------------------------------------------------


def mutual_nn_rate(sim: torch.Tensor) -> float:
    """Fraction of src items whose argmax-tgt also picks them back (mutual NN)."""
    V_src, V_tgt = sim.shape
    fwd = sim.argmax(dim=1)            # [V_src] -> tgt
    bwd = sim.argmax(dim=0)            # [V_tgt] -> src
    src_ids = torch.arange(V_src, device=sim.device)
    mutual = (bwd[fwd] == src_ids).sum().item()
    return mutual / V_src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki-store", default="cache/wiki/store/wiki_400k")
    ap.add_argument("--n", type=int, default=3000, help="items per corpus")
    ap.add_argument("--k", type=int, default=20, help="kNN-gram neighbors")
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eyeball", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # --- Load corpora ------------------------------------------------------
    print("Loading OpenAlex NLP slice (raw GIST) ...")
    oa, oa_titles, oa_ids, oa_fp = load_openalex_nlp()
    print(f"  OA NLP full: {oa.shape}")
    print("Loading Wikipedia corpus ...")
    wiki, wiki_titles, wiki_ids, wiki_fp = load_wikipedia(args.wiki_store)
    print(f"  Wiki full: {wiki.shape}")
    assert oa_fp == wiki_fp, f"fingerprint mismatch OA={oa_fp} wiki={wiki_fp}"

    def l2(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return (X / n).astype(np.float32)

    oa = l2(oa)
    wiki = l2(wiki)

    # --- NLP-wiki slice: top-n wiki by cosine to OA-NLP centroid -----------
    oa_centroid = l2(oa.mean(axis=0, keepdims=True))[0]
    wiki_to_oa = wiki @ oa_centroid
    wiki_sel = np.argsort(-wiki_to_oa)[: args.n]
    wiki_s = wiki[wiki_sel]
    wiki_s_titles = [wiki_titles[i] for i in wiki_sel]

    # --- OA-NLP slice: take up to n (closest to its own centroid for stability)
    n_oa = min(args.n, len(oa))
    oa_sel = np.arange(len(oa)) if len(oa) <= args.n else \
        np.argsort(-(oa @ oa_centroid))[: args.n]
    oa_s = oa[oa_sel]
    oa_s_titles = [oa_titles[i] for i in oa_sel]

    V_W, V_OA = len(wiki_s), len(oa_s)
    print(f"\nSLICE SHAPES: wiki_s={wiki_s.shape}  oa_s={oa_s.shape}")

    # --- Build intra-cloud association matrices ----------------------------
    M_W = knn_gram(wiki_s, args.k, device)
    M_OA = knn_gram(oa_s, args.k, device)
    print(f"M_W shape={tuple(M_W.shape)}  M_OA shape={tuple(M_OA.shape)}  k={args.k}")

    # --- Prime from shared encoder -----------------------------------------
    sim = prime_sim(wiki_s, oa_s, device)
    print(f"sim (prime) shape={tuple(sim.shape)}")

    # The RAW prime matches (for the coocmap-vs-raw comparison). Use CSLS'd
    # prime as the "raw NN" baseline argmax (coocmap's loop also CSLS's, so the
    # honest baseline is the CSLS'd shared-encoder NN).
    raw_sim = sim.clone()
    raw_csls = _csls(raw_sim, args.csls_k)
    raw_match = raw_csls.argmax(dim=1).cpu().numpy()   # wiki_i -> oa_j (raw)
    raw_mnn = mutual_nn_rate(raw_csls)
    print(f"\nRAW prime (CSLS'd) mutual-NN rate = {raw_mnn:.4f}")

    # --- coocmap loop ------------------------------------------------------
    mnn_history = [("prime(CSLS)", raw_mnn)]
    for step in range(1, args.iters + 1):
        sc = _csls(sim, args.csls_k)
        inds1, inds2 = _fbm(sc)
        Xf = _unit_normalize(M_W[:, inds1])
        Zf = _unit_normalize(M_OA[:, inds2])
        sim = Xf @ Zf.T
        mnn = mutual_nn_rate(sim)
        # how many cooc matches differ from raw, how stable vs prev
        cur_match = sim.argmax(dim=1).cpu().numpy()
        changed_vs_raw = int((cur_match != raw_match).sum())
        mnn_history.append((f"iter{step}", mnn))
        print(f"  iter {step:2d}: mutual-NN={mnn:.4f}  "
              f"n_anchors={len(inds1)}  matches!=raw: {changed_vs_raw}/{V_W}")

    cooc_csls = _csls(sim, args.csls_k)
    cooc_match = cooc_csls.argmax(dim=1).cpu().numpy()  # final coocmap match
    cooc_mnn = mutual_nn_rate(cooc_csls)

    # --- MNN-over-iterations summary ---------------------------------------
    print("\n=== MUTUAL-NN over iterations ===")
    for tag, v in mnn_history:
        print(f"  {tag:14s} {v:.4f}")

    # --- Eyeball top matches (by match confidence) -------------------------
    # Rank sampled wiki items by their match similarity, show top n_eyeball.
    rng = np.random.default_rng(args.seed)

    def show_matches(match_idx, sim_mat, tag):
        # confidence = the sim value at the matched column
        conf = sim_mat[torch.arange(V_W, device=device), torch.tensor(match_idx, device=device)]
        conf = conf.cpu().numpy()
        order = np.argsort(-conf)[: args.n_eyeball]
        print(f"\n### TOP {args.n_eyeball} {tag} MATCHES (by confidence) ###")
        for r in order:
            print(f"  [{conf[r]:+.3f}] WIKI: {wiki_s_titles[r][:70]}")
            print(f"          ->  OA: {oa_s_titles[match_idx[r]][:70]}")
        return order

    print("\n" + "=" * 78)
    print("RAW shared-encoder cosine NN (the prime, CSLS'd) — the baseline")
    raw_order = show_matches(raw_match, raw_csls, "RAW")

    print("\n" + "=" * 78)
    print("COOCMAP induced matches (after iteration) — the test")
    show_matches(cooc_match, cooc_csls, "COOCMAP")

    # --- coocmap vs raw on the SAME sampled wiki items ---------------------
    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE on the same random sample (raw vs coocmap match):")
    sample = rng.choice(V_W, size=min(args.n_eyeball, V_W), replace=False)
    n_diff = 0
    for r in sample:
        rm, cm = raw_match[r], cooc_match[r]
        flag = "  (same)" if rm == cm else "  (DIFF)"
        if rm != cm:
            n_diff += 1
        print(f"  WIKI: {wiki_s_titles[r][:60]}{flag}")
        print(f"    raw  -> OA: {oa_s_titles[rm][:60]}")
        if rm != cm:
            print(f"    cooc -> OA: {oa_s_titles[cm][:60]}")
    print(f"\n  coocmap changed the match for {n_diff}/{len(sample)} sampled wiki items")

    print("\n=== SUMMARY ===")
    print(f"  V_W={V_W}  V_OA={V_OA}  k={args.k}  csls_k={args.csls_k}  iters={args.iters}")
    print(f"  raw prime mutual-NN  = {raw_mnn:.4f}")
    print(f"  coocmap mutual-NN    = {cooc_mnn:.4f}")
    total_changed = int((cooc_match != raw_match).sum())
    print(f"  coocmap changed {total_changed}/{V_W} matches vs raw "
          f"({100*total_changed/V_W:.1f}%)")


if __name__ == "__main__":
    main()
