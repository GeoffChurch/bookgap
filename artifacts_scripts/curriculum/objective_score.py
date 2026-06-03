"""Objective, LLM-free scoring of curriculum clusterings — the external/intrinsic anchor for the judge.

Motivated by the 2026-06-03 worry "are we relying fully on the faulty LLM eval?". Three objective lenses,
none of which touch an LLM:

  1. ToC-RECONSTRUCTION agreement (AMI/NMI) — the EXTERNAL ground truth. The goal is to reconstruct the
     book's actual chapters, so induce a reference partition by assigning each concept work to its nearest
     book-chapter signature (from the ToC title+description), in BOTH keyword and embedding space, and
     measure how well each method's clusters agree with it. The method that best recovers the book wins the
     stated task. (Reported in two reference spaces; a robust win agrees in both.)
  2. INTER-CHAPTER REDUNDANCY — the thing the judge caught in link-pLSA (TTS×2, paratext T18≡T19) that the
     balance metrics (n_tiny, coverage) completely missed. Mean cosine of each chapter's keyword-lift
     vector to its NEAREST other chapter; high = duplicated chapters.
  3. CROSS-VIEW SILHOUETTE — is the partition REAL or a one-view artifact? A method's clusters should stay
     cohesive even in feature spaces it did NOT use. Silhouette of its labels in keyword / embedding /
     citation space (sampled).

Use these to VALIDATE the judge (does judge-preferred = best reconstruction?), not to replace it.

Regenerate (do not edit by hand):
  python -m artifacts_scripts.curriculum.objective_score \
    --toc-json artifacts/book_gaps/jurafsky_toc.json \
    --concept-gap artifacts/book_gaps/jurafsky_concept_full/concept_gap.json --field 17 \
    --runs jurafsky_trackA_dec jurafsky_trackA_cca jurafsky_cca_coupling jurafsky_concat_vocab \
           jurafsky_link_plsa jurafsky_discardclash jurafsky_citonly
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score
from sklearn.preprocessing import normalize

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.field_term_index import build_field_term_index
from bookgap.util import LOGGER, setup_logging

AD = "artifacts/curriculum"


def _nearest_chapter(work_vecs, chap_vecs) -> np.ndarray:
    """Assign each work to the argmax-cosine book chapter (both row-normalized)."""
    return np.asarray((work_vecs @ chap_vecs.T).argmax(1)).ravel()


def _redundancy(V, labels, K) -> float:
    """Nearest-other-chapter cosine of the per-chapter CENTROIDS in feature space V (V[i] = work rows[i]).
    Jurafsky-INDEPENDENT, works for any partition incl. the ToC; ↓ = more distinct chapters. Centroids
    (full keyword space) avoid the disjoint-top-k-terms degeneracy."""
    m = labels >= 0
    lab = labels[m].astype(int); cols = np.where(m)[0]
    present = sorted(set(lab.tolist()))
    if len(present) < 2:
        return float("nan")
    remap = {k: i for i, k in enumerate(present)}
    Ind = sp.csr_matrix((np.ones(len(lab)), ([remap[k] for k in lab], np.arange(len(lab)))),
                        shape=(len(present), len(lab)))
    C = normalize(Ind @ V[cols])                       # (K' × D) row-normalized centroids
    S = (C @ C.T).toarray(); np.fill_diagonal(S, -1.0)
    return float(S.max(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toc-json", required=True, type=Path)
    ap.add_argument("--concept-gap", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--runs", nargs="+", required=True, help="run-dir names under artifacts/curriculum/")
    ap.add_argument("--lsa-dim", type=int, default=200)
    ap.add_argument("--sil-sample", type=int, default=3000)
    ap.add_argument("--restrict-to", default=None,
                    help="coverage-match: score every run only on the works covered by THIS run-dir "
                         "(e.g. the 13%-coverage embedding selection) for an apples-to-apples AMI.")
    args = ap.parse_args()
    setup_logging(log_name="obj_score", log_level="INFO", include_time=False)

    X, vocab, _ = build_field_term_index(args.field, None, 20, 1_000_000)
    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}
    ids = json.loads(args.concept_gap.read_text())["book"]["gap_article_ids"]
    rows = [id2idx[w] for w in ids if w in id2idx]
    wid_rows = [wid[r] for r in rows]; row_of_wid = {w: i for i, w in enumerate(wid_rows)}
    N = len(rows)
    LOGGER.info("concept: %d works", N)

    # work vectors in two spaces
    Xc = X[rows].astype(np.float64)
    idf = np.log(N / (np.asarray((Xc > 0).sum(0)).ravel() + 1.0))
    KWv = normalize(Xc.multiply(idf).tocsr())                                  # keyword TF-IDF
    EMv = normalize(np.asarray(cloud.embeddings(), dtype=np.float64)[rows])    # GIST embedding

    # ToC chapter signatures -> reference partitions (keyword + embedding)
    toc = json.loads(args.toc_json.read_text())["chapters"]
    ctext = [f"{c.get('title','')}. {c.get('description','')}" for c in toc]
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2))
    Cw = normalize(cv.transform(ctext).multiply(idf).tocsr())
    ref_kw = _nearest_chapter(KWv, Cw)
    ref_em = None
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0", device="cpu")
        Ce = normalize(np.asarray(st.encode(ctext), dtype=np.float64))
        ref_em = _nearest_chapter(EMv, Ce)
    except Exception as e:                                                     # robust to no model
        LOGGER.warning("embedding ToC reference skipped: %s", e)

    # citation (coupling) space for cross-view silhouette
    Gc = cloud.citation_graph(False).tocsr().astype(np.float64); Gc.setdiag(0.0); Gc.eliminate_zeros()
    Xo = Gc[rows]; cdf = np.asarray((Xo > 0).sum(0)).ravel(); Xo = Xo[:, cdf > 0]
    CITv = normalize(Xo.multiply(np.log(N / (cdf[cdf > 0] + 1.0))).tocsr())

    # coverage-match: restrict scoring to the works covered by one run (e.g. the 13% embedding selection)
    allowed = None
    if args.restrict_to:
        rt = json.loads((Path(AD) / args.restrict_to / "topic_work_ids.json").read_text())
        allowed = {row_of_wid[w] for ws in rt.values() for w in ws if w in row_of_wid}
        LOGGER.info("restrict-to %s: %d works", args.restrict_to, len(allowed))

    rng = np.random.RandomState(0)
    samp = rng.choice(N, min(args.sil_sample, N), replace=False)

    def sil(Vv, lab, mask):
        s = np.array([i for i in samp if mask[i]])
        if len(set(lab[s].tolist())) < 2:
            return float("nan")
        return float(silhouette_score(Vv[s], lab[s], metric="cosine"))

    # the book's own ToC chapters, scored as if it were a method (AMI vs itself is trivially 1 → "ref")
    items = [("TOC(jurafsky)", ref_kw, len(toc))]
    for r in args.runs:
        twi = json.loads((Path(AD) / r / "topic_work_ids.json").read_text())
        lab = np.full(N, -1)
        for t, ws in twi.items():
            for w in ws:
                if w in row_of_wid:
                    lab[row_of_wid[w]] = int(t)
        items.append((r, lab, len(twi)))

    print(f"\n{'run':<24} {'AMI_kw':>7} {'AMI_emb':>8} {'redund':>7} {'sil_kw':>7} {'sil_emb':>8} "
          f"{'sil_cit':>8} {'n_ch':>5} {'cover':>6}")
    print("-" * 92)
    for name, lab, nch in items:
        mask = lab >= 0
        if allowed is not None:
            mask = mask & np.array([i in allowed for i in range(N)])
        is_ref = name.startswith("TOC")
        akw = float("nan") if is_ref else ami(ref_kw[mask], lab[mask])
        aem = (float("nan") if is_ref or ref_em is None else ami(ref_em[mask], lab[mask]))
        red = _redundancy(KWv, lab, max(nch, int(lab.max()) + 1))
        cover = 100.0 * mask.mean()
        akws = "  ref" if is_ref else f"{akw:.3f}"
        aems = "   ref" if is_ref else f"{aem:.3f}"
        print(f"{name:<24} {akws:>7} {aems:>8} {red:>7.3f} {sil(KWv, lab, mask):>7.3f} "
              f"{sil(EMv, lab, mask):>8.3f} {sil(CITv, lab, mask):>8.3f} {nch:>5} {cover:>5.0f}%")
    print("\nAMI_kw/emb = agreement with the book's ToC-induced partition (↑=more FAITHFUL to Jurafsky — NOT "
          "a quality ceiling; a better-than-Jurafsky set would score LOWER). redund = nearest-other-chapter "
          "keyword-lift cosine, Jurafsky-INDEPENDENT (↓=more distinct chapters); beat TOC's redund = arguably "
          "better-organized than the book. sil_* = cross-view cohesion." + (
          f"  [restricted to {args.restrict_to}'s works]" if args.restrict_to else ""))


if __name__ == "__main__":
    main()
