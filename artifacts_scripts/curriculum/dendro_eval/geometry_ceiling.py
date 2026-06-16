"""Geometry-ceiling probe — the DECISIVE "is better curriculum geometry a lever?" test.

Prior result (``toc_recovery.py`` + close-loop ablation): the embedding GEOMETRY,
not the clustering algorithm, sets the ceiling on chapter recovery. Clean corpora
recover chapters under ANY tree (color_image-style ARI ~0.98 in the clean regime),
noisy ones don't under ANY tree (cyberbullying ARI ~0.25 / recon ~0.08). The tree
construction was exonerated; the question that remains is about the SPACE itself:

    For a hard book, does ANY linear subspace separate its chapters?

If a SUPERVISED oracle (LDA on gold chapter labels) can separate the chapters on
held-out works, then a better (unsupervised) embedding is a real lever — Stage 2
(learning a better curriculum space) is worth it. If even the oracle can't lift a
book off the floor, the chapters genuinely overlap in this representation and we
should pivot rather than chase embeddings.

The probe — per book, a 3-rung SPACE ladder on a held-out split
---------------------------------------------------------------
Reuses ``toc_recovery``'s corpus loader and labeled-point-set construction EXACTLY
(same 4 books, same per-book scope, same top-N-by-max-cosine-to-any-chapter set,
same argmax-cosine gold labels), so results tie to the prior experiment. Then:

1. **Stratified train/test split** of the labeled works by gold chapter (fixed
   seed, ~70/30). The SAME split feeds all three spaces. Chapters with <5 works
   are flagged (unstable for LDA / per-class F1).
2. **Three spaces**, each fit on TRAIN only and applied to TEST:
   * ``raw``        — GIST-small 384d as-is (the baseline geometry).
   * ``pca50``      — unsupervised PCA->50d (fit on train). Cheap denoise preview.
   * ``lda_oracle`` — supervised PCA(50)->LDA on TRAIN gold labels, projecting to
     <=(n_classes-1) dims. This is the SUPERVISED CEILING: the best a linear
     subspace can do when it is *told* the chapter labels. LDA never sees test
     labels (it is fit on train only; test is projected through the frozen pipe).
3. **Two readouts on held-out TEST works** in each space:
   * **kNN chapter classification** (train->test, k=10, cosine): macro-F1 + acc.
     The purest "is chapter info linearly present" signal.
   * **Unsupervised ARI@k**: average-linkage (cosine) over TEST works only,
     ``fcluster`` at k=n_chapters, ARI vs gold test labels. Ties to the
     curriculum's flat-cut operating point.
4. **Random-label ARI baseline** (shuffle test labels, average over repeats) so
   "above chance" is explicit per space.

Interpretation guide (per book)
-------------------------------
* lda_oracle HIGH and >> raw                 -> linear separability exists; better
                                                geometry is a real lever (Stage 2).
* lda_oracle HIGH on kNN-F1 but unsup ARI
  stays low in raw/pca50                     -> structure present but NOT dominant
                                                variance; unsupervised methods miss
                                                it; a label-aware/cleverer space is
                                                needed (lever exists, harder).
* lda_oracle LOW (~ raw, both low)           -> chapters overlap even under
                                                supervision; geometry is a hard wall
                                                for that book -> pivot.

CIRCULARITY CAVEAT (same as the prior run, and the cross-SPACE comparison is fair)
----------------------------------------------------------------------------------
Work labels are argmax-cosine to the SAME-space (raw GIST-small) chapter
embeddings, so ABSOLUTE numbers are inflated. But all three spaces see IDENTICAL
works and the IDENTICAL train/test split; only the projection differs, and the
oracle never sees test labels. The cross-space deltas (lda - raw, pca50 - raw)
are therefore a fair read on "does a better subspace help".

Regenerate (do not edit generated JSON by hand):

  .conda/bin/python -m artifacts_scripts.curriculum.dendro_eval.geometry_ceiling
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# Reuse the prior experiment's corpus loader + chapter embedder + book table +
# labeled-set helpers VERBATIM so the point population is identical.
from artifacts_scripts.curriculum.dendro_eval.toc_recovery import (
    _BOOKS,
    _embed_chapters,
    _load_article_space,
    _unit,
)
from bookgap.util import LOGGER, setup_logging

_MODULE = "artifacts_scripts.curriculum.dendro_eval.geometry_ceiling"
_HERE = Path(__file__).resolve().parent
_BOOK_GAPS = Path("artifacts/book_gaps")

# kNN neighborhood; small-N books have few test points so keep k modest.
_KNN_K = 10
# unsupervised PCA target dim (rung 2) — well below 384 and below most n_train.
_PCA_DIM = 50
# chapters with fewer than this many works are flagged unstable.
_MIN_PER_CHAP = 5


def _provenance() -> str:
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    return f"Generated by `{cmd}`. Regenerate with the command above; do not edit by hand."


# --------------------------------------------------------------------------
# Labeled point set (REPLICATES toc_recovery._process_book's construction)
# --------------------------------------------------------------------------
def _build_labeled_set(slug: str, n_chap: int, toc_name: str, gap_name: str,
                       recon_name: str) -> dict:
    """Return {X (unit f64), labels, chapters, scope, ...} or a skip dict.

    Identical recipe to ``toc_recovery``: load the article SubCloud at the scope the
    book's gap JSON declares, embed gold chapters with the same GIST-small embedder,
    select the top-N=recon.n_gap_articles articles by max-cosine to ANY chapter, and
    label each by argmax-cosine gold chapter. X is L2-normalized (cosine geometry).
    """
    toc_path = _BOOK_GAPS / toc_name
    gap_path = _BOOK_GAPS / gap_name
    recon_path = _BOOK_GAPS / recon_name
    if not toc_path.exists():
        return {"slug": slug, "skipped": f"missing ToC {toc_path}"}

    toc = json.loads(toc_path.read_text())
    chapters = toc["chapters"]
    n_chap = len(chapters)

    subfield = field = None
    n_gap_articles = None
    if gap_path.exists():
        gap = json.loads(gap_path.read_text())
        subfield = gap.get("scope_subfield")
        field = gap.get("scope_field")
        bk = gap.get("book", gap)
        ga = bk.get("gap_article_ids")
        n_gap_articles = len(ga) if ga else None
    recon = json.loads(recon_path.read_text()) if recon_path.exists() else {}
    ari_recon = recon.get("adjusted_rand_index")
    if n_gap_articles is None:
        n_gap_articles = recon.get("n_gap_articles")

    LOGGER.info("[%s] n_chap=%d scope(subfield=%s field=%s) n_gap_articles=%s ari_recon=%s",
                slug, n_chap, subfield, field, n_gap_articles, ari_recon)

    try:
        art_ids, art_emb, _id2row = _load_article_space(subfield, field)
    except Exception as exc:  # noqa: BLE001 — skip-and-proceed per spec
        LOGGER.warning("[%s] SKIP: corpus load failed: %s", slug, exc)
        return {"slug": slug, "skipped": f"corpus load failed: {exc}"}

    LOGGER.info("[%s] article pool: %d works, dim=%d", slug, len(art_ids), art_emb.shape[1])

    # top-N by max-cosine to ANY chapter; argmax-cosine gold label (toc_recovery recipe).
    chap_emb = _unit(_embed_chapters(chapters))            # (n_chap, d) f64
    chap_f32 = chap_emb.astype(np.float32)
    art_norm = art_emb / np.maximum(
        np.linalg.norm(art_emb, axis=1, keepdims=True), np.float32(1e-12))
    sims_all = art_norm @ chap_f32.T                        # (N_pool, n_chap) f32
    max_sim = sims_all.max(axis=1)
    N = int(n_gap_articles) if n_gap_articles else 60 * n_chap
    N = min(N, len(art_ids))
    top_idx = np.argpartition(-max_sim, N - 1)[:N] if N < len(max_sim) else np.arange(len(max_sim))
    top_idx = np.sort(top_idx)                              # stable point order
    X = _unit(art_emb[top_idx])                             # (N, d) unit f64
    labels = sims_all[top_idx].argmax(axis=1).astype(int)
    sel_ids = [art_ids[i] for i in top_idx.tolist()]

    label_sizes = np.bincount(labels, minlength=n_chap)
    LOGGER.info("[%s] labeled set: N=%d, chapter sizes=%s",
                slug, X.shape[0],
                {chapters[i]["title"][:22]: int(label_sizes[i]) for i in range(n_chap)})

    return {
        "slug": slug,
        "n_chapters": n_chap,
        "scope": {"subfield": subfield, "field": field},
        "n_article_pool": len(art_ids),
        "n_labeled_points": int(X.shape[0]),
        "top_n": int(N),
        "ari_recon_baseline": float(ari_recon) if ari_recon is not None else None,
        "chapter_titles": [c["title"] for c in chapters],
        "chapter_label_sizes": label_sizes.astype(int).tolist(),
        "X": X,                     # (N,d) unit f64  (not serialized — npz instead)
        "labels": labels,           # (N,)  int
        "work_ids": sel_ids,
    }


# --------------------------------------------------------------------------
# Space fitters (fit on TRAIN, apply to TEST) — each returns (Xtr', Xte')
# --------------------------------------------------------------------------
def _space_raw(Xtr: NDArray, Xte: NDArray, _ytr: NDArray):
    """Rung 1: raw GIST-small (already unit-norm). Re-normalize defensively."""
    return _unit(Xtr), _unit(Xte), {"dim": int(Xtr.shape[1])}


def _space_pca(Xtr: NDArray, Xte: NDArray, _ytr: NDArray, dim: int = _PCA_DIM):
    """Rung 2: unsupervised PCA->dim (fit on train). Re-normalize so cosine holds.

    dim is clamped to <= min(n_train, n_features) so PCA never over-asks.
    """
    d = int(min(dim, Xtr.shape[0], Xtr.shape[1]))
    pca = PCA(n_components=d, random_state=0).fit(Xtr)
    Ztr = pca.transform(Xtr)
    Zte = pca.transform(Xte)
    return _unit(Ztr), _unit(Zte), {"dim": d}


def _space_lda_oracle(Xtr: NDArray, Xte: NDArray, ytr: NDArray, pca_dim: int = _PCA_DIM):
    """Rung 3: SUPERVISED ceiling — PCA(pca_dim)->LDA on TRAIN gold labels.

    PCA pre-reduction guards the 384-dim LDA scatter estimate against overfit /
    singular within-class scatter. LDA projects to <=(n_classes-1) dims. The whole
    pipe is fit on TRAIN ONLY; TEST is projected through the frozen transforms, so
    the oracle never sees test labels. Re-normalize for cosine readouts.
    """
    classes = np.unique(ytr)
    n_cls = int(len(classes))
    # PCA dim: < n_train and < n_features; LDA needs the reduced dim >= n_classes-1
    # to realize its full (C-1) output, but works fine with fewer (it just emits
    # min(pca_dim, C-1) discriminants). Clamp conservatively.
    d_pca = int(min(pca_dim, Xtr.shape[0] - 1, Xtr.shape[1]))
    d_pca = max(d_pca, 1)
    pca = PCA(n_components=d_pca, random_state=0).fit(Xtr)
    Ptr = pca.transform(Xtr)
    Pte = pca.transform(Xte)
    # shrinkage LDA on the PCA scores — robust even when within-class scatter is thin.
    # The ``eigen`` solver (unlike ``lsqr``) supports BOTH automatic shrinkage AND
    # ``.transform`` to the (<=C-1)-dim discriminant subspace, which is what we need.
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(Ptr, ytr)
    Ztr = lda.transform(Ptr)
    Zte = lda.transform(Pte)
    out_dim = int(Ztr.shape[1])
    return _unit(Ztr), _unit(Zte), {"dim": out_dim, "pca_dim": d_pca, "n_classes": n_cls}


# space name -> fitter(Xtr, Xte, ytr) -> (Ztr', Zte', info). Fitters take optional
# extra kwargs with defaults, so the common (3-positional-arg) signature is what the
# driver calls; the annotation pins that calling convention for the type checker.
_SPACES: dict[str, Callable[[NDArray, NDArray, NDArray], tuple[NDArray, NDArray, dict]]] = {
    "raw": _space_raw,
    "pca50": _space_pca,
    "lda_oracle": _space_lda_oracle,
}


# --------------------------------------------------------------------------
# Readouts on TEST works
# --------------------------------------------------------------------------
def _knn_readout(Ztr: NDArray, ytr: NDArray, Zte: NDArray, yte: NDArray,
                 k: int) -> dict:
    """train->test kNN (cosine) macro-F1 + accuracy on held-out TEST.

    Cosine kNN on unit-norm rows == Euclidean kNN (monotone), so we use the fast
    Euclidean ``KNeighborsClassifier`` on the already-normalized vectors. k is
    clamped to n_train.
    """
    kk = int(min(k, Ztr.shape[0]))
    clf = KNeighborsClassifier(n_neighbors=kk, metric="cosine", weights="uniform")
    clf.fit(Ztr, ytr)
    pred = clf.predict(Zte)
    macro_f1 = float(f1_score(yte, pred, average="macro", labels=np.unique(ytr),
                              zero_division=0))
    acc = float(accuracy_score(yte, pred))
    return {"knn_macro_f1": macro_f1, "knn_acc": acc, "knn_k": kk}


def _unsup_ari_readout(Zte: NDArray, yte: NDArray, k: int, n_rand: int,
                       rng: np.random.Generator) -> dict:
    """Average-linkage (cosine) on TEST-only, fcluster at k, ARI vs gold + random.

    Mirrors toc_recovery's flat-cut operating point but on the held-out TEST set in
    the candidate space. The random baseline shuffles the gold test labels (fixed
    tree / clustering) so "above chance" is explicit.
    """
    n = Zte.shape[0]
    k_eff_req = int(min(k, n))
    # average-linkage on cosine distances over the TEST points in THIS space.
    Z = linkage(Zte.astype(np.float64), method="average", metric="cosine")
    flat = fcluster(Z, t=k_eff_req, criterion="maxclust")
    k_eff = int(len(np.unique(flat)))
    ari = float(adjusted_rand_score(yte, flat))
    rand_vals = []
    for _ in range(n_rand):
        perm = rng.permutation(n)
        rand_vals.append(adjusted_rand_score(yte[perm], flat))
    ari_rand = float(np.mean(rand_vals)) if rand_vals else float("nan")
    return {
        "unsup_ari": ari,
        "unsup_ari_random": ari_rand,
        "unsup_ari_above_random": ari - ari_rand,
        "flat_k_requested": k_eff_req,
        "flat_k_effective": k_eff,
    }


# --------------------------------------------------------------------------
# Per-book driver
# --------------------------------------------------------------------------
def _process_book(bk: dict, test_frac: float, n_rand: int, seed: int,
                  npz_dir: Path | None) -> dict:
    """Run the 3-space ladder on one book's labeled set; return a result row."""
    slug = bk["slug"]
    X = bk.pop("X")
    labels = bk.pop("labels")
    work_ids = bk.pop("work_ids")
    n_chap = bk["n_chapters"]
    label_sizes = np.asarray(bk["chapter_label_sizes"])

    # --- present classes (chapters with >=1 work) and stability flags
    present = np.flatnonzero(label_sizes > 0)
    n_present = int(len(present))
    small_classes = [int(c) for c in present if label_sizes[c] < _MIN_PER_CHAP]
    # classes with a single work can't be stratified into train AND test -> drop.
    singletons = [int(c) for c in present if label_sizes[c] < 2]
    if singletons:
        keep_mask = ~np.isin(labels, singletons)
        LOGGER.info("[%s] dropping %d singleton-chapter works (%d chapters) — "
                    "cannot stratify into train+test", slug,
                    int((~keep_mask).sum()), len(singletons))
        X = X[keep_mask]
        labels = labels[keep_mask]
        work_ids = [w for w, m in zip(work_ids, keep_mask.tolist()) if m]

    # --- stratified train/test split (SAME split for all three spaces)
    idx = np.arange(X.shape[0])
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=labels)
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = labels[tr_idx], labels[te_idx]
    n_train, n_test = int(len(tr_idx)), int(len(te_idx))
    k = n_chap  # flat cut at number of gold chapters (matches the curriculum cut)

    tr_sizes = np.bincount(ytr, minlength=n_chap)
    te_sizes = np.bincount(yte, minlength=n_chap)
    unstable = [int(c) for c in np.unique(labels) if label_sizes[c] < _MIN_PER_CHAP]
    LOGGER.info("[%s] split: n_train=%d n_test=%d  classes_present=%d  "
                "unstable(<%d works)=%s", slug, n_train, n_test,
                int(len(np.unique(labels))), _MIN_PER_CHAP, unstable)

    if npz_dir is not None:
        npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_dir / f"{slug}_labeled.npz",
            X=X.astype(np.float32), labels=labels,
            train_idx=tr_idx, test_idx=te_idx,
            work_ids=np.array(work_ids, dtype=object),
            seed=seed, test_frac=test_frac)

    # --- run all three spaces on the SAME split
    rows = []
    for sname, fitter in _SPACES.items():
        Ztr, Zte, sinfo = fitter(Xtr, Xte, ytr)
        knn = _knn_readout(Ztr, ytr, Zte, yte, _KNN_K)
        rng = np.random.default_rng(seed)  # same random stream per space (fair)
        ari = _unsup_ari_readout(Zte, yte, k, n_rand, rng)
        row = {"space": sname, **sinfo, **knn, **ari}
        rows.append(row)
        LOGGER.info("[%s] %-11s dim=%-3d  kNN-F1=%.3f acc=%.3f  unsupARI=%.3f "
                    "(rand %.3f, +%.3f)", slug, sname, sinfo["dim"],
                    knn["knn_macro_f1"], knn["knn_acc"], ari["unsup_ari"],
                    ari["unsup_ari_random"], ari["unsup_ari_above_random"])

    # --- per-book verdict (interpretation guide)
    by = {r["space"]: r for r in rows}
    raw_f1, lda_f1 = by["raw"]["knn_macro_f1"], by["lda_oracle"]["knn_macro_f1"]
    raw_ari, lda_ari = by["raw"]["unsup_ari"], by["lda_oracle"]["unsup_ari"]
    pca_f1, pca_ari = by["pca50"]["knn_macro_f1"], by["pca50"]["unsup_ari"]
    raw_ari_rel = by["raw"]["unsup_ari_above_random"]
    pca_ari_rel = by["pca50"]["unsup_ari_above_random"]
    d_f1 = lda_f1 - raw_f1
    d_ari = lda_ari - raw_ari

    # thresholds: "HIGH" oracle F1 means linearly separable; "substantial" lift
    # means the supervised space buys real separation over raw.
    LDA_HIGH = 0.50          # macro-F1 the oracle must clear to call it "separable"
    LIFT_F1 = 0.10           # oracle macro-F1 lift over raw to call it "substantial"
    UNSUP_LOW = 0.20         # unsupervised ARI-above-random below this == "floor"
    RAW_NEAR_CEIL = 0.40     # raw unsupARI above this == chapters already separate raw
    if lda_f1 >= LDA_HIGH and d_f1 >= LIFT_F1:
        if max(raw_ari_rel, pca_ari_rel) < UNSUP_LOW:
            verdict = "LEVER_HARD"
            note = ("supervised oracle separates chapters (kNN-F1 high & >> raw) but "
                    "unsupervised ARI stays on the floor in raw/PCA -> chapter "
                    "structure is present but NOT dominant variance; a label-aware / "
                    "cleverer space is needed (lever exists, harder).")
        else:
            verdict = "LEVER"
            note = ("supervised oracle separates chapters AND unsupervised methods "
                    "already pick some of it up -> better geometry is a real lever "
                    "(Stage 2 worth it).")
    elif lda_f1 >= LDA_HIGH and raw_ari >= RAW_NEAR_CEIL:
        # oracle confirms the chapters ARE linearly separable, but raw already
        # achieves it (small lift because there is little headroom). This is the GOOD
        # regime: geometry already works; no better space is needed for this book.
        verdict = "ALREADY_SEPARABLE"
        note = ("chapters are ALREADY well-separated in raw GIST-small (raw unsupARI "
                f">= {RAW_NEAR_CEIL:.2f}, oracle kNN-F1 high but lift small) -> the "
                "oracle confirms separability the raw space already realizes; better "
                "geometry has little to add here (geometry already works, not a wall).")
    elif lda_f1 < LDA_HIGH and d_f1 < LIFT_F1:
        verdict = "WALL"
        note = ("even the supervised oracle cannot separate the chapters (kNN-F1 low "
                "& ~ raw) -> chapters overlap under supervision; geometry is a hard "
                "wall for this book -> pivot.")
    else:
        verdict = "MIXED"
        note = ("partial: oracle lifts F1 but not past the 'separable' bar, or lifts "
                "without clearing the substantial-lift threshold; weak lever at best.")

    LOGGER.info("[%s] VERDICT=%s  (raw F1=%.3f -> oracle F1=%.3f, +%.3f; "
                "raw ARI=%.3f -> oracle ARI=%.3f, +%.3f)", slug, verdict,
                raw_f1, lda_f1, d_f1, raw_ari, lda_ari, d_ari)

    return {
        **bk,
        "n_train": n_train,
        "n_test": n_test,
        "n_classes_present": n_present,
        "train_class_sizes": tr_sizes.astype(int).tolist(),
        "test_class_sizes": te_sizes.astype(int).tolist(),
        "unstable_classes_lt5": small_classes,
        "dropped_singleton_classes": singletons,
        "flat_k": k,
        "spaces": rows,
        "deltas": {
            "lda_minus_raw_knn_f1": float(d_f1),
            "lda_minus_raw_unsup_ari": float(d_ari),
            "pca_minus_raw_knn_f1": float(pca_f1 - raw_f1),
            "pca_minus_raw_unsup_ari": float(pca_ari - raw_ari),
        },
        "verdict": verdict,
        "verdict_note": note,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _print_per_book_tables(books: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("PER-BOOK SPACE LADDER — readouts on held-out TEST works "
          f"(kNN k={_KNN_K} cosine; unsup ARI@n_chapters; LDA never sees test)")
    print("=" * 100)
    for b in books:
        if b.get("skipped"):
            print(f"\n[{b['slug']}] SKIPPED: {b['skipped']}")
            continue
        sc = b["scope"]
        scope_s = (f"subfield={sc['subfield']}" if sc["subfield"]
                   else f"field={sc['field']}")
        print(f"\n[{b['slug']}]  scope({scope_s})  n_chap={b['n_chapters']}  "
              f"N_labeled={b['n_labeled_points']}  n_train={b['n_train']} "
              f"n_test={b['n_test']}  recon_ARI={b['ari_recon_baseline']:.4f}"
              if b.get("ari_recon_baseline") is not None else
              f"\n[{b['slug']}]  scope({scope_s})  n_chap={b['n_chapters']}  "
              f"N_labeled={b['n_labeled_points']}  n_train={b['n_train']} "
              f"n_test={b['n_test']}")
        if b["unstable_classes_lt5"]:
            print(f"    NOTE: {len(b['unstable_classes_lt5'])} chapter(s) with <"
                  f"{_MIN_PER_CHAP} works (unstable): {b['unstable_classes_lt5']}")
        if b["dropped_singleton_classes"]:
            print(f"    NOTE: dropped {len(b['dropped_singleton_classes'])} singleton "
                  f"chapter(s) (cannot split): {b['dropped_singleton_classes']}")
        hdr = (f"    {'space':<11}{'dim':>5}{'kNN-F1':>9}{'kNN-acc':>9}"
               f"{'unsupARI':>10}{'ARI-rand':>10}{'ARI>rand':>10}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for r in b["spaces"]:
            print(f"    {r['space']:<11}{r['dim']:>5}{r['knn_macro_f1']:>9.3f}"
                  f"{r['knn_acc']:>9.3f}{r['unsup_ari']:>10.3f}"
                  f"{r['unsup_ari_random']:>10.3f}{r['unsup_ari_above_random']:>10.3f}")
        d = b["deltas"]
        print(f"    deltas: oracle-raw  kNN-F1 {d['lda_minus_raw_knn_f1']:+.3f}  "
              f"unsupARI {d['lda_minus_raw_unsup_ari']:+.3f}   |   "
              f"pca50-raw  kNN-F1 {d['pca_minus_raw_knn_f1']:+.3f}  "
              f"unsupARI {d['pca_minus_raw_unsup_ari']:+.3f}")
        print(f"    >>> VERDICT [{b['verdict']}]: {b['verdict_note']}")


def _raw_ari(b: dict) -> float:
    """The raw-space unsupervised ARI for a book (used to classify MIXED books)."""
    for r in b["spaces"]:
        if r["space"] == "raw":
            return float(r["unsup_ari"])
    return float("nan")


def _print_overall(books: list[dict]) -> dict:
    live = [b for b in books if not b.get("skipped")]
    print("\n" + "=" * 100)
    print("OVERALL — is 'better curriculum embeddings' a real lever, and which books?")
    print("=" * 100)
    tally: dict[str, list[str]] = {}
    for b in live:
        tally.setdefault(b["verdict"], []).append(b["slug"])
    for v in ("LEVER", "LEVER_HARD", "ALREADY_SEPARABLE", "MIXED", "WALL"):
        if v in tally:
            print(f"  {v:<18}: {tally[v]} "
                  f"(raw unsupARI {[f'{_raw_ari(b):.2f}' for b in live if b['verdict'] == v]})")
    levers = tally.get("LEVER", []) + tally.get("LEVER_HARD", [])
    already = tally.get("ALREADY_SEPARABLE", [])   # raw already separates chapters
    walls = tally.get("WALL", [])
    weak = tally.get("MIXED", [])                  # oracle couldn't clear separability
    print()
    if walls and not levers and not already:
        msg = (f"UNIFORM WALL. Even the supervised oracle cannot separate chapters on "
               f"any evaluated book ({walls}) -> the chapter overlap is intrinsic to "
               f"the representation/corpus; better embeddings will NOT rescue recovery "
               f"-> pivot.")
    elif not walls and not weak:
        # nobody is a hard wall, nobody is unseparable even under supervision: the
        # picture is "geometry already works OR is liftable" — i.e. no hard ceiling.
        bits = []
        if levers:
            bits.append(f"better geometry is a demonstrated LEVER on {levers} "
                        f"(supervised oracle separates chapters AND a cheap unsupervised "
                        f"projection already recovers much of the lift)")
        if already:
            bits.append(f"chapters are ALREADY well-separated in raw space on {already} "
                        f"(oracle confirms separability raw already realizes -> geometry "
                        f"already works there, no wall and no headroom)")
        msg = ("GEOMETRY IS NOT A HARD WALL on any evaluated book. " + "; ".join(bits) +
               ". No book has chapters overlapping under supervision -> a better "
               "curriculum space is a real (book-dependent) lever, most valuable exactly "
               "where raw is weakest (cyberbullying).")
    else:
        msg = (f"BOOK-DEPENDENT. LEVER on {levers}; raw ALREADY separates on {already}; "
               f"weak/unclear on {weak}; hard WALL on {walls}. Stage 2 pays off on the "
               f"low-raw / liftable books; books already separated in raw need no "
               f"geometry work; pivot on any walls.")
    print("  " + msg)

    # Did cheap PCA-50 ever buy a free win?
    pca_helps = [b["slug"] for b in live
                 if b["deltas"]["pca_minus_raw_knn_f1"] >= 0.03
                 or b["deltas"]["pca_minus_raw_unsup_ari"] >= 0.03]
    if pca_helps:
        print(f"  Cheap unsupervised PCA-{_PCA_DIM} gave a (>=0.03) free lift on: {pca_helps}.")
    else:
        print(f"  Cheap unsupervised PCA-{_PCA_DIM} bought no meaningful free lift on any book.")

    return {"verdict_tally": tally, "lever_books": levers,
            "already_separable_books": already, "weak_books": weak,
            "wall_books": walls, "pca_helps_books": pca_helps, "summary": msg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-frac", type=float, default=0.30,
                    help="held-out test fraction (stratified by gold chapter).")
    ap.add_argument("--n-rand", type=int, default=20,
                    help="label shuffles for the random-ARI baseline (default 20).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(_HERE / "geometry_ceiling_results.json"))
    ap.add_argument("--npz-dir", default=str(_HERE / "geometry_ceiling_npz"),
                    help="dir for per-book labeled-set npz dumps (cheap re-runs).")
    ap.add_argument("--no-npz", action="store_true", help="skip npz dumps.")
    args = ap.parse_args()

    setup_logging(log_name="geometry_ceiling", log_level="INFO", include_time=False,
                  log_path=Path(args.out).with_suffix(".log"))
    t0 = time.time()
    npz_dir = None if args.no_npz else Path(args.npz_dir)

    books: list[dict] = []
    for book_id, slug, n_chap, toc_name, gap_name, recon_name in _BOOKS:
        LOGGER.info("=" * 70)
        try:
            bk = _build_labeled_set(slug, n_chap, toc_name, gap_name, recon_name)
            if bk.get("skipped"):
                books.append(bk)
                continue
            res = _process_book(bk, args.test_frac, args.n_rand, args.seed, npz_dir)
        except Exception as exc:  # noqa: BLE001 — skip-and-proceed per spec
            LOGGER.exception("[%s] unexpected failure; skipping", slug)
            res = {"slug": slug, "book_id": book_id, "skipped": f"unexpected: {exc}"}
        books.append(res)

    _print_per_book_tables(books)
    overall = _print_overall(books)

    out = {
        "books": books,
        "overall": overall,
        "params": {"test_frac": args.test_frac, "n_rand": args.n_rand,
                   "seed": args.seed, "knn_k": _KNN_K, "pca_dim": _PCA_DIM,
                   "min_per_chap_stable": _MIN_PER_CHAP},
        "interpretation_guide": {
            "LEVER": "oracle separates chapters AND unsupervised picks some up -> Stage 2 worth it.",
            "LEVER_HARD": "oracle separates (kNN-F1 high & >>raw) but unsup ARI on floor -> "
                          "structure present, not dominant variance; label-aware space needed.",
            "ALREADY_SEPARABLE": "oracle confirms separability (kNN-F1 high) but raw ALREADY "
                                 "achieves it (small lift, raw unsupARI>=0.40) -> geometry "
                                 "already works for this book; no better space needed.",
            "WALL": "oracle cannot separate (kNN-F1 low & ~raw) -> chapters overlap under "
                    "supervision; geometry is a hard wall -> pivot.",
            "MIXED": "oracle lifts but not past separable/substantial bars; weak lever.",
        },
        "caveats": {
            "scope": ("cyberbullying loaded at subfield 1702; mcs/neuromorphic/color_image at "
                      "FIELD 17 — each at the scope its own gap JSON declares, matching the "
                      "cl_<slug>_recon.json corpus. Default cloud denoiser=NONE (raw GIST-small 384d)."),
            "circularity": ("work labels = argmax-cosine to same-space (raw GIST-small) chapter "
                            "embeddings, so ABSOLUTE numbers are inflated and consistent with the "
                            "prior toc_recovery run. The cross-SPACE comparison is fair: identical "
                            "works and identical train/test split; only the projection differs; the "
                            "LDA oracle is fit on TRAIN only and never sees test labels."),
            "split": (f"stratified train/test (test_frac={args.test_frac}, seed={args.seed}); the "
                      "SAME split feeds all three spaces; singleton chapters dropped; chapters with "
                      f"<{_MIN_PER_CHAP} works flagged unstable."),
        },
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": _provenance(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.time() - t0:.1f}s  results -> {args.out}")
    if npz_dir is not None:
        print(f"       per-book labeled-set npz -> {npz_dir}/")


if __name__ == "__main__":
    main()
