"""Keyword-concept seeding: a soft target-works weight vector from a single seed text.

docs/backlog/curriculum-keyword-concept-seeding.md. The embedding-ball gap is empty for any
non-isolated book (Koehn/Jurafsky → gap_size 0). A keyword-concept exists for every book: seed
from its text (a ToC), iterate the nonlinear W2K↔K2W fixed point (a soft FCA formal concept),
and take the fixed-point K2W scores as the soft target-works weight vector that REPLACES the ball
at the curriculum's personalization interface.

  W2K (soft intent): lift = log[ P(t|concept) / P(t|FIELD) ] — distinctive vs the whole field.
  K2W (soft extent): score works by keyword-match; take top-m.

Backend (``--backend full``, default): K2W ranges over the WHOLE field via the cached work×term
index (``field_term_index.build_field_term_index``) — not an embedding-nearest prefilter that
funnels every chapter into the pool's dominant basin. ``--backend nearest`` keeps the
embedding-nearest-N candidate pool (cheaper, no cached index, but basin-funnels).

A broad book = the UNION of its chapter-concepts (``--per-chapter``): one whole-book fixed point
collapses to the single tightest sub-basin (Jurafsky → all speech/ASR), so seed each chapter,
iterate to its basin, union (max of per-chapter-normalized scores). Use few ``--iters`` (the fixed
point drifts off-topic; early iterates stay on the seed).

Emits a report (per-chapter keywords, top works w/ types, the book-attribute lift = the spin-off
book-deficit gap signal) + ``concept_target_weights.json`` (work_id → soft weight) for the
curriculum's target-works personalization.

Example:
  python -m artifacts_scripts.curriculum.gap_keyword_concept \\
      --toc-json artifacts/book_gaps/jurafsky_toc.json --field 17 --backend full \\
      --per-chapter --iters 3 --out artifacts/book_gaps/jurafsky_concept_full
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from bookgap.util import LOGGER, setup_logging

_MODULE = "artifacts_scripts.curriculum.gap_keyword_concept"
_TOK = re.compile(r"[a-z0-9]{3,}")
_STOP = set("the a an of and or to in for on with is are be by we our this that as at from using".split())
_EPS = 0.5


def _terms(text: str) -> list:
    t = [w for w in _TOK.findall((text or "").lower()) if w not in _STOP]
    return t + [f"{t[i]} {t[i+1]}" for i in range(len(t) - 1)]   # space-join to match sklearn


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toc-json", type=Path, help="book ToC (chapters[].title/description) as the seed")
    ap.add_argument("--seed-text", type=str, help="raw seed text (alternative to --toc-json)")
    ap.add_argument("--field", default="17")
    ap.add_argument("--backend", choices=["full", "nearest"], default="full",
                    help="full = K2W over the whole field via the cached work×term index; "
                         "nearest = embedding-nearest-N candidate pool (basin-funnels).")
    ap.add_argument("--n-candidates", type=int, default=20000, help="[nearest] candidate pool size")
    ap.add_argument("--bg-sample", type=int, default=80000, help="[nearest] random works for the lift bg DF")
    ap.add_argument("--min-df", type=int, default=20, help="[full] index min_df (cache key)")
    ap.add_argument("--max-features", type=int, default=1_000_000, help="[full] index max_features (cache key)")
    ap.add_argument("--snapshot-date", default="2025-11-06", help="[full] index cache key")
    ap.add_argument("--top-k", type=int, default=40, help="keywords per W2K step")
    ap.add_argument("--top-m", type=int, default=400, help="works per K2W step")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--union-floor", type=float, default=0.15, help="[per-chapter] keep works whose best "
                    "per-chapter-normalized score >= this")
    ap.add_argument("--per-chapter", action="store_true", help="union of per-chapter concepts (preserves breadth)")
    ap.add_argument("--drop-incoherent-basins", action="store_true",
                    help="[per-chapter] corrected #1: drop basins that are keyword-coherent but "
                         "embedding-INCOHERENT (content-scattered) — the paratext/proceedings basin — by "
                         "comparing each basin's internal embedding cohesion to a random same-size null. "
                         "Removes the generic hub that gates the #2 coherence-merge.")
    ap.add_argument("--distinctiveness", choices=["none", "emptysink", "peakedness", "maxresp"], default="none",
                    help="[per-chapter] down-weight each work's selection weight by how distinctively it "
                         "belongs to ONE basin vs being ambient/spread (drops paratext at the source). All "
                         "forms derive from the SAME vMF-mixture-with-null responsibilities over the basin "
                         "embedding centroids: emptysink=1-r_∅ (canonical mixture-with-background posterior), "
                         "peakedness=1-H(r)/logK (negentropy), maxresp=max_k r_k.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    setup_logging(log_name="kw_concept", log_level="INFO", include_time=False)

    cfg = OmegaConf.structured(CurriculumConfig())
    cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); titles = cloud.work_titles(); abstracts = cloud.work_abstracts()
    try:
        wtypes = list(cloud.work_types())
    except Exception:
        wtypes = None

    # seed text(s)
    if args.toc_json:
        toc = json.loads(args.toc_json.read_text()); chs = toc["chapters"]
        seed_text = ". ".join(f"{c.get('title','')} {c.get('description','')}" for c in chs)
        seed_name = toc.get("title", args.toc_json.stem)
    else:
        chs = [{"title": "seed", "description": args.seed_text}]
        seed_text = args.seed_text; seed_name = "seed_text"

    # ---- backend: define seed_lift(text)->{term:wt}, w2k(S)->{term:wt}, k2w(kw)->(set, score) ----
    if args.backend == "full":
        from artifacts_scripts.curriculum.field_term_index import build_field_term_index
        from sklearn.feature_extraction.text import CountVectorizer
        X, vocab, field_df = build_field_term_index(
            args.field, args.snapshot_date, args.min_df, args.max_features)
        N = X.shape[0]; BG = N
        Xc = X.tocsc()
        term2id = {t: i for i, t in enumerate(vocab)}
        cv = CountVectorizer(vocabulary=term2id, ngram_range=(1, 2),
                             token_pattern=r"(?u)\b[a-z0-9]{3,}\b", lowercase=True,
                             stop_words=list(_STOP), binary=True)
        cand = np.arange(N)                                   # identity: positions ARE work indices
        LOGGER.info("full-index backend: X %s, vocab=%d, %d works", X.shape, len(vocab), N)

        def _lift_from_counts(ids, cnt, nS):
            kw = {}
            for tid, c in zip(ids, cnt):
                pS = (c + _EPS) / (nS + _EPS); pF = (field_df[tid] + _EPS) / (BG + _EPS)
                lift = np.log(pS / pF)
                if lift > 0:
                    kw[tid] = lift * np.log1p(c)
            return dict(sorted(kw.items(), key=lambda x: -x[1])[: args.top_k])

        def seed_lift(text):
            v = cv.transform([text]); ids = v.indices
            return _lift_from_counts(ids, np.ones(len(ids)), 1)

        def w2k(S):
            sub = X[list(S)]; cnt = np.asarray(sub.sum(0)).ravel()
            ids = cnt.nonzero()[0]
            return _lift_from_counts(ids, cnt[ids], len(S))

        def k2w(kw):
            ids = np.fromiter(kw.keys(), dtype=np.int64); w = np.fromiter(kw.values(), dtype=np.float64)
            score = np.asarray(Xc[:, ids] @ w).ravel()
            m = min(args.top_m, len(score))
            top = set(np.argpartition(score, -m)[-m:].tolist())
            return top, score

        def term_name(tid):
            return vocab[tid]
    else:
        E = np.asarray(cloud.embeddings(), dtype=np.float32)
        embedder = instantiate(cfg.experiment.cloud.embedder)
        with embedder as _e:
            seed_vec = _unit(np.asarray(_e.embed_texts([("", seed_text)]), dtype=np.float32)[0])
        cand = np.argsort(E @ seed_vec)[::-1][: args.n_candidates]
        cand_sets = [set(_terms(f"{titles[i]} {abstracts[i] or ''}")) for i in cand]
        rng = np.random.default_rng(0)
        bg = rng.choice(len(wid), min(args.bg_sample, len(wid)), replace=False)
        field_df = {}
        for i in bg:
            for t in set(_terms(f"{titles[i]} {abstracts[i] or ''}")):
                field_df[t] = field_df.get(t, 0) + 1
        BG = len(bg)
        inv = {}
        for d, s in enumerate(cand_sets):
            for t in s:
                inv.setdefault(t, []).append(d)
        LOGGER.info("nearest backend: %d candidates, field-bg %d", len(cand), BG)

        def _lift(cnt, nS):
            kw = {}
            for t, c in cnt.items():
                pS = (c + _EPS) / (nS + _EPS); pF = (field_df.get(t, 0) + _EPS) / (BG + _EPS)
                lift = np.log(pS / pF)
                if lift > 0:
                    kw[t] = lift * np.log1p(c)
            return dict(sorted(kw.items(), key=lambda x: -x[1])[: args.top_k])

        def seed_lift(text):
            cnt = {}
            for t in _terms(text):
                cnt[t] = cnt.get(t, 0) + 1
            return _lift(cnt, 1)

        def w2k(S):
            cnt = {}
            for d in S:
                for t in cand_sets[d]:
                    cnt[t] = cnt.get(t, 0) + 1
            return _lift(cnt, len(S))

        def k2w(kw):
            score = np.zeros(len(cand))
            for t, wt in kw.items():
                for d in inv.get(t, []):
                    score[d] += wt
            top = set(np.argsort(score)[::-1][: args.top_m].tolist())
            return top, score

        def term_name(t):
            return t

    # ---- concept = W2K<->K2W fixed point (per-chapter union, or single whole-seed) ----
    def find_concept(text):
        S, sc = k2w(seed_lift(text)); traj = []
        for _ in range(args.iters):
            Snew, sc = k2w(w2k(S))
            traj.append(len(S & Snew) / len(S | Snew) if (S or Snew) else 1.0)
            S = Snew
        return S, sc, traj

    n_cand = len(cand)
    if args.per_chapter and args.toc_json:
        agg = np.zeros(n_cand); per_ch = []; basins: set = set(); basin_sets: list = []
        for c in chs:
            ctext = f"{c.get('title','')} {c.get('description','')}".strip()
            Sc, sc, _ = find_concept(ctext)
            mx = sc.max()
            if mx > 0:
                agg = np.maximum(agg, sc / mx)
            basins |= Sc                                      # the per-chapter basin (top-m)
            basin_sets.append(Sc)
            per_ch.append((c.get('title', '')[:42], [term_name(t) for t in list(w2k(Sc))[:6]]))
        score = agg
        # FOCUSED target = union of per-chapter basins (≤ n_chapters·top_m), not the broad
        # union_floor support over the full score — keeps the personalization concentrated.
        S = set(basins)
        kw_final = w2k(S); traj = []
        LOGGER.info("per-chapter union (%d chapters) -> focused |concept|=%d (∪ basins)", len(chs), len(S))
        for t, k in per_ch:
            LOGGER.info("  ch '%s': %s", t, k)

        # ---- #1: concept-distinctiveness down-weighting (drops ambient/paratext at the source) ----
        if args.distinctiveness != "none":
            from artifacts_scripts.curriculum.gap_fuzzy_topics import soft_responsibilities, _unit as _u
            Eall = E if args.backend == "nearest" else np.asarray(cloud.embeddings(), dtype=np.float32)
            cw = sorted(S)                                    # concept work positions
            Cb = np.stack([_u(Eall[cand[list(b)]].mean(0)) for b in basin_sets])   # basin embed centroids
            Ec = _u(Eall[cand[cw]].astype(np.float64))
            w = np.array([score[d] for d in cw], dtype=np.float64)                 # concept-score = ambient mass
            ambient = _u(np.average(Ec, axis=0, weights=w if w.sum() > 0 else None))
            R, tau = soft_responsibilities(Ec, Cb, args.soft_perplexity if hasattr(args, "soft_perplexity") else 1.5,
                                           ambient=ambient)   # (|cw| x n_chapters), rows sum to 1-r_∅
            rsum = R.sum(1)                                   # = 1 - r_∅ (concept membership vs background)
            rn = R / np.maximum(rsum[:, None], 1e-12)         # renormalized over basins (drop ∅)
            if args.distinctiveness == "emptysink":
                dist = rsum                                   # canonical: posterior of a real chapter vs ∅
            elif args.distinctiveness == "maxresp":
                dist = rn.max(1)
            else:  # peakedness (negentropy)
                H = -np.sum(np.where(rn > 0, rn * np.log(rn), 0.0), axis=1)
                dist = 1.0 - H / np.log(R.shape[1])
            d_full = np.zeros(n_cand); d_full[cw] = dist
            score = score * d_full                            # down-weight ambient works
            order_d = np.argsort(dist)
            LOGGER.info("distinctiveness=%s: mean=%.3f; LOW (ambient/paratext): %s",
                        args.distinctiveness, float(dist.mean()),
                        [titles[cand[cw[i]]][:40] for i in order_d[:4]])
            LOGGER.info("  HIGH (distinctive): %s", [titles[cand[cw[i]]][:40] for i in order_d[::-1][:4]])
    else:
        S, score, traj = find_concept(seed_text)
        kw_final = w2k(S)
        LOGGER.info("converged: step Jaccard %s", ["%.2f" % j for j in traj])

    # ---- corrected #1: drop embedding-INCOHERENT basins (paratext: venue-coherent, content-scattered) ----
    if args.drop_incoherent_basins and args.per_chapter:
        from artifacts_scripts.curriculum.gap_fuzzy_topics import _unit as _u
        Eall = E if args.backend == "nearest" else np.asarray(cloud.embeddings(), dtype=np.float32)
        rng = np.random.default_rng(0)
        crows = cand[sorted(S)]                                   # all concept work indices
        coh = np.zeros(len(basin_sets)); nul = np.zeros(len(basin_sets))
        for k, Sc in enumerate(basin_sets):
            Eb = _u(Eall[cand[list(Sc)]].astype(np.float64))
            coh[k] = float((Eb @ _u(Eb.mean(0))).mean())          # mean cos-to-centroid (cohesion)
            samp = rng.choice(len(crows), min(len(Sc), len(crows)), replace=False)
            Er = _u(Eall[crows[samp]].astype(np.float64))
            nul[k] = float((Er @ _u(Er.mean(0))).mean())          # random same-size null
        keep = coh > nul                                          # basin tighter than a random set
        rank = np.argsort(coh)
        LOGGER.info("basin embedding cohesion vs random-null (lowest first):")
        for k in rank:
            LOGGER.info("  %-26s coh=%.3f null=%.3f %s", per_ch[k][0][:26], coh[k], nul[k],
                        "DROP" if not keep[k] else "")
        kept_works = set().union(*(basin_sets[k] for k in range(len(basin_sets)) if keep[k])) if keep.any() else set(S)
        dropped_ch = [per_ch[k][0] for k in range(len(basin_sets)) if not keep[k]]
        S = set(S) & kept_works
        LOGGER.info("dropped %d incoherent basin(s): %s -> concept now %d works",
                    int((~keep).sum()), dropped_ch, len(S))

    # ---- book-attribute lift (spin-off book-deficit gap signal) ----
    type_line = ""
    if wtypes is not None and S:
        from collections import Counter
        ct = Counter(wtypes[cand[d]] for d in S)
        c_book = (ct.get("book", 0) + ct.get("book-chapter", 0)) / len(S)
        # field base rate over a 100k sample
        rng = np.random.default_rng(1)
        samp = rng.choice(len(wid), min(100000, len(wid)), replace=False)
        bt = Counter(wtypes[i] for i in samp)
        f_book = (bt.get("book", 0) + bt.get("book-chapter", 0)) / len(samp)
        book_lift = (c_book + 1e-9) / (f_book + 1e-9)
        type_line = (f"concept types {dict(ct)} | book_lift P(book|concept)/P(book|field) = "
                     f"{c_book:.4f}/{f_book:.4f} = **{book_lift:.2f}** "
                     f"({'book-DEFICIT (gap-like)' if book_lift < 1 else 'book-rich'})")
        LOGGER.info(type_line)

    # ---- emit ----
    args.out.mkdir(parents=True, exist_ok=True)
    order = sorted(S, key=lambda d: -score[d])
    weights = {wid[cand[d]]: float(score[d]) for d in order}
    (args.out / "concept_target_weights.json").write_text(json.dumps(weights))
    # gap-json-compatible file so book_curriculum.py consumes the concept directly as its
    # target_work_ids personalization (this replaces the empty embedding-ball gap).
    gap_ids = [wid[cand[d]] for d in order]
    gap_wts = [float(score[d]) for d in order]            # aligned soft concept weights
    (args.out / "concept_gap.json").write_text(json.dumps({
        "scope_field": str(args.field), "scope_subfield": None, "method": "keyword_concept",
        "book": {"book": seed_name, "book_id": seed_name, "gap_size": len(gap_ids),
                 "gap_article_ids": gap_ids, "gap_article_weights": gap_wts}}))
    cmd = f"python -m {_MODULE} " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    lines = [f"# Keyword-concept — {seed_name}", "",
             f"backend={args.backend} · |concept|={len(S)} · top_k={args.top_k} · top_m={args.top_m} · "
             f"iters={args.iters} · per_chapter={args.per_chapter}", "",
             (type_line + "\n" if type_line else ""),
             "## Top concept keywords (lift vs field)", "",
             ", ".join(f"`{term_name(t)}`" for t in list(kw_final)[:40]), ""]
    if args.per_chapter and args.toc_json:
        lines += ["## Per-chapter basins", ""]
        lines += [f"- **{t}** — {', '.join(k)}" for t, k in per_ch] + [""]
    lines += ["## Top concept works (soft weight)", ""]
    for d in order[:25]:
        ty = f" [{wtypes[cand[d]]}]" if wtypes is not None else ""
        lines.append(f"{score[d]:.2f}  {titles[cand[d]][:90]}{ty}")
    lines += ["", f"---\n*Generated by `{cmd}`*  \n*Regenerate; do not edit by hand.*"]
    (args.out / "report.md").write_text("\n".join(lines))
    print("\n".join([l for l in lines if l][:18]))
    print(f"\nwrote {args.out}/report.md + concept_target_weights.json ({len(weights)} works)")


if __name__ == "__main__":
    main()
