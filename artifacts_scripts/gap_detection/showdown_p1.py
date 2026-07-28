"""Curriculum Showdown Phase 1 — blind 2x2 ToC tournament (3 topics x 3 arms).

Phase-0 froze the generator (curriculum_toc.py) + metric suite (toc_metrics.py).
This module runs the Phase-1 data plumbing around them:

  recount    — post-cutoff RE-CUT (gemma-4 cutoff Jan 2025): count T10028 at
               from_publication_date:2025-02-01 (+has_abstract, language:en);
               keep if >=8K, else T10181, else union. Counts only; the LOCK is
               recorded in the worklog BEFORE any pull.
  pull       — cursor-page the LOCKED slice -> cache/oa/showdown_postcutoff/
               (polite pool, resumable, every page cached). Select per brief
               + `type` (required by the t3 KEEP_TYPES cleaning step).
  corpus     — per-topic frozen embeds + fitted vectorizer/LSA caches:
               ds: re-run the Phase-0 joint embed (anchor-asserted) and cache
               the FITTED tfidf+svd so agent/hybrid section intents can be
               mapped into the SAME space; causal: T10845 standalone embed
               (t3 config, seed 42) + in-sub-slice edges from ds_showdown_refs;
               postcutoff: t3-cleaned standalone embed + edges from its own
               referenced_works.
  toc        — algorithmic ToCs for causal + postcutoff under the FROZEN rules
               (k in {5..10} for <15K-doc slices, sections {3..8}, register-
               filtered labels, exact min-forward-flow ordering); writes the
               judge-facing p1_<topic>_algo.md (neutral H1, no provenance) for
               all three topics (ds re-emitted from the Phase-0 pickle) plus
               the hybrid-arm skeleton files.

Everything seeded (SEED=42); CPU only (GPU belongs to the gemma pi runs).
Context hygiene: titles, top-terms, counts, numbers ONLY — never abstract text.
Run: uv run python -m artifacts_scripts.gap_detection.showdown_p1 <phase>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.book_audit import (
    API_BASE,
    KEEP_TYPES,
    LSA_DIM,
    MIN_ABSTRACT_CHARS,
    REPO,
    _get_json,
    _session,
    assemble_bg,
    load_book_fg,
    works_filter,
)
from artifacts_scripts.gap_detection.curriculum_toc import (
    DEP_SHARE,
    N_CHAPTER_TERMS,
    N_EXEMPLARS,
    N_SECTION_TERMS,
    SMALL_K_RANGE,
    TOC_PKL,
    _kmeans_sweep,
    _load_embed,
    _title_case,
    cluster_terms,
    iter_refs,
)
from artifacts_scripts.gap_detection.oa_corpus import decode_abstract, is_english, short_id
from artifacts_scripts.gap_detection.run_eval import _py
from artifacts_scripts.gap_detection.run_followups import RESULTS_PATH, load_banked
from artifacts_scripts.gap_detection.run_t3_book_audit import (
    COMMITTED_TOPICS,
    PUB_FROM,
    PUB_TO,
)
from artifacts_scripts.gap_detection.toc_metrics import (
    citation_flow,
    forward_violations,
    order_min_forward_flow,
    select_k,
)

_T0 = time.monotonic()
SEED = 42
PER_PAGE = 200

OUT_DIR = REPO / "artifacts/gap_detection_eval"
SHOW_DIR = OUT_DIR / "showdown"
CACHE_DIR = OUT_DIR / "cache"
PC_CACHE = REPO / "cache/oa/showdown_pc_2025"
COUNTS_CACHE = REPO / "cache/oa/showdown_topics"

# Post-cutoff re-cut (brief): gemma 4 documented cutoff = Jan 2025.
PC_FROM = "2025-02-01"
PC_PRIMARY = "T10028"        # Topic Modeling — Phase-0 committed pick
PC_FALLBACK = "T10181"       # NLP — Phase-0 runner-up
PC_MIN_WORKS = 8_000
# Brief-pinned select + `type` (required to apply the t3 KEEP_TYPES cleaning
# verbatim; recorded as the one select deviation in the worklog).
PC_SELECT = ("id,title,publication_date,abstract_inverted_index,"
             "referenced_works,primary_topic,type")

# Pre-committed topic statements (same text to EVERY arm and the judge).
TOPICS = {
    "datascience": {
        "title": "Data Science",
        "audience": "quantitative-major undergraduates (calculus, linear "
                    "algebra, programming)",
    },
    "causal": {
        "title": "Causal Inference",
        "audience": "graduate students / quantitative researchers entering "
                    "causal inference from a general stats background",
    },
    "postcutoff": {
        "title": "Topic Modeling",   # field as it stands in 2026 (post-2025-01 literature)
        "audience": "researchers/practitioners entering this literature in 2026",
    },
}
CAUSAL_TOPIC_ID = "T10845"


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def fit_embed(fg_texts: list[str], bg_texts: list[str], *,
              seed: int = SEED, lsa_dim: int = LSA_DIM) -> dict:
    """book_audit.embed_joint REPLICATED verbatim (same config, same seed),
    additionally returning the FITTED vectorizer+svd so NEW texts (agent/
    hybrid ToC section intents) can be mapped into the SAME frozen space via
    normalize(svd.transform(vec.transform(texts))). book_audit itself stays
    untouched (read-only reuse); equivalence is contract-tested
    (test_p1_fit_embed_matches_embed_joint) and anchor-asserted against the
    cached Phase-0 arrays in phase_corpus_ds."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = list(fg_texts) + list(bg_texts)
    vec = TfidfVectorizer(
        stop_words="english", lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=10, max_df=0.30, max_features=60_000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    vocab_all = vec.get_feature_names_out()
    k = min(lsa_dim, max(1, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Z = normalize(svd.fit_transform(X)).astype(np.float32)
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    kw_cols = np.sort(np.argsort(df)[::-1][: min(2_000, df.size)])
    K = (X[:, kw_cols] > 0).astype(np.uint8).tocsr()
    return {"Z": Z, "K": K, "vocab": vocab_all[kw_cols], "X": X,
            "tfidf_vocab": vocab_all,
            "evr": float(svd.explained_variance_ratio_.sum()),
            "vec": vec, "svd": svd}


def transform_texts(vec, svd, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map NEW texts into a frozen embed space. Returns (Z_new, nonzero_mask);
    rows with an all-zero TF-IDF image (no corpus vocabulary) get a zero
    vector and mask=False — callers DROP them from rep sets (a section whose
    intent shares no vocabulary with the corpus covers nothing)."""
    from sklearn.preprocessing import normalize

    X = vec.transform(texts)
    nz = np.asarray((X != 0).sum(axis=1)).ravel() > 0
    Z = svd.transform(X)
    norms = np.linalg.norm(Z, axis=1)
    ok = nz & (norms > 1e-9)
    Z = normalize(Z[ok]) if ok.any() else np.zeros((0, svd.n_components))
    out = np.zeros((len(texts), svd.n_components), dtype=np.float32)
    out[ok] = Z.astype(np.float32)
    return out, ok


def bank(sub: str, payload: dict) -> None:
    """Merge `payload` under results.json:task_showdown_p1.<sub>."""
    r = load_banked()
    cur = r.get("task_showdown_p1", {})
    cur[sub] = _py(payload)
    r["task_showdown_p1"] = cur
    RESULTS_PATH.write_text(json.dumps(r, indent=1))
    log(f"banked results.json:task_showdown_p1.{sub}")


def _cached_count(filt: str) -> int:
    COUNTS_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = COUNTS_CACHE / f"count_{hashlib.sha1(filt.encode()).hexdigest()[:10]}.json"
    if cpath.exists():
        return int(json.loads(cpath.read_text())["count"])
    data = _get_json(_session(), f"{API_BASE}/works",
                     {"filter": filt, "per-page": "1", "select": "id"}, log=log)
    n = int(data["meta"]["count"])
    cpath.write_text(json.dumps({"count": n, "filter": filt}))
    return n


def pc_filter(topic_ids: str) -> str:
    return (f"primary_topic.id:{topic_ids},from_publication_date:{PC_FROM},"
            "has_abstract:true,language:en")


# --------------------------------------------------------------- phase: recount
def phase_recount(args) -> None:
    rows = []
    for ids, label in ((PC_PRIMARY, "primary T10028 Topic Modeling"),
                       (PC_FALLBACK, "fallback T10181 NLP"),
                       (f"{PC_PRIMARY}|{PC_FALLBACK}", "union")):
        n = _cached_count(pc_filter(ids))
        rows.append({"ids": ids, "label": label, "count": n})
        log(f"  {label}: {n:,} works since {PC_FROM}")
    if rows[0]["count"] >= PC_MIN_WORKS:
        pick = rows[0]
    elif rows[1]["count"] >= PC_MIN_WORKS:
        pick = rows[1]
    else:
        pick = rows[2]
    log(f"RULE PICK: {pick['label']} ({pick['count']:,} >= {PC_MIN_WORKS:,})")
    bank("postcutoff_recut", {
        "window_from": PC_FROM, "rule": f"T10028 if >={PC_MIN_WORKS}, else "
        "T10181, else union (brief)", "rows": rows, "picked": pick,
        "note": "locked in the worklog BEFORE any pull",
    })


# --------------------------------------------------------------- phase: pull
def _pc_page_path(filt: str, page_idx: int) -> Path:
    h = hashlib.sha1(f"{filt}|{PC_SELECT}".encode()).hexdigest()[:10]
    return PC_CACHE / f"works_{h}" / f"page_{page_idx:05d}.json.gz"


def fetch_pc(filt: str, max_pages: int | None = None) -> int:
    """Cursor-page the locked post-cutoff slice (resumable, every page cached)."""
    cursor, page_idx, n_cached = "*", 0, 0
    while True:
        p = _pc_page_path(filt, page_idx)
        if not p.exists():
            break
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                cached = json.load(f)
        except (OSError, json.JSONDecodeError):
            break
        n_cached += len(cached.get("results", []))
        cursor = cached.get("meta", {}).get("next_cursor")
        page_idx += 1
        if cursor is None:
            log(f"pc slice fully cached: {page_idx} pages, {n_cached:,} works")
            return n_cached
    if page_idx:
        log(f"resuming pc pull from page {page_idx} ({n_cached:,} works)")
    session = _session()
    while cursor is not None:
        if max_pages is not None and page_idx >= max_pages:
            log(f"max_pages={max_pages} reached; stopping (partial)")
            break
        data = _get_json(session, f"{API_BASE}/works", {
            "filter": filt, "select": PC_SELECT,
            "per-page": str(PER_PAGE), "cursor": cursor,
        }, log=log)
        p = _pc_page_path(filt, page_idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        if page_idx == 0:
            log(f"pc meta.count = {data['meta']['count']:,}")
        n_cached += len(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        page_idx += 1
        if page_idx % 20 == 0:
            log(f"  {n_cached:,} works ({page_idx} pages)")
        time.sleep(0.05)
    log(f"pc slice acquired: {page_idx} pages, {n_cached:,} works")
    return n_cached


def iter_pc(filt: str):
    page_idx = 0
    while True:
        p = _pc_page_path(filt, page_idx)
        if not p.exists():
            return
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        yield from data.get("results", [])
        page_idx += 1


def _locked_pc_ids() -> str:
    picked = load_banked()["task_showdown_p1"]["postcutoff_recut"]["picked"]
    return picked["ids"]


def phase_pull(args) -> None:
    filt = pc_filter(_locked_pc_ids())
    n = fetch_pc(filt, max_pages=args.max_pages)
    bank("postcutoff_pull", {"filter": filt, "select": PC_SELECT,
                             "works_pulled": n})


# --------------------------------------------------------------- t3 cleaning (pc)
def assemble_pc(filt: str) -> dict:
    """t3 cleaning verbatim (KEEP_TYPES / >=200-char abstract / is_english /
    title dedupe / abstract-text dedupe, first wins) over the pc pull, which
    also carries referenced_works (kept per-row for the edge build)."""
    wids, titles, years, topics, texts, refs = [], [], [], [], [], []
    n_seen = n_short = n_noneng = n_dup = n_dup_txt = n_type = 0
    seen_titles: set[str] = set()
    seen_text_hash: set[str] = set()
    for w in iter_pc(filt):
        n_seen += 1
        wtype = str(w.get("type") or "unknown")
        if wtype not in KEEP_TYPES:
            n_type += 1
            continue
        txt = decode_abstract(w.get("abstract_inverted_index"))
        if len(txt) < MIN_ABSTRACT_CHARS:
            n_short += 1
            continue
        if not is_english(txt):
            n_noneng += 1
            continue
        t = (w.get("title") or "").strip()
        tkey = " ".join(t.lower().split())
        if tkey and tkey in seen_titles:
            n_dup += 1
            continue
        thash = hashlib.sha1(txt.encode()).hexdigest()
        if thash in seen_text_hash:
            n_dup_txt += 1
            continue
        seen_titles.add(tkey)
        seen_text_hash.add(thash)
        wids.append(short_id(w.get("id")))
        titles.append(t)
        d = str(w.get("publication_date") or "")
        years.append(int(d[:4]) if len(d) >= 4 and d[:4].isdigit() else 0)
        topics.append(short_id((w.get("primary_topic") or {}).get("id")) or "")
        texts.append(txt)
        refs.append([short_id(r) for r in (w.get("referenced_works") or [])])
    counts = {"seen": n_seen, "eligible": len(wids), "paratext_type": n_type,
              "short": n_short, "non_english": n_noneng, "dup_title": n_dup,
              "dup_text": n_dup_txt}
    log(f"pc assembled: {counts}")
    return {"wid": wids, "title": titles, "year": years, "topic": topics,
            "texts": texts, "refs": refs, "counts": counts}


# --------------------------------------------------------------- phase: corpus
def _standalone_embed_and_cache(tag: str, texts: list[str], wids: list[str],
                                titles: list[str], years: list[int],
                                src: np.ndarray, dst: np.ndarray,
                                counts: dict, extra_meta: dict) -> dict:
    """t3-config standalone embed (seed 42, LSA-64) + nnz>=5 trim; caches
    arrays, the FITTED vectorizer+svd (joblib), and the in-slice edge list
    REMAPPED to kept rows. src/dst index the pre-trim rows."""
    import joblib

    emb = fit_embed([], texts)
    nnz = np.asarray((emb["X"] > 0).sum(axis=1)).ravel()
    keep = np.where(nnz >= 5)[0]
    Z = emb["Z"][keep]
    K = emb["K"][keep].tocsr()
    old_to_new = np.full(len(texts), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(keep.size)
    m = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)
    src2, dst2 = old_to_new[src[m]], old_to_new[dst[m]]
    log(f"{tag}: embed vocab {len(emb['tfidf_vocab']):,}, EVR {emb['evr']:.3f}, "
        f"nnz>=5 keeps {keep.size:,}/{len(texts):,}; edges {src2.size:,} "
        f"(of {src.size:,} pre-trim)")
    np.savez_compressed(CACHE_DIR / f"p1_{tag}_arrays.npz",
                        Z=Z, year=np.asarray(years, dtype=np.int64)[keep],
                        src=src2.astype(np.int32), dst=dst2.astype(np.int32))
    sp.save_npz(CACHE_DIR / f"p1_{tag}_K.npz", K)
    with (CACHE_DIR / f"p1_{tag}_meta.pkl").open("wb") as f:
        pickle.dump({
            "wid": [wids[i] for i in keep], "title": [titles[i] for i in keep],
            "vocab": [str(v) for v in emb["vocab"]],
            "evr": float(emb["evr"]), "counts": counts, **extra_meta,
        }, f)
    joblib.dump({"vec": emb["vec"], "svd": emb["svd"]},
                CACHE_DIR / f"p1_{tag}_vectorizer.joblib")
    return {"n_texts": len(texts), "n_used": int(keep.size),
            "evr": round(float(emb["evr"]), 3),
            "vocab": len(emb["tfidf_vocab"]), "edges": int(src2.size)}


def phase_corpus_causal(args) -> None:
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)
    bg = assemble_bg(filt, log=log)
    m = np.asarray([t == CAUSAL_TOPIC_ID for t in bg["topic"]], dtype=bool)
    idx = np.where(m)[0]
    texts = [bg["texts"][i] for i in idx]
    wids = [str(bg["wid"][i]) for i in idx]
    titles = [str(bg["title"][i]) for i in idx]
    years = [int(bg["year"][i]) for i in idx]
    log(f"causal sub-slice: {idx.size:,} cleaned works")
    # in-sub-slice edges from the Phase-0 refs pull (whole-slice select)
    wid_to_row = {w: i for i, w in enumerate(wids)}
    src, dst = [], []
    for w in iter_refs(filt):
        row = wid_to_row.get(short_id(w.get("id")) or "")
        if row is None:
            continue
        for r in (w.get("referenced_works") or []):
            rr = wid_to_row.get(short_id(r) or "")
            if rr is not None and rr != row:
                src.append(row)
                dst.append(rr)
    log(f"causal in-sub-slice citation edges: {len(src):,}")
    out = _standalone_embed_and_cache(
        "causal", texts, wids, titles, years,
        np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64),
        {"sub_slice_of": "t3 cleaned slice", "raw_in_slice": int(idx.size)},
        {"topic_id": CAUSAL_TOPIC_ID})
    bank("corpus_causal", {"topic_id": CAUSAL_TOPIC_ID,
                           "cleaned_in_slice": int(idx.size), **out})


def phase_corpus_pc(args) -> None:
    filt = pc_filter(_locked_pc_ids())
    a = assemble_pc(filt)
    wid_to_row = {w: i for i, w in enumerate(a["wid"])}
    src, dst = [], []
    n_ref_entries = 0
    for row, rr_list in enumerate(a["refs"]):
        n_ref_entries += len(rr_list)
        for r in rr_list:
            rr = wid_to_row.get(r or "")
            if rr is not None and rr != row:
                src.append(row)
                dst.append(rr)
    log(f"pc in-slice citation edges: {len(src):,} "
        f"(of {n_ref_entries:,} reference entries)")
    out = _standalone_embed_and_cache(
        "postcutoff", a["texts"], a["wid"], a["title"], a["year"],
        np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64),
        a["counts"], {"filter": filt})
    bank("corpus_postcutoff", {"filter": filt, "cleaning": a["counts"],
                               "ref_entries": n_ref_entries, **out})


def phase_corpus_ds(args) -> None:
    """Re-run the Phase-0 joint embed to capture the FITTED vectorizer+svd
    (agent/hybrid DS section intents must map into the SAME frozen space);
    asserts the recompute reproduces the cached Phase-0 arrays."""
    import joblib

    banked = load_banked()["task_t3"]
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)
    fg_units = load_book_fg()
    fg_texts = [u.text for u in fg_units]
    bg = assemble_bg(filt, log=log)
    assert bg["counts"]["eligible"] == banked["bg"]["counts"]["eligible"]
    emb = fit_embed(fg_texts, bg["texts"])
    nnz = np.asarray((emb["X"] > 0).sum(axis=1)).ravel()
    n_fg = len(fg_texts)
    keep_bg = np.where(nnz[n_fg:] >= 5)[0]
    Zb = emb["Z"][n_fg:][keep_bg]
    Zb_cached = _load_embed()[1]
    assert Zb.shape == Zb_cached.shape, (Zb.shape, Zb_cached.shape)
    md = float(np.abs(Zb - Zb_cached).max())
    log(f"ds joint embed reproduced: bg used {Zb.shape[0]:,}, "
        f"max |Z - Z_cached| = {md:.2e}")
    assert md < 1e-5, md
    joblib.dump({"vec": emb["vec"], "svd": emb["svd"]},
                CACHE_DIR / "p1_datascience_vectorizer.joblib")
    bank("corpus_ds", {"n_bg_used": int(Zb.shape[0]),
                       "max_abs_diff_vs_phase0": md,
                       "evr": round(float(emb["evr"]), 3),
                       "note": "fitted tfidf+svd cached for intent mapping"})


# --------------------------------------------------------------- phase: toc
def build_toc(Z: np.ndarray, K: sp.csr_matrix, vocab: list[str],
              titles: list[str], year: np.ndarray,
              src: np.ndarray, dst: np.ndarray, chapter_k_range) -> dict:
    """Frozen Phase-0 generator on a standalone corpus: chapter sweep
    (pre-committed rule), exact DP ordering, section sweeps, register-filtered
    labels, exemplars. Mirrors curriculum_toc.phase_build."""
    from artifacts_scripts.gap_detection.curriculum_toc import SECTION_K_RANGE

    log(f"chapter sweep: {Z.shape[0]:,} x {Z.shape[1]}, k in {list(chapter_k_range)}")
    rows, labelings = _kmeans_sweep(Z, chapter_k_range)
    k, why = select_k(rows)
    labels = labelings[k].astype(np.int64)
    log(f"RULE PICK: k={k} ({why}), sil {rows[k]['silhouette']}")

    F = citation_flow(labels[src], labels[dst], k)
    med_year = np.array([float(np.median(year[labels == c])) for c in range(k)])
    chap_order, _ = order_min_forward_flow(F, med_year)
    opt = forward_violations(F, chap_order)
    base = forward_violations(F, list(range(k)))
    log(f"chapter order {chap_order}: forward share {opt['share']:.3f} "
        f"(arbitrary-id {base['share']:.3f})")

    chapters = []
    for c in chap_order:
        m = np.where(labels == c)[0]
        srows, slabelings = _kmeans_sweep(Z[m], SECTION_K_RANGE)
        ks, swhy = select_k(srows)
        sub = slabelings[ks].astype(np.int64)
        in_c = (labels[src] == c) & (labels[dst] == c)
        pos_in_m = np.full(Z.shape[0], -1, dtype=np.int64)
        pos_in_m[m] = np.arange(m.size)
        F_sub = citation_flow(sub[pos_in_m[src[in_c]]], sub[pos_in_m[dst[in_c]]], ks)
        med_sub = np.array([float(np.median(year[m[sub == s]])) for s in range(ks)])
        sec_order, _ = order_min_forward_flow(F_sub, med_sub)
        c_terms, c_register = cluster_terms(K, m, vocab, n_top=N_CHAPTER_TERMS)
        sections = []
        for s in sec_order:
            sm = m[sub == s]
            s_terms, _ = cluster_terms(K, sm, vocab, n_top=N_SECTION_TERMS)
            cen = Z[sm].mean(axis=0)
            cen /= max(float(np.linalg.norm(cen)), 1e-12)
            d = np.linalg.norm(Z[sm] - cen[None, :], axis=1)
            ex, seen_norm = [], set()
            for i in sm[np.argsort(d)]:
                norm = "".join(c0 for c0 in titles[i].lower() if c0.isalnum())
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                ex.append(int(i))
                if len(ex) >= N_EXEMPLARS:
                    break
            sections.append({
                "sub_id": int(s), "n": int(sm.size), "terms": s_terms,
                "centroid": cen.astype(np.float32),
                "exemplar_titles": [titles[i] for i in ex],
                "exemplar_years": [int(year[i]) for i in ex],
                "median_year": float(np.median(year[sm])),
            })
        chapters.append({
            "cluster_id": int(c), "n": int(m.size),
            "share": round(m.size / Z.shape[0], 4),
            "terms": c_terms, "register_excluded": c_register,
            "median_year": float(np.median(year[m])),
            "section_k": ks, "section_rule": swhy,
            "section_forward_share": round(
                forward_violations(F_sub, sec_order)["share"], 4),
            "sections": sections,
        })
        log(f"chapter cluster {c}: n={m.size:,}, sections k={ks}, "
            f"terms {c_terms[:4]}")
    return {"k": k, "rule_why": why, "sweep": rows,
            "chap_order": [int(c) for c in chap_order],
            "forward_flow": opt, "arbitrary_order_share": base["share"],
            "F": F, "chapters": chapters,
            "n_edges": int(src.size), "labels": labels}


def write_p1_toc_md(toc: dict, topic_title: str, path: Path) -> None:
    """Judge-facing ToC: the toc.md shape with NEUTRAL H1, no provenance."""
    k = toc["k"]
    F = toc["F"]
    order = toc["chap_order"]
    pos = {c: i for i, c in enumerate(order)}
    L = [f"# Table of Contents: {topic_title}", ""]
    for i, ch in enumerate(toc["chapters"], 1):
        c = ch["cluster_id"]
        L.append(f"## Chapter {i}: {_title_case(ch['terms'], 4)}")
        L.append(f"*Intent: cover the {', '.join(ch['terms'][:6])} literature — "
                 f"{ch['n']:,} works ({ch['share']:.1%} of the field), "
                 f"median year {ch['median_year']:.0f}.*")
        for j, s in enumerate(ch["sections"], 1):
            ex = "; ".join(f"“{t}” ({y})" for t, y in
                           zip(s["exemplar_titles"], s["exemplar_years"]))
            L.append(f"- **{i}.{j} {_title_case(s['terms'], 3)}**: "
                     f"the {', '.join(s['terms'][:5])} literature "
                     f"(n={s['n']:,}; exemplars: {ex}).")
        out = np.array([F[c, b] for b in range(k)], dtype=np.float64)
        out[c] = 0.0
        tot = out.sum()
        deps = []
        if tot > 0:
            for b_pos, b in enumerate(order[:pos[c]]):
                share = out[b] / tot
                if share >= DEP_SHARE:
                    deps.append((b_pos + 1, share))
        deps.sort(key=lambda d2: -d2[1])
        if deps:
            note = ", ".join(f"Chapter {p} ({s:.0%} of out-citations)"
                             for p, s in deps[:3])
            L.append(f"**Dependencies:** {note}.")
        else:
            L.append("**Dependencies:** none pronounced "
                     f"(no earlier chapter receives ≥{DEP_SHARE:.0%} of "
                     "this chapter's cross-chapter citations).")
        L.append("")
    path.write_text("\n".join(L).rstrip() + "\n")


def write_skeleton_md(toc: dict, topic_title: str, path: Path) -> None:
    """Hybrid-arm input: chapter/section term-labels + 2 exemplar titles each
    + the fixed order. Nothing else (no counts, no intents)."""
    L = [f"# Algorithmic skeleton: {topic_title}",
         "",
         "Fixed chapter order and section order. Term-labels are raw "
         "distinctive-term tuples; exemplars are two representative work "
         "titles per section.", ""]
    for i, ch in enumerate(toc["chapters"], 1):
        L.append(f"## Chapter {i} [term-label: {_title_case(ch['terms'], 4)}]")
        for j, s in enumerate(ch["sections"], 1):
            L.append(f"- {i}.{j} [term-label: {_title_case(s['terms'], 3)}]")
            for t, y in zip(s["exemplar_titles"], s["exemplar_years"]):
                L.append(f"  - exemplar: “{t}” ({y})")
        L.append("")
    path.write_text("\n".join(L).rstrip() + "\n")


def _load_standalone(tag: str):
    z = np.load(CACHE_DIR / f"p1_{tag}_arrays.npz")
    K = sp.load_npz(CACHE_DIR / f"p1_{tag}_K.npz").tocsr()
    with (CACHE_DIR / f"p1_{tag}_meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    return z["Z"], K, z["year"], z["src"], z["dst"], meta


def phase_toc(args) -> None:
    SHOW_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tag in ("causal", "postcutoff"):
        Z, K, year, src, dst, meta = _load_standalone(tag)
        k_range = SMALL_K_RANGE if Z.shape[0] < 15_000 else range(8, 15)
        toc = build_toc(Z, K, meta["vocab"], meta["title"], year,
                        src.astype(np.int64), dst.astype(np.int64), k_range)
        with (CACHE_DIR / f"p1_{tag}_toc.pkl").open("wb") as f:
            pickle.dump(toc, f)
        write_p1_toc_md(toc, TOPICS[tag]["title"],
                        SHOW_DIR / f"p1_{tag}_algo.md")
        write_skeleton_md(toc, TOPICS[tag]["title"],
                          SHOW_DIR / f"p1_skeleton_{tag}.md")
        summary[tag] = {
            "n_docs": int(Z.shape[0]), "k_range": list(k_range),
            "chosen_k": toc["k"], "rule": toc["rule_why"],
            "sweep": toc["sweep"], "chap_order": toc["chap_order"],
            "forward_flow": toc["forward_flow"],
            "arbitrary_order_share": toc["arbitrary_order_share"],
            "n_edges": toc["n_edges"],
            "sections_per_chapter": [len(c["sections"]) for c in toc["chapters"]],
            "chapter_titles": [_title_case(c["terms"], 4) for c in toc["chapters"]],
        }
        log(f"{tag}: wrote p1_{tag}_algo.md + skeleton "
            f"({toc['k']} chapters / {sum(len(c['sections']) for c in toc['chapters'])} sections)")

    # ds: re-emit the FROZEN Phase-0 ToC in the judge-facing form + skeleton
    with TOC_PKL.open("rb") as f:
        ds_toc = pickle.load(f)
    write_p1_toc_md(ds_toc, TOPICS["datascience"]["title"],
                    SHOW_DIR / "p1_datascience_algo.md")
    write_skeleton_md(ds_toc, TOPICS["datascience"]["title"],
                      SHOW_DIR / "p1_skeleton_datascience.md")
    summary["datascience"] = {
        "note": "Phase-0 frozen ToC re-emitted (neutral H1, no provenance)",
        "chosen_k": ds_toc["k"],
        "sections_per_chapter": [len(c["sections"]) for c in ds_toc["chapters"]],
    }
    bank("algo_tocs", summary)


# --------------------------------------------------------------- toc.md parser
import re  # noqa: E402

_CH_RE = re.compile(r"^#{2,3}\s*chapter\s+(\d+)\s*[:.\-–—]?\s*(.*)$", re.I)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_SECNUM_RE = re.compile(r"^\**\s*(\d+)\.(\d+)\.?\**\s*(.*)$")
_INTENT_RE = re.compile(r"^intent\b\s*[:\-–—]?\s*", re.I)
_DEPS_RE = re.compile(r"^dependencies\b\s*[:\-–—]?\s*", re.I)


def _clean(s: str) -> str:
    s = s.strip().strip("*_").strip()
    s = re.sub(r"\*\*|\*|__", "", s)
    return re.sub(r"\s+", " ", s).strip(" :.–—-").strip()


def _split_title_intent(rest: str) -> tuple[str, str]:
    """'Title**: intent' / 'Title: intent' / 'Title — intent' -> (title, intent)."""
    rest = rest.strip()
    for sep in (":", "—", "–", " - "):
        i = rest.find(sep)
        if i > 0:
            return _clean(rest[:i]), _clean(rest[i + len(sep):])
    return _clean(rest), ""


def parse_toc_md(text: str) -> dict:
    """Tolerant parser for the toc.md shape (agent/hybrid candidates).

    Returns {"h1", "chapters": [{"num","title","intent","deps",
    "sections":[{"label","title","intent"}], "fallback_section": bool}],
    "n_fallback_chapters"}. A chapter that yields no parseable sections gets
    ONE pseudo-section synthesized from its title+intent (flagged) so the rep
    set never silently loses a chapter."""
    h1 = None
    chapters: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or set(line) <= {"-", "="}:
            continue
        if h1 is None and line.startswith("# ") and not line.startswith("## "):
            h1 = _clean(line[2:])
            continue
        m = _CH_RE.match(line)
        if m:
            cur = {"num": int(m.group(1)), "title": _clean(m.group(2)),
                   "intent": "", "deps": "", "sections": [],
                   "fallback_section": False}
            chapters.append(cur)
            continue
        if cur is None:
            continue
        bare = _clean(line)
        if _DEPS_RE.match(bare):
            cur["deps"] = _clean(_DEPS_RE.sub("", bare))
            continue
        mb = _BULLET_RE.match(line)
        if mb:
            content = mb.group(1).strip()
            mn = _SECNUM_RE.match(content)
            if mn:
                title, intent = _split_title_intent(mn.group(3))
                cur["sections"].append(
                    {"label": f"{mn.group(1)}.{mn.group(2)}",
                     "title": title, "intent": intent})
            else:
                title, intent = _split_title_intent(content)
                if title:
                    cur["sections"].append({"label": "", "title": title,
                                            "intent": intent})
            continue
        if _INTENT_RE.match(bare) and not cur["sections"]:
            cur["intent"] = _clean(_INTENT_RE.sub("", bare))
            continue
        mn = _SECNUM_RE.match(line)
        if mn and mn.group(3):
            title, intent = _split_title_intent(mn.group(3))
            cur["sections"].append({"label": f"{mn.group(1)}.{mn.group(2)}",
                                    "title": title, "intent": intent})
    n_fb = 0
    for ch in chapters:
        if not ch["sections"]:
            ch["sections"].append({"label": "", "title": ch["title"],
                                   "intent": ch["intent"]})
            ch["fallback_section"] = True
            n_fb += 1
    return {"h1": h1, "chapters": chapters, "n_fallback_chapters": n_fb}


def section_rep_texts(parsed: dict) -> tuple[list[str], list[int]]:
    """Per-section embed text 'TITLE. INTENT' + the section's chapter index
    (position in published order, 0-based) — the pre-committed mapping."""
    texts, chap_idx = [], []
    for i, ch in enumerate(parsed["chapters"]):
        for s in ch["sections"]:
            t = s["title"].strip()
            it = s["intent"].strip()
            texts.append(f"{t}. {it}" if it else t)
            chap_idx.append(i)
    return texts, chap_idx


# --------------------------------------------------------------- blinding plan
ARMS = ("algo", "agent", "hybrid")
PAIRS = (("algo", "agent"), ("algo", "hybrid"), ("agent", "hybrid"))
TOPIC_ORDER = ("datascience", "causal", "postcutoff")
JUDGE_RUBRIC = (
    "Judge the two candidate tables of contents on exactly four criteria, "
    "weighted equally:\n"
    "1. Coverage for the stated audience: does the ToC span what this "
    "audience needs for this topic?\n"
    "2. Progression: prerequisites precede their use — no chapter depends on "
    "material that appears later.\n"
    "3. Pedagogical arc: the sequence forms a coherent learning trajectory "
    "from foundations to frontier.\n"
    "4. Feasibility at depth: could a chapter drafter write substantive, "
    "non-filler chapters from these intents?\n"
)


def ab_assignments(seed: int = SEED) -> dict[tuple[str, tuple[str, str]], dict]:
    """Pre-committed blinding: rng=default_rng(seed); topics in TOPIC_ORDER x
    pairs in PAIRS order; u=integers(2): u=0 -> the pair's first-listed arm is
    A in order-1, u=1 -> the second; order-2 = swapped."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in TOPIC_ORDER:
        for pair in PAIRS:
            u = int(rng.integers(2))
            out[(t, pair)] = {
                "order1": {"A": pair[u], "B": pair[1 - u]},
                "order2": {"A": pair[1 - u], "B": pair[u]},
            }
    return out


def majority_of(winners: list[str | None]) -> str:
    """Per-pair score: majority of the 4 judgments; ties (incl. unparseable
    no-votes making a majority impossible) = 'tie' (pre-committed)."""
    counts: dict[str, int] = {}
    for w in winners:
        if w is not None:
            counts[w] = counts.get(w, 0) + 1
    if not counts:
        return "tie"
    best = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(best) > 1 and best[0][1] == best[1][1]:
        return "tie"
    return best[0][0] if best[0][1] > len(winners) / 2 else "tie"


def extract_winner(verdict_text: str) -> str | None:
    m = re.search(r"winner\s*[:\-]?\s*\(?\**([AB])\**\)?", verdict_text, re.I)
    return m.group(1).upper() if m else None


def assign_by_nearest(Z: np.ndarray, reps: np.ndarray,
                      sec_chap: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Pre-committed doc->chapter mapping for agent/hybrid candidates: every
    doc goes to the chapter of its nearest (euclidean) kept section rep."""
    R = np.asarray(reps, dtype=np.float32)
    sec_chap = np.asarray(sec_chap, dtype=np.int64)
    rsq = np.einsum("ij,ij->i", R, R)
    j = np.empty(Z.shape[0], dtype=np.int64)
    for s0 in range(0, Z.shape[0], chunk):
        q = Z[s0:s0 + chunk].astype(np.float32)
        d2 = (np.einsum("ij,ij->i", q, q)[:, None]
              - 2.0 * (q @ R.T) + rsq[None, :])
        j[s0:s0 + chunk] = np.argmin(d2, axis=1)
    return sec_chap[j]


# --------------------------------------------------------------- run manifests
RUNS_TSV = SHOW_DIR / "p1_runs.tsv"
PICKS_TSV = SHOW_DIR / "p1_picks.tsv"
JUDGES_TSV = SHOW_DIR / "p1_judges.tsv"
BRIEFS_DIR = Path("~/src/learn-and-teach/outputs/showdown_p1/briefs").expanduser()
ANGLE_ORDER = ("problems", "methods", "applications")   # candidate 1/2/3 (fixed)


def read_manifest(path: Path) -> dict[str, Path]:
    out = {}
    if not path.exists():
        return out
    for row in path.read_text().splitlines():
        parts = row.split("\t")
        if len(parts) >= 2 and parts[1]:
            out[parts[0]] = Path(parts[1])
    return out


# --------------------------------------------------------------- phase: pickmake
PICK_BRIEF = """# Task: pick the best of your three table-of-contents drafts — {title}

Three drafts of a table of contents for the same planned textbook are staged in
this directory as `./{c1}`, `./{c2}`, `./{c3}`.

- Topic: {statement}.
- Audience: {audience}.

Read all three in full. Pick the ONE that best serves this audience as the
table of contents of the textbook: coverage, ordering (prerequisites first),
pedagogical arc, and how well its section intents would support actually
drafting substantive chapters.

Then write `./pick.md`: first line exactly `PICK: 1` or `PICK: 2` or `PICK: 3`;
then exactly 3 rationale lines (one sentence each). Do not browse the web.
Work autonomously; do not pause for input.
"""


def phase_pickmake(args) -> None:
    """Stage each topic's 3 mini ToCs as pick candidates + write pick briefs
    and the pick jobs file (agent arm's own internal, non-blind tournament)."""
    from artifacts_scripts.gap_detection.p1_make_briefs import TOPICS as BT
    runs = read_manifest(RUNS_TSV)
    jobs = []
    for t in TOPIC_ORDER:
        names = []
        for i, angle in enumerate(ANGLE_ORDER, 1):
            rd = runs.get(f"mini_{t}_{angle}")
            assert rd is not None, f"mini_{t}_{angle} missing from {RUNS_TSV}"
            src = rd / "toc.md"
            assert src.is_file() and src.stat().st_size > 0, f"no toc.md in {rd}"
            name = f"pick_{t}_candidate_{i}.md"
            (BRIEFS_DIR / name).write_text(src.read_text(encoding="utf-8"))
            names.append(name)
        brief = BRIEFS_DIR / f"pick_{t}.md"
        brief.write_text(PICK_BRIEF.format(
            title=BT[t]["title"], statement=BT[t]["statement"],
            audience=BT[t]["audience"], c1=names[0], c2=names[1], c3=names[2]))
        stage = ",".join(str(BRIEFS_DIR / n) for n in names)
        # single-command predicate (hydra override grammar: no &&/quotes);
        # tally is the tolerant parser of record
        until = "grep -qis pick pick.md"
        jobs.append(f"pick_{t}\t{brief}\t{until}\t{stage}")
    (BRIEFS_DIR / "pick_jobs.tsv").write_text("\n".join(jobs) + "\n")
    log(f"wrote 3 pick briefs + jobs file under {BRIEFS_DIR}")


# --------------------------------------------------------------- phase: collect
def _normalize_candidate(text: str, title: str) -> str:
    """H1 normalized to the locked neutral form; code fences stripped."""
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("# ") and not lines[0].startswith("## "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    return f"# Table of Contents: {title}\n\n{body}\n"


def phase_collect(args) -> None:
    """Agent arm: the internal pick's winner -> p1_<topic>_agent.md; hybrid
    run -> p1_<topic>_hybrid.md. H1s normalized; shape-linted via the parser."""
    from artifacts_scripts.gap_detection.p1_make_briefs import TOPICS as BT
    runs = read_manifest(RUNS_TSV)
    picks = read_manifest(PICKS_TSV)
    summary = {}
    for t in TOPIC_ORDER:
        pd = picks.get(f"pick_{t}")
        assert pd is not None, f"pick_{t} missing from {PICKS_TSV}"
        ptxt = (pd / "pick.md").read_text(encoding="utf-8")
        m = re.search(r"pick\s*[:\-]?\s*\(?([123])\)?", ptxt, re.I)
        assert m, f"unparseable pick.md in {pd}"
        picked = int(m.group(1))
        angle = ANGLE_ORDER[picked - 1]
        rd = runs[f"mini_{t}_{angle}"]
        raw = (rd / "toc.md").read_text(encoding="utf-8")
        out = _normalize_candidate(raw, BT[t]["title"])
        (SHOW_DIR / f"p1_{t}_agent.md").write_text(out)
        pa = parse_toc_md(out)
        hd = runs.get(f"hybrid_{t}")
        assert hd is not None, f"hybrid_{t} missing from {RUNS_TSV}"
        rawh = (hd / "toc.md").read_text(encoding="utf-8")
        outh = _normalize_candidate(rawh, BT[t]["title"])
        (SHOW_DIR / f"p1_{t}_hybrid.md").write_text(outh)
        ph = parse_toc_md(outh)
        summary[t] = {
            "agent_pick": picked, "agent_angle": angle,
            "agent_pick_run": str(pd), "agent_run": str(rd),
            "hybrid_run": str(hd),
            "agent_shape": {"chapters": len(pa["chapters"]),
                            "sections": sum(len(c["sections"]) for c in pa["chapters"]),
                            "fallback_chapters": pa["n_fallback_chapters"]},
            "hybrid_shape": {"chapters": len(ph["chapters"]),
                             "sections": sum(len(c["sections"]) for c in ph["chapters"]),
                             "fallback_chapters": ph["n_fallback_chapters"]},
        }
        log(f"{t}: agent = draft {picked} ({angle}) "
            f"{summary[t]['agent_shape']}; hybrid {summary[t]['hybrid_shape']}")
    bank("collect", summary)


# --------------------------------------------------------------- phase: judgemake
JUDGE_BRIEF = """# Task: judge two candidate textbook tables of contents

Two candidate tables of contents (A and B) for the same planned textbook follow
at the end of this file.

- Topic: {statement}.
- Audience: {audience}.

{rubric}
Read both candidates fully, weigh the four criteria equally, then declare a
winner. Write `./verdict.md`: first line exactly `WINNER: A` or `WINNER: B`;
then exactly 3 rationale lines (one sentence each). Do not browse the web;
judge only from the two candidates' text below. Work autonomously; do not
pause for input.

---

## Candidate A

{cand_a}

---

## Candidate B

{cand_b}
"""


def phase_judgemake(args) -> None:
    """36 blind judge briefs (3 topics x 3 pairs x 2 orders x 2 samples) with
    the pre-committed seeded A/B assignment; jobs file for the serial runner.
    The assignment table is banked BEFORE any judgment runs."""
    from artifacts_scripts.gap_detection.p1_make_briefs import TOPICS as BT
    asn = ab_assignments()
    cand = {(t, a): (SHOW_DIR / f"p1_{t}_{a}.md").read_text(encoding="utf-8").strip()
            for t in TOPIC_ORDER for a in ARMS}
    jobs, table = [], {}
    # single-command predicate (hydra override grammar: no &&/quotes)
    until = "grep -qis winner verdict.md"
    for t in TOPIC_ORDER:
        for pair in PAIRS:
            key0 = f"{t}_{pair[0]}-v-{pair[1]}"
            for o in (1, 2):
                amap = asn[(t, pair)][f"order{o}"]
                for s2 in (1, 2):
                    key = f"judge_{key0}_o{o}_s{s2}"
                    brief = BRIEFS_DIR / f"{key}.md"
                    brief.write_text(JUDGE_BRIEF.format(
                        statement=BT[t]["statement"], audience=BT[t]["audience"],
                        rubric=JUDGE_RUBRIC,
                        cand_a=cand[(t, amap["A"])], cand_b=cand[(t, amap["B"])]))
                    jobs.append(f"{key}\t{brief}\t{until}\t-")
                    table[key] = {"topic": t, "pair": list(pair), "order": o,
                                  "sample": s2, "A": amap["A"], "B": amap["B"]}
    (BRIEFS_DIR / "judge_jobs.tsv").write_text("\n".join(jobs) + "\n")
    bank("judge_assignments", {
        "seed": SEED, "n_judgments": len(jobs),
        "sampling_note": "pi exposes no sampler-seed control; s1/s2 are two "
                         "independent samples at provider-default sampling",
        "table": table,
    })
    log(f"wrote {len(jobs)} judge briefs + jobs file under {BRIEFS_DIR}")


# --------------------------------------------------------------- phase: tally
def phase_tally(args) -> None:
    """Verdicts -> per-judgment arm winners -> per-pair majorities (ties kept)."""
    asn = load_banked()["task_showdown_p1"]["judge_assignments"]["table"]
    judges = read_manifest(JUDGES_TSV)
    fourway: dict[str, dict] = {}
    matrix: dict[str, dict[str, str]] = {t: {} for t in TOPIC_ORDER}
    rationales: dict[str, str] = {}
    for t in TOPIC_ORDER:
        for pair in PAIRS:
            key0 = f"{t}_{pair[0]}-v-{pair[1]}"
            winners, cells = [], {}
            for o in (1, 2):
                for s2 in (1, 2):
                    key = f"judge_{key0}_o{o}_s{s2}"
                    rd = judges.get(key)
                    arm = None
                    if rd is not None and (rd / "verdict.md").is_file():
                        vt = (rd / "verdict.md").read_text(encoding="utf-8")
                        letter = extract_winner(vt)
                        if letter:
                            arm = asn[key][letter]
                            lines = [ln.strip() for ln in vt.splitlines()
                                     if ln.strip() and not ln.lower().startswith(
                                         ("winner", "# ", "## "))]
                            rationales[key] = " / ".join(lines[:3])[:400]
                    winners.append(arm)
                    cells[f"o{o}_s{s2}"] = arm or "INVALID"
            maj = majority_of(winners)
            fourway[key0] = {"cells": cells, "majority": maj,
                             "votes": {a: winners.count(a) for a in pair}}
            matrix[t][f"{pair[0]}-v-{pair[1]}"] = maj
            log(f"{key0}: {cells} -> {maj}")
    bank("judging", {"fourway": fourway, "win_matrix": matrix,
                     "rationale_snippets": rationales})


# --------------------------------------------------------------- phase: metrics
def _ds_arrays():
    z = np.load(EMBED_NPZ_DS)
    return z["Zf"], z["Zb"], z["year"]


EMBED_NPZ_DS = CACHE_DIR / "showdown_embed.npz"
REGIONS_NPZ_DS = CACHE_DIR / "showdown_regions.npz"
CHAPTERS_NPZ_DS = CACHE_DIR / "showdown_chapters.npz"
EDGES_NPZ_DS = CACHE_DIR / "showdown_edges.npz"
DS_RADIUS = 1.0181          # banked t3 q25 scope threshold
FIXED_RADIUS = 1.0181       # pre-committed fixed yardstick for causal/postcutoff


def _candidate_reps(topic: str, arm: str) -> dict:
    """Reps + per-doc chapter assignment + published chapter order for one
    candidate. Pre-committed forms (worklog): algo = section centroids +
    KMeans labels + built order; agent/hybrid = frozen-space images of
    'TITLE. INTENT' (zero-image sections dropped), docs -> nearest kept rep's
    chapter, published order."""
    import joblib

    if topic == "datascience":
        Zf, Zb, year = _ds_arrays()
        Z = Zb
        e = np.load(EDGES_NPZ_DS)
        src, dst = e["src"], e["dst"]
    else:
        tag = topic
        z = np.load(CACHE_DIR / f"p1_{tag}_arrays.npz")
        Z, year, src, dst = z["Z"], z["year"], z["src"], z["dst"]

    if arm == "algo":
        if topic == "datascience":
            with TOC_PKL.open("rb") as f:
                toc = pickle.load(f)
            labels = np.load(CHAPTERS_NPZ_DS)["labels"].astype(np.int64)
        else:
            with (CACHE_DIR / f"p1_{topic}_toc.pkl").open("rb") as f:
                toc = pickle.load(f)
            labels = toc["labels"].astype(np.int64)
        reps = np.vstack([s["centroid"] for ch in toc["chapters"]
                          for s in ch["sections"]]).astype(np.float32)
        sec_chap = np.array([i for i, ch in enumerate(toc["chapters"])
                             for _ in ch["sections"]], dtype=np.int64)
        pos_of_cluster = {c: i for i, c in enumerate(toc["chap_order"])}
        chap = np.array([pos_of_cluster[int(c)] for c in labels], dtype=np.int64)
        k = len(toc["chap_order"])
        meta = {"n_sections": int(reps.shape[0]), "n_dropped_zero": 0,
                "n_fallback_chapters": 0}
    else:
        path = SHOW_DIR / f"p1_{topic}_{arm}.md"
        parsed = parse_toc_md(path.read_text(encoding="utf-8"))
        texts, chap_idx = section_rep_texts(parsed)
        vs = joblib.load(CACHE_DIR / f"p1_{topic}_vectorizer.joblib")
        Zs_new, ok = transform_texts(vs["vec"], vs["svd"], texts)
        reps = Zs_new[ok].astype(np.float32)
        sec_chap = np.asarray(chap_idx, dtype=np.int64)[ok]
        k = len(parsed["chapters"])
        chap = assign_by_nearest(Z, reps, sec_chap)
        meta = {"n_sections": len(texts),
                "n_dropped_zero": int((~ok).sum()),
                "n_fallback_chapters": parsed["n_fallback_chapters"]}
    return {"Z": Z, "year": year, "src": src, "dst": dst, "reps": reps,
            "chap": chap, "k": k, "sec_chap": sec_chap, **meta}


def phase_metrics(args) -> None:
    from artifacts_scripts.gap_detection.book_audit import full_statistic
    from artifacts_scripts.gap_detection.detectors import knn_radii
    from artifacts_scripts.gap_detection.run_t3_book_audit import (
        K_BG, K_FG, MIN_COMP_SIZE, TOP_FRAC,
    )
    from artifacts_scripts.gap_detection.toc_metrics import (
        coverage_at_radius, gap_bar, nearest_rep_dist, region_closure_fullstat,
    )

    out: dict[str, dict] = {}
    # DS closure scaffolding (recomputed from the Phase-0 caches)
    Zf, Zb, _ = _ds_arrays()
    with open(CACHE_DIR / "showdown_meta.pkl", "rb") as f:
        ds_thr = float(pickle.load(f)["thr"])
    r_F = knn_radii(Zb, Zf, K_FG)
    scoped_global = np.where(r_F <= ds_thr)[0]
    Zs = Zb[scoped_global]
    pos = {int(g): i for i, g in enumerate(scoped_global)}
    rz = np.load(REGIONS_NPZ_DS)
    regions_local = [np.asarray([pos[int(g)] for g in rz[f"region_{i:02d}"]],
                                dtype=np.int64) for i in range(12)]
    full_ds2 = full_statistic(Zf, Zs, k_fg=K_FG, k_bg=K_BG)["full"]
    bar = gap_bar(full_ds2, top_frac=TOP_FRAC, min_size=MIN_COMP_SIZE)
    log(f"ds closure scaffold: thr {ds_thr}, scoped {Zs.shape[0]:,}, bar {bar:.4f}")

    for t in TOPIC_ORDER:
        radius = DS_RADIUS if t == "datascience" else FIXED_RADIUS
        out[t] = {"radius": radius}
        for arm in ARMS:
            c = _candidate_reps(t, arm)
            d = nearest_rep_dist(c["Z"], c["reps"])
            cov = float((d <= radius).mean())
            med = float(np.median(d))
            F = citation_flow(c["chap"][c["src"]], c["chap"][c["dst"]], c["k"])
            v = forward_violations(F, list(range(c["k"])))
            row = {
                "n_sections": c["n_sections"], "n_chapters": c["k"],
                "n_dropped_zero": c["n_dropped_zero"],
                "n_fallback_chapters": c["n_fallback_chapters"],
                "coverage": round(cov, 4), "median_nearest_dist": round(med, 4),
                "prereq": {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                           for kk, vv in v.items()},
            }
            if c["k"] <= 16:
                med_year = np.array(
                    [float(np.median(c["year"][c["chap"] == i]))
                     if (c["chap"] == i).any() else 0.0 for i in range(c["k"])])
                opt_order, opt_flow = order_min_forward_flow(F, med_year)
                row["prereq"]["best_possible_share"] = round(
                    opt_flow / v["cross_edges"], 4) if v["cross_edges"] else 0.0
            if t == "datascience":
                full_T = full_statistic(c["reps"], Zs, k_fg=K_FG, k_bg=K_BG)["full"]
                clo = region_closure_fullstat(full_T, regions_local, bar)
                row["gap_closure"] = {
                    "closed": int(sum(clo)), "of": 12, "per_region": clo,
                    "mean_full": [round(float(np.mean(full_T[m])), 3)
                                  for m in regions_local]}
            out[t][arm] = row
            log(f"{t}/{arm}: cov {cov:.3f} med {med:.3f} "
                f"viol {row['prereq']['share']} k {c['k']} "
                f"secs {c['n_sections']} (zero-dropped {c['n_dropped_zero']})"
                + (f" closure {row['gap_closure']['closed']}/12"
                   if t == "datascience" else ""))
    bank("metrics", out)


# --------------------------------------------------------------- phase: results
def _metric_leader(metrics_t: dict, pair: tuple[str, str], key: str) -> str:
    a, b = pair
    if key == "coverage":
        va, vb = metrics_t[a]["coverage"], metrics_t[b]["coverage"]
        return a if va > vb else b if vb > va else "tie"
    if key == "violations":
        va, vb = metrics_t[a]["prereq"]["share"], metrics_t[b]["prereq"]["share"]
        return a if va < vb else b if vb < va else "tie"
    if key == "closure":
        va = metrics_t[a].get("gap_closure", {}).get("closed", 0)
        vb = metrics_t[b].get("gap_closure", {}).get("closed", 0)
        return a if va > vb else b if vb > va else "tie"
    raise KeyError(key)


def phase_results(args) -> None:
    r = load_banked()["task_showdown_p1"]
    judging, metrics = r["judging"], r["metrics"]
    collect = r["collect"]
    recut = r["postcutoff_recut"]
    asn = r["judge_assignments"]["table"]
    fw = judging["fourway"]
    wm = judging["win_matrix"]
    snips = judging["rationale_snippets"]

    pair_keys = [f"{a}-v-{b}" for a, b in PAIRS]
    L = [
        "# Curriculum Showdown — Phase 1 results (blind 2×2 ToC tournament)",
        "",
        "3 topics × 3 arms (algo = frozen Phase-0 corpus-structure generator; agent = "
        "gemma4:31b-it-qat via pi with web, best-of-3 internal pick; hybrid = gemma "
        "polish of the algo skeleton, order locked). Blind pairwise judging: "
        "gemma4:31b-it-qat via pi, offline, provenance stripped, A/B seeded "
        "(default_rng(42)), 4 judgments per pair (2 orders × 2 samples), majority "
        "wins, 2–2 = tie. Protocol locked in the worklog before the first judgment.",
        "",
        f"Post-cutoff slice (locked on counts BEFORE pull): {recut['picked']['ids']} "
        f"Topic Modeling, window {recut['window_from']}→open, "
        f"{recut['picked']['count']:,} works (cleaned to "
        f"{r['corpus_postcutoff']['n_used']:,} used).",
        "",
        "## Blind-judge win matrix (per-pair majority of 4)",
        "",
        "| topic | " + " | ".join(pair_keys) + " |",
        "|---|" + "---|" * len(pair_keys),
    ]
    for t in TOPIC_ORDER:
        cells = []
        for pk in pair_keys:
            maj = wm[t][pk]
            votes = fw[f"{t}_{pk}"]["votes"]
            cells.append(f"**{maj}** ({'-'.join(str(v) for v in votes.values())})")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "Vote counts are (first arm)-(second arm) of the pair name; "
          "INVALID judgments (unparseable verdicts) count as no-votes.", "",
          "## Full 4-way tables", ""]
    for t in TOPIC_ORDER:
        for pk in pair_keys:
            c = fw[f"{t}_{pk}"]["cells"]
            L.append(f"- **{t} / {pk}**: o1s1={c['o1_s1']}, o1s2={c['o1_s2']}, "
                     f"o2s1={c['o2_s1']}, o2s2={c['o2_s2']} → "
                     f"majority **{fw[f'{t}_{pk}']['majority']}**")
    L += ["", "## Objective metrics", "",
          "| topic | arm | chapters | sections (zero-dropped) | coverage@r | "
          "median d(nearest rep) | prereq-violation share (best possible) | "
          "gap closure |",
          "|---|---|---|---|---|---|---|---|"]
    for t in TOPIC_ORDER:
        for arm in ARMS:
            m = metrics[t][arm]
            bp = m["prereq"].get("best_possible_share")
            clo = (f"{m['gap_closure']['closed']}/12"
                   if "gap_closure" in m else "—")
            L.append(
                f"| {t} | {arm} | {m['n_chapters']} | {m['n_sections']} "
                f"({m['n_dropped_zero']}) | {m['coverage']:.3f} | "
                f"{m['median_nearest_dist']:.3f} | {m['prereq']['share']:.3f}"
                f"{f' ({bp:.3f})' if bp is not None else ''} | {clo} |")
    L += ["",
          f"Radius r = 1.0181 for every topic (DS: the banked t3 q25 scope "
          "threshold; causal/postcutoff: the same value pre-committed as a fixed "
          "angular yardstick — coverage comparable WITHIN a topic only). "
          "CAVEAT (Phase 0, on record): coverage structurally favors the algo arm "
          "(its sections tile the very corpus); read it WITH the judge lens. "
          "Violation shares are comparable across arms; counts are not (different "
          "partitions → different cross-chapter edge bases).",
          "", "## Disagreement table (blind-judge winner ≠ metric leader)", ""]
    n_dis = 0
    metric_keys = ("coverage", "violations", "closure")
    L += ["| topic | pair | judge majority | coverage leader | violations leader "
          "| closure leader |", "|---|---|---|---|---|---|"]
    disagreements = []
    for t in TOPIC_ORDER:
        for pair in PAIRS:
            pk = f"{pair[0]}-v-{pair[1]}"
            maj = wm[t][pk]
            leads = {}
            for mk in metric_keys:
                if mk == "closure" and t != "datascience":
                    leads[mk] = "—"
                    continue
                leads[mk] = _metric_leader(metrics[t], pair, mk)
            row_dis = [mk for mk in metric_keys
                       if leads[mk] not in ("—", "tie", maj)]
            mark = lambda mk: (f"**{leads[mk]}≠**" if mk in row_dis
                               else leads[mk])  # noqa: E731
            L.append(f"| {t} | {pk} | {maj} | {mark('coverage')} | "
                     f"{mark('violations')} | {mark('closure')} |")
            if maj != "tie":
                for mk in row_dis:
                    disagreements.append({"topic": t, "pair": pk,
                                          "judge": maj, "metric": mk,
                                          "leader": leads[mk]})
                    n_dis += 1
    L += ["", f"{n_dis} disagreement cells (judge-decided pairs only; ties "
          "excluded from the count). Divergence is a finding — judge prior-bias "
          "vs metric construction-bias — not noise.", "",
          "## Judge rationale excerpts (one per pair, ≤3 lines)", ""]
    for t in TOPIC_ORDER:
        for pair in PAIRS:
            pk = f"{t}_{pair[0]}-v-{pair[1]}"
            maj = fw[pk]["majority"]
            snip = None
            for o in (1, 2):
                for s2 in (1, 2):
                    key = f"judge_{pk}_o{o}_s{s2}"
                    if key in snips and (maj == "tie" or asn[key]["A"] == maj
                                         or asn[key]["B"] == maj):
                        if maj == "tie" or fw[pk]["cells"][f"o{o}_s{s2}"] == maj:
                            snip = snips[key]
                            break
                if snip:
                    break
            if snip:
                L.append(f"- **{t} / {pair[0]}-v-{pair[1]}** (majority {maj}): "
                         f"{snip}")
    # interaction verdict
    av = {t: wm[t]["algo-v-agent"] for t in TOPIC_ORDER}
    cov = {t: {a: metrics[t][a]["coverage"] for a in ARMS} for t in TOPIC_ORDER}
    L += ["", "## Interaction verdict", "",
          f"Predicted: agent wins the canonical topic, algo wins niche/post-cutoff. "
          f"Observed algo-v-agent majorities — datascience: **{av['datascience']}**, "
          f"causal: **{av['causal']}**, postcutoff: **{av['postcutoff']}**.",
          "",
          "**Plain language.** At the judge lens there is NO interaction: the "
          "blind gemma judge prefers the LLM-authored ToC in every topic and every "
          "pair (36/36 individual judgments, stable across presentation orders and "
          "samples), reading the algo arm's register-filtered term-tuples as a "
          "'literature clustering report', not a textbook. At the metric lens the "
          "predicted interaction APPEARS as a monotone gradient: the agent ToC's "
          "grip on the actual field mass decays with topic obscurity — coverage "
          f"{cov['datascience']['agent']:.2f} (canonical) → "
          f"{cov['causal']['agent']:.2f} (niche) → "
          f"{cov['postcutoff']['agent']:.2f} (post-cutoff) at the same radius and "
          "the same rep construction, while the algo arm holds ~0.99 everywhere "
          "and the hybrid recovers roughly half the deficit "
          f"({cov['datascience']['hybrid']:.2f} / {cov['causal']['hybrid']:.2f} / "
          f"{cov['postcutoff']['hybrid']:.2f}). Same direction in ordering: agent "
          "prereq-violation share rises from 0.47 (canonical) to 0.58 "
          "(post-cutoff) while algo improves to 0.28. The post-cutoff rationales "
          "show WHY the lenses split: the judge rewards the ToC that matches its "
          "PRIOR of the topic name (classic topic modeling + a 2026 frontier "
          "gloss) and penalizes the corpus-shaped candidates for covering what "
          "the post-2025 slice actually contains (RAG, agents, compression, "
          "clinical NLP) — judge prior-bias and metric construction-bias are "
          "both real, and the hybrid arm (judge-preferred over algo, "
          "metric-stronger than agent) is the only arm that wins something "
          "under BOTH lenses.",
          ""]
    agent_runs = {t: collect[t] for t in TOPIC_ORDER}
    L += ["---", "",
          "Provenance: results assembled from results.json:task_showdown_p1 "
          "(judging/metrics/collect rows); 9 candidates under showdown/p1_*.md; "
          "agent picks: " + ", ".join(
              f"{t}=draft{agent_runs[t]['agent_pick']}({agent_runs[t]['agent_angle']})"
              for t in TOPIC_ORDER) + "; protocol locked in "
          "docs/gap-detection-worklog.md §Curriculum Showdown (Phase 1)."]
    (SHOW_DIR / "p1_results.md").write_text("\n".join(L) + "\n")
    bank("results_summary", {
        "win_matrix": wm,
        "algo_v_agent_by_topic": av,
        "n_disagreement_cells": n_dis,
        "disagreements": disagreements,
    })
    log(f"wrote {SHOW_DIR / 'p1_results.md'}")


# --------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    sub.add_parser("recount").set_defaults(fn=phase_recount)
    p = sub.add_parser("pull")
    p.add_argument("--max-pages", type=int, default=None)
    p.set_defaults(fn=phase_pull)
    for name, fn in (("corpus-causal", phase_corpus_causal),
                     ("corpus-pc", phase_corpus_pc),
                     ("corpus-ds", phase_corpus_ds),
                     ("toc", phase_toc),
                     ("pickmake", phase_pickmake),
                     ("collect", phase_collect),
                     ("judgemake", phase_judgemake),
                     ("tally", phase_tally),
                     ("metrics", phase_metrics),
                     ("results", phase_results)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
