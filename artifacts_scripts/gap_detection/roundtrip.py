"""Curriculum Showdown R0 — round-trip operators E (extract) / D (derive), M-side cells.

Round-trip program: E maps a ToC's chapters to per-chapter keyword sets; D maps
keyword sets back to a ToC. R0 builds the mechanical (M) side and computes every
CPU-only cell against the FROZEN Phase-0/1 embeddings and caches (no network):

  E_M closure  — chapter doc set -> register-filtered closed-frequent-itemset
                 family (detectors.mine_frequent_itemsets + closed_filter,
                 minsup = max(2, ceil(MINSUP_SHARE*|D|)), max_len=2) -> term union
                 ranked by within-chapter support, capped at 40 (practical FCA
                 intent). MINSUP_SHARE revised 0.25 -> 0.05 on record (worklog):
                 the brief-suggested 25% was degenerate on every real candidate.
  E_M c-TF-IDF — per-chapter top-40 c-TF-IDF terms (the P1b comparator), same
                 junk+register column filters; f_t / register always relative to
                 the FULL topic corpus.
  D_M respect  — chapters = extent(K_i) (best-match share, ties -> earliest
                 published chapter; zero-match docs unassigned); re-derive ONLY
                 sections, labels, exemplars, ordering (frozen generator rules).
  D_M free     — extent(pool) = docs containing >=1 pooled keyword; the frozen
                 Phase-1 generator (showdown_p1.build_toc) end-to-end on the
                 induced sub-corpus (chapter k in {8..14} if n>=15K else {5..10}).

Cells (worklog "Curriculum Showdown (R0 round-trip)" pre-commits, recorded
BEFORE any computation): R_MM idempotence on the 3 algo ToCs; R_AM improvement
on the 3 agent ToCs (both modes, banked metric suite, derived ToCs emitted as
showdown/r0_<topic>_agent_RAM_{respect,free}.md); fidelity tables 2 extractors
x 2 modes x 9 candidates; alignment energy delta(T) = 1 - mean per-chapter
best-match Jaccard vs R_MM-free(T) (closure headline, c-TF-IDF sensitivity)
paired with the coverage delta; extractor arbitration.

Phases (run from bookgap root):
  uv run python -m artifacts_scripts.gap_detection.roundtrip <phase>
  extract — E_M for all 9 P1 candidates x 2 extractors (cached pickle).
  derive  — D_M for all 9 x 2 extractors x 2 modes (36 derivations, per-item
            pickle cache; the 6 DS free derivations dominate the runtime).
  score   — fidelity tables, R_AM metric suite (+ originals), alignment energy,
            extractor arbitration; banked under results.json:task_showdown_r0.
  report  — showdown/r0_roundtrip.md + the 6 derived agent ToC .md files.

Everything seeded (SEED=42); CPU only; threads capped at 8.
Context hygiene: titles, top-terms, counts, numbers ONLY — never abstract text.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # a GPU book build owns the card; CPU only
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import pickle
import time
from datetime import date
from math import ceil
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.book_audit import full_statistic
from artifacts_scripts.gap_detection.curriculum_toc import (
    EXTRA_JUNK,
    N_EXEMPLARS,
    N_SECTION_TERMS,
    _kmeans_sweep,
    cluster_terms,
)
from artifacts_scripts.gap_detection.detectors import (
    closed_filter,
    knn_radii,
    mine_frequent_itemsets,
)
from artifacts_scripts.gap_detection.run_eval import _py
from artifacts_scripts.gap_detection.run_followups import RESULTS_PATH, load_banked
from artifacts_scripts.gap_detection.run_t3_book_audit import (
    JUNK_TERMS,
    K_BG,
    K_FG,
    MIN_COMP_SIZE,
    REGISTER_DF_SHARE,
    TOP_FRAC,
)
from artifacts_scripts.gap_detection.showdown_p1 import (
    ARMS,
    CACHE_DIR,
    EMBED_NPZ_DS,
    REGIONS_NPZ_DS,
    SHOW_DIR,
    TOPIC_ORDER,
    TOPICS,
    _candidate_reps,
    assign_by_nearest,
    build_toc,
    write_p1_toc_md,
)
from artifacts_scripts.gap_detection.toc_metrics import (
    citation_flow,
    forward_violations,
    gap_bar,
    nearest_rep_dist,
    order_min_forward_flow,
    region_closure_fullstat,
    select_k,
)

_T0 = time.monotonic()
SEED = 42
MINSUP_SHARE = 0.05          # iceberg min-support share. REVISED 0.25 -> 0.05 on record
                             # (worklog): 0.25 was degenerate (0-6 terms, empty chapters);
                             # 0.05 = largest probed share keeping >=5 terms per chapter
                             # everywhere; the 40-cap binds as intended.
TERM_CAP = 40                # pre-committed keyword-set cap (both extractors)
SMALL_CORPUS = 15_000        # frozen Phase-1 k-range rescale threshold
RADIUS = 1.0181              # banked t3 q25 scope threshold / fixed yardstick
MIN_SECTION_DOCS = 8         # pre-committed: chapters below this get ONE section
EXTRACTORS = ("closure", "ctfidf")
MODES = ("respect", "free")

RT_DIR = CACHE_DIR / "roundtrip"
KW_PKL = RT_DIR / "r0_keywords.pkl"


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def bank(sub: str, payload: dict) -> None:
    """Merge `payload` under results.json:task_showdown_r0.<sub>."""
    r = load_banked()
    cur = r.get("task_showdown_r0", {})
    cur[sub] = _py(payload)
    r["task_showdown_r0"] = cur
    RESULTS_PATH.write_text(json.dumps(r, indent=1))
    log(f"banked results.json:task_showdown_r0.{sub}")


# ------------------------------------------------------------------ topic data
def topic_data(topic: str) -> dict:
    """Frozen per-topic arrays: Z, binary keyword matrix K (top-2000-df vocab),
    vocab, titles, year, citation edges. Rows = the topic corpus."""
    if topic == "datascience":
        z = np.load(EMBED_NPZ_DS)
        Z, year = z["Zb"], z["year"]
        K = sp.load_npz(CACHE_DIR / "showdown_K.npz").tocsr()
        with (CACHE_DIR / "showdown_meta.pkl").open("rb") as f:
            meta = pickle.load(f)
        e = np.load(CACHE_DIR / "showdown_edges.npz")
        src, dst = e["src"], e["dst"]
    else:
        z = np.load(CACHE_DIR / f"p1_{topic}_arrays.npz")
        Z, year, src, dst = z["Z"], z["year"], z["src"], z["dst"]
        K = sp.load_npz(CACHE_DIR / f"p1_{topic}_K.npz").tocsr()
        with (CACHE_DIR / f"p1_{topic}_meta.pkl").open("rb") as f:
            meta = pickle.load(f)
    vocab = [str(v) for v in meta["vocab"]]
    junk = np.array([v in JUNK_TERMS or v in EXTRA_JUNK for v in vocab], dtype=bool)
    return {"Z": Z.astype(np.float32), "K": K, "vocab": vocab,
            "titles": [str(t) for t in meta["title"]],
            "year": year.astype(np.int64),
            "src": src.astype(np.int64), "dst": dst.astype(np.int64),
            "junk": junk}


def chapter_doc_sets(chap: np.ndarray, k: int) -> list[np.ndarray]:
    """Doc-assignment vector (published chapter positions) -> per-chapter doc sets."""
    return [np.where(chap == i)[0] for i in range(k)]


# ------------------------------------------------------------------ extractors
def register_mask(K: sp.csr_matrix, members: np.ndarray,
                  share: float = REGISTER_DF_SHARE) -> tuple[np.ndarray, np.ndarray]:
    """(register_cols, inside_df) for doc set `members` vs corpus-minus-members.
    The t3 register rule ported chapter-wise: a term with df-share >= `share`
    BOTH inside D and in corpus\\D is genre register, not chapter vocabulary."""
    members = np.asarray(members, dtype=np.int64)
    n = K.shape[0]
    n_in = float(members.size)
    n_out = float(n - members.size)
    inside = np.asarray(K[members].sum(axis=0), dtype=np.float64).ravel()
    total = np.asarray(K.sum(axis=0), dtype=np.float64).ravel()
    outside = total - inside
    reg = (inside / max(n_in, 1.0) >= share) & (outside / max(n_out, 1.0) >= share)
    return reg, inside


def extract_closure(K: sp.csr_matrix, members: np.ndarray, junk: np.ndarray,
                    minsup_share: float = MINSUP_SHARE, cap: int = TERM_CAP) -> dict:
    """E_M closure variant (pre-committed): register-filtered closed-frequent-
    itemset machinery, iceberg-style; keyword set = the closed family's term
    union ranked by within-chapter support desc (tie: vocab index asc), cap 40.

    Returns {"cols": [vocab col ids], "support": [...], "minsup", "n_docs",
    "n_register", "n_closed_family"}."""
    members = np.asarray(members, dtype=np.int64)
    n_in = int(members.size)
    if n_in < 2:
        return {"cols": [], "support": [], "minsup": None, "n_docs": n_in,
                "n_register": 0, "n_closed_family": 0}
    reg, inside = register_mask(K, members)
    kept = np.where(~junk & ~reg)[0]
    minsup = max(2, int(ceil(minsup_share * n_in)))
    Kd = np.asarray(K[members][:, kept].todense(), dtype=bool)
    family = closed_filter(
        mine_frequent_itemsets(Kd, min_support=minsup, max_len=2))
    local_terms = sorted({t for S in family for t in S})
    cols = [int(kept[t]) for t in local_terms]
    sup = [int(inside[c]) for c in cols]
    order = sorted(range(len(cols)), key=lambda i: (-sup[i], cols[i]))[:cap]
    return {"cols": [cols[i] for i in order], "support": [sup[i] for i in order],
            "minsup": minsup, "n_docs": n_in,
            "n_register": int((reg & ~junk).sum()), "n_closed_family": len(family)}


def extract_ctfidf(K: sp.csr_matrix, chapters: list[np.ndarray],
                   junk: np.ndarray, cap: int = TERM_CAP) -> list[dict]:
    """E_M c-TF-IDF variant (the P1b comparator; pre-committed): per chapter,
    top-`cap` terms by W(t,c) = tf(t,c)/sum_t tf(t,c) * ln(1 + A/f_t), with
    tf = within-chapter df count (binary K), f_t = FULL topic-corpus df,
    A = mean class total; junk + per-chapter register columns excluded.
    Totals (class totals, f_t, A) are over the junk-dropped vocabulary."""
    nonjunk = ~junk
    f_t = np.asarray(K.sum(axis=0), dtype=np.float64).ravel()
    f_t[junk] = 0.0
    tfs, regs = [], []
    for m in chapters:
        m = np.asarray(m, dtype=np.int64)
        reg, inside = register_mask(K, m) if m.size else (
            np.zeros(K.shape[1], dtype=bool), np.zeros(K.shape[1]))
        tfs.append(inside)
        regs.append(reg)
    A = float(np.mean([tf[nonjunk].sum() for tf in tfs])) if chapters else 0.0
    out = []
    for tf, reg, m in zip(tfs, regs, chapters):
        tot = float(tf[nonjunk].sum())
        if m.size == 0 or tot <= 0:
            out.append({"cols": [], "weight": [], "n_docs": int(np.size(m)),
                        "n_register": int((reg & ~junk).sum())})
            continue
        with np.errstate(divide="ignore"):
            W = (tf / tot) * np.log1p(A / np.maximum(f_t, 1.0))
        W[junk | reg | (tf <= 0)] = -np.inf
        order = np.lexsort((np.arange(W.size), -W))
        cols = [int(c) for c in order[:cap] if np.isfinite(W[c])]
        out.append({"cols": cols, "weight": [round(float(W[c]), 6) for c in cols],
                    "n_docs": int(m.size), "n_register": int((reg & ~junk).sum())})
    return out


def extract_sets(K: sp.csr_matrix, chapters: list[np.ndarray], junk: np.ndarray,
                 extractor: str) -> list[dict]:
    """Uniform driver: per-chapter keyword dicts for either extractor."""
    if extractor == "closure":
        return [extract_closure(K, m, junk) for m in chapters]
    if extractor == "ctfidf":
        return extract_ctfidf(K, chapters, junk)
    raise KeyError(extractor)


# ------------------------------------------------------------------ extents (D_M)
def extent_respect(K: sp.csr_matrix, kw_sets: list[list[int]]) -> np.ndarray:
    """Pre-committed respect-partition extent: doc -> argmax_i |d ∩ K_i|/|K_i|
    over chapters with match > 0; ties -> earliest published chapter (argmax
    first-max); zero match everywhere -> -1 (unassigned). Empty K_i never wins."""
    n, _ = K.shape
    k = len(kw_sets)
    M = np.zeros((n, k), dtype=np.float32)
    Kf = K.astype(np.float32)
    for i, cols in enumerate(kw_sets):
        if not cols:
            continue
        hits = np.asarray(
            Kf[:, np.asarray(cols, dtype=np.int64)].sum(axis=1)).ravel()
        M[:, i] = hits / float(len(cols))
    labels = np.argmax(M, axis=1).astype(np.int64)
    labels[M.max(axis=1) <= 0.0] = -1
    return labels


def extent_free(K: sp.csr_matrix, kw_sets: list[list[int]]) -> np.ndarray:
    """Pre-committed free extent: pool = union of keyword sets; the induced
    sub-corpus = docs containing >= 1 pooled term."""
    pool = sorted({c for cols in kw_sets for c in cols})
    if not pool:
        return np.empty(0, dtype=np.int64)
    hits = np.asarray(
        K[:, np.asarray(pool, dtype=np.int64)].sum(axis=1)).ravel()
    return np.where(hits >= 1)[0].astype(np.int64)


# ------------------------------------------------------------------ derivation
def _derive_sections(Z: np.ndarray, K: sp.csr_matrix, vocab: list[str],
                     titles: list[str], year: np.ndarray, m: np.ndarray,
                     src: np.ndarray, dst: np.ndarray,
                     in_chapter: np.ndarray) -> tuple[list[dict], int, str, float]:
    """Frozen section machinery for one chapter doc set `m` (global indices):
    KMeans k in {3..min(8, n-1)} by the pre-committed rule (n < MIN_SECTION_DOCS
    -> one section), within-chapter exact-DP ordering, register-filtered labels,
    title-deduped exemplars. `in_chapter` masks edges with both endpoints in m.
    Returns (sections, section_k, rule_why, section_forward_share)."""
    n = int(m.size)
    if n >= MIN_SECTION_DOCS:
        rows, labelings = _kmeans_sweep(Z[m], range(3, min(8, n - 1) + 1))
        ks, why = select_k(rows)
        sub = labelings[ks].astype(np.int64)
    else:
        ks, why = 1, "single-section (n < 8, pre-committed edge rule)"
        sub = np.zeros(n, dtype=np.int64)
    pos_in_m = np.full(Z.shape[0], -1, dtype=np.int64)
    pos_in_m[m] = np.arange(n)
    F_sub = citation_flow(sub[pos_in_m[src[in_chapter]]],
                          sub[pos_in_m[dst[in_chapter]]], ks)
    med_sub = np.array([float(np.median(year[m[sub == s]])) for s in range(ks)])
    sec_order, _ = order_min_forward_flow(F_sub, med_sub)
    sec_viol = forward_violations(F_sub, sec_order)
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
    return sections, ks, why, round(sec_viol["share"], 4)


def derive_respect(td: dict, labels: np.ndarray, orig_k: int) -> dict:
    """D_M respect-partition: chapter identities = extent(K_i) (labels array,
    -1 = unassigned); re-derive sections, labels, exemplars, and ordering only
    (frozen generator rules). Returns a build_toc-shaped dict; cluster_id =
    the chapter's ORIGINAL published position; F indexed by original position."""
    Z, K, vocab, titles, year = td["Z"], td["K"], td["vocab"], td["titles"], td["year"]
    src, dst = td["src"], td["dst"]
    present = [c for c in range(orig_k) if int((labels == c).sum()) > 0]
    ok_edge = (labels[src] >= 0) & (labels[dst] >= 0)
    F = citation_flow(labels[src[ok_edge]], labels[dst[ok_edge]], orig_k)
    med_year = np.array([float(np.median(year[labels == c])) if c in present
                         else np.inf for c in range(orig_k)])
    # DP over PRESENT chapters only (remap to compact ids, then map back)
    Fp = F[np.ix_(present, present)]
    sub_order, _ = order_min_forward_flow(Fp, med_year[np.asarray(present)])
    chap_order = [int(present[i]) for i in sub_order]
    opt = forward_violations(F, chap_order)
    chapters = []
    for c in chap_order:
        m = np.where(labels == c)[0]
        in_c = ok_edge & (labels[src] == c) & (labels[dst] == c)
        sections, ks, why, sec_share = _derive_sections(
            Z, K, vocab, titles, year, m, src, dst, in_c)
        c_terms, c_register = cluster_terms(K, m, vocab, n_top=10)
        chapters.append({
            "cluster_id": int(c), "n": int(m.size),
            "share": round(m.size / Z.shape[0], 4),
            "terms": c_terms, "register_excluded": c_register,
            "median_year": float(np.median(year[m])),
            "section_k": ks, "section_rule": why,
            "section_forward_share": sec_share,
            "sections": sections,
        })
        log(f"  respect chapter pos {c}: n={m.size:,}, sections k={ks}, "
            f"terms {c_terms[:4]}")
    return {"k": orig_k, "mode": "respect", "chap_order": chap_order,
            "forward_flow": opt, "F": F, "chapters": chapters,
            "labels": labels, "present": present,
            "n_unassigned": int((labels == -1).sum()),
            "n_dropped_chapters": orig_k - len(present)}


def derive_free(td: dict, rows: np.ndarray) -> dict:
    """D_M free: the frozen Phase-1 generator end-to-end on the induced
    sub-corpus (chapter k in {8..14} if n >= 15K else {5..10})."""
    Z, K, vocab, titles, year = td["Z"], td["K"], td["vocab"], td["titles"], td["year"]
    src, dst = td["src"], td["dst"]
    old_to_new = np.full(Z.shape[0], -1, dtype=np.int64)
    old_to_new[rows] = np.arange(rows.size)
    ok = (old_to_new[src] >= 0) & (old_to_new[dst] >= 0)
    src2, dst2 = old_to_new[src[ok]], old_to_new[dst[ok]]
    k_range = range(8, 15) if rows.size >= SMALL_CORPUS else range(5, 11)
    toc = build_toc(Z[rows], K[rows].tocsr(), vocab,
                    [titles[i] for i in rows], year[rows],
                    src2, dst2, k_range)
    toc["mode"] = "free"
    toc["rows"] = rows
    return toc


def derived_doc_sets(d: dict, n_docs: int) -> list[np.ndarray]:
    """GLOBAL per-chapter doc sets of a derived ToC, in DERIVED order."""
    if d["mode"] == "respect":
        return [np.where(d["labels"] == c)[0] for c in d["chap_order"]]
    rows = d["rows"]
    return [rows[d["labels"] == c] for c in d["chap_order"]]


def derived_reps(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(section centroids, section->derived-chapter index) for a derived ToC."""
    reps = np.vstack([s["centroid"] for ch in d["chapters"] for s in ch["sections"]])
    sec_chap = np.array([i for i, ch in enumerate(d["chapters"])
                         for _ in ch["sections"]], dtype=np.int64)
    return reps.astype(np.float32), sec_chap


# ------------------------------------------------------------------ fidelity
def jaccard(a: set, b: set) -> float:
    """Pre-committed: J(empty, empty) = 1; J(A, empty) = 0."""
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def fidelity_respect(orig_sets: list[list[int]], new_sets_by_pos: dict[int, list[int]],
                     ) -> tuple[float, list[float]]:
    """Identity-matched: mean_i J(K_i, E(extent(K_i))); dropped chapters -> 0."""
    per = []
    for i, cols in enumerate(orig_sets):
        new = new_sets_by_pos.get(i)
        per.append(0.0 if new is None else jaccard(set(cols), set(new)))
    return float(np.mean(per)) if per else 0.0, [round(p, 4) for p in per]


def fidelity_free(orig_sets: list[list[int]], new_sets: list[list[int]],
                  ) -> tuple[float, list[float]]:
    """Best-match: mean_i max_j J(K_i, K'_j)."""
    per = []
    for cols in orig_sets:
        a = set(cols)
        per.append(max((jaccard(a, set(b)) for b in new_sets), default=0.0))
    return float(np.mean(per)) if per else 0.0, [round(p, 4) for p in per]


# ------------------------------------------------------------------ phase: extract
def candidate_assignment(topic: str, arm: str) -> dict:
    """P1 doc-assignment convention reused verbatim (_candidate_reps)."""
    c = _candidate_reps(topic, arm)
    return c


def phase_extract(args) -> None:
    RT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    diag_bank: dict = {}
    for topic in TOPIC_ORDER:
        td = topic_data(topic)
        for arm in ARMS:
            c = candidate_assignment(topic, arm)
            chapters = chapter_doc_sets(c["chap"], c["k"])
            for ext in EXTRACTORS:
                sets = extract_sets(td["K"], chapters, td["junk"], ext)
                out[(topic, arm, ext)] = sets
                sizes = [len(s["cols"]) for s in sets]
                diag_bank[f"{topic}/{arm}/{ext}"] = {
                    "k": c["k"], "chapter_sizes": [int(m.size) for m in chapters],
                    "set_sizes": sizes,
                    "minsup": [s.get("minsup") for s in sets] if ext == "closure" else None,
                    "n_closed_family": [s.get("n_closed_family") for s in sets]
                    if ext == "closure" else None,
                }
                log(f"{topic}/{arm}/{ext}: k={c['k']} set sizes {sizes}")
    with KW_PKL.open("wb") as f:
        pickle.dump(out, f)
    bank("extraction", {
        "minsup_share": MINSUP_SHARE, "term_cap": TERM_CAP,
        "register_df_share": REGISTER_DF_SHARE,
        "doc_assignment": "P1 _candidate_reps convention (algo=KMeans partition; "
                          "agent/hybrid=nearest kept section rep)",
        "rows": diag_bank,
    })
    log(f"wrote {KW_PKL}")


# ------------------------------------------------------------------ phase: derive
def _dpath(topic: str, arm: str, ext: str, mode: str) -> Path:
    return RT_DIR / f"d_{topic}_{arm}_{ext}_{mode}.pkl"


def phase_derive(args) -> None:
    with KW_PKL.open("rb") as f:
        kw = pickle.load(f)
    todo = [(t, a, e, m) for t in TOPIC_ORDER for a in ARMS
            for e in EXTRACTORS for m in MODES]
    if args.only:
        todo = [x for x in todo if "_".join(x) in set(args.only)]
    for topic, arm, ext, mode in todo:
        p = _dpath(topic, arm, ext, mode)
        if p.exists():
            log(f"cached: {p.name}")
            continue
        td = topic_data(topic)
        sets = [s["cols"] for s in kw[(topic, arm, ext)]]
        log(f"derive {topic}/{arm}/{ext}/{mode}: k={len(sets)}")
        if mode == "respect":
            labels = extent_respect(td["K"], sets)
            d = derive_respect(td, labels, len(sets))
        else:
            rows = extent_free(td["K"], sets)
            log(f"  free extent: {rows.size:,}/{td['Z'].shape[0]:,} docs")
            d = derive_free(td, rows)
            d["n_extent"] = int(rows.size)
        with p.open("wb") as f:
            pickle.dump(d, f)
        log(f"wrote {p.name} ({len(d['chap_order'])} chapters)")


# ------------------------------------------------------------------ phase: score
def _ds_closure_scaffold() -> dict:
    """The banked DS gap-closure machinery (P1 phase_metrics verbatim)."""
    z = np.load(EMBED_NPZ_DS)
    Zf, Zb = z["Zf"], z["Zb"]
    with (CACHE_DIR / "showdown_meta.pkl").open("rb") as f:
        thr = float(pickle.load(f)["thr"])
    r_F = knn_radii(Zb, Zf, K_FG)
    scoped_global = np.where(r_F <= thr)[0]
    Zs = Zb[scoped_global]
    pos = {int(g): i for i, g in enumerate(scoped_global)}
    rz = np.load(REGIONS_NPZ_DS)
    regions_local = [np.asarray([pos[int(g)] for g in rz[f"region_{i:02d}"]],
                                dtype=np.int64) for i in range(12)]
    full_ds2 = full_statistic(Zf, Zs, k_fg=K_FG, k_bg=K_BG)["full"]
    bar = gap_bar(full_ds2, top_frac=TOP_FRAC, min_size=MIN_COMP_SIZE)
    log(f"ds closure scaffold: scoped {Zs.shape[0]:,}, bar {bar:.4f}")
    return {"Zs": Zs, "regions_local": regions_local, "bar": bar}


def metric_row(topic: str, Z: np.ndarray, year: np.ndarray,
               src: np.ndarray, dst: np.ndarray,
               reps: np.ndarray, chap: np.ndarray, k: int, order: list[int],
               scaffold: dict | None) -> dict:
    """The banked P1 metric suite for one (candidate or derived) ToC."""
    d = nearest_rep_dist(Z, reps)
    F = citation_flow(chap[src], chap[dst], k)
    v = forward_violations(F, order)
    row = {"n_chapters": k, "n_sections": int(reps.shape[0]),
           "coverage": round(float((d <= RADIUS).mean()), 4),
           "median_nearest_dist": round(float(np.median(d)), 4),
           "prereq": {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                      for kk, vv in v.items()}}
    if k <= 16:
        med_year = np.array([float(np.median(year[chap == i]))
                             if (chap == i).any() else 0.0 for i in range(k)])
        _, opt_flow = order_min_forward_flow(F, med_year)
        row["prereq"]["best_possible_share"] = round(
            opt_flow / v["cross_edges"], 4) if v["cross_edges"] else 0.0
    if topic == "datascience" and scaffold is not None:
        full_T = full_statistic(reps, scaffold["Zs"], k_fg=K_FG, k_bg=K_BG)["full"]
        clo = region_closure_fullstat(full_T, scaffold["regions_local"],
                                      scaffold["bar"])
        row["gap_closure"] = {"closed": int(sum(clo)), "of": 12, "per_region": clo}
    return row


def phase_score(args) -> None:
    with KW_PKL.open("rb") as f:
        kw = pickle.load(f)
    scaffold = _ds_closure_scaffold()
    banked_p1 = load_banked()["task_showdown_p1"]["metrics"]

    fidelity: dict = {}
    derived_metrics: dict = {}
    original_metrics: dict = {}
    energy: dict = {}
    deriv_diag: dict = {}

    for topic in TOPIC_ORDER:
        td = topic_data(topic)
        Z, year, src, dst = td["Z"], td["year"], td["src"], td["dst"]
        for arm in ARMS:
            # ----- original candidate metrics (P1 conventions, sanity-gated)
            c = candidate_assignment(topic, arm)
            row0 = metric_row(topic, Z, year, src, dst, c["reps"], c["chap"],
                              c["k"], list(range(c["k"])),
                              scaffold if topic == "datascience" else None)
            b = banked_p1[topic][arm]
            assert abs(row0["coverage"] - b["coverage"]) < 5e-4, (topic, arm)
            assert abs(row0["prereq"]["share"] - b["prereq"]["share"]) < 5e-4
            original_metrics[f"{topic}/{arm}"] = row0
            orig_sets = {e: [s["cols"] for s in kw[(topic, arm, e)]]
                         for e in EXTRACTORS}
            for ext in EXTRACTORS:
                for mode in MODES:
                    with _dpath(topic, arm, ext, mode).open("rb") as f:
                        d = pickle.load(f)
                    docsets = derived_doc_sets(d, Z.shape[0])
                    # ----- re-extraction with the SAME extractor
                    if ext == "closure":
                        new = [extract_closure(td["K"], m, td["junk"])["cols"]
                               for m in docsets]
                    else:
                        new = [s["cols"] for s in
                               extract_ctfidf(td["K"], docsets, td["junk"])]
                    if mode == "respect":
                        by_pos = {int(c0): cols for c0, cols in
                                  zip(d["chap_order"], new)}
                        mean_j, per = fidelity_respect(orig_sets[ext], by_pos)
                    else:
                        mean_j, per = fidelity_free(orig_sets[ext], new)
                    fidelity[f"{topic}/{arm}/{ext}/{mode}"] = {
                        "mean_jaccard": round(mean_j, 4), "per_chapter": per}
                    # ----- metric suite on the derived ToC (uniform convention)
                    reps, sec_chap = derived_reps(d)
                    chap = assign_by_nearest(Z, reps, sec_chap)
                    kd = len(d["chap_order"])
                    rowd = metric_row(topic, Z, year, src, dst, reps, chap, kd,
                                      list(range(kd)),
                                      scaffold if topic == "datascience" else None)
                    rowd["n_chapters_dropped"] = d.get("n_dropped_chapters", 0)
                    rowd["n_unassigned"] = d.get("n_unassigned")
                    rowd["n_extent"] = d.get("n_extent")
                    derived_metrics[f"{topic}/{arm}/{ext}/{mode}"] = rowd
                    if mode == "free":
                        key = f"{topic}/{arm}"
                        energy.setdefault(key, {})[f"delta_{ext}"] = round(
                            1.0 - mean_j, 4)
                        if ext == "closure":
                            energy[key]["dcov_closure"] = round(
                                rowd["coverage"] - row0["coverage"], 4)
                            energy[key]["cov_orig"] = row0["coverage"]
                            energy[key]["cov_rt"] = rowd["coverage"]
                    deriv_diag[f"{topic}/{arm}/{ext}/{mode}"] = {
                        "k_derived": kd,
                        "sections": int(reps.shape[0]),
                        "new_set_sizes": [len(x) for x in new]}
                    log(f"{topic}/{arm}/{ext}/{mode}: J {mean_j:.3f} "
                        f"cov {rowd['coverage']:.3f} "
                        f"viol {rowd['prereq']['share']}")

    # ----- extractor arbitration (pre-committed rule)
    idem = {e: {m: round(float(np.mean(
        [fidelity[f"{t}/algo/{e}/{m}"]["mean_jaccard"] for t in TOPIC_ORDER])), 4)
        for m in MODES} for e in EXTRACTORS}
    deltas = {}
    for metric, key in (("coverage", "coverage"), ("violations", "share")):
        for e in EXTRACTORS:
            for m in MODES:
                for t in TOPIC_ORDER:
                    r0 = original_metrics[f"{t}/agent"]
                    rd = derived_metrics[f"{t}/agent/{e}/{m}"]
                    v0 = r0["coverage"] if metric == "coverage" else r0["prereq"]["share"]
                    vd = rd["coverage"] if metric == "coverage" else rd["prereq"]["share"]
                    deltas[(metric, e, m, t)] = vd - v0
    stab = {}
    for metric in ("coverage", "violations"):
        diffs, signs = [], []
        for m in MODES:
            for t in TOPIC_ORDER:
                dc = deltas[(metric, "closure", m, t)]
                dt = deltas[(metric, "ctfidf", m, t)]
                diffs.append(abs(dc - dt))
                signs.append(int(np.sign(dc) == np.sign(dt)))
        stab[metric] = {"mean_abs_delta_gap": round(float(np.mean(diffs)), 4),
                        "sign_agreement": f"{sum(signs)}/{len(signs)}"}
    arbitration = {"m_side_idempotence": idem, "ram_delta_stability": stab}

    bank("fidelity", fidelity)
    bank("derived_metrics", derived_metrics)
    bank("original_metrics", original_metrics)
    bank("energy", energy)
    bank("arbitration", arbitration)
    bank("derivation_diag", deriv_diag)
    log("score phase complete")


# ------------------------------------------------------------------ phase: report
def phase_report(args) -> None:
    r = load_banked()["task_showdown_r0"]
    fid, dm, om, en, arb = (r["fidelity"], r["derived_metrics"],
                            r["original_metrics"], r["energy"], r["arbitration"])

    # emit the 6 derived agent ToCs (closure-fed, pre-committed)
    emitted = []
    for topic in TOPIC_ORDER:
        for mode in MODES:
            with _dpath(topic, "agent", "closure", mode).open("rb") as f:
                d = pickle.load(f)
            path = SHOW_DIR / f"r0_{topic}_agent_RAM_{mode}.md"
            write_p1_toc_md(d, TOPICS[topic]["title"], path)
            emitted.append(path.name)
            log(f"emitted {path.name}")

    def _f(key):
        return fid[key]["mean_jaccard"]

    L = [
        "# Curriculum Showdown — R0 round-trip operators (M-side cells)",
        "",
        "E_M (closure = register-filtered closed-itemset iceberg intents, minsup "
        f"25% cap {TERM_CAP}; c-TF-IDF = the P1b comparator, same filters) and "
        "D_M (respect-partition = chapters from extent(K_i), re-derive "
        "sections/labels/ordering only; free = frozen generator on extent(pool)). "
        "All constructions pre-committed in the worklog BEFORE any cell; "
        "frozen Phase-0/1 spaces; seed 42; CPU only.",
        "",
        "## 1. R_MM idempotence (3 algo ToCs) — mean per-chapter keyword Jaccard",
        "",
        "| topic | closure/respect | closure/free | ctfidf/respect | ctfidf/free |",
        "|---|---|---|---|---|",
    ]
    for t in TOPIC_ORDER:
        L.append(f"| {t} | {_f(f'{t}/algo/closure/respect'):.3f} | "
                 f"{_f(f'{t}/algo/closure/free'):.3f} | "
                 f"{_f(f'{t}/algo/ctfidf/respect'):.3f} | "
                 f"{_f(f'{t}/algo/ctfidf/free'):.3f} |")
    idem = arb["m_side_idempotence"]
    L += ["",
          f"Mean over topics — closure: respect {idem['closure']['respect']:.3f}, "
          f"free {idem['closure']['free']:.3f}; c-TF-IDF: respect "
          f"{idem['ctfidf']['respect']:.3f}, free {idem['ctfidf']['free']:.3f}.",
          "",
          "## 2. R_AM — the improvement operator on the 3 agent ToCs "
          "(closure-fed; original next to derived)",
          "",
          "| topic | candidate | chapters | sections | coverage@r | median d | "
          "violation share (best) | DS closure |",
          "|---|---|---|---|---|---|---|---|"]
    for t in TOPIC_ORDER:
        rows = [("original", om[f"{t}/agent"])]
        rows += [(m, dm[f"{t}/agent/closure/{m}"]) for m in MODES]
        for name, m0 in rows:
            bp = m0["prereq"].get("best_possible_share")
            clo = (f"{m0['gap_closure']['closed']}/12"
                   if "gap_closure" in m0 else "—")
            L.append(f"| {t} | {name} | {m0['n_chapters']} | {m0['n_sections']} | "
                     f"{m0['coverage']:.3f} | {m0['median_nearest_dist']:.3f} | "
                     f"{m0['prereq']['share']:.3f}"
                     f"{f' ({bp:.3f})' if bp is not None else ''} | {clo} |")
    L += ["",
          "Derived ToCs (closure-fed): " + ", ".join(f"`{e}`" for e in emitted) +
          ". Respect keeps the agent's keyword-set chapter identities (scoping) "
          "and re-derives ordering/sections; free reorganizes the pooled "
          "keywords' sub-corpus end-to-end.",
          "",
          "## 3. Round-trip fidelity — all 9 candidates "
          "(mean per-chapter Jaccard; respect identity-matched, free best-match)",
          "",
          "| topic | arm | closure/respect | closure/free | ctfidf/respect | "
          "ctfidf/free |",
          "|---|---|---|---|---|---|"]
    for t in TOPIC_ORDER:
        for a in ARMS:
            L.append(f"| {t} | {a} | {_f(f'{t}/{a}/closure/respect'):.3f} | "
                     f"{_f(f'{t}/{a}/closure/free'):.3f} | "
                     f"{_f(f'{t}/{a}/ctfidf/respect'):.3f} | "
                     f"{_f(f'{t}/{a}/ctfidf/free'):.3f} |")
    L += ["",
          "## 4. Alignment energy δ(T) = 1 − mean best-match Jaccard vs "
          "R_MM-free(T) (closure headline; ctfidf sensitivity; Δcoverage paired)",
          "",
          "| topic | arm | δ closure | δ ctfidf | cov(T) | cov(RT) | Δcov |",
          "|---|---|---|---|---|---|---|"]
    for t in TOPIC_ORDER:
        for a in ARMS:
            e = en[f"{t}/{a}"]
            L.append(f"| {t} | {a} | {e['delta_closure']:.3f} | "
                     f"{e['delta_ctfidf']:.3f} | {e['cov_orig']:.3f} | "
                     f"{e['cov_rt']:.3f} | {e['dcov_closure']:+.3f} |")
    ag = [en[f"{t}/agent"]["delta_closure"] for t in TOPIC_ORDER]
    L += ["",
          f"Agent-arm energy gradient (canonical → niche → post-cutoff): "
          f"{ag[0]:.3f} → {ag[1]:.3f} → {ag[2]:.3f} "
          f"(hypothesis on record: rising).",
          "",
          "## 5. Extractor arbitration",
          "",
          f"- M-side idempotence (cell a): {json.dumps(idem)}",
          f"- R_AM delta stability (cell b): {json.dumps(arb['ram_delta_stability'])}",
          ""]
    if args.read:
        L += ["## Plain-language read", ""] + [f"{ln}" for ln in args.read.split("|")]
    L += ["", "---",
          f"*Provenance: roundtrip.py (phases extract/derive/score/report), "
          f"seed {SEED}, generated {date.today().isoformat()}; pre-commits in "
          "docs/gap-detection-worklog.md §Curriculum Showdown (R0 round-trip); "
          "rows banked under results.json:task_showdown_r0. Regenerate with "
          "`uv run python -m artifacts_scripts.gap_detection.roundtrip "
          "report` rather than editing.*"]
    (SHOW_DIR / "r0_roundtrip.md").write_text("\n".join(L) + "\n")
    bank("report", {"emitted": emitted,
                    "agent_energy_gradient": ag})
    log(f"wrote {SHOW_DIR / 'r0_roundtrip.md'}")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    sub.add_parser("extract").set_defaults(fn=phase_extract)
    d = sub.add_parser("derive")
    d.add_argument("--only", nargs="*", default=None,
                   help="restrict to topic_arm_ext_mode ids")
    d.set_defaults(fn=phase_derive)
    sub.add_parser("score").set_defaults(fn=phase_score)
    rp = sub.add_parser("report")
    rp.add_argument("--read", default=None,
                    help="plain-language read, lines separated by |")
    rp.set_defaults(fn=phase_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
