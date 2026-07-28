"""Curriculum Showdown Phase 0 — algorithmic ToC generator from corpus structure.

Builds a textbook-shaped table of contents PURELY from the structure of a field
corpus (no LLM anywhere in the loop):

  chapters  = KMeans clusters of the corpus in the banked t3 LSA-64 space,
              k chosen by the PRE-COMMITTED rule (max mean silhouette s.t. no
              cluster <3% or >25% of mass; recorded in the worklog before any
              sweep output was looked at);
  sections  = sub-clusters within each chapter (k in {3..8}, same rule);
  labels    = log-odds top terms per cluster over the 2,000-keyword space with
              the t3 register split ported cluster-wise (a term frequent in
              >=4% of docs BOTH inside the cluster and in the rest of the
              slice is genre register, not a topic) + 2 exemplar titles per
              section (nearest the centroid; TITLES ONLY ever emitted);
  ordering  = foundational-first: exact minimum of forward-pointing cross-
              cluster citation flow (members of A citing members of B means B
              is prerequisite mass for A), bitmask DP, median publication year
              as the pre-committed tie-break — chapter AND section grain.

Corpus: the cached Track 3 slice (cache/oa/t3_ds_book, 19 pre-committed
topics, 2019-2025), t3 cleaning verbatim via book_audit.assemble_bg, joint
embed with the 126 ds2 fg units (embed_joint, seed 42, LSA-64) so the
generator, both ToCs, and the 12 banked task_t3 gap regions share ONE space.
fg rows are 126 of 122,486 TF-IDF docs (0.1%; joint-fit caveat on record);
clustering runs on bg rows only. Citation edges come from a dedicated re-pull
of the same slice with select=id,publication_date,referenced_works
(cache/oa/ds_showdown_refs/) because the t3 selects carried no refs.

Recompute-validity gate (asserted in `embed`): cleaning counts, scope thr,
n_scoped and the 12 component sizes must reproduce banked results.json:task_t3.

Phases (subcommands; run from bookgap root as
  uv run python -m artifacts_scripts.gap_detection.curriculum_toc <phase>):
  refs       — cursor-page the refs re-pull (polite pool, resumable, cached).
  embed      — t3 corpus + joint embed + banked-anchor asserts + the 12 region
               member sets; caches arrays for the later phases.
  sweep      — chapter k sweep (k in {8..14}): silhouettes + mass shares; the
               pick is recorded BEFORE any labels exist.
  build      — sections, labels, exemplars, citation ordering; writes
               artifacts/gap_detection_eval/showdown/toc_algo_datascience.md.
  metrics    — the 3-metric objective suite (toc_metrics) for toc-algo AND the
               ds2 agent ToC; writes showdown/side_by_side_ds2.md.
  causal     — Phase 0b: T10845 sub-slice standalone counts + cluster-ability
               (k in {5..10}, the <15K-docs rescale); counts only, NO ToC.
  postcutoff — Phase 0b: /topics + meta.count probes for the post-cutoff
               comparison topic; counts only, NO pull.

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
import gzip
import hashlib
import json
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from artifacts_scripts.gap_detection.book_audit import (
    API_BASE,
    REPO,
    _get_json,
    _session,
    assemble_bg,
    embed_joint,
    full_statistic,
    load_book_fg,
    works_filter,
)
from artifacts_scripts.gap_detection.detectors import knn_radii, level_set_components
from artifacts_scripts.gap_detection.oa_corpus import short_id
from artifacts_scripts.gap_detection.run_eval import _py
from artifacts_scripts.gap_detection.run_followups import RESULTS_PATH, load_banked
from artifacts_scripts.gap_detection.run_t3_book_audit import (
    COMMITTED_TOPICS,
    JUNK_TERMS,
    K_BG,
    K_FG,
    K_GRAPH,
    LSA_DIM,
    MIN_COMP_SIZE,
    PUB_FROM,
    PUB_TO,
    REGISTER_DF_SHARE,
    SCOPE_QUANTILE,
    TOP_FRAC,
    TOPIC_NAMES,
)
from artifacts_scripts.gap_detection.toc_metrics import (
    citation_flow,
    coverage_at_radius,
    forward_violations,
    gap_bar,
    order_min_forward_flow,
    region_closure,
    region_closure_fullstat,
    select_k,
)

_T0 = time.monotonic()
SEED = 42
# MathML/XML namespace debris in OA abstracts (same class as t3's JUNK_TERMS
# LaTeX debris: markup, never topical vocabulary). Added as an on-record
# labeling-hygiene iteration after the first build surfaced {mathml, xmlns,
# mml, xlink} as cluster labels; LABELS ONLY — k, clusters, ordering untouched.
EXTRA_JUNK = frozenset(
    "mathml xmlns mml xlink math mi mo mrow msub msup mfrac amp lt gt".split()
)
N_REGIONS = 12                       # the banked task_t3 report regions
CHAPTER_K_RANGE = range(8, 15)       # brief: k in {8..14}
SECTION_K_RANGE = range(3, 9)        # brief: 3-8 subclusters
SMALL_K_RANGE = range(5, 11)         # brief 0b: k in {5..10} under 15K docs
SIL_SAMPLE = 20_000                  # pre-committed silhouette subsample
N_EXEMPLARS = 2                      # exemplar titles per section
DEP_SHARE = 0.10                     # dependency note: earlier chapters with >=10% out-flow
N_CHAPTER_TERMS = 10
N_SECTION_TERMS = 8

OUT_DIR = REPO / "artifacts/gap_detection_eval"
SHOW_DIR = OUT_DIR / "showdown"
CACHE_DIR = OUT_DIR / "cache"
REFS_CACHE = REPO / "cache/oa/ds_showdown_refs"
TOPICS_CACHE = REPO / "cache/oa/showdown_topics"
REFS_SELECT = "id,publication_date,referenced_works"
PER_PAGE = 200

DS2_TOC_PATH = Path(
    "~/src/learn-and-teach/outputs/2026-06-09/23-15-05/build/state/toc.md"
).expanduser()
DS2_N_CHAPTERS = 9

EMBED_NPZ = CACHE_DIR / "showdown_embed.npz"
META_PKL = CACHE_DIR / "showdown_meta.pkl"
K_NPZ = CACHE_DIR / "showdown_K.npz"
REGIONS_NPZ = CACHE_DIR / "showdown_regions.npz"
SWEEP_JSON = CACHE_DIR / "showdown_sweep.json"
CHAPTERS_NPZ = CACHE_DIR / "showdown_chapters.npz"
EDGES_NPZ = CACHE_DIR / "showdown_edges.npz"
TOC_PKL = CACHE_DIR / "showdown_toc.pkl"


def log(msg: str) -> None:
    print(f"[{time.monotonic() - _T0:7.1f}s] {msg}", flush=True)


def bank(sub: str, payload: dict) -> None:
    """Merge `payload` under results.json:task_showdown_p0.<sub>."""
    r = load_banked()
    cur = r.get("task_showdown_p0", {})
    cur[sub] = _py(payload)
    r["task_showdown_p0"] = cur
    RESULTS_PATH.write_text(json.dumps(r, indent=1))
    log(f"banked results.json:task_showdown_p0.{sub}")


# ----------------------------------------------------------------- phase: refs
def _refs_page_path(filt: str, page_idx: int) -> Path:
    h = hashlib.sha1(f"{filt}|{REFS_SELECT}".encode()).hexdigest()[:10]
    return REFS_CACHE / f"works_{h}" / f"page_{page_idx:05d}.json.gz"


def fetch_refs(filt: str, max_pages: int | None = None) -> int:
    """Cursor-page the refs select of the pre-committed slice (resumable)."""
    cursor, page_idx, n_cached = "*", 0, 0
    while True:
        p = _refs_page_path(filt, page_idx)
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
            log(f"refs slice fully cached: {page_idx} pages, {n_cached:,} works")
            return n_cached
    if page_idx:
        log(f"resuming refs pull from page {page_idx} ({n_cached:,} works)")
    session = _session()
    while cursor is not None:
        if max_pages is not None and page_idx >= max_pages:
            log(f"max_pages={max_pages} reached; stopping (partial)")
            break
        data = _get_json(session, f"{API_BASE}/works", {
            "filter": filt, "select": REFS_SELECT,
            "per-page": str(PER_PAGE), "cursor": cursor,
        }, log=log)
        p = _refs_page_path(filt, page_idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        if page_idx == 0:
            log(f"refs meta.count = {data['meta']['count']:,}")
        n_cached += len(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        page_idx += 1
        if page_idx % 50 == 0:
            log(f"  {n_cached:,} works ({page_idx} pages)")
        time.sleep(0.05)
    log(f"refs slice acquired: {page_idx} pages, {n_cached:,} works")
    return n_cached


def iter_refs(filt: str):
    page_idx = 0
    while True:
        p = _refs_page_path(filt, page_idx)
        if not p.exists():
            return
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        yield from data.get("results", [])
        page_idx += 1


def phase_refs(args) -> None:
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)
    n = fetch_refs(filt, max_pages=args.max_pages)
    log(f"refs pull done: {n:,} works with {REFS_SELECT}")


# ----------------------------------------------------------------- phase: embed
def phase_embed(args) -> None:
    banked = load_banked()["task_t3"]
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)

    fg_units = load_book_fg()
    fg_texts = [u.text for u in fg_units]
    log(f"fg: {len(fg_units)} units (banked {banked['fg']['n_units']})")
    assert len(fg_units) == banked["fg"]["n_units"]

    bg = assemble_bg(filt, log=log)
    assert bg["counts"]["eligible"] == banked["bg"]["counts"]["eligible"], (
        bg["counts"], banked["bg"]["counts"])

    emb = embed_joint(fg_texts, bg["texts"], seed=SEED, lsa_dim=LSA_DIM)
    nnz = np.asarray((emb["X"] > 0).sum(axis=1)).ravel()
    n_fg = len(fg_texts)
    keep_bg = np.where(nnz[n_fg:] >= 5)[0]
    Zf = emb["Z"][:n_fg]
    Zb = emb["Z"][n_fg:][keep_bg]
    for k in ("wid", "title", "year", "topic", "cited_by"):
        bg[k] = bg[k][keep_bg]
    Kb = emb["K"][n_fg:][keep_bg].tocsr()
    vocab = emb["vocab"]
    assert Zb.shape[0] == banked["bg"]["n_used"], (Zb.shape, banked["bg"]["n_used"])
    log(f"embed: LSA-{LSA_DIM} EVR {emb['evr']:.3f} (banked {banked['embed']['evr']}); "
        f"bg used {Zb.shape[0]:,}")

    # t3 scope + 12 banked gap regions, recomputed and asserted against the bank
    r_F = knn_radii(Zb, Zf, K_FG)
    thr = float(np.quantile(r_F, SCOPE_QUANTILE))
    scoped_global = np.where(r_F <= thr)[0]
    assert round(thr, 4) == banked["scope"]["thr"], (thr, banked["scope"]["thr"])
    assert scoped_global.size == banked["scope"]["n_scoped"]
    Zs = Zb[scoped_global]
    stat = full_statistic(Zf, Zs, k_fg=K_FG, k_bg=K_BG)
    ones = np.ones(scoped_global.size, dtype=bool)
    comps = level_set_components(stat["full"], Zs, ones, top_frac=TOP_FRAC,
                                 k_graph=K_GRAPH, min_size=MIN_COMP_SIZE)
    sizes = [int(c["size"]) for c in comps[:N_REGIONS]]
    banked_sizes = [c["size"] for c in banked["levelset"]["components"]]
    assert sizes == banked_sizes, (sizes, banked_sizes)
    log(f"t3 anchors reproduced: thr {thr:.4f}, scoped {scoped_global.size:,}, "
        f"12 region sizes {sizes}")
    regions = [scoped_global[c["members"]] for c in comps[:N_REGIONS]]

    # ds2 chapter assignment of every bg doc = chapter of its nearest fg unit
    ch_names = sorted({u.chapter for u in fg_units}, key=lambda c: int(c[2:]))
    ch_index = {c: i for i, c in enumerate(ch_names)}
    fg_chap = np.array([ch_index[u.chapter] for u in fg_units], dtype=np.int16)
    j_near = np.empty(Zb.shape[0], dtype=np.int64)
    for s in range(0, Zb.shape[0], 8192):
        blk = Zb[s:s + 8192]
        d2 = (np.einsum("ij,ij->i", blk, blk)[:, None]
              - 2.0 * (blk @ Zf.T) + np.einsum("ij,ij->i", Zf, Zf)[None, :])
        j_near[s:s + 8192] = np.argmin(d2, axis=1)
    chap_ds2 = fg_chap[j_near]
    log(f"ds2 doc->chapter map: {np.bincount(chap_ds2, minlength=DS2_N_CHAPTERS).tolist()}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_NPZ, Zf=Zf, Zb=Zb, year=bg["year"],
                        chap_ds2=chap_ds2.astype(np.int16))
    sp.save_npz(K_NPZ, Kb)
    np.savez_compressed(REGIONS_NPZ,
                        **{f"region_{i:02d}": m for i, m in enumerate(regions)})
    with META_PKL.open("wb") as f:
        pickle.dump({
            "wid": [str(w) for w in bg["wid"]],
            "title": [str(t) for t in bg["title"]],
            "topic": [str(t) for t in bg["topic"]],
            "vocab": [str(v) for v in vocab],
            "fg_labels": [u.label for u in fg_units],
            "fg_chapters": [u.chapter for u in fg_units],
            "thr": thr, "evr": float(emb["evr"]),
            "filter": filt,
        }, f)
    log(f"cached embed/meta/K/regions under {CACHE_DIR}")
    bank("embed_anchor", {
        "n_fg_units": len(fg_units), "n_bg_used": int(Zb.shape[0]),
        "evr": round(float(emb["evr"]), 3), "scope_thr": round(thr, 4),
        "n_scoped": int(scoped_global.size), "region_sizes": sizes,
        "anchors_reproduced_banked_task_t3": True,
        "joint_fit_caveat": "fg = 126 of 122,486 TF-IDF docs (0.1%); clustering on bg rows only",
    })


def _load_embed():
    z = np.load(EMBED_NPZ)
    with META_PKL.open("rb") as f:
        meta = pickle.load(f)
    return z["Zf"], z["Zb"], z["year"], z["chap_ds2"], meta


# ----------------------------------------------------------------- phase: sweep
def _kmeans_sweep(Z: np.ndarray, k_range, seed: int = SEED,
                  sil_sample: int = SIL_SAMPLE) -> tuple[dict, dict]:
    """KMeans + silhouette over a k range; returns ({k: {silhouette, shares}},
    {k: labels}). Pre-committed: n_init=10, random_state=seed, euclidean
    silhouette on a seeded subsample of min(n, sil_sample)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows: dict[int, dict] = {}
    labelings: dict[int, np.ndarray] = {}
    for k in k_range:
        if k >= Z.shape[0]:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sil = float(silhouette_score(
            Z, km.labels_, metric="euclidean",
            sample_size=min(Z.shape[0], sil_sample), random_state=seed))
        shares = (np.bincount(km.labels_, minlength=k) / Z.shape[0]).tolist()
        rows[k] = {"silhouette": round(sil, 5),
                   "shares": [round(s, 5) for s in shares],
                   "inertia": round(float(km.inertia_), 2)}
        labelings[k] = km.labels_.astype(np.int16)
        log(f"  k={k}: sil {sil:.5f}, share min/max "
            f"{min(shares):.4f}/{max(shares):.4f}")
    return rows, labelings


def phase_sweep(args) -> None:
    _, Zb, _, _, _ = _load_embed()
    log(f"chapter sweep on {Zb.shape[0]:,} x {Zb.shape[1]} bg vectors, "
        f"k in {list(CHAPTER_K_RANGE)}")
    rows, labelings = _kmeans_sweep(Zb, CHAPTER_K_RANGE)
    k, why = select_k(rows)
    log(f"PRE-COMMITTED RULE PICK: k={k} ({why}), silhouette "
        f"{rows[k]['silhouette']}")
    SWEEP_JSON.write_text(json.dumps(
        {"rule": "max mean silhouette s.t. no cluster <3% or >25% of mass; "
                 "fallback fewest violations then higher silhouette",
         "k_range": list(CHAPTER_K_RANGE), "sil_sample": SIL_SAMPLE,
         "seed": SEED, "rows": _py(rows), "chosen_k": k, "why": why}, indent=1))
    np.savez_compressed(CHAPTERS_NPZ, labels=labelings[k], chosen_k=np.int64(k))
    bank("chapter_sweep", {"rows": rows, "chosen_k": k, "why": why,
                           "sil_sample": SIL_SAMPLE, "k_range": list(CHAPTER_K_RANGE)})


# ----------------------------------------------------------------- labels
def cluster_terms(K: sp.csr_matrix, members: np.ndarray, vocab: list[str],
                  n_top: int = 12, alpha: float = 1.0, min_df: int = 3,
                  register_share: float = REGISTER_DF_SHARE) -> tuple[list[str], list[str]]:
    """Top distinctive terms of a cluster (log-odds member df vs rest), with the
    t3 register split ported cluster-wise: a term with df-share >= register_share
    BOTH inside the cluster and in the rest of the corpus is REGISTER (genre
    vocabulary uniform across field areas), excluded from topical labels.
    Returns (topical_terms, register_terms_excluded). Same log-odds form as the
    banked region_terms; sparse-friendly."""
    members = np.asarray(members, dtype=np.int64)
    n, _ = K.shape
    n_in = float(members.size)
    n_out = float(n - members.size)
    inside = np.asarray(K[members].sum(axis=0), dtype=np.float64).ravel()
    total = np.asarray(K.sum(axis=0), dtype=np.float64).ravel()
    outside = total - inside
    register = (inside / max(n_in, 1) >= register_share) & \
               (outside / max(n_out, 1) >= register_share)
    score = (np.log((inside + alpha) / (n_in - inside + alpha))
             - np.log((outside + alpha) / (n_out - outside + alpha)))
    score[inside < min_df] = -np.inf
    order = np.argsort(score)[::-1]
    topical, reg_hits = [], []
    for j in order:
        if not np.isfinite(score[j]):
            break
        t = vocab[j]
        if t in JUNK_TERMS or t in EXTRA_JUNK:
            continue
        if register[j]:
            if len(reg_hits) < n_top:
                reg_hits.append(t)
            continue
        topical.append(t)
        if len(topical) >= n_top:
            break
    return topical, reg_hits


def _title_case(terms: list[str], n: int) -> str:
    return " · ".join(t.capitalize() for t in terms[:n])


# ----------------------------------------------------------------- phase: build
def _build_edges(wid_to_row: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """In-slice citation edge list (src_row cites dst_row) from the refs pull."""
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)
    src, dst = [], []
    n_refs_works = n_matched = n_refs_total = 0
    for w in iter_refs(filt):
        n_refs_works += 1
        row = wid_to_row.get(short_id(w.get("id")) or "")
        refs = w.get("referenced_works") or []
        n_refs_total += len(refs)
        if row is None:
            continue
        n_matched += 1
        for r in refs:
            rr = wid_to_row.get(short_id(r) or "")
            if rr is not None and rr != row:
                src.append(row)
                dst.append(rr)
    stats = {"refs_pull_works": n_refs_works, "matched_to_slice": n_matched,
             "total_reference_entries": n_refs_total, "in_slice_edges": len(src)}
    log(f"edges: {stats}")
    return (np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32), stats)


def phase_build(args) -> None:
    Zf, Zb, year, chap_ds2, meta = _load_embed()
    K = sp.load_npz(K_NPZ).tocsr()
    vocab = meta["vocab"]
    titles = meta["title"]
    ch = np.load(CHAPTERS_NPZ)
    labels, k = ch["labels"].astype(np.int64), int(ch["chosen_k"])
    sweep = json.loads(SWEEP_JSON.read_text())
    log(f"build: chosen k={k} (rule: {sweep['why']})")

    # ---- citation edges (cached after first build)
    if EDGES_NPZ.exists():
        e = np.load(EDGES_NPZ)
        src, dst, estats = e["src"], e["dst"], json.loads(str(e["stats"]))
        log(f"edges loaded: {estats}")
    else:
        wid_to_row = {w: i for i, w in enumerate(meta["wid"])}
        src, dst, estats = _build_edges(wid_to_row)
        np.savez_compressed(EDGES_NPZ, src=src, dst=dst, stats=json.dumps(estats))

    # ---- chapter ordering: min forward citation flow, median-year tie-break
    F = citation_flow(labels[src], labels[dst], k)
    med_year = np.array([float(np.median(year[labels == c])) for c in range(k)])
    chap_order, _ = order_min_forward_flow(F, med_year)
    base = forward_violations(F, list(range(k)))
    opt = forward_violations(F, chap_order)
    log(f"chapter order {chap_order}: forward flow {opt['violations']:.0f} of "
        f"{opt['cross_edges']:.0f} cross ({opt['share']:.3f}); arbitrary-id order "
        f"would be {base['share']:.3f}")

    # ---- sections per chapter: sub-cluster sweep (same pre-committed rule)
    chapters = []
    for c in chap_order:
        m = np.where(labels == c)[0]
        rows, labelings = _kmeans_sweep(Zb[m], SECTION_K_RANGE)
        ks, why = select_k(rows)
        sub = labelings[ks].astype(np.int64)
        # within-chapter citation flow over subclusters
        in_c = (labels[src] == c) & (labels[dst] == c)
        pos_in_m = np.full(Zb.shape[0], -1, dtype=np.int64)
        pos_in_m[m] = np.arange(m.size)
        F_sub = citation_flow(sub[pos_in_m[src[in_c]]], sub[pos_in_m[dst[in_c]]], ks)
        med_sub = np.array([float(np.median(year[m[sub == s]])) for s in range(ks)])
        sec_order, _ = order_min_forward_flow(F_sub, med_sub)
        sec_viol = forward_violations(F_sub, sec_order)
        c_terms, c_register = cluster_terms(K, m, vocab, n_top=N_CHAPTER_TERMS)
        sections = []
        for s in sec_order:
            sm = m[sub == s]
            s_terms, _ = cluster_terms(K, sm, vocab, n_top=N_SECTION_TERMS)
            cen = Zb[sm].mean(axis=0)
            cen /= max(float(np.linalg.norm(cen)), 1e-12)
            d = np.linalg.norm(Zb[sm] - cen[None, :], axis=1)
            ex, seen_norm = [], set()
            for i in sm[np.argsort(d)]:       # nearest-first; skip near-dup titles
                norm = "".join(ch for ch in titles[i].lower() if ch.isalnum())
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
            "share": round(m.size / Zb.shape[0], 4),
            "terms": c_terms, "register_excluded": c_register,
            "median_year": float(np.median(year[m])),
            "section_k": ks, "section_rule": why,
            "section_sweep": rows,
            "section_forward_share": round(sec_viol["share"], 4),
            "sections": sections,
        })
        log(f"chapter cluster {c}: n={m.size:,}, sections k={ks} ({why}), "
            f"terms {c_terms[:5]}")

    toc = {
        "k": k, "chap_order": [int(c) for c in chap_order],
        "forward_flow": opt, "F": F, "chapters": chapters,
        "edge_stats": estats, "labels_key": str(CHAPTERS_NPZ),
    }
    with TOC_PKL.open("wb") as f:
        pickle.dump(toc, f)

    _write_toc_md(toc, meta)
    log(f"wrote {SHOW_DIR / 'toc_algo_datascience.md'}")
    bank("toc_build", {
        "chosen_k": k, "chapter_order_cluster_ids": [int(c) for c in chap_order],
        "chapter_forward_violations": opt,
        "edge_stats": estats,
        "chapters": [{kk: v for kk, v in c.items() if kk != "sections"}
                     | {"n_sections": len(c["sections"]),
                        "section_titles": [_title_case(s["terms"], 3)
                                           for s in c["sections"]]}
                     for c in chapters],
    })


def _params_hash() -> str:
    cfg = {
        "seed": SEED, "lsa_dim": LSA_DIM, "chapter_k_range": list(CHAPTER_K_RANGE),
        "section_k_range": list(SECTION_K_RANGE), "sil_sample": SIL_SAMPLE,
        "rule": "max-sil s.t. shares in [0.03,0.25]; fallback fewest violations",
        "ordering": "exact min forward citation flow (bitmask DP), "
                    "median-year tie-break",
        "register_df_share": REGISTER_DF_SHARE, "kmeans": "n_init=10,random_state=42",
        "corpus": works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO),
    }
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _write_toc_md(toc: dict, meta: dict) -> None:
    SHOW_DIR.mkdir(parents=True, exist_ok=True)
    k = toc["k"]
    F = toc["F"]
    order = toc["chap_order"]
    pos = {c: i for i, c in enumerate(order)}
    L = [
        "# Table of Contents: Foundations of Data Science (corpus-derived)",
        "",
        f"*Generated algorithmically from the structure of {sum(c['n'] for c in toc['chapters']):,} "
        f"field works (no LLM): chapters = KMeans-{k} clusters of the corpus in "
        "LSA-64, sections = subclusters, labels = register-filtered distinctive "
        "terms, order = minimum forward citation flow (prerequisites first).*",
        "",
    ]
    for i, ch in enumerate(toc["chapters"], 1):
        c = ch["cluster_id"]
        title = _title_case(ch["terms"], 4)
        L.append(f"## Chapter {i}: {title}")
        L.append(f"*Intent: cover the {', '.join(ch['terms'][:6])} literature — "
                 f"{ch['n']:,} works ({ch['share']:.1%} of the field), "
                 f"median year {ch['median_year']:.0f}.*")
        for j, s in enumerate(ch["sections"], 1):
            ex = "; ".join(f"“{t}” ({y})" for t, y in
                           zip(s["exemplar_titles"], s["exemplar_years"]))
            L.append(f"- **{i}.{j} {_title_case(s['terms'], 3)}**: "
                     f"the {', '.join(s['terms'][:5])} literature "
                     f"(n={s['n']:,}; exemplars: {ex}).")
        # prerequisite note: earlier chapters receiving >= DEP_SHARE of out-flow
        out = np.array([F[c, b] for b in range(k)])
        out[c] = 0.0
        tot = out.sum()
        deps = []
        if tot > 0:
            for b_pos, b in enumerate(order[:pos[c]]):
                share = out[b] / tot
                if share >= DEP_SHARE:
                    deps.append((b_pos + 1, share))
        deps.sort(key=lambda d: -d[1])
        if deps:
            note = ", ".join(f"Chapter {p} ({s:.0%} of out-citations)"
                             for p, s in deps[:3])
            L.append(f"**Dependencies:** {note}.")
        else:
            L.append("**Dependencies:** none pronounced "
                     f"(no earlier chapter receives ≥{DEP_SHARE:.0%} of "
                     "this chapter's cross-chapter citations).")
        L.append("")
    L += [
        "---",
        f"*Provenance: curriculum_toc.py params {_params_hash()} | corpus: 19-topic "
        f"OpenAlex slice (t3, {PUB_FROM[:4]}–{PUB_TO[:4]}), cache "
        f"cache/oa/t3_ds_book + ds_showdown_refs | seed {SEED} | LSA-64 EVR "
        f"{meta['evr']:.3f} | chapter rule: max silhouette s.t. cluster shares in "
        f"[3%, 25%] over k∈{{8..14}} | ordering: exact min forward citation flow, "
        f"median-year tie-break | generated {date.today().isoformat()} | NO LLM.*",
    ]
    (SHOW_DIR / "toc_algo_datascience.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------- phase: metrics
def phase_metrics(args) -> None:
    Zf, Zb, year, chap_ds2, meta = _load_embed()
    thr = float(meta["thr"])
    with TOC_PKL.open("rb") as f:
        toc = pickle.load(f)
    regions = []
    rz = np.load(REGIONS_NPZ)
    for i in range(N_REGIONS):
        regions.append(rz[f"region_{i:02d}"])
    e = np.load(EDGES_NPZ)
    src, dst = e["src"], e["dst"]

    # representatives: algo = its section centroids; ds2 = its 126 unit vectors
    reps_algo = np.vstack([s["centroid"] for ch in toc["chapters"]
                           for s in ch["sections"]])
    reps_ds2 = Zf
    log(f"reps: algo {reps_algo.shape[0]} section centroids, ds2 {reps_ds2.shape[0]} units")

    cov_algo = coverage_at_radius(Zb, reps_algo, thr)
    cov_ds2 = coverage_at_radius(Zb, reps_ds2, thr)
    # first (degenerate) closure form — kept on record, superseded (worklog)
    clo_algo_d = region_closure(Zb, regions, reps_algo, thr)
    clo_ds2_d = region_closure(Zb, regions, reps_ds2, thr)

    # REVISED closure (pre-committed in the worklog before computing): re-run
    # the t3 full statistic against each ToC's reps over the t3 scoped pool;
    # closed iff the region's mean score falls below the t3 gap bar.
    r_F = knn_radii(Zb, Zf, K_FG)
    scoped_global = np.where(r_F <= thr)[0]
    Zs = Zb[scoped_global]
    pos = {int(g): i for i, g in enumerate(scoped_global)}
    regions_local = [np.asarray([pos[int(g)] for g in m], dtype=np.int64)
                     for m in regions]
    full_ds2 = full_statistic(Zf, Zs, k_fg=K_FG, k_bg=K_BG)["full"]
    bar = gap_bar(full_ds2, top_frac=TOP_FRAC, min_size=MIN_COMP_SIZE)
    full_algo = full_statistic(reps_algo, Zs, k_fg=K_FG, k_bg=K_BG)["full"]
    clo_algo = region_closure_fullstat(full_algo, regions_local, bar)
    clo_ds2 = region_closure_fullstat(full_ds2, regions_local, bar)
    log(f"gap bar {bar:.4f}; mean region full: ds2 "
        f"{[round(float(np.mean(full_ds2[m])), 2) for m in regions_local]}, algo "
        f"{[round(float(np.mean(full_algo[m])), 2) for m in regions_local]}")

    # prereq violations: algo under its built order; ds2 under ch1..ch9
    k = toc["k"]
    F_algo = toc["F"]
    v_algo = forward_violations(F_algo, toc["chap_order"])
    F_ds2 = citation_flow(chap_ds2[src], chap_ds2[dst], DS2_N_CHAPTERS)
    v_ds2 = forward_violations(F_ds2, list(range(DS2_N_CHAPTERS)))
    # reference: how good could ds2's own partition be ordered? (context number)
    med_ds2 = np.array([float(np.median(year[chap_ds2 == c])) if (chap_ds2 == c).any()
                        else 0.0 for c in range(DS2_N_CHAPTERS)])
    ds2_opt_order, ds2_opt = order_min_forward_flow(F_ds2, med_ds2)

    payload = {
        "radius": round(thr, 4),
        "rep_counts": {"toc_algo": int(reps_algo.shape[0]),
                       "ds2": int(reps_ds2.shape[0])},
        "field_mass_coverage": {"toc_algo": round(cov_algo, 4),
                                "ds2": round(cov_ds2, 4)},
        "gap_region_closure": {
            "rule": "REVISED (worklog): re-run full statistic vs ToC reps; "
                    "closed iff mean region score < t3 gap bar",
            "gap_bar": round(bar, 4),
            "toc_algo": {"closed": int(sum(clo_algo)), "of": N_REGIONS,
                         "per_region": clo_algo,
                         "mean_full": [round(float(np.mean(full_algo[m])), 3)
                                       for m in regions_local]},
            "ds2": {"closed": int(sum(clo_ds2)), "of": N_REGIONS,
                    "per_region": clo_ds2,
                    "mean_full": [round(float(np.mean(full_ds2[m])), 3)
                                  for m in regions_local]},
            "degenerate_first_form_on_record": {
                "rule": "median member dist-to-nearest-rep <= scope radius",
                "toc_algo_closed": int(sum(clo_algo_d)),
                "ds2_closed": int(sum(clo_ds2_d)),
                "verdict": "non-discriminating (12/12 both): the regions are "
                           "in-scope by construction",
            },
        },
        "prereq_violations": {
            "toc_algo": v_algo, "ds2": v_ds2,
            "ds2_best_possible_share": round(
                ds2_opt / v_ds2["cross_edges"], 4) if v_ds2["cross_edges"] else 0.0,
            "ds2_best_possible_order": [int(c) + 1 for c in ds2_opt_order],
        },
    }
    log(json.dumps(_py(payload), indent=1))
    _write_side_by_side(payload, toc, meta)
    log(f"wrote {SHOW_DIR / 'side_by_side_ds2.md'}")
    bank("metrics", payload)


def _write_side_by_side(p: dict, toc: dict, meta: dict) -> None:
    SHOW_DIR.mkdir(parents=True, exist_ok=True)
    ds2_text = DS2_TOC_PATH.read_text(encoding="utf-8").strip()
    ca, cd = p["gap_region_closure"]["toc_algo"], p["gap_region_closure"]["ds2"]
    va, vd = p["prereq_violations"]["toc_algo"], p["prereq_violations"]["ds2"]
    L = [
        "# Curriculum Showdown — toc-algo vs ds2 agent ToC (DS tuning topic)",
        "",
        "Objective metric suite (pre-committed definitions in the worklog; functions in",
        "`toc_metrics.py`, toy-tested). Representatives: toc-algo = its "
        f"{p['rep_counts']['toc_algo']} section centroids; ds2 = its "
        f"{p['rep_counts']['ds2']} section-level fg units (its real section texts).",
        f"Radius = the banked t3 q25 scope threshold ({p['radius']}); corpus = the",
        f"cleaned t3 slice (122,360 works); citation edges = "
        f"{toc['edge_stats']['in_slice_edges']:,} in-slice reference pairs.",
        "",
        "| metric | toc-algo | ds2 agent ToC |",
        "|---|---|---|",
        f"| field-mass coverage @ r={p['radius']} (k=1) | "
        f"**{p['field_mass_coverage']['toc_algo']:.3f}** | "
        f"{p['field_mass_coverage']['ds2']:.3f} |",
        f"| gap-region closure (of the 12 task_t3 regions, re-run-audit rule) | "
        f"**{ca['closed']}/12** | {cd['closed']}/12 |",
        f"| prereq violations (forward citation mass) | "
        f"{va['violations']:,.0f} of {va['cross_edges']:,.0f} cross "
        f"({va['share']:.3f}) | {vd['violations']:,.0f} of "
        f"{vd['cross_edges']:,.0f} cross ({vd['share']:.3f}) |",
        "",
        "Notes:",
        f"- *Closure rule (revised, worklog on record)*: a region is closed iff its "
        f"members' mean full statistic, re-computed against the ToC's representatives, "
        f"drops below the t3 gap bar ({p['gap_region_closure']['gap_bar']}) — i.e. the "
        "region would no longer be selected as a gap in a re-run audit. ds2 closes 0 "
        "BY CONSTRUCTION (its audit defined the regions) — it is the incumbent "
        "baseline. The first-form median-distance rule was computed and found "
        "degenerate (12/12 both, on record in results.json).",
        "- Per-region closure, task_t3 rank order 1–12: toc-algo "
        f"{''.join('CT'[c] for c in ca['per_region'])} (C=open, T=closed); mean "
        f"region full-score under toc-algo {ca['mean_full']} vs under ds2 "
        f"{cd['mean_full']}.",
        "- *Coverage*: k=1 (nearest representative) at the t3 scope radius. The "
        "algorithmic ToC tiles the corpus by construction (its sections are corpus "
        "clusters), so high coverage is expected — the number quantifies how much "
        "field mass an agent ToC leaves outside its topical reach, not a win per se.",
        f"- *Ordering*: ds2's own 9-part partition re-ordered optimally would reach "
        f"forward share {p['prereq_violations']['ds2_best_possible_share']} "
        f"(order {p['prereq_violations']['ds2_best_possible_order']}); the agent's "
        "published order is compared as published. The algorithmic order is optimal "
        "for its own partition by construction (exact DP).",
        "",
        "---",
        "",
        "## The algorithmic ToC",
        "",
        (SHOW_DIR / "toc_algo_datascience.md").read_text(encoding="utf-8").strip(),
        "",
        "---",
        "",
        "## The ds2 agent ToC (verbatim, read-only source)",
        "",
        f"Source: `{DS2_TOC_PATH}`",
        "",
        ds2_text,
        "",
        "---",
        f"*Provenance: params {_params_hash()}, generated {date.today().isoformat()}; "
        "metric definitions pre-committed in docs/gap-detection-worklog.md "
        "§Curriculum Showdown; banked under results.json:task_showdown_p0.metrics.*",
    ]
    (SHOW_DIR / "side_by_side_ds2.md").write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------- phase: causal
def phase_causal(args) -> None:
    filt = works_filter(COMMITTED_TOPICS, PUB_FROM, PUB_TO)
    bg = assemble_bg(filt, log=log)
    m = np.asarray([t == "T10845" for t in bg["topic"]], dtype=bool)
    texts = [t for t, keep in zip(bg["texts"], m) if keep]
    log(f"T10845 causal sub-slice: {int(m.sum()):,} cleaned works "
        f"(of {bg['counts']['eligible']:,} slice-cleaned)")
    emb = embed_joint([], texts, seed=SEED, lsa_dim=LSA_DIM)
    nnz = np.asarray((emb["X"] > 0).sum(axis=1)).ravel()
    keep = np.where(nnz >= 5)[0]
    Z = emb["Z"][keep]
    log(f"standalone embed: vocab {len(emb['tfidf_vocab']):,}, LSA-{LSA_DIM} "
        f"EVR {emb['evr']:.3f}, nnz>=5 keeps {keep.size:,}/{len(texts):,}")
    rows, _ = _kmeans_sweep(Z, SMALL_K_RANGE)
    k, why = select_k(rows)
    log(f"cluster-ability: pre-committed rule picks k={k} ({why}), "
        f"sil {rows[k]['silhouette']}")
    bank("causal_subslice", {
        "topic": "T10845", "name": TOPIC_NAMES["T10845"],
        "raw_in_slice": int(m.sum()),
        "embedded_after_nnz_trim": int(keep.size),
        "evr": round(float(emb["evr"]), 3),
        "k_range": list(SMALL_K_RANGE), "sweep": rows,
        "would_pick_k": k, "why": why,
        "note": "counts + cluster-ability only; ToC deferred to Phase 1 (frozen params)",
    })


# ----------------------------------------------------------------- phase: postcutoff
POSTCUTOFF_QUERIES = (
    # the brief's four candidate searches…
    "large language model agents", "LLM agents",
    "retrieval augmented generation", "mechanistic interpretability",
    "diffusion models", "large language models",
    # …plus the OpenAlex taxonomy's own names for where that literature files
    # (the fixed /topics taxonomy predates these fields — no literal hits above)
    "language model", "natural language processing", "text generation",
    "topic modeling", "interpretability", "explainable artificial intelligence",
    "generative adversarial networks", "image synthesis", "transformer",
    "question answering", "speech recognition", "multimodal machine learning",
    "neural machine translation", "reinforcement learning",
    # due-diligence round (worklog): agent/codegen/text-mining names
    "multi-agent systems", "chatbot dialogue", "code generation",
    "prompt engineering", "intelligent agents", "text mining",
)
POSTCUTOFF_FROM = "2024-07-01"


def _cached_get(url: str, params: dict, cache_name: str) -> dict:
    TOPICS_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = TOPICS_CACHE / f"{cache_name}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())
    data = _get_json(_session(), url, params, log=log)
    cpath.write_text(json.dumps(data))
    return data


def phase_postcutoff(args) -> None:
    seen: dict[str, dict] = {}
    for q in POSTCUTOFF_QUERIES:
        h = hashlib.sha1(q.encode()).hexdigest()[:10]
        data = _cached_get(f"{API_BASE}/topics",
                           {"search": q, "per-page": "25"}, f"topics_{h}")
        for t in data.get("results", []):
            tid = short_id(t.get("id"))
            if tid and tid not in seen:
                seen[tid] = {
                    "id": tid, "display_name": t.get("display_name"),
                    "works_count": t.get("works_count"),
                    "subfield": (t.get("subfield") or {}).get("display_name"),
                    "query": q,
                }
    cands = [t for t in sorted(seen.values(), key=lambda t: -(t["works_count"] or 0))
             if (t["works_count"] or 0) >= 10_000]
    log(f"{len(cands)} candidate topics (>=10K total works) from "
        f"{len(POSTCUTOFF_QUERIES)} queries")
    rows = []
    for t in cands:
        filt = (f"primary_topic.id:{t['id']},from_publication_date:{POSTCUTOFF_FROM},"
                "has_abstract:true,language:en")
        h = hashlib.sha1(filt.encode()).hexdigest()[:10]
        data = _cached_get(f"{API_BASE}/works",
                           {"filter": filt, "per-page": "1", "select": "id"},
                           f"count_{h}")
        n = int(data["meta"]["count"])
        rows.append({**t, "post_cutoff_count": n})
        print(f"{t['id']:>7} | {n:>7,} since {POSTCUTOFF_FROM} | "
              f"{t['works_count']:>8,} total | {t['subfield']} | "
              f"{t['display_name']} [q: {t['query']}]")
    rows.sort(key=lambda r: -r["post_cutoff_count"])
    bank("postcutoff_candidates", {
        "from_publication_date": POSTCUTOFF_FROM,
        "filter_form": "primary_topic.id:X,from_publication_date:...,"
                       "has_abstract:true,language:en",
        "rows": rows,
        "note": "pick committed in the worklog on these counts; corpus pull is Phase 1",
    })


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    r = sub.add_parser("refs")
    r.add_argument("--max-pages", type=int, default=None)
    r.set_defaults(fn=phase_refs)
    for name, fn in (("embed", phase_embed), ("sweep", phase_sweep),
                     ("build", phase_build), ("metrics", phase_metrics),
                     ("causal", phase_causal), ("postcutoff", phase_postcutoff)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
