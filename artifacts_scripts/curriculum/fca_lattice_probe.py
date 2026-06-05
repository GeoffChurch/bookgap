"""EXPLORATORY PROBE (throwaway) — FCA concept lattice over a concept-seeded corpus.

Tests the backlog idea (docs/backlog/curriculum-keyword-concept-seeding.md, "FCA antichain"): build a
formal context over the concept works, mine closed concepts, score each by intensional STABILITY
(the canonical FCA robustness measure) and EMBEDDING COHESION (the co-clustering guardrail: drop
concepts that are lexically coherent but embedding-scattered = paratext), then select a size-N
max-stability ANTICHAIN (mutually-incomparable, non-redundant, covering concepts) as the chapter set.

Compares the antichain to the coherence-merge's clean 11-topic neuromorphic structure WITHOUT a tau,
WITHOUT breaking ordering, and WITHOUT the jurafsky paratext over-merge.

Input variants (--context):
  terms   : works x keyword-terms                 (pure keywords)
  refs    : works x cited-references (coupling)    (pure citations; coverage-capped)
  concat  : works x {terms U refs} joint vocab     (the concat_vocab unifier — DEFAULT)

Attributes are pruned to support >= --min-supp among the concept works (keeps the context small and the
closures clean). Candidate concepts are the ATTRIBUTE concepts mu(a) (closed by construction).

Example:
  .conda/bin/python -m artifacts_scripts.curriculum.fca_lattice_probe \
      --concept-gap artifacts/book_gaps/neuromorphic_concept/concept_gap.json --field 17 \
      --n 11 --context concat
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from artifacts_scripts.curriculum.config import CurriculumConfig
from artifacts_scripts.curriculum.field_term_index import build_field_term_index
from bookgap.util import LOGGER, setup_logging

_RNG = np.random.default_rng(0)


def _attribute_concepts(B_csr, B_csc, min_supp, lo, hi):
    """Attribute concepts mu(a) for each column a: extent = supp(a); intent = closure (cols all-1 over
    extent). Closed by construction. Dedup by extent. Keep extents in [lo, hi] (chapter-grain)."""
    supp = np.asarray(B_csc.sum(0)).ravel()
    seen = set()
    concepts = []  # (extent ndarray[int], intent ndarray[bool])
    for a in np.where(supp >= min_supp)[0]:
        ext = B_csc.getcol(a).indices
        if not (lo <= len(ext) <= hi):
            continue
        key = frozenset(ext.tolist())
        if key in seen:
            continue
        seen.add(key)
        intent = np.asarray(B_csr[ext].sum(0)).ravel() == len(ext)
        concepts.append((ext, intent))
    return concepts


def _stability(B_csr, ext, intent, n_samp=120):
    """Intensional stability estimate: P(random nonempty subset C of extent has C' == intent)."""
    if len(ext) < 2:
        return 0.0
    sub = B_csr[ext].toarray().astype(bool)                # (|ext| x n_att)
    hits = 0
    for _ in range(n_samp):
        mask = _RNG.random(len(ext)) < 0.5
        if not mask.any():
            mask[_RNG.integers(len(ext))] = True
        if np.array_equal(sub[mask].all(0), intent):       # intent of the subset == concept intent
            hits += 1
    return hits / n_samp


def _closed_intents(B_csr, B_csc, min_supp):
    """Closed concept intents (itemsets, |intent|>=2) anchored by frequent attributes. (intent, support)."""
    supp = np.asarray(B_csc.sum(0)).ravel()
    seen, out = set(), []
    for a in np.where(supp >= min_supp)[0]:
        ext = B_csc.getcol(a).indices
        intent = np.where(np.asarray(B_csr[ext].sum(0)).ravel() == len(ext))[0]
        if len(intent) < 2:
            continue
        key = frozenset(int(i) for i in intent)
        if key not in seen:
            seen.add(key); out.append((key, len(ext)))
    return out


def _cover_L(transactions, patterns, Lst):
    """Krimp two-part code length for code table = singletons + `patterns`. Returns (L, usage)."""
    order = sorted(patterns, key=lambda p: (-len(p[0]), -p[1]))    # cover order: long, then frequent
    usage = defaultdict(int)
    for t in transactions:
        rem = set(t)
        for P, _s in order:
            if P <= rem:
                usage[P] += 1; rem -= P
        for a in rem:
            usage[a] += 1
    tot = sum(usage.values()) or 1
    LD = sum(u * -np.log2(u / tot) for u in usage.values())        # L(D | CT)
    LCT = sum(sum(Lst[a] for a in P) for P, _s in patterns)        # L(CT | ST): non-singleton patterns
    return LD + LCT, usage


def _krimp(transactions, cands, singleton_supp, max_cands):
    """Greedy Krimp: keep each candidate pattern iff it reduces total description length."""
    S0 = sum(singleton_supp.values()) or 1
    Lst = {a: -np.log2(s / S0) for a, s in singleton_supp.items()}
    CT = []
    bestL, _ = _cover_L(transactions, CT, Lst)
    L0 = bestL
    for P, s in sorted(cands, key=lambda c: (-c[1], -len(c[0])))[:max_cands]:   # support desc, length desc
        L, _ = _cover_L(transactions, CT + [(P, s)], Lst)
        if L < bestL:
            CT.append((P, s)); bestL = L
    return CT, L0, bestL


def _merge_cols(Bcsc, groups, labels, dfs):
    """Merge column groups (OR), returning (csc, labels, dfs) with shortest-phrase representatives."""
    cols, lab, df = [], [], []
    for js in groups:
        merged = (np.asarray(Bcsc[:, js].sum(1)).ravel() > 0).astype(np.int8)
        cols.append(sp.csc_matrix(merged[:, None]))
        rep = min((labels[j] for j in js), key=len)
        lab.append(rep + (f" +{len(js) - 1}" if len(js) > 1 else ""))
        df.append(min(dfs[j] for j in js))
    return sp.hstack(cols).tocsc(), lab, df


def _canonicalize(term_bin, labels, dfs, jacc):
    """Embedder-free surface canonicalization. (1) stem-merge: columns whose stemmed, stopword-stripped
    token-SETS match (kills network/networks, snn/snns, neural-network/neural-networks). (2) extensional
    clump: union-find merge columns with extent-Jaccard >= jacc (kills co-occurring synonyms snn/spiking-
    neural, mzi/mach-zehnder). Returns (csr, labels, dfs). No embedder used."""
    try:
        from nltk.stem import PorterStemmer
        stem = PorterStemmer().stem
    except Exception:
        def stem(w):
            return w[:-2] if w.endswith("es") and len(w) > 5 else w[:-1] if w.endswith("s") and len(w) > 4 else w
    STOP = {"have", "which", "based", "using", "used", "via", "high", "low", "new", "art", "state",
            "compared", "different", "novel", "recent", "various", "several", "general", "choice", "have"}
    m = term_bin.shape[1]
    Bcsc = term_bin.tocsc()
    key2cols = defaultdict(list)
    for j in range(m):
        toks = frozenset(stem(t) for t in labels[j].split() if t not in STOP)
        key2cols[toks if toks else f"__drop{j}"].append(j)
    stem_groups = [js for k, js in key2cols.items() if not isinstance(k, str)]
    M1, lab1, df1 = _merge_cols(Bcsc, stem_groups, labels, dfs)
    # extensional clump
    Bb = (M1 > 0).astype(np.float64)
    inter = (Bb.T @ Bb).toarray()
    sz = np.diag(inter).copy()
    J = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1.0)
    np.fill_diagonal(J, 0.0)
    parent = list(range(M1.shape[1]))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a, b in zip(*np.where(np.triu(J >= jacc, 1))):
        parent[find(int(a))] = find(int(b))
    clump_groups = list(defaultdict(list, {find(j): [k for k in range(M1.shape[1]) if find(k) == find(j)]
                                           for j in range(M1.shape[1])}).values())
    clump_groups = [list(set(g)) for g in {frozenset(g) for g in clump_groups}]
    M2, lab2, df2 = _merge_cols(M1.tocsc(), clump_groups, lab1, df1)
    LOGGER.info("canonicalize: %d -> %d (stem) -> %d (clump) term attrs", m, M1.shape[1], M2.shape[1])
    return M2.tocsr(), lab2, np.array(df2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept-gap", required=True, type=Path)
    ap.add_argument("--field", default="17")
    ap.add_argument("--n", type=int, default=11)
    ap.add_argument("--context", choices=["terms", "refs", "concat"], default="concat")
    ap.add_argument("--min-supp", type=int, default=25, help="min support for an attribute to be kept")
    ap.add_argument("--min-lift", type=float, default=2.0,
                    help="keep only terms over-represented vs the field by this factor (distinctiveness)")
    ap.add_argument("--coverage-target", type=float, default=0.9,
                    help="stop the set-cover once this fraction of works is covered (or --n reached)")
    ap.add_argument("--select", choices=["mdl", "setcover"], default="mdl",
                    help="mdl = Krimp-style compression-optimal concept set (knob-free, chooses count); "
                         "setcover = stability-weighted greedy overlapping cover (needs --n)")
    ap.add_argument("--max-cands", type=int, default=300, help="MDL: cap candidate patterns (by support)")
    ap.add_argument("--canonicalize", action="store_true",
                    help="embedder-free surface canonicalization: stem-merge + extensional (Jaccard) clump terms")
    ap.add_argument("--clump-jaccard", type=float, default=0.7,
                    help="merge term attributes whose extents have Jaccard >= this (co-occurring synonyms)")
    ap.add_argument("--snapshot-date", default="2025-11-06")
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=1_000_000)
    args = ap.parse_args()
    setup_logging(log_name="fca_probe", log_level="INFO", include_time=False)

    X, vocab, field_df = build_field_term_index(args.field, args.snapshot_date, args.min_df, args.max_features)
    cfg = OmegaConf.structured(CurriculumConfig()); cfg.experiment.cloud.filter_pair.pre.field = str(args.field)
    cloud = instantiate(cfg.experiment.cloud)
    wid = cloud.work_ids(); id2idx = {w: i for i, w in enumerate(wid)}
    ids = json.loads(args.concept_gap.read_text())["book"]["gap_article_ids"]
    rows = [id2idx[w] for w in ids if w in id2idx]
    N = len(rows)

    # --- build the binary formal context (objects = concept works); prune attrs to support >= min_supp ---
    Xr = (X[rows] > 0)
    tsupp = np.asarray(Xr.sum(0)).ravel()
    BG = X.shape[0]                                         # field corpus size
    lift = (tsupp / max(N, 1)) / ((field_df + 1e-9) / BG)  # concept-vs-field over-representation
    keep_t = np.where((tsupp >= args.min_supp) & (lift > args.min_lift))[0]   # distinctive + frequent
    term_bin = Xr[:, keep_t].astype(np.int8).tocsr()
    term_labels = [vocab[i] for i in keep_t]
    term_df = field_df[keep_t].astype(np.float64)
    if args.canonicalize:
        term_bin, term_labels, term_df = _canonicalize(term_bin, term_labels, term_df, args.clump_jaccard)

    Gc = cloud.citation_graph(False).tocsr().astype(np.float64); Gc.setdiag(0.0); Gc.eliminate_zeros()
    Xo = (Gc[rows] > 0)
    rsupp = np.asarray(Xo.sum(0)).ravel()
    keep_r = np.where(rsupp >= args.min_supp)[0]
    refs_bin = Xo[:, keep_r].astype(np.int8).tocsr()

    if args.context == "terms":
        B = term_bin.tocsr(); n_terms = term_bin.shape[1]
    elif args.context == "refs":
        B = refs_bin.tocsr(); n_terms = 0
    else:
        B = sp.hstack([term_bin, refs_bin]).tocsr(); n_terms = term_bin.shape[1]
    B_csc = B.tocsc()
    is_term = np.arange(B.shape[1]) < n_terms
    obj_cov = (np.asarray(B.sum(1)).ravel() > 0)
    LOGGER.info("context=%s: %d works x %d attrs (%d terms + %d refs); object coverage %.1f%%",
                args.context, B.shape[0], B.shape[1], n_terms, B.shape[1] - n_terms, 100 * obj_cov.mean())

    if args.select == "mdl":
        # --- MDL (Krimp): compression-optimal concept set; chooses the chapter count itself ---
        transactions = [set(B.indices[B.indptr[i]:B.indptr[i + 1]].tolist()) for i in range(B.shape[0])]
        singleton_supp = {int(a): int(s) for a, s in enumerate(np.asarray(B_csc.sum(0)).ravel()) if s > 0}
        cands = _closed_intents(B, B_csc, args.min_supp)
        LOGGER.info("MDL candidate closed intents (|intent|>=2): %d", len(cands))
        CT, L0, LK = _krimp(transactions, cands, singleton_supp, args.max_cands)
        order = sorted(CT, key=lambda p: (-len(p[0]), -p[1]))
        groups = defaultdict(list)
        for wi, t in enumerate(transactions):
            chosen = next((P for P, _s in order if P <= t), None)     # largest pattern covering the work
            groups[chosen].append(wi)
        EMB = normalize(np.asarray(cloud.embeddings(), dtype=np.float64)[rows])

        def _lab(P):
            ta = sorted((a for a in P if a < n_terms), key=lambda a: term_df[a])
            return [term_labels[a] for a in ta[:6]] or [f"{sum(a >= n_terms for a in P)} shared refs"]

        bg = len(groups.get(None, []))
        chapters = sorted(((P, m) for P, m in groups.items() if P is not None), key=lambda x: -len(x[1]))
        print(f"\n===== MDL (Krimp) selection  (context={args.context}, min_supp={args.min_supp}, "
              f"min_lift={args.min_lift}) =====")
        print(f"MDL chose {len(chapters)} chapters; {100*(1-LK/L0):.1f}% compression; "
              f"{bg} works unassigned (background, {100*bg/max(N,1):.0f}%)")
        print(f"{'#':>2} {'works':>5} {'|int|':>5} {'coh':>5}  label (intent terms)")
        for i, (P, m) in enumerate(chapters):
            e = EMB[m]; coh = float((e @ e.mean(0)).mean()) if len(m) > 1 else 1.0
            print(f"{i:>2} {len(m):>5} {len(P):>5} {coh:>5.2f}  " + ", ".join(_lab(P)))
        return

    # --- mine candidate closed concepts (attribute concepts, chapter-grain) ---
    lo, hi = max(args.min_supp, 20), int(0.45 * N)
    concepts = _attribute_concepts(B, B_csc, args.min_supp, lo, hi)
    LOGGER.info("candidate closed concepts (extent in [%d,%d]): %d", lo, hi, len(concepts))

    # --- score: stability + embedding cohesion; label every concept by its extent's top lift terms ---
    EMB = normalize(np.asarray(cloud.embeddings(), dtype=np.float64)[rows])
    Xr_full = Xr.tocsr()                                     # works x full vocab (presence)

    def _label(ext):
        sup = np.asarray(Xr_full[ext].sum(0)).ravel()
        common = sup / len(ext)
        liftx = common / ((field_df + 1e-9) / BG)           # over-rep vs field
        cand = np.where(common > 0.25)[0]                   # present in >25% of the extent
        cand = cand[np.argsort(liftx[cand])[::-1]][:5]
        return [vocab[i] for i in cand]

    scored = []
    for ext, intent in concepts:
        stab = _stability(B, ext, intent)
        e = EMB[ext]; coh = float((e @ e.mean(0)).mean())   # mean cos to centroid
        n_ref_intent = int(intent[n_terms:].sum()) if n_terms < len(intent) else int(intent.sum())
        scored.append({"ext": ext, "stab": stab, "coh": coh, "size": len(ext),
                       "label": _label(ext), "refI": n_ref_intent})
    if not scored:
        print("no candidate concepts — lower --min-supp"); return
    coh_gate = float(np.percentile([s["coh"] for s in scored], 25))  # data-driven paratext drop
    LOGGER.info("embedding-cohesion 25th-pctile gate = %.3f (concepts below = paratext-like, dropped)", coh_gate)

    # --- greedy STABILITY-WEIGHTED overlapping set cover (cohesion gate) ---
    # pick the concept maximizing stability * (NEW works covered); a near-duplicate of a pick adds ~no new
    # coverage -> auto-skipped, and big stable foundational topics come first. Stop at coverage target or N.
    pool = [s for s in scored if s["coh"] >= coh_gate]
    picked, covered, used = [], np.zeros(N, dtype=bool), set()
    for _ in range(args.n):
        best_i, best_gain, best_new = -1, 0.0, 0
        for i, s in enumerate(pool):
            if i in used:
                continue
            new = int((~covered[s["ext"]]).sum())
            gain = s["stab"] * new
            if gain > best_gain:
                best_i, best_gain, best_new = i, gain, new
        if best_i < 0 or best_new < 20:                     # nothing left adds real new coverage
            break
        used.add(best_i); picked.append(pool[best_i]); covered[pool[best_i]["ext"]] = True
        if covered.mean() >= args.coverage_target:
            break

    print(f"\n===== stability-weighted set cover  (context={args.context}, min_supp={args.min_supp}, "
          f"min_lift={args.min_lift}) =====")
    print(f"object coverage by context: {100*obj_cov.mean():.1f}%   cover reaches {100*covered.mean():.1f}% "
          f"of works with {len(picked)} concepts (cap {args.n})")
    print(f"{'#':>2} {'size':>5} {'stab':>5} {'coh':>5} {'refI':>5}  label (top extent lift-terms)")
    for i, s in enumerate(sorted(picked, key=lambda s: -s["size"])):
        print(f"{i:>2} {s['size']:>5} {s['stab']:>5.2f} {s['coh']:>5.2f} {s['refI']:>5}  "
              + ", ".join(s["label"]))


if __name__ == "__main__":
    main()
