"""Wiki-gap spike v2 — OA-concept + LIVE-Wikipedia-API coverage check.

v1 (`wiki_gap_spike.py`) used an embedded wiki k-ball and scored 0/8: a 5k
topic-filtered wiki slice meant "no wiki landmark in the ball" mostly meant "not
in the slice," manufacturing false gaps (slice-incompleteness, the embedding twin
of OSM's reference-data-incompleteness). It also surfaced peak-density topics
(MT, parsing) that are exactly the best Wikipedia-covered ones.

v2 fixes all of that:
  * DROP the embedded wiki slice / k-ball. The coverage signal is the LIVE
    Wikipedia API (`classify_gap_via_oracle`), which moots both slice-
    incompleteness AND the wiki<->OA register residual (no cross-corpus
    embedding alignment needed at all).
  * ENGLISH-filter the OA works (drop the multilingual contamination — v1's
    Finnish "Saturday Sauna" cluster).
  * Finer clustering for SPECIFIC concepts (v1's blobs of 'semantic, corpus'
    were too coarse); report niche concepts, not peak-density.

A gap = a coherent OA concept (research exists, multiple works) for which the
live Wikipedia API finds NO focused article.

Regenerate:
    uv run \
        python -m artifacts_scripts.wiki.wiki_gap_spike2 --k 120 --max-oracle 90
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from artifacts_scripts.wiki.wiki_gap_spike import (
    classify_gap_via_oracle,
    distinctive_terms_for_groups,
    load_oa_nlp_cached,
)

# English function words: ~10-15% of English prose, ~0% of foreign text — a clean,
# dependency-free language filter that catches Latin-script foreign (Finnish,
# Estonian, Turkish) which an ASCII check would miss.
ENGLISH_FUNC = {
    "the", "of", "and", "to", "in", "is", "for", "with", "that", "this", "are",
    "on", "as", "by", "be", "from", "at", "which", "or", "can", "we", "an", "it",
    "these", "have", "has", "was", "were", "their", "its", "but", "not", "also",
}


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def english_fraction(text: str) -> float:
    toks = [w.strip(".,;:()[]'\"-") for w in text.lower().split()]
    toks = [w for w in toks if w]
    if not toks:
        return 0.0
    return sum(1 for w in toks if w in ENGLISH_FUNC) / len(toks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=120, help="k-means clusters (finer = more specific concepts)")
    ap.add_argument("--min-size", type=int, default=8, help="drop clusters smaller than this (noise)")
    ap.add_argument("--max-size", type=int, default=2000, help="drop giant generic blobs")
    ap.add_argument("--english-min", type=float, default=0.05, help="min English-function-word fraction")
    ap.add_argument("--max-oracle", type=int, default=90, help="cap live-API concepts (politeness)")
    ap.add_argument("--no-oracle", action="store_true", help="dry run: cluster+terms only, no API")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("cache/wiki/wiki_gap_spike2_result.json"))
    ap.add_argument("--oa-sample", type=int, default=0, help="subsample N OA works (0=all; fast iteration)")
    ap.add_argument("--mine-sample", type=int, default=12000, help="phrase-mining doc sample (smaller=faster)")
    ap.add_argument("--mine-phrases", dest="mine_phrases", action="store_true", default=True)
    ap.add_argument("--no-mine-phrases", dest="mine_phrases", action="store_false")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel workers for term extraction (-1=all cores)")
    ap.add_argument("--slice-npz", type=Path, default=None,
                    help="load arts_emb/arts_titles/arts_abstracts from this npz instead of OA-NLP (cross-field test)")
    args = ap.parse_args()

    if args.slice_npz:
        z = np.load(args.slice_npz, allow_pickle=True)
        oa = {"arts_emb": z["arts_emb"], "arts_titles": z["arts_titles"], "arts_abstracts": z["arts_abstracts"]}
        _log(f"loaded slice {args.slice_npz}")
    else:
        oa = load_oa_nlp_cached()
    emb = np.ascontiguousarray(oa["arts_emb"], dtype=np.float32)
    titles = list(oa["arts_titles"])
    abstracts = list(oa["arts_abstracts"])
    n = emb.shape[0]
    _log(f"OA articles: {n}")

    # ---- English filter (drop multilingual contamination) ----
    texts = [f"{t} {a}".strip() for t, a in zip(titles, abstracts)]
    ef = np.array([english_fraction(x) for x in texts])
    keep = ef >= args.english_min
    idx = np.where(keep)[0]
    _log(f"English-filtered: {idx.size}/{n} kept (>= {args.english_min} func-word frac)")
    if args.oa_sample and idx.size > args.oa_sample:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, size=args.oa_sample, replace=False))
        _log(f"subsampled to {idx.size} works (--oa-sample {args.oa_sample})")
    emb_e = np.ascontiguousarray(emb[idx])
    titles_e = [titles[i] for i in idx]
    texts_e = [texts[i] for i in idx]

    # ---- normalize + cluster ----
    embn = emb_e / np.maximum(np.linalg.norm(emb_e, axis=1, keepdims=True), 1e-9)
    _log(f"clustering {embn.shape[0]} works into k={args.k} (MiniBatchKMeans) ...")
    km = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
    labels = km.fit_predict(embn)

    groups = []
    for c in range(args.k):
        members = np.where(labels == c)[0]
        if args.min_size <= members.size <= args.max_size:
            groups.append((int(c), members))
    _log(f"{len(groups)} clusters in [{args.min_size},{args.max_size}]")

    group_docs = [[texts_e[i] for i in members] for _, members in groups]
    _log(f"extracting Fightin'-Words distinctive terms (phrases={args.mine_phrases}) ...")
    termlists = distinctive_terms_for_groups(
        group_docs, top_k=12, seed=args.seed,
        mine_phrases=args.mine_phrases, mine_sample=args.mine_sample, n_jobs=args.n_jobs,
    )

    concepts = []
    for (c, members), terms_scored in zip(groups, termlists):
        terms = [t for t, _ in terms_scored]
        concepts.append({
            "cluster": c, "size": int(members.size),
            "distinctive_terms": terms,
            "rep_titles": [titles_e[i] for i in members[:6].tolist()],
        })
    # smallest (most niche) first — the real gaps live away from peak density
    concepts.sort(key=lambda x: x["size"])

    if not args.no_oracle:
        _log(f"live Wikipedia oracle on up to {args.max_oracle} concepts ...")
        for cc in concepts[: args.max_oracle]:
            terms = cc["distinctive_terms"]
            # Synthesize a concept-name phrase from the top single-word terms so the
            # oracle issues a focused phrase query ("semantic role labeling") even
            # when phrase-mining is off — avoids the v1 mining timeout.
            head = [t for t in terms if "_" not in t][:3]
            aug = (["_".join(head)] if len(head) >= 2 else []) + terms
            try:
                cc["oracle"] = classify_gap_via_oracle(aug, cc["rep_titles"])
            except Exception as e:  # noqa: BLE001 — spike: never die on one API hiccup
                cc["oracle"] = {"verdict": f"ERR:{e}", "hits": []}
            _log(f"  [{cc['cluster']:3d}] {cc['oracle']['verdict']:14s} sz={cc['size']:4d} "
                 f"{cc['distinctive_terms'][:5]}")

    verds = Counter(c["oracle"]["verdict"] for c in concepts if "oracle" in c)
    real = [c for c in concepts if c.get("oracle", {}).get("verdict") == "REAL-GAP"]
    result = {
        "k": args.k, "n_total": n, "n_english": int(idx.size),
        "n_clusters_kept": len(groups), "verdict_counts": dict(verds),
        "n_real_gap": len(real), "concepts": concepts,
        "params": vars(args),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nVERDICTS: {dict(verds)}", flush=True)
    print(f"REAL-GAP concepts: {len(real)}  -> {args.out}", flush=True)
    for c in real[:20]:
        print(f"  [{c['cluster']}] sz={c['size']} {c['distinctive_terms'][:6]}", flush=True)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# PROVENANCE: regenerate (do not hand-edit cache/wiki/wiki_gap_spike2_result.json):
#   uv run \
#       python -m artifacts_scripts.wiki.wiki_gap_spike2 --k 120 --max-oracle 90
# Reuses load_oa_nlp_cached + distinctive_terms_for_groups + classify_gap_via_oracle
# from wiki_gap_spike.py. Coverage signal = LIVE Wikipedia API (no embedded slice,
# no cross-corpus alignment). English-filtered, niche-first concept ranking.
# ---------------------------------------------------------------------------
