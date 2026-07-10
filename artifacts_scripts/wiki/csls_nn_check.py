#!/usr/bin/env python
# artifacts_scripts/wiki/csls_nn_check.py
"""Topic-relevant reverse NN + CSLS hubness correction for the OpenAlex<->Wikipedia
cross-corpus nearest-neighbor check.

Two fixes over the cross-corpus NN block in ``shared_space_analysis.py``:

**A. Topic-relevant reverse direction.** The original wiki->OpenAlex check sampled
*uniform* Wikipedia rows (Varazdin, Pearl Drums, ...) — off-topic encyclopedic
pages with no genuine NLP neighbor, so "do they land on sensible OpenAlex NLP
articles?" was an unanswerable (meaningless) question. Here both directions use
**NLP-relevant** samples: the wiki query points are drawn from the top-K Wikipedia
by cosine to the mean-centered OpenAlex-NLP-article centroid — the SAME selection
``cross_corpus_gap_demo.py --topic-filter`` uses. Then the reverse NN is a fair
test: do NLP-relevant wiki pages (Machine translation, WordNet, ...) map to
sensible OpenAlex NLP articles?

**B. CSLS (Cross-domain Similarity Local Scaling).** As an alternative to raw
cosine for the cross-corpus NN, to reduce hubness (a few generic target points
being everyone's nearest neighbor). For a source query x and target candidate y::

    CSLS(x, y) = 2 * cos(x, y) - r_T(x) - r_S(y)

where ``r_T(x)`` is the mean cosine of x to its k nearest *target*-corpus points
and ``r_S(y)`` is the mean cosine of y to its k nearest *source*-corpus points
(k~=10). Hub target points have a high ``r_S`` (they are densely surrounded by
source points), so CSLS penalizes them, demoting them from the top-1 slot.

We report, on the NLP-relevant samples and in BOTH directions:
  - NN top-1 hub share under raw cosine vs CSLS (lower = less hubness),
  - example NN lists raw-vs-CSLS.

Everything is the same raw 384-D GIST space as ``shared_space_analysis.py``:
the OpenAlex side is the NLP slice (denoiser=NONE, drop_k=0); the Wikipedia side
reuses ``cache/wiki/store/wiki_400k`` (no re-embed). Like the topic-filter demo,
each corpus is mean-centered (OpenAlex centroid over the NLP articles, Wikipedia
centroid over the FULL store) then re-normalized before the NN check.

Usage::

    uv run python -m artifacts_scripts.wiki.csls_nn_check \
        --wiki-store cache/wiki/store/wiki_400k --wiki-topk 3000 \
        --csls-k 10 --n-nn-examples 200 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from bookgap.corpus import embedder_fingerprint
from bookgap.wikipedia.corpus import WikipediaCorpus

# OpenAlex raw-GIST overrides: same pinned AI subfield (1702) as tests/conftest,
# denoiser=NONE / drop_k=0 so embeddings are raw 384-D GIST (the Wikipedia space).
OA_OVERRIDES = [
    "cloud.filter_pair.pre.domain=null",
    "cloud.filter_pair.pre.subfield='1702'",
    "cloud.filter_pair.pre.work_types=[book,article]",
    "cloud.denoiser=NONE",
    "cloud.drop_k=0",
]
NLP_TOPIC = "T10181"  # NLP-Techniques topic within subfield 1702.


def _normalize(X: NDArray[np.float32]) -> NDArray[np.float32]:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n).astype(np.float32)


def load_openalex_nlp() -> tuple[NDArray[np.float32], list[str], list[str], str]:
    """Return (embeddings, titles, ids, fingerprint) for OpenAlex NLP articles, raw GIST."""
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    from bookgap.configs.config_store import register_configs
    from bookgap.subcloud import SubCloud
    from bookgap.util import Filter

    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        cfg = compose(config_name="standard", overrides=OA_OVERRIDES)
    cloud = instantiate(cfg.cloud)
    fp = embedder_fingerprint(cloud.embedder)
    nlp = SubCloud(parent=cloud, post_filter=Filter(topic=NLP_TOPIC, work_types=("article",)))
    emb = np.ascontiguousarray(nlp.embeddings(), dtype=np.float32)
    return emb, nlp.work_titles(), nlp.work_ids(), fp


def load_wikipedia(store_dir: Path) -> tuple[NDArray[np.float32], list[str], list[str], str]:
    wc = WikipediaCorpus(store_dir)
    emb = np.ascontiguousarray(wc.embeddings(), dtype=np.float32)
    return emb, wc.work_titles(), wc.work_ids(), wc.embedder_fingerprint


# --- CSLS -------------------------------------------------------------------
# All inputs are L2-normalized -> cosine == dot product. CSLS local-scaling term
# r_C(z) = mean cosine of z to its k nearest points in corpus C. For a cross-corpus
# score, the source query x is scaled by its mean cosine to the k nearest TARGET
# points (r_T), and each target candidate y by its mean cosine to the k nearest
# SOURCE points (r_S). The hubness fix is the r_S(y) term: a target hub densely
# surrounded by source points has a high r_S and is demoted.


def mean_topk_cosine(query: NDArray, ref: NDArray, k: int) -> NDArray[np.float32]:
    """For each row of `query`, mean cosine to its k nearest rows of `ref`.

    Both assumed L2-normalized -> cosine == dot. Returns shape (len(query),).
    `query` and `ref` may be the same array region; we do NOT exclude self —
    for cross-corpus scaling query and ref are different corpora so self-match
    cannot occur, which is exactly the CSLS setting here.
    """
    out = np.empty(query.shape[0], dtype=np.float32)
    # Chunk over queries to bound the (q, len(ref)) sim matrix memory.
    chunk = 2048
    for start in range(0, query.shape[0], chunk):
        sims = query[start:start + chunk] @ ref.T  # (c, len(ref))
        # top-k along axis 1 without full sort.
        if k < sims.shape[1]:
            part = np.partition(sims, sims.shape[1] - k, axis=1)[:, -k:]
        else:
            part = sims
        out[start:start + chunk] = part.mean(axis=1)
    return out


def csls_scores(
    query_emb: NDArray, target_emb: NDArray, r_T_query: NDArray, r_S_target: NDArray,
) -> NDArray[np.float32]:
    """Full CSLS score matrix (len(query), len(target)).

    r_T_query[i]   = mean cosine of query i to its k nearest TARGET points.
    r_S_target[j]  = mean cosine of target j to its k nearest SOURCE points.
    CSLS(i, j) = 2*cos(i, j) - r_T_query[i] - r_S_target[j].
    """
    sims = query_emb @ target_emb.T  # (q, M), cosine
    return (2.0 * sims - r_T_query[:, None] - r_S_target[None, :]).astype(np.float32)


# --- NN + hubness on a sample ----------------------------------------------

def nn_top1(score_row_fn, sample: NDArray, n_target: int) -> NDArray[np.int64]:
    """Argmax target index per sampled query, given a callable that returns the
    score row (length n_target) for a query row index."""
    nn1 = np.empty(len(sample), dtype=np.int64)
    for r, qi in enumerate(sample):
        nn1[r] = int(np.argmax(score_row_fn(qi)))
    return nn1


def hub_top1_share(nn1: NDArray[np.int64]) -> tuple[int, int, float]:
    """(n_unique_nn, n_queries, top1_share) — top1_share = fraction whose single
    nearest target is the single most-popular target across the sample."""
    uniq, counts = np.unique(nn1, return_counts=True)
    return len(uniq), len(nn1), float(counts.max() / len(nn1))


def topk_titles(score_row: NDArray, target_titles: list[str], k: int) -> list[str]:
    idx = np.argsort(-score_row)[:k]
    return [target_titles[j] for j in idx]


# --- Reporting --------------------------------------------------------------

def _fmt_examples(
    rows: list[tuple[str, list[str], list[str]]], n_show: int,
) -> list[str]:
    """rows: (query_title, raw_nn_titles, csls_nn_titles)."""
    lines: list[str] = []
    for q, raw_nn, csls_nn in rows[:n_show]:
        lines.append(f"- **{q[:90]}**")
        lines.append(f"  - raw : {', '.join(t[:55] for t in raw_nn)}")
        lines.append(f"  - csls: {', '.join(t[:55] for t in csls_nn)}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/wiki_400k"))
    ap.add_argument("--wiki-topk", type=int, default=3000,
                    help="how many top-cosine-to-NLP-centroid Wikipedia rows form "
                         "the NLP-relevant subset (same selection as --topic-filter)")
    ap.add_argument("--csls-k", type=int, default=10, help="k for CSLS local scaling")
    ap.add_argument("--n-nn-examples", type=int, default=200,
                    help="sampled queries per direction for the hub-share statistic")
    ap.add_argument("--n-show", type=int, default=5, help="example NN lists to print/write")
    ap.add_argument("--nn-k", type=int, default=3, help="top-k NN titles shown per example")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("docs/wikipedia-openalex-shared-space.md"))
    args = ap.parse_args()

    print("Loading OpenAlex NLP slice (raw GIST) ...")
    oa, oa_titles, _oa_ids, oa_fp = load_openalex_nlp()
    print(f"  OA NLP articles: {oa.shape}, fp={oa_fp}")
    print("Loading Wikipedia corpus ...")
    wiki, wiki_titles, _wiki_ids, wiki_fp = load_wikipedia(args.wiki_store)
    print(f"  Wiki store: {wiki.shape}, fp={wiki_fp}")
    assert oa_fp == wiki_fp, f"fingerprint mismatch: OA={oa_fp} wiki={wiki_fp}"

    oa = _normalize(oa)
    wiki = _normalize(wiki)

    # --- Mean-center each corpus, re-normalize (same alignment as the gap demo) --
    # OpenAlex centroid over the NLP articles; Wikipedia centroid over the FULL
    # store (the corpus-offset estimate reflects the full encyclopedic
    # distribution). Topic selection then happens on the centered wiki rows.
    oa_c = _normalize(oa - oa.mean(axis=0, keepdims=True))
    wiki_c = _normalize(wiki - wiki.mean(axis=0, keepdims=True))

    # --- NLP-relevant Wikipedia subset (the --topic-filter selection) -----------
    nlp_dir = oa_c.mean(axis=0)
    nlp_norm = np.linalg.norm(nlp_dir)
    nlp_dir = (nlp_dir / nlp_norm).astype(np.float32) if nlp_norm > 0 else nlp_dir
    wiki_cos = wiki_c @ nlp_dir
    topk = min(args.wiki_topk, wiki_cos.shape[0])
    part = np.argpartition(-wiki_cos, topk - 1)[:topk]
    wiki_rel_sel = part[np.argsort(-wiki_cos[part])].astype(np.int64)
    wiki_rel = np.ascontiguousarray(wiki_c[wiki_rel_sel])
    wiki_rel_titles = [wiki_titles[i] for i in wiki_rel_sel.tolist()]
    print(f"  NLP-relevant Wikipedia subset: top-{topk} "
          f"(cos {wiki_cos[wiki_rel_sel].min():.3f}..{wiki_cos[wiki_rel_sel].max():.3f})")
    print("  Top 8 selected:", ", ".join(wiki_rel_titles[i][:40] for i in range(8)))

    # --- CSLS local-scaling terms (computed once over the relevant subsets) ------
    # Direction OA->wiki: source = OA, target = wiki_rel.
    #   r_T(x in OA)      = mean cos of OA row to its k nearest wiki_rel rows.
    #   r_S(y in wiki_rel)= mean cos of wiki_rel row to its k nearest OA rows.
    # Direction wiki->OA: source = wiki_rel, target = OA.
    #   r_T(x in wiki_rel)= mean cos of wiki_rel row to its k nearest OA rows.
    #   r_S(y in OA)      = mean cos of OA row to its k nearest wiki_rel rows.
    # The two r_S/r_T pairs reuse the same two quantities (symmetric):
    print(f"\nComputing CSLS local-scaling terms (k={args.csls_k}) ...")
    r_oa_to_wiki = mean_topk_cosine(oa_c, wiki_rel, args.csls_k)        # OA row -> k nearest wiki_rel
    r_wiki_to_oa = mean_topk_cosine(wiki_rel, oa_c, args.csls_k)        # wiki_rel row -> k nearest OA
    print("  done.")

    rng = np.random.default_rng(args.seed)

    # ============ Direction 1: OpenAlex NLP article -> Wikipedia ============
    oa_sample = rng.choice(len(oa_c), size=min(args.n_nn_examples, len(oa_c)), replace=False)

    def oa_raw_row(qi: int) -> NDArray:
        return oa_c[qi] @ wiki_rel.T

    def oa_csls_row(qi: int) -> NDArray:
        sims = oa_c[qi] @ wiki_rel.T
        return 2.0 * sims - r_oa_to_wiki[qi] - r_wiki_to_oa

    oa_nn1_raw = nn_top1(oa_raw_row, oa_sample, wiki_rel.shape[0])
    oa_nn1_csls = nn_top1(oa_csls_row, oa_sample, wiki_rel.shape[0])
    oa_hub_raw = hub_top1_share(oa_nn1_raw)
    oa_hub_csls = hub_top1_share(oa_nn1_csls)

    # ============ Direction 2: Wikipedia (NLP-relevant) -> OpenAlex NLP ============
    wiki_sample = rng.choice(wiki_rel.shape[0], size=min(args.n_nn_examples, wiki_rel.shape[0]), replace=False)

    def wiki_raw_row(qi: int) -> NDArray:
        return wiki_rel[qi] @ oa_c.T

    def wiki_csls_row(qi: int) -> NDArray:
        sims = wiki_rel[qi] @ oa_c.T
        return 2.0 * sims - r_wiki_to_oa[qi] - r_oa_to_wiki

    wiki_nn1_raw = nn_top1(wiki_raw_row, wiki_sample, oa_c.shape[0])
    wiki_nn1_csls = nn_top1(wiki_csls_row, wiki_sample, oa_c.shape[0])
    wiki_hub_raw = hub_top1_share(wiki_nn1_raw)
    wiki_hub_csls = hub_top1_share(wiki_nn1_csls)

    print("\n=== Hub top-1 share (lower = less hubness) ===")
    print(f"  OA->wiki   raw={oa_hub_raw[2]:.3f} (uniq {oa_hub_raw[0]}/{oa_hub_raw[1]})  "
          f"csls={oa_hub_csls[2]:.3f} (uniq {oa_hub_csls[0]}/{oa_hub_csls[1]})")
    print(f"  wiki->OA   raw={wiki_hub_raw[2]:.3f} (uniq {wiki_hub_raw[0]}/{wiki_hub_raw[1]})  "
          f"csls={wiki_hub_csls[2]:.3f} (uniq {wiki_hub_csls[0]}/{wiki_hub_csls[1]})")

    # --- Example NN lists raw vs CSLS (use the first n_show sampled queries) ------
    oa_ex: list[tuple[str, list[str], list[str]]] = []
    for qi in oa_sample[:args.n_show]:
        oa_ex.append((
            oa_titles[qi],
            topk_titles(oa_raw_row(qi), wiki_rel_titles, args.nn_k),
            topk_titles(oa_csls_row(qi), wiki_rel_titles, args.nn_k),
        ))
    wiki_ex: list[tuple[str, list[str], list[str]]] = []
    for qi in wiki_sample[:args.n_show]:
        wiki_ex.append((
            wiki_rel_titles[qi],
            topk_titles(wiki_raw_row(qi), oa_titles, args.nn_k),
            topk_titles(wiki_csls_row(qi), oa_titles, args.nn_k),
        ))

    _append_doc(
        args, oa, wiki, oa_fp, topk,
        oa_hub_raw, oa_hub_csls, wiki_hub_raw, wiki_hub_csls,
        oa_ex, wiki_ex,
    )
    print(f"\nAppended CSLS section to {args.out}")


def _append_doc(
    args, oa, wiki, fp, topk,
    oa_hub_raw, oa_hub_csls, wiki_hub_raw, wiki_hub_csls,
    oa_ex, wiki_ex,
) -> None:
    L: list[str] = []
    L.append("## CSLS + topic-relevant reverse NN\n")
    L.append(
        "The cross-corpus NN block in §2 had two weaknesses, fixed here. **(A)** Its "
        "*reverse* direction sampled **uniform** Wikipedia rows (*Varazdin*, *Pearl "
        "Drums*, ...) — off-topic encyclopedic pages with no genuine NLP neighbor, so "
        "\"do they map to sensible OpenAlex NLP articles?\" was unanswerable. Here both "
        "directions use **NLP-relevant** samples: the Wikipedia query points are drawn "
        f"from the **top-{topk}** Wikipedia by cosine to the mean-centered "
        "OpenAlex-NLP-article centroid (the same selection "
        "`cross_corpus_gap_demo.py --topic-filter` uses). **(B)** It scored NN with raw "
        "cosine, which is hubness-prone (a few generic target points become everyone's "
        "nearest neighbor). We add **CSLS** (Cross-domain Similarity Local Scaling) as "
        "an alternative and measure whether it reduces hubness.\n"
    )
    L.append(
        "CSLS(x, y) = 2·cos(x, y) − r_T(x) − r_S(y), where r_T(x) is the mean cosine of "
        f"the source query x to its k={args.csls_k} nearest *target*-corpus points and "
        f"r_S(y) is the mean cosine of the target candidate y to its k={args.csls_k} "
        "nearest *source*-corpus points. A target **hub** is densely surrounded by "
        "source points, so it has a high r_S(y) and CSLS subtracts it off — demoting the "
        "hub from the top-1 slot.\n"
    )
    L.append("### Setup\n")
    L.append(
        f"- Same raw 384-D GIST space as §1–2: OpenAlex NLP articles (N={oa.shape[0]}, "
        f"denoiser=NONE drop_k=0) and the full Wikipedia store `{args.wiki_store}` "
        f"(N={wiki.shape[0]}, fp `{fp}`)."
    )
    L.append(
        "- Each corpus mean-centered (OpenAlex centroid over the NLP articles, Wikipedia "
        "centroid over the **full** store) then re-normalized — identical alignment to "
        "the topic-filtered gap demo."
    )
    L.append(
        f"- NLP-relevant Wikipedia subset = top-{topk} by cosine to the mean-centered "
        "NLP-article centroid. Hub-share over "
        f"{oa_hub_raw[1]} (OA→wiki) / {wiki_hub_raw[1]} (wiki→OA) sampled queries, seed "
        f"{args.seed}.\n"
    )

    L.append("### Hub top-1 share — raw cosine vs CSLS (both directions)\n")
    L.append(
        "Top-1 hub share = fraction of sampled queries whose single nearest target is "
        "the *same* most-popular target across the sample. Lower = less hub collapse. "
        "(unique NN / queries in parentheses.)\n"
    )
    L.append("| direction | scorer | NN top-1 hub share | unique NN / queries |")
    L.append("|---|---|---|---|")
    L.append(f"| OpenAlex → Wikipedia | raw cosine | {oa_hub_raw[2]:.3f} | {oa_hub_raw[0]} / {oa_hub_raw[1]} |")
    L.append(f"| OpenAlex → Wikipedia | CSLS | {oa_hub_csls[2]:.3f} | {oa_hub_csls[0]} / {oa_hub_csls[1]} |")
    L.append(f"| Wikipedia → OpenAlex | raw cosine | {wiki_hub_raw[2]:.3f} | {wiki_hub_raw[0]} / {wiki_hub_raw[1]} |")
    L.append(f"| Wikipedia → OpenAlex | CSLS | {wiki_hub_csls[2]:.3f} | {wiki_hub_csls[0]} / {wiki_hub_csls[1]} |")
    L.append("")

    L.append("### Example NN lists — raw vs CSLS\n")
    L.append("#### OpenAlex NLP article → nearest NLP-relevant Wikipedia\n")
    L.extend(_fmt_examples(oa_ex, args.n_show))
    L.append("#### NLP-relevant Wikipedia → nearest OpenAlex NLP article\n")
    L.extend(_fmt_examples(wiki_ex, args.n_show))

    # Verdict is appended by the script with the measured numbers interpolated; the
    # CSLS-in-gap-detection recommendation is added (and kept current) by hand below
    # the table since it is a reasoned argument, not a measured number.
    out = args.out
    existing = out.read_text() if out.exists() else ""
    sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
    out.write_text(existing + sep + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
