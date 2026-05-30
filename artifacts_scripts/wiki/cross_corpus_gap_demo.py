#!/usr/bin/env python
# artifacts_scripts/wiki/cross_corpus_gap_demo.py
"""Cross-corpus gap detection: OpenAlex NLP books as landmarks, a UnionCorpus of
OpenAlex NLP articles + a Wikipedia slice as the ambient.

This is the SP3 capstone demo: ``detect_gaps`` is now source-agnostic (the
gap algorithms accept any ``Corpus``), so the *ambient* can be a cross-corpus
``UnionCorpus``. It exercises the relaxed ``get_kball_coverage`` (which now only
needs ``.index()`` / ``.embeddings()`` and a shared ``embedder_fingerprint``,
not a shared ``SubCloud.parent``) and the range-searchable ``UnionCorpus.index()``.

Alignment: the domain-conflation experiment
(docs/wikipedia-openalex-shared-space.md) found the OpenAlex<->Wikipedia shift in
raw 384-D GIST is a **pure per-corpus offset**. So we **mean-center each corpus
before the union** (subtract the per-corpus centroid, then re-normalize to the
unit sphere). The OpenAlex *books* (landmarks) are centered by the SAME OpenAlex
centroid as the OpenAlex *articles*, so books and articles stay coherent while
the academic<->encyclopedic offset between OpenAlex and Wikipedia is removed.

Everything is raw 384-D GIST (denoiser=NONE, drop_k=0) so the OpenAlex slice is
byte-comparable to the Wikipedia store.

Usage::

    conda run -p .conda python -m artifacts_scripts.wiki.cross_corpus_gap_demo \
        --wiki-store cache/wiki/store/wiki_400k \
        --num-gaps 3 --radius-fraction 0.75 --seed 0

Topic-filtered variant (``--topic-filter``): instead of a uniform Wikipedia
subsample (dominated by off-topic stubs), keep the ``--wiki-topk`` Wikipedia
articles whose mean-centered embedding has the highest cosine to the
mean-centered OpenAlex-NLP-article centroid. This surfaces *cross-corpus*
NLP gaps (regions where OpenAlex articles and topically-relevant Wikipedia
articles co-occupy a book-free region) rather than generic encyclopedic stubs.
No re-embedding: it reuses the already-embedded ``wiki_400k`` store and only
changes which rows enter the union::

    conda run -p .conda python -m artifacts_scripts.wiki.cross_corpus_gap_demo \
        --wiki-store cache/wiki/store/wiki_400k --topic-filter --wiki-topk 3000 \
        --oa-limit 4000 --num-gaps 5 --radius-fraction 0.75 --seed 0
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

import faiss
import numpy as np
from numpy.typing import NDArray

from bookgap.corpus import UnionCorpus
from bookgap.gaps import detect_gaps
from bookgap.index import FlatL2, build_index
from bookgap.types import CellType
from bookgap.wikipedia.corpus import WikipediaCorpus

# Same pinned AI subfield (1702) as tests/conftest, but denoiser=NONE / drop_k=0
# so embeddings are raw 384-D GIST (the Wikipedia space).
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


@dataclass(frozen=True)
class MemCorpus:
    """A frozen in-memory ``Corpus`` over already-materialized (e.g. centered)
    embeddings. Used to feed the mean-centered embeddings into ``detect_gaps``
    without round-tripping through a store. Hashable (for joblib-cached ``knn``)
    via an explicit ``identity`` tag the caller supplies."""

    _embeddings: NDArray[np.float32]
    _work_ids: tuple[str, ...]
    _work_titles: tuple[str, ...]
    _fingerprint: str
    _name: str
    _identity: Hashable

    def embeddings(self) -> NDArray[np.float32]:
        return self._embeddings

    def index(self) -> faiss.Index:
        return build_index(FlatL2(), np.ascontiguousarray(self._embeddings, dtype=np.float32))

    def work_ids(self) -> list[str]:
        return list(self._work_ids)

    def work_titles(self) -> list[str]:
        return list(self._work_titles)

    def work_abstracts(self) -> list[str]:
        return ["" for _ in self._work_ids]

    def name(self) -> str:
        return self._name

    @property
    def embedder_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def identity(self) -> Hashable:
        return self._identity


def load_openalex_nlp():
    """Return (books_subcloud, articles_subcloud, fingerprint) for OpenAlex NLP, raw GIST."""
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    from bookgap.configs.config_store import register_configs
    from bookgap.corpus import embedder_fingerprint
    from bookgap.subcloud import SubCloud
    from bookgap.util import Filter

    register_configs()
    with initialize_config_module(version_base=None, config_module="bookgap.configs"):
        cfg = compose(config_name="standard", overrides=OA_OVERRIDES)
    cloud = instantiate(cfg.cloud)
    fp = embedder_fingerprint(cloud.embedder)
    books = SubCloud(parent=cloud, post_filter=Filter(topic=NLP_TOPIC, work_types=("book",)))
    articles = SubCloud(parent=cloud, post_filter=Filter(topic=NLP_TOPIC, work_types=("article",)))
    return books, articles, fp


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/wiki_400k"))
    ap.add_argument("--num-gaps", type=int, default=3)
    ap.add_argument("--radius-fraction", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=5, help="example titles per source per gap")
    # Subsample caps. The union ambient is also the QUERY, and kball coverage
    # does one single-threaded range search PER query point against the whole
    # ambient flat index (~50 ms/query at 400k), so the union size is the runtime
    # driver. Cap both sources for a tractable demo; 0 = no cap (full corpus).
    ap.add_argument("--oa-limit", type=int, default=8000,
                    help="max OpenAlex NLP articles in the union (0 = all ~87k)")
    ap.add_argument("--wiki-limit", type=int, default=8000,
                    help="max Wikipedia articles in the union (0 = all in store); "
                         "ignored when --topic-filter is set (the top-k selection caps it)")
    ap.add_argument("--topic-filter", action="store_true",
                    help="select NLP-relevant Wikipedia articles (highest cosine to the "
                         "mean-centered OpenAlex-NLP-article centroid) instead of a uniform "
                         "subsample. Reuses the embedded store; no re-embedding.")
    ap.add_argument("--wiki-topk", type=int, default=3000,
                    help="with --topic-filter: how many top-cosine Wikipedia articles to keep")
    ap.add_argument("--out", type=Path, default=Path("docs/wikipedia-openalex-shared-space.md"))
    args = ap.parse_args()

    # Determinism: single-threaded faiss (matches the parity gate's discipline).
    faiss.omp_set_num_threads(1)

    rng_sub = np.random.default_rng(args.seed)

    def _subsample(n: int, limit: int) -> NDArray[np.int64]:
        """Deterministic sorted subsample of row indices (identity if limit>=n or 0)."""
        if limit <= 0 or limit >= n:
            return np.arange(n, dtype=np.int64)
        return np.sort(rng_sub.choice(n, size=limit, replace=False)).astype(np.int64)

    print("Loading OpenAlex NLP slice (raw GIST) ...")
    oa_books, oa_articles, oa_fp = load_openalex_nlp()
    books_emb = _normalize(np.ascontiguousarray(oa_books.embeddings(), dtype=np.float32))
    arts_emb_full = _normalize(np.ascontiguousarray(oa_articles.embeddings(), dtype=np.float32))
    art_titles_full = oa_articles.work_titles()
    art_ids_full = oa_articles.work_ids()
    oa_sel = _subsample(arts_emb_full.shape[0], args.oa_limit)
    arts_emb = np.ascontiguousarray(arts_emb_full[oa_sel])
    art_titles_full = [art_titles_full[i] for i in oa_sel.tolist()]
    art_ids_full = [art_ids_full[i] for i in oa_sel.tolist()]
    print(f"  OA NLP books:    {books_emb.shape}")
    print(f"  OA NLP articles: {arts_emb.shape}  (subsampled from {arts_emb_full.shape[0]})")

    print("Loading Wikipedia corpus ...")
    wiki = WikipediaCorpus(args.wiki_store)
    wiki_emb_all = _normalize(np.ascontiguousarray(wiki.embeddings(), dtype=np.float32))
    wiki_titles_all = wiki.work_titles()
    wiki_ids_all = wiki.work_ids()
    wiki_fp = wiki.embedder_fingerprint
    assert oa_fp == wiki_fp, f"fingerprint mismatch: OA={oa_fp} wiki={wiki_fp}"
    print(f"  Wiki store: {wiki_emb_all.shape}, fp={wiki_fp}")

    # --- Mean-center each corpus, then re-normalize (the validated alignment) ----
    # Books and articles are both OpenAlex -> they share the OpenAlex centroid so
    # the (books -> articles) landmark geometry is preserved; only the
    # OpenAlex<->Wikipedia offset is removed. The OpenAlex centroid is taken over
    # the OpenAlex *articles* (the OA child of the union ambient). The Wikipedia
    # centroid is taken over the WHOLE store (the corpus-offset estimate must
    # reflect the full encyclopedic distribution, not a topical subset), and
    # topic-filter selection then happens on the already-centered Wikipedia rows.
    oa_centroid = arts_emb.mean(axis=0, keepdims=True)
    wiki_centroid = wiki_emb_all.mean(axis=0, keepdims=True)
    books_c = _normalize(books_emb - oa_centroid)
    arts_c = _normalize(arts_emb - oa_centroid)
    wiki_c_all = _normalize(wiki_emb_all - wiki_centroid)

    # --- Select which Wikipedia rows enter the union -----------------------------
    if args.topic_filter:
        # NLP-relevant selection: keep the wiki rows whose mean-centered embedding
        # has the highest cosine to the mean-centered OpenAlex-NLP-article centroid.
        # arts_c is already unit-norm; its mean is the centered NLP direction. We
        # L2-normalize that direction so the score is a true cosine, then rank.
        nlp_dir = arts_c.mean(axis=0)
        nlp_norm = np.linalg.norm(nlp_dir)
        nlp_dir = (nlp_dir / nlp_norm).astype(np.float32) if nlp_norm > 0 else nlp_dir
        wiki_cos = wiki_c_all @ nlp_dir  # cosine (both unit-norm)
        topk = min(args.wiki_topk, wiki_cos.shape[0])
        # argpartition for the top-k, then sort that slice by cosine descending.
        part = np.argpartition(-wiki_cos, topk - 1)[:topk]
        wiki_sel = part[np.argsort(-wiki_cos[part])].astype(np.int64)
        print(f"  Topic-filter: kept top-{topk} Wikipedia by cosine to NLP centroid "
              f"(cos range {wiki_cos[wiki_sel].min():.3f}..{wiki_cos[wiki_sel].max():.3f}).")
        print("  Example selected Wikipedia titles (top 15 by cosine):")
        for i in wiki_sel[:15].tolist():
            print(f"    [{wiki_cos[i]:.3f}] {wiki_titles_all[i][:80]}")
    else:
        wiki_sel = _subsample(wiki_emb_all.shape[0], args.wiki_limit)

    wiki_c = np.ascontiguousarray(wiki_c_all[wiki_sel])
    wiki_titles_full = [wiki_titles_all[i] for i in wiki_sel.tolist()]
    wiki_ids_full = [wiki_ids_all[i] for i in wiki_sel.tolist()]
    print(f"  Wiki in union: {wiki_c.shape}  "
          f"({'topic-filtered' if args.topic_filter else 'uniform subsample'} "
          f"from {wiki_emb_all.shape[0]})")

    # Bare ids inside each child; UnionCorpus.work_ids() applies the source
    # prefix derived from child .name() -> "openalex:" / "wiki:" (per task spec).
    oa_art_ids = tuple(art_ids_full)
    oa_art_titles = tuple(art_titles_full)
    wiki_ids = tuple(wiki_ids_full)
    wiki_titles = tuple(wiki_titles_full)

    sel_tag = ("topk", args.wiki_topk) if args.topic_filter else ("uniform", args.wiki_limit)
    sub_tag = (args.seed, args.oa_limit, sel_tag)
    landmark = MemCorpus(
        _embeddings=books_c, _work_ids=tuple(f"openalex:{w}" for w in oa_books.work_ids()),
        _work_titles=tuple(oa_books.work_titles()), _fingerprint=oa_fp,
        _name="openalex_books_centered",
        _identity=("oa_books_centered", NLP_TOPIC, oa_fp, sub_tag),
    )
    oa_art_corpus = MemCorpus(
        _embeddings=arts_c, _work_ids=oa_art_ids, _work_titles=oa_art_titles,
        _fingerprint=oa_fp, _name="openalex",
        _identity=("oa_arts_centered", NLP_TOPIC, oa_fp, sub_tag),
    )
    wiki_corpus = MemCorpus(
        _embeddings=wiki_c, _work_ids=wiki_ids, _work_titles=wiki_titles,
        _fingerprint=oa_fp, _name="wiki",
        _identity=("wiki_centered", str(args.wiki_store.resolve()), oa_fp, sub_tag),
    )

    union = UnionCorpus(children=(oa_art_corpus, wiki_corpus))
    union_ids = union.work_ids()       # already namespaced via child prefixes
    union_titles = union.work_titles()
    n_oa = len(oa_art_ids)
    print(f"  Union ambient: {union.embeddings().shape} "
          f"({n_oa} openalex + {len(wiki_ids)} wiki)")

    print(f"\nRunning detect_gaps (KBALL, num_gaps={args.num_gaps}, "
          f"radius_fraction={args.radius_fraction}) ...")
    gaps = detect_gaps(
        ambient=union, landmark=landmark, query=union,
        cell_type=CellType.KBALL, num_gaps=args.num_gaps,
        radius_fraction=args.radius_fraction,
        label_id="cross-corpus-nlp", label_name="cross-corpus NLP",
    )
    print(f"  detect_gaps returned {len(gaps)} gap(s).")

    report_lines: list[str] = []
    if args.topic_filter:
        report_lines.append("## Topic-filtered cross-corpus gaps (NLP-relevant Wikipedia ambient)\n")
        report_lines.append(
            "Follow-up to the uniform-sample run below: instead of a uniform Wikipedia "
            "subsample (dominated by off-topic stubs), the Wikipedia ambient is the "
            f"**top-{args.wiki_topk}** Wikipedia articles by cosine to the mean-centered "
            "OpenAlex-NLP-article centroid (reusing the already-embedded store; no "
            "re-embedding). Everything else is identical: OpenAlex NLP books are the "
            "landmarks, the ambient/query is a `UnionCorpus` of OpenAlex NLP articles + "
            "the NLP-relevant Wikipedia slice, all raw 384-D GIST, each corpus "
            "mean-centered then re-normalized, faiss pinned to one thread.\n"
        )
    else:
        report_lines.append("## Cross-corpus gap detection (OpenAlex books vs OpenAlex+Wikipedia ambient)\n")
        report_lines.append(
            "SP3 capstone: the gap algorithms now accept any `Corpus`, so `detect_gaps` "
            "runs with a cross-corpus `UnionCorpus` ambient. Landmarks are OpenAlex NLP "
            f"books (topic `{NLP_TOPIC}`); the ambient/query is a `UnionCorpus` of OpenAlex "
            "NLP articles + a Wikipedia slice, all in raw 384-D GIST. Each corpus is "
            "**mean-centered then re-normalized before the union** (the validated alignment: "
            "the OpenAlex<->Wikipedia shift is a pure per-corpus offset). faiss is pinned to "
            "one thread.\n"
        )
    report_lines.append("### Setup\n")
    report_lines.append(f"- **Landmark:** OpenAlex NLP books, N={books_c.shape[0]}.")
    wiki_desc = (
        f"top-{args.wiki_topk} NLP-relevant" if args.topic_filter else "uniform-sampled"
    )
    report_lines.append(
        f"- **Ambient/query:** `UnionCorpus` of OpenAlex NLP articles (N={n_oa}) + "
        f"{wiki_desc} Wikipedia `{args.wiki_store}` (N={len(wiki_ids)}); "
        f"total N={union.embeddings().shape[0]}."
    )
    report_lines.append(
        f"- Each corpus mean-centered (OpenAlex centroid for books+articles, "
        f"Wikipedia centroid for wiki) then L2-renormalized. "
        f"`CellType.KBALL`, `num_gaps={args.num_gaps}`, `radius_fraction={args.radius_fraction}`, "
        f"single-thread faiss.\n"
    )

    report_lines.append("### Result\n")
    report_lines.append(f"- `detect_gaps` ran end-to-end and returned **{len(gaps)} gap(s)**.")
    report_lines.append("")

    rng = np.random.default_rng(args.seed)
    for gwc in gaps:
        members = np.asarray(gwc.gap.get_dists_idxs().idxs, dtype=np.int64)
        is_oa = members < n_oa
        n_oa_in = int(is_oa.sum())
        n_wiki_in = int((~is_oa).sum())
        print(f"\n=== Gap {gwc.gap_idx}: size={gwc.gap_size}  "
              f"(openalex={n_oa_in}, wiki={n_wiki_in})  "
              f"percentile={gwc.gap_size_percentile:.3f} ===")
        report_lines.append(
            f"#### Gap {gwc.gap_idx} — size {gwc.gap_size} "
            f"(percentile {gwc.gap_size_percentile:.3f})\n"
        )
        report_lines.append(
            f"- Membership: **{n_oa_in} OpenAlex** + **{n_wiki_in} Wikipedia** "
            f"= {gwc.gap_size} in-gap points.\n"
        )

        oa_members = members[is_oa]
        wiki_members = members[~is_oa]

        def _examples(idxs: NDArray[np.int64], k: int) -> list[str]:
            if len(idxs) == 0:
                return []
            pick = idxs if len(idxs) <= k else idxs[rng.choice(len(idxs), size=k, replace=False)]
            return [f"{union_ids[i]}  —  {union_titles[i][:90]}" for i in pick.tolist()]

        oa_ex = _examples(oa_members, args.n_examples)
        wiki_ex = _examples(wiki_members, args.n_examples)
        if oa_ex:
            report_lines.append("- Example OpenAlex members:")
            for e in oa_ex:
                report_lines.append(f"  - {e}")
                print(f"    [OA]   {e}")
        if wiki_ex:
            report_lines.append("- Example Wikipedia members:")
            for e in wiki_ex:
                report_lines.append(f"  - {e}")
                print(f"    [WIKI] {e}")
        report_lines.append("")

    # Append the section to the shared-space doc.
    existing = args.out.read_text() if args.out.exists() else ""
    sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
    args.out.write_text(existing + sep + "\n".join(report_lines) + "\n")
    print(f"\nAppended cross-corpus section to {args.out}")


if __name__ == "__main__":
    main()
