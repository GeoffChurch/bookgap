"""Cross-method gap comparison table generator.

REFERENCE IMPLEMENTATION for cross-method comparison artifacts. New
comparison scripts (loading two gap CSVs, computing keywords with the
production log-odds + lattice-bigram pipeline, looking up nearest-book
votes, emitting markdown) should copy from this file. See
``docs/gap-method-development.md`` for the comparison-table pattern.

Compares top-20 gaps from raw K-ball (size-ranked) vs cohesion-aware
article-as-seed (logit-ranked, with min_post >= 0.5 boundary), side by
side in markdown form. For each gap:

- 5 distinguishing keywords via log-odds-ratio (Fightin' Words style,
  marginal-background + freq-weighting), with bigram phrase mining as
  in the curriculum production pipeline.
- Top-3 nearest books (by article-vote frequency).
- Gap size and key cohesion metrics (article-as-seed only).
- 2 example article titles.

Usage::

    python -m artifacts_scripts.gap_method_comparison_table \\
        --scope 17/1702/T10181 \\
        --kball-csv artifacts/kball_gap_probe/17-1702-T10181__pca__drop0.csv \\
        --article-seed-csv artifacts/gap_article_as_seed/T10181__article_seed.csv \\
        --top-n 20
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from artifacts_scripts.curriculum.keyword_extraction import (
    extract_topic_keywords,
    filter_stopwords,
    mine_phrases_lattice,
    parse_with_phrases,
    tokenize,
    KeywordMethod,
    DocWeighting,
    BackgroundMode,
    TopicWeighting,
)
from artifacts_scripts.hdbscan_gap_probe import load_scope
from bookgap.index import build_index, FlatL2


def _load_kball(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "gap_id": int(r["gap_idx"]),
                "size": int(r["gap_size"]),
                "members": json.loads(r["article_indices_json"]),
            })
    return rows


def _load_article_seed(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "rank": int(r["rank"]),
                "seed": int(r["seed_article"]),
                "size": int(r["final_size"]),
                "logit_sum": float(r["logit_sum"]),
                "mean_post": float(r["mean_post"]),
                "min_post": float(r["min_post"]),
                "members": json.loads(r["article_indices_json"]),
            })
    return rows


def _load_articles_parquet(scope: str) -> tuple[list[str], list[str]]:
    """Return (titles, abstracts) for articles in the scope, indexed by
    the same global position as gap_article_as_seed/K-ball CSVs use."""
    parts = scope.split("/")
    parquet_dir = Path("cache/works_parquet").joinpath(*parts)
    article_pq = parquet_dir / "article.parquet"
    if not article_pq.exists():
        raise FileNotFoundError(f"No article.parquet at {article_pq}")
    tbl = pq.read_table(article_pq, columns=["display_name", "abstract"])
    titles = tbl.column("display_name").to_pylist()
    abstracts = tbl.column("abstract").to_pylist()
    return titles, abstracts


def _gap_to_raw_tokens(members: list[int], titles, abstracts) -> list[list[str]]:
    """Per-article tokenize (no stopword filter yet — bigram mining
    needs raw token sequences)."""
    out = []
    for m in members:
        if m >= len(titles):
            continue
        text_parts = []
        if titles[m]:
            text_parts.append(titles[m])
        if abstracts[m]:
            text_parts.append(abstracts[m])
        text = " ".join(text_parts)
        toks = tokenize(text)
        if toks:
            out.append(toks)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", required=True)
    p.add_argument("--kball-csv", required=True)
    p.add_argument("--article-seed-csv", required=True)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--keywords-per-gap", type=int, default=5)
    p.add_argument("--titles-per-gap", type=int, default=2)
    p.add_argument("--bigram-iterations", type=int, default=2,
                   help="phrase-mining iterations; production default is 2 "
                        "(trigrams via bigram-of-bigram); higher surfaces longer phrases")
    p.add_argument("--output-md", default="docs/gap_method_comparison.md")
    args = p.parse_args()

    print(f"Loading article titles+abstracts for scope {args.scope}...")
    titles, abstracts = _load_articles_parquet(args.scope)
    print(f"  {len(titles):,} articles loaded")

    # Also load full scope for book lookups (need book embeddings + titles).
    print(f"Loading scope embeddings for nearest-book lookup...")
    emb, ids, types, all_titles, _topic_ids, _refs = load_scope(args.scope, with_refs=False)
    types_arr = np.array(types)
    is_book = types_arr == "book"
    book_global_idx = np.where(is_book)[0]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    emb = np.ascontiguousarray(emb)
    book_emb = np.ascontiguousarray(emb[book_global_idx])
    book_index = build_index(FlatL2(), book_emb)
    all_titles_arr = np.array(all_titles, dtype=object)

    print(f"Loading K-ball gaps (sorted by size descending)...")
    kball_all = _load_kball(Path(args.kball_csv))
    kball_top = sorted(kball_all, key=lambda r: -r["size"])[:args.top_n]

    print(f"Loading article-as-seed gaps (already sorted by logit_sum)...")
    seed_all = _load_article_seed(Path(args.article_seed_csv))
    seed_top = sorted(seed_all, key=lambda r: -r["logit_sum"])[:args.top_n]

    # Pass 1: tokenize all docs across all 40 gaps (raw, no stopword filter).
    print("Tokenizing gap members...")
    raw_topic_tokens: list[list[list[str]]] = []
    for g in kball_top:
        raw_topic_tokens.append(_gap_to_raw_tokens(g["members"], titles, abstracts))
    for g in seed_top:
        raw_topic_tokens.append(_gap_to_raw_tokens(g["members"], titles, abstracts))

    # Pass 2: lattice phrase mining on the joint corpus (matches the
    # curriculum production pipeline's approach: bigram_iterations=2,
    # re-parse from raw with longest-match each iteration so trigrams
    # surface as merged-bigram-of-bigram).
    flat_raw = [doc for topic_docs in raw_topic_tokens for doc in topic_docs]
    print(f"Mining phrases ({args.bigram_iterations} iterations, lattice longest-match) "
          f"from {len(flat_raw)} docs...")
    phrase_vocab: set = set()
    parsed = flat_raw
    import time
    for it in range(args.bigram_iterations):
        t0 = time.time()
        new_phrases = mine_phrases_lattice(
            parsed,
            npmi_threshold=0.5,
            min_boundary_be=0.5,
        )
        t_mine = time.time() - t0
        if not new_phrases:
            print(f"  iter {it+1}: 0 new phrases mined ({t_mine:.1f}s); stopping")
            break
        prev_vocab = len(phrase_vocab)
        phrase_vocab |= new_phrases
        added = len(phrase_vocab) - prev_vocab
        t1 = time.time()
        parsed = [parse_with_phrases(doc, phrase_vocab) for doc in flat_raw]
        t_parse = time.time() - t1
        print(f"  iter {it+1}: +{added} phrases ({t_mine:.1f}s mine + {t_parse:.1f}s reparse), "
              f"total vocab = {len(phrase_vocab)}")
    print(f"  final phrase vocab: {len(phrase_vocab)} multi-word terms")

    # Distribute parsed docs back per topic
    merged_per_topic: list[list[list[str]]] = []
    offset = 0
    for topic_docs in raw_topic_tokens:
        n = len(topic_docs)
        merged_per_topic.append(parsed[offset : offset + n])
        offset += n

    # Pass 3: stopword filter
    topic_doc_tokens: list[list[list[str]]] = []
    for topic_docs in merged_per_topic:
        kept = []
        for tokens in topic_docs:
            tokens = filter_stopwords(tokens)
            if tokens:
                kept.append(tokens)
        topic_doc_tokens.append(kept)

    # Pass 4: log-odds keyword extraction (production default)
    print("Extracting keywords (log-odds, marginal-bg, freq-weighted)...")
    keywords = extract_topic_keywords(
        topic_doc_tokens,
        method=KeywordMethod.LOG_ODDS,
        top_k=args.keywords_per_gap,
        doc_weighting=DocWeighting.EQUAL,
        topic_weighting=TopicWeighting.POOLED,
        background_mode=BackgroundMode.MARGINAL,
        weight_by_freq=True,
    )

    kball_kws = keywords[:len(kball_top)]
    seed_kws = keywords[len(kball_top):]

    # Compute top-3 nearest books per gap (by article-vote frequency).
    print("Computing nearest books per gap...")
    def _top_books(members: list[int], k: int = 3) -> list[tuple[str, int]]:
        if not members:
            return []
        # The gap members are scope-global indices; pull their embeddings
        # and find the 1-NN book of each, then count.
        from collections import Counter
        gap_emb = np.ascontiguousarray(emb[members])
        _, I_b = book_index.search(gap_emb, k=1)
        nearest_books_global = book_global_idx[I_b[:, 0]]
        cnt = Counter(nearest_books_global.tolist())
        result = []
        for bi, c in cnt.most_common(k):
            t = all_titles_arr[bi] or "(no title)"
            result.append((str(t), c))
        return result

    kball_books = [_top_books(g["members"]) for g in kball_top]
    seed_books = [_top_books(g["members"]) for g in seed_top]

    def _kw_str(kw_list, k: int) -> str:
        return ", ".join(w for w, _ in kw_list[:k])

    def _title_lines(g, k: int) -> str:
        ts = []
        for m in g["members"][:k]:
            t = titles[m] if m < len(titles) else "(no title)"
            t = (t or "").replace("|", "\\|").replace("\n", " ").strip()
            ts.append(f"  • {t[:90]}")
        return "<br>".join(ts)

    def _books_str(book_list: list) -> str:
        return "<br>".join(
            f"  • [×{c}] {t.replace('|', chr(92) + '|')[:80]}"
            for t, c in book_list
        )

    out = []
    out.append(f"# Gap-detection method comparison: {args.scope}")
    out.append("")
    out.append("Top-20 gaps from each pipeline, side by side. K-ball is")
    out.append("the original size-ranked baseline; article-as-seed uses")
    out.append("the cohesion-aware HNSW pipeline (moving-centroid, strict")
    out.append("policy, min_post≥0.5 boundary, logit_sum ranking).")
    out.append("Keywords use log-odds-ratio (Fightin' Words) with marginal")
    out.append("background and freq-weighting, with bigram phrase mining,")
    out.append("matching the curriculum production pipeline. Top books are")
    out.append("the most-voted nearest book per gap article (1-NN).")
    out.append("")

    out.append("## K-ball (size-ranked)")
    out.append("")
    out.append("| rank | size | keywords | top books (×votes) | example titles |")
    out.append("|------|------|----------|-------------------|----------------|")
    for i, g in enumerate(kball_top):
        out.append(
            f"| {i+1} | {g['size']} | {_kw_str(kball_kws[i], args.keywords_per_gap)} | "
            f"{_books_str(kball_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
        )
    out.append("")

    out.append("## Article-as-seed (logit_sum-ranked, cohesion-aware)")
    out.append("")
    out.append("| rank | size | mean_p / min_p | logit | keywords | top books (×votes) | example titles |")
    out.append("|------|------|----------------|-------|----------|-------------------|----------------|")
    for i, g in enumerate(seed_top):
        out.append(
            f"| {i+1} | {g['size']} | {g['mean_post']:.2f} / {g['min_post']:.2f} | "
            f"{g['logit_sum']:.0f} | {_kw_str(seed_kws[i], args.keywords_per_gap)} | "
            f"{_books_str(seed_books[i])} | {_title_lines(g, args.titles_per_gap)} |"
        )
    out.append("")

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
