#!/usr/bin/env python
# artifacts_scripts/wiki/wiki_gap_spike.py
"""DECISIVE SPIKE — "OA-dense, Wikipedia-MISSING" research topics, two detectors.

Question: detect research concepts well-covered in OpenAlex (OA) but with NO
corresponding Wikipedia article, bounded to ONE subfield (NLP, topic T10181 /
subfield 1702), with TWO detectors, then ask the LIVE Wikipedia API which gaps
are genuine missing articles.

Detector #1 — NEURAL density gap (reuse, not rebuilt).
    The cross-corpus k-ball over a UnionCorpus(OA-NLP-articles + topic-filtered
    Wikipedia), per-corpus mean-center+renorm, OA-NLP *books* as landmarks. This
    is exactly ``cross_corpus_gap_demo.py``'s machinery (imported, not copied):
    same ``load_openalex_nlp``, same centering, same ``--topic-filter`` Wikipedia
    selection, same ``detect_gaps(KBALL)``. A gap = a dense book-free region; we
    keep the gaps with OA membership (OA-dense) and report their distinctive
    terms. This detector sees the wiki<->OA register residual (it operates in the
    shared embedding geometry), which is the hypothesis under test.

Detector #2 — EMBEDDER-FREE has_wiki-deficit (the star).
    The backlog "book-deficit lift" analog (docs/backlog/
    curriculum-keyword-concept-seeding.md): build OA concepts over the NLP OA
    works, extract LIFT-DISTINCTIVE keywords per concept
    (``compute_log_odds_ratio_docs``, Fightin'-Words, BackgroundMode.MARGINAL,
    weight_by_freq=True — the validated distinctiveness primitive, PREFERRED over
    tf-idf), then for each concept compute a ``has_wiki`` attribute: does the
    topic-filtered Wikipedia corpus contain an article whose TITLE/terms match the
    concept's distinctive terms (matched on TERMS, not embedding geometry — the
    register-robust point)? has_wiki lift < 1 (wiki-deficient relative to the
    NLP-concept base rate of wiki coverage) = an embedder-free gap.

    REGISTER-NEUTRALIZER: terms are scored PER-CORPUS — OA concept terms are
    distinctive vs the OA-NLP marginal; the Wikipedia title index is built from
    Wikipedia's own text. Each corpus is compared against its own background;
    the only cross-corpus operation is a string match on distinctive terms,
    which carries no register geometry. (Precedent: the ``--per-corpus-select``
    z-norm in joint_graph_curriculum.py:1149.)

Concept FORMATION uses embedding clustering (fast, validated; within-OA, so the
cross-corpus register residual does not enter). The GAP SIGNAL (has_wiki, term-
matched) is the embedder-free part — exactly as the backlog defines book-deficit
lift: the *attribute lift* is the gap, regardless of how the concept was formed.

VALIDATION oracle = Wikipedia's live inventory. For the top ~15 gaps from EACH
detector, query the live Wikipedia API (opensearch + search, read-only, polite
UA + rate-limit) on the gap's distinctive terms, and classify each:
    REAL-GAP        — no focused article exists for the concept
    ALREADY-COVERED — an article exists (maybe under another title)
    TOO-NICHE       — no encyclopedia would carry it
API responses are cached under cache/wiki/api_cache/ so re-runs are free.

Usage::

    uv run python -m artifacts_scripts.wiki.wiki_gap_spike \
        --wiki-store cache/wiki/store/wiki_400k \
        --oa-limit 8000 --wiki-topk 8000 --n-gaps 15 \
        --radius-fraction 0.75 --seed 0

Provenance footer at end of file. Regenerate rather than edit outputs by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Optional

import faiss
import numpy as np
from numpy.typing import NDArray

# --- detector #1 machinery (reuse, do not rebuild) ---------------------------
from artifacts_scripts.wiki.cross_corpus_gap_demo import (
    MemCorpus,
    NLP_TOPIC,
    _normalize,
    load_openalex_nlp,
)
from bookgap.corpus import UnionCorpus
from bookgap.gaps import detect_gaps
from bookgap.types import CellType
from bookgap.wikipedia.corpus import WikipediaCorpus

# --- detector #2 primitive (validated distinctiveness) -----------------------
from curriculum.layers._keyword_extraction import (
    BackgroundMode,
    DocWeighting,
    TopicWeighting,
    compute_log_odds_ratio_docs,
    filter_stopwords,
    mine_phrases_lattice,
    parse_with_phrases,
    tokenize,
)

API_CACHE_DIR = Path("cache/wiki/api_cache")
OA_SLICE_NPZ = Path("cache/wiki/oa_nlp_slice_spike.npz")  # our own tidy cache
USER_AGENT = (
    "bookgap-research-spike/1.0 (geoffrey.a.churchill@gmail.com; "
    "read-only Wikipedia inventory check)"
)


# =============================================================================
# Shared: OA NLP slice loader (cached npz so re-runs skip the cloud entirely)
# =============================================================================
def load_oa_nlp_cached() -> dict:
    """Return dict with books_emb, books_titles, arts_emb, arts_titles, arts_ids,
    arts_abstracts, fp. Loads via the validated ``load_openalex_nlp`` once, then
    caches to a small npz (the abstracts make it ~a few hundred MB; tidy)."""
    if OA_SLICE_NPZ.exists():
        d = np.load(OA_SLICE_NPZ, allow_pickle=True)
        return {
            "books_emb": d["books_emb"],
            "books_titles": list(d["books_titles"]),
            "arts_emb": d["arts_emb"],
            "arts_titles": list(d["arts_titles"]),
            "arts_ids": list(d["arts_ids"]),
            "arts_abstracts": list(d["arts_abstracts"]),
            "fp": str(d["fp"]),
        }
    print("Loading OpenAlex NLP slice via cloud (one-time; caching to npz) ...", flush=True)
    books, articles, fp = load_openalex_nlp()
    out = {
        "books_emb": np.ascontiguousarray(books.embeddings(), dtype=np.float32),
        "books_titles": list(books.work_titles()),
        "arts_emb": np.ascontiguousarray(articles.embeddings(), dtype=np.float32),
        "arts_titles": list(articles.work_titles()),
        "arts_ids": list(articles.work_ids()),
        "arts_abstracts": list(articles.work_abstracts()),
        "fp": fp,
    }
    OA_SLICE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OA_SLICE_NPZ,
        books_emb=out["books_emb"],
        books_titles=np.asarray(out["books_titles"], dtype=object),
        arts_emb=out["arts_emb"],
        arts_titles=np.asarray(out["arts_titles"], dtype=object),
        arts_ids=np.asarray(out["arts_ids"], dtype=object),
        arts_abstracts=np.asarray(out["arts_abstracts"], dtype=object),
        fp=fp,
    )
    print(f"  cached OA NLP slice -> {OA_SLICE_NPZ} "
          f"({OA_SLICE_NPZ.stat().st_size/1e6:.0f} MB)", flush=True)
    return out


# =============================================================================
# Distinctive terms (the embedder-free distinctiveness primitive)
# =============================================================================
from functools import partial as _partial


def _pmap(fn, items, n_jobs):
    """Order-preserving parallel map; serial fallback for n_jobs==1 or small inputs
    (multiprocessing's pickling overhead isn't worth it under a few thousand items)."""
    if n_jobs == 1 or len(items) < 4000:
        return [fn(x) for x in items]
    import os
    from multiprocessing import Pool
    ncpu = os.cpu_count() or 2
    n = ncpu if n_jobs in (-1, 0) else n_jobs
    n = max(1, min(n, ncpu))
    with Pool(n) as pool:
        return pool.map(fn, items, chunksize=max(1, len(items) // (n * 8)))


def _grouped_pmap(fn, groups, n_jobs):
    """Apply fn to every element across all groups in parallel; preserve group structure."""
    lens = [len(g) for g in groups]
    flat = [x for g in groups for x in g]
    out = _pmap(fn, flat, n_jobs)
    res, i = [], 0
    for L in lens:
        res.append(out[i:i + L])
        i += L
    return res


def distinctive_terms_for_groups(
    group_docs: list[list[str]],   # for each group, its member texts (title+abstract)
    *,
    top_k: int = 12,
    mine_phrases: bool = True,
    mine_sample: int = 12000,
    min_count: int = 3,
    seed: int = 0,
    n_jobs: int = 1,
) -> list[list[tuple[str, float]]]:
    """LIFT-distinctive terms per group via Fightin'-Words (MARGINAL background,
    weight_by_freq=True). PHRASE-mined first (the validated curriculum chain:
    tokenize -> iterative ``mine_phrases_lattice`` -> ``parse_with_phrases`` ->
    ``filter_stopwords`` -> FW), so distinctive terms include multi-word phrases
    (``machine_translation``, ``named_entity``) that can match Wikipedia titles.
    Each group is scored vs the pooled marginal (its own corpus prior).

    Scalability: the iterative phrase *mining* (the O(N·iters) step) runs on a
    deterministic SAMPLE of ``mine_sample`` docs (a phrase like
    ``machine_translation`` recurs in many docs, so a sample discovers the same
    vocabulary); the discovered phrase vocab then parses ALL docs once (O(N), the
    longest-match trie pass), and FW scoring uses the full corpus. This keeps the
    full-corpus distinctiveness while bounding the expensive mining loop."""
    # Pass 1: raw tokens per doc (NO stopword filter — phrase mining needs raw
    # sequences so phrases aren't merged across removed stopwords).
    group_raw: list[list[list[str]]] = _grouped_pmap(tokenize, group_docs, n_jobs)
    group_raw = [[d for d in docs if d] for docs in group_raw]

    if mine_phrases:
        flat = [d for docs in group_raw for d in docs]
        # mining corpus = deterministic sample (bounds the iterative re-parse cost)
        if len(flat) > mine_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(flat), size=mine_sample, replace=False)
            mine_corpus = [flat[i] for i in idx.tolist()]
        else:
            mine_corpus = flat
        phrase_vocab: set[str] = set()
        parsed_mine = mine_corpus
        for _ in range(3):  # iterative exclusive-entropy re-parse (curriculum default)
            new = mine_phrases_lattice(parsed_mine, npmi_threshold=0.4, min_boundary_be=0.5,
                                       count_floor=3, max_phrases=5000)
            new -= phrase_vocab
            if not new:
                break
            phrase_vocab |= new
            parsed_mine = _pmap(_partial(parse_with_phrases, phrases=phrase_vocab), mine_corpus, n_jobs)
        # parse ALL docs once with the discovered vocab (O(N) longest-match) — parallel
        group_raw = _grouped_pmap(_partial(parse_with_phrases, phrases=phrase_vocab), group_raw, n_jobs)

    # Final: stopword-filter (merged phrases use "_" and survive) — parallel, one call/doc.
    _filt = _grouped_pmap(filter_stopwords, group_raw, n_jobs)
    topic_doc_tokens: list[list[list[str]]] = [[d for d in docs if d] for docs in _filt]
    # Vocabulary prune (speed): drop terms with total corpus count < min_count.
    # FW's cost is O(|vocab| × |topics|) in Python; hapax/rare terms can never be
    # top distinctive (weight_by_freq=True multiplies by in-topic count, so a
    # count<3 term is crushed) — pruning them shrinks the vocab ~10x with no effect
    # on the top-k. This is a pure performance prune, not a behavior change to the
    # validated primitive.
    if min_count > 1:
        gc: Counter[str] = Counter()
        for docs in topic_doc_tokens:
            for d in docs:
                gc.update(d)
        keep = {w for w, c in gc.items() if c >= min_count}
        topic_doc_tokens = [
            [[t for t in d if t in keep] for d in docs] for docs in topic_doc_tokens
        ]
        topic_doc_tokens = [[d for d in docs if d] for docs in topic_doc_tokens]
    return compute_log_odds_ratio_docs(
        topic_doc_tokens,
        top_k=top_k,
        doc_weighting=DocWeighting.EQUAL,
        topic_weighting=TopicWeighting.EQUAL,
        background_mode=BackgroundMode.MARGINAL,
    )


# =============================================================================
# Wikipedia TERM/TITLE index (embedder-free has_wiki matching)
# =============================================================================
@dataclass
class WikiTitleIndex:
    """Term/title index over the topic-filtered Wikipedia slice for embedder-free
    has_wiki matching. Holds:
      - title_token_sets: per-article set of title tokens (for term-overlap match)
      - title_norm: per-article normalized title string (for phrase match)
      - token_postings: token -> set of article row indices (inverted index)
    """
    titles: list[str]
    title_token_sets: list[frozenset[str]]
    title_norm: list[str]
    token_postings: dict[str, set[int]]
    # exact single-token Wikipedia titles, normalized -> the original title
    single_title: dict[str, str]

    @classmethod
    def build(cls, titles: list[str]) -> "WikiTitleIndex":
        tok_sets: list[frozenset[str]] = []
        norm: list[str] = []
        postings: dict[str, set[int]] = {}
        single_title: dict[str, str] = {}
        for i, t in enumerate(titles):
            toks = set(filter_stopwords(tokenize(t)))
            tok_sets.append(frozenset(toks))
            tn = _norm_phrase(t)
            norm.append(tn)
            for tok in toks:
                postings.setdefault(tok, set()).add(i)
            # a 1-content-word title (e.g. "Parsing", "Lemmatisation"): index its
            # single normalized token so a distinctive unigram can match it exactly.
            if len(toks) == 1:
                single_title.setdefault(next(iter(toks)), t)
        return cls(titles=titles, title_token_sets=tok_sets, title_norm=norm,
                   token_postings=postings, single_title=single_title)

    def has_match(self, terms: list[str], *, top_unigrams: int = 4) -> tuple[bool, Optional[str]]:
        """Embedder-free has_wiki test for a concept's distinctive terms.

        A concept is "covered" by the Wikipedia landmark slice if ANY of:
          (a) a distinctive multi-word phrase (``machine_translation``) appears as,
              within, or as a token-subset of a Wikipedia title (e.g. concept term
              ``named_entity_recognition`` -> article "Named-entity recognition");
          (b) a PAIR of the concept's top distinctive unigrams BOTH appear in one
              Wikipedia title (e.g. {parsing, dependency} -> "Dependency parsing");
          (c) a TOP distinctive unigram IS itself a Wikipedia article title (a
              1-word title — e.g. ``parsing`` -> "Parsing", ``segmentation`` ->
              "Segmentation"): the natural "has_wiki" sense for a broad concept
              whose head term titles an article.
        A single GENERIC head token can't trigger (a)/(b); (c) only fires on a term
        that the FW lift already certified as distinctive AND that exactly titles an
        article. The match is on TERMS, never embedding geometry (register-robust).

        Returns (matched, matched_title_or_None).
        """
        unigrams: list[str] = []
        for term in terms:
            parts = [p for p in term.split("_") if len(p) >= 3]
            # (a) multi-word phrase containment / token-subset
            if len(parts) >= 2:
                pn = _norm_phrase(" ".join(parts))
                for i in self._candidates(parts):
                    if pn and pn in self.title_norm[i]:
                        return True, self.titles[i]
                    if set(parts).issubset(self.title_token_sets[i]):
                        return True, self.titles[i]
            else:
                unigrams.extend(parts)
        top = unigrams[:top_unigrams]
        # (c) a top distinctive unigram exactly titles a (1-word) Wikipedia article
        for u in top:
            if u in self.single_title:
                return True, self.single_title[u]
        # (b) any pair of the top distinctive unigrams co-occurring in one title
        for a in range(len(top)):
            for b in range(a + 1, len(top)):
                cand = self._candidates([top[a], top[b]])
                for i in cand:
                    if {top[a], top[b]}.issubset(self.title_token_sets[i]):
                        return True, self.titles[i]
        return False, None

    def _candidates(self, parts: list[str]) -> set[int]:
        postings = [self.token_postings.get(p, set()) for p in parts]
        postings = [p for p in postings if p]
        if not postings:
            return set()
        # intersection over all parts present (title containing all phrase tokens)
        out = set(min(postings, key=len))
        for p in postings:
            out &= p
        return out


_PHRASE_NONWORD = re.compile(r"[^a-z0-9]+")


def _norm_phrase(s: str) -> str:
    return _PHRASE_NONWORD.sub(" ", (s or "").lower()).strip()


# =============================================================================
# Live Wikipedia API oracle (read-only, cached, rate-limited)
# =============================================================================
_LAST_CALL = [0.0]


def _rate_limit(min_interval: float = 0.5) -> None:
    dt = time.time() - _LAST_CALL[0]
    if dt < min_interval:
        time.sleep(min_interval - dt)
    _LAST_CALL[0] = time.time()


def _api_get(params: dict) -> dict:
    """Cached GET against en.wikipedia.org/w/api.php. Read-only."""
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = json.dumps(params, sort_keys=True)
    import hashlib

    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    cache_f = API_CACHE_DIR / f"{h}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    _rate_limit()
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    cache_f.write_text(json.dumps(data))
    return data


def wiki_search(query: str, limit: int = 5) -> list[dict]:
    """Full-text search; returns [{title, snippet}]. Cached."""
    data = _api_get({
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": limit, "format": "json", "srprop": "snippet",
    })
    return data.get("query", {}).get("search", [])


def wiki_opensearch(query: str, limit: int = 5) -> list[str]:
    """Prefix/opensearch titles. Cached."""
    data = _api_get({
        "action": "opensearch", "search": query, "limit": limit,
        "namespace": 0, "format": "json",
    })
    return data[1] if isinstance(data, list) and len(data) > 1 else []


def _title_is_focused_match(title: str, terms_words: list[str]) -> bool:
    """Heuristic: a returned article TITLE counts as a focused match for the
    concept if its normalized title shares >=2 content words with the concept's
    distinctive words, OR contains a full multi-word distinctive phrase. Conserva-
    tive: a generic 1-word overlap (e.g. only "model") does not count."""
    tnorm = set(_norm_phrase(title).split())
    cwords = set(w for w in terms_words if len(w) >= 3)
    inter = tnorm & cwords
    return len(inter) >= 2


def classify_gap_via_oracle(
    distinctive_terms: list[str],
    rep_titles: list[str],
) -> dict:
    """Query the live Wikipedia API on the gap's distinctive terms and representa-
    tive OA titles; return a verdict dict.

    Strategy:
      1. Build query strings from the top multi-word distinctive terms (spaces).
      2. opensearch (title prefix) + full-text search for each.
      3. If a near-exact title match exists for the concept phrase -> ALREADY-COVERED.
      4. Else if full-text search returns articles whose titles focus on the
         concept (>=2 distinctive content-word overlap) -> ALREADY-COVERED.
      5. Else REAL-GAP (no focused article), unless the distinctive terms are
         degenerate (place-name/disambiguation/too-generic) -> TOO-NICHE.
    """
    # words from underscore terms (for overlap test), and phrase queries
    words: list[str] = []
    phrases: list[str] = []
    for term in distinctive_terms:
        parts = [p for p in term.split("_") if len(p) >= 3]
        words.extend(parts)
        if len(parts) >= 2:
            phrases.append(" ".join(parts))
    # de-dup, keep order
    seen = set()
    words = [w for w in words if not (w in seen or seen.add(w))]
    seen = set()
    phrases = [p for p in phrases if not (p in seen or seen.add(p))]

    hits: list[dict] = []
    verdict = "REAL-GAP"
    reason = ""
    matched_title = None

    # Queries: the top distinctive multi-word PHRASES (specific concept names) AND
    # the top individual distinctive TERMS (so a broad concept whose head term
    # titles an article — "parsing" -> "Parsing" — is correctly found). Plus a
    # joined-top-words query as a fallback. The oracle is the fair "does a focused
    # article exist ANYWHERE" judge, so we cast the net wide on the concept's own
    # distinctive vocabulary.
    queries: list[str] = []
    queries.extend(phrases[:3])
    # individual top distinctive single words (the FW lift already certified them)
    queries.extend(words[:3])
    if not queries:
        queries = [" ".join(words[:4])]
    # de-dup, keep order, drop empties
    seenq = set()
    queries = [q for q in queries if q.strip() and not (q in seenq or seenq.add(q))]

    exact_cover = False
    focused_cover = False
    for q in queries:
        os_titles = wiki_opensearch(q, limit=5)
        srch = wiki_search(q, limit=5)
        hit = {
            "query": q,
            "opensearch_titles": os_titles,
            "search_titles": [s.get("title", "") for s in srch],
        }
        hits.append(hit)
        qn = _norm_phrase(q)
        # exact-ish: an opensearch/search title equals or tightly contains the query
        for ot in os_titles + [s.get("title", "") for s in srch]:
            otn = _norm_phrase(ot)
            if qn and (qn == otn or (" " in qn and qn in otn)):
                exact_cover = True
                matched_title = ot
        # focused: a search-result title shares >=2 content words with the concept
        for s in srch:
            st = s.get("title", "")
            if _title_is_focused_match(st, words):
                focused_cover = True
                if matched_title is None:
                    matched_title = st

    if exact_cover:
        verdict, reason = "ALREADY-COVERED", f"title match: {matched_title!r}"
    elif focused_cover:
        verdict, reason = "ALREADY-COVERED", f"focused article: {matched_title!r}"
    else:
        # Distinguish REAL-GAP from TOO-NICHE by inspecting term character.
        if _looks_too_niche(words):
            verdict, reason = "TOO-NICHE", "terms read as proper-noun/place/disambiguation or too generic"
        else:
            verdict, reason = "REAL-GAP", "no focused Wikipedia article found for the concept phrase"
    return {
        "verdict": verdict,
        "reason": reason,
        "matched_title": matched_title,
        "queries": queries,
        "hits": hits,
    }


_PLACE_HINTS = {"county", "township", "village", "city", "river", "lake", "district",
                "kansas", "georgia", "idaho", "texas", "michigan", "pennsylvania",
                "ohio", "iowa", "minnesota"}

# Non-English function words observed dominating language-cluster gaps (Finnish,
# Estonian, Swedish, Dutch, German). A concept whose DISTINCTIVE terms are these
# is "NLP papers written in language X" — an artifact of multilingual corpora,
# not a missing encyclopedia CONCEPT. Transparent, small, observed-only list.
_FOREIGN_FUNCTION = {
    # Finnish / Estonian
    "suomen", "eesti", "kuidas", "miten", "viranomaistekstien", "som", "och",
    "mret", "kielen", "kielenarvostelu", "ajat",
    # Swedish / Norwegian / Danish
    "med", "och", "den", "det", "att", "och",
    # Dutch / German / French / Spanish
    "van", "den", "der", "und", "het", "een", "och", "les", "des", "une",
    "bir",  # Turkish
}


def _looks_too_niche(words: list[str]) -> bool:
    """A concept reads TOO-NICHE if its distinctive words are dominated by place/
    proper-noun hints, or by non-English function words (a "papers in language X"
    cluster — a multilingual-corpus artifact, not a missing concept), or are so
    few/generic that no encyclopedia would focus on the concept. Conservative —
    flags only clear cases."""
    if not words:
        return True
    if sum(1 for w in words if w in _PLACE_HINTS) >= 2:
        return True
    if sum(1 for w in words[:8] if w in _FOREIGN_FUNCTION) >= 2:
        return True
    return False


# =============================================================================
# Detector #1 — neural cross-corpus k-ball (reuse the demo machinery)
# =============================================================================
def run_detector1(
    oa: dict, wiki_titles_tf: list[str], wiki_emb_tf: NDArray[np.float32],
    *, oa_limit: int, n_gaps: int, radius_fraction: float, seed: int,
) -> list[dict]:
    """NEURAL cross-corpus k-ball, in the task's framing: WIKIPEDIA is the LANDMARK
    (the covering corpus), OA NLP articles are the AMBIENT/QUERY. A gap is therefore
    a region DENSE IN OA RESEARCH that no topic-filtered Wikipedia article covers —
    i.e. OA-dense, wiki-sparse, exactly the target. Per-corpus mean-center+renorm is
    still applied (the validated alignment that removes the wiki<->OA offset). This
    detector operates in the shared embedding geometry, so it SEES the register
    residual (the hypothesis under test).

    ``wiki_emb_tf`` / ``wiki_titles_tf`` are the already topic-filtered NLP-relevant
    Wikipedia slice (selected in main, shared with detector #2's has_wiki index).
    Returns a list of gap dicts with distinctive OA terms for the oracle."""
    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(seed)

    arts_emb_full = _normalize(oa["arts_emb"])
    art_titles_full = oa["arts_titles"]
    art_abstracts_full = oa["arts_abstracts"]

    nA = arts_emb_full.shape[0]
    if oa_limit <= 0 or oa_limit >= nA:
        oa_sel = np.arange(nA, dtype=np.int64)
    else:
        oa_sel = np.sort(rng.choice(nA, size=oa_limit, replace=False)).astype(np.int64)
    arts_emb = np.ascontiguousarray(arts_emb_full[oa_sel])
    art_titles = [art_titles_full[i] for i in oa_sel.tolist()]
    art_abstracts = [art_abstracts_full[i] for i in oa_sel.tolist()]

    wiki_emb_tf = _normalize(wiki_emb_tf)

    # per-corpus mean-center + renorm (the validated alignment). The wiki slice is
    # already topic-filtered; center it by ITS OWN centroid (the corpus-offset
    # estimate), center OA by the OA centroid.
    oa_centroid = arts_emb.mean(axis=0, keepdims=True)
    wiki_centroid = wiki_emb_tf.mean(axis=0, keepdims=True)
    arts_c = _normalize(arts_emb - oa_centroid)
    wiki_c = _normalize(wiki_emb_tf - wiki_centroid)

    sub_tag = (seed, oa_limit, len(wiki_titles_tf))
    # LANDMARK = topic-filtered Wikipedia (the covering corpus).
    landmark = MemCorpus(
        _embeddings=wiki_c, _work_ids=tuple(f"w{i}" for i in range(len(wiki_titles_tf))),
        _work_titles=tuple(wiki_titles_tf), _fingerprint=oa["fp"], _name="wiki_landmark",
        _identity=("wiki_landmark_spike", "wiki_400k_tf", oa["fp"], sub_tag),
    )
    # AMBIENT/QUERY = OA NLP articles (the dense research corpus we hunt gaps in).
    oa_art_corpus = MemCorpus(
        _embeddings=arts_c, _work_ids=tuple(f"a{i}" for i in range(len(art_titles))),
        _work_titles=tuple(art_titles), _fingerprint=oa["fp"], _name="openalex",
        _identity=("oa_arts_centered_spike", NLP_TOPIC, oa["fp"], sub_tag),
    )
    print(f"  [D1] landmark=wiki ({wiki_c.shape[0]} TF articles), "
          f"ambient/query=OA ({arts_c.shape[0]} NLP articles); detecting {n_gaps} gaps ...",
          flush=True)

    gaps = detect_gaps(
        ambient=oa_art_corpus, landmark=landmark, query=oa_art_corpus,
        cell_type=CellType.KBALL, num_gaps=n_gaps, radius_fraction=radius_fraction,
        label_id="spike-nlp", label_name="spike NLP",
    )
    print(f"  [D1] detect_gaps returned {len(gaps)} gaps", flush=True)

    out: list[dict] = []
    for gwc in gaps:
        members = np.asarray(gwc.gap.get_dists_idxs().idxs, dtype=np.int64)
        oa_member_titles = [art_titles[i] for i in members.tolist()]
        oa_member_abstracts = [art_abstracts[i] for i in members.tolist()]
        # distinctive terms for this gap's OA members vs an OA background sample.
        if len(oa_member_titles) >= 2:
            bg_docs = [f"{t}" for t in art_titles[:2000]]
            gap_docs = [f"{t}. {a}" for t, a in zip(oa_member_titles, oa_member_abstracts)]
            terms = distinctive_terms_for_groups([gap_docs, bg_docs], top_k=12)[0]
            dterms = [w for w, _ in terms]
        else:
            dterms = []
        out.append({
            "gap_idx": int(gwc.gap_idx),
            "size": int(gwc.gap_size),
            "percentile": float(gwc.gap_size_percentile),
            "n_oa": int(len(members)),
            "n_wiki": 0,
            "distinctive_terms": dterms,
            "oa_titles": oa_member_titles[:8],
            "wiki_titles": [],
        })
    return out


# =============================================================================
# Detector #2 — embedder-free has_wiki-deficit concepts
# =============================================================================
def run_detector2(
    oa: dict, wiki_index: WikiTitleIndex,
    *, oa_limit: int, n_concepts: int, n_gaps: int, seed: int,
) -> tuple[list[dict], float]:
    """Cluster OA NLP articles into concepts (embedding clustering for FORMATION),
    extract lift-distinctive terms per concept, compute the embedder-free has_wiki
    attribute (term-matched against the topic-filtered wiki index), and rank
    concepts by has_wiki deficit (lift < 1). Returns (top_gap_concepts, base_rate).
    """
    rng = np.random.default_rng(seed)
    arts_emb = _normalize(oa["arts_emb"])
    art_titles = oa["arts_titles"]
    art_abstracts = oa["arts_abstracts"]
    nA = arts_emb.shape[0]
    if oa_limit <= 0 or oa_limit >= nA:
        sel = np.arange(nA, dtype=np.int64)
    else:
        sel = np.sort(rng.choice(nA, size=oa_limit, replace=False)).astype(np.int64)
    X = np.ascontiguousarray(arts_emb[sel])
    titles = [art_titles[i] for i in sel.tolist()]
    abstracts = [art_abstracts[i] for i in sel.tolist()]

    # --- concept formation: spherical k-means (faiss) on unit embeddings ------
    k = n_concepts
    print(f"  [D2] clustering {X.shape[0]} OA NLP articles into {k} concepts ...", flush=True)
    faiss.omp_set_num_threads(4)
    km = faiss.Kmeans(X.shape[1], k, niter=25, verbose=False, seed=seed, spherical=True)
    km.train(X)
    _, assign = km.index.search(X, 1)
    assign = assign.ravel()

    # group member texts per concept
    concept_docs: list[list[str]] = [[] for _ in range(k)]
    concept_member_titles: list[list[str]] = [[] for _ in range(k)]
    for i, c in enumerate(assign.tolist()):
        concept_docs[c].append(f"{titles[i]}. {abstracts[i]}")
        concept_member_titles[c].append(titles[i])

    # --- lift-distinctive terms per concept (validated primitive) -------------
    print(f"  [D2] extracting lift-distinctive terms per concept ...", flush=True)
    terms_per_concept = distinctive_terms_for_groups(concept_docs, top_k=12)

    # --- has_wiki attribute per concept (EMBEDDER-FREE term match) ------------
    concepts: list[dict] = []
    for c in range(k):
        size = len(concept_member_titles[c])
        if size < 5:
            continue  # skip tiny/degenerate clusters
        dterms = [w for w, _ in terms_per_concept[c]]
        matched, mtitle = wiki_index.has_match(dterms)
        concepts.append({
            "concept_id": c,
            "size": size,
            "distinctive_terms": dterms,
            "has_wiki": bool(matched),
            "matched_wiki_title": mtitle,
            "rep_titles": concept_member_titles[c][:8],
        })

    # base rate of has_wiki across concepts (the denominator of lift)
    n_with = sum(1 for c in concepts if c["has_wiki"])
    base_rate = n_with / max(len(concepts), 1)
    # has_wiki lift per concept: a concept's has_wiki indicator / base_rate.
    # Concepts with has_wiki=False have lift 0 (< 1) = wiki-deficient gaps.
    # Rank wiki-deficient concepts by SIZE (largest OA-dense wiki-missing first).
    for c in concepts:
        c["has_wiki_lift"] = (1.0 if c["has_wiki"] else 0.0) / max(base_rate, 1e-9)
    gaps = [c for c in concepts if not c["has_wiki"]]
    gaps.sort(key=lambda c: c["size"], reverse=True)
    return gaps[:n_gaps], base_rate


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki-store", type=Path, default=Path("cache/wiki/store/wiki_400k"))
    ap.add_argument("--oa-limit", type=int, default=8000,
                    help="max OA NLP articles in detector #1 union (0=all ~87k)")
    ap.add_argument("--wiki-topk", type=int, default=8000,
                    help="topic-filtered wiki articles kept (D1 ambient + D2 has_wiki index)")
    ap.add_argument("--n-concepts", type=int, default=60,
                    help="detector #2: number of OA NLP concepts (k-means clusters)")
    ap.add_argument("--d2-oa-limit", type=int, default=0,
                    help="max OA NLP articles for detector #2 clustering (0=all)")
    ap.add_argument("--n-gaps", type=int, default=15)
    ap.add_argument("--radius-fraction", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the live Wikipedia API validation (offline dry run)")
    ap.add_argument("--out", type=Path, default=Path("cache/wiki/wiki_gap_spike_result.json"))
    args = ap.parse_args()

    print("=" * 78, flush=True)
    print("WIKI GAP SPIKE — OA-dense, Wikipedia-missing NLP research topics", flush=True)
    print("=" * 78, flush=True)

    oa = load_oa_nlp_cached()
    print(f"OA NLP: {oa['books_emb'].shape[0]} books, {oa['arts_emb'].shape[0]} articles, "
          f"fp={oa['fp']}", flush=True)

    wiki = WikipediaCorpus(args.wiki_store)
    assert wiki.embedder_fingerprint == oa["fp"], "fingerprint mismatch OA vs wiki"
    wiki_emb_all = np.ascontiguousarray(wiki.embeddings(), dtype=np.float32)
    wiki_titles_all = wiki.work_titles()
    print(f"Wiki store: {wiki_emb_all.shape[0]} articles", flush=True)

    # Topic-filter the wiki slice ONCE (shared selection for D1 ambient and the
    # D2 has_wiki landmark index — both need the NLP-relevant slice, not uniform).
    # We re-derive the same selection inside D1; for D2 we build the index from
    # the same topic-filtered titles so has_wiki is judged against the NLP-relevant
    # encyclopedic neighborhood (footgun: never the uniform 400k).
    wiki_emb_n = _normalize(wiki_emb_all)
    arts_emb_n = _normalize(oa["arts_emb"])
    oa_centroid = arts_emb_n.mean(axis=0, keepdims=True)
    wiki_centroid = wiki_emb_n.mean(axis=0, keepdims=True)
    arts_c = _normalize(arts_emb_n - oa_centroid)
    wiki_c_all = _normalize(wiki_emb_n - wiki_centroid)
    nlp_dir = arts_c.mean(axis=0)
    nn = np.linalg.norm(nlp_dir)
    nlp_dir = (nlp_dir / nn).astype(np.float32) if nn > 0 else nlp_dir
    wiki_cos = wiki_c_all @ nlp_dir
    topk = min(args.wiki_topk, wiki_cos.shape[0])
    part = np.argpartition(-wiki_cos, topk - 1)[:topk]
    wiki_sel = part[np.argsort(-wiki_cos[part])].astype(np.int64)
    wiki_titles_tf = [wiki_titles_all[i] for i in wiki_sel.tolist()]
    wiki_emb_tf = np.ascontiguousarray(wiki_emb_all[wiki_sel])  # RAW (un-normed) tf slice
    print(f"Topic-filtered wiki landmark slice: {len(wiki_titles_tf)} titles "
          f"(cos {wiki_cos[wiki_sel].min():.3f}..{wiki_cos[wiki_sel].max():.3f})", flush=True)
    print(f"  sample TF wiki titles: {wiki_titles_tf[:12]}", flush=True)

    wiki_index = WikiTitleIndex.build(wiki_titles_tf)

    # ---- Detector #1 ----
    print("\n--- Detector #1: neural cross-corpus k-ball (wiki=landmark, OA=ambient) ---", flush=True)
    d1_gaps = run_detector1(
        oa, wiki_titles_tf, wiki_emb_tf,
        oa_limit=args.oa_limit, n_gaps=args.n_gaps,
        radius_fraction=args.radius_fraction, seed=args.seed,
    )

    # ---- Detector #2 ----
    print("\n--- Detector #2: embedder-free has_wiki-deficit ---", flush=True)
    d2_gaps, base_rate = run_detector2(
        oa, wiki_index,
        oa_limit=args.d2_oa_limit, n_concepts=args.n_concepts, n_gaps=args.n_gaps,
        seed=args.seed,
    )
    print(f"  [D2] has_wiki base rate across concepts: {base_rate:.3f}; "
          f"{len(d2_gaps)} wiki-deficient gaps", flush=True)

    # ---- Oracle ----
    if not args.no_oracle:
        print("\n--- Oracle: live Wikipedia API on top gaps ---", flush=True)
        print("  [D1] querying ...", flush=True)
        for g in d1_gaps:
            if g["distinctive_terms"]:
                g["oracle"] = classify_gap_via_oracle(g["distinctive_terms"], g["oa_titles"])
            else:
                g["oracle"] = {"verdict": "TOO-NICHE", "reason": "no distinctive terms", "hits": []}
        print("  [D2] querying ...", flush=True)
        for g in d2_gaps:
            g["oracle"] = classify_gap_via_oracle(g["distinctive_terms"], g["rep_titles"])

    result = {
        "params": vars(args) | {"wiki_store": str(args.wiki_store), "out": str(args.out)},
        "oa_books": int(oa["books_emb"].shape[0]),
        "oa_articles": int(oa["arts_emb"].shape[0]),
        "wiki_topic_filtered": len(wiki_titles_tf),
        "d2_has_wiki_base_rate": base_rate,
        "detector1_gaps": d1_gaps,
        "detector2_gaps": d2_gaps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote full result -> {args.out}", flush=True)

    _print_summary(d1_gaps, d2_gaps, base_rate, no_oracle=args.no_oracle)


def _verdict_counts(gaps: list[dict]) -> Counter:
    return Counter(g.get("oracle", {}).get("verdict", "?") for g in gaps)


def _print_summary(d1_gaps, d2_gaps, base_rate, *, no_oracle: bool) -> None:
    print("\n" + "=" * 78, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 78, flush=True)

    def _show(name, gaps):
        print(f"\n### {name} — top {len(gaps)} gaps", flush=True)
        for i, g in enumerate(gaps):
            terms = ", ".join(g["distinctive_terms"][:6])
            ver = g.get("oracle", {}).get("verdict", "(no oracle)")
            reason = g.get("oracle", {}).get("reason", "")
            mt = g.get("oracle", {}).get("matched_title")
            extra = f"size={g.get('size')}"
            if "n_oa" in g:
                extra += f" n_oa={g['n_oa']} n_wiki={g['n_wiki']}"
            print(f"  [{i:2d}] {ver:15s} | {extra}", flush=True)
            print(f"       terms: {terms}", flush=True)
            if mt:
                print(f"       wiki hit: {mt!r}", flush=True)
            elif reason:
                print(f"       {reason}", flush=True)
        if not no_oracle:
            vc = _verdict_counts(gaps)
            n = len(gaps)
            real = vc.get("REAL-GAP", 0)
            prec = real / n if n else 0.0
            print(f"  -> verdicts: {dict(vc)}  PRECISION(REAL-GAP/total)={prec:.2f}", flush=True)

    _show("Detector #1 (neural k-ball)", d1_gaps)
    _show("Detector #2 (embedder-free has_wiki-deficit)", d2_gaps)

    # agreement: overlap of distinctive-term sets
    def _termset(g):
        return set(w for t in g["distinctive_terms"] for w in t.split("_") if len(w) >= 4)
    d1_terms = set().union(*[_termset(g) for g in d1_gaps]) if d1_gaps else set()
    d2_terms = set().union(*[_termset(g) for g in d2_gaps]) if d2_gaps else set()
    print(f"\n### Agreement (distinctive-term vocabulary overlap)", flush=True)
    inter = d1_terms & d2_terms
    print(f"  D1 term-vocab {len(d1_terms)}, D2 term-vocab {len(d2_terms)}, "
          f"shared {len(inter)}: {sorted(inter)[:30]}", flush=True)
    print(f"  D2 has_wiki base rate (fraction of concepts WITH a wiki match): {base_rate:.3f}", flush=True)


# --- provenance footer -------------------------------------------------------
def _provenance() -> str:
    return "python -m artifacts_scripts.wiki.wiki_gap_spike " + " ".join(sys.argv[1:])


if __name__ == "__main__":
    main()
    # Regenerate, don't hand-edit outputs:
    #   uv run python -m artifacts_scripts.wiki.wiki_gap_spike <args>
    # (full command for this run reconstructed from argv:)
    #   {_provenance()}
