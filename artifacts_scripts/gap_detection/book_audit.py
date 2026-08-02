"""Track 3 — book gap-audit demo: fg extraction, OA field slice, joint embed.

DESCRIPTIVE DEMO — BELOW THE VALIDATED FRONTIER. Every validated configuration
(k=10, scope q0.25, LSA-128, miner min_support 30, the 64-doc recoverable-gap
frontier) was tuned at 30K docs with a 1,446-doc fg; here fg is ~10^2 book
sections, so parameters are rescaled (k_fg=3, LSA-64, support scaled to the
scoped pool) and nothing in this module supports a validated-recovery claim.

fg = section-level chunks of `ch*.tex` from the ds2 build
     (~/src/learn-and-teach/outputs/2026-06-09/23-15-05/build/sections; READ-ONLY;
     solutions.tex excluded — exercise answers, not coverage; notation.tex is not
     a chapter and does not match the glob).

UNIT RULE (documented contract, tested):
  * a unit starts at every \\section{...} OR \\subsection{...} command;
    rationale: the book has 42 \\section + 93 \\subsection commands — splitting
    on \\section alone gives ~51 units, well under the brief's expected 80-150;
    subsection splitting lands in-range and sharpens nearest-unit attribution.
  * the text between \\chapter{...} and the first \\section is a "chapter intro"
    unit (chapter title prepended — it is that unit's only headline).
  * each unit's own section/subsection title is prepended to its text (titles
    are topical content); subsection units inherit their parent section title
    as metadata only.
  * units whose stripped text is < MIN_UNIT_CHARS (200, the bg abstract
    eligibility rule) are MERGED into the previous unit of the same chapter
    (first unit merges forward) — book coverage is never dropped.

LATEX -> TEXT RULE (documented contract, tested), applied per unit:
  1. comments: unescaped `%` to end-of-line -> space;
  2. preamble/wrapper: only the body between \\begin{document}..\\end{document}
     is scanned (each chapter is a standalone document);
  3. `lstlisting` environments dropped wholesale (code, not prose — and the bg
     abstracts contain no code);
  4. math dropped: `$$..$$`, `\\[..\\]`, `\\(..\\)`, `$..$`, and
     equation/align/gather/eqnarray/multline/bmatrix/pmatrix/matrix/vmatrix/
     cases/array environments (starred or not);
  5. reference/graphics commands dropped WITH their argument: label, ref,
     eqref, pageref, cite(p/t), includegraphics, input, include, usepackage,
     documentclass, bibliography(style), vspace, hspace;
  6. text-carrying commands unwrapped (argument kept): textbf, textit, emph,
     text, texttt, textsc, underline, caption, mbox, paragraph — captions
     therefore survive as prose while their figures are dropped;
  7. \\begin{..}/\\end{..} tokens, \\item[..], and any remaining \\command
     tokens -> space; escaped \\& \\% \\_ \\# \\$ -> bare char;
  8. leftover structural chars ({ } [ ] $ ~ ^ _ backslash) -> space (a bare &
     can only come from \\& since math/tabular are already gone); whitespace
     collapsed.

bg = an EXTERNAL OpenAlex slice (the ds2 build is anchorless — no reference
corpus exists). Topic set is PRE-COMMITTED on meta.count probes recorded in the
worklog before any scoring. Polite pool: every request carries
mailto=$OPENALEX_EMAIL + a descriptive User-Agent, exponential
backoff on 429/5xx, every page cached under cache/oa/t3_ds_book/ so reruns are
offline.

Context hygiene: this module processes book text and abstracts internally but
only ever EMITS titles, top terms, counts, and numeric aggregates.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPU owned by a sweep; CPU only.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import gzip
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from artifacts_scripts.gap_detection.oa_corpus import (
    MAILTO,
    decode_abstract,
    is_english,
    short_id,
)

REPO = Path(__file__).resolve().parents[2]
BOOK_SECTIONS_DIR = Path(
    "~/src/learn-and-teach/outputs/2026-06-09/23-15-05/build/sections"
).expanduser()
T3_OA_CACHE = REPO / "cache/oa/t3_ds_book"
OUT_DIR = REPO / "artifacts/gap_detection_eval"
CACHE_DIR = OUT_DIR / "cache"

API_BASE = "https://api.openalex.org"
USER_AGENT = (
    "bookgap-gap-detection-research/0.1 "
    f"(t3-book-audit field-slice{f'; mailto:{MAILTO}' if MAILTO else ''}) "
    "cursor-paging"
)
T3_SELECT_FIELDS = (
    "id,display_name,publication_year,abstract_inverted_index,"
    "primary_topic,cited_by_count,type"
)
# Work types that are field LITERATURE; the rest are catalog paratext (dataset
# records, referee reports, errata, editorials...) observed polluting the v1
# pull ("Peer Review #N of ..." x4, "msd-errors - random walk data" x78).
KEEP_TYPES = frozenset(
    "article book-chapter book preprint review dissertation report".split()
)
PER_PAGE = 200
MIN_UNIT_CHARS = 200
MIN_ABSTRACT_CHARS = 200

SEED = 42
LSA_DIM = 64
N_KEYWORDS = 2_000


# =============================================================== fg: LaTeX -> units
_MATH_ENVS = (
    "equation|align|gather|eqnarray|multline|bmatrix|pmatrix|matrix|vmatrix|cases|array"
)
_DROP_WITH_ARG = (
    "label|ref|eqref|pageref|citep|citet|cite|includegraphics|input|include|"
    "usepackage|documentclass|bibliographystyle|bibliography|vspace|hspace"
)
_UNWRAP = "textbf|textit|emph|texttt|textsc|text|underline|caption|mbox|paragraph"


def strip_latex(s: str) -> str:
    """LaTeX -> plain text per the documented module rule (steps 1, 3-8)."""
    s = re.sub(r"(?<!\\)%[^\n]*", " ", s)                                    # 1 comments
    s = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", " ", s,      # 3 code
               flags=re.DOTALL)
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.DOTALL)                      # 4 math
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.DOTALL)
    s = re.sub(r"\\\(.*?\\\)", " ", s, flags=re.DOTALL)
    s = re.sub(rf"\\begin\{{({_MATH_ENVS})\*?\}}.*?\\end\{{\1\*?\}}", " ", s,
               flags=re.DOTALL)
    s = re.sub(r"(?<!\\)\$[^$]*\$", " ", s, flags=re.DOTALL)
    s = re.sub(rf"\\({_DROP_WITH_ARG})\*?(\[[^\]]*\])?\{{[^{{}}]*\}}", " ", s)  # 5 drop
    for _ in range(3):                                                       # 6 unwrap
        s, n = re.subn(rf"\\({_UNWRAP})\*?\{{([^{{}}]*)\}}", r" \2 ", s)
        if not n:
            break
    s = re.sub(r"\\(?:begin|end)\{[a-zA-Z*]+\}", " ", s)                     # 7 tokens
    s = re.sub(r"\\item(\[[^\]]*\])?", " ", s)
    s = re.sub(r"\\([&%_#$])", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}\[\]$~^_\\]", " ", s)                                    # 8 leftovers
    return re.sub(r"\s+", " ", s).strip()


def _read_braced(s: str, i: int) -> tuple[str, int]:
    """s[i] must be '{'; return (content, index after matching '}')."""
    assert s[i] == "{", f"expected '{{' at {i}"
    depth, j = 1, i + 1
    while j < len(s) and depth:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    return s[i + 1 : j - 1], j


@dataclass
class BookUnit:
    chapter: str                       # file stem, e.g. "ch5"
    chapter_title: str
    section_title: str                 # "(chapter intro)" for the intro unit
    subsection_title: str | None
    text: str = ""
    merged_titles: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        lab = f"{self.chapter} {self.chapter_title} § {self.section_title}"
        if self.subsection_title:
            lab += f" ▸ {self.subsection_title}"
        return lab


def split_units(tex: str, chapter: str) -> list[BookUnit]:
    """One chapter .tex -> ordered BookUnits per the documented unit rule."""
    if "\\begin{document}" in tex:                                           # 2 wrapper
        tex = tex.split("\\begin{document}", 1)[1]
    if "\\end{document}" in tex:
        tex = tex.rsplit("\\end{document}", 1)[0]

    ch_title = ""
    m = re.search(r"\\chapter\s*\{", tex)
    if m:
        raw, _ = _read_braced(tex, m.end() - 1)
        ch_title = strip_latex(raw)

    marks = []  # (start_of_cmd, end_of_title_brace, kind, title)
    for mm in re.finditer(r"\\(section|subsection)\s*\{", tex):
        raw, j = _read_braced(tex, mm.end() - 1)
        marks.append((mm.start(), j, mm.group(1), strip_latex(raw)))

    units: list[BookUnit] = []
    intro_end = marks[0][0] if marks else len(tex)
    intro_txt = strip_latex(tex[:intro_end])
    units.append(BookUnit(chapter, ch_title, "(chapter intro)", None,
                          text=f"{ch_title} {intro_txt}".strip()))
    cur_section = "(chapter intro)"
    for n, (start, title_end, kind, title) in enumerate(marks):
        body_end = marks[n + 1][0] if n + 1 < len(marks) else len(tex)
        body = strip_latex(tex[title_end:body_end])
        if kind == "section":
            cur_section = title
            u = BookUnit(chapter, ch_title, title, None)
        else:
            u = BookUnit(chapter, ch_title, cur_section, title)
        u.text = f"{title} {body}".strip()
        units.append(u)

    # merge-tiny rule: < MIN_UNIT_CHARS merges into the PREVIOUS unit of the
    # chapter (a short first unit defers and merges forward into the next);
    # coverage is never dropped.
    out: list[BookUnit] = []
    pending: BookUnit | None = None
    for u in units:
        if pending is not None:
            u.text = f"{pending.text} {u.text}".strip()
            u.merged_titles = (pending.merged_titles
                               + [pending.subsection_title or pending.section_title]
                               + u.merged_titles)
            pending = None
        if len(u.text) < MIN_UNIT_CHARS:
            if out:
                out[-1].text = f"{out[-1].text} {u.text}".strip()
                out[-1].merged_titles += [u.subsection_title or u.section_title]
            else:
                pending = u
        else:
            out.append(u)
    if pending is not None:  # pathological: the whole chapter is tiny
        out.append(pending)
    return out


def _chapter_sort_key(p: Path) -> tuple[int, str]:
    m = re.match(r"ch(\d+)$", p.stem)
    return (int(m.group(1)) if m else 999, p.stem)


def load_book_fg(sections_dir: Path = BOOK_SECTIONS_DIR) -> list[BookUnit]:
    """All chapter units in book order. Glob ch*.tex only (solutions/notation out)."""
    units: list[BookUnit] = []
    for p in sorted(sections_dir.glob("ch*.tex"), key=_chapter_sort_key):
        units.extend(split_units(p.read_text(encoding="utf-8"), p.stem))
    return units


# =============================================================== bg: OpenAlex slice
def _session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get_json(session, url: str, params: dict, log=print) -> dict:
    backoff = 1.0
    for _ in range(8):
        try:
            resp = session.get(url, params={**params, "mailto": MAILTO}, timeout=60)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 64.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 — network/decode; retry with backoff
            log(f"  retry after {type(e).__name__}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 64.0)
    raise RuntimeError(f"GET {url} failed after retries")


def search_topics(query: str, per_page: int = 25, log=print) -> list[dict]:
    """GET /topics?search=...; cached. Returns id/display_name/works_count/subfield."""
    T3_OA_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = T3_OA_CACHE / f"topics_{hashlib.sha1(query.encode()).hexdigest()[:10]}.json"
    if cpath.exists():
        data = json.loads(cpath.read_text())
    else:
        data = _get_json(_session(), f"{API_BASE}/topics",
                         {"search": query, "per-page": str(per_page)}, log=log)
        cpath.write_text(json.dumps(data))
    out = []
    for t in data.get("results", []):
        out.append({
            "id": short_id(t.get("id")),
            "display_name": t.get("display_name"),
            "works_count": t.get("works_count"),
            "subfield": ((t.get("subfield") or {}).get("display_name")),
            "field": ((t.get("field") or {}).get("display_name")),
        })
    return out


def works_filter(topic_ids: list[str], pub_from: str, pub_to: str) -> str:
    return (f"primary_topic.id:{'|'.join(topic_ids)},"
            f"from_publication_date:{pub_from},to_publication_date:{pub_to},"
            f"has_abstract:true,language:en")


def works_count(filt: str, log=print) -> int:
    """meta.count probe for a works filter (per-page=1); cached."""
    T3_OA_CACHE.mkdir(parents=True, exist_ok=True)
    cpath = T3_OA_CACHE / f"count_{hashlib.sha1(filt.encode()).hexdigest()[:10]}.json"
    if cpath.exists():
        return int(json.loads(cpath.read_text())["count"])
    data = _get_json(_session(), f"{API_BASE}/works",
                     {"filter": filt, "per-page": "1", "select": "id"}, log=log)
    n = int(data["meta"]["count"])
    cpath.write_text(json.dumps({"count": n, "filter": filt}))
    return n


def _page_path(filt: str, page_idx: int) -> Path:
    # hash covers filter AND select fields: changing the select busts the cache
    # (the v1 pull without `type` stays on disk under its own key).
    h = hashlib.sha1(f"{filt}|{T3_SELECT_FIELDS}".encode()).hexdigest()[:10]
    return T3_OA_CACHE / f"works_{h}" / f"page_{page_idx:05d}.json.gz"


def fetch_works(filt: str, log=print, max_pages: int | None = None) -> str:
    """Cursor-page the works slice to cache (resumable); returns the filter key."""
    cursor, page_idx, n_cached = "*", 0, 0
    while True:
        p = _page_path(filt, page_idx)
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
            log(f"slice fully cached: {page_idx} pages, {n_cached:,} works")
            return filt
    if page_idx:
        log(f"resuming from cached page {page_idx} ({n_cached:,} works)")
    session = _session()
    while cursor is not None:
        if max_pages is not None and page_idx >= max_pages:
            log(f"max_pages={max_pages} reached; stopping (partial)")
            break
        data = _get_json(session, f"{API_BASE}/works", {
            "filter": filt, "select": T3_SELECT_FIELDS,
            "per-page": str(PER_PAGE), "cursor": cursor,
        }, log=log)
        p = _page_path(filt, page_idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        if page_idx == 0:
            log(f"meta.count = {data['meta']['count']:,}")
        n_cached += len(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        page_idx += 1
        if page_idx % 50 == 0:
            log(f"  {n_cached:,} works ({page_idx} pages)")
        time.sleep(0.05)
    log(f"slice acquired: {page_idx} pages, {n_cached:,} works")
    return filt


def iter_cached_works(filt: str):
    page_idx = 0
    while True:
        p = _page_path(filt, page_idx)
        if not p.exists():
            return
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        yield from data.get("results", [])
        page_idx += 1


def assemble_bg(filt: str, log=print) -> dict:
    """Cached pages -> parallel arrays + decoded abstract texts (in-process only).

    Eligibility (data-cleaning within the pre-committed universe; documented):
    type in KEEP_TYPES (literature, not catalog paratext); abstract >=
    MIN_ABSTRACT_CHARS; is_english (belt-and-braces on top of the language:en
    filter); exact lowercase-title dedupe; exact abstract-TEXT dedupe (first
    wins — kills shared-publisher-blurb chapter piles whose r_B ~ 0 explodes
    the habitability weight; observed v2 pathology, component score 1.4e12).
    """
    wids, titles, years, topics, cited, texts = [], [], [], [], [], []
    n_seen = n_short = n_noneng = n_dup = n_dup_txt = n_type = 0
    seen_titles: set[str] = set()
    seen_text_hash: set[str] = set()
    dup_counter: dict[str, int] = {}
    type_counter: dict[str, int] = {}
    for w in iter_cached_works(filt):
        n_seen += 1
        wtype = str(w.get("type") or "unknown")
        if wtype not in KEEP_TYPES:
            n_type += 1
            type_counter[wtype] = type_counter.get(wtype, 0) + 1
            continue
        txt = decode_abstract(w.get("abstract_inverted_index"))
        if len(txt) < MIN_ABSTRACT_CHARS:
            n_short += 1
            continue
        if not is_english(txt):
            n_noneng += 1
            continue
        t = (w.get("display_name") or "").strip()
        tkey = " ".join(t.lower().split())  # whitespace-normalized dedupe key
        if tkey and tkey in seen_titles:
            n_dup += 1
            dup_counter[tkey] = dup_counter.get(tkey, 0) + 1
            continue
        thash = hashlib.sha1(txt.encode()).hexdigest()
        if thash in seen_text_hash:
            n_dup_txt += 1
            continue
        seen_titles.add(tkey)
        seen_text_hash.add(thash)
        wids.append(short_id(w.get("id")))
        titles.append(t)
        years.append(int(w.get("publication_year") or 0))
        topics.append(short_id((w.get("primary_topic") or {}).get("id")) or "")
        cited.append(int(w.get("cited_by_count") or 0))
        texts.append(txt)
    log(f"bg assembled: {n_seen:,} cached -> {len(wids):,} eligible "
        f"(paratext-type {n_type:,}, short {n_short:,}, non-English {n_noneng:,}, "
        f"dup-title {n_dup:,}, dup-text {n_dup_txt:,})")
    return {
        "wid": np.asarray(wids, dtype=object),
        "title": np.asarray(titles, dtype=object),
        "year": np.asarray(years, dtype=np.int64),
        "topic": np.asarray(topics, dtype=object),
        "cited_by": np.asarray(cited, dtype=np.int64),
        "texts": texts,
        "counts": {"seen": n_seen, "eligible": len(wids), "paratext_type": n_type,
                   "short": n_short, "non_english": n_noneng, "dup_title": n_dup,
                   "dup_text": n_dup_txt,
                   "dropped_types": dict(sorted(type_counter.items(),
                                                key=lambda kv: -kv[1])),
                   "top_dup_titles": sorted(dup_counter.items(),
                                            key=lambda kv: -kv[1])[:10]},
    }


# =============================================================== joint embedding
def embed_joint(
    fg_texts: list[str], bg_texts: list[str], *,
    seed: int = SEED, lsa_dim: int = LSA_DIM, n_keywords: int = N_KEYWORDS,
    min_df: int = 10, max_df: float = 0.30, max_features: int = 60_000,
) -> dict:
    """fg+bg -> joint TF-IDF -> LSA (L2-normalized) + binary top-df keyword space.

    Contract (tested): row order is fg_texts then bg_texts — fg_mask marks the
    first len(fg_texts) rows; NO row is dropped (the nnz>=5 bg trim is the
    caller's job so masks stay aligned); deterministic for a fixed seed.
    Returns Z (n,d) float32 unit-norm, K (n,m) uint8 csr, vocab (m,), fg_mask,
    X (tf-idf, for top-term labeling), tfidf_vocab, evr.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = list(fg_texts) + list(bg_texts)
    vec = TfidfVectorizer(
        stop_words="english", lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        min_df=min_df, max_df=max_df, max_features=max_features,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    vocab_all = vec.get_feature_names_out()
    k = min(lsa_dim, max(1, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    Z = normalize(svd.fit_transform(X)).astype(np.float32)
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    kw_cols = np.sort(np.argsort(df)[::-1][: min(n_keywords, df.size)])
    K = (X[:, kw_cols] > 0).astype(np.uint8).tocsr()
    fg_mask = np.zeros(X.shape[0], dtype=bool)
    fg_mask[: len(fg_texts)] = True
    return {
        "Z": Z, "K": K, "vocab": vocab_all[kw_cols], "fg_mask": fg_mask, "X": X,
        "tfidf_vocab": vocab_all,
        "evr": float(svd.explained_variance_ratio_.sum()),
    }


def full_statistic(
    F: np.ndarray, B: np.ndarray, *, k_fg: int = 3, k_bg: int = 10,
    hab_exp: float = 0.5, lcb_lambda: float = 0.5, alpha_prior: float = 1.0,
) -> dict:
    """The validated GapScorer 'full' statistic — hab * (Delta - lam*sigma) —
    over candidates = B (no probes), with the fg-side k DECOUPLED from the
    bg-side k. The decoupling is the t3 scale rescale (fg ~ 10^2 book units:
    k_fg=3 ~ the validated 0.7%-of-fg; bg keeps the validated k=10).

    Identity contract (tested): with k_fg == k_bg == k this reproduces
    detectors.GapScorer(bg=B, probes=None, k=k).score(F)['full'] exactly.
    """
    from artifacts_scripts.gap_detection.detectors import (
        EPS,
        count_within_radius,
        knn_radii,
    )

    r_F = knn_radii(B, F, k_fg)
    r_B = knn_radii(B, B, k_bg, ref_is_query=True)
    log_rB = np.log(r_B)
    mu, sd = log_rB.mean(), log_rB.std() + EPS
    hab = np.exp(hab_exp * (-(log_rB - mu) / sd))
    delta = np.log(r_F) - log_rB
    R = np.maximum(r_F, r_B)
    n_F = count_within_radius(B, F, R)
    n_B = count_within_radius(B, B, R, ref_is_query=True)
    sigma = np.sqrt(1.0 / (n_F + alpha_prior) + 1.0 / (np.maximum(n_B, 0) + alpha_prior))
    z = delta - lcb_lambda * sigma
    return {"full": hab * z, "r_F": r_F, "r_B": r_B, "delta": delta,
            "z": z, "hab": hab}


def region_terms(K_pool: np.ndarray, members: np.ndarray, vocab: np.ndarray,
                 n_top: int = 12, alpha: float = 1.0, min_df: int = 3) -> list[str]:
    """Top distinctive keyword terms of a region (log-odds member df vs rest).

    Same contract as describe_t2_oa.region_terms (copied: that module hard-imports
    the t2 OA phase-3 stack at import time)."""
    K_pool = np.asarray(K_pool, dtype=bool)
    members = np.asarray(members, dtype=np.int64)
    if members.size == 0:
        return []
    inside = K_pool[members].sum(axis=0).astype(np.float64)
    rest = ~np.isin(np.arange(K_pool.shape[0]), members)
    outside = K_pool[rest].sum(axis=0).astype(np.float64)
    n_in, n_out = float(members.size), float(rest.sum())
    score = (np.log((inside + alpha) / (n_in - inside + alpha))
             - np.log((outside + alpha) / (n_out - outside + alpha)))
    score[inside < min_df] = -np.inf
    order = np.argsort(score)[::-1][:n_top]
    return [str(vocab[j]) for j in order if np.isfinite(score[j])]


if __name__ == "__main__":  # offline smoke: fg extraction stats only
    units = load_book_fg()
    lens = np.array([len(u.text) for u in units])
    by_ch: dict[str, int] = {}
    for u in units:
        by_ch[u.chapter] = by_ch.get(u.chapter, 0) + 1
    print(f"fg units: {len(units)} | chars min/med/mean/max = "
          f"{lens.min()}/{int(np.median(lens))}/{int(lens.mean())}/{lens.max()}")
    print("units per chapter:", json.dumps(by_ch))
    n_merged = sum(1 for u in units if u.merged_titles)
    print(f"units carrying merges: {n_merged}")
