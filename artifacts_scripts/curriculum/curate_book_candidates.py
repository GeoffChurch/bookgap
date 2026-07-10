"""Curate ToC-reconstruction targets from the own-gap ranking.

The reconstruction eval is a direct proxy for the product goal: the largest
GENUINE-book own-gaps are the most underserved real regions a human wrote a book
to fill, so "does our curriculum reconstruct that book's ToC?" measures whether
our system serves the highest-value underserved gaps as well as a domain expert.

But the raw top of the own-gap ranking is debris (misclassified non-English
textbooks, theses, off-domain works are extreme outliers with huge gaps). This
script filters the ranking (`book_own_gap.py --rank-top` output) to genuine,
English, coherent-gap books and re-ranks by gap size -- the ordered list to
ToC-check (via Google Books) and feed to the panel.

Usage:
  uv run python -m artifacts_scripts.curriculum.curate_book_candidates \
      artifacts/book_gaps/csfield_ranked.json artifacts/book_gaps/ai_subfield_ranked.json
"""
import json
import re
import sys
from collections import Counter

_STOP = set((
    "the of for a an and or to in on with using based via from into over under is are be "
    "this that these those we our their it its as at by such new approach method methods "
    "system systems model models use survey review study analysis").split())
# obvious paper / thesis / non-textbook title phrasing
_PAPER = re.compile(
    r"\b(using|towards|a (survey|study|novel|new)|detection of|based on|analysis of|"
    r"design and implementation|implementation of|optimi[sz]ation of|prediction of|"
    r"application of|research on algorithm|effect of|case study|approach for|"
    r"framework for|method for|perancangan|rancang bangun|penerapan)\b", re.I)


def _nonascii(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / max(1, len(s))


def _toks(s: str):
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", s.lower()) if w not in _STOP]


def _coherence(samples) -> float:
    """Fraction of sample gap-article titles sharing the single most common content
    word -- a cheap proxy for a coherent (vs debris/random) gap."""
    eng = [s for s in samples if _nonascii(s) < 0.15]
    if len(eng) < 2:
        return 0.0
    c: Counter = Counter()
    for s in eng:
        c.update(set(_toks(s)))
    if not c:
        return 0.0
    return c.most_common(1)[0][1] / len(samples)


def main() -> None:
    files = sys.argv[1:] or ["artifacts/book_gaps/csfield_ranked.json",
                             "artifacts/book_gaps/ai_subfield_ranked.json"]
    cands: dict = {}
    for f in files:
        d = json.load(open(f))
        scope = d.get("scope_field") or d.get("scope_subfield") or "domain"
        for b in d.get("top_books", []):
            t, g, wid, samp = b["book"], b["gap_size"], b["book_id"], b.get("sample_titles", [])
            coh = _coherence(samp)
            ok = (g >= 120 and _nonascii(t) < 0.08 and coh >= 0.4 and not _PAPER.search(t))
            rec = {"gap": g, "title": t, "wid": wid, "coherence": round(coh, 2),
                   "scope": scope, "sample_titles": samp[:3], "passes": ok}
            if wid not in cands or g > cands[wid]["gap"]:
                cands[wid] = rec
    survivors = sorted([r for r in cands.values() if r["passes"]], key=lambda r: -r["gap"])
    print(f"=== {len(survivors)} genuine-book candidates (largest underserved gaps; ToC-check next) ===")
    print(f"{'gap':>5} {'coh':>4}  {'book':<50} W-id")
    for r in survivors[:40]:
        print(f"{r['gap']:5d} {r['coherence']:.2f}  {r['title'][:48]:<50} {r['wid'].rsplit('/', 1)[-1]}")
    out = "artifacts/book_gaps/curated_candidates.json"
    json.dump(survivors, open(out, "w"), indent=1)
    print(f"\nwrote {len(survivors)} -> {out}  (rejected {len(cands)-len(survivors)} as debris/paper/non-English)")


if __name__ == "__main__":
    main()
