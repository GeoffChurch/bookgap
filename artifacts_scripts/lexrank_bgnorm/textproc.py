"""Text processing for the bg-norm LexRank eval: wikitext finishing, splitting, tokens.

Vendored on purpose: the bookgap conda env has neither ``mwparserfromhell`` nor
``nltk``, and the house rule is "no pip installs into the owner's env — vendor a
minimal implementation and document the divergence".

Input corpus: ``bookgen``'s ``data/corpus/*.txt`` files are PARTIALLY cleaned
wikitext (its ``src/eval/scraper.py`` already removed ``<ref>`` tags,
cite/infobox/navbox/reflist templates, and Category/File/Image links). This
module finishes the job: comments, tables, remaining templates (with a small
keep-map for text-bearing ones), HTML tags, wikilinks, external links,
bold/italic quotes, entities — while PRESERVING ``== heading ==`` lines so the
lead (text before the first heading) can be split from the body.

Divergences from a full wikitext renderer (documented, applied uniformly to all
compared methods):
- unknown templates are dropped wholesale (mwparserfromhell's ``strip_code``
  does the same); a tiny keep-map renders {{convert}}, {{lang}}, {{nowrap}}-like
  wrappers to their visible text so sentences keep their values;
- list/table/caption lines are dropped (bullets are not prose sentences);
- the sentence splitter is a rule-based scanner (abbreviation list, initials,
  decimals), not a trained model.
"""

from __future__ import annotations

import html
import re

# --------------------------------------------------------------------------- #
# stopwords (small functional-word list; content numbers/years are kept)
# --------------------------------------------------------------------------- #

STOPWORDS = frozenset(
    """
    a an the and or but nor if then else when while where which who whom whose what why how
    this that these those there here is are was were be been being am do does did doing
    have has had having can will shall should would could may might must
    i me my mine we us our ours you your yours he him his she her hers it its they them
    their theirs himself herself itself themselves myself yourself ourselves
    of at by for to in on from as with into onto through during before after above below
    between among under over again further out off up down about against
    not no so than too very just also such only own same other another both each few
    more most some any all once because until upon within without along across behind
    beyond per via amid toward towards
    s t d ll m re ve y
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def word_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens (mirrors rouge_score's default tokenizer)."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """``word_tokens`` minus stopwords and single-character tokens (B-matrix stream)."""
    return [t for t in word_tokens(text) if len(t) > 1 and t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# wikitext finishing cleaner
# --------------------------------------------------------------------------- #

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MAGIC_RE = re.compile(r"__[A-Z]+__")
# Tags whose CONTENT must be dropped too (not prose).
_DROP_CONTENT_TAGS = ("math", "gallery", "timeline", "score", "ref", "imagemap",
                      "syntaxhighlight", "source", "chem", "hiero", "mapframe", "graph")
_DROP_CONTENT_RES = [
    re.compile(rf"<{t}\b[^>/]*>.*?</{t}\s*>", re.DOTALL | re.IGNORECASE) for t in _DROP_CONTENT_TAGS
] + [re.compile(rf"<{t}\b[^>]*/>", re.IGNORECASE) for t in _DROP_CONTENT_TAGS]
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]*>")  # remaining tags: strip tag, keep inner text

_EXT_LINK_LABELED_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\s+([^\]]*)\]")
_EXT_LINK_BARE_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+\]")
_QUOTES_RE = re.compile(r"'{2,5}")

_HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")

#: level-2 sections dropped from the body (reference cruft, not prose).
DROP_SECTIONS = frozenset({
    "references", "external links", "see also", "notes", "footnotes", "citations",
    "bibliography", "further reading", "sources", "works cited", "gallery",
    "explanatory notes", "notes and references", "references and notes",
    "references and sources", "sources and references", "citations and notes",
})


def _strip_spans(text: str, open_tok: str, close_tok: str, render) -> str:
    """Remove/replace nested ``open_tok``...``close_tok`` spans (outermost-first scan).

    ``render(inner)`` maps the INNER text of each top-level span to replacement
    text. Unbalanced opens swallow to end-of-text (matches renderer behaviour on
    malformed markup).
    """
    out: list[str] = []
    i, n = 0, len(text)
    lo, lc = len(open_tok), len(close_tok)
    while i < n:
        if text.startswith(open_tok, i):
            depth, j = 1, i + lo
            while j < n and depth:
                if text.startswith(open_tok, j):
                    depth += 1
                    j += lo
                elif text.startswith(close_tok, j):
                    depth -= 1
                    j += lc
                else:
                    j += 1
            inner = text[i + lo: j - lc] if depth == 0 else text[i + lo:]
            out.append(render(inner))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _split_top_level(text: str, sep: str = "|") -> list[str]:
    """Split on ``sep`` ignoring separators nested inside ``[[...]]`` / ``{{...}}``."""
    parts: list[str] = []
    buf: list[str] = []
    depth_link = depth_tmpl = 0
    i, n = 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "[[":
            depth_link += 1
            buf.append(two)
            i += 2
        elif two == "]]":
            depth_link = max(0, depth_link - 1)
            buf.append(two)
            i += 2
        elif two == "{{":
            depth_tmpl += 1
            buf.append(two)
            i += 2
        elif two == "}}":
            depth_tmpl = max(0, depth_tmpl - 1)
            buf.append(two)
            i += 2
        elif text[i] == sep and depth_link == 0 and depth_tmpl == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(text[i])
            i += 1
    parts.append("".join(buf))
    return parts


_NAMED_PARAM_RE = re.compile(r"^\s*[\w\- ]+\s*=")


def _render_template(inner: str) -> str:
    """Render a template's visible text via a tiny keep-map; default: drop."""
    inner = _strip_spans(inner, "{{", "}}", _render_template)  # nested first
    parts = _split_top_level(inner)
    name = parts[0].strip().lower()
    positional = [p for p in parts[1:] if not _NAMED_PARAM_RE.match(p)]

    if name in ("convert", "cvt") and len(positional) >= 2:
        # {{convert|14|km|mi}} -> "14 km"; ranges {{convert|3|to|5|km}} -> "3 to 5 km"
        joiners = {"to", "-", "and", "or", "by", "x", "×"}
        toks = [positional[0]]
        k = 1
        while k + 1 < len(positional) and positional[k].strip().lower() in joiners:
            toks += [positional[k].strip(), positional[k + 1]]
            k += 2
        if k < len(positional):
            toks.append(positional[k])
        return " ".join(t.strip() for t in toks)
    if name in ("lang", "langx", "transliteration", "transl") and len(positional) >= 2:
        return positional[1].strip()
    if (name.startswith("lang-") or name.startswith("lang ")) and positional:
        return positional[0].strip()
    if name in ("nowrap", "small", "em", "strong", "visible anchor", "vanchor",
                "nobr", "big") and positional:
        return positional[0].strip()
    if name in ("spnd", "snd", "ndash", "endash", "spaced ndash", "dash", "mdash", "emdash"):
        return " – "
    if name in ("'", "`", "circa", "c."):
        return "c. " if name in ("circa", "c.") else "'"
    return ""


def finish_clean_wikitext(text: str) -> str:
    """Finish cleaning bookgen's partially-cleaned wikitext into headed plain text.

    Heading lines (``== ... ==``) are preserved for the lead/body split.
    """
    text = _COMMENT_RE.sub("", text)
    text = _MAGIC_RE.sub("", text)
    for rx in _DROP_CONTENT_RES:
        text = rx.sub(" ", text)
    text = _strip_spans(text, "{|", "|}", lambda inner: "")     # tables (may contain templates)
    text = _strip_spans(text, "{{", "}}", _render_template)     # remaining templates
    text = _BR_RE.sub(" ", text)
    text = _TAG_RE.sub("", text)                                # keep inner text of other tags

    def _render_link(inner: str) -> str:
        # [[target|label]] -> label, [[target]] -> target; drop residual ns links.
        parts = _split_top_level(inner)
        target = parts[0].strip()
        low = target.lower()
        if ":" in low and low.split(":", 1)[0] in (
                "category", "file", "image", "wikt", "wiktionary", "s", "commons"):
            return parts[-1].strip() if len(parts) > 1 else ""
        label = parts[-1].strip() if len(parts) > 1 else target
        return label.split("#", 1)[0] if label == target else label

    text = _strip_spans(text, "[[", "]]", _render_link)
    text = _EXT_LINK_LABELED_RE.sub(r"\1", text)
    text = _EXT_LINK_BARE_RE.sub("", text)
    text = _QUOTES_RE.sub("", text)
    text = html.unescape(text).replace("\xa0", " ")
    return text


_LIST_LINE_RE = re.compile(r"^\s*[*#;:|!]")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def _prose_lines(lines: list[str]) -> str:
    """Keep prose lines only (drop bullets/table residue/no-alpha lines)."""
    kept: list[str] = []
    for ln in lines:
        if _LIST_LINE_RE.match(ln):
            continue
        if ln.strip() and not _ALPHA_RE.search(ln):
            continue
        kept.append(ln)
    return "\n".join(kept)


def split_lead_body(cleaned: str) -> tuple[str, str]:
    """Split finished-cleaned text into (lead, body) at the first heading.

    The body drops blocklisted level-2 sections (References etc.) including
    their subsections, and drops the heading lines themselves.
    """
    lines = cleaned.split("\n")
    lead_lines: list[str] = []
    body_lines: list[str] = []
    in_body = False
    skipping = False
    for ln in lines:
        m = _HEADING_RE.match(ln)
        if m:
            in_body = True
            level = len(m.group(1))
            if level == 2:
                skipping = m.group(2).strip().lower() in DROP_SECTIONS
            continue  # heading lines are never sentences
        if not in_body:
            lead_lines.append(ln)
        elif not skipping:
            body_lines.append(ln)
    return _prose_lines(lead_lines), _prose_lines(body_lines)


# --------------------------------------------------------------------------- #
# sentence splitter
# --------------------------------------------------------------------------- #

#: lowercased abbreviations (no trailing dot) after which '. ' is NOT a boundary.
ABBREVIATIONS = frozenset(
    """
    mr mrs ms dr prof st sr jr rev hon fr gen col lt capt sgt maj adm gov sen rep pres
    mt ft no nos vol vols pp p ch chs sec fig figs ed eds trans etc eg ie cf ca viz vs v
    approx est dept univ co inc ltd corp bros al r b d op art
    jan feb mar apr jun jul aug sep sept oct nov dec
    """.split()
)

_BOUNDARY_RE = re.compile(r"[.!?]+[\"'”’)\]]*\s+")
_OPENERS = "\"'“‘([A-Z0-9"
_PREV_TOKEN_RE = re.compile(r"(\S+)$")
_DOTTED_ABBREV_RE = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]?$")  # U.S. / J.R.R. / Ph.D-ish

MIN_SENT_TOKENS = 4  # prep filter: shorter candidates are markup residue, not prose


def _is_boundary(text: str, m: re.Match[str]) -> bool:
    nxt = text[m.end()] if m.end() < len(text) else ""
    if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in _OPENERS):
        return False
    if "." not in m.group(0):
        return True  # ! or ? always end a sentence (given a valid opener follows)
    prev = _PREV_TOKEN_RE.search(text, 0, m.start())
    if not prev:
        return True
    tok = prev.group(1)
    full = tok + m.group(0).strip()
    if _DOTTED_ABBREV_RE.match(full):
        return False  # "U.S.", "J. R. R." initials, "Ph.D." (single letters + dots)
    word = tok.rstrip(".").split(".")[-1].lower().lstrip("\"'“‘([")
    if word in ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isalpha() and tok[-1:].isupper():
        return False  # single-letter initial like "J."
    return True


def split_sentences(text: str, min_tokens: int = MIN_SENT_TOKENS) -> list[str]:
    """Split prose into sentences (paragraph breaks are hard boundaries).

    Candidates with fewer than ``min_tokens`` word-tokens (or no alphabetic
    token) are dropped as markup residue. This prep filter applies uniformly
    to every compared method.
    """
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        flat = " ".join(para.split())
        if not flat:
            continue
        start = 0
        for m in _BOUNDARY_RE.finditer(flat):
            if _is_boundary(flat, m):
                sentences.append(flat[start: m.end()].strip())
                start = m.end()
        tail = flat[start:].strip()
        if tail:
            sentences.append(tail)
    return [
        s for s in sentences
        if len(word_tokens(s)) >= min_tokens and _ALPHA_RE.search(s)
    ]
