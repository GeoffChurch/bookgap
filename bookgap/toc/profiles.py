"""Blind structural ToC extractors over book abstracts.

Each Profile is a pure str -> ParsedToc | None. A parse is kept only when it
yields an ordered, sanely-titled chapter sequence -- the structure IS the
filter, so profiles run blindly on every abstract with no venue gating.

Part-aware: when the abstract contains >=2 keyword "PART"/"SECTION" markers,
chapters are nested under their enclosing part (restart-tolerant: chapter
numbering may reset to 1 across a part boundary). Back-matter (appendices,
glossary, index, ...) is truncated at the abstract level before any marker
scan, so back-matter words never need to be stripped out of a title.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TocNodeRaw:
    title: str
    kind: str = "chapter"
    children: tuple["TocNodeRaw", ...] = ()


@dataclass(frozen=True)
class ParsedToc:
    profile: str
    nodes: tuple[TocNodeRaw, ...]


Profile = Callable[[str], "ParsedToc | None"]
PROFILES: dict[str, Profile] = {}

_HTML = re.compile(r"<[^>]+>")
# embedded structural divider leaked into a title (bare-roman-before-ALLCAPS / SECTION / PART):
_DIV = re.compile(r"\.\s+(?:[IVXLC]+\.\s+[A-Z][A-Z]|SECTION\b|PART\b).*$")
# structural trailers ONLY -- back-matter words are handled by _truncate_backmatter
# at the abstract level, so they must NOT be stripped here (that would mangle
# legitimate titles like "Soil Index Properties" -> "Soil").
_TRAILER = re.compile(r"\s+(?:PART\s+[IVXLC\d]+|Summary|Session|부록)\b.*$", re.I)
_CONNECTOR = re.compile(r"\s*--\s*[A-Za-z]{0,8}\.?\s*$")  # trailing "-- ch" / "-- Part"
_CHAP_TAIL = re.compile(r"\s+Chapter(?:\s+\d+)?\s*$", re.I)
_PAGE_TAIL = re.compile(r"(?:\s+|(?<=[a-z]))\d{1,4}\s*$")


def _norm(s: str | None) -> str:
    return (s or "").replace("\xa0", " ").replace("–", "-")


def _clean_title(raw: str) -> str:
    t = raw.replace("\xa0", " ").replace("–", "-").replace("[[MISSING]]", " ")
    t = _HTML.sub(" ", t)
    t = _DIV.sub("", t)
    t = _TRAILER.sub("", t)
    t = _CONNECTOR.sub("", t)
    t = _CHAP_TAIL.sub("", t)
    t = _PAGE_TAIL.sub("", t)
    return re.sub(r"\s+", " ", t).strip(" .:-,;\t")


_BACK = re.compile(
    r"(?:^|[.\s;])\s*(?:APPENDIX|APPENDICES|GLOSSARY|REFERENCES|BIBLIOGRAPHY|"
    r"LIST OF ACRONYMS|Problems\s+Solutions)\b|\.\s*Index\b(?!\s+[A-Z][a-z])", re.I)


def _truncate_backmatter(ab: str) -> str:
    for m in _BACK.finditer(ab):
        if m.start() > 40:
            return ab[:m.start()]
    return ab


_SPELLED = ("ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|"
            "THIRTEEN|FOURTEEN")
_PART = re.compile(r"(?:(?<=[.\s;\-])|^)(?:PART|SECTION|Part|Section)\s+"
                   r"(?:" + _SPELLED + r"|[IVXLC]+|\d+)\b[.:]?\s*")


def _find_parts(ab: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PART.finditer(ab)]


# ----- marker finders (one per profile) -> list[(start, end, number)] -----
_CH = re.compile(r"(?:^|[\s.;:,\-—])(?:CHAPTER|Chapter|Ch\.)\s+(\d{1,3})\b[.:\s]*", re.I)
_ARA = re.compile(r"(?<![\d.])(\d{1,3})[.:)](?!\d)\s+")
_ROM = re.compile(r"(?<![a-zA-Z])(i{1,3}|iv|vi{0,3}|ix|xi{0,3})\.\s+", re.I)
_ROMORD = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv"]


def markers_chapter_n(ab: str) -> list[tuple[int, int, int]]:
    return [(m.start(), m.end(), int(m.group(1))) for m in _CH.finditer(ab)]


def markers_arabic(ab: str) -> list[tuple[int, int, int]]:
    return [(m.start(), m.end(), int(m.group(1))) for m in _ARA.finditer(ab)]


def markers_roman(ab: str) -> list[tuple[int, int, int]]:
    return [(m.start(), m.end(), _ROMORD.index(m.group(1).lower()) + 1)
            for m in _ROM.finditer(ab) if m.group(1).lower() in _ROMORD]


def assemble(abstract: str, markers_fn: Callable[[str], list[tuple[int, int, int]]],
             name: str, *, min_leaves: int = 3) -> "ParsedToc | None":
    ab = _norm(_truncate_backmatter(abstract))
    parts = _find_parts(ab)
    in_part = lambda p: any(s <= p < e for s, e in parts)
    div_between = lambda a, b: any(a < s < b for s, e in parts)

    chap = [(s, e, n) for (s, e, n) in markers_fn(ab) if not in_part(s)]

    # restart-tolerant ordered selection (reset-to-1 allowed only across a keyword part)
    kept: list[tuple[int, int, int]] = []
    exp: int | None = None
    for (s, e, n) in chap:
        if exp is None:
            if n in (0, 1):
                kept.append((s, e, n))
                exp = n + 1
        elif n == exp:
            kept.append((s, e, n))
            exp += 1
        elif n == 1 and kept and div_between(kept[-1][0], s):
            kept.append((s, e, n))
            exp = 2
        else:
            break

    stops = sorted([s for s, _, _ in kept] + [s for s, _ in parts] + [len(ab)])

    def _after(end: int, cap: int) -> str:
        nxt = min([x for x in stops if x > end], default=min(len(ab), end + cap))
        return _clean_title(ab[end:min(nxt, end + cap)])

    chapters = [(s, _after(e, 140)) for (s, e, n) in kept]

    def _plabel(pe: int) -> str:
        nxt = min([x for x in stops if x > pe], default=min(len(ab), pe + 90))
        return _clean_title(ab[pe:min(nxt, pe + 90)].split(".")[0])  # bound at first period

    if len(parts) >= 2:
        first = parts[0][0]
        tree: list[TocNodeRaw] = [
            TocNodeRaw(title=t) for (s, t) in chapters if s < first and t]
        part_nodes: list[TocNodeRaw] = []
        for i, (ps, pe) in enumerate(parts):
            nxt = parts[i + 1][0] if i + 1 < len(parts) else len(ab)
            kids = tuple(TocNodeRaw(title=t) for (cs, t) in chapters if pe <= cs < nxt and t)
            part_nodes.append(TocNodeRaw(title=_plabel(pe), kind="part", children=kids))
        seen: dict[str, TocNodeRaw] = {}
        order: list[str] = []
        for n in part_nodes:
            k = n.title.lower()
            if k not in seen:
                order.append(k)
                seen[k] = n
            elif len(n.children) > len(seen[k].children):
                seen[k] = n
        tree += [seen[k] for k in order]
    else:
        tree = [TocNodeRaw(title=t) for (s, t) in chapters if t]

    def _leaves(ns: list[TocNodeRaw]) -> int:
        return sum(_leaves(list(n.children)) if n.children else 1 for n in ns)

    def _ltitles(ns: list[TocNodeRaw]) -> list[str]:
        out: list[str] = []
        for n in ns:
            out += _ltitles(list(n.children)) if n.children else [n.title]
        return out

    if _leaves(tree) < min_leaves:
        return None
    lt = _ltitles(tree)
    if sum(1 for t in lt if t and not t[0].islower()) < 0.7 * len(lt):
        return None
    return ParsedToc(profile=name, nodes=tuple(tree))


def chapter_n_profile(abstract: str) -> "ParsedToc | None":
    return assemble(abstract, markers_chapter_n, "chapter-n")


PROFILES["chapter-n"] = chapter_n_profile


def arabic_list_profile(abstract: str) -> "ParsedToc | None":
    return assemble(abstract, markers_arabic, "arabic-list")


PROFILES["arabic-list"] = arabic_list_profile


def roman_list_profile(abstract: str) -> "ParsedToc | None":
    # roman numerals collide easily with prose ("i.", "v."); require a longer
    # ordered run than the chapter-n/arabic profiles before trusting a parse.
    return assemble(abstract, markers_roman, "roman-list", min_leaves=5)


PROFILES["roman-list"] = roman_list_profile
