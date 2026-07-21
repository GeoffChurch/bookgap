"""Blind structural ToC extractors over book abstracts.

Each Profile is a pure str -> ParsedToc | None. A parse is kept only when it
yields an ordered, sanely-titled chapter sequence -- the structure IS the
filter, so profiles run blindly on every abstract with no venue gating.
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
# Trailing structural markers that are not part of a chapter title. PART is
# matched case-insensitively but only when followed by a roman/number token.
_TRAILER = re.compile(
    r"\s+(?:PART\s+[IVXLC\d]+|Index|Appendix|Author index|Object index|"
    r"Bibliography|Summary|Session|부록)\b.*$", re.I)
_PAGE_TAIL = re.compile(r"\s+\d{1,4}\s*$")


def _norm(s: str | None) -> str:
    return (s or "").replace("\xa0", " ").replace("–", "-")


def _clean_title(raw: str) -> str:
    t = raw.replace("\xa0", " ").replace("–", "-").replace("[[MISSING]]", " ")
    t = _HTML.sub(" ", t)
    t = _TRAILER.sub("", t)
    t = _PAGE_TAIL.sub("", t)
    return re.sub(r"\s+", " ", t).strip(" .:-\t")


def _slice_titles(ab: str, ms: list[re.Match]) -> list[str]:
    out: list[str] = []
    for i, m in enumerate(ms):
        start = m.end()
        end = ms[i + 1].start() if i + 1 < len(ms) else min(len(ab), start + 120)
        out.append(_clean_title(ab[start:end]))
    return out


def _titles_sane(titles: list[str], frac: float = 0.7) -> bool:
    if not titles:
        return False
    ok = sum(1 for t in titles if 2 <= len(t) <= 100)
    return ok >= len(titles) * frac


def _parsed(profile: str, titles: list[str]) -> "ParsedToc | None":
    nodes = tuple(TocNodeRaw(title=t) for t in titles if t)
    if len(nodes) < 3:
        return None
    return ParsedToc(profile=profile, nodes=nodes)


_CH = re.compile(r"(?:^|[\s.;:\-—])(?:CHAPTER|Chapter|Ch\.)\s+(\d{1,3})\b[.:\s]*")


def chapter_n_profile(abstract: str, *, minimum: int = 3) -> "ParsedToc | None":
    ab = _norm(abstract)
    ms = list(_CH.finditer(ab))
    if len(ms) < minimum:
        return None
    nums = [int(m.group(1)) for m in ms]
    if nums != sorted(nums) or len(set(nums)) != len(nums):
        return None
    if nums[0] not in (0, 1):
        return None
    titles = _slice_titles(ab, ms)
    if not _titles_sane(titles):
        return None
    return _parsed("chapter-n", titles)


PROFILES["chapter-n"] = chapter_n_profile


_ARA = re.compile(r"(?<![\d.])(\d{1,3})\.(?!\d)\s+")


def arabic_list_profile(abstract: str, *, minimum: int = 5) -> "ParsedToc | None":
    ab = _norm(abstract)
    ms = list(_ARA.finditer(ab))
    if len(ms) < minimum:
        return None
    nums = [int(m.group(1)) for m in ms]
    if nums[0] != 1:
        return None
    k = 1
    while k < len(nums) and nums[k] == nums[k - 1] + 1:
        k += 1
    if k < minimum:
        return None
    titles = _slice_titles(ab, ms[:k])
    if not _titles_sane(titles, frac=0.8):
        return None
    return _parsed("arabic-list", titles)


PROFILES["arabic-list"] = arabic_list_profile


_ROM = re.compile(r"(?<![a-zA-Z])(i{1,3}|iv|vi{0,3}|ix|xi{0,3})\.\s+", re.I)
_ROMORD = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]


def roman_list_profile(abstract: str, *, minimum: int = 5) -> "ParsedToc | None":
    ab = _norm(abstract)
    ms = list(_ROM.finditer(ab))
    if len(ms) < minimum:
        return None
    labs = [m.group(1).lower() for m in ms]
    if labs[0] != "i":
        return None
    k = 1
    while k < len(labs) and k < len(_ROMORD) and labs[k] == _ROMORD[k]:
        k += 1
    if k < minimum:
        return None
    titles = _slice_titles(ab, ms[:k])
    if not _titles_sane(titles, frac=0.8):
        return None
    return _parsed("roman-list", titles)


PROFILES["roman-list"] = roman_list_profile
