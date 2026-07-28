"""Curriculum Showdown Phase 1 — generate the pi mini-run briefs (agent + hybrid arms).

Deterministic text generation (no LLM). Writes to the learn-and-teach scratch
area (outputs/ is the sanctioned run-artifact area; repo files stay read-only):

    ~/src/learn-and-teach/outputs/showdown_p1/briefs/mini_<topic>_<angle>.md   (9)
    ~/src/learn-and-teach/outputs/showdown_p1/briefs/hybrid_<topic>.md         (3)

Topic statements / audiences / H1s / hybrid rules are the worklog-locked ones.
Run: uv run python -m artifacts_scripts.gap_detection.p1_make_briefs
"""

from __future__ import annotations

from pathlib import Path

BRIEFS = Path("~/src/learn-and-teach/outputs/showdown_p1/briefs").expanduser()

TOPICS = {
    "datascience": {
        "title": "Data Science",
        "statement": "Data Science",
        "audience": "quantitative-major undergraduates (calculus, linear "
                    "algebra, programming)",
    },
    "causal": {
        "title": "Causal Inference",
        "statement": "Causal Inference",
        "audience": "graduate students / quantitative researchers entering "
                    "causal inference from a general stats background",
    },
    "postcutoff": {
        "title": "Topic Modeling",
        "statement": "Topic Modeling — the field as it stands in 2026, i.e. "
                     "the research literature published since early 2025",
        "audience": "researchers/practitioners entering this literature in 2026",
    },
}

ANGLES = {
    "problems": "organize the book around the field's driving PROBLEMS — each "
                "chapter is a class of problems the audience must learn to "
                "solve, with methods introduced where a problem demands them",
    "methods": "organize the book around the field's METHODS and theory — "
               "each chapter is a coherent methodological family, building "
               "from mathematical foundations to advanced machinery",
    "applications": "organize the book around APPLICATIONS and use-cases — "
                    "each chapter is anchored in a domain or application "
                    "area, pulling in the theory it needs",
}

FORMAT_BLOCK = """## Output (REQUIRED)

Write the finished table of contents to `./toc.md`. That file must contain ONLY the
table of contents — no process notes, no commentary, no tool logs. EXACT format:

# Table of Contents: {title}

## Chapter 1: <chapter title>
*Intent: <one line — what this chapter covers and why it is here>.*
- **1.1 <section title>**: <one-line intent>.
- **1.2 <section title>**: <one-line intent>.
**Dependencies:** <one line — which earlier chapters this chapter builds on, or "none">.

## Chapter 2: <chapter title>
(... repeat the same structure for every chapter ...)

Rules: number every chapter and section exactly as above; every chapter gets the
one-line *Intent* line, 3-8 sections each with a one-line intent, and the one-line
**Dependencies:** line. Order chapters so prerequisites come before their use.
When `./toc.md` is complete and in this exact shape, you are done.
"""

WEB_BLOCK = """## Survey the field first

Two research tools are staged in this directory (run them with bash; results print
to stdout):

- `python3 websearch.py "your query" -n 8` — web search.
- `python3 fetch_url.py <URL> --max-chars 12000` — fetch one page/PDF as clean text.

Use websearch.py/fetch_url.py to survey the field's actual structure: its subareas,
what is taught before what, what a course for this audience must cover, and what the
current literature emphasizes. Run several searches (and fetch a few of the most
useful hits) BEFORE you design the table of contents.
"""

MINI = """# Task: design a textbook table of contents — {title}

You are the sole author designing the table of contents for a new textbook.

- Topic: {statement}.
- Audience: {audience}. Pitch coverage, prerequisites, and depth for exactly this
  audience.
- Design angle for THIS draft: {angle}.

{web}
{fmt}
Work autonomously to completion; do not pause for input.
"""

HYBRID = """# Task: polish an existing textbook skeleton into a finished table of contents — {title}

You are given a fixed skeleton for a textbook's table of contents in
`./{skeleton_name}` (staged in this directory). Its chapters and sections are
labeled only by raw distinctive-term tuples plus two exemplar work titles per
section, in a FIXED order.

- Topic: {statement}.
- Audience: {audience}.

## Your job

Produce a polished table of contents from that skeleton: rename chapters/sections
into real titles, write the intents, you MAY merge at most 2 adjacent chapters, you
MUST NOT reorder or move content between chapters.

In detail:
1. Read `./{skeleton_name}` in full.
2. For every chapter and section, infer what literature the term-label + exemplar
   titles describe, and give it a real textbook-quality title.
3. Write the one-line intent for every chapter and every section (what it covers and
   why it sits at that point of the book).
4. Keep the given chapter order and the given section order within each chapter
   EXACTLY. You may merge at most 2 adjacent chapters into one (if two neighbors are
   really one subject); otherwise the chapter count stays the same. Never move a
   section from one chapter into another, never reorder.
5. Write the one-line **Dependencies:** note per chapter (which earlier chapters it
   builds on, or "none").

You may use the staged research tools to check what a term or an exemplar title
refers to: `python3 websearch.py "query" -n 8` and
`python3 fetch_url.py <URL> --max-chars 12000`.

{fmt}
Work autonomously to completion; do not pause for input.
"""


def main() -> None:
    BRIEFS.mkdir(parents=True, exist_ok=True)
    n = 0
    for tkey, t in TOPICS.items():
        fmt = FORMAT_BLOCK.format(title=t["title"])
        for akey, angle in ANGLES.items():
            p = BRIEFS / f"mini_{tkey}_{akey}.md"
            p.write_text(MINI.format(title=t["title"], statement=t["statement"],
                                     audience=t["audience"], angle=angle,
                                     web=WEB_BLOCK, fmt=fmt))
            n += 1
        p = BRIEFS / f"hybrid_{tkey}.md"
        p.write_text(HYBRID.format(title=t["title"], statement=t["statement"],
                                   audience=t["audience"],
                                   skeleton_name=f"p1_skeleton_{tkey}.md",
                                   fmt=fmt))
        n += 1
    print(f"wrote {n} briefs under {BRIEFS}")


if __name__ == "__main__":
    main()
