"""Export the three sides of one ToC-reconstruction case as readable markdown:

  1. ours_curriculum.md         -- our generated curriculum (Track A, from the book's gap)
  2. original_book.md           -- the real book: title + abstract + table of contents
  3. toc_derived_curriculum.md  -- the curriculum derived from the book's real ToC (Track B)

Reusable across books. Tracks A/B are read straight from their curriculum run-dirs
(via eval_book_judge.load_book_outline -- same parse the LLM judge uses), so the
markdown matches exactly what was evaluated. The book's abstract is fetched from the
OpenAlex API (falls back to the ToC json's description for external books).

Example:
  .conda/bin/python -m artifacts_scripts.curriculum.export_curriculum_markdown \
      --track-a artifacts/curriculum/mcs_trackA \
      --track-b artifacts/curriculum/mcs_trackB \
      --toc-json artifacts/book_gaps/W4235013513_toc.json \
      --out-dir docs/curriculum_eval_examples/mobile_crowdsensing
"""
import argparse
import json
import urllib.request
from pathlib import Path

from artifacts_scripts.curriculum.eval_book_judge import load_book_outline


def _render_curriculum_md(run_dir: str, heading: str, n_keywords: int = 30, n_works: int = 5) -> str:
    bo = load_book_outline(Path(run_dir))
    out = [f"# {bo.topic} — {heading}", "",
           f"*{len(bo.chapters)} chapters, in learning order.*", ""]
    for ch in bo.chapters:
        out += [f"## Chapter {ch.idx}: {ch.label}", ""]
        # Keep underscores so multi-word phrase-mined terms (e.g. neural_networks)
        # read as single units rather than blending into the comma list as unigrams.
        kw = ", ".join(ch.keywords[:n_keywords]) or "(none)"
        out += [f"**Key topics:** {kw}", ""]
        if ch.titles:
            out.append("**Representative works:**")
            out += [f"- {t}" for t in ch.titles[:n_works]]
            out.append("")
    return "\n".join(out)


def _fetch_abstract(book_id) -> str | None:
    if not book_id or not str(book_id).startswith("https://openalex.org/W"):
        return None
    try:
        wid = str(book_id).rsplit("/", 1)[-1]
        url = (f"https://api.openalex.org/works/{wid}"
               "?select=abstract_inverted_index&mailto=geoffrey.a.churchill@gmail.com")
        with urllib.request.urlopen(url, timeout=25) as r:
            inv = json.load(r).get("abstract_inverted_index")
        if not inv:
            return None
        pos = {i: w for w, idxs in inv.items() for i in idxs}
        return " ".join(pos[i] for i in sorted(pos))
    except Exception:
        return None


def _render_book_md(toc_json: str) -> str:
    toc = json.loads(Path(toc_json).read_text())
    title, wid = toc["title"], toc.get("book_id")
    abstract = _fetch_abstract(wid) or toc.get("query_text") or "*(abstract unavailable)*"
    out = [f"# {title} — original book", "",
           f"**OpenAlex id:** {wid}", "",
           "## Abstract", "", abstract, "",
           "## Table of Contents (the real book)", ""]
    for c in toc["chapters"]:
        line = f"{c['n']}. **{c['title']}**"
        desc = c.get("description")
        if desc:
            line += f" — {desc}"
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-a", required=True, help="our curriculum run-dir (Track A)")
    ap.add_argument("--track-b", help="ToC-derived curriculum run-dir (Track B); not needed with --only-ours")
    ap.add_argument("--toc-json", help="the book's ToC json (title + chapters + book_id); not needed with --only-ours")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--only-ours", action="store_true",
                    help="regenerate only ours_curriculum.md (Track A); leave the book + Track B "
                         "files untouched. Use after improving the Track A curriculum.")
    args = ap.parse_args()
    if not args.only_ours and (not args.track_b or not args.toc_json):
        ap.error("--track-b and --toc-json are required unless --only-ours is set")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ours_curriculum.md").write_text(
        _render_curriculum_md(args.track_a, "our generated curriculum (from the book's gap)"))
    if args.only_ours:
        print(f"wrote ours_curriculum.md -> {args.out_dir}")
        return
    (args.out_dir / "toc_derived_curriculum.md").write_text(
        _render_curriculum_md(args.track_b, "curriculum derived from the book's real ToC"))
    (args.out_dir / "original_book.md").write_text(_render_book_md(args.toc_json))
    print(f"wrote ours_curriculum.md, original_book.md, toc_derived_curriculum.md -> {args.out_dir}")


if __name__ == "__main__":
    main()
