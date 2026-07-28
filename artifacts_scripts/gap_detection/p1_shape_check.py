"""P1 scratch: shape-lint every mini/hybrid run toc.md via the module parser."""
from artifacts_scripts.gap_detection.showdown_p1 import (
    RUNS_TSV,
    parse_toc_md,
    read_manifest,
    section_rep_texts,
)

runs = read_manifest(RUNS_TSV)
for key, rd in sorted(runs.items()):
    p = rd / "toc.md"
    if not p.is_file():
        print(f"{key}: NO toc.md in {rd}")
        continue
    t = p.read_text(encoding="utf-8")
    try:
        parsed = parse_toc_md(t)
        texts, ci = section_rep_texts(parsed)
        print(
            f"{key}: ch={len(parsed['chapters'])} secs={len(texts)} "
            f"fallback_ch={parsed['n_fallback_chapters']}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"{key}: PARSE FAIL {e}")
