# paper/ repo: reconsider CI

Follow-up noticed while setting up the paper build/watch/commit process
(2026-07-28). The build tooling itself is done and lives in `paper/` —
`Makefile`, `.latexmkrc`, `.githooks/`, `README.md`.

Note `paper/` is a **separate git repo** (remote `sbu-algorithms/nextbook`),
excluded from bookgap via `/paper/` in `.gitignore:28`. Changes there are
committed separately.

*(The `*_OLD*` prune is **done** — paper commit `da8898b`, 2026-07-29, removed
all 101 files. None was referenced by any `.tex` even at the EMNLP submission
commit, and rebuilding after deletion produced an identical word stream. The
`related_work*` ambiguity resolved too: `related_work.tex` is a foracl /
notforacl switch and both branches are live; only the `_OLD` fourth variant was
dead. `main_new.pdf` / `main_old.pdf` are gone in favour of `make diff`.)*

## Reconsider CI

Explicitly declined on 2026-07-28 in favour of local hooks only, on the grounds
that `sbu-algorithms/nextbook` is a shared org repo. Worth revisiting because
of a hazard the local hook cannot cover:

**`pdftotext` cannot recover `fi`/`fl` ligatures from the Overleaf-built PDF**
(`identified` → `identi ed`, `filled` → `lled`), while the local TeX Live 2023
build extracts them fine. So a collaborator who builds on Overleaf and commits
will trip the hook spuriously on every ligature word — and if they instead
commit with `--no-verify`, nothing catches the resulting staleness.

CI would be the backstop. The workflow needs a `texlive-full` image with
`-shell-escape` enabled (minted), and should reuse the `pdftotext -raw`
normalised-word comparison from `.githooks/pre-commit` rather than `-layout`
(measured: `-layout` reports 1141 differing lines between two builds of
identical source; `-raw` normalised reports 0).

Cheaper partial mitigation: tell collaborators to use `make docker`, which
pins the same image for everyone.
