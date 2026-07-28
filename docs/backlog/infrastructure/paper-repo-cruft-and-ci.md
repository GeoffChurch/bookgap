# paper/ repo: prune `*_OLD*` cruft, and reconsider CI

Follow-ups noticed while setting up the paper build/watch/commit process
(2026-07-28). The build tooling itself is done and lives in `paper/` —
`Makefile`, `.latexmkrc`, `.githooks/`, `README.md`. These are the two things
deliberately left undone.

Note `paper/` is a **separate git repo** (remote `sbu-algorithms/nextbook`),
excluded from bookgap via `/paper/` in `.gitignore:28`. Changes there are
committed separately.

## 1. Prune the `*_OLD*` tracked files

101 tracked files sit under superseded paths:

- `artifacts/latex_tables_OLD/` and `artifacts/latex_tables_OLD2/`
- `related_work_OLD.tex` (alongside `related_work.tex`,
  `related_work_foracl.tex`, `related_work_notforacl.tex` — unclear which are live)

`main.tex` `\input`s none of the `_OLD` paths, so they are dead weight in a repo
whose `.git` is already 45 MB against a 5 MB working tree. Per the no-legacy
rule these should just be deleted — history retains them.

Before deleting, confirm the `related_work*` variants: `related_work.tex` is
121 bytes and looks like a switch that `\input`s one of the others, so at least
two of the four are live.

Also untracked and now superseded by `make diff`: `main_new.pdf`, `main_old.pdf`.

## 2. Reconsider CI

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
