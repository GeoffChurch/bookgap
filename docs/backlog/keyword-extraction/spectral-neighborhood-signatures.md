# Spectral neighborhood signatures for token quality

(Migrated from `artifacts_scripts/curriculum/TODO.md`. Original cross-pollination from `mycooc/docs/backlog/signatures-and-features/spectral-neighborhood-signatures.md`.)

Take each token's k×k cooc submatrix (its k nearest neighbours in cooc space), compute sorted eigenvalues as a signature. Boilerplate tokens form *degenerate* neighborhoods (all pointing to one hub); genuine topic tokens form *richer* neighborhoods.

**Use cases**:
- Rank candidates pre-keyword-extraction
- De-rank tokens whose spectral signature flags as boilerplate-shaped

**Status**: idea-only. Not yet prototyped.
