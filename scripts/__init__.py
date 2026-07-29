"""Standalone helper scripts (not part of the installed bookgap distribution).

This file exists to make `scripts` a *regular* package rather than a PEP 420
namespace package. The sibling curriculum repo, which bookgap depends on as an
editable install, also ships a top-level `scripts` package with its own
__init__.py. Python resolves a regular package ahead of any namespace portion
regardless of sys.path order, so without this file curriculum's `scripts`
shadows this directory and `from scripts.capture_gap_parity_golden import ...`
fails with a confusing "No module named scripts.capture_gap_parity_golden" --
the parent resolves, the submodule does not.

Paired with `pythonpath = ["."]` in pyproject.toml's pytest config, which puts
the repo root on sys.path in the first place.
"""
