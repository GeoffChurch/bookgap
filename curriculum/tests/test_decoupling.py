"""Guard tests: the package must not import bookgap (esp. in core)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent  # curriculum/


def _module_files(subdir: str) -> list[Path]:
    base = PKG / subdir if subdir else PKG
    return sorted(base.rglob("*.py"))


def _imports_bookgap(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "bookgap" or alias.name.startswith("bookgap."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "bookgap" or mod.startswith("bookgap."):
                return True
    return False


@pytest.mark.parametrize("path", _module_files(""), ids=lambda p: p.name)
def test_no_bookgap_import_anywhere(path: Path) -> None:
    assert not _imports_bookgap(path), f"{path} imports bookgap"


def test_core_subpackage_is_pure() -> None:
    offenders = [p for p in _module_files("core") if _imports_bookgap(p)]
    assert not offenders, f"core modules import bookgap: {offenders}"
