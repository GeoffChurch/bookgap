"""Guard the coupling to the sibling ``curriculum`` repo.

bookgap consumes ``curriculum`` as an editable path dependency (``../curriculum``),
so the two drift independently and nothing on either side runs the other's tests.
That coupling broke bookgap twice within three days in July 2026, by two different
mechanisms, neither catchable by ``py_compile``:

1. ``curriculum`` c4bd916 removed the ``weight_by_freq`` keyword from
   ``extract_topic_keywords``. Three bookgap call sites kept passing it and raised
   ``TypeError`` at runtime. Two were committed that same day, already broken.
2. ``curriculum`` 229316b added a top-level ``scripts`` package with an
   ``__init__.py``, which shadowed bookgap's own ``__init__``-less ``scripts/``.
   Python resolves a regular package ahead of a namespace portion regardless of
   sys.path order, so ``from scripts.capture_gap_parity_golden import ...`` failed
   with a misleading ``No module named 'scripts.capture_gap_parity_golden'`` --
   the parent resolved, the submodule did not.

Both tests here discover their own subject matter rather than reading a
hand-maintained list, so a new call site is covered the moment it is written.
Everything is import- and introspection-only: no I/O, no network, sub-second.
"""
from __future__ import annotations


import pytest

# The sibling `curriculum` repo is private and no longer a declared dependency
# (see pyproject.toml). These tests cover code that still imports it, so they
# skip rather than error on a clone without `uv pip install -e ../curriculum`.
pytest.importorskip("curriculum", reason="sibling curriculum checkout not installed")

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Trees whose curriculum usage we hold to the contract.
SOURCE_TREES = ("bookgap", "artifacts_scripts", "scripts")


def _is_curriculum(module: str | None) -> bool:
    """True for the sibling repo, false for the local ``curriculum_lean`` package.

    The prefix collision is a live trap: a plain ``startswith("curriculum")``
    sweeps in ~200 ``curriculum_lean`` imports that have nothing to do with the
    sibling repo.
    """
    return module is not None and (module == "curriculum" or module.startswith("curriculum."))


def _python_files() -> list[Path]:
    out: list[Path] = []
    for tree in SOURCE_TREES:
        root = REPO_ROOT / tree
        if root.is_dir():
            out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _curriculum_imports() -> list[tuple[Path, str, str, str]]:
    """(file, local_name, module, attribute) for every sibling-curriculum import.

    ``attribute`` is "" for whole-module imports (``import curriculum.core.rank``).
    Covers imports nested inside functions too -- several are deliberately lazy.
    """
    found: list[tuple[Path, str, str, str]] = []
    for path in _python_files():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _is_curriculum(node.module):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    found.append((path, local, node.module or "", alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_curriculum(alias.name):
                        local = alias.asname or alias.name.split(".")[0]
                        found.append((path, local, alias.name, ""))
    return found


def _resolve(module: str, attribute: str):
    mod = importlib.import_module(module)
    return getattr(mod, attribute) if attribute else mod


@pytest.mark.parametrize(
    "module,attribute",
    sorted({(m, a) for _, _, m, a in _curriculum_imports()}),
    ids=lambda v: v or "<module>",
)
def test_curriculum_symbol_still_exists(module: str, attribute: str):
    """Every symbol bookgap imports from curriculum resolves."""
    try:
        _resolve(module, attribute)
    except ImportError as exc:
        pytest.fail(f"curriculum module '{module}' no longer importable: {exc}")
    except AttributeError:
        pytest.fail(
            f"curriculum.{module} no longer exports '{attribute}'. "
            "The sibling repo changed its API; update the bookgap call sites "
            "rather than pinning curriculum."
        )


def _dict_var_keys(scope: ast.AST) -> dict[str, set[str]]:
    """Map ``name -> literal string keys`` for dicts built inside one scope.

    Recognises the shape that actually appears in this codebase::

        kwargs: dict = dict(method=..., top_k=...)   # AnnAssign, not Assign
        kwargs = {"method": ...}
        kwargs["doc_weighting"] = ...                # conditional additions

    This exists because the call sites that matter splat rather than passing
    keywords inline -- ``extract_topic_keywords(tokens, **extract_kwargs)``. An
    earlier version of this test skipped splats and therefore could not see the
    one call site that the weight_by_freq removal actually broke.
    """
    keys: dict[str, set[str]] = {}

    def record(name: str, value: ast.AST | None) -> None:
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict":
            keys.setdefault(name, set()).update(
                kw.arg for kw in value.keywords if kw.arg is not None
            )
        elif isinstance(value, ast.Dict):
            keys.setdefault(name, set()).update(
                k.value for k in value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            )

    for node in ast.walk(scope):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            record(node.target.id, node.value)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    record(tgt.id, node.value)
                elif (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and isinstance(tgt.slice, ast.Constant)
                    and isinstance(tgt.slice.value, str)
                ):
                    keys.setdefault(tgt.value.id, set()).add(tgt.slice.value)
    return keys


def _curriculum_call_kwargs() -> list[tuple[Path, int, str, str, frozenset[str]]]:
    """(file, lineno, module, attribute, kwarg names) for calls to imported symbols.

    Covers both inline keywords and ``**name`` splats whose dict is built in the
    same scope. A splat we cannot resolve is skipped rather than guessed at --
    under-reporting is safe here, inventing keys would produce false failures.
    """
    out: list[tuple[Path, int, str, str, frozenset[str]]] = []
    for path in _python_files():
        tree = _parse(path)
        if tree is None:
            continue
        local_map: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _is_curriculum(node.module):
                for alias in node.names:
                    if alias.name != "*":
                        local_map[alias.asname or alias.name] = (node.module or "", alias.name)
        if not local_map:
            continue

        # Per-scope, so a `kwargs` in one function cannot leak into another.
        scopes: list[ast.AST] = [tree] + [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for scope in scopes:
            dict_keys = _dict_var_keys(scope)
            for node in ast.walk(scope):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                target = local_map.get(node.func.id)
                if target is None:
                    continue
                names: set[str] = set()
                for kw in node.keywords:
                    if kw.arg is not None:
                        names.add(kw.arg)
                    elif isinstance(kw.value, ast.Name):
                        names |= dict_keys.get(kw.value.id, set())
                if names:
                    entry = (path, node.lineno, target[0], target[1], frozenset(names))
                    if entry not in out:
                        out.append(entry)
    return out


@pytest.mark.parametrize(
    "path,lineno,module,attribute,kwargs",
    _curriculum_call_kwargs(),
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_curriculum_call_kwargs_still_bind(
    path: Path, lineno: int, module: str, attribute: str, kwargs: frozenset[str]
):
    """Keyword arguments we pass to curriculum functions still exist.

    This is the ``weight_by_freq`` guard. bind_partial rather than a real call:
    it validates names against the signature without executing anything.
    """
    try:
        fn = _resolve(module, attribute)
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"{module}.{attribute} unresolvable: {exc}")

    if not callable(fn):
        pytest.skip(f"{attribute} is not callable")
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        pytest.skip(f"{attribute} has no introspectable signature")

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return  # **kwargs accepts anything

    unknown = sorted(k for k in kwargs if k not in signature.parameters)
    assert not unknown, (
        f"{path.relative_to(REPO_ROOT)}:{lineno} passes {unknown} to "
        f"{module}.{attribute}, which no longer accepts it.\n"
        f"Current signature: {signature}\n"
        "curriculum changed its API. Migrate the call site -- do not add a shim."
    )


# Repo-root directories that are not importable top-level names we own.
_NOT_PACKAGES = {
    "cache", "outputs", "artifacts", "docs", "multirun", "paper",
    "node_modules", "site-packages", "build", "dist",
}


def _repo_root_packages() -> list[str]:
    """Top-level importable names this repo owns.

    Deliberately keyed on "contains Python", NOT on having an ``__init__.py``.
    An earlier version required ``__init__.py``, which made this check vanish in
    exactly the situation it exists to catch: deleting ``scripts/__init__.py``
    is what lets curriculum's regular ``scripts`` package win, and it also
    dropped ``scripts`` from this list, so the suite went green while shadowed.
    A guard that disappears under the condition it guards is worse than none.
    """
    out = []
    for p in sorted(REPO_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name in _NOT_PACKAGES:
            continue
        if any(p.rglob("*.py")):
            out.append(p.name)
    return out


@pytest.mark.parametrize("package", _repo_root_packages())
def test_repo_package_is_not_shadowed(package: str):
    """A top-level package here resolves to *here*, not to a dependency.

    curriculum shipping its own ``scripts`` package silently captured ours. Any
    top-level name we own is exposed to the same capture by any dependency that
    later claims it, so this checks the property rather than that one name.

    find_spec rather than import_module: no module side effects, and it reports
    the path that *would* be used.
    """
    spec = importlib.util.find_spec(package)
    assert spec is not None, f"top-level package '{package}' is not importable at all"

    origin = spec.origin
    if origin in (None, "namespace"):  # namespace package: check its search path
        locations = list(getattr(spec, "submodule_search_locations", []) or [])
        assert locations, f"'{package}' resolves to a namespace package with no location"
        origin = locations[0]

    resolved = Path(origin).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"'{package}' resolves to {resolved}, outside this repo.\n"
        "A dependency ships a top-level package of the same name and is shadowing "
        "ours. A regular package (one with __init__.py) wins over a namespace "
        "package regardless of sys.path order, so the fix is usually to give our "
        "directory an __init__.py -- or to rename one of the two."
    )
