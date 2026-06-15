"""Static ``ast``-based Python -> code-dependency-graph extractor.

This is the "noisy oracle" anchor of the gap-detection harness. It turns a Python
package into a graph of code *nodes* split into:

    B = concepts   = classes               (the things composed into other things)
    F = operations = functions/methods/tests (the things that exercise concepts)

An edge ``src -> dst`` means "src depends on / uses dst". Resolution is
*deliberately incomplete*: only cleanly-resolvable references (inheritance,
bare-name calls, ``self.m(...)``, and annotations resolving to a known class) are
kept. Duck-typed / dynamically-dispatched attribute calls on untyped objects are
unresolvable and dropped on purpose -- that dropped noise is the scientific point
(Python is the noisy anchor against the clean Lean oracle).

Stdlib ``ast`` only -- no third-party call-graph tooling.

Per-node counts consumed downstream:
    structural_users(X)  = # of B-nodes (classes)   that depend on X
    operational_users(X) = # of F-nodes (functions) that depend on X
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# joblib-cached singleton logger if available; fall back to module logging.
try:  # pragma: no cover - trivial import wiring
    from util import LOGGER
except Exception:  # pragma: no cover
    import logging

    LOGGER = logging.getLogger("gap_recover.python.extract")

_TEXT_MAX = 300


# --------------------------------------------------------------------------- #
# data structures
# --------------------------------------------------------------------------- #
@dataclass
class ExtractResult:
    names: list[str]            # qualified node name "module.Qual.name"
    kinds: list[str]            # "class" | "function"
    mods: list[str]             # module name "pkg.sub.mod" (bootstrap cluster key)
    texts: list[str]            # signature + first docstring line (NO bodies)
    is_F: np.ndarray            # bool: function/method/test
    is_B: np.ndarray            # bool: class
    is_test: np.ndarray         # bool: F-node defined in a test module
    concept_pref: np.ndarray    # bool: public class in a NON-test module
    edges_src: np.ndarray       # int idx
    edges_dst: np.ndarray       # int idx (src depends-on dst)

    def structural_users(self) -> np.ndarray:
        """Per-node count of distinct-edge B (class) citers."""
        out = np.zeros(len(self.names), dtype=np.int64)
        if self.edges_src.size:
            mask = self.is_B[self.edges_src]
            np.add.at(out, self.edges_dst[mask], 1)
        return out

    def operational_users(self) -> np.ndarray:
        """Per-node count of distinct-edge F (function) citers."""
        out = np.zeros(len(self.names), dtype=np.int64)
        if self.edges_src.size:
            mask = self.is_F[self.edges_src]
            np.add.at(out, self.edges_dst[mask], 1)
        return out


@dataclass
class _ModuleDefs:
    """Per-module bookkeeping accumulated during pass 1."""

    module: str
    local: dict[str, int] = field(default_factory=dict)        # top-level name -> node idx
    imports: dict[str, str] = field(default_factory=dict)      # alias -> "module.qual"
    # node idx -> (qualname, enclosing_class_qual_or_None, ast_node)
    bodies: dict[int, tuple[str, str | None, ast.AST]] = field(default_factory=dict)


@dataclass
class _Defs:
    """Aggregate output of pass 1, consumed by pass 2."""

    names: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    mods: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    is_F: list[bool] = field(default_factory=list)
    is_B: list[bool] = field(default_factory=list)
    is_test: list[bool] = field(default_factory=list)
    concept_pref: list[bool] = field(default_factory=list)
    name_to_idx: dict[str, int] = field(default_factory=dict)
    modules: dict[str, _ModuleDefs] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def module_name_for(py_path: Path, pkg_dir: Path, import_name: str) -> str:
    """Dotted module name for ``py_path`` inside package rooted at ``pkg_dir``.

    ``<import_root>/a/b/c.py``        -> ``a.b.c``
    ``<import_root>/a/b/__init__.py`` -> ``a.b``
    The package dir itself (``pkg_dir``) maps onto ``import_name``.
    """
    rel = py_path.relative_to(pkg_dir.parent)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip ".py"
    return ".".join(parts)


def _is_test_module(module: str) -> bool:
    """A node is a "test operation" if its module path is test-flavored."""
    leaf = module.rsplit(".", 1)[-1]
    return (
        leaf.startswith("test_")
        or leaf.endswith("_test")
        or leaf == "tests"
        or leaf == "conftest"
        or ".tests." in f".{module}."
        or ".test." in f".{module}."
    )


def _dotted_attr(node: ast.AST) -> str | None:
    """Render ``a.b.c`` (Name/Attribute chain) to its dotted string, else None."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _docstring_first_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    try:
        doc = ast.get_docstring(node, clean=True)
    except Exception:
        doc = None
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip() if doc.strip() else ""


def _func_text(qualname: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    a = node.args
    arg_nodes = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    arg_names = [x.arg for x in arg_nodes]
    if a.vararg:
        arg_names.append("*" + a.vararg.arg)
    if a.kwarg:
        arg_names.append("**" + a.kwarg.arg)
    sig = ", ".join(arg_names)
    text = f"def {qualname}({sig}) : {_docstring_first_line(node)}".rstrip()
    return text[:_TEXT_MAX]


def _class_text(qualname: str, node: ast.ClassDef) -> str:
    bases = [b for b in (_dotted_attr(x) for x in node.bases) if b]
    text = f"class {qualname}({', '.join(bases)}) : {_docstring_first_line(node)}".rstrip()
    return text[:_TEXT_MAX]


# --------------------------------------------------------------------------- #
# pass 1: collect defs + per-module symbol tables
# --------------------------------------------------------------------------- #
def build_import_map(
    tree: ast.Module, module: str, *, is_package: bool = False,
) -> dict[str, str]:
    """alias -> fully-qualified ``"module.qual"`` target string.

    Handles ``import x`` / ``import x as y`` / ``from m import n`` /
    ``from m import n as a`` and intra-package relative ``from .rel import n`` /
    ``from ..pkg import n`` (leading dots resolved against ``module``'s package).
    Targets pointing outside the scanned package are recorded anyway; they simply
    fail to map to a node in pass 2.

    ``is_package`` is True when ``module`` comes from a ``__init__.py``: a
    package's ``__init__`` IS the package, so a single leading dot
    (``from .core import X``) anchors at ``module`` itself, not its parent.
    """
    imports: dict[str, str] = {}
    # Containing-package path that a single leading dot anchors at. For a
    # package __init__ that is the module itself; for a regular module it is the
    # module minus its own last component.
    pkg_parts = module.split(".") if is_package else module.split(".")[:-1]
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                target = alias.name                       # "a.b.c"
                bound = alias.asname or alias.name.split(".")[0]
                # `import a.b.c` binds name `a`; `import a.b.c as x` binds `x` to a.b.c.
                imports[bound] = target if alias.asname else alias.name.split(".")[0]
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level:  # relative import: resolve leading dots
                base = pkg_parts[: len(pkg_parts) - (stmt.level - 1)]
                mod_prefix = base + (stmt.module.split(".") if stmt.module else [])
            else:
                mod_prefix = stmt.module.split(".") if stmt.module else []
            mod_str = ".".join(mod_prefix)
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                target = f"{mod_str}.{alias.name}" if mod_str else alias.name
                imports[bound] = target
    return imports


def _collect_module(
    tree: ast.Module, module: str, defs: _Defs, *, is_package: bool = False,
) -> None:
    """Register every class/function node in ``module`` into ``defs``."""
    is_test_mod = _is_test_module(module)
    md = _ModuleDefs(
        module=module,
        imports=build_import_map(tree, module, is_package=is_package),
    )
    defs.modules[module] = md

    def visit(node: ast.AST, qual_prefix: str, enclosing_class: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qual = f"{qual_prefix}{child.name}"
                full = f"{module}.{qual}"
                idx = _add_node(
                    defs, full, "class", module, _class_text(qual, child),
                    is_F=False, is_B=True, is_test=False,
                    concept_pref=(not is_test_mod and not child.name.startswith("_")),
                )
                md.bodies[idx] = (qual, enclosing_class, child)
                if qual_prefix == "":  # module-top-level class
                    md.local[child.name] = idx
                visit(child, f"{qual}.", qual)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{qual_prefix}{child.name}"
                full = f"{module}.{qual}"
                idx = _add_node(
                    defs, full, "function", module, _func_text(qual, child),
                    is_F=True, is_B=False, is_test=is_test_mod,
                    concept_pref=False,
                )
                md.bodies[idx] = (qual, enclosing_class, child)
                if qual_prefix == "":  # module-top-level function
                    md.local[child.name] = idx
                # recurse for nested defs / methods (enclosing class stays this
                # class for a method; nested funcs keep their own class context)
                visit(child, f"{qual}.", enclosing_class)
            # non-def/class nodes are not recursed for *definitions* here; their
            # references are walked in pass 2 against the registered def nodes.

    visit(tree, "", None)


def _add_node(
    defs: _Defs, name: str, kind: str, module: str, text: str,
    *, is_F: bool, is_B: bool, is_test: bool, concept_pref: bool,
) -> int:
    if not text:  # gate: require non-empty text (almost always satisfied)
        text = name
    if name in defs.name_to_idx:
        return defs.name_to_idx[name]  # duplicate qualname in module: keep first
    idx = len(defs.names)
    defs.names.append(name)
    defs.kinds.append(kind)
    defs.mods.append(module)
    defs.texts.append(text)
    defs.is_F.append(is_F)
    defs.is_B.append(is_B)
    defs.is_test.append(is_test)
    defs.concept_pref.append(concept_pref)
    defs.name_to_idx[name] = idx
    return idx


def _parse_file(py_path: Path) -> ast.Module | None:
    """Parse one ``.py`` file, returning its AST or None if unparseable."""
    try:
        source = py_path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(py_path))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None


def _module_name_under(py_path: Path, root: Path) -> str:
    """Dotted module name for ``py_path`` relative to ``root`` (root excluded).

    ``<root>/tests/test_core.py``        -> ``tests.test_core``
    ``<root>/tests/__init__.py``         -> ``tests``
    Used for root-level sibling test dirs, whose modules are NOT under the
    located package and so cannot use :func:`module_name_for`.
    """
    rel = py_path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip ".py"
    return ".".join(parts)


def collect_defs(pkg_dir: Path, import_name: str) -> _Defs:
    """Pass 1: walk every ``.py`` under the package, build nodes + symbol tables."""
    defs = _Defs()
    skipped = 0
    for py_path in sorted(pkg_dir.rglob("*.py")):
        module = module_name_for(py_path, pkg_dir, import_name)
        tree = _parse_file(py_path)
        if tree is None:
            skipped += 1
            continue
        _collect_module(tree, module, defs, is_package=py_path.name == "__init__.py")
    if skipped:
        LOGGER.info("python-extract: skipped %d unparseable file(s)", skipped)
    return defs


def collect_root_tests(repo_path: Path, pkg_dir: Path, defs: _Defs) -> None:
    """Pass 1b: scan top-level sibling ``tests/`` / ``test/`` dirs.

    Repos like click/flask/requests keep their tests in a top-level directory
    that is a sibling of the package, so :func:`collect_defs` never reaches
    them. Module names here are taken relative to ``repo_path`` (e.g.
    ``tests/test_core.py`` -> ``tests.test_core``). Every def is registered as
    a test-module node (``is_test=True`` for functions, ``concept_pref=False``
    for everything). Dirs already living inside the package (``pkg/tests/``,
    already scanned by :func:`collect_defs`) are skipped to avoid double-count.
    """
    skipped = 0
    try:
        pkg_resolved = pkg_dir.resolve()
    except OSError:
        pkg_resolved = pkg_dir
    for dname in ("tests", "test"):
        tdir = repo_path / dname
        if not tdir.is_dir():
            continue
        # skip a tests dir that is the package itself or lives inside it.
        try:
            tdir_resolved = tdir.resolve()
        except OSError:
            tdir_resolved = tdir
        if tdir_resolved == pkg_resolved or pkg_resolved in tdir_resolved.parents:
            continue
        for py_path in sorted(tdir.rglob("*.py")):
            module = _module_name_under(py_path, repo_path)
            if module in defs.modules:  # already scanned -> never double-count
                continue
            tree = _parse_file(py_path)
            if tree is None:
                skipped += 1
                continue
            _collect_module(
                tree, module, defs, is_package=py_path.name == "__init__.py",
            )
    if skipped:
        LOGGER.info("python-extract: skipped %d unparseable root-test file(s)", skipped)


# --------------------------------------------------------------------------- #
# pass 2: resolve edges
# --------------------------------------------------------------------------- #
def _resolve_target(target: str, defs: _Defs, _hops: int = 2) -> int | None:
    """Map a fully-qualified target string ``"M.q"`` to a node idx.

    Directly if it already names a node; otherwise *follow package re-exports*:
    if ``M`` is a scanned module that re-exports ``q`` (i.e. ``q`` lives in
    ``M``'s import map, as produced by ``pkg/__init__.py`` doing
    ``from .core import q``), chase that mapping to the real node. Bounded to
    ``_hops`` chases (default 2) to avoid import cycles; unresolved -> None.
    """
    seen: set[str] = set()
    for _ in range(_hops + 1):
        if target in defs.name_to_idx:
            return defs.name_to_idx[target]
        if target in seen:
            break
        seen.add(target)
        # split into containing module + symbol; chase a re-export if present.
        if "." not in target:
            break
        mod_str, _, sym = target.rpartition(".")
        re_mod = defs.modules.get(mod_str)
        if re_mod is None or sym not in re_mod.imports:
            break
        target = re_mod.imports[sym]
    return None


def resolve_name(
    name: str, md: _ModuleDefs, defs: _Defs, enclosing_class: str | None,
) -> int | None:
    """Resolve a referenced symbol to a node idx, or None if unresolvable.

    Lookup order:
      1. same-module top-level ``local``;
      2. module ``imports`` -> fully-qualified target -> global name index
         (also try a one-hop ``from-module import alias`` where the alias is a
         submodule and ``name`` was ``alias.attr``); a target that names a
         package re-export (``from .core import X`` in ``pkg/__init__.py``) is
         chased through ``_resolve_target`` to the real node;
    ``name`` may be dotted (``mod.Thing`` / ``pkg.mod.Thing``).
    """
    # bare same-module name
    head = name.split(".", 1)[0]
    if name in md.local:
        return md.local[name]
    if head in md.local and "." not in name:
        return md.local[head]

    # dotted/bare via import alias
    if head in md.imports:
        target_head = md.imports[head]
        rest = name.split(".", 1)[1] if "." in name else ""
        target = f"{target_head}.{rest}" if rest else target_head
        idx = _resolve_target(target, defs)
        if idx is not None:
            return idx
    # fully-qualified reference that already names a node directly
    if name in defs.name_to_idx:
        return defs.name_to_idx[name]
    return None


def _resolve_class_node(
    name: str, md: _ModuleDefs, defs: _Defs,
) -> int | None:
    idx = resolve_name(name, md, defs, None)
    if idx is not None and defs.kinds[idx] == "class":
        return idx
    return None


def _annotation_classes(ann: ast.AST, md: _ModuleDefs, defs: _Defs) -> list[int]:
    """Class nodes referenced anywhere inside an annotation expression."""
    out: list[int] = []
    for sub in ast.walk(ann):
        if isinstance(sub, (ast.Name, ast.Attribute)):
            dotted = _dotted_attr(sub)
            if dotted:
                idx = _resolve_class_node(dotted, md, defs)
                if idx is not None:
                    out.append(idx)
    return out


def _resolve_body_edges(
    caller: int, qual: str, enclosing_class: str | None,
    node: ast.AST, md: _ModuleDefs, defs: _Defs, edges: set[tuple[int, int]],
) -> None:
    """Walk a function/method body for resolvable references -> edges."""
    for sub in ast.walk(node):
        # --- calls --------------------------------------------------------- #
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name):
                dst = resolve_name(fn.id, md, defs, enclosing_class)
                if dst is not None:
                    edges.add((caller, dst))
            elif isinstance(fn, ast.Attribute):
                base = fn.value
                if isinstance(base, ast.Name) and base.id == "self" and enclosing_class:
                    # self.m(...) -> enclosing_class.m if that method node exists
                    target = f"{md.module}.{enclosing_class}.{fn.attr}"
                    dst = defs.name_to_idx.get(target)
                    if dst is not None:
                        edges.add((caller, dst))
                else:
                    # obj.m(...) with obj != self: try resolving obj.m as an
                    # imported / module-level dotted symbol (e.g. `mod.func(...)`,
                    # `pkg.Klass(...)`). Pure duck-typed attr calls won't resolve
                    # and are intentionally dropped.
                    dotted = _dotted_attr(fn)
                    if dotted:
                        dst = resolve_name(dotted, md, defs, enclosing_class)
                        if dst is not None:
                            edges.add((caller, dst))
        # --- annotations on params / returns / variables ------------------- #
        elif isinstance(sub, ast.arg) and sub.annotation is not None:
            for dst in _annotation_classes(sub.annotation, md, defs):
                edges.add((caller, dst))
        elif isinstance(sub, ast.AnnAssign) and sub.annotation is not None:
            for dst in _annotation_classes(sub.annotation, md, defs):
                edges.add((caller, dst))
    # return annotation lives on the def node itself
    ret = getattr(node, "returns", None)
    if ret is not None:
        for dst in _annotation_classes(ret, md, defs):
            edges.add((caller, dst))


def _resolve_class_edges(
    citer: int, node: ast.ClassDef, md: _ModuleDefs, defs: _Defs,
    edges: set[tuple[int, int]],
) -> None:
    """Inheritance bases + class-scope attribute annotations (structural)."""
    for base in node.bases:
        dotted = _dotted_attr(base)
        if dotted:
            dst = _resolve_class_node(dotted, md, defs)
            if dst is not None:
                edges.add((citer, dst))
    # class-body attribute annotations: `x: Cls` / `x: Cls = ...`
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.annotation is not None:
            for dst in _annotation_classes(stmt.annotation, md, defs):
                edges.add((citer, dst))


def resolve_edges(defs: _Defs) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: produce deduped, self-loop-free ``(edges_src, edges_dst)``."""
    edges: set[tuple[int, int]] = set()
    for md in defs.modules.values():
        for idx, (qual, enclosing_class, node) in md.bodies.items():
            if isinstance(node, ast.ClassDef):
                _resolve_class_edges(idx, node, md, defs, edges)
            else:  # FunctionDef / AsyncFunctionDef
                _resolve_body_edges(idx, qual, enclosing_class, node, md, defs, edges)
    pairs = [(s, d) for (s, d) in edges if s != d]  # drop self-loops
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    src = np.fromiter((s for s, _ in pairs), dtype=np.int64, count=len(pairs))
    dst = np.fromiter((d for _, d in pairs), dtype=np.int64, count=len(pairs))
    return src, dst


# --------------------------------------------------------------------------- #
# package location + top-level entry point
# --------------------------------------------------------------------------- #
def _locate_pkg_dir(repo_path: Path, import_name: str) -> Path | None:
    candidates = [
        repo_path / import_name,
        repo_path / "src" / import_name,
    ]
    for c in candidates:
        if (c / "__init__.py").exists():
            return c
    # fall back: first top-level dir containing __init__.py under repo / repo/src
    for root in (repo_path, repo_path / "src"):
        if not root.is_dir():
            continue
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            if (child / "__init__.py").exists():
                LOGGER.info(
                    "python-extract: '%s' package dir not found for import_name=%r; "
                    "falling back to first package dir %s",
                    import_name, import_name, child,
                )
                return child
    return None


def extract_repo(repo_path, import_name: str) -> ExtractResult:
    """Locate the package, run pass 1 + pass 2, assemble the ``ExtractResult``."""
    repo_path = Path(repo_path)
    pkg_dir = _locate_pkg_dir(repo_path, import_name)
    if pkg_dir is None:
        raise FileNotFoundError(
            f"no Python package (dir with __init__.py) found under {repo_path} "
            f"for import_name={import_name!r}"
        )
    LOGGER.info("python-extract: scanning package %s (import_name=%r)", pkg_dir, import_name)

    defs = collect_defs(pkg_dir, import_name)
    collect_root_tests(repo_path, pkg_dir, defs)
    edges_src, edges_dst = resolve_edges(defs)

    return ExtractResult(
        names=defs.names,
        kinds=defs.kinds,
        mods=defs.mods,
        texts=defs.texts,
        is_F=np.asarray(defs.is_F, dtype=bool),
        is_B=np.asarray(defs.is_B, dtype=bool),
        is_test=np.asarray(defs.is_test, dtype=bool),
        concept_pref=np.asarray(defs.concept_pref, dtype=bool),
        edges_src=edges_src,
        edges_dst=edges_dst,
    )
