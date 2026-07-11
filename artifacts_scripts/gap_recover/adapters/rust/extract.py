"""tree-sitter Rust -> code-dependency-graph extractor (4th-language modality check).

Source-only Rust analogue of the Python ``ast`` extractor (NO cargo / no Rust
toolchain — pure ``tree-sitter-rust`` parsing). Produces the SAME
``ExtractResult`` shape so the shared gap-recover core runs unchanged.

The generic graph model maps onto Rust as:

    B = concepts   = TYPES   (struct / enum / union / trait / type-alias)
    F = operations = functions / methods / tests

An edge ``src -> dst`` means "src uses dst". Resolution is *deliberately
incomplete* and resolves by NAME against the set of type/function names defined
in the SAME repo (a heuristic — noisy/low-recall is expected and is the point:
Rust is a 4th, independently-noisy modality). Unresolved references are dropped.

    structural (type->type): struct/enum/union field types, ``impl Trait for T``
        (T -> Trait), type-alias target, trait supertraits & generic bounds.
    operational (func->type): inside a fn/method body — construction
        (``T { .. }`` struct-expr, ``T::new(..)`` / ``T::assoc(..)`` scoped
        calls) and param/return/local type annotations referencing an in-repo
        type. A method call ``x.method()`` is unresolvable without types and is
        SKIPPED on purpose; ``T::method(..)`` scoped calls DO resolve to T.

Per-node counts consumed downstream:
    structural_users(X)  = # of B-nodes (types) that use X
    operational_users(X) = # of F-nodes (funcs) that use X
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tree_sitter_rust
from tree_sitter import Language, Node, Parser

try:  # pragma: no cover - trivial import wiring
    from util import LOGGER
except Exception:  # pragma: no cover
    import logging

    LOGGER = logging.getLogger("gap_recover.rust.extract")

_TEXT_MAX = 300

# Rust node-type vocabulary (tree-sitter-rust grammar).
_TYPE_ITEMS = frozenset({"struct_item", "enum_item", "union_item", "trait_item", "type_item"})
_TEST_ATTRS = frozenset({"test", "bench"})

_LANGUAGE = Language(tree_sitter_rust.language())


def _parser() -> Parser:
    return Parser(_LANGUAGE)


# --------------------------------------------------------------------------- #
# data structures (mirror the Python adapter exactly)
# --------------------------------------------------------------------------- #
@dataclass
class ExtractResult:
    names: list[str]            # qualified node name "repo::module::Item"
    kinds: list[str]            # "type" | "func" | "method" | "test"
    mods: list[str]             # module name "repo::a::b" (bootstrap cluster key)
    texts: list[str]            # signature only, first line (NO bodies)
    is_F: np.ndarray            # bool: function / method / test
    is_B: np.ndarray            # bool: type (struct/enum/union/trait/alias)
    is_test: np.ndarray         # bool: F-node that is a test
    concept_pref: np.ndarray    # bool: pub type in a NON-test module
    edges_src: np.ndarray       # int idx
    edges_dst: np.ndarray       # int idx (src uses dst)

    def structural_users(self) -> np.ndarray:
        """Per-node count of distinct-edge B (type) citers."""
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
class _Item:
    """One collected def node, pending edge resolution in pass 2."""

    idx: int
    kind: str                   # "type" | "func" | "method" | "test"
    leaf: str                   # bare item name (e.g. "Shape", "make_shape")
    node: Node                  # tree-sitter node (header for types, fn for funcs)
    self_type: str | None       # for a method: leaf name of its impl Self type


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
    # leaf type-name -> node idx (last writer wins; resolution is heuristic).
    type_by_leaf: dict[str, int] = field(default_factory=dict)
    # leaf func/method-name -> node idx.
    func_by_leaf: dict[str, int] = field(default_factory=dict)
    items: list[_Item] = field(default_factory=list)
    # ``impl Trait for T`` headers as (self_type_leaf, trait_leaf) -> structural
    # ``T -> Trait`` edges in pass 2. impl blocks are not graph nodes themselves.
    impls: list[tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _txt(node: Node | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", "replace")


def _oneline(s: str) -> str:
    """First non-empty line, tabs/newlines -> spaces, clamped to _TEXT_MAX."""
    s = s.replace("\t", " ").replace("\r", " ")
    line = next((ln for ln in s.split("\n") if ln.strip()), "")
    return " ".join(line.split())[:_TEXT_MAX]


def module_name_for(rs_path: Path, repo_path: Path, src_root: Path) -> str:
    """Module path for ``rs_path``, anchored at the crate ``src`` root.

    File path components are ``::``-joined; ``.rs`` is stripped, and the Rust
    module-root file names (``lib.rs`` / ``main.rs`` / ``mod.rs``) collapse onto
    their directory (``foo/mod.rs`` -> ``foo``; ``src/lib.rs`` -> ``""``). Files
    outside ``src`` (a top-level ``tests/`` / ``benches/`` dir) are anchored at
    the repo root instead (``tests/test_x.rs`` -> ``tests::test_x``). Kept
    deliberately simple — it is only a clustering key.
    """
    try:
        rel = rs_path.relative_to(src_root)
    except ValueError:
        try:
            rel = rs_path.relative_to(repo_path)
        except ValueError:
            rel = Path(rs_path.name)
    parts = list(rel.parts)
    if parts and parts[-1].endswith(".rs"):
        stem = parts[-1][:-3]
        if stem in ("lib", "main", "mod"):
            parts = parts[:-1]
        else:
            parts[-1] = stem
    return "::".join(parts)


def _attr_names(attr_nodes: list[Node]) -> set[str]:
    """Identifier leaves of a run of ``attribute_item`` nodes (e.g. {"test","cfg"})."""
    out: set[str] = set()
    for a in attr_nodes:
        for sub in _walk(a):
            if sub.type == "identifier":
                out.add(_txt(sub))
    return out


def _is_pub(node: Node) -> bool:
    return any(c.type == "visibility_modifier" for c in node.children)


def _type_leaves(node: Node | None) -> list[str]:
    """All ``type_identifier`` leaf names anywhere inside a type expression.

    Covers bare types, generics (``Vec<Shape>`` -> Shape), tuples, references,
    ``Box<Inner>`` etc. Skips lone ``Self``.
    """
    if node is None:
        return []
    out: list[str] = []
    for sub in _walk(node):
        if sub.type == "type_identifier":
            nm = _txt(sub)
            if nm and nm != "Self":
                out.append(nm)
    return out


def _walk(node: Node):
    """Pre-order traversal over named + unnamed descendants (incl. ``node``)."""
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


# --------------------------------------------------------------------------- #
# text (signature only — NEVER a body)
# --------------------------------------------------------------------------- #
def _fn_text(node: Node) -> str:
    """``fn name(params) -> ret`` first line, body stripped."""
    body = node.child_by_field_name("body")
    if body is not None:
        sig = _txt(node)[: body.start_byte - node.start_byte]
    else:
        sig = _txt(node)
    return _oneline(sig)


def _type_text(node: Node) -> str:
    """Type header first line: ``struct Name`` / ``trait Name { .. }`` / alias.

    Drops the body so no field-level detail/impl leaks; the leading ``pub`` and
    generics are preserved.
    """
    body = node.child_by_field_name("body")
    if body is not None:
        head = _txt(node)[: body.start_byte - node.start_byte]
    else:
        head = _txt(node)
    return _oneline(head)


# --------------------------------------------------------------------------- #
# pass 1: collect items + per-repo symbol tables
# --------------------------------------------------------------------------- #
def _add_node(
    defs: _Defs, name: str, kind: str, module: str, text: str,
    *, is_F: bool, is_B: bool, is_test: bool, concept_pref: bool,
) -> int:
    if not text:  # gate: require non-empty text
        text = name
    idx = len(defs.names)
    defs.names.append(name)
    defs.kinds.append(kind)
    defs.mods.append(module)
    defs.texts.append(text)
    defs.is_F.append(is_F)
    defs.is_B.append(is_B)
    defs.is_test.append(is_test)
    defs.concept_pref.append(concept_pref)
    return idx


def _leading_attrs(child: Node, siblings: list[Node], pos: int) -> list[Node]:
    """Contiguous ``attribute_item`` siblings immediately preceding ``siblings[pos]``."""
    attrs: list[Node] = []
    j = pos - 1
    while j >= 0 and siblings[j].type == "attribute_item":
        attrs.append(siblings[j])
        j -= 1
    return attrs


def collect_items(
    body_nodes: list[Node], repo: str, module: str, defs: _Defs,
    *, in_test_ctx: bool,
) -> None:
    """Register type/function/method defs among ``body_nodes`` (one scope level).

    Recurses into ``mod_item`` blocks (extending the module path) and ``impl_item``
    blocks (binding the Self type onto contained methods). ``in_test_ctx`` is True
    inside a ``#[cfg(test)]`` mod or a ``tests/`` file — all functions become tests.
    """
    is_test_mod = in_test_ctx
    for pos, child in enumerate(body_nodes):
        ctype = child.type
        if ctype in _TYPE_ITEMS:
            leaf = _txt(child.child_by_field_name("name"))
            if not leaf:
                continue
            name = f"{repo}::{module}::{leaf}"
            idx = _add_node(
                defs, name, "type", f"{repo}::{module}", _type_text(child),
                is_F=False, is_B=True, is_test=False,
                concept_pref=(_is_pub(child) and not is_test_mod),
            )
            defs.type_by_leaf[leaf] = idx
            defs.items.append(_Item(idx, "type", leaf, child, None))

        elif ctype == "function_item":
            leaf = _txt(child.child_by_field_name("name"))
            if not leaf:
                continue
            attrs = _attr_names(_leading_attrs(child, body_nodes, pos))
            is_test = is_test_mod or bool(attrs & _TEST_ATTRS)
            kind = "test" if is_test else "func"
            name = f"{repo}::{module}::{leaf}"
            idx = _add_node(
                defs, name, kind, f"{repo}::{module}", _fn_text(child),
                is_F=True, is_B=False, is_test=is_test, concept_pref=False,
            )
            # last-writer-wins; free functions and methods share one namespace.
            defs.func_by_leaf.setdefault(leaf, idx)
            defs.items.append(_Item(idx, kind, leaf, child, None))

        elif ctype == "impl_item":
            self_type = _txt(impl_self(child))
            trait_leaf = _txt(impl_trait(child))
            if self_type and trait_leaf:
                defs.impls.append((self_type, trait_leaf))  # T -> Trait (structural)
            body = child.child_by_field_name("body")
            if body is None:
                continue
            inner = [c for c in body.children if c.is_named]
            for ipos, m in enumerate(inner):
                if m.type != "function_item":
                    continue
                leaf = _txt(m.child_by_field_name("name"))
                if not leaf:
                    continue
                attrs = _attr_names(_leading_attrs(m, inner, ipos))
                is_test = is_test_mod or bool(attrs & _TEST_ATTRS)
                kind = "test" if is_test else "method"
                name = f"{repo}::{module}::{self_type}::{leaf}" if self_type else \
                    f"{repo}::{module}::{leaf}"
                idx = _add_node(
                    defs, name, kind, f"{repo}::{module}", _fn_text(m),
                    is_F=True, is_B=False, is_test=is_test, concept_pref=False,
                )
                defs.func_by_leaf.setdefault(leaf, idx)
                defs.items.append(_Item(idx, kind, leaf, m, self_type or None))

        elif ctype == "mod_item":
            sub = _txt(child.child_by_field_name("name"))
            body = child.child_by_field_name("body")
            if body is None or not sub:
                continue
            attrs = _attr_names(_leading_attrs(child, body_nodes, pos))
            child_test_ctx = is_test_mod or ("test" in attrs)
            sub_module = f"{module}::{sub}" if module else sub
            inner = [c for c in body.children if c.is_named]
            collect_items(inner, repo, sub_module, defs, in_test_ctx=child_test_ctx)


def impl_self(impl: Node) -> Node | None:
    """The Self ``type`` of an ``impl`` block (``impl .. for T`` / ``impl T``)."""
    t = impl.child_by_field_name("type")
    if t is None:
        return None
    # ``impl Trait for Wrapper<T>`` -> the Self type may be a generic_type; take
    # its first type_identifier leaf as the bare name.
    if t.type == "type_identifier":
        return t
    for sub in _walk(t):
        if sub.type == "type_identifier":
            return sub
    return None


def impl_trait(impl: Node) -> Node | None:
    """The implemented ``trait`` of an ``impl`` block (None for inherent impls).

    ``impl Trait<T> for Wrapper<T>`` -> the trait's bare ``type_identifier`` leaf.
    """
    tr = impl.child_by_field_name("trait")
    if tr is None:
        return None
    if tr.type == "type_identifier":
        return tr
    for sub in _walk(tr):
        if sub.type == "type_identifier":
            return sub
    return None


# --------------------------------------------------------------------------- #
# pass 2: resolve edges (by leaf name, in-repo only)
# --------------------------------------------------------------------------- #
def _struct_field_types(node: Node) -> list[str]:
    """In-repo type leaves named by struct/enum/union field declarations.

    Named fields (``field_declaration``) carry a ``type`` field; tuple
    fields/variants (``ordered_field_declaration_list``) have no ``type`` field,
    so their type leaves are walked directly.
    """
    out: list[str] = []
    for sub in _walk(node):
        if sub.type == "field_declaration":
            out.extend(_type_leaves(sub.child_by_field_name("type")))
        elif sub.type == "ordered_field_declaration_list":
            out.extend(_type_leaves(sub))
    return out


def _bound_types(node: Node) -> list[str]:
    """Supertrait / generic-bound type leaves in a header (``: Bound``, where-clauses)."""
    out: list[str] = []
    for sub in _walk(node):
        if sub.type in ("trait_bounds", "where_clause", "type_parameters"):
            out.extend(_type_leaves(sub))
    return out


def _resolve_type_edges(item: _Item, defs: _Defs, edges: set[tuple[int, int]]) -> None:
    """Structural edges for a type def node."""
    node = item.node
    src = item.idx
    refs: list[str] = []
    if node.type in ("struct_item", "enum_item", "union_item"):
        refs += _struct_field_types(node)
        refs += _bound_types(node)
    elif node.type == "trait_item":
        refs += _bound_types(node)  # supertraits + assoc bounds
    elif node.type == "type_item":
        refs += _type_leaves(node.child_by_field_name("type"))  # alias target
    for leaf in refs:
        dst = defs.type_by_leaf.get(leaf)
        if dst is not None and dst != src:
            edges.add((src, dst))


def _scoped_path_leaf(scoped: Node) -> str | None:
    """Leaf type name for the ``path`` of a ``scoped_identifier`` (``T::f`` -> ``T``).

    Handles a nested-path tail (``a::b::T::f`` -> ``T``) by taking the last
    identifier of the immediate path. ``self::``/``crate::`` heads are ignored.
    """
    path = scoped.child_by_field_name("path")
    if path is None:
        return None
    if path.type == "identifier":
        nm = _txt(path)
        return nm if nm not in ("self", "crate", "super", "Self") else None
    if path.type == "scoped_identifier":  # take the last segment
        tail = path.child_by_field_name("name")
        if tail is not None and tail.type == "identifier":
            nm = _txt(tail)
            return nm if nm not in ("self", "crate", "super", "Self") else None
    return None


def _resolve_func_edges(item: _Item, defs: _Defs, edges: set[tuple[int, int]]) -> None:
    """Operational edges for a fn/method body + its signature annotations."""
    node = item.node
    src = item.idx
    types = defs.type_by_leaf

    def link(leaf: str | None) -> None:
        if not leaf:
            return
        dst = types.get(leaf)
        if dst is not None and dst != src:
            edges.add((src, dst))

    # signature: parameter / return type annotations.
    params = node.child_by_field_name("parameters")
    if params is not None:
        for leaf in _type_leaves(params):
            link(leaf)
    link_all = node.child_by_field_name("return_type")
    for leaf in _type_leaves(link_all):
        link(leaf)
    # a method's own Self type is a used concept.
    if item.self_type:
        link(item.self_type)

    body = node.child_by_field_name("body")
    if body is None:
        return
    for sub in _walk(body):
        st = sub.type
        if st == "struct_expression":          # T { .. } construction
            link_node = sub.child_by_field_name("name")
            if link_node is not None:
                for leaf in _type_leaves(link_node) or [_txt(link_node)]:
                    link(leaf)
        elif st == "call_expression":          # T::assoc(..) scoped call
            fn = sub.child_by_field_name("function")
            if fn is not None and fn.type == "scoped_identifier":
                link(_scoped_path_leaf(fn))
            # field_expression call (x.method()) -> intentionally NOT resolved.
        elif st == "let_declaration":          # local type annotation
            for leaf in _type_leaves(sub.child_by_field_name("type")):
                link(leaf)


def resolve_edges(defs: _Defs) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: produce deduped, self-loop-free ``(edges_src, edges_dst)``."""
    edges: set[tuple[int, int]] = set()
    for item in defs.items:
        if item.kind == "type":
            _resolve_type_edges(item, defs, edges)
        else:  # func / method / test
            _resolve_func_edges(item, defs, edges)
    # ``impl Trait for T`` -> structural edge T -> Trait (the type uses the trait).
    for self_leaf, trait_leaf in defs.impls:
        s = defs.type_by_leaf.get(self_leaf)
        d = defs.type_by_leaf.get(trait_leaf)
        if s is not None and d is not None and s != d:
            edges.add((s, d))
    pairs = [(s, d) for (s, d) in edges if s != d]
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    src = np.fromiter((s for s, _ in pairs), dtype=np.int64, count=len(pairs))
    dst = np.fromiter((d for _, d in pairs), dtype=np.int64, count=len(pairs))
    return src, dst


# --------------------------------------------------------------------------- #
# file scan + top-level entry point
# --------------------------------------------------------------------------- #
def _iter_rs_files(repo_path: Path):
    """Every ``*.rs`` under the repo, skipping ``target/`` build output."""
    for rs in sorted(repo_path.rglob("*.rs")):
        if "target" in rs.parts:
            continue
        yield rs


def _src_root(repo_path: Path) -> Path:
    """Crate source root for module-path purposes (``<repo>/src`` if present)."""
    src = repo_path / "src"
    return src if src.is_dir() else repo_path


def extract_repo(repo_path, repo_name: str) -> ExtractResult:
    """Parse every ``.rs`` under the repo and assemble the ``ExtractResult``."""
    repo_path = Path(repo_path)
    src_root = _src_root(repo_path)
    parser = _parser()
    defs = _Defs()
    skipped = 0

    for rs in _iter_rs_files(repo_path):
        try:
            data = rs.read_bytes()
        except OSError:
            skipped += 1
            continue
        # A file is a test context if it lives in a top-level (non-src) tests/ or
        # benches/ dir, or any path component is "tests"/"test".
        rel_parts = rs.relative_to(repo_path).parts
        in_test_file = (
            rel_parts and rel_parts[0] in ("tests", "benches")
        ) or ("tests" in rel_parts) or ("test" in rel_parts)
        module = module_name_for(rs, repo_path, src_root)
        tree = parser.parse(data)
        if tree.root_node is None:
            skipped += 1
            continue
        top = [c for c in tree.root_node.children if c.is_named]
        collect_items(top, repo_name, module, defs, in_test_ctx=bool(in_test_file))

    edges_src, edges_dst = resolve_edges(defs)
    if skipped:
        LOGGER.info("rust-extract: skipped %d unreadable file(s)", skipped)

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
