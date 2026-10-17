"""Tree-sitter based extraction of symbols and relations (no LLM involved)."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from tree_sitter_language_pack import get_parser

sys.setrecursionlimit(20000)

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".c": "cpp", ".h": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
}

_JS = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
}
_TS = {
    **_JS,
    "abstract_class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}
DEFS = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": _JS,
    "typescript": _TS,
    "tsx": _TS,
    "go": {"function_declaration": "function", "method_declaration": "method", "type_spec": "type"},
    "rust": {
        "function_item": "function", "struct_item": "struct", "enum_item": "enum",
        "trait_item": "trait", "impl_item": "impl", "mod_item": "module_decl",
    },
    "cpp": {
        "function_definition": "function", "class_specifier": "class", "struct_specifier": "struct",
        "namespace_definition": "namespace", "enum_specifier": "enum",
    },
}
# node type -> field holding the callee
CALLS = {
    "python": {"call": "function"},
    "javascript": {"call_expression": "function", "new_expression": "constructor"},
    "typescript": {"call_expression": "function", "new_expression": "constructor"},
    "tsx": {"call_expression": "function", "new_expression": "constructor"},
    "go": {"call_expression": "function"},
    "rust": {"call_expression": "function"},
    "cpp": {"call_expression": "function"},
}
IMPORTS = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "go": {"import_spec"},
    "rust": {"use_declaration"},
    "cpp": {"preproc_include"},
}
# callee shapes like `obj.name()`: the receiver type is unknown, so the target is only a guess
ATTR_CALLEES = {"attribute", "member_expression", "field_expression", "selector_expression"}
JSLIKE = {"javascript", "typescript", "tsx"}
FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
CONTAINER_KINDS = {"class", "impl", "struct", "trait", "interface"}
HERITAGE = {"class_heritage", "extends_type_clause", "extends_clause", "implements_clause"}
KW = {"extends", "implements", "public", "private", "protected", "virtual", "class", "struct"}


@dataclass
class Sym:
    id: int
    name: str
    qualname: str
    kind: str
    start: int
    end: int
    sig: str
    doc: str
    parent: int | None


@dataclass
class Parsed:
    syms: list[Sym] = field(default_factory=list)
    refs: list[tuple[int, str, str]] = field(default_factory=list)  # (src local id, name, kind)
    imports: list[str] = field(default_factory=list)


_parsers: dict = {}


def _parser(lang: str):
    if lang not in _parsers:
        _parsers[lang] = get_parser(lang)
    return _parsers[lang]


def _first_line(s: str) -> str:
    for line in s.splitlines():
        line = line.strip(" \t*/#!\"'")
        if line:
            return line[:160]
    return ""


def parse(src: bytes, lang: str, modname: str) -> Parsed:
    tree = _parser(lang).parse(src)
    out = Parsed()
    nlines = src.count(b"\n") + 1
    out.syms.append(Sym(0, modname, "", "module", 1, nlines, "", "", None))
    defs, calls, imports = DEFS[lang], CALLS[lang], IMPORTS[lang]

    def text(n) -> str:
        return src[n.start_byte:n.end_byte].decode("utf8", "replace")

    def last_ident(s: str):
        ids = IDENT.findall(s)
        return ids[-1] if ids else None

    def get_name(n):
        """Return (name, owner_override)."""
        t = n.type
        if t == "impl_item":
            ty = n.child_by_field_name("type")
            ids = IDENT.findall(text(ty)) if ty is not None else []
            return (ids[0] if ids else None), None
        if lang == "cpp" and t == "function_definition":
            d = n.child_by_field_name("declarator")
            while d is not None and d.child_by_field_name("declarator") is not None:
                d = d.child_by_field_name("declarator")
            if d is None:
                return None, None
            ids = IDENT.findall(text(d))
            if not ids:
                return None, None
            owner = ids[-2] if "::" in text(d) and len(ids) > 1 else None
            return ids[-1], owner
        if lang == "go" and t == "method_declaration":
            nm = n.child_by_field_name("name")
            rc = n.child_by_field_name("receiver")
            owner = last_ident(text(rc)) if rc is not None else None
            return (text(nm) if nm is not None else None), owner
        nm = n.child_by_field_name("name")
        return (text(nm) if nm is not None else None), None

    def get_doc(n) -> str:
        if lang == "python":
            body = n.child_by_field_name("body")
            if body is not None and body.named_children:
                f = body.named_children[0]
                if f.type == "expression_statement" and f.named_children and f.named_children[0].type == "string":
                    return _first_line(text(f.named_children[0]))
            return ""
        p = n.prev_named_sibling
        if p is not None and "comment" in p.type and p.end_point[0] >= n.start_point[0] - 1:
            return _first_line(text(p))
        return ""

    def sig(n) -> str:
        first = text(n).split("\n", 1)[0].strip()
        return first[:160]

    def collect_inherits(n, sid: int):
        t = n.type
        if lang == "python" and t == "class_definition":
            sc = n.child_by_field_name("superclasses")
            if sc is not None:
                for c in sc.named_children:
                    if c.type in ("identifier", "attribute"):
                        nm = last_ident(text(c))
                        if nm:
                            out.refs.append((sid, nm, "inherits"))
        elif lang in JSLIKE and t in ("class_declaration", "abstract_class_declaration", "interface_declaration"):
            for c in n.children:
                if c.type in HERITAGE:
                    for ident in IDENT.findall(text(c)):
                        if ident not in KW:
                            out.refs.append((sid, ident, "inherits"))
        elif lang == "cpp" and t in ("class_specifier", "struct_specifier"):
            for c in n.children:
                if c.type == "base_class_clause":
                    for ident in IDENT.findall(text(c)):
                        if ident not in KW:
                            out.refs.append((sid, ident, "inherits"))
        elif lang == "rust" and t == "impl_item":
            tr = n.child_by_field_name("trait")
            if tr is not None:
                nm = last_ident(text(tr).split("<")[0])
                if nm:
                    out.refs.append((sid, nm, "inherits"))

    def import_raws(n) -> list[str]:
        t = n.type
        if lang == "python":
            if t == "import_from_statement":
                m = n.child_by_field_name("module_name")
                return [text(m)] if m is not None else []
            res = []
            for c in n.named_children:
                if c.type == "aliased_import":
                    c = c.child_by_field_name("name") or c
                res.append(text(c))
            return res
        if lang in JSLIKE:
            s = n.child_by_field_name("source")
            return [text(s).strip("'\"`")] if s is not None else []
        if lang == "go":
            p = n.child_by_field_name("path")
            return [text(p).strip('"`')] if p is not None else []
        if lang == "rust":
            a = n.child_by_field_name("argument")
            return [text(a)] if a is not None else []
        if lang == "cpp":
            p = n.child_by_field_name("path")
            return [text(p).strip('<>"')] if p is not None else []
        return []

    def is_self_call(f) -> bool:
        """`self.x()` / `cls.x()` (python), `this.x()` (js/ts/c++), `self.x()` (rust)."""
        if f.type == "attribute":
            o = f.child_by_field_name("object")
            return o is not None and o.type == "identifier" and text(o) in ("self", "cls")
        if f.type == "member_expression":
            o = f.child_by_field_name("object")
            return o is not None and o.type == "this"
        if f.type == "field_expression":
            o = f.child_by_field_name("value") or f.child_by_field_name("argument")
            return o is not None and o.end_byte - o.start_byte <= 4 and text(o) in ("self", "this")
        return False

    def walk(n, cur: int):
        t = n.type
        kind = defs.get(t)
        if kind is None and lang in JSLIKE and t == "variable_declarator":
            v = n.child_by_field_name("value")
            if v is not None and v.type in FUNC_VALUES:
                kind = "function"
        new = cur
        if kind:
            name, owner = get_name(n)
            if name:
                parent = out.syms[cur]
                prefix = owner or parent.qualname
                qn = f"{prefix}.{name}" if prefix else name
                if kind == "function" and parent.kind in CONTAINER_KINDS:
                    kind = "method"
                if kind == "module_decl":
                    kind = "module"
                s = Sym(len(out.syms), name, qn, kind, n.start_point[0] + 1, n.end_point[0] + 1,
                        sig(n), get_doc(n), cur)
                out.syms.append(s)
                new = s.id
                collect_inherits(n, s.id)
        cm = calls.get(t)
        if cm:
            f = n.child_by_field_name(cm)
            if f is not None and f.end_byte - f.start_byte <= 120:
                nm = last_ident(text(f))
                if nm:
                    rk = "calls_self" if is_self_call(f) else "calls_attr" if f.type in ATTR_CALLEES else "calls"
                    out.refs.append((cur, nm, rk))
        if t in imports:
            out.imports.extend(import_raws(n))
        for c in n.children:
            walk(c, new)

    walk(tree.root_node, 0)
    return out
