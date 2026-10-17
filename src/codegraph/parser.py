"""Tree-sitter based extraction of symbols and relations (no LLM involved)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tree_sitter_language_pack import get_parser

from . import keys

IDENT = re.compile(r"[^\W\d][\w]*", re.UNICODE)
CONST_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

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
    "pair": "method",            # `get x() {}` / `set x(v) {}` in an object literal
    "public_field_definition": "field",
}
_TS = {
    **_JS,
    "abstract_class_declaration": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "method_signature": "method",       # `m(x: T): R;` inside an interface
    "property_signature": "field",
}
DEFS = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": _JS,
    "typescript": _TS,
    "tsx": _TS,
    "go": {"function_declaration": "function", "method_declaration": "method",
           "type_spec": "type", "method_elem": "method"},
    "rust": {
        "function_item": "function", "struct_item": "struct", "enum_item": "enum",
        "trait_item": "trait", "impl_item": "impl", "mod_item": "module_decl",
        "trait_item_signature": "method", "function_signature_item": "function",
    },
    "cpp": {
        "function_definition": "function", "class_specifier": "class", "struct_specifier": "struct",
        "namespace_definition": "namespace", "enum_specifier": "enum", "type_definition": "type",
    },
}
# node type -> field holding the callee
CALLS = {
    "python": {"call": "function"},
    "javascript": {"call_expression": "function", "new_expression": "constructor"},
    "typescript": {"call_expression": "function", "new_expression": "constructor"},
    "tsx": {"call_expression": "function", "new_expression": "constructor"},
    "go": {"call_expression": "function"},
    "rust": {"call_expression": "function", "macro_invocation": "macro"},
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
ATTR_CALLEES = {"attribute", "member_expression", "field_expression", "selector_expression",
                "scoped_identifier", "qualified_name"}
JSLIKE = {"javascript", "typescript", "tsx"}
FUNC_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
CONTAINER_KINDS = {"class", "impl", "struct", "trait", "interface", "namespace"}
HERITAGE = {"class_heritage", "extends_type_clause", "extends_clause", "implements_clause",
            "base_class_clause", "superclasses", "type_parameters"}
# subtrees inside a heritage clause that must never contribute a name (generic arguments,
# type parameters) -- this is what used to turn `extends Base<T>` into a phantom `T` edge
SKIP_SUBTREES = {"type_arguments", "type_parameters", "type_parameter", "type_parameter_list",
                 "modifiers", "attribute_item", "variance", "constraints"}
NAME_NODES = {"identifier", "type_identifier", "field_identifier", "constant", "property_identifier"}
KW = {"extends", "implements", "public", "private", "protected", "virtual", "class", "struct",
      "template", "typename", "using", "interface", "enum", "namespace", "operator"}


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
    keys: list[tuple[int, str, str, str]] = field(default_factory=list)  # (src local id, ns, key, role)


_parsers: dict = {}


def _parser(lang: str):
    if lang not in _parsers:
        _parsers[lang] = get_parser(lang)
    return _parsers[lang]


def _clean_comment(raw: str) -> str:
    """Turn any supported comment syntax into its first line of prose."""
    for line in raw.splitlines():
        line = line.strip()
        for prefix in ("///", "//!", "//", "/**", "/*", "*/", "///<", "//!<", "*", "#!", "#", "<!--", "-->"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        line = line.strip(" \t*/#!\"'").strip()
        if line and set(line) != {"*"}:
            return line[:160]
    return ""


def _first_line(s: str) -> str:
    return _clean_comment(s)


class _FileParser:
    """Walks one file's syntax tree once and collects symbols, references and imports."""

    def __init__(self, src: bytes, lang: str, modname: str):
        self.src = src
        self.lang = lang
        self.modname = modname
        self.out = Parsed()
        self.defs = DEFS[lang]
        self.calls = CALLS[lang]
        self.imports_set = IMPORTS[lang]

    # ---- small helpers ----
    def text(self, n) -> str:
        return self.src[n.start_byte:n.end_byte].decode("utf8", "replace")

    def field(self, n, *names):
        for nm in names:
            c = n.child_by_field_name(nm)
            if c is not None:
                return c
        return None

    @staticmethod
    def last_ident(s: str) -> str | None:
        ids = IDENT.findall(s)
        return ids[-1] if ids else None

    # ---- naming ----
    def get_name(self, n) -> tuple[str | None, str | None]:
        """Return (name, owner_override) for a definition node."""
        t = n.type
        if t == "impl_item":
            ty = self.field(n, "type")
            ids = IDENT.findall(self.text(ty)) if ty is not None else []
            return (ids[0] if ids else None), None
        if self.lang == "cpp" and t == "function_definition":
            d = self.field(n, "declarator")
            while d is not None and self.field(d, "declarator") is not None:
                d = self.field(d, "declarator")
            if d is None:
                return None, None
            ids = IDENT.findall(self.text(d))
            if not ids:
                return None, None
            owner = ids[-2] if "::" in self.text(d) and len(ids) > 1 else None
            return ids[-1], owner
        if self.lang == "go" and t == "method_declaration":
            nm = self.field(n, "name")
            rc = self.field(n, "receiver")
            owner = self.last_ident(self.text(rc)) if rc is not None else None
            return (self.text(nm) if nm is not None else None), owner
        if t in ("method_signature", "property_signature", "pair", "public_field_definition"):
            nm = self.field(n, "name", "key", "property")
            if nm is None or nm.type in ("string", "string_literal", "computed_property_name"):
                return None, None  # `{"a-b": fn}` has no identifier to link
            return self.last_ident(self.text(nm)), None
        if t == "assignment" and self.lang == "python":
            left = self.field(n, "left")
            return (self.text(left) if left is not None and left.type == "identifier" else None), None
        nm = self.field(n, "name")
        return (self.text(nm) if nm is not None else None), None

    # ---- doc comments ----
    def get_doc(self, n) -> str:
        if self.lang == "python":
            body = self.field(n, "body")
            if body is not None and body.named_children:
                f = body.named_children[0]
                if f.type == "expression_statement" and f.named_children and f.named_children[0].type == "string":
                    return _clean_comment(self.text(f.named_children[0]))
            return ""
        # collect the run of comment lines directly above the definition
        parts: list[str] = []
        p = n.prev_named_sibling
        while p is not None and "comment" in p.type and p.end_point[0] >= n.start_point[0] - 1:
            line = _clean_comment(self.text(p))
            if line:
                parts.insert(0, line)
            p = p.prev_named_sibling
        return " ".join(parts)[:160]

    # ---- signatures ----
    def sig(self, n) -> str:
        """Header of the definition: the first line, extended while parentheses stay unbalanced."""
        lines = self.text(n).split("\n")
        head = lines[0].strip()
        for extra in lines[1:4]:  # a signature may wrap over a few lines
            if head.count("(") <= head.count(")"):
                break
            head = (head + " " + extra.strip()).strip()
        return head[:160]

    # ---- inheritance ----
    def _collect_ids(self, n, bucket: list[str], depth: int = 0):
        if depth > 8 or n.type in SKIP_SUBTREES:
            return
        if n.type in NAME_NODES:
            txt = self.text(n)
            if txt and txt not in KW:
                bucket.append(txt)
            return
        for c in n.children:
            self._collect_ids(c, bucket, depth + 1)

    def _clause_names(self, clause, sid: int):
        """Each comma-separated entry of a heritage clause contributes exactly one name:
        the last identifier in its subtree, so `ns::Outer` and `Base<T>` both resolve right."""
        bucket: list[str] = []

        def flush():
            if bucket:
                name = bucket[-1]
                if name not in KW:
                    self.out.refs.append((sid, name, "inherits"))
                bucket.clear()

        for c in clause.children:
            # keywords (`extends`, `implements`, access specifiers) end the current entry; note that
            # `extends_clause` / `implements_clause` are containers and must be descended into
            if c.type == "," or c.type in KW:
                flush()
                continue
            self._collect_ids(c, bucket)
        flush()

    def collect_inherits(self, n, sid: int):
        lang = self.lang
        if lang == "python" and n.type == "class_definition":
            sc = self.field(n, "superclasses")
            if sc is not None:
                for c in sc.named_children:  # python keeps one entry per base, no clause commas
                    bucket: list[str] = []
                    self._collect_ids(c, bucket)
                    if bucket and bucket[-1] not in KW:
                        self.out.refs.append((sid, bucket[-1], "inherits"))
        elif lang in JSLIKE and n.type in ("class_declaration", "abstract_class_declaration",
                                           "interface_declaration"):
            for c in n.children:
                if c.type in HERITAGE:
                    self._clause_names(c, sid)
        elif lang == "cpp" and n.type in ("class_specifier", "struct_specifier"):
            for c in n.children:
                if c.type == "base_class_clause":
                    self._clause_names(c, sid)
        elif lang == "rust" and n.type == "impl_item":
            tr = self.field(n, "trait")
            if tr is not None:
                bucket: list[str] = []
                self._collect_ids(tr, bucket)
                if bucket and bucket[-1] not in KW:
                    self.out.refs.append((sid, bucket[-1], "inherits"))
        # Go interfaces have no base clause; their methods are indexed as method_elem

    # ---- imports ----
    def import_raws(self, n) -> list[str]:
        t = n.type
        lang = self.lang
        if lang == "python":
            if t == "import_from_statement":
                m = self.field(n, "module_name")
                if m is None:
                    return []
                mod = self.text(m)
                # `from app import util` may mean the submodule app/util.py rather than the
                # package itself, so offer both readings and let the store pick what exists
                cands = [mod]
                seen_import = False
                for c in n.children:  # only what follows the `import` keyword
                    if c.type == "import":
                        seen_import = True
                        continue
                    if not seen_import:
                        continue
                    if c.type == "dotted_name":
                        cands.append(f"{mod}.{self.text(c)}")
                    elif c.type == "aliased_import":
                        target = self.field(c, "name")
                        if target is not None and target.type != "dotted_name":
                            cands.append(f"{mod}.{self.text(target)}")
                return cands
            res = []
            for c in n.named_children:
                if c.type == "aliased_import":
                    c = self.field(c, "name") or c
                res.append(self.text(c))
            return res
        if lang in JSLIKE:
            s = self.field(n, "source")
            return [self.text(s).strip("'\"`")] if s is not None else []
        if lang == "go":
            p = self.field(n, "path")
            return [self.text(p).strip('"`')] if p is not None else []
        if lang == "rust":
            a = self.field(n, "argument")
            return [self.text(a)] if a is not None else []
        if lang == "cpp":
            p = self.field(n, "path")
            return [self.text(p).strip('<>"\'')] if p is not None else []
        return []

    # ---- calls ----
    def is_self_call(self, f) -> bool:
        """`self.x()` / `cls.x()` (python), `this.x()` (js/ts/c++), `self.x()` (rust)."""
        if f.type == "attribute":
            o = self.field(f, "object")
            return o is not None and o.type == "identifier" and self.text(o) in ("self", "cls")
        if f.type == "member_expression":
            o = self.field(f, "object")
            return o is not None and o.type == "this"
        if f.type == "field_expression":
            o = self.field(f, "value", "argument")
            return o is not None and o.end_byte - o.start_byte <= 4 and self.text(o) in ("self", "this")
        return False

    # ---- string-keyed links (routes, events, config keys; see keys.py) ----
    def first_string_arg(self, call) -> str | None:
        args = self.field(call, "arguments")
        if args is None or not args.named_children:
            return None
        a = args.named_children[0]
        if a.type not in ("string", "template_string", "string_literal"):
            return None
        return keys.string_value(self.text(a))

    def collect_key(self, call, callee, src: int, decorator: bool = False):
        if self.lang not in keys.ANY:
            return
        key = self.first_string_arg(call)
        if key is not None:
            for ns, k, role in keys.match(self.lang, self.text(callee), key, decorator):
                self.out.keys.append((src, ns, k, role))

    def collect_decorator_keys(self, defn, sid: int):
        """`@app.get("/x")` registers the *decorated* function, not the scope the decorator sits in."""
        parent = defn.parent
        if self.lang != "python" or parent is None or parent.type != "decorated_definition":
            return
        for dec in parent.children:
            call = next((c for c in dec.named_children if c.type == "call"), None) if dec.type == "decorator" else None
            callee = self.field(call, "function") if call is not None else None
            if callee is not None:
                self.collect_key(call, callee, sid, decorator=True)

    # ---- main walk (iterative: no recursion limit games) ----
    def run(self) -> Parsed:
        tree = _parser(self.lang).parse(self.src)
        nlines = self.src.count(b"\n") + 1
        self.out.syms.append(Sym(0, self.modname, "", "module", 1, nlines, "", "", None))
        stack = [(tree.root_node, 0)]
        while stack:
            n, cur = stack.pop()
            t = n.type
            kind = self.defs.get(t)
            if kind is None and self.lang in JSLIKE and t == "variable_declarator":
                kind = self._js_declarator_kind(n, cur)
            elif kind is None and self.lang == "python" and t == "assignment":
                kind = self._py_assignment_kind(n, cur)
            new = cur
            if kind:
                name, owner = self.get_name(n)
                if name:
                    parent = self.out.syms[cur]
                    prefix = owner or parent.qualname
                    qn = f"{prefix}.{name}" if prefix else name
                    if kind == "function" and parent.kind in CONTAINER_KINDS:
                        kind = "method"
                    # `int Shape::area() {}` at file scope: the owner comes from the declarator
                    if kind == "function" and owner:
                        kind = "method"
                    if kind == "module_decl":
                        kind = "module"
                    s = Sym(len(self.out.syms), name, qn, kind,
                            n.start_point[0] + 1, n.end_point[0] + 1,
                            self.sig(n), self.get_doc(n), cur)
                    self.out.syms.append(s)
                    new = s.id
                    self.collect_inherits(n, s.id)
                    self.collect_decorator_keys(n, s.id)
            cm = self.calls.get(t)
            if cm:
                f = self.field(n, cm)
                if f is not None and f.end_byte - f.start_byte <= 120:
                    nm = self.last_ident(self.text(f))
                    if nm:
                        rk = ("calls_self" if self.is_self_call(f)
                              else "calls_attr" if f.type in ATTR_CALLEES else "calls")
                        self.out.refs.append((cur, nm, rk))
                    if not (n.parent is not None and n.parent.type == "decorator"):  # handled at the def
                        self.collect_key(n, f, cur)
            if t in self.imports_set:
                self.out.imports.extend(self.import_raws(n))
            for c in reversed(n.children):
                stack.append((c, new))
        return self.out

    def _js_declarator_kind(self, n, cur) -> str | None:
        parent = self.out.syms[cur]
        v = self.field(n, "value")
        if v is not None and v.type in FUNC_VALUES:
            return "function"
        if parent.kind in ("module",) or parent.kind in CONTAINER_KINDS:
            return "variable"
        return None

    def _py_assignment_kind(self, n, cur) -> str | None:
        """Only ALL_CAPS module-level bindings, so constants show up without flooding."""
        parent = self.out.syms[cur]
        if parent.kind not in ("module", "class"):
            return None
        targets = self.field(n, "left") or (n.named_children[0] if n.named_children else None)
        if targets is None or targets.type != "identifier":
            return None
        txt = self.text(targets)
        return "constant" if CONST_NAME.match(txt) else None


def parse(src: bytes, lang: str, modname: str) -> Parsed:
    return _FileParser(src, lang, modname).run()
