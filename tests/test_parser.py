"""Parser-level tests: symbol extraction, heritage parsing and call classification."""
import pytest

from codegraph.parser import DEFS, parse


def syms(src, lang, mod="m"):
    return [(s.qualname, s.kind) for s in parse(src.encode(), lang, mod).syms][1:]  # drop module


def test_python_symbols_and_docs():
    out = syms('"""Mod."""\nCONST = 1\n\n\ndef f(a, b=2):\n    """Doc."""\n    return a\n', "python")
    assert ("CONST", "constant") in out
    assert ("f", "function") in out


def test_python_multiline_signature_joined():
    out = parse(b"def f(\n    a,\n    b,\n):\n    pass\n", "python", "m").syms[1]
    assert "(" in out.sig and ")" in out.sig
    assert "\n" not in out.sig


def test_self_call_is_classified_separately():
    # the ref is attributed to the enclosing method (id 2), not to the class
    p = parse(b"class C:\n    def a(self):\n        self.b()\n    def b(self):\n        pass\n", "python", "m")
    assert (2, "b", "calls_self") in p.refs


def test_self_call_is_marked_at_parse_time():
    # downgrading to calls_attr when `self` is outside a class happens in the store, not the parser
    p = parse(b"def top():\n    self.helper()\n", "python", "m")
    assert (1, "helper", "calls_self") in p.refs


@pytest.mark.parametrize("lang,src,base", [
    # the regression: `extends Base<T>` must not produce a phantom edge to the type parameter T
    ("typescript", "class A extends Base<T> {}", "Base"),
    ("tsx", "class A extends Base<T> {}", "Base"),
    ("javascript", "class A extends Base {}", "Base"),
    ("cpp", "struct D : ns::Base<int> { int x; };", "Base"),
])
def test_heritage_ignores_type_arguments(lang, src, base):
    p = parse(src.encode(), lang, "m")
    inherits = [name for _, name, kind in p.refs if kind == "inherits"]
    assert inherits == [base], f"{lang}: got {inherits}, want only ['{base}']"


def test_heritage_multiple_bases_comma_separated():
    src = b"interface A extends Base<T>, Other { m(): void }"
    p = parse(src, "typescript", "m")
    assert sorted(n for _, n, k in p.refs if k == "inherits") == ["Base", "Other"]


def test_implements_clause_is_parsed():
    p = parse(b"class W implements B { m(): void {} }", "typescript", "m")
    assert ("inherits", "B") in [(k, n) for _, n, k in p.refs]


def test_rust_trait_impl_inherits():
    p = parse(b"struct S;\ntrait T {}\nimpl T for S {}", "rust", "m")
    assert ("inherits", "T") in [(k, n) for _, n, k in p.refs]


def test_rust_doc_comment():
    p = parse(b"/// Greets.\n/// Second line.\npub fn g() {}\n", "rust", "m")
    assert p.syms[1].doc == "Greets. Second line."


def test_go_interface_methods_are_symbols():
    p = parse(b"package a\ntype I interface{ Do() error }\n", "go", "a")
    assert ("I.Do", "method") in [(s.qualname, s.kind) for s in p.syms]


def test_go_method_owner_qualname():
    p = parse(b"package a\nfunc (s *S) M() {}\n", "go", "a")
    assert ("S.M", "method") in [(s.qualname, s.kind) for s in p.syms]


def test_ts_interface_members():
    p = parse(b"interface I { m(x: number): void; readonly p: string }", "typescript", "m")
    got = {(s.qualname, s.kind) for s in p.syms}
    assert ("I.m", "method") in got and ("I.p", "field") in got


def test_cpp_out_of_line_method_is_a_method():
    p = parse(b"class Shape { public: int area(); };\nint Shape::area() { return 1; }\n", "cpp", "m")
    kinds = {s.qualname: s.kind for s in p.syms}
    assert kinds.get("Shape.area") == "method"


def test_js_arrow_function_and_object_literal_getter():
    src = b"const f = (x) => x + 1;\nconst o = { get g() { return 1; } };\n"
    got = {s.qualname for s in parse(src, "javascript", "m").syms}
    assert {"f", "o", "o.g"} <= got   # the getter is nested under the object variable


def test_quoted_keys_are_not_symbols():
    p = parse(b'const o = { "not-an-ident": function () {} };\n', "javascript", "m")
    assert not [s for s in p.syms if s.name == "not-an-ident"]


def test_deeply_nested_source_does_not_blow_the_stack():
    # the walk is iterative: deep nesting must not depend on the recursion limit
    depth = 3000
    src = ("def f():\n" + "".join("    if x:\n" for _ in range(depth)) + "    pass\n").encode()
    p = parse(src, "python", "m")
    assert any(s.qualname == "f" for s in p.syms)


def test_unicode_identifiers():
    p = parse("def café(x):\n    return x\n".encode(), "python", "m")
    assert p.syms[1].name == "café"


def test_every_language_in_ext_maps_to_a_def_table():
    from codegraph.parser import EXT
    for lang in EXT.values():
        assert lang in DEFS, f"{lang} has no DEFS entry"
        assert lang in {"python", "javascript", "typescript", "tsx", "go", "rust", "cpp"}


def test_call_length_cap_skips_huge_callees():
    long = "x" * 200
    p = parse(f"def f():\n    d.{long}()\n".encode(), "python", "m")
    assert not [r for r in p.refs if long in r[1]]
