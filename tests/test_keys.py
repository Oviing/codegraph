"""String-key extraction: what the parser records for routes, events and config keys."""
import pytest

from codegraph.keys import string_value
from codegraph.parser import parse


def keys_of(src: str, lang: str) -> set[tuple[str, str, str, str]]:
    p = parse(src.encode(), lang, "m")
    names = {s.id: s.qualname or s.name for s in p.syms}
    return {(names[i], ns, k, role) for i, ns, k, role in p.keys}


def test_python_decorator_route_belongs_to_the_decorated_function():
    got = keys_of('@app.get("/users")\ndef list_users():\n    pass\n', "python")
    assert got == {("list_users", "route", "/users", "provides")}


def test_python_client_call_uses_the_route_but_does_not_provide_it():
    got = keys_of('def f():\n    return requests.get("/users")\n', "python")
    assert got == {("f", "route", "/users", "uses")}


def test_python_events_and_config():
    got = keys_of('def f():\n    bus.emit("order_created")\n    return os.getenv("DB_URL")\n', "python")
    assert got == {("f", "event", "order_created", "emits"), ("f", "config", "DB_URL", "reads")}


@pytest.mark.parametrize("lang", ["javascript", "typescript", "tsx"])
def test_js_routes_and_events(lang):
    src = ('function setup() { app.post("/items", h); em.on("order_created", cb); }\n'
           'function load() { return fetch("/items"); }\n')
    got = keys_of(src, lang)
    assert ("setup", "route", "/items", "provides") in got
    assert ("setup", "event", "order_created", "listens") in got
    assert ("load", "route", "/items", "uses") in got


def test_generic_events_and_computed_keys_are_ignored():
    got = keys_of('function f(x) { s.on("error", cb); s.on("data", cb); bus.emit(`a${x}`); bus.emit(name); }',
                  "javascript")
    assert got == set()


def test_other_languages_are_not_scanned():
    assert parse(b'package m\nfunc f() { app.Get("/x") }\n', "go", "m").keys == []


@pytest.mark.parametrize("text,expected", [
    ('"abc"', "abc"), ("'abc'", "abc"), ('r"a\\b"', "a\\b"), ("`abc`", "abc"),
    ("`a${x}`", None), ('f"a{x}"', None), ("name", None),
])
def test_string_value(text, expected):
    assert string_value(text) == expected
