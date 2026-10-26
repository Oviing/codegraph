"""Roles are a ranking prior only: they reorder equally relevant hits and never hide a symbol."""
from codegraph import roles
from codegraph.indexer import index_repo
from codegraph.queries import Graph
from tests.conftest import make_repo


def test_role_of():
    assert roles.role_of("pkg/api.py", "grad", "python") == "public"
    assert roles.role_of("pkg/api.py", "_split", "python") == "private"
    assert roles.role_of("pkg/api.py", "Foo._helper", "python") == "private"
    assert roles.role_of("pkg/api.py", "Foo.__init__", "python") == "public"  # dunders are API
    assert roles.role_of("lib/kernel.cc", "Cholesky", "python") == "native"
    assert roles.role_of("lib/kernel.cc", "Cholesky", "cpp") == "public"  # not foreign in a C++ repo
    assert roles.role_of("pkg/x_pb2.py", "Msg", "python") == "generated"


def test_dominant_lang():
    assert roles.dominant_lang(["a.py", "b.py", "c.cc", "README.md"]) == "python"
    assert roles.dominant_lang(["README.md"]) is None


def test_prior_keeps_an_explicit_private_query():
    assert roles.prior("a.py", "_split", "python", "split random key") < 1.0
    assert roles.prior("a.py", "_split", "python", "_split") == 1.0


FILES = {
    "lib/auth.py": "def _authenticate_session(n):\n    return n\n\ndef authenticate_session(n):\n    return n\n",
    "lib/mail.py": "def send_receipt(a):\n    return a\n",
}


def names(g, query):
    return [r["name"] for r, _ in g._hybrid_rows(query, 10)]


def test_public_symbol_outranks_private_twin(tmp_path, semantic, monkeypatch):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    g = Graph(root)
    on = names(g, "authenticate session")
    assert on[:2] == ["authenticate_session", "_authenticate_session"]
    monkeypatch.setenv("CODEGRAPH_ROLES", "off")
    off = names(g, "authenticate session")
    assert set(on) == set(off)  # roles reorder, they never hide a symbol


def test_spelling_the_private_name_keeps_it_first(tmp_path, semantic):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    assert names(Graph(root), "_authenticate_session")[0] == "_authenticate_session"
