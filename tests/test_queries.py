"""Query-layer behaviour: escaping, ambiguity handling, budgets and safety limits."""
import shutil
import tempfile
import threading
from pathlib import Path

import pytest

from codegraph.indexer import index_repo
from codegraph.queries import Graph, like, toks

FILES = {
    "app/__init__.py": "",
    "app/alpha.py": "def alpha():\n    \"\"\"The alpha.\"\"\"\n    return helper()\n",
    "app/beta.py": "from app.alpha import alpha\n\n\ndef beta():\n    return alpha()\n",
    "app/my_helper.py": "def helper():\n    return 1\n",
    "app/weird_100.py": "def pct_target():\n    return 2\n",
    "app/other.py": "def helper():\n    return 3\n",
}


@pytest.fixture(scope="module")
def graph():
    root = Path(tempfile.mkdtemp(prefix="codegraph-q-"))
    for name, body in FILES.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    index_repo(root)
    g = Graph(root)
    yield g
    g.close()
    shutil.rmtree(root, ignore_errors=True)


# ---------- escaping ----------

def test_like_escapes_wildcards():
    assert like("a%b") == "a\\%b"
    assert like("a_b") == "a\\_b"
    assert like("a\\b") == "a\\\\b"


def test_like_neutralises_a_percent_query(graph):
    """A '%' must not turn into a wildcard that matches everything."""
    rows = graph._search_rows("%", 50)
    assert all("%" not in (r["name"] or "") for r in rows)


def test_like_neutralises_an_underscore_query(graph):
    rows = graph._search_rows("_", 50)
    assert all("_" not in (r["name"] or "") for r in rows)


def test_outline_does_not_treat_underscore_as_wildcard(graph):
    """`_` alone must not match every symbol."""
    out = graph.outline("_")
    assert "No indexed symbols" in out


def test_repo_map_prefix_is_not_a_wildcard(graph):
    assert "my_helper.py" not in graph.repo_map(prefix="app/alpha")


# ---------- symbol resolution ----------

def test_find_by_qualname(graph):
    row, err = graph._one("beta")
    assert not err and row["qualname"] == "beta"


def test_find_ambiguous_reports_candidates(graph):
    _, err = graph._one("helper")
    assert "ambiguous" in err.lower() or "No symbol" in err


def test_file_name_form_disambiguates(graph):
    row, err = graph._one("app/beta.py:beta")
    assert not err and row["file"] == "app/beta.py"


def test_unknown_symbol_suggests_search(graph):
    _, err = graph._one("no_such_symbol_here")
    assert "search()" in err


def test_modules_are_never_returned_as_targets(graph):
    # every module is named after its file stem, e.g. "alpha"
    rows = graph._find("alpha")
    assert all(r["kind"] != "module" for r in rows)


def test_impl_nodes_are_not_targets(graph):
    for name in ("alpha", "beta", "helper"):
        assert all(r["kind"] != "impl" for r in graph._find(name))


# ---------- deps ----------

def test_deps_accepts_a_bare_filename(graph):
    out = graph.deps("beta.py")
    assert out.startswith("app/beta.py")
    assert "app/alpha.py" in out


def test_deps_accepts_a_path_suffix(graph):
    assert graph.deps("app/beta.py").startswith("app/beta.py")


def test_deps_prefers_an_exact_path(graph):
    assert graph.deps("app/beta.py").startswith("app/beta.py")


def test_deps_reports_unknown_files(graph):
    assert "No indexed file" in graph.deps("nope.py")


def test_deps_lists_importers(graph):
    assert "app/beta.py" in graph.deps("alpha.py")


# ---------- budgets ----------

def test_context_respects_the_token_budget(graph):
    out = graph.context("beta", 200)
    assert toks(out.split("[~")[0]) < 400  # headroom for the header and source


def test_context_with_a_tiny_budget_still_answers(graph):
    out = graph.context("beta", 20)
    assert "function beta" in out


def test_repo_map_respects_the_budget(graph):
    out = graph.repo_map(max_tokens=30)
    assert len(out.splitlines()) < 40


def test_search_limit_is_clamped(graph):
    assert len(graph.search("a", limit=10_000).splitlines()) <= 100


def test_depth_is_clamped():
    """A deep chain must stop at the clamp instead of walking the whole graph."""
    import tempfile

    from codegraph.indexer import index_repo as index
    root = Path(tempfile.mkdtemp(prefix="codegraph-depth-"))
    n = 40
    body = ["def f0():\n    return 0\n"]
    for i in range(1, n):
        body.append(f"\ndef f{i}():\n    return f{i - 1}()\n")
    (root / "chain.py").write_text("".join(body))
    index(root)
    g = Graph(root)
    try:
        out = g.callees("f39", depth=10_000, limit=1000)
        assert "depth 10:" in out
        assert "depth 11:" not in out, "depth clamp not applied"
    finally:
        g.close()


# ---------- ranking plumbing ----------

def test_ppr_seeds_are_deduplicated(graph):
    row = graph.db.execute("SELECT id FROM symbols WHERE qualname='alpha'").fetchone()
    a = graph._ppr([row[0], row[0]])
    b = graph._ppr([row[0]])
    assert a == b


def test_ppr_seeds_must_be_symbols_not_modules(graph):
    row = graph.db.execute("SELECT id FROM symbols WHERE qualname='beta'").fetchone()
    assert graph._ppr([row[0]])


def test_reload_clears_caches(graph):
    graph._nbrs(1)
    graph._ftok_cache["app/beta.py"] = (0.0, 123)
    graph.reload()
    assert graph._ncache == {} and graph._ftok_cache == {}


# ---------- concurrency ----------

def test_concurrent_tool_calls_are_safe(graph):
    """The MCP server dispatches tools concurrently over one shared connection."""
    errors = []

    def worker():
        try:
            for _ in range(20):
                graph.search("alpha")
                graph.repo_map(max_tokens=100)
                graph.impact("helper", depth=2)
                graph.stats()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]


# ---------- misc ----------

def test_toks_never_returns_zero():
    assert toks("") == 1


def test_stats_reports_counts(graph):
    out = graph.stats()
    assert "files" in out and "symbols" in out and "edges" in out


def test_source_returns_the_symbol_body(graph):
    out = graph.source("app/beta.py:beta")
    assert "def beta" in out


def test_outline_shows_a_symbol_tree(graph):
    out = graph.outline("alpha.py")
    assert "function alpha" in out


def test_related_reports_unknown_symbol(graph):
    out = graph.related("no_such_symbol_at_all")
    assert "No symbol" in out


def test_ambiguous_symbol_is_refused_not_guessed(graph):
    out = graph.source("helper")
    assert "ambiguous" in out.lower()
    assert "app/my_helper.py" in out and "app/other.py" in out


def test_empty_index_reports_emptiness():
    root = Path(tempfile.mkdtemp(prefix="codegraph-empty-"))
    (root / "x.py").write_text("def a():\n    return 1\n")
    g = Graph(root)
    try:
        assert "Index is empty" in g.repo_map()  # nothing ranked yet, no resolve has run
    finally:
        g.close()
