"""Hidden links: tests <-> subject, shared string keys, git co-change."""
import shutil
import subprocess

import pytest

from codegraph.indexer import index_repo
from codegraph.links import cochange_pairs, is_test_path, subject_stem
from codegraph.queries import Graph
from tests.conftest import link_set, make_repo


def build(tmp_path, files) -> Graph:
    root = make_repo(tmp_path, files)
    index_repo(root)
    return Graph(root)


@pytest.mark.parametrize("path,stem", [
    ("tests/test_store.py", "store"), ("pkg/store_test.py", "store"), ("src/foo.test.ts", "foo"),
    ("src/foo.spec.tsx", "foo"), ("internal/foo_test.go", "foo"), ("src/foo.py", None),
])
def test_subject_stem(path, stem):
    assert subject_stem(path) == stem


def test_is_test_path():
    assert is_test_path("tests/helpers.py") and is_test_path("a/b.test.ts") and not is_test_path("src/app.py")


def test_tests_link_to_subject_file_and_symbol_python(tmp_path):
    g = build(tmp_path, {
        "src/shop/cart.py": "def add_item(c, i):\n    return c\n\ndef total(c):\n    return 0\n",
        "tests/test_cart.py": "def test_add_item():\n    pass\n\ndef test_total_of_empty_cart():\n    pass\n\n"
                              "def test_unrelated_thing():\n    pass\n",
    })
    got = link_set(g, "tests")
    assert ("tests/test_cart.py:test_cart", "src/shop/cart.py:cart", "file") in got
    assert ("tests/test_cart.py:test_add_item", "src/shop/cart.py:add_item", "name") in got
    assert ("tests/test_cart.py:test_total_of_empty_cart", "src/shop/cart.py:total", "name") in got
    assert not any("test_unrelated_thing" in a for a, _, _ in got)


def test_tests_link_js_and_go_conventions(tmp_path):
    g = build(tmp_path, {
        "web/price.ts": "export function discount(p: number) { return p }\n",
        "web/price.test.ts": "function discountWorks() {}\nfunction testDiscount() {}\n",
        "pkg/calc.go": "package pkg\nfunc Add(a, b int) int { return a + b }\n",
        "pkg/calc_test.go": "package pkg\nfunc TestAdd(t *testing.T) {}\n",
    })
    got = link_set(g, "tests")
    assert ("web/price.test.ts:testDiscount", "web/price.ts:discount", "name") in got
    assert ("pkg/calc_test.go:TestAdd", "pkg/calc.go:Add", "name") in got


def test_ambiguous_stem_picks_the_nearest_file_or_nothing(tmp_path):
    g = build(tmp_path, {
        "a/util.py": "def f():\n    pass\n", "b/util.py": "def g():\n    pass\n",
        "tests/test_util.py": "def test_f():\n    pass\n",
    })
    assert link_set(g, "tests") == set()  # two equally distant candidates: no guess


def test_test_callers_are_labelled_tested_by(tmp_path):
    g = build(tmp_path, {
        "app.py": "def work():\n    return 1\n\ndef main():\n    return work()\n",
        "tests/test_app.py": "from app import work\n\ndef test_it():\n    assert work()\n",
    })
    out = g.related("work")
    assert "tested by" in out and "used by" in out
    assert "tested by" in g.context("work")


def test_route_and_event_links_across_files(tmp_path):
    g = build(tmp_path, {
        "server.py": '@app.get("/users")\ndef list_users():\n    bus.emit("users_listed")\n',
        "client.py": 'def load():\n    bus.on("users_listed", load)\n    return requests.get("/users")\n',
    })
    got = link_set(g, "key")
    assert ("server.py:list_users", "client.py:load", "route:/users") in got
    assert ("server.py:list_users", "client.py:load", "event:users_listed") in got
    assert "route:/users" in g.related("list_users")


def test_config_key_links_every_reader_and_same_file_pairs_are_skipped(tmp_path):
    g = build(tmp_path, {
        "a.py": 'def a():\n    return os.getenv("DB_URL")\n\ndef a2():\n    return os.getenv("DB_URL")\n',
        "b.py": 'def b():\n    return os.getenv("DB_URL")\n',
    })
    got = link_set(g, "key")
    assert ("a.py:a", "b.py:b", "config:DB_URL") in got and ("a.py:a2", "b.py:b", "config:DB_URL") in got
    assert not any(s.startswith("a.py") and d.startswith("a.py") for s, d, _ in got)


def test_a_key_shared_by_too_many_symbols_is_a_convention_not_a_link(tmp_path):
    files = {f"m{i}.py": f'def f{i}():\n    return os.getenv("HOME_DIR")\n' for i in range(12)}
    assert link_set(build(tmp_path, files), "key") == set()


def test_soft_links_never_leak_into_structural_tools(tmp_path):
    g = build(tmp_path, {
        "server.py": '@app.get("/users")\ndef list_users():\n    pass\n',
        "client.py": 'def load():\n    return requests.get("/users")\n',
    })
    assert "load" not in g.callers("list_users") and "load" not in g.impact("list_users")
    assert g.db.execute("SELECT COUNT(*) FROM edges WHERE kind NOT IN ('calls','inherits')").fetchone()[0] == 0


def test_links_follow_edits_and_deletions(tmp_path):
    root = make_repo(tmp_path, {
        "server.py": '@app.get("/users")\ndef list_users():\n    pass\n',
        "client.py": 'def load():\n    return requests.get("/users")\n',
    })
    index_repo(root)
    (root / "client.py").write_text("def load():\n    return 1\n")
    index_repo(root)
    assert link_set(Graph(root), "key") == set()
    (root / "client.py").write_text('def load():\n    return requests.get("/users")\n')
    index_repo(root)
    assert len(link_set(Graph(root), "key")) == 1
    (root / "client.py").unlink()
    index_repo(root)
    assert link_set(Graph(root), "key") == set()


# ---- git co-change ----
def test_cochange_pairs_weights_and_filters():
    known = {"a.py", "b.py", "c.py", "big1", "big2"}
    log = "\x1e\na.py\nb.py\n" * 3 + "\x1e\nb.py\nc.py\n" + "\x1e\n" + "\n".join(f"f{i}" for i in range(40)) + "\na.py\nc.py\n"
    pairs = cochange_pairs(log, known)
    assert ("a.py", "b.py") in pairs and pairs[("a.py", "b.py")] > 0.5
    assert ("b.py", "c.py") not in pairs  # once is not a habit
    assert ("a.py", "c.py") not in pairs  # co-changed only inside a huge commit


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


@needs_git
def test_cochange_edges_from_real_history_and_deps_line(tmp_path):
    root = make_repo(tmp_path, {"a.py": "def a():\n    pass\n", "b.py": "def b():\n    pass\n",
                                "c.py": "def c():\n    pass\n"})
    git(root, "init", "-q")
    for i in range(3):
        for f in ("a.py", "b.py"):
            (root / f).write_text((root / f).read_text() + f"# {i}\n")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", f"c{i}")
    index_repo(root)
    g = Graph(root)
    rows = g.db.execute("SELECT a, b FROM file_links WHERE kind='cochange'").fetchall()
    assert [tuple(r) for r in rows] == [("a.py", "b.py")]
    assert "co-changes with: b.py" in g.deps("a.py")
    assert "co-changes" not in g.deps("c.py")
    # unchanged HEAD: second index leaves it alone
    assert index_repo(root)["links"]["cochange"] == -1


def test_no_git_degrades_to_no_edges(tmp_path, monkeypatch):
    root = make_repo(tmp_path, {"a.py": "def a():\n    pass\n"})
    stats = index_repo(root)  # tmp_path is not a repository
    assert stats["links"]["cochange"] == 0
    monkeypatch.setattr("shutil.which", lambda _: None)  # and git itself missing
    assert index_repo(root, force=True)["links"]["cochange"] == 0
    assert "co-changes" not in Graph(root).deps("a.py")


@needs_git
def test_repo_map_uses_cochange_to_surface_a_file(tmp_path):
    root = make_repo(tmp_path, {"core.py": "def core():\n    pass\n\ndef core2():\n    core()\n",
                                "side.py": "def side():\n    pass\n", "lone.py": "def lone():\n    pass\n"})
    git(root, "init", "-q")
    for i in range(3):
        for f in ("core.py", "side.py"):
            (root / f).write_text((root / f).read_text() + f"# {i}\n")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", f"c{i}")
    index_repo(root)
    out = Graph(root).repo_map(focus="core")
    assert "side.py" in out and "lone.py" not in out  # no call links side to core: only history does
