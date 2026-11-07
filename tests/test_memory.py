"""Agent memories: files as source of truth, anchors that survive reindexing, staleness, integrations."""
import json
import shutil
import sqlite3
import subprocess

import pytest

from codegraph import cli, memory
from codegraph.indexer import index_repo
from codegraph.install import STOP_COMMAND, install, stop_nudge, uninstall
from codegraph.queries import Graph
from tests.conftest import make_repo

SRC = '''def helper():
    return 1


def run():
    return helper()
'''
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@pytest.fixture
def repo(tmp_path):
    make_repo(tmp_path, {"app.py": SRC})
    index_repo(tmp_path)
    return tmp_path


def graph(root):
    return Graph(root)


def mem_id(out: str) -> str:
    return out.split()[1]


def test_remember_writes_a_committable_file_and_recall_finds_it(repo):
    g = graph(repo)
    out = g.remember("helper must stay pure", about="helper", kind="invariant",
                     why="run() caches it", alternatives="memoize inside helper: hides state")
    mid = mem_id(out)
    f = repo / ".codegraph" / "memory" / f"{mid}.md"
    text = f.read_text()
    assert text.startswith("---\nid: " + mid) and "kind: invariant" in text
    assert "  - app.py::helper#" in text and "## Why" in text and "## Alternatives considered" in text
    assert (repo / ".codegraph" / ".gitignore").read_text().startswith("graph.db*")
    assert "helper must stay pure" in g.recall("pure")                    # by words
    assert "helper must stay pure" in g.recall(about="helper")            # by anchor
    assert "helper must stay pure" in g.recall(about="run")               # by a neighbour (run calls helper)
    assert "No matching" in g.recall(about="helper", kind="decision")
    g.close()


def test_context_impact_and_stats_show_memories(repo):
    g = graph(repo)
    g.remember("never call helper twice", about="helper", kind="failed_attempt", why="double charge")
    assert "## memories" in g.context("helper") and "never call helper twice" in g.context("helper")
    assert "recorded memories" in g.impact("helper")
    assert "1 agent memories" in g.stats() and "recall()" in g.repo_map()
    assert "## memories" not in g.context("run")  # run is not anchored, and is not a neighbour here
    g.close()


def test_stale_when_code_changes_and_orphaned_when_it_goes(repo):
    g = graph(repo)
    g.remember("helper returns 1 on purpose", about="helper")
    g.close()
    (repo / "app.py").write_text(SRC.replace("return 1", "return 2"))
    index_repo(repo)
    g = graph(repo)
    assert "[stale: code changed since" in g.recall(about="helper")
    assert "[stale" in g.context("helper")
    g.close()
    (repo / "app.py").write_text("def run():\n    return 0\n")
    index_repo(repo)
    g = graph(repo)
    assert "[orphaned" in g.recall(query="purpose")
    assert "## memories" not in g.context("run")
    g.close()


def test_reformatting_is_not_stale_and_a_move_relinks(repo):
    g = graph(repo)
    g.remember("keep helper", about="helper")
    g.close()
    (repo / "app.py").write_text("def helper():\n        return 1\n\n\n\ndef run():\n    return helper()\n")
    index_repo(repo)
    g = graph(repo)
    assert "[stale" not in g.recall(about="helper")
    g.close()
    make_repo(repo, {"app.py": "def run():\n    return 0\n", "util.py": "def helper():\n    return 1\n"})
    index_repo(repo)
    g = graph(repo)
    out = g.recall(about="util.py:helper")
    assert "keep helper" in out and "[orphaned" not in out and "[stale" not in out
    g.close()


def test_memories_survive_force_reindex_and_follow_the_files(repo):
    g = graph(repo)
    mid = mem_id(g.remember("survives", about="helper"))
    g.close()
    index_repo(repo, force=True)
    g = graph(repo)
    assert "survives" in g.recall()
    g.close()
    (repo / ".codegraph" / "memory" / f"{mid}.md").unlink()
    index_repo(repo)
    g = graph(repo)
    assert "No memories yet" in g.recall()
    g.close()


def test_a_memory_file_added_by_hand_or_git_is_picked_up(repo):
    g = graph(repo)
    first = mem_id(g.remember("original", about="run"))
    src = (repo / ".codegraph" / "memory" / f"{first}.md").read_text()
    (repo / ".codegraph" / "memory" / "20260101-abcdef.md").write_text(
        src.replace(first, "20260101-abcdef").replace("original", "from a teammate"))
    assert "from a teammate" in g.recall(query="teammate")  # no reindex needed
    g.close()


def test_forget_retires_and_supersedes(repo):
    g = graph(repo)
    old = mem_id(g.remember("old plan", about="helper"))
    new = mem_id(g.remember("new plan", about="helper"))
    assert "retired" in g.forget(old[:12])
    assert "old plan" not in g.context("helper") and "new plan" in g.context("helper")
    assert (repo / ".codegraph" / "memory" / f"{old}.md").exists()  # kept as history
    assert "superseded by " + old in g.forget(new, superseded_by=old)
    assert "new plan" not in g.recall(about="helper")
    g.close()
    g = graph(repo)
    assert "No memory matches" in g.forget("nope")
    assert g.status == "missing"
    g.close()


def test_bad_input_is_reported_not_raised(repo):
    g = graph(repo)
    assert "Unknown kind" in g.remember("x", kind="rumour")
    assert "empty" in g.remember("   ")
    assert "No symbol or file matches" in g.remember("x", about="does_not_exist")
    assert g.status == "missing"
    assert "credential" in g.remember("use api_key=abcd1234efgh5678 to call it")
    assert "credential" in g.remember("fine", why="AKIAABCDEFGHIJKLMNOP")
    assert not (repo / ".codegraph" / "memory").exists()
    make_repo(repo, {"other.py": "def helper():\n    return 3\n"})
    index_repo(repo)
    g.reload()
    out = g.remember("x", about="helper")
    assert "ambiguous" in out and g.status == "ambiguous"
    g.close()


def test_file_anchor_and_repo_wide_memory(repo):
    g = graph(repo)
    g.remember("app.py is the entry point", about="app.py", kind="intent")
    g.remember("tests use fixtures/", kind="decision")  # no anchor
    assert "entry point" in g.context("run")  # file anchors show up on every symbol of the file
    assert "tests use fixtures/" in g.recall()
    assert "tests use fixtures/" not in g.context("run")
    g.close()


def test_invariants_are_listed_before_other_kinds(repo):
    g = graph(repo)
    g.remember("a decision", about="helper", kind="decision")
    g.remember("an invariant", about="helper", kind="invariant")
    ctx = g.context("helper")
    assert ctx.index("an invariant") < ctx.index("a decision")
    g.close()


def test_section_stays_inside_its_budget(repo):
    g = graph(repo)
    for i in range(12):
        g.remember(f"memory number {i} " + "word " * 30, about="helper")
    out = g.context("helper", max_tokens=400)
    assert "more (recall())" in out
    g.close()


@needs_git
def test_review_lists_memories_on_changed_code(repo):
    def git(*a):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                       check=True, capture_output=True)
    git("init", "-q")
    git("add", "app.py")
    git("commit", "-qm", "init")
    g = graph(repo)
    g.remember("helper is load bearing", about="helper", kind="invariant")
    g.close()
    (repo / "app.py").write_text(SRC.replace("return 1", "return 5"))
    g = graph(repo)
    out = g.review()
    assert "memories on the code this diff touches" in out and "helper is load bearing" in out
    assert "[stale" in out
    g.close()


def test_old_database_gets_the_tables(repo):
    db = sqlite3.connect(repo / ".codegraph" / "graph.db")
    for t in ("memories", "memory_anchors", "memory_links", "memory_fts"):
        db.execute(f"DROP TABLE {t}")
    db.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    db.commit()
    db.close()
    g = graph(repo)
    assert g.remember("works after upgrade", about="helper").startswith("remembered")
    assert "works after upgrade" in g.recall()
    g.close()


def test_file_format_roundtrip():
    meta = {"id": "x", "kind": "decision", "title": "a: title\nwith newline", "status": "active",
            "anchors": ["a.py::A.b#abc", "c.py"]}
    m, body = memory.parse(memory.render(meta, "text", "because", "other"))
    assert m["anchors"] == ["a.py::A.b#abc", "c.py"] and m["title"] == "a: title with newline"
    assert memory.split_body(body) == ("text", "because", "other")
    assert memory.parse_anchor("src/a.rs::Foo::bar#1f") == ("src/a.rs", "Foo::bar", "1f")
    with pytest.raises(ValueError):
        memory.parse("no frontmatter")


def test_cli_remember_and_recall(repo, capsys):
    assert cli.main(["-r", str(repo), "remember", "cli note", "--about", "helper", "--why", "because"]) == 0
    assert cli.main(["-r", str(repo), "recall", "--about", "helper"]) == 0
    assert "cli note" in capsys.readouterr().out
    assert cli.main(["-r", str(repo), "remember", "x", "--about", "nope"]) == 4


def test_install_adds_and_removes_the_stop_hook(tmp_path):
    install(tmp_path, "claude")
    install(tmp_path, "claude")
    data = json.loads((tmp_path / ".claude/settings.json").read_text())
    assert [e["hooks"][0]["command"] for e in data["hooks"]["Stop"]] == [STOP_COMMAND]
    assert "remember" in (tmp_path / "CLAUDE.md").read_text()
    uninstall(tmp_path)
    data = json.loads((tmp_path / ".claude/settings.json").read_text())
    assert data["hooks"]["Stop"] == [] and data["hooks"]["PreToolUse"] == []


@needs_git
def test_stop_nudge_blocks_once_and_only_when_code_changed_without_a_memory(repo):
    def git(*a):
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                       check=True, capture_output=True)
    git("init", "-q")
    git("add", "app.py")
    git("commit", "-qm", "init")
    assert stop_nudge(repo, "{}") == ""                          # clean tree
    (repo / "app.py").write_text(SRC + "\n# edit\n")
    assert json.loads(stop_nudge(repo, "{}"))["decision"] == "block"
    assert stop_nudge(repo, '{"stop_hook_active": true}') == ""  # never loops
    g = graph(repo)
    g.remember("explained", about="helper")
    g.close()
    assert stop_nudge(repo, "{}") == ""                          # a memory was written
