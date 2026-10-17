"""Edge confidence, exit codes, .codegraphignore, docs/SQL input, communities, report, review, install."""
import json
import subprocess
from pathlib import Path

import pytest

from codegraph import cli
from codegraph.indexer import index_repo
from codegraph.queries import Graph

PY = '''
def helper():
    return 1


def run():
    return helper()


class A:
    def go(self):
        return self.stop()

    def stop(self):
        return 2
'''
OTHER = "def helper():\n    return 2\n\n\ndef use():\n    return other_thing()\n"


def make(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def edges(root: Path):
    g = Graph(root)
    try:
        return {tuple(r) for r in g.db.execute(
            "SELECT sa.qualname, sb.qualname, e.conf FROM edges e "
            "JOIN symbols sa ON sa.id=e.src JOIN symbols sb ON sb.id=e.dst")}
    finally:
        g.close()


def test_edge_confidence(tmp_path):
    root = make(tmp_path, {"a.py": PY, "b.py": OTHER})
    index_repo(root)
    e = edges(root)
    assert ("run", "helper", "extracted") in e          # same file, one candidate
    assert ("A.go", "A.stop", "extracted") in e         # self.x() inside the class
    g = Graph(root)
    out = g.callers("a.py:helper")
    g.close()
    assert "run" in out and "[" not in out.split("\n", 1)[1]  # extracted edges carry no tag


def test_ambiguous_edge_is_tagged(tmp_path):
    root = make(tmp_path, {"x.py": "def go():\n    return shared()\n",
                           "p/shared.py": "def shared():\n    return 1\n",
                           "q/shared.py": "def shared():\n    return 2\n"})
    index_repo(root)
    assert {c for _, d, c in edges(root) if d == "shared"} == {"ambiguous"}
    g = Graph(root)
    assert "[ambiguous]" in g.callees("go")
    g.close()


def test_exit_codes(tmp_path, capsys):
    root = make(tmp_path, {"a.py": PY, "b.py": OTHER})
    index_repo(root)
    assert cli.main(["-r", str(root), "source", "a.py:run"]) == 0
    assert cli.main(["-r", str(root), "source", "helper"]) == 3     # defined in a.py and b.py
    assert cli.main(["-r", str(root), "source", "nope_nothing"]) == 4
    assert cli.main(["-r", str(root), "outline", "missing.py"]) == 4
    capsys.readouterr()


def test_codegraphignore(tmp_path):
    root = make(tmp_path, {"a.py": PY, "gen/out.py": PY, "keep/gen_not.py": PY, "scratch.py": PY,
                           ".codegraphignore": "# generated\ngen/\n/scratch.py\n*.pyc\n"})
    r = index_repo(root)
    g = Graph(root)
    files = {x[0] for x in g.db.execute("SELECT path FROM files")}
    g.close()
    assert files == {"a.py", "keep/gen_not.py"} and r["files"] == 2
    (root / ".codegraphignore").write_text("")           # un-ignoring brings them back
    assert index_repo(root)["files"] == 4


def test_codegraphignore_negation(tmp_path):
    root = make(tmp_path, {"a.py": PY, "b.py": PY, ".codegraphignore": "*.py\n!a.py\n"})
    assert index_repo(root)["files"] == 1


def test_markdown_and_sql(tmp_path):
    root = make(tmp_path, {
        "a.py": PY,
        "README.md": "# Title\n\nintro\n\n## Usage\n\nCall `run()` or `A.go`.\n\n```\n# not a heading\n```\n",
        "schema.sql": "-- users\nCREATE TABLE users (id int);\n\n"
                      "CREATE VIEW active AS\n  SELECT * FROM users JOIN orders ON 1=1;\n\n"
                      "CREATE TABLE orders (user_id int REFERENCES users(id));\n"})
    index_repo(root)
    g = Graph(root)
    kinds = {(r["kind"], r["name"]) for r in g.db.execute("SELECT kind, name FROM symbols")}
    assert ("section", "Usage") in kinds and ("section", "Title") in kinds
    assert ("section", "not a heading") not in kinds
    assert {("table", "users"), ("view", "active"), ("table", "orders")} <= kinds
    e = edges(root)
    assert any(s.endswith("Usage") and d == "run" for s, d, _ in e)       # doc -> code
    assert ("active", "users", "inferred") in e or ("active", "users", "extracted") in e
    assert any(s == "orders" and d == "users" for s, d, _ in e)
    assert "active" in g.search("active")
    g.close()


def test_incremental_matches_full(tmp_path):
    root = make(tmp_path, {"a.py": PY, "b.py": OTHER,
                           "c.py": "from typing import cast\n\ndef f(x):\n    return cast(int, x) + helper()\n"})
    index_repo(root)
    (root / "a.py").write_text(PY + "\n# touched\n")
    index_repo(root)
    inc = edges(root)
    index_repo(root, force=True)
    assert inc == edges(root)


def test_communities_and_report(tmp_path):
    root = make(tmp_path, {"a.py": PY, "b.py": OTHER})
    index_repo(root)
    g = Graph(root)
    assert g.db.execute("SELECT COUNT(*) FROM symbols WHERE community>=0").fetchone()[0] > 0
    text = g.report()
    g.close()
    assert "## Key symbols" in text and "## Communities" in text
    assert (root / ".codegraph" / "GRAPH_REPORT.md").exists()
    html = (root / ".codegraph" / "graph.html").read_text()
    assert "const D=" in html and '"nodes"' in html


def _git(root, *a):
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                   check=True, capture_output=True)


def test_review_maps_diff_to_symbols(tmp_path):
    root = make(tmp_path, {"a.py": PY, "tests/test_a.py": "from a import helper\n\ndef test_helper():\n    assert helper()\n"})
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    (root / "a.py").write_text(PY.replace("return 1", "return 11"))
    g = Graph(root)
    index_repo(root)
    out = g.review()
    g.close()
    assert "1 symbol(s) changed" in out
    assert "helper" in out and "dependents" in out


def test_review_clean_tree(tmp_path):
    root = make(tmp_path, {"a.py": PY})
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    g = Graph(root)
    assert g.review().startswith("No changes")
    g.close()


def test_install_is_idempotent_and_reversible(tmp_path):
    from codegraph.install import install, uninstall

    root = make(tmp_path, {"AGENTS.md": "# Mine\nkeep this\n",
                           ".mcp.json": json.dumps({"mcpServers": {"other": {"command": "x"}}})})
    install(root)
    first = {p.name: p.read_text() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}
    install(root)
    assert first == {p.name: p.read_text() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}
    servers = json.loads((root / ".mcp.json").read_text())["mcpServers"]
    assert set(servers) == {"other", "codegraph"}
    assert "keep this" in (root / "AGENTS.md").read_text()
    assert "codegraph" in json.loads((root / ".claude/settings.json").read_text())["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    uninstall(root)
    assert (root / "AGENTS.md").read_text().strip() == "# Mine\nkeep this"
    assert set(json.loads((root / ".mcp.json").read_text())["mcpServers"]) == {"other"}
    assert not (root / "GEMINI.md").exists() and not (root / "CLAUDE.md").exists()


def test_install_rejects_broken_json(tmp_path):
    from codegraph.install import install

    root = make(tmp_path, {".mcp.json": "{not json"})
    with pytest.raises(SystemExit):
        install(root, "claude")
    assert (root / ".mcp.json").read_text() == "{not json"   # never overwritten


def test_git_hooks_append_and_remove(tmp_path):
    from codegraph.install import hook_install, hook_uninstall

    root = make(tmp_path, {})
    _git(root, "init", "-q")
    hook = root / ".git" / "hooks" / "post-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\necho mine\n")
    hook_install(root)
    hook_install(root)
    text = hook.read_text()
    assert text.count("# >>> codegraph >>>") == 1 and "echo mine" in text
    assert (root / ".git" / "hooks" / "post-merge").exists()
    hook_uninstall(root)
    assert hook.read_text().strip() == "#!/bin/sh\necho mine"
    assert not (root / ".git" / "hooks" / "post-merge").exists()
