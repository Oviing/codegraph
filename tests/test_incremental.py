"""Incremental indexing must produce exactly the same graph as a full rebuild.

This is the highest-risk part of the design: partial edge re-resolution is only correct if the
dirty set is a superset of what actually changed.
"""
import shutil
import tempfile
from pathlib import Path

import pytest

from codegraph.indexer import index_repo
from codegraph.queries import Graph

BASE = {
    "core.py": "def alpha():\n    return 1\n\n\ndef beta():\n    return alpha()\n",
    "app.py": "from core import alpha\n\n\ndef gamma():\n    return alpha()\n",
    "orphan.py": "def delta():\n    return 2\n",
    "cls.py": "class Base:\n    def run(self):\n        return alpha()\n\n\nclass Kid(Base):\n    def run(self):\n        return self.run2()\n\n    def run2(self):\n        return alpha()\n",
}


def edges(root: Path) -> set[tuple[str, str, str]]:
    g = Graph(root)
    rows = g.db.execute(
        "SELECT s.file || ':' || s.qualname, d.file || ':' || d.qualname, e.kind "
        "FROM edges e JOIN symbols s ON s.id = e.src JOIN symbols d ON d.id = e.dst").fetchall()
    imports = [tuple(r) for r in g.db.execute("SELECT file, dst FROM imports")]
    g.close()
    return {tuple(r) for r in rows} | {("import", a, b) for a, b in imports}


def make_repo(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="codegraph-inc-"))
    for name, body in files.items():
        (root / name).write_text(body)
    index_repo(root)
    return root


def rebuild_fresh(src: Path) -> set[tuple[str, str, str]]:
    """Copy the working tree without the index and index it from scratch."""
    dst = Path(tempfile.mkdtemp(prefix="codegraph-full-")) / "repo"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".codegraph"))
    index_repo(dst, force=True)
    return edges(dst)


MUTATIONS = {
    "edit a call target's body": ("core.py", "def alpha():\n    return 99\n\n\ndef beta():\n    return alpha()\n"),
    "remove a call": ("core.py", "def alpha():\n    return 1\n\n\ndef beta():\n    return 0\n"),
    "delete a definition other files call": ("core.py", "def beta():\n    return 1\n"),
    "add a competing definition": ("orphan.py", "def delta():\n    return alpha()\n"),
    "change imports so narrowing moves": ("app.py", "from core import beta\n\n\ndef gamma():\n    return beta()\n"),
    "drop all imports": ("app.py", "\n\ndef gamma():\n    return alpha()\n"),
    "change a class hierarchy": ("cls.py", "class Base:\n    def run(self):\n        return 0\n\n\nclass Kid(Base):\n    def run(self):\n        return alpha()\n\n    def run2(self):\n        return alpha()\n"),
}


@pytest.mark.parametrize("label", list(MUTATIONS))
def test_incremental_matches_full_rebuild(label):
    name, content = MUTATIONS[label]
    src = make_repo(BASE)
    (src / name).write_text(content)
    stats = index_repo(src)
    assert stats["changed"] >= 1
    assert edges(src) == rebuild_fresh(src), f"incremental graph differs after: {label}"


def test_unchanged_repo_does_no_work():
    src = make_repo(BASE)
    stats = index_repo(src)
    assert stats["changed"] == 0 and stats["removed"] == 0


def test_deleted_file_removes_its_symbols_and_edges():
    src = make_repo(BASE)
    before = edges(src)
    assert any("core.py:alpha" in e for e in before)
    (src / "core.py").unlink()
    index_repo(src)
    after = edges(src)
    assert not any("core.py:alpha" in str(e) for e in after)
    assert after == rebuild_fresh(src)


def test_deleting_a_target_file_clears_dangling_edges():
    src = make_repo(BASE)
    (src / "core.py").unlink()
    index_repo(src)
    g = Graph(src)
    dangling = g.db.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN symbols s ON s.id = e.dst WHERE s.id IS NULL"
    ).fetchone()[0]
    dangling_src = g.db.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN symbols s ON s.id = e.src WHERE s.id IS NULL"
    ).fetchone()[0]
    g.close()
    assert dangling == 0 and dangling_src == 0


def test_renaming_a_file_keeps_the_graph_consistent():
    src = make_repo(BASE)
    (src / "orphan.py").rename(src / "orphan2.py")
    index_repo(src)
    assert edges(src) == rebuild_fresh(src)


def test_parser_version_change_forces_a_full_reindex():
    src = make_repo(BASE)
    g = Graph(src)
    g.store.set_meta("parser_version", "0")
    g.store.commit()
    g.close()
    stats = index_repo(src)
    assert stats["changed"] == len(BASE)


def test_reindex_is_idempotent():
    src = make_repo(BASE)
    first = edges(src)
    index_repo(src, force=True)
    assert edges(src) == first == rebuild_fresh(src)
