import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from codegraph.indexer import index_repo
from codegraph.queries import Graph

# the suite must not download or run a model just because the embed extra happens to be installed;
# tests that exercise the semantic layer opt in with their own provider
os.environ.setdefault("CODEGRAPH_EMBED", "off")

FIXTURES = Path(__file__).parent / "fixtures"
LANGUAGES = ("py", "ts", "go", "rust", "cpp")


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURES


def load_expected(lang: str) -> dict:
    return json.loads((FIXTURES / lang / "expected.json").read_text())


def build_graph(src: Path, force: bool = False) -> Graph:
    """Copy a fixture into a scratch directory, index it, and return a Graph over the copy.

    Copying keeps the fixtures themselves free of .codegraph directories.
    """
    tmp = Path(tempfile.mkdtemp(prefix=f"codegraph-{src.name}-"))
    dst = tmp / src.name
    shutil.copytree(src, dst)
    index_repo(dst, force)
    return Graph(dst)


def edge_set(graph: Graph) -> set[tuple[str, str, str]]:
    rows = graph.db.execute(
        "SELECT s.file || ':' || s.qualname, d.file || ':' || d.qualname, e.kind "
        "FROM edges e JOIN symbols s ON s.id = e.src JOIN symbols d ON d.id = e.dst"
    ).fetchall()
    return {tuple(r) for r in rows}


def import_set(graph: Graph) -> set[tuple[str, str]]:
    return {tuple(r) for r in graph.db.execute("SELECT file, dst FROM imports")}


def symbol_set(graph: Graph) -> set[tuple[str, str, str]]:
    rows = graph.db.execute("SELECT file, qualname, kind FROM symbols WHERE kind != 'module'").fetchall()
    return {tuple(r) for r in rows}


def make_repo(root: Path, files: dict[str, str]) -> Path:
    """Write `files` (path -> source) under `root` and return it."""
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


@pytest.fixture
def semantic(monkeypatch):
    """Switch the semantic layer on with the deterministic fake model (no download, no network)."""
    from tests.fake_embed import FakeEmbedder
    monkeypatch.setenv("CODEGRAPH_EMBED", "tests.fake_embed:FakeEmbedder")
    FakeEmbedder.calls = 0
    return FakeEmbedder


def link_set(graph: Graph, kind: str) -> set[tuple[str, str, str]]:
    rows = graph.db.execute(
        "SELECT s.file || ':' || s.qualname, d.file || ':' || d.qualname, l.note "
        "FROM links l JOIN symbols s ON s.id = l.src JOIN symbols d ON d.id = l.dst WHERE l.kind = ?",
        (kind,)).fetchall()
    return {tuple(r) for r in rows}
