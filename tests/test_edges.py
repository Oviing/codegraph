"""Edge-resolution quality: every edge the tool emits must be accounted for.

`true` edges are hand-verified real call/inherit relations and must all be found.
`known_ambiguous` edges are accepted false positives caused by not tracking receiver types;
each one must be listed explicitly, so a *new* kind of false positive fails the test.
"""
import tempfile
from pathlib import Path

import pytest

from codegraph.queries import Graph

from .conftest import LANGUAGES, build_graph, edge_set, import_set, load_expected


@pytest.fixture(scope="module", params=LANGUAGES)
def lang(request):
    return request.param


@pytest.fixture(scope="module")
def graph(lang):
    from .conftest import FIXTURES
    g = build_graph(FIXTURES / lang)
    yield g
    g.close()


def test_no_unexpected_edges(graph, lang):
    expected = load_expected(lang)
    documented = {tuple(e) for e in expected["true"]} | {tuple(e) for e in expected["known_ambiguous"]}
    found = edge_set(graph)
    unexpected = found - documented
    assert not unexpected, (
        f"{lang}: {len(unexpected)} edge(s) not documented in expected.json:\n"
        + "\n".join("  " + " -> ".join(e) for e in sorted(unexpected))
        + "\nEither fix the resolver or add them to known_ambiguous with a reason."
    )


def test_all_true_edges_found(graph, lang):
    expected = load_expected(lang)
    found = edge_set(graph)
    missing = {tuple(e) for e in expected["true"]} - found
    assert not missing, (
        f"{lang}: missed {len(missing)} true edge(s):\n"
        + "\n".join("  " + " -> ".join(e) for e in sorted(missing))
    )


def test_ambiguous_edges_are_justified(lang):
    expected = load_expected(lang)
    if expected["known_ambiguous"]:
        assert expected["why_ambiguous"].strip(), f"{lang}: known_ambiguous needs a 'why_ambiguous'"


def test_imports(graph, lang):
    assert import_set(graph) == {tuple(i) for i in load_expected(lang)["imports"]}


def test_precision_and_recall(graph, lang):
    expected = load_expected(lang)
    found = edge_set(graph)
    true = {tuple(e) for e in expected["true"]}
    recall = len(true & found) / len(true) if true else 1.0
    documented = true | {tuple(e) for e in expected["known_ambiguous"]}
    precision = len(found & documented) / len(found) if found else 1.0
    assert recall == 1.0, f"{lang}: recall {recall:.2f}"
    assert precision == 1.0, f"{lang}: precision {precision:.2f}"


def test_bare_builtin_calls_are_not_linked(tmp_path):
    """`isinstance(x, T)` must not become an edge to a repo method named `isinstance`.

    Such a method would otherwise collect thousands of phantom in-edges and dominate the global
    PageRank that `repo_map` ranks files by.
    """
    import shutil

    from codegraph.indexer import index_repo

    root = Path(tempfile.mkdtemp(prefix="codegraph-builtin-"))
    (root / "a.py").write_text(
        "class Util:\n"
        "    def isinstance(self, x):\n"
        "        return True\n"
        "\n"
        "\n"
        "def uses(v):\n"
        "    return isinstance(v, int)\n"
    )
    (root / "b.py").write_text(
        "def many(v):\n" + "    if isinstance(v, int):\n        pass\n" * 20
    )
    try:
        index_repo(root)
        g = Graph(root)
        try:
            edges = edge_set(g)
            assert not any(e[2] == "calls" and "isinstance" in e[1] for e in edges), edges
        finally:
            g.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_name_imported_from_an_external_module_is_not_linked(tmp_path):
    """`from typing import cast` must not bind a bare `cast(...)` to a repo method of that name."""
    import shutil

    from codegraph.indexer import index_repo

    root = Path(tempfile.mkdtemp(prefix="codegraph-external-"))
    (root / "thing.py").write_text("class Thing:\n    def cast(self):\n        return 1\n")
    (root / "user.py").write_text("from typing import cast\n\n\ndef go(v):\n    return cast(int, v)\n")
    try:
        index_repo(root)
        g = Graph(root)
        try:
            assert not any("Thing.cast" in e[1] for e in edge_set(g)), edge_set(g)
        finally:
            g.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_shadowed_builtin_in_the_same_file_still_links(tmp_path):
    """Suppressing builtins must not break code that defines and calls one locally."""
    import shutil

    from codegraph.indexer import index_repo

    root = Path(tempfile.mkdtemp(prefix="codegraph-shadow-"))
    (root / "m.py").write_text("def open(path):\n    return path\n\n\ndef use(p):\n    return open(p)\n")
    try:
        index_repo(root)
        g = Graph(root)
        try:
            assert ("m.py:use", "m.py:open", "calls") in edge_set(g)
        finally:
            g.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)
