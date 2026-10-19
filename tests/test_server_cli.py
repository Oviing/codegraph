"""The MCP server and the CLI must both be driven by the tool registry, with no drift."""
import shutil
import tempfile
from pathlib import Path

import pytest

from codegraph.indexer import index_repo
from codegraph.server import build
from codegraph.tools import TOOLS, TOOLS_BY_NAME, Param

FILES = {"m.py": "def a():\n    return b()\n\n\ndef b():\n    return 1\n"}


@pytest.fixture(scope="module")
def repo():
    root = Path(tempfile.mkdtemp(prefix="codegraph-srv-"))
    (root / "m.py").write_text(FILES["m.py"])
    index_repo(root)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_registry_has_no_duplicate_names():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))


def test_every_tool_matches_a_graph_method(repo):
    from codegraph.queries import Graph
    g = Graph(repo)
    try:
        for tool in TOOLS:
            if tool.name == "reindex":
                continue
            assert callable(getattr(g, tool.name)), tool.name
    finally:
        g.close()


def test_server_registers_every_tool(repo):
    mcp = build(repo, index=False)
    registered = {t.name for t in await_tools(mcp)}
    assert registered == {t.name for t in TOOLS}


def await_tools(mcp):
    import asyncio

    async def go():
        tools = await mcp.list_tools()
        return tools

    return asyncio.run(go())


def test_tool_schemas_carry_the_right_types(repo):
    import asyncio
    mcp = build(repo, index=False)

    async def go():
        return {t.name: t for t in await mcp.list_tools()}

    tools = asyncio.run(go())
    ctx = tools["context"]
    schema = getattr(ctx, "inputSchema", None) or ctx.input_schema
    props = schema["properties"]
    assert props["symbol"]["type"] == "string"
    assert props["max_tokens"]["type"] == "integer"
    assert schema["required"] == ["symbol"]


def test_tool_descriptions_come_from_the_registry(repo):
    import asyncio
    mcp = build(repo, index=False)
    tools = {t.name: t for t in asyncio.run(_list(mcp))}
    for tool in TOOLS:
        assert tools[tool.name].description == tool.summary


async def _list(mcp):
    return await mcp.list_tools()


def test_cli_exposes_every_tool():
    import argparse

    parser = _build_parser()
    subactions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert subactions, "no subcommands registered"
    names = set(subactions[0].choices)
    from codegraph.cli import CLI_SPELLINGS
    expected = {"serve", "ui", "bench", "install", "hook"} | {spelling for spellings in CLI_SPELLINGS.values() for spelling in spellings}
    assert names == expected
    # documented aliases must keep working
    assert {"map"} <= names


def test_cli_index_is_json(repo, capsys):
    import json
    import sys

    from codegraph.cli import main
    argv = sys.argv
    sys.argv = ["codegraph", "-r", str(repo), "index"]
    try:
        main()
    finally:
        sys.argv = argv
    out = json.loads(capsys.readouterr().out)
    assert out["files"] == 1 and "symbols" in out


def _build_parser():
    """Rebuild the CLI parser without running main()."""
    import argparse

    import codegraph.cli as cli
    captured = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    def fake(self, *a, **kw):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = fake
    try:
        cli.main()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse_args
    return captured["parser"]


def test_cli_flag_aliases_do_not_collide():
    from codegraph.tools import CLI_FLAGS
    longs = [v[0] for v in CLI_FLAGS.values()]
    shorts = [v[1] for v in CLI_FLAGS.values() if v[1]]
    assert len(longs) == len(set(longs))
    assert len(shorts) == len(set(shorts))


def test_tools_with_only_optional_params_run_with_defaults(repo):
    from codegraph.queries import Graph
    g = Graph(repo)
    try:
        checked = 0
        for tool in TOOLS:
            if tool.name == "reindex" or any(p.default is None for p in tool.params):
                continue  # has a required argument
            assert isinstance(getattr(g, tool.name)(**tool.defaults()), str)
            checked += 1
        assert checked >= 2
    finally:
        g.close()


def test_tools_with_required_params_refuse_a_missing_argument(repo):
    """The required argument is enforced by the signature, not by a runtime check."""
    from codegraph.queries import Graph
    g = Graph(repo)
    try:
        for tool in TOOLS:
            if not any(p.default is None for p in tool.params):
                continue
            with pytest.raises(TypeError):
                getattr(g, tool.name)()
    finally:
        g.close()


def test_param_helper_builds_the_right_type():
    p = Param("depth", "int", 3, "how deep")
    assert (p.name, p.type, p.default) == ("depth", "int", 3)


def test_tools_by_name_lookup():
    assert TOOLS_BY_NAME["impact"].name == "impact"
    assert TOOLS_BY_NAME["impact"].defaults() == {"symbol": None, "depth": 3}
