"""End-to-end checks against a real MCP stdio server.

Schema-only tests are not enough: the MCP runtime invokes tools with *keyword* arguments, which
a tool wrapper can get wrong while its advertised schema still looks perfect.
"""
import os
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


async def _session(root):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "codegraph.cli", "-r", str(root), "serve"],
        env={**os.environ, "PYTHONPATH": SRC},
    )
    return params


def _run(root, body):
    import asyncio

    async def go():
        params = await _session(root)
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            return await body(s)

    return asyncio.run(go())


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("mcp-repo")
    (root / "m.py").write_text(
        "def a():\n"
        "    return b()\n"
        "\n"
        "\n"
        "def b():\n"
        "    return 1\n"
    )
    return root


def test_server_initializes_with_instructions(repo):
    """The agent-facing instructions must actually reach the client at initialize time."""
    captured = {}

    async def body(s):
        return None

    async def go():
        params = await _session(repo)
        async with stdio_client(params) as (r, w), ClientSession(r, w) as sess:
            init = await sess.initialize()
            captured["name"] = init.server_info.name
            captured["instructions"] = init.instructions
            await sess.call_tool("stats", {})

    import asyncio
    asyncio.run(go())
    assert captured["name"] == "codegraph"
    assert "repo_map" in (captured["instructions"] or "")


def test_every_tool_can_actually_be_invoked(repo):
    """Call each tool with valid arguments; a wrapper bug shows up here and nowhere else."""
    calls = [
        ("repo_map", {"max_tokens": 100}),
        ("search", {"query": "a"}),
        ("outline", {"file": "m.py"}),
        ("context", {"symbol": "a", "max_tokens": 200}),
        ("related", {"symbol": "a"}),
        ("similar", {"symbol": "a"}),
        ("source", {"symbol": "a"}),
        ("callers", {"symbol": "b"}),
        ("callees", {"symbol": "a"}),
        ("impact", {"symbol": "b"}),
        ("deps", {"file": "m.py"}),
        ("stats", {}),
        ("reindex", {}),
    ]

    async def body(s):
        out = {}
        for name, args in calls:
            res = await s.call_tool(name, args)
            out[name] = res.content[0].text
        return out

    out = _run(repo, body)
    for name, _ in calls:
        assert "Error executing tool" not in out[name], f"{name} failed: {out[name]}"
        assert out[name].strip(), f"{name} returned nothing"
    assert "1 files" in out["stats"]


def test_unknown_symbol_is_reported_not_raised(repo):
    async def body(s):
        res = await s.call_tool("context", {"symbol": "definitely_not_here"})
        return res.content[0].text

    text = _run(repo, body)
    assert "No symbol matches" in text
    assert "Error executing tool" not in text


def test_keyword_invocation_is_used(repo):
    """Guard against regressing to positional forwarding in the generated wrappers."""
    async def body(s):
        return (await s.call_tool("context", {"symbol": "a", "max_tokens": 300})).content[0].text

    assert "def a" in _run(repo, body)


def test_tool_names_have_no_leading_underscore(repo):
    async def body(s):
        return sorted(t.name for t in (await s.list_tools()).tools)

    names = _run(repo, body)
    assert not [n for n in names if n.startswith("_")]
    assert "repo_map" in names and "reindex" in names
