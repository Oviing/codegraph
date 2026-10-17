"""MCP server exposing the code graph to LLM agents."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .indexer import index_repo
from .queries import Graph
from .tools import TOOLS

try:  # mcp 1.x compatibility: FastMCP is the same object under its old name
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - mcp 2.x
    FastMCP = MCPServer

INSTRUCTIONS = (
    "Code graph for this repository. Prefer these tools over reading whole files: "
    "start with repo_map, find code with search, then call context(symbol) to get a symbol's source, "
    "the signatures it uses and who uses it in one compact response. Use impact(symbol) before changing "
    "a function, and similar(symbol) to find code that does the same thing under different names. "
    "Fall back to reading files only for exact surrounding code."
)


def _log(msg: str):
    """stderr only: stdout is the MCP transport."""
    print(f"codegraph: {msg}", file=sys.stderr, flush=True)


def build(root: str | Path, index: bool = True) -> MCPServer:
    root = Path(root).resolve()
    mcp = MCPServer("codegraph", instructions=INSTRUCTIONS)
    if index:
        _log(f"indexing {root} ...")
        stats = index_repo(root, progress=lambda m: _log(m))
        _log(f"indexed {stats['files']} files, {stats['symbols']} symbols, {stats['edges']} edges")
    graph = Graph(root)  # one long-lived graph: adjacency lists stay in memory

    for tool in TOOLS:
        _register(mcp, tool, graph, root)

    return mcp


def _register(mcp, tool, graph: Graph, root: Path):
    """Expose one Tool spec. The signature is generated from the spec, annotations included,
    so the advertised schema cannot drift from what the tool actually accepts."""
    if tool.name == "reindex":
        def reindex(force: bool = False) -> str:
            r = index_repo(root, force)
            graph.reload()
            return (f"{r['files']} files, {r['changed']} re-parsed, "
                    f"{r['symbols']} symbols, {r['edges']} edges")
        reindex.__doc__ = tool.summary
        mcp.tool(name=tool.name)(reindex)
        return

    def dispatch(**kwargs):
        return getattr(graph, tool.name)(**kwargs)

    params = tool.params
    args = []
    for p in params:
        args.append(f"{p.name}: {p.type}" if p.default is None
                    else f"{p.name}: {p.type} = {p.default!r}")
    # keyword forwarding: the MCP runtime calls tools with keyword arguments
    src = (f"def _{tool.name}({', '.join(args)}) -> str:\n"
           f"    return _dispatch({', '.join(f'{n}={n}' for n in [p.name for p in params])})")
    ns: dict = {"_dispatch": dispatch}
    exec(compile(src, f"<tool {tool.name}>", "exec"), ns)
    fn = ns[f"_{tool.name}"]
    fn.__name__ = tool.name  # the MCP tool name must not inherit the underscore
    fn.__doc__ = tool.summary
    mcp.tool(name=tool.name)(fn)


def main(root: str | None = None):
    build(root or os.environ.get("CODEGRAPH_ROOT") or os.getcwd()).run()
