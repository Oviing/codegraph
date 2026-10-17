"""MCP server exposing the code graph to LLM agents."""
from __future__ import annotations

import os
from pathlib import Path

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from .indexer import index_repo
from .queries import Graph

INSTRUCTIONS = (
    "Code graph for this repository. Prefer these tools over reading whole files: "
    "start with repo_map, find code with search, then call context(symbol) to get a symbol's source, "
    "the signatures it uses and who uses it in one compact response. Use impact(symbol) before changing "
    "a function. Fall back to reading files only for exact surrounding code."
)


def build(root: str | Path) -> FastMCP:
    root = Path(root).resolve()
    mcp = FastMCP("codegraph", instructions=INSTRUCTIONS)
    index_repo(root)  # incremental: only changed files are re-parsed
    graph = Graph(root)  # one long-lived graph: adjacency lists stay in memory

    def g() -> Graph:
        return graph

    @mcp.tool()
    def repo_map(prefix: str = "", max_tokens: int = 1500, focus: str = "") -> str:
        """Ranked overview of the repo (PageRank). Pass focus="symbolA, some/file.py, auth flow" to rank by
        closeness to what you are working on. Use prefix to zoom into a directory."""
        return g().repo_map(prefix, max_tokens, focus)

    @mcp.tool()
    def search(query: str, limit: int = 10) -> str:
        """Find symbols by name/words/docstring (camelCase and snake_case are split). Returns file:lines and signature."""
        return g().search(query, limit)

    @mcp.tool()
    def outline(file: str) -> str:
        """Symbol tree of one file (path suffix is enough) without its source."""
        return g().outline(file)

    @mcp.tool()
    def context(symbol: str, max_tokens: int = 1500) -> str:
        """Best first call for understanding a symbol: its source, signatures of what it uses, and its callers."""
        return g().context(symbol, max_tokens)

    @mcp.tool()
    def related(symbol: str, max_tokens: int = 800) -> str:
        """Symbols most closely connected to this one (direct and multi-hop), ranked, within a token budget."""
        return g().related(symbol, max_tokens)

    @mcp.tool()
    def source(symbol: str) -> str:
        """Exact source of one symbol only."""
        return g().source(symbol)

    @mcp.tool()
    def callers(symbol: str, depth: int = 1) -> str:
        """Who calls/uses/inherits from this symbol (depth>1 walks further up)."""
        return g().callers(symbol, depth)

    @mcp.tool()
    def callees(symbol: str, depth: int = 1) -> str:
        """What this symbol calls (depth>1 walks further down)."""
        return g().callees(symbol, depth)

    @mcp.tool()
    def impact(symbol: str, depth: int = 3) -> str:
        """Blast radius of changing a symbol, grouped by file."""
        return g().impact(symbol, depth)

    @mcp.tool()
    def file_deps(file: str) -> str:
        """What a file imports and which files import it."""
        return g().deps(file)

    @mcp.tool()
    def reindex(force: bool = False) -> str:
        """Re-scan the repo (incremental). Call after large edits made outside this session."""
        r = index_repo(root, force)
        graph.reload()
        return f"{r['files']} files, {r['changed']} re-parsed, {r['symbols']} symbols, {r['edges']} edges"

    return mcp


def main(root: str | None = None):
    build(root or os.environ.get("CODEGRAPH_ROOT") or os.getcwd()).run()
