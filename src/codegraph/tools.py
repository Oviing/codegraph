"""One definition of the codegraph tool surface, shared by the MCP server and the CLI.

Keeping the signature, the argument spec and the handler in one place is the only way the two
front ends cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "str" | "int" | "bool"
    default: object = None
    help: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    summary: str  # what the agent should use it for
    params: tuple[Param, ...] = field(default_factory=tuple)

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def usage(self) -> str:
        parts = [p.name if p.default is None else f"[{p.name}]" for p in self.params]
        return f"codegraph {self.name} " + " ".join(parts)


def _p(name, type_, default=None, help_=""):
    return Param(name, type_, default, help_)


TOOLS: tuple[Tool, ...] = (
    Tool("repo_map", "Ranked overview of the repo. Start here instead of reading whole files.",
         (_p("prefix", "str", "", "directory prefix to zoom into"),
          _p("max_tokens", "int", 1500),
          _p("focus", "str", "", "symbols, words or paths, comma separated"))),
    Tool("search", "Find symbols by name, split words or docstring; by meaning too when the embed extra is installed.",
         (_p("query", "str", None), _p("limit", "int", 10))),
    Tool("outline", "Symbol tree of a file, without any code.",
         (_p("file", "str", None),)),
    Tool("context", "Best first call for a symbol: source, what it uses, who uses it.",
         (_p("symbol", "str", None), _p("max_tokens", "int", 1500))),
    Tool("related", "Closest symbols to one symbol, direct and multi-hop, ranked.",
         (_p("symbol", "str", None), _p("max_tokens", "int", 800))),
    Tool("similar", "Symbols that mean the same thing as this one, even with no call between them "
                    "(needs the embed extra).",
         (_p("symbol", "str", None), _p("limit", "int", 10))),
    Tool("source", "Exact source of one symbol.",
         (_p("symbol", "str", None),)),
    Tool("callers", "Who calls / uses / inherits from this symbol.",
         (_p("symbol", "str", None), _p("depth", "int", 1), _p("limit", "int", 30))),
    Tool("callees", "What this symbol calls.",
         (_p("symbol", "str", None), _p("depth", "int", 1), _p("limit", "int", 30))),
    Tool("impact", "Blast radius of a change, grouped by file. Use before editing.",
         (_p("symbol", "str", None), _p("depth", "int", 3))),
    Tool("deps", "What a file imports and which files import it.",
         (_p("file", "str", None),)),
    Tool("stats", "Index size: files, symbols, edges by confidence, imports, communities."),
    Tool("report", "Whole-repo overview: key symbols, communities, surprising connections. "
                   "Also writes .codegraph/GRAPH_REPORT.md and a browsable graph.html.",
         (_p("max_tokens", "int", 1500),)),
    Tool("review", "Review a change: symbols the git diff touches, their blast radius, missing tests, "
                   "files that usually change with it.",
         (_p("base", "str", "", "branch/commit to diff against (default: where this branch left main)"),
          _p("max_tokens", "int", 1500))),
    Tool("remember", "Record WHY: a decision (with reasons and rejected alternatives), a failed attempt, an "
                     "invariant or unfinished intent, anchored to the symbols/files in `about`. "
                     "kind: decision | failed_attempt | invariant | intent.",
         (_p("text", "str", None, "what to remember"),
          _p("about", "str", "", "symbols or files it is about, comma separated"),
          _p("kind", "str", "decision", "decision | failed_attempt | invariant | intent"),
          _p("why", "str", "", "the reasoning"),
          _p("alternatives", "str", "", "what was considered and rejected, and why"),
          _p("title", "str", "", "short title (default: first line of text)"))),
    Tool("recall", "Memories (decisions, failed attempts, invariants) about a symbol/file and its neighbours, "
                   "or matching words. Check before changing code. Stale = the code changed since.",
         (_p("query", "str", "", "words to search for"),
          _p("about", "str", "", "symbols or files, comma separated"),
          _p("kind", "str", "", "only this kind"),
          _p("limit", "int", 10))),
    Tool("forget", "Retire a memory that no longer holds, or mark it superseded by a newer one. "
                   "The file is kept as history.",
         (_p("id", "str", None, "memory id (unique prefix is enough)"),
          _p("superseded_by", "str", "", "id of the memory that replaces it"))),
    Tool("reindex", "Re-scan the repo incrementally. Use after large external edits.",
         (_p("force", "bool", False),)),
)

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# extra CLI spellings, e.g. `codegraph map` for `repo_map`
CLI_ALIASES = {"repo_map": ("map",), "review": ("prs", "pr")}
QUERY_TOOLS = tuple(t for t in TOOLS if t.name not in ("reindex",))

# CLI flag aliases: param name -> (long flag, short flag or None).
# Anything not listed here is exposed under its own parameter name.
CLI_FLAGS = {
    "max_tokens": ("tokens", "-t"),
    "focus": ("focus", "-f"),
    "depth": ("depth", "-d"),
}


def make_callable(tool: Tool, graph, reindex_fn=None):
    """Bind a Tool spec to a Graph, producing the callable used by both front ends."""
    def call(**kwargs):
        return getattr(graph, tool.name)(**kwargs)
    call.__name__ = tool.name
    call.__doc__ = tool.summary
    if tool.name == "reindex":
        assert reindex_fn is not None
        call = reindex_fn
    return call
