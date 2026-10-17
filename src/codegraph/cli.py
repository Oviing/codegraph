from __future__ import annotations

import argparse

from .indexer import index_repo
from .queries import Graph


def main():
    ap = argparse.ArgumentParser(prog="codegraph", description="Graph-based code understanding for LLMs")
    ap.add_argument("-r", "--root", default=".", help="repository root (default: cwd)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="build/update the graph"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("map"); p.add_argument("prefix", nargs="?", default=""); p.add_argument("-t", "--tokens", type=int, default=1500); p.add_argument("-f", "--focus", default="")
    p = sub.add_parser("search"); p.add_argument("query")
    p = sub.add_parser("context"); p.add_argument("symbol"); p.add_argument("-t", "--tokens", type=int, default=1500)
    p = sub.add_parser("source"); p.add_argument("symbol")
    p = sub.add_parser("related"); p.add_argument("symbol"); p.add_argument("-t", "--tokens", type=int, default=800)
    p = sub.add_parser("outline"); p.add_argument("file")
    p = sub.add_parser("callers"); p.add_argument("symbol"); p.add_argument("-d", "--depth", type=int, default=1)
    p = sub.add_parser("callees"); p.add_argument("symbol"); p.add_argument("-d", "--depth", type=int, default=1)
    p = sub.add_parser("impact"); p.add_argument("symbol"); p.add_argument("-d", "--depth", type=int, default=3)
    p = sub.add_parser("deps"); p.add_argument("file")
    sub.add_parser("stats")
    sub.add_parser("serve", help="run the MCP server over stdio")
    a = ap.parse_args()

    if a.cmd == "serve":
        from .server import main as serve
        return serve(a.root)
    if a.cmd == "index":
        print(index_repo(a.root, a.force))
        return
    index_repo(a.root)  # cheap when nothing changed
    g = Graph(a.root)
    out = {
        "map": lambda: g.repo_map(a.prefix, a.tokens, a.focus),
        "related": lambda: g.related(a.symbol, a.tokens),
        "search": lambda: g.search(a.query),
        "context": lambda: g.context(a.symbol, a.tokens),
        "source": lambda: g.source(a.symbol),
        "outline": lambda: g.outline(a.file),
        "callers": lambda: g.callers(a.symbol, a.depth),
        "callees": lambda: g.callees(a.symbol, a.depth),
        "impact": lambda: g.impact(a.symbol, a.depth),
        "deps": lambda: g.deps(a.file),
        "stats": g.stats,
    }[a.cmd]()
    print(out)


if __name__ == "__main__":
    main()
