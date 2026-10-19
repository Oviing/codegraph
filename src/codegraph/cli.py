"""Command line front end. Argument parsing is derived from the shared tool registry."""
from __future__ import annotations

import argparse
import json
import sys

from .indexer import index_repo
from .install import PLATFORMS, hook_install, hook_uninstall, install, uninstall
from .queries import Graph
from .tools import CLI_ALIASES, CLI_FLAGS, TOOLS

# 0 answered, 3 the name matches several symbols, 4 it matches none: scripts can branch without parsing text
EXIT_CODES = {"ok": 0, "ambiguous": 3, "missing": 4}

# the reindex tool is presented under a friendlier name
CLI_NAME = {"reindex": "index"}


def _cli_name(tool_name: str) -> str:
    return CLI_NAME.get(tool_name, tool_name)


# every spelling a tool answers to on the command line
CLI_SPELLINGS = {t.name: (_cli_name(t.name), *CLI_ALIASES.get(t.name, ())) for t in TOOLS}


def _cli_name(tool_name: str) -> str:
    return CLI_NAME.get(tool_name, tool_name)


def _tool_for(spelling: str):
    for tool in TOOLS:
        if spelling in CLI_SPELLINGS[tool.name]:
            return tool
    raise KeyError(spelling)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="codegraph",
        description="Graph-based code understanding for LLMs",
        epilog="run 'codegraph <tool> --help' for the arguments of one tool")
    ap.add_argument("-r", "--root", default=".", help="repository root (default: cwd)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the MCP server over stdio")
    ui = sub.add_parser("ui", help="open an interactive graph explorer in the browser (local only)")
    ui.add_argument("--port", type=int, default=0, help="port to listen on (default: any free port)")
    ui.add_argument("--no-open", action="store_true", help="print the URL instead of opening a browser")
    bench = sub.add_parser("bench", help="measure tokens and recall (lexical vs hybrid) on a tasks file")
    bench.add_argument("tasks", help="YAML/JSON list of {task: ..., expect: [symbols]}")
    bench.add_argument("-k", type=int, default=10, help="results per search (default 10)")

    inst = sub.add_parser("install", help="wire codegraph into coding assistants (skill, MCP config, guidance)")
    inst.add_argument("platform", nargs="?", default="all", choices=(*PLATFORMS, "all"),
                      help="which assistant to configure (default: all)")
    inst.add_argument("--no-hooks", action="store_true", help="skip the Claude Code Glob|Grep hint hook")
    inst.add_argument("--uninstall", action="store_true", help="remove what install wrote")
    hk = sub.add_parser("hook", help="keep the index fresh from git: post-commit/checkout/merge hooks")
    hk.add_argument("action", choices=("install", "uninstall"))

    for tool in TOOLS:
        spellings = CLI_SPELLINGS[tool.name]
        p = sub.add_parser(spellings[0], aliases=spellings[1:],
                           help=tool.summary, description=tool.summary)
        for param in tool.params:
            if param.default is None:
                p.add_argument("value", help=param.help or param.name)
                continue
            flag, short = CLI_FLAGS.get(param.name, (param.name, None))
            opts = [f"--{flag}"] + ([short] if short else [])
            if param.type == "bool":
                p.add_argument(*opts, action="store_true", help=param.help)
            elif param.type == "int":
                p.add_argument(*opts, type=int, default=param.default, help=param.help)
            else:
                p.add_argument(*opts, default=param.default, help=param.help)
    return ap


def _kwargs(tool, args) -> dict:
    out = {}
    for param in tool.params:
        if param.default is None:
            out[param.name] = args.value
            continue
        flag = CLI_FLAGS.get(param.name, (param.name, None))[0]
        v = getattr(args, flag.replace("-", "_"))
        out[param.name] = int(v) if param.type == "int" and v is not None else v
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cmd == "serve":
        from .server import main as serve
        return serve(args.root)

    if args.cmd == "ui":
        from .ui import serve as ui_serve
        return ui_serve(args.root, args.port, not args.no_open)

    if args.cmd == "bench":
        from .bench import run as bench
        print(bench(args.root, args.tasks, args.k))
        return None

    if args.cmd in ("install", "hook"):
        if args.cmd == "hook":
            lines = (hook_install if args.action == "install" else hook_uninstall)(args.root)
        elif args.uninstall:
            lines = uninstall(args.root)
        else:
            lines = install(args.root, args.platform, hooks=not args.no_hooks)
        print("\n".join(lines))
        return None

    if args.cmd == "index":
        print(json.dumps(index_repo(args.root, args.force), indent=2, sort_keys=True))
        return None

    index_repo(args.root)  # cheap when nothing changed
    tool = _tool_for(args.cmd)
    graph = Graph(args.root)
    graph.status = "ok"
    try:
        print(getattr(graph, tool.name)(**_kwargs(tool, args)))
        return EXIT_CODES[graph.status]
    finally:
        graph.close()


if __name__ == "__main__":
    sys.exit(main())
