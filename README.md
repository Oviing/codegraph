# codegraph

Graph-based code understanding for LLM agents. Parses a repo once with tree-sitter (no LLM, no cost),
stores symbols and relations in SQLite, and serves compact, token-budgeted answers over MCP.

Instead of reading whole files, the agent asks the graph.

## Install

    pip install -e .

## Use with Claude Code

    claude mcp add codegraph -- codegraph -r /path/to/repo serve

Or in any MCP client config:

    {"mcpServers": {"codegraph": {"command": "codegraph", "args": ["-r", "/path/to/repo", "serve"]}}}

The index lives in `<repo>/.codegraph/graph.db` (add it to .gitignore) and updates incrementally
(by file hash) every time the server starts, or when the agent calls `reindex`.

## Tools

| Tool | Use it to |
|---|---|
| `repo_map` | PageRank-ranked overview within a token budget; `focus=` personalizes it to what you're working on |
| `related` | Closest symbols to one symbol (direct + multi-hop), ranked by personalized PageRank |
| `search` | Find symbols by name, words (camel/snake split) or docstring |
| `context` | Best first call: source + signatures it uses + who uses it, in one reply |
| `source` | Exact source of one symbol |
| `outline` | Symbol tree of a file, without code |
| `callers` / `callees` | Walk the call/inherit graph (depth N) |
| `impact` | Blast radius of a change, grouped by file |
| `file_deps` | What a file imports and what imports it |
| `reindex` | Re-scan after big external edits |

CLI mirrors the tools: `codegraph map`, `codegraph context Store.resolve`, `codegraph impact parse`, ...

## Languages

Python, TypeScript/TSX, JavaScript, Go, Rust, C/C++. Adding one means adding entries to the tables
at the top of `parser.py` (definition nodes, call nodes, import nodes).

## Known limits (by design, for v0.1)

- Edges are resolved by name, not by type, using the strongest evidence available:
  - `self.x()` / `this.x()` link only to methods of the enclosing class, its bases and its subclasses
    (no match => inherited from an external base => no edge).
  - `obj.x()` links only if a candidate is in the same file, an imported file or the same directory.
  - bare `x()` prefers same file, then imported files, then same directory, then anywhere.
  - Targets defined in more than 4 places are dropped as too ambiguous.
- Remaining false positives: `obj.x()` where `obj`'s type is unknown and an imported file defines a
  method called `x` (e.g. `Path(p).resolve()` vs your own `resolve`). Type-aware indexers (SCIP) fix this.
- Dynamic dispatch, reflection and macros are invisible.
- Go/Rust/C++ import resolution is heuristic (directory/stem matching).

## Next steps

- Type-aware resolution via SCIP/LSIF indexers for exact call edges
- Optional one-line LLM summaries per symbol, generated once at index time
- Git-aware ranking (recently changed / frequently co-changed files)
- A `codegraph bench` command that replays real agent tasks with and without the graph
