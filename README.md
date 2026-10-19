# codegraph

Graph-based code understanding for LLM agents. Parses a repo once with tree-sitter (no LLM, no cost),
stores symbols and relations in SQLite, and serves compact, token-budgeted answers over MCP.

Instead of reading whole files, the agent asks the graph.

## Install

    pip install codegraph-network              # provides the `codegraph` command
    pip install "codegraph-network[embed]"     # optional: semantic search, see "Hybrid search"

From a checkout:

    pip install -e ".[dev]"     # drop [dev] if you don't need pytest/ruff
    pip install -e ".[embed]"   # optional: semantic search (fastembed + numpy), see "Hybrid search"

## Use with Claude Code

    claude mcp add codegraph -- codegraph -r /path/to/repo serve

Or in any MCP client config:

    {"mcpServers": {"codegraph": {"command": "codegraph", "args": ["-r", "/path/to/repo", "serve"]}}}

The index lives in `<repo>/.codegraph/graph.db` and updates incrementally on every server start;
`reindex` re-scans after edits made outside the session. `.gitignore` it.

## Explore visually

    codegraph -r /path/to/repo ui

Opens an interactive map of the repo in your browser (served from `127.0.0.1` only, nothing leaves your machine).
Everything is computed from the existing index with the standard library; no extra dependencies.

**Start from communities, zoom to code.** The first view is the community map: groups of symbols that are tightly
linked to each other, found with Louvain modularity clustering (the *clusters* slider merges or splits them). Double-click
a community to open it into files, a file into its symbols, a symbol to pull in what it calls and what calls it.
Drag nodes to rearrange and pin them; scroll to zoom. Click a community for cohesion, key symbols and its most
linked neighbours. The legend isolates a community with one click.

**Lenses** colour the graph by a graph-theory metric:

| Lens | Algorithm | Shows |
|---|---|---|
| importance | PageRank | what many others lean on |
| bottlenecks | betweenness centrality (sampled on big graphs) | symbols most paths run through |
| cycles | strongly connected components | circular dependencies between symbols and files |
| layers | condensation + longest path | callers on top, callees below |
| coupling / misplaced | cross-community link ratio | symbols that belong with another community |
| core vs periphery | k-core decomposition | the dense kernel vs leaf code |
| dead code | zero in-degree + no unresolved references | candidates nothing calls |
| hotspots | git co-change x importance | central code that also changes a lot |

**Layouts:** force, grouped by community, layered (callers on top), radial around the selection.
**Filters:** minimum link weight, certain-only links, hops from the selection, tests, kind, folder.
**Tools:** *Show impact* (transitive callers), *Path* between two symbols (shortest call chain), *Surprises*
(certain links between different communities and folders), *Save / Compare snapshot* (which symbols changed
community since you saved), and export to PNG, SVG or JSON. Press `/` to search.

## Tools

| Tool | Use it to |
|---|---|
| `repo_map` | PageRank-ranked overview within a token budget; `focus=` personalizes it to what you're working on |
| `search` | Find symbols by name, words (camel/snake split) or docstring; with the embed extra also by meaning, each hit tagged `[lexical]` / `[semantic]` / `[both]` |
| `context` | Best first call: source + signatures it uses + who uses it, in one reply |
| `source` | Exact source of one symbol |
| `outline` | Symbol tree of a file, without code |
| `callers` / `callees` | Walk the call/inherit graph (depth N) |
| `impact` | Blast radius of a change, grouped by file |
| `related` | Closest symbols to one symbol (direct + multi-hop), ranked by personalized PageRank; each line says how it is related (`uses`, `used by`, `tested by`, `tests`, `key`, `similar`) |
| `similar` | Symbols that mean the same thing as this one even with no call between them (needs the embed extra) |
| `file_deps` | What a file imports and what imports it |
| `stats` | Index size, edges by confidence, communities |
| `report` | Key symbols, communities, surprising connections; writes `.codegraph/GRAPH_REPORT.md` and a browsable `graph.html` |
| `review` (`prs`) | What the git diff touches: blast radius per changed symbol, missing tests, files that usually change with it |
| `reindex` | Re-scan after big external edits |

The MCP server and the CLI are both generated from one tool registry (`src/codegraph/tools.py`), so
they cannot drift apart. The CLI mirrors the tools: `codegraph map`, `codegraph context Store.resolve`,
`codegraph impact parse`, ...

## Languages

Python, TypeScript/TSX, JavaScript, Go, Rust, C/C++.

Beyond code, also parsed statically (no LLM): Markdown (headings become `section` symbols; `` `code spans` `` link
to the symbols they name, relative links count as imports), SQL (`CREATE TABLE/VIEW/FUNCTION/PROCEDURE`, with
`FROM`/`JOIN`/`REFERENCES` links), and PDF pages (`pip install -e ".[pdf]"`). Prose and SQL mentions link only
when the name is unambiguous.

## Edge confidence

Every edge records how it was resolved: `extracted` (one target named by the source itself: same file, an import,
`self.x()`), `inferred` (a name match by proximity) or `ambiguous` (several equally plausible targets). `callers`,
`callees` and `context` tag anything that is not `extracted`; `impact` marks it with `?`. Treat tagged edges as
upper bounds.

## Scripting and ignoring

`codegraph <tool>` exits 0 on an answer, 3 when a name matches several symbols, 4 when it matches none.
`.codegraphignore` (gitignore-style: `#` comments, `dir/`, `/anchored`, `*.glob`, `!negation`) excludes paths
from the index, on top of the built-in skips (`node_modules`, `venv`, `dist`, hidden directories, ...).

## Installing into assistants and git

    codegraph install [claude|cursor|gemini|agents|all]   # skill, MCP config, guidance block; --uninstall to undo
    codegraph hook install                                # re-index after commit, checkout, merge, rebase

`install` is idempotent and merges into existing files (guidance lives between `<!-- codegraph:start/end -->`
markers; JSON configs keep their other entries). For Claude Code it also adds a `Glob|Grep` hint hook
(`--no-hooks` to skip). Git hooks are appended to existing ones and run the indexer in the background.

## How it works

Indexing is incremental in two layers. Files are re-parsed only when their hash changes, and edge
resolution is scoped: a change re-resolves only the raw references whose name, or whose source file,
is affected. Parsing runs across processes and writes are batched; PageRank and community detection
warm-start from the previous run. On a 3,091-file / 59k-symbol / 79k-edge corpus (8 cores) a full index takes
~7.6s (was ~22s single-process), changing one file ~2s, and a no-op reindex ~0.3s. PageRank is the one step
that always walks the whole graph; whether it converged is reported by `codegraph index` rather than assumed.

Edge resolution is name-based, using the strongest evidence available:

- `self.x()` / `this.x()` link only to methods of the enclosing class, its bases and its subclasses
  (no match => inherited from an external base => no edge).
- `obj.x()` links only if a candidate is in the same file, an imported file or the same directory.
- bare `x()` prefers same file, then imported files, then same directory, then anywhere.
- a bare call to a Python builtin, or to a name imported from a module outside the repo
  (`from typing import cast`), links to nothing — otherwise `isinstance(...)` would attach thousands
  of phantom edges to whatever repo method happens to share the name.
- Targets defined in more than 4 places are dropped as too ambiguous.

## Hybrid search and hidden links

Everything above is exact and structural. A second layer adds *recall* for relationships that are not
spelled out in code. It never touches `callers` / `callees` / `impact` or global PageRank: those stay
structural. Soft links live in their own tables (`links`, `file_links`) and enter the personalized
PageRank behind `context` / `related` / `repo_map(focus=)` at a lower weight than a call edge
(`LINK_WEIGHT` in `queries.py`).

| Link | Source | Needs |
|---|---|---|
| `tests` | `test_foo.py` / `foo.test.ts` / `foo_test.go` -> the file named after it, and `test_bar` -> `bar` inside that pair; test callers are labelled `tested by` | nothing |
| `key` | the same string literal used in matching roles in different files: route path (`@app.get("/x")` vs `requests.get("/x")`, express vs `fetch`), event name (`emit` vs `on`), config key (`getenv`, `config.get`). Table-driven: add a row to `PATTERNS` in `keys.py`. Keys shared by more than 8 symbols and generic events (`error`, `data`, ...) are ignored | nothing |
| co-change | `git log --name-only` (last 1000 commits, commits over 30 files ignored, pairs need 3+ co-changes); shown by `deps` and used by `repo_map`'s ranking. No git or no history: no edges | git |
| `similar` | top-5 nearest neighbours by embedding cosine >= 0.85, skipping pairs already linked by a call/inheritance, overloads, and test-to-test pairs | embed extra |

**Semantic search** is optional: `pip install codegraph[embed]` (fastembed + numpy). Each function / method /
class / type is embedded from its name as words, qualified name, signature and doc line with a small local
ONNX model (`BAAI/bge-small-en-v1.5`); there is no network access at query time, the model files are
downloaded once at the first index. Vectors are cached in SQLite by content hash, so only changed symbols are
ever re-embedded and a no-change reindex stays near-instant. `search` fuses FTS5 and vector rankings with
reciprocal rank fusion (semantic hits in test files count half). Without the extra, or with
`CODEGRAPH_EMBED=off`, nothing changes: the output is the plain lexical search, byte for byte.

Providers are pluggable: `CODEGRAPH_EMBED=package.module:factory` loads any object with a `name`,
`embed_passages(list[str])` and `embed_query(str)` returning L2-normalised float32 vectors (see
`tests/fake_embed.py`); `CODEGRAPH_EMBED_MODEL` picks another fastembed model.

### Measured (fastembed bge-small, Apple silicon, CPU)

| | Flask (Python) | ky (TypeScript) |
|---|---|---|
| files / symbols | 88 / 1,723 | 100 / 11,427 |
| cold index, no embeddings | 0.4 s | 0.9 s |
| first-time embedding (one-off, then cached) | 27 s / 1,624 symbols | 58 s / 7,389 symbols |
| no-change reindex | 0.06 s | 0.04 s |
| edges (structural) | 1,011 calls, 74 inherits | 2,915 calls, 8 inherits |
| soft links | 70 tests, 52 key, 353 similar, 39 co-change | 3 key, 1,261 similar, 40 co-change |

`codegraph bench tasks.yaml` replays your own `(task, expected symbols)` list and reports, per task and in
total, recall@k of lexical-only vs hybrid search and the tokens an agent reads through the graph vs reading
the files that contain the expected symbols (`pip install codegraph[bench]` for YAML; JSON works without).
On 7 Flask and 5 ky natural-language tasks written for this README: hybrid recall 8/9 and 3/5, lexical-only
1/9 and 1/5. The lexical baseline is the existing search, which needs every query word to match, so it
scores ~0 on questions phrased unlike the code by construction. Treat the gap as "what the semantic layer adds
for descriptive queries", not as a general accuracy figure.

The similarity threshold (0.85) was set on codegraph's own code: bge-small cosines are high in general
(median nearest-neighbour 0.80 here); real near-duplicates (`make_repo` in two test files, `pagerank` vs
`personalized_pagerank`) start around 0.85, and below it the pairs are mostly generically similar test names.

## Tests

    pytest              # 232 tests
    pytest tests/test_edges.py -rA      # per-language edge precision/recall report

`tests/fixtures/<lang>/expected.json` holds hand-verified edges per language. Each fixture lists
`true` edges (must all be found) and `known_ambiguous` ones (accepted false positives from not
tracking receiver types, each with a stated reason). Anything the tool emits that is in neither list
fails the test, so a new kind of false positive cannot slip in unnoticed.

## Known limits

- Remaining false positives: `obj.x()` where `obj`'s type is unknown and an imported file defines a
  method called `x` (e.g. `Path(p).resolve()` vs your own `resolve`). Type-aware indexers (SCIP) fix this.
- Dynamic dispatch, reflection and macros are mostly invisible.
- Semantic search is as good as a small local model: it can miss (`jsonify` for "turn python objects into a
  json response" in Flask) and it ranks by meaning of the name/signature/doc line, not of the body. Variables
  and TypeScript interface fields are not embedded.
- String keys match on identical literals only (no `/users/{id}` vs `/users/1`), and only for Python and
  JS/TS call shapes listed in `keys.py`. `os.environ["X"]` and `process.env.X` are not recognised yet.
- Co-change uses the last 1000 commits and treats renames as new files.
- Object-literal properties in JS/TS (`{ json: body }`) are indexed as methods by the parser, which inflates
  TypeScript indexes (ky: 11k symbols for 100 files) and puts noise in semantic results.
- Go/Rust/C++ import resolution is heuristic. Notably a stdlib import can collide with a local
  directory of the same name: `net/http` matches an `internal/http` directory, because there is no
  `go.mod` to tell the module prefix apart. Python imports are exact (see `py_lookup` in `store.py`).

## Next steps

- Type-aware resolution via SCIP/LSIF indexers for exact call edges
- Optional one-line LLM summaries per symbol, generated once at index time
- Git-aware ranking by recency (co-change is in; "recently changed" is not)