"""`codegraph bench`: does the graph answer real tasks cheaper than reading files, and does semantic
search find symbols that lexical search misses?

Tasks file (YAML, or JSON which is a subset of it):

    - task: "where do we check a user's credentials on login"
      expect: [authenticate_user, auth.py:Session.open]

`expect` entries are anything `context()` accepts (name, qualname or file:name). Per task we measure
  recall      share of expected symbols inside the top-k results, lexical-only vs hybrid
  tokens      what an agent reads with the graph (the hybrid search output plus context() of every
              expected symbol it found) vs reading every file that contains an expected symbol
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .indexer import index_repo
from .queries import Graph, toks


@dataclass
class TaskResult:
    task: str
    expected: int
    unresolved: list[str] = field(default_factory=list)
    lexical_found: int = 0
    hybrid_found: int = 0
    graph_tokens: int = 0
    file_tokens: int = 0


def load_tasks(path: str | Path) -> list[dict]:
    text = Path(path).read_text()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        try:
            data = json.loads(text)
        except ValueError as e:
            raise SystemExit("bench: reading YAML needs PyYAML (pip install 'codegraph[bench]'); "
                             f"as JSON the file failed to parse: {e}") from None
    if not isinstance(data, list) or not all(isinstance(t, dict) and "task" in t for t in data):
        raise SystemExit("bench: the tasks file must be a list of {task: ..., expect: [symbols]}")
    return data


def run_task(g: Graph, spec: dict, k: int) -> TaskResult:
    expect = [str(e) for e in spec.get("expect") or []]
    res = TaskResult(str(spec["task"]), len(expect))
    wanted: dict[str, set[int]] = {}
    for e in expect:
        rows = g._find(e)
        if rows:
            wanted[e] = {r["id"] for r in rows}
        else:
            res.unresolved.append(e)
    lex_ids = {r["id"] for r in g._search_rows(res.task, k)}
    hyb = g._hybrid_rows(res.task, k)
    hyb_ids = {r["id"] for r, _ in hyb}
    res.lexical_found = sum(1 for ids in wanted.values() if ids & lex_ids)
    found = [e for e, ids in wanted.items() if ids & hyb_ids]
    res.hybrid_found = len(found)
    res.graph_tokens = toks(g.search(res.task, k)) + sum(toks(g.context(e)) for e in found)
    files = {r["file"] for ids in wanted.values() for r in g._rows_by_id(ids)}
    res.file_tokens = g._file_tokens(files)
    return res


def run(root: str | Path, tasks_path: str | Path, k: int = 10) -> str:
    index_repo(root)
    g = Graph(root)
    try:
        semantic = g._semantic() is not None
        results = [run_task(g, t, k) for t in load_tasks(tasks_path)]
    finally:
        g.close()
    out = [f"codegraph bench  top-{k}  semantic index: {'yes' if semantic else 'NO (hybrid == lexical)'}", ""]
    for r in results:
        out.append(f"- {r.task}")
        out.append(f"    recall   lexical {r.lexical_found}/{r.expected}   hybrid {r.hybrid_found}/{r.expected}"
                   f"   tokens graph ~{r.graph_tokens} vs files ~{r.file_tokens}")
        if r.unresolved:
            out.append(f"    not in index: {', '.join(r.unresolved)}")
    exp = sum(r.expected - len(r.unresolved) for r in results) or 1
    lex, hyb = sum(r.lexical_found for r in results), sum(r.hybrid_found for r in results)
    gt, ft = sum(r.graph_tokens for r in results), sum(r.file_tokens for r in results)
    out += ["", f"recall   lexical {lex}/{exp} ({lex / exp:.0%})   hybrid {hyb}/{exp} ({hyb / exp:.0%})",
            f"tokens   graph ~{gt} vs reading files ~{ft}" + (f"  ({ft / gt:.1f}x fewer)" if gt else "")]
    return "\n".join(out)
