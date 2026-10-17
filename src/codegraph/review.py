"""`review` / `prs`: what does this change touch, and what depends on it?

Maps a git diff onto graph symbols, then reports each changed symbol's blast radius, whether any test is
linked to it, and files that usually change together with the diff but are missing from it.
"""
from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

from .links import is_test_path

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
MAIN_BRANCHES = ("origin/main", "origin/master", "main", "master")
COCHANGE_MIN = 0.3  # a partner file below this weight is coincidence, not a habit
MAX_SYMBOLS = 15


def _git(root: Path, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def default_base(root: Path) -> str:
    """The point the change started from: merge-base with the main branch, or HEAD (uncommitted work)
    when already on it."""
    for name in MAIN_BRANCHES:
        out = _git(root, "merge-base", "HEAD", name)
        if out and out.strip():
            base = out.strip()
            head = (_git(root, "rev-parse", "HEAD") or "").strip()
            return base if base != head else "HEAD"
    return "HEAD"


def parse_diff(diff: str) -> tuple[dict[str, list[tuple[int, int]]], list[str]]:
    """Unified diff (-U0) -> ({file: [(first, last) new-side line ranges]}, [deleted files])."""
    changed: dict[str, list[tuple[int, int]]] = defaultdict(list)
    deleted: list[str] = []
    old = cur = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            old = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
            if cur is None and old:
                deleted.append(old)
        elif cur and (m := HUNK.match(line)):
            start, n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            changed[cur].append((start, start + max(n, 1) - 1))  # n == 0: pure deletion at `start`
    return dict(changed), deleted


def _innermost(rows, ranges):
    """Symbols overlapping any range, minus containers whose own child also overlaps."""
    hit = [r for r in rows if any(r["line_start"] <= b and a <= r["line_end"] for a, b in ranges)]
    ids = {r["parent"] for r in hit}
    return [r for r in hit if r["id"] not in ids]


def review(graph, base: str = "", max_tokens: int = 1500) -> str:
    root = graph.root
    if _git(root, "rev-parse", "--git-dir") is None:
        return "Not a git repository: nothing to diff."
    base = base.strip() or default_base(root)
    diff = _git(root, "diff", "-U0", "--no-color", base, "--")
    if diff is None:
        return f"git could not diff against '{base}'. Pass a branch, tag or commit."
    changed, deleted = parse_diff(diff)
    if not changed and not deleted:
        return f"No changes against {base}."

    db = graph.db
    rows_by_file = defaultdict(list)
    for r in db.execute("SELECT * FROM symbols WHERE kind NOT IN ('module','impl','namespace')"):
        if r["file"] in changed:
            rows_by_file[r["file"]].append(r)
    touched, unindexed = [], []
    for f, ranges in sorted(changed.items()):
        rows = rows_by_file.get(f)
        if rows is None:
            unindexed.append(f)
            continue
        touched += _innermost(rows, ranges)

    report = []
    for r in touched:
        levels = graph._bfs(r, "in", 3, 10_000)
        deps = [n for ns in levels.values() for n in ns]
        report.append((r, len(deps), len({n["file"] for n in deps}),
                       sum(1 for n in deps if n["conf"] != "extracted"), _tests_for(graph, r["id"])))
    report.sort(key=lambda t: (-t[1], t[0]["qualname"]))

    files = sorted(changed)
    out = [f"review vs {base}: {len(files)} file(s), {len(touched)} symbol(s) changed"
           + (f", {len(deleted)} file(s) deleted" if deleted else "")]
    if report:
        out.append("highest blast radius first:")
    for r, n, nf, soft, tests in report[:MAX_SYMBOLS]:
        what = f"{n} dependents in {nf} file(s)" + (f", {soft} via inferred/ambiguous" if soft else "") if n else "no dependents"
        verdict = "" if is_test_path(r["file"]) else ("  tests: " + (", ".join(tests[:2]) if tests else "NONE linked"))
        out.append(f"  {r['qualname']}  {r['file']}:{r['line_start']}  -> {what}{verdict}")
    if len(report) > MAX_SYMBOLS:
        out.append(f"  ... and {len(report) - MAX_SYMBOLS} more")
    untested = [r["qualname"] for r, _, _, _, t in report if not t and not is_test_path(r["file"])]
    if untested:
        out.append("changed without a linked test: " + ", ".join(untested[:8])
                   + (f" (+{len(untested) - 8})" if len(untested) > 8 else ""))
    missing = _cochange_missing(graph, set(files))
    if missing:
        out.append("usually changes together with this diff, but untouched: "
                   + ", ".join(f"{m} ({w:.2f})" for m, w in missing[:5]))
    if deleted:
        out.append("deleted: " + ", ".join(deleted[:8]))
    if unindexed:
        out.append(f"not indexed (unsupported type or ignored): {', '.join(unindexed[:6])}"
                   + (f" (+{len(unindexed) - 6})" if len(unindexed) > 6 else ""))
    text = "\n".join(out)
    cap = max_tokens * 4
    return text if len(text) <= cap else text[:cap].rsplit("\n", 1)[0] + "\n  ... truncated"


def _tests_for(graph, sid: int) -> list[str]:
    """Test symbols linked to `sid` (links of kind 'tests', in either direction)."""
    names = []
    for other, kind, _w, _note, _fwd in graph._lnk.get(sid, ()):
        if kind == "tests":
            row = graph.db.execute("SELECT qualname FROM symbols WHERE id=?", (other,)).fetchone()
            if row:
                names.append(row[0])
    return names


def _cochange_missing(graph, files: set[str]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for f in files:
        for other, w in graph._fl.get(f, ()):
            if other not in files and w >= COCHANGE_MIN:
                best[other] = max(best.get(other, 0.0), w)
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
