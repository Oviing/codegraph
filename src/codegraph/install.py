"""`codegraph install` and `codegraph hook`: wire the graph into coding assistants and into git.

Everything written here is idempotent and reversible: instruction text lives between markers inside
files the user may also edit by hand, JSON configs are merged key by key, and git hooks are appended to
rather than replaced.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

START, END = "<!-- codegraph:start -->", "<!-- codegraph:end -->"
SH_START, SH_END = "# >>> codegraph >>>", "# <<< codegraph <<<"
GIT_HOOKS = ("post-commit", "post-checkout", "post-merge", "post-rewrite")
NUDGE_MARK = "codegraph-nudge"  # lets us find our own hook entry again
PLATFORMS = ("claude", "cursor", "gemini", "agents")

GUIDE = """\
## Code navigation: use codegraph before reading files

This repository is indexed by codegraph (symbols, call graph, imports, docs, SQL). Ask the graph instead
of reading whole files or grepping:

1. `repo_map` for an overview (pass `focus=` to personalise it), `search` to find a symbol by name or meaning.
2. `context <symbol>` for its source, what it uses and who uses it, in one reply.
3. `impact <symbol>` before changing anything; `review` before opening a PR (what the diff touches,
   blast radius, untested changes).
4. `callers` / `callees` / `related` / `deps` to walk the graph.

Edges tagged `[inferred]` or `[ambiguous]` (or `?` in `impact`) are name-based guesses, so treat them as
upper bounds and confirm by reading the code. Read files directly only for exact surrounding code.

No MCP server connected? The same tools run from the shell: `codegraph context <symbol>`,
`codegraph impact <symbol>`, `codegraph review`. Exit code 3 means the name is ambiguous, 4 that it was
not found.
"""

SKILL = f"""---
name: codegraph
description: Navigate this codebase through its code graph (symbols, callers, impact, docs, SQL) instead of reading whole files. Use before searching or reading code, before editing a function, and before opening a PR.
---

{GUIDE}"""

NUDGE = ("codegraph is available for this repo: try repo_map / search / context before grepping or "
         "reading whole files (CLI: codegraph search <query>).")


def _write(path: Path, text: str, report: list[str], what: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        report.append(f"unchanged  {what}")
        return
    verb = "updated" if path.exists() else "created"
    path.write_text(text)
    report.append(f"{verb:<10} {what}")


def _block(text: str) -> str:
    return f"{START}\n{text.rstrip()}\n{END}\n"


def _upsert_block(path: Path, text: str, report: list[str], what: str):
    """Put `text` between markers in a file that may hold other content; replace it if already there."""
    block = _block(text)
    cur = path.read_text() if path.exists() else ""
    if START in cur and END in cur:
        a, b = cur.index(START), cur.index(END) + len(END)
        new = cur[:a] + block.rstrip("\n") + cur[b:]
    else:
        new = cur + ("\n" if cur and not cur.endswith("\n\n") else "") + block
    _write(path, new, report, what)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"codegraph: {path} is not valid JSON ({e}); fix it or move it aside, then re-run") from e
    if not isinstance(data, dict):
        raise SystemExit(f"codegraph: {path} must hold a JSON object")
    return data


def _merge_mcp(path: Path, report: list[str], what: str):
    data = _load_json(path)
    servers = data.setdefault("mcpServers", {})
    # relative root: MCP clients start servers in the project directory, so the config stays portable
    servers["codegraph"] = {"command": "codegraph", "args": ["-r", ".", "serve"]}
    _write(path, json.dumps(data, indent=2) + "\n", report, what)


def _claude_hook(root: Path, report: list[str]):
    path = root / ".claude" / "settings.json"
    data = _load_json(path)
    payload = json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": NUDGE}})
    entry = {"matcher": "Glob|Grep", "hooks": [{"type": "command", "command": f"echo '{payload}' # {NUDGE_MARK}"}]}
    pre = data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    pre[:] = [e for e in pre if NUDGE_MARK not in json.dumps(e)] + [entry]
    _write(path, json.dumps(data, indent=2) + "\n", report, ".claude/settings.json (Glob|Grep hint)")


def install(root: str | Path, platform: str = "all", hooks: bool = True) -> list[str]:
    root = Path(root).resolve()
    chosen = PLATFORMS if platform == "all" else (platform,)
    unknown = [p for p in chosen if p not in PLATFORMS]
    if unknown:
        raise SystemExit(f"codegraph: unknown platform {unknown[0]!r}; choose from {', '.join(PLATFORMS)}, all")
    report: list[str] = []
    for p in chosen:
        if p == "claude":
            _write(root / ".claude" / "skills" / "codegraph" / "SKILL.md", SKILL, report,
                   ".claude/skills/codegraph/SKILL.md")
            _merge_mcp(root / ".mcp.json", report, ".mcp.json (MCP server)")
            _upsert_block(root / "CLAUDE.md", GUIDE, report, "CLAUDE.md (guidance block)")
            if hooks:
                _claude_hook(root, report)
        elif p == "cursor":
            rule = ("---\ndescription: Navigate the codebase with codegraph before reading files\n"
                    f"alwaysApply: true\n---\n\n{GUIDE}")
            _write(root / ".cursor" / "rules" / "codegraph.mdc", rule, report, ".cursor/rules/codegraph.mdc")
            _merge_mcp(root / ".cursor" / "mcp.json", report, ".cursor/mcp.json (MCP server)")
        elif p == "gemini":
            _upsert_block(root / "GEMINI.md", GUIDE, report, "GEMINI.md (guidance block)")
            _merge_mcp(root / ".gemini" / "settings.json", report, ".gemini/settings.json (MCP server)")
        elif p == "agents":  # Codex, opencode, Aider (via --read), Copilot and others read AGENTS.md
            _upsert_block(root / "AGENTS.md", GUIDE, report, "AGENTS.md (guidance block)")
    return report


def uninstall(root: str | Path) -> list[str]:
    """Remove what `install` wrote. Files that held only our block are deleted."""
    root = Path(root).resolve()
    report: list[str] = []
    for rel in ("CLAUDE.md", "GEMINI.md", "AGENTS.md"):
        path = root / rel
        if path.exists() and START in (cur := path.read_text()) and END in cur:
            rest = (cur[:cur.index(START)] + cur[cur.index(END) + len(END):]).strip()
            if rest:
                path.write_text(rest + "\n")
            else:
                path.unlink()
            report.append(f"cleaned    {rel}")
    for rel in (".claude/skills/codegraph/SKILL.md", ".cursor/rules/codegraph.mdc"):
        if (root / rel).exists():
            (root / rel).unlink()
            report.append(f"removed    {rel}")
    for rel in (".mcp.json", ".cursor/mcp.json", ".gemini/settings.json"):
        path = root / rel
        if path.exists() and "codegraph" in _load_json(path).get("mcpServers", {}):
            data = _load_json(path)
            del data["mcpServers"]["codegraph"]
            path.write_text(json.dumps(data, indent=2) + "\n")
            report.append(f"cleaned    {rel}")
    path = root / ".claude" / "settings.json"
    if path.exists() and NUDGE_MARK in path.read_text():
        data = _load_json(path)
        pre = data.get("hooks", {}).get("PreToolUse", [])
        pre[:] = [e for e in pre if NUDGE_MARK not in json.dumps(e)]
        path.write_text(json.dumps(data, indent=2) + "\n")
        report.append("cleaned    .claude/settings.json")
    return report or ["nothing to remove"]


# ---- git hooks ----

def _hooks_dir(root: Path) -> Path:
    p = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-path", "hooks"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("codegraph: not a git repository")
    d = Path(p.stdout.strip())
    return d if d.is_absolute() else root / d


def _hook_block() -> str:
    # backgrounded: a commit or checkout must never wait for, or fail because of, the indexer
    return (f"{SH_START}\ncommand -v codegraph >/dev/null 2>&1 && "
            f'(codegraph -r "$(git rev-parse --show-toplevel)" index >/dev/null 2>&1 &)\n{SH_END}\n')


def hook_install(root: str | Path) -> list[str]:
    root = Path(root).resolve()
    d = _hooks_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    report = []
    for name in GIT_HOOKS:
        path = d / name
        cur = path.read_text() if path.exists() else "#!/bin/sh\n"
        if SH_START in cur:
            a, b = cur.index(SH_START), cur.index(SH_END) + len(SH_END)
            new = cur[:a] + _hook_block().rstrip("\n") + cur[b:]
        else:
            new = cur + ("" if cur.endswith("\n") else "\n") + _hook_block()
        path.write_text(new)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        report.append(f"installed  {os.path.relpath(path, root)}")
    return report


def hook_uninstall(root: str | Path) -> list[str]:
    root = Path(root).resolve()
    d, report = _hooks_dir(root), []
    for name in GIT_HOOKS:
        path = d / name
        if not path.exists() or SH_START not in (cur := path.read_text()):
            continue
        rest = (cur[:cur.index(SH_START)] + cur[cur.index(SH_END) + len(SH_END):]).strip()
        if rest in ("", "#!/bin/sh"):
            path.unlink()
        else:
            path.write_text(rest + "\n")
        report.append(f"removed    {os.path.relpath(path, root)}")
    return report or ["no codegraph hooks found"]
