"""Agent memories: decisions, failed attempts, invariants and intent, anchored to symbols and files.

The source of truth is one Markdown file per memory in `.codegraph/memory/` (commit them: they are
reviewable in a PR and never conflict with each other). The tables in graph.db are a derived index,
rebuilt from those files on every index run, like `edges` are rebuilt from `raw_edges`.

Symbol ids change whenever a file is re-parsed, so a memory never stores one. It stores a stable anchor
(`file::qualname#hash-of-the-source-when-written`, or just a file) and `relink` re-attaches it to the
current symbol after every index: same source -> `ok`, changed source -> `stale`, gone -> `orphaned`.
"""
from __future__ import annotations

import calendar
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

MEMORY_DIR = ".codegraph/memory"
KINDS = ("decision", "failed_attempt", "invariant", "intent")
STATUSES = ("active", "superseded", "retired")
KIND_ORDER = {k: i for i, k in enumerate(("invariant", "failed_attempt", "decision", "intent"))}
STATE_ORDER = {"ok": 0, "stale": 1, "orphaned": 2}
FIELDS = ("id", "kind", "title", "status", "supersedes", "superseded_by", "author", "commit", "created",
          "anchors")
CHARS_PER_TOKEN = 4

# Tiers, like a brain: a new memory is *working*; recalled often enough it consolidates into *long_term*;
# left unused it fades to *dormant* (out of sight in context/impact/review, still found by recall, and
# revived by a hit). Invariants never fade. Usage is stored locally in graph.db (memory_access), not in
# the committed files, so recalling never rewrites them.
TIERS = ("working", "long_term", "dormant")
PROMOTE_RECALLS = 3        # recalls needed to consolidate into long term
WORKING_GRACE_DAYS = 14    # a working memory nobody recalls fades after this many idle days
LONG_IDLE_DAYS = 90        # a long-term memory fades after this many idle days
DAY = 86400
GITIGNORE = "graph.db*\nGRAPH_REPORT.md\ngraph.html\n"

# memory text is committed and shown to other agents, so obvious credentials are refused
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
)


class SecretError(ValueError):
    pass


# ---- file format ----

def parse(text: str) -> tuple[dict, str]:
    """Memory file -> (frontmatter dict, body). `anchors` is a list, everything else a string."""
    m = re.match(r"---\n(.*?)\n---[ \t]*(?:\n|\Z)(.*)\Z", text, re.S)
    if not m:
        raise ValueError("missing frontmatter")
    meta: dict = {}
    key = None
    for line in m.group(1).splitlines():
        if line.startswith("  - ") and key == "anchors":
            meta["anchors"].append(line[4:].strip())
            continue
        k, sep, v = line.partition(":")
        if not sep:
            continue
        key, v = k.strip(), v.strip()
        meta[key] = [] if key == "anchors" else v
    return meta, m.group(2).strip()


def render(meta: dict, text: str, why: str = "", alternatives: str = "") -> str:
    lines = ["---"]
    for k in FIELDS:
        v = meta.get(k, "")
        if k == "anchors":
            lines.append("anchors:")
            lines += [f"  - {a}" for a in v]
        else:
            lines.append(f"{k}: {' '.join(str(v).split())}".rstrip())
    lines.append("---")
    body = text.strip()
    if why.strip():
        body += f"\n\n## Why\n{why.strip()}"
    if alternatives.strip():
        body += f"\n\n## Alternatives considered\n{alternatives.strip()}"
    return "\n".join(lines) + "\n\n" + body + "\n"


def split_body(body: str) -> tuple[str, str, str]:
    parts: dict[str, list[str]] = {"": [], "why": [], "alt": []}
    cur = ""
    for line in body.splitlines():
        s = line.strip()
        if s == "## Why":
            cur = "why"
        elif s == "## Alternatives considered":
            cur = "alt"
        else:
            parts[cur].append(line)
    return tuple("\n".join(parts[k]).strip() for k in ("", "why", "alt"))  # type: ignore[return-value]


def parse_anchor(a: str) -> tuple[str, str, str]:
    """`file::qualname#hash` | `file` -> (file, qualname, hash)."""
    f, sep, rest = a.partition("::")
    if not sep:
        return f, "", ""
    qual, _, h = rest.rpartition("#") if "#" in rest else (rest, "", "")
    return f, qual, h


def mem_dir(root: str | Path) -> Path:
    return Path(root) / MEMORY_DIR


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def check_secrets(*texts: str):
    for t in texts:
        for pat in SECRET_PATTERNS:
            if pat.search(t):
                raise SecretError("Refusing to store this memory: it looks like it contains a credential "
                                  "(memory files are committed to git). Rephrase without the secret.")


def symbol_hash(root: str | Path, s, cache: dict | None = None) -> str:
    """Hash of a symbol's source, ignoring indentation and blank lines (so a reformat is not 'stale')."""
    if s["file"].endswith(".pdf"):
        text = s["doc"] or ""
    else:
        lines = cache.get(s["file"]) if cache is not None else None
        if lines is None:
            try:
                lines = (Path(root) / s["file"]).read_text(errors="replace").splitlines()
            except OSError:
                lines = []
            if cache is not None:
                cache[s["file"]] = lines
        text = "\n".join(lines[s["line_start"] - 1:s["line_end"]])
    return _sha("\n".join(ln.strip() for ln in text.splitlines() if ln.strip()))[:10]


def _git_head(root: Path) -> str:
    try:
        p = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


# ---- writing files ----

def write(root: str | Path, *, kind: str, title: str, text: str, why: str = "", alternatives: str = "",
          anchors: list[str] | None = None, author: str = "agent") -> str:
    """Create a memory file and return its id. The caller syncs the index afterwards."""
    check_secrets(title, text, why, alternatives)
    root = Path(root)
    d = mem_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    gi = root / ".codegraph" / ".gitignore"
    if not gi.exists():
        gi.write_text(GITIGNORE)
    mid = f"{time.strftime('%Y%m%d')}-{_sha(title + str(time.time_ns()))[:6]}"
    meta = {"id": mid, "kind": kind, "title": title, "status": "active", "author": author,
            "commit": _git_head(root), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "anchors": anchors or []}
    (d / f"{mid}.md").write_text(render(meta, text, why, alternatives))
    return mid


def find_file(root: str | Path, mid: str) -> Path | None:
    """Memory file for an id, or a unique id prefix."""
    d = mem_dir(root)
    if not d.is_dir():
        return None
    exact = d / f"{mid}.md"
    if exact.is_file():
        return exact
    hits = [p for p in d.glob(f"{mid}*.md")] if mid else []
    return hits[0] if len(hits) == 1 else None


def set_status(root: str | Path, mid: str, status: str, superseded_by: str = "") -> str | None:
    """Mark a memory retired/superseded. The file is kept (audit trail). Returns its id, or None."""
    p = find_file(root, mid)
    if p is None:
        return None
    meta, body = parse(p.read_text())
    text, why, alt = split_body(body)
    meta["status"] = status
    meta["superseded_by"] = superseded_by
    p.write_text(render(meta, text, why, alt))
    return p.stem


# ---- the derived index ----

def _delete(db, mid: str):
    for t in ("memories", "memory_anchors", "memory_links"):
        db.execute(f"DELETE FROM {t} WHERE {'id' if t == 'memories' else 'mem_id'}=?", (mid,))
    db.execute("DELETE FROM memory_fts WHERE mem_id=?", (mid,))
    # memory_access is kept: an edited or re-indexed memory keeps its usage; sync prunes vanished ids


def sync(db, root: str | Path) -> dict:
    """Make the memory tables match the files (re-reading only files whose hash changed)."""
    d = mem_dir(root)
    files = {p.stem: p for p in sorted(d.glob("*.md"))} if d.is_dir() else {}
    old = {r[0]: r[1] for r in db.execute("SELECT id, hash FROM memories")}
    for gone in set(old) - set(files):
        _delete(db, gone)
    changed = 0
    for mid, p in files.items():
        raw = p.read_text(errors="replace")
        h = _sha(raw)
        if old.get(mid) == h:
            continue
        try:
            meta, body = parse(raw)
        except ValueError as e:
            print(f"codegraph: skipped memory {p.name}: {e}", file=sys.stderr)
            continue
        text, why, alt = split_body(body)
        _delete(db, mid)
        db.execute("INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (mid, h, meta.get("kind", "decision"), meta.get("title") or mid, text, why, alt,
                    meta.get("status", "active") or "active", meta.get("superseded_by", ""),
                    meta.get("author", ""), meta.get("commit", ""), meta.get("created", "")))
        db.executemany("INSERT INTO memory_anchors VALUES(?,?,?,?)",
                       [(mid, *parse_anchor(a)) for a in meta.get("anchors", [])])
        db.execute("INSERT INTO memory_fts(mem_id,title,body,kind) VALUES(?,?,?,?)",
                   (mid, meta.get("title", ""), "\n".join((text, why, alt)), meta.get("kind", "")))
        changed += 1
    db.execute("DELETE FROM memory_access WHERE mem_id NOT IN (SELECT id FROM memories)")
    db.execute("INSERT OR IGNORE INTO memory_access(mem_id, tier, tier_since) "
               "SELECT id, 'working', created FROM memories")
    return {"memories": len(files), "changed": changed}


# ---- tiers ----

def _epoch(iso: str, default: float = 0.0) -> float:
    try:
        return float(calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return default


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def tier_for(kind: str, created: str, recalls: int, last_recalled: str, now: float) -> str:
    """Where a memory belongs, from how often and how recently it was recalled."""
    idle = (now - max(_epoch(created, now), _epoch(last_recalled))) / DAY
    pinned = kind == "invariant"
    if recalls >= PROMOTE_RECALLS:
        return "dormant" if idle > LONG_IDLE_DAYS and not pinned else "long_term"
    return "dormant" if idle > WORKING_GRACE_DAYS and not pinned else "working"


def reclassify(db, now: float | None = None, ids=None) -> int:
    """Move memories between tiers. Lazy (run on every update/recall), `now` is injectable for tests."""
    now = time.time() if now is None else now
    moved = 0
    q = "SELECT m.id, m.kind, m.created, a.recalls, a.last_recalled, a.tier " \
        "FROM memories m JOIN memory_access a ON a.mem_id=m.id"
    for r in db.execute(q).fetchall():
        if ids is not None and r["id"] not in ids:
            continue
        t = tier_for(r["kind"], r["created"], r["recalls"] or 0, r["last_recalled"], now)
        if t != r["tier"]:
            db.execute("UPDATE memory_access SET tier=?, tier_since=? WHERE mem_id=?", (t, _iso(now), r["id"]))
            moved += 1
    return moved


def touch(db, ids, now: float | None = None):
    """Record that these memories were recalled (explicitly), then re-judge their tier."""
    now = time.time() if now is None else now
    ids = list(ids)
    db.executemany("UPDATE memory_access SET recalls=recalls+1, last_recalled=? WHERE mem_id=?",
                   [(_iso(now), i) for i in ids])
    reclassify(db, now, set(ids))


def tier_counts(db) -> dict[str, int]:
    out = dict.fromkeys(TIERS, 0)
    for t, n in db.execute("SELECT a.tier, COUNT(*) FROM memories m JOIN memory_access a ON a.mem_id=m.id "
                           "WHERE m.status='active' GROUP BY a.tier"):
        out[t] = n
    return out


def relink(db, root: str | Path):
    """Re-attach every anchor to the symbols of the current index and judge whether it still holds."""
    db.execute("DELETE FROM memory_links")
    anchors = db.execute("SELECT mem_id, file, qualname, src_hash FROM memory_anchors").fetchall()
    if not anchors:
        return
    files = {r[0] for r in db.execute("SELECT path FROM files")}
    cache: dict = {}
    rows = []
    for mem_id, f, qual, src_hash in anchors:
        if not qual:  # a file anchor: it holds as long as the file exists
            rows.append((mem_id, f, None, "ok" if f in files else "orphaned"))
            continue
        syms = db.execute("SELECT * FROM symbols WHERE file=? AND qualname=?", (f, qual)).fetchall()
        if not syms:  # moved? only trust an unambiguous match elsewhere
            moved = db.execute("SELECT * FROM symbols WHERE qualname=? AND kind NOT IN "
                               "('module','impl','namespace')", (qual,)).fetchall()
            syms = moved if len(moved) == 1 else []
        if not syms:
            rows.append((mem_id, f, None, "orphaned"))
            continue
        hashes = {symbol_hash(root, s, cache) for s in syms}
        state = "ok" if (not src_hash or src_hash in hashes) else "stale"
        rows += [(mem_id, s["file"], s["id"], state) for s in syms]
    db.executemany("INSERT INTO memory_links VALUES(?,?,?,?)", rows)


def update(db, root: str | Path) -> dict:
    stats = sync(db, root)
    relink(db, root)
    reclassify(db)
    return stats


# ---- reading ----

def _worse(a: str, b: str) -> str:
    return a if STATE_ORDER.get(a, 0) >= STATE_ORDER.get(b, 0) else b


def _sorted(mems):
    mems = sorted(mems, key=lambda m: m["created"] or "", reverse=True)
    mems = sorted(mems, key=lambda m: KIND_ORDER.get(m["kind"], 9))
    return sorted(mems, key=lambda m: m.get("tier") == "dormant")  # faded memories last


def related(db, sym_ids, files, dormant: bool = False) -> list[dict]:
    """Active memories anchored to any of these symbols, or to one of these files. Each carries the
    worst `state` among its matching anchors. Invariants and failed attempts come first. Dormant
    (faded) memories are left out unless `dormant`."""
    sym_ids, files = set(sym_ids), set(files)
    out: dict[str, dict] = {}
    for r in db.execute("SELECT m.*, l.sym_id AS sid, l.file AS lfile, l.state AS state, "
                        "a.tier AS tier, a.recalls AS recalls, a.last_recalled AS last_recalled "
                        "FROM memories m JOIN memory_links l ON l.mem_id=m.id "
                        "LEFT JOIN memory_access a ON a.mem_id=m.id "
                        "WHERE m.status='active' AND (?=1 OR COALESCE(a.tier,'working')!='dormant')",
                        (1 if dormant else 0,)):
        hit = r["sid"] in sym_ids if r["sid"] is not None else (r["state"] == "ok" and r["lfile"] in files)
        if not hit:
            continue
        cur = out.get(r["id"])
        if cur is None:
            out[r["id"]] = dict(r)
        else:
            cur["state"] = _worse(cur["state"], r["state"])
    return _sorted(out.values())


def load(db, ids) -> list[dict]:
    """Full records (with anchors and overall state), in the order of `ids`."""
    out = []
    for mid in ids:
        r = db.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        if r is None:
            continue
        m = dict(r)
        a = db.execute("SELECT tier, recalls, last_recalled FROM memory_access WHERE mem_id=?", (mid,)).fetchone()
        m.update(dict(a) if a else {"tier": "working", "recalls": 0, "last_recalled": ""})
        m["anchors"] = [f"{f}::{q}" if q else f for f, q in
                        db.execute("SELECT file, qualname FROM memory_anchors WHERE mem_id=?", (mid,))]
        state = "ok"
        for (s,) in db.execute("SELECT state FROM memory_links WHERE mem_id=?", (mid,)):
            state = _worse(state, s)
        m["state"] = state
        out.append(m)
    return out


def search(db, query: str, kind: str = "", limit: int = 10) -> list[str]:
    words = re.findall(r"[^\W_]+", query, re.UNICODE)
    if not words:
        return []
    fts = " OR ".join(f'"{w}"*' for w in words)
    rows = db.execute("SELECT f.mem_id FROM memory_fts f JOIN memories m ON m.id=f.mem_id "
                      "LEFT JOIN memory_access a ON a.mem_id=m.id "
                      "WHERE memory_fts MATCH ? AND m.status='active' AND (?='' OR m.kind=?) "
                      "ORDER BY COALESCE(a.tier,'working')='dormant', rank LIMIT ?",
                      (fts, kind, kind, limit)).fetchall()
    return [r[0] for r in rows]


def recent(db, kind: str = "", limit: int = 10) -> list[str]:
    rows = db.execute("SELECT m.id FROM memories m LEFT JOIN memory_access a ON a.mem_id=m.id "
                      "WHERE m.status='active' AND (?='' OR m.kind=?) "
                      "ORDER BY COALESCE(a.tier,'working')='dormant', m.created DESC, m.id DESC LIMIT ?",
                      (kind, kind, limit)).fetchall()
    return [r[0] for r in rows]


def counts(db) -> tuple[int, int]:
    """(active memories, those whose anchor is stale or orphaned)."""
    total = db.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
    bad = db.execute("SELECT COUNT(DISTINCT m.id) FROM memories m JOIN memory_links l ON l.mem_id=m.id "
                     "WHERE m.status='active' AND l.state!='ok'").fetchone()[0]
    return total, bad


# ---- rendering ----

def tag(m: dict) -> str:
    if m.get("tier") == "dormant":
        return "  [dormant: unused for a while]" + _state_tag(m)
    return _state_tag(m)


def _state_tag(m: dict) -> str:
    if m["state"] == "stale":
        return f"  [stale: code changed since {m.get('commit_sha') or 'it was written'}]"
    if m["state"] == "orphaned":
        return "  [orphaned: anchor no longer found]"
    return ""


def section(header: str, mems: list[dict], max_tokens: int) -> str:
    """A compact list (title, kind, state, first line of the reason) that stays within a token budget."""
    if not mems:
        return ""
    cap = max(max_tokens, 60) * CHARS_PER_TOKEN
    out, used = [header], len(header)
    for i, m in enumerate(mems):
        lines = f"[{m['kind']}] {m['title']}  ({m['id']}){tag(m)}"
        why = (m.get("why") or "").splitlines()
        if why:
            lines += f"\n    why: {why[0][:140]}"
        if used + len(lines) > cap and i:
            out.append(f"... {len(mems) - i} more (recall())")
            break
        out.append(lines)
        used += len(lines)
    return "\n".join(out)


def card(m: dict, body_chars: int = 800) -> str:
    out = [f"[{m['kind']}] {m['title']}  ({m['id']}){tag(m)}"]
    if m.get("anchors"):
        out.append("  anchors: " + ", ".join(m["anchors"]))
    meta = " ".join(x for x in (f"{m['tier'].replace('_', '-')}, recalled {m.get('recalls') or 0}x"
                                if m.get("tier") else "",
                                f"by {m['author']}" if m.get("author") else "",
                                f"@ {m['commit_sha']}" if m.get("commit_sha") else "",
                                (m.get("created") or "")[:10]) if x)
    if meta:
        out.append("  " + meta)
    body = m["body"]
    out.append("  " + (body if len(body) <= body_chars else body[:body_chars] + " ...").replace("\n", "\n  "))
    if m.get("why"):
        out.append("  why: " + m["why"].replace("\n", "\n  "))
    if m.get("alternatives"):
        out.append("  alternatives: " + m["alternatives"].replace("\n", "\n  "))
    return "\n".join(out)


def summary(db) -> str:
    total, bad = counts(db)
    if not total:
        return ""
    t = tier_counts(db)
    by_tier = ", ".join(f"{t[k]} {k.replace('_', '-')}" for k in TIERS if t[k])
    return (f"{total} agent memories" + (f" ({by_tier})" if by_tier else "")
            + (f" ({bad} stale or orphaned)" if bad else "") + " - use recall()")
