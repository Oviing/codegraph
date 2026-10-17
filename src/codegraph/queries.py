"""Compact, token-budgeted views over the graph. Every method returns plain text."""
from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from pathlib import Path

from .indexer import db_path
from .rank import personalized_pagerank
from .store import Store, split_words

CHARS_PER_TOKEN = 4


def toks(s: str) -> int:
    return max(1, len(s) // CHARS_PER_TOKEN)


class Graph:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.store = Store(db_path(self.root))
        self.db = self.store.db
        self.reload()

    def reload(self):
        """Load adjacency lists into memory (ids only, cheap). Call after the index changes."""
        out, inn = defaultdict(list), defaultdict(list)
        for src, dst in self.db.execute("SELECT src, dst FROM edges"):
            out[src].append(dst)
            inn[dst].append(src)
        self._out, self._in, self._ncache = out, inn, {}

    def _nbrs(self, u: int):
        """Undirected neighbours of u, each weighted down if it is a hub (IDF-style)."""
        c = self._ncache.get(u)
        if c is None:
            c = [(v, 1.0 / math.log(2 + len(self._out.get(v, ())) + len(self._in.get(v, ()))))
                 for v in self._out.get(u, []) + self._in.get(u, [])]
            self._ncache[u] = c
        return c

    def _ppr(self, seed_ids) -> dict[int, float]:
        seed_ids = list(seed_ids)
        if not seed_ids:
            return {}
        return personalized_pagerank(self._nbrs, {i: 1.0 / len(seed_ids) for i in seed_ids})

    # ---------- helpers ----------
    def _line(self, r, with_sig=False) -> str:
        s = f"{r['kind']} {r['qualname']}  {r['file']}:{r['line_start']}-{r['line_end']}"
        if r["doc"]:
            s += f"  # {r['doc'][:80]}"
        if with_sig and r["sig"]:
            s += f"\n    {r['sig']}"
        return s

    def _find(self, q: str):
        """Resolve a user-supplied symbol to rows: qualname, then name, then suffix, then file:name."""
        db = self.db
        q = q.strip()
        if ":" in q and not q.startswith(":"):
            f, _, n = q.rpartition(":")
            rows = db.execute("SELECT * FROM symbols WHERE file LIKE ? AND (qualname=? OR name=?) "
                              "AND kind!='module'", (f"%{f}", n, n)).fetchall()
            if rows:
                return rows
        for sql, arg in (
            ("SELECT * FROM symbols WHERE qualname=? AND kind NOT IN ('module','impl')", q),
            ("SELECT * FROM symbols WHERE name=? AND kind NOT IN ('module','impl')", q),
            ("SELECT * FROM symbols WHERE qualname LIKE ? AND kind NOT IN ('module','impl')", "%." + q),
        ):
            rows = db.execute(sql, (arg,)).fetchall()
            if rows:
                return rows
        return []

    def _one(self, q: str):
        rows = self._find(q)
        if not rows:
            return None, f"No symbol matches '{q}'. Try search()."
        if len(rows) > 1:
            lines = "\n".join(self._line(r) for r in rows[:10])
            return None, f"'{q}' is ambiguous ({len(rows)} matches). Re-run with qualname or file:name:\n{lines}"
        return rows[0], ""

    def _read(self, r) -> str:
        try:
            lines = (self.root / r["file"]).read_text(errors="replace").splitlines()
        except OSError:
            return "(file unreadable)"
        return "\n".join(lines[r["line_start"] - 1:r["line_end"]])

    def _file_tokens(self, files) -> int:
        total = 0
        for f in set(files):
            try:
                total += toks((self.root / f).read_text(errors="replace"))
            except OSError:
                pass
        return total

    # ---------- tools ----------
    def _focus_seeds(self, focus: str) -> list[int]:
        seeds: list[int] = []
        for term in [t.strip() for t in focus.split(",") if t.strip()]:
            rows = self.db.execute("SELECT id FROM symbols WHERE file LIKE ? AND kind!='module'",
                                   ("%" + term,)).fetchall()
            if rows:
                seeds += [r[0] for r in rows][:50]
                continue
            seeds += [r["id"] for r in self._search_rows(term, 3)]
        return seeds

    def repo_map(self, prefix: str = "", max_tokens: int = 1500, focus: str = "") -> str:
        """Files ranked by PageRank; with `focus` (symbols, words or paths, comma separated) ranked by
        personalized PageRank, i.e. closeness to what you are working on."""
        db = self.db
        note = ""
        scores: dict[int, float] = {}
        if focus:
            seeds = self._focus_seeds(focus)
            if seeds:
                scores = self._ppr(seeds)
                note = f"(personalized to: {focus})\n"
            else:
                note = f"(no symbols matched focus '{focus}'; showing global ranking)\n"
        if not scores:
            scores = {r[0]: r[1] for r in db.execute("SELECT id, rank FROM symbols")}
        syms = db.execute("SELECT * FROM symbols WHERE kind NOT IN ('module','impl') AND file LIKE ?",
                          (prefix + "%",)).fetchall()
        by_file = defaultdict(list)
        for s_ in syms:
            if scores.get(s_["id"], 0) > 0:
                by_file[s_["file"]].append(s_)
        scored = sorted(((sum(scores[x["id"]] for x in ss), f) for f, ss in by_file.items()),
                        key=lambda t: (-t[0], t[1]))
        out, used = [], toks(note)
        for i, (_, f) in enumerate(scored):
            ss = sorted(by_file[f], key=lambda x: -scores[x["id"]])[:8]
            ss.sort(key=lambda x: x["line_start"])
            block = f + "\n" + "\n".join(
                f"  {x['kind']} {x['qualname']}" + (f"  ({len(self._in[x['id']])} refs)" if self._in.get(x["id"]) else "")
                for x in ss)
            if used + toks(block) > max_tokens and out:
                out.append(f"... {len(scored) - i} more files (use prefix= or focus= to zoom in)")
                break
            out.append(block)
            used += toks(block)
        return note + "\n".join(out) if out else "Index is empty. Run reindex()."

    def _search_rows(self, query: str, limit: int):
        words = re.findall(r"[A-Za-z0-9_]+", split_words(query))
        if not words:
            return []
        fts = " AND ".join(f'"{w}"*' for w in words)
        rows = self.db.execute(
            "SELECT s.* FROM sym_fts f JOIN symbols s ON s.id=f.rowid WHERE sym_fts MATCH ? "
            "AND s.kind NOT IN ('module','impl') ORDER BY bm25(sym_fts, 10, 5, 3, 1) LIMIT ?",
            (fts, limit)).fetchall()
        if not rows:
            rows = self.db.execute("SELECT * FROM symbols WHERE name LIKE ? AND kind NOT IN ('module','impl') LIMIT ?",
                                   (f"%{query}%", limit)).fetchall()
        return rows

    def search(self, query: str, limit: int = 10) -> str:
        rows = self._search_rows(query, limit)
        return "\n".join(self._line(r, with_sig=True) for r in rows) or "No matches."

    def outline(self, file: str) -> str:
        rows = self.db.execute("SELECT * FROM symbols WHERE file LIKE ? AND kind!='module' ORDER BY file, line_start",
                               ("%" + file,)).fetchall()
        if not rows:
            return f"No indexed symbols in '{file}'."
        out, cur = [], None
        depth = {}
        for r in rows:
            if r["file"] != cur:
                cur = r["file"]
                out.append(cur)
            depth[r["id"]] = depth.get(r["parent"], -1) + 1 if r["parent"] in depth else 0
            out.append("  " * (depth[r["id"]] + 1) + f"{r['kind']} {r['name']}  L{r['line_start']}-{r['line_end']}"
                       + (f"  # {r['doc'][:60]}" if r["doc"] else ""))
        return "\n".join(out)

    def source(self, symbol: str) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        return f"{r['file']}:{r['line_start']}-{r['line_end']}\n{self._read(r)}"

    def _neighbors(self, sid: int, direction: str):
        if direction == "out":
            sql = "SELECT s.*, e.kind AS ek FROM edges e JOIN symbols s ON s.id=e.dst WHERE e.src=?"
        else:
            sql = "SELECT s.*, e.kind AS ek FROM edges e JOIN symbols s ON s.id=e.src WHERE e.dst=?"
        return self.db.execute(sql, (sid,)).fetchall()

    def _bfs(self, r, direction: str, depth: int, limit: int):
        seen = {r["id"]}
        q = deque([(r["id"], 0)])
        levels = defaultdict(list)
        while q:
            sid, d = q.popleft()
            if d >= depth:
                continue
            for n in self._neighbors(sid, direction):
                if n["id"] in seen:
                    continue
                seen.add(n["id"])
                levels[d + 1].append(n)
                q.append((n["id"], d + 1))
        return levels

    def _render_levels(self, levels, limit: int) -> str:
        out, count = [], 0
        for d in sorted(levels):
            out.append(f"depth {d}:")
            for n in levels[d]:
                if count >= limit:
                    out.append(f"  ... truncated at {limit}")
                    return "\n".join(out)
                out.append("  " + self._line(n))
                count += 1
        return "\n".join(out) or "(none)"

    def callers(self, symbol: str, depth: int = 1, limit: int = 30) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        return f"callers of {r['qualname']}:\n" + self._render_levels(self._bfs(r, "in", depth, limit), limit)

    def callees(self, symbol: str, depth: int = 1, limit: int = 30) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        return f"callees of {r['qualname']}:\n" + self._render_levels(self._bfs(r, "out", depth, limit), limit)

    def impact(self, symbol: str, depth: int = 3) -> str:
        """Blast radius: everything that transitively depends on the symbol."""
        r, err = self._one(symbol)
        if err:
            return err
        levels = self._bfs(r, "in", depth, 10_000)
        files = defaultdict(list)
        for d, ns in levels.items():
            for n in ns:
                files[n["file"]].append((d, n["qualname"]))
        if not files:
            return f"Nothing depends on {r['qualname']} (static analysis)."
        total = sum(len(v) for v in files.values())
        out = [f"changing {r['qualname']} may affect {total} symbols in {len(files)} files:"]
        for f, items in sorted(files.items(), key=lambda kv: min(d for d, _ in kv[1])):
            items.sort()
            out.append(f"  {f}: " + ", ".join(f"{q}(d{d})" for d, q in items[:8]) +
                       (f", +{len(items) - 8} more" if len(items) > 8 else ""))
        return "\n".join(out)

    def _related(self, r, budget: int):
        """Best-first neighbourhood: direct callees/callers first, then the highest personalized-PageRank
        symbols further out, filled until the token budget is spent. Returns [(relation, row)]."""
        sid = r["id"]
        scores = self._ppr([sid])
        scores.pop(sid, None)
        ids = sorted(scores, key=scores.get, reverse=True)[:80]
        direct_out, direct_in = set(self._out.get(sid, ())), set(self._in.get(sid, ()))
        ids += [i for i in (direct_out | direct_in) if i not in scores]
        if not ids:
            return []
        rows = {x["id"]: x for x in self.db.execute(
            f"SELECT * FROM symbols WHERE id IN ({','.join('?' * len(ids))})", ids)}
        direct = sorted(direct_out | direct_in, key=lambda i: -scores.get(i, 0))
        far = [i for i in ids if i not in direct_out and i not in direct_in]
        picked, used = [], 0
        for i in direct + far:
            x = rows.get(i)
            if x is None or (x["kind"] in ("module", "impl") and i not in direct_out | direct_in):
                continue
            c = toks(f"{x['file']}:{x['line_start']}  {x['sig'] or x['qualname']}\n")
            if used + c > budget:
                continue
            rel = "uses" if i in direct_out else "used by" if i in direct_in else "related"
            picked.append((rel, x))
            used += c
        return picked

    def related(self, symbol: str, max_tokens: int = 800) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        picked = self._related(r, max_tokens)
        return "\n".join(f"{rel:8} {x['file']}:{x['line_start']}  {x['sig'] or x['qualname']}"
                         for rel, x in picked) or "(isolated symbol)"

    def context(self, symbol: str, max_tokens: int = 1500) -> str:
        """One call that replaces reading several files: source + the best-ranked neighbourhood that fits."""
        r, err = self._one(symbol)
        if err:
            return err
        head = f"# {r['kind']} {r['qualname']}  {r['file']}:{r['line_start']}-{r['line_end']}"
        src = self._read(r)
        budget_src = max_tokens // 2
        if toks(src) > budget_src:
            src = src[:budget_src * CHARS_PER_TOKEN] + "\n... [source truncated, use source() / Read for the rest]"
        parts = [head, src]
        used_files = [r["file"]]
        if r["parent"]:
            p = self.db.execute("SELECT * FROM symbols WHERE id=?", (r["parent"],)).fetchone()
            if p and p["kind"] != "module":
                parts.append(f"## inside\n{p['file']}:{p['line_start']}  {p['sig'] or p['qualname']}")
        remaining = max_tokens - toks("\n".join(parts)) - 20
        picked = self._related(r, max(remaining, 0))
        for rel, title in (("uses", "## uses"), ("used by", "## used by"),
                           ("related", "## related (further out, ranked by closeness)")):
            group = [x for k, x in picked if k == rel]
            if group:
                parts.append(title)
                for x in group:
                    parts.append(f"{x['file']}:{x['line_start']}  {x['sig'] or x['qualname']}")
                    used_files.append(x["file"])
        body = "\n".join(parts)
        naive = self._file_tokens(used_files)
        return body + f"\n[~{toks(body)} tokens vs ~{naive} to read {len(set(used_files))} whole files]"

    def deps(self, file: str) -> str:
        rows = self.db.execute("SELECT path FROM files WHERE path LIKE ?", ("%" + file,)).fetchall()
        if len(rows) != 1:
            return "No match." if not rows else "Ambiguous:\n" + "\n".join(r[0] for r in rows[:10])
        f = rows[0][0]
        imports = [r[0] for r in self.db.execute("SELECT dst FROM imports WHERE file=?", (f,))]
        importers = [r[0] for r in self.db.execute("SELECT file FROM imports WHERE dst=?", (f,))]
        return (f"{f}\nimports: {', '.join(sorted(imports)) or '-'}\nimported by: {', '.join(sorted(importers)) or '-'}")

    def stats(self) -> str:
        db = self.db
        q = lambda s: db.execute(s).fetchone()[0]
        return (f"{q('SELECT COUNT(*) FROM files')} files, {q('SELECT COUNT(*) FROM symbols')} symbols, "
                f"{q('SELECT COUNT(*) FROM edges')} edges, {q('SELECT COUNT(*) FROM imports')} file imports")
