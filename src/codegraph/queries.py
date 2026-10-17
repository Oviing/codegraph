"""Compact, token-budgeted views over the graph. Every method returns plain text."""
from __future__ import annotations

import functools
import math
import re
import sys
import threading
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from .embed import load_embedder, load_matrix, symbol_text
from .indexer import db_path
from .links import is_test_path
from .rank import personalized_pagerank
from .store import Store, split_words

CHARS_PER_TOKEN = 4
PPR_MAX_NODES = 20_000
NBR_CACHE_MAX = 20_000  # neighbour lists kept in memory (LRU)
NEAR_CANDIDATES = 80
# how much a non-structural link carries a random walk, relative to a call edge (1.0). Lower means
# the walk prefers structure and only spills over the soft links when structure runs out.
LINK_WEIGHT = {"similar": 0.3, "tests": 0.6, "key": 0.6}
COCHANGE_BLEND = 0.25  # share of a co-changed file's score a file inherits in repo_map
RRF_K = 60  # reciprocal rank fusion constant (the usual value)
TEST_DAMPING = 0.5  # weight of a semantic hit that lives in a test file
SEM_MIN = 0.55  # a semantic hit needs at least this cosine ...
SEM_BAND = 0.12  # ... and must be within this of the best one, so a weak query does not return noise


def toks(s: str) -> int:
    return max(1, len(s) // CHARS_PER_TOKEN)


def like(s: str) -> str:
    """Escape a user-supplied string for use inside a LIKE pattern.

    Without this, a query containing `%` or `_` silently becomes a wildcard and returns
    confident, wrong results.
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _suffix(term: str) -> str:
    return "%" + like(term)


def synchronized(fn):
    """Serialise access to the shared SQLite connection and its caches.

    The MCP server dispatches tool calls concurrently, and one sqlite3 connection cannot serve
    overlapping statements. Reentrant so that composed tools (context -> _related -> _ppr) work.
    """

    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)

    return wrapper


class Graph:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.store = Store(db_path(self.root))
        self.db = self.store.db
        # one connection and one cache are shared by concurrent MCP tool calls
        self._lock = threading.RLock()
        self._ftok_cache: dict[str, tuple[float, int]] = {}
        self._sem = None  # loaded lazily: see _semantic()
        # outcome of the last symbol lookup: "ok" | "ambiguous" | "missing". The CLI maps it to an exit code.
        self.status = "ok"
        self.reload()

    def reload(self):
        """Load adjacency lists into memory (ids only, cheap). Call after the index changes."""
        with self._lock:
            out, inn = defaultdict(list), defaultdict(list)
            for src, dst in self.db.execute("SELECT src, dst FROM edges"):
                out[src].append(dst)
                inn[dst].append(src)
            lnk = defaultdict(list)  # soft links, both directions: (other, kind, strength, note, forward)
            for src, dst, kind, w, note in self.db.execute("SELECT src, dst, kind, weight, note FROM links"):
                lnk[src].append((dst, kind, w, note, True))
                lnk[dst].append((src, kind, w, note, False))
            fl = defaultdict(list)
            for a, b, w in self.db.execute("SELECT a, b, weight FROM file_links"):
                fl[a].append((b, w))
                fl[b].append((a, w))
            self._out, self._in, self._ncache = out, inn, OrderedDict()
            self._lnk, self._fl, self._sem = lnk, fl, None
            self._ftok_cache.clear()

    def close(self):
        with self._lock:
            self.store.close()

    @synchronized
    def _nbrs(self, u: int):
        """Undirected neighbours of u, each weighted down if it is a hub (IDF-style). Soft links
        (similar / tests / key) join the walk at a lower weight than call edges."""
        c = self._ncache.get(u)
        if c is not None:
            self._ncache.move_to_end(u)
        else:
            def hub(v: int) -> float:
                return 1.0 / math.log(2 + len(self._out.get(v, ())) + len(self._in.get(v, ())) + len(self._lnk.get(v, ())))

            c = [(v, hub(v)) for v in self._out.get(u, []) + self._in.get(u, [])]
            c += [(v, LINK_WEIGHT.get(kind, 0.3) * w * hub(v)) for v, kind, w, _, _ in self._lnk.get(u, ())]
            self._ncache[u] = c
            if len(self._ncache) > NBR_CACHE_MAX:  # bounded: a long session must not cache the whole graph
                self._ncache.popitem(last=False)
        return c

    @synchronized
    def _ppr(self, seed_ids) -> dict[int, float]:
        seed_ids = list(dict.fromkeys(seed_ids))
        if not seed_ids:
            return {}
        return personalized_pagerank(self._nbrs, {i: 1.0 / len(seed_ids) for i in seed_ids},
                                     max_nodes=PPR_MAX_NODES)

    # ---------- helpers ----------
    def _line(self, r, with_sig=False) -> str:
        s = f"{r['kind']} {r['qualname']}  {r['file']}:{r['line_start']}-{r['line_end']}"
        if r["doc"]:
            s += f"  # {r['doc'][:80]}"
        if with_sig and r["sig"]:
            s += f"\n    {r['sig']}"
        return s

    # kinds that can never be the target of a user query
    SKIP_KINDS = "('module','impl','namespace')"

    @synchronized
    def _find(self, q: str):
        """Resolve a user-supplied symbol to rows: file:name, then qualname, name, then suffix."""
        db = self.db
        q = q.strip()
        if ":" in q and not q.startswith(":"):
            f, _, n = q.rpartition(":")
            rows = db.execute(f"SELECT * FROM symbols WHERE file LIKE ? ESCAPE '\\' "
                              f"AND (qualname=? OR name=?) AND kind NOT IN {self.SKIP_KINDS}",
                              (_suffix(f), n, n)).fetchall()
            if rows:
                return rows
        for sql, arg in (
            (f"SELECT * FROM symbols WHERE qualname=? AND kind NOT IN {self.SKIP_KINDS}", q),
            (f"SELECT * FROM symbols WHERE name=? AND kind NOT IN {self.SKIP_KINDS}", q),
            (f"SELECT * FROM symbols WHERE qualname LIKE ? ESCAPE '\\' "
             f"AND kind NOT IN {self.SKIP_KINDS}", "%." + like(q)),
        ):
            rows = db.execute(sql, (arg,)).fetchall()
            if rows:
                return rows
        return []

    @synchronized
    def _one(self, q: str):
        rows = self._find(q)
        self.status = "missing" if not rows else "ambiguous" if len(rows) > 1 else "ok"
        if not rows:
            return None, f"No symbol matches '{q}'. Try search()."
        if len(rows) > 1:
            lines = "\n".join(self._line(r) for r in rows[:10])
            return None, f"'{q}' is ambiguous ({len(rows)} matches). Re-run with qualname or file:name:\n{lines}"
        return rows[0], ""

    @synchronized
    def _read(self, r) -> str:
        if r["file"].endswith(".pdf"):  # page text lives in the index, line numbers are page numbers
            return r["doc"] or ""
        try:
            lines = (self.root / r["file"]).read_text(errors="replace").splitlines()
        except OSError:
            return "(file unreadable)"
        return "\n".join(lines[r["line_start"] - 1:r["line_end"]])

    @synchronized
    def _file_tokens(self, files) -> int:
        """Approximate whole-file token cost, cached per mtime.

        This is the denominator of the savings estimate, so it is cached rather than re-read on
        every context() call -- reading whole files to describe a snippet is the one thing this
        tool is trying to avoid.
        """
        total = 0
        with self._lock:
            for f in sorted(set(files)):
                p = self.root / f
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                cached = self._ftok_cache.get(f)
                if cached is not None and cached[0] == mtime:
                    total += cached[1]
                    continue
                try:
                    n = toks(p.read_text(errors="replace"))
                except OSError:
                    continue
                self._ftok_cache[f] = (mtime, n)
                total += n
        return total

    # ---------- tools ----------
    @synchronized
    def _focus_seeds(self, focus: str) -> list[int]:
        seeds: list[int] = []
        for term in [t.strip() for t in focus.split(",") if t.strip()]:
            rows = self.db.execute(
                f"SELECT id FROM symbols WHERE file LIKE ? ESCAPE '\\' AND kind NOT IN "
                f"{self.SKIP_KINDS} ORDER BY id LIMIT 50", (_suffix(term),)).fetchall()
            if rows:
                seeds += [r[0] for r in rows]
                continue
            seeds += [r["id"] for r, _ in self._hybrid_rows(term, 3)]
        return list(dict.fromkeys(seeds))

    @synchronized
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
        syms = db.execute(f"SELECT * FROM symbols WHERE kind NOT IN {self.SKIP_KINDS} "
                          f"AND file LIKE ? ESCAPE '\\'", (_suffix(prefix),)).fetchall()
        by_file = defaultdict(list)
        for s_ in syms:
            if scores.get(s_["id"], 0) > 0:
                by_file[s_["file"]].append(s_)
        totals = {f: sum(scores[x["id"]] for x in ss) for f, ss in by_file.items()}
        glob: dict[int, float] = {}
        if self._fl:  # files that change together with important files are important too
            boost: dict[str, float] = defaultdict(float)
            for f, t in totals.items():
                for g, w in self._fl.get(f, ()):
                    boost[g] += COCHANGE_BLEND * w * t
            for g, b in boost.items():
                if g not in totals:  # reachable only through history: show it by its global rank
                    if not glob:
                        glob = {r[0]: r[1] for r in db.execute("SELECT id, rank FROM symbols")}
                    extra = [x for x in syms if x["file"] == g and glob.get(x["id"], 0) > 0]
                    if extra:
                        by_file[g] = extra
                        totals[g] = 0.0
                if g in totals:
                    totals[g] += b
        scored = sorted(((t, f) for f, t in totals.items()), key=lambda t: (-t[0], t[1]))
        out, used = [], toks(note)
        for i, (_, f) in enumerate(scored):
            ss = sorted(by_file[f], key=lambda x: -(scores.get(x["id"]) or glob.get(x["id"], 0)))
            seen_qn: set[str] = set()  # overloads share a qualname; show each once
            ss = [x for x in ss if not (x["qualname"] in seen_qn or seen_qn.add(x["qualname"]))][:8]
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

    @synchronized
    def _search_rows(self, query: str, limit: int):
        words = [w for w in re.findall(r"[^\W_]+", split_words(query), re.UNICODE) if w]
        if not words:
            return []
        fts = " AND ".join(f'"{w}"*' for w in words)
        try:
            rows = self.db.execute(
                "SELECT s.* FROM sym_fts f JOIN symbols s ON s.id=f.rowid WHERE sym_fts MATCH ? "
                f"AND s.kind NOT IN {self.SKIP_KINDS} "
                "ORDER BY bm25(sym_fts, 10, 5, 3, 1) LIMIT ?", (fts, limit)).fetchall()
        except Exception:  # a malformed FTS expression must not take the tool down
            rows = []
        if not rows:
            rows = self.db.execute(
                f"SELECT * FROM symbols WHERE name LIKE ? ESCAPE '\\' AND kind NOT IN "
                f"{self.SKIP_KINDS} LIMIT ?", (_suffix(query), limit)).fetchall()
        return rows

    @synchronized
    def _semantic(self):
        """(embedder, ids, matrix, id->row) when this index has vectors made by the configured model,
        else None. Loaded once per index version; the model itself loads on the first query embedding."""
        if self._sem is None:
            self._sem = False
            try:
                emb = load_embedder()
                if emb is not None and self.store.get_meta("embed_model") == emb.name:
                    ids, mat = load_matrix(self.db)
                    if ids is not None:
                        self._sem = (emb, ids, mat, {int(i): n for n, i in enumerate(ids)})
            except Exception as e:  # semantic search is a bonus: never take a tool down for it
                print(f"codegraph: semantic search disabled: {e}", file=sys.stderr, flush=True)
        return self._sem or None

    @synchronized
    def _rows_by_id(self, ids):
        ids = list(ids)
        if not ids:
            return []
        found = {r["id"]: r for r in self.db.execute(
            f"SELECT * FROM symbols WHERE id IN ({','.join('?' * len(ids))})", ids)}
        return [found[i] for i in ids if i in found]

    @synchronized
    def _vector_hits(self, vec, n: int, exclude: int | None = None) -> list[tuple[int, float]]:
        """The n symbols closest to `vec` by cosine, best first, as (id, similarity)."""
        import numpy as np
        _, ids, mat, _ = self._semantic()
        sims = mat @ vec
        order = np.argsort(-sims)[:n + 1]
        return [(int(ids[i]), float(sims[i])) for i in order if exclude is None or int(ids[i]) != exclude][:n]

    @synchronized
    def _hybrid_rows(self, query: str, limit: int):
        """[(row, signal)] fusing lexical (FTS5) and semantic (vector) rankings with reciprocal rank
        fusion; signal is "lexical", "semantic" or "both". Without vectors this is the plain FTS result."""
        pool = max(limit * 3, 30)
        lex = self._search_rows(query, pool)
        sem = self._semantic()
        if sem is None:
            return [(r, "lexical") for r in lex[:limit]]
        hits = self._vector_hits(sem[0].embed_query(query), pool)
        if hits:
            floor = max(SEM_MIN, hits[0][1] - SEM_BAND)
            hits = [h for h in hits if h[1] >= floor]
        score: dict[int, float] = defaultdict(float)
        seen_lex, seen_sem = set(), set()
        for rank, r in enumerate(lex):
            score[r["id"]] += 1.0 / (RRF_K + rank)
            seen_lex.add(r["id"])
        rows = {r["id"]: r for r in lex}
        rows.update({r["id"]: r for r in self._rows_by_id([i for i, _ in hits if i not in rows])})
        for rank, (i, _) in enumerate(hits):
            # test code resembles everything it exercises; it should not outrank the code itself
            test = i in rows and is_test_path(rows[i]["file"])
            score[i] += (TEST_DAMPING if test else 1.0) / (RRF_K + rank)
            seen_sem.add(i)
        best = sorted(score, key=lambda i: -score[i])[:limit]
        return [(rows[i], "both" if i in seen_lex and i in seen_sem else "semantic" if i in seen_sem else "lexical")
                for i in best if i in rows]

    @synchronized
    def search(self, query: str, limit: int = 10) -> str:
        limit = max(1, min(limit, 100))
        if self._semantic() is None:  # no vectors: the original lexical output, unchanged
            rows = self._search_rows(query, limit)
            return "\n".join(self._line(r, with_sig=True) for r in rows) or "No matches."
        out = []
        for r, signal in self._hybrid_rows(query, limit):
            head, _, sig = self._line(r, with_sig=True).partition("\n")
            out.append(f"{head}  [{signal}]" + (f"\n{sig}" if sig else ""))
        return "\n".join(out) or "No matches."

    @synchronized
    def similar(self, symbol: str, limit: int = 10) -> str:
        """Symbols whose meaning is closest to this one, whether or not anything links them."""
        sem = self._semantic()
        if sem is None:
            return ("Semantic index not available. Install the extra (pip install 'codegraph[embed]') "
                    "and run reindex(), or use related() for structural neighbours.")
        r, err = self._one(symbol)
        if err:
            return err
        limit = max(1, min(limit, 50))
        emb, _, mat, pos = sem
        vec = mat[pos[r["id"]]] if r["id"] in pos else emb.embed_passages([symbol_text(r)])[0]
        direct = set(self._out.get(r["id"], ())) | set(self._in.get(r["id"], ()))
        hits = [h for h in self._vector_hits(vec, limit, exclude=r["id"]) if h[1] >= SEM_MIN]
        rows = {x["id"]: x for x in self._rows_by_id([i for i, _ in hits])}
        return "\n".join(
            f"{sim:.2f}  {self._line(rows[i], with_sig=True)}" + ("  [linked]" if i in direct else "")
            for i, sim in hits if i in rows) or "(nothing similar)"

    @synchronized
    def outline(self, file: str) -> str:
        rows = self.db.execute(
            "SELECT * FROM symbols WHERE file LIKE ? ESCAPE '\\' AND kind!='module' "
            "ORDER BY file, line_start", (_suffix(file),)).fetchall()
        self.status = "ok" if rows else "missing"
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

    @synchronized
    def source(self, symbol: str) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        return f"{r['file']}:{r['line_start']}-{r['line_end']}\n{self._read(r)}"

    def _neighbors(self, sid: int, direction: str):
        if direction == "out":
            sql = "SELECT s.*, e.kind AS ek, e.conf AS conf FROM edges e JOIN symbols s ON s.id=e.dst WHERE e.src=?"
        else:
            sql = "SELECT s.*, e.kind AS ek, e.conf AS conf FROM edges e JOIN symbols s ON s.id=e.src WHERE e.dst=?"
        return self.db.execute(sql, (sid,)).fetchall()

    @synchronized
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

    @staticmethod
    def _conf_tag(n) -> str:
        """Only edges that are not certain are tagged, so the common case stays free of noise."""
        conf = n["conf"] if "conf" in n.keys() else "extracted"  # noqa: SIM118 (sqlite3.Row: `in` tests values)
        return "" if conf == "extracted" else f"  [{conf}]"

    def _render_levels(self, levels, limit: int) -> str:
        out, count = [], 0
        for d in sorted(levels):
            out.append(f"depth {d}:")
            for n in levels[d]:
                if count >= limit:
                    out.append(f"  ... truncated at {limit}")
                    return "\n".join(out)
                out.append("  " + self._line(n) + self._conf_tag(n))
                count += 1
        return "\n".join(out) or "(none)"

    @synchronized
    def callers(self, symbol: str, depth: int = 1, limit: int = 30) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        depth = max(1, min(depth, 10))
        return f"callers of {r['qualname']}:\n" + self._render_levels(self._bfs(r, "in", depth, limit), limit)

    @synchronized
    def callees(self, symbol: str, depth: int = 1, limit: int = 30) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        depth = max(1, min(depth, 10))
        return f"callees of {r['qualname']}:\n" + self._render_levels(self._bfs(r, "out", depth, limit), limit)

    @synchronized
    def impact(self, symbol: str, depth: int = 3) -> str:
        """Blast radius: everything that transitively depends on the symbol."""
        r, err = self._one(symbol)
        if err:
            return err
        depth = max(1, min(depth, 10))
        levels = self._bfs(r, "in", depth, 10_000)
        files = defaultdict(list)
        for d, ns in levels.items():
            for n in ns:
                files[n["file"]].append((d, n["qualname"] + ("" if n["conf"] == "extracted" else "?")))
        if not files:
            return f"Nothing depends on {r['qualname']} (static analysis)."
        total = sum(len(v) for v in files.values())
        plural = "symbol" if total == 1 else "symbols"
        out = [f"changing {r['qualname']} may affect {total} {plural} in {len(files)} file(s)"
               f"{' (? = reached through an inferred or ambiguous link)' if any(q.endswith('?') for v in files.values() for _, q in v) else ''}:"]
        for f, items in sorted(files.items(), key=lambda kv: min(d for d, _ in kv[1])):
            items.sort()
            out.append(f"  {f}: " + ", ".join(f"{q[:-1] if q.endswith('?') else q}(d{d}{'?' if q.endswith('?') else ''})" for d, q in items[:8]) +
                       (f", +{len(items) - 8} more" if len(items) > 8 else ""))
        return "\n".join(out)

    @staticmethod
    def _link_label(kind: str, forward: bool) -> str:
        if kind == "tests":
            return "tests" if forward else "tested by"
        return kind  # "similar", "key"

    @synchronized
    def _related(self, r, budget: int):
        """Best-first neighbourhood: direct callees/callers first, then soft links (tests, shared keys,
        similar), then the highest personalized-PageRank symbols further out, filled until the token
        budget is spent. Returns [(relation, row, note)]."""
        sid = r["id"]
        scores = self._ppr([sid])
        scores.pop(sid, None)
        direct_out, direct_in = set(self._out.get(sid, ())), set(self._in.get(sid, ()))
        soft: dict[int, tuple[str, str]] = {}  # soft links only; a structural edge always wins
        for v, kind, _, note, fwd in sorted(self._lnk.get(sid, ()), key=lambda t: -scores.get(t[0], 0)):
            if v in direct_out or v in direct_in:
                continue
            label, seen_note = soft.get(v, (self._link_label(kind, fwd), ""))
            soft[v] = (label, ", ".join(n for n in dict.fromkeys([*seen_note.split(", "), note]) if n))
        ids = sorted(scores, key=scores.get, reverse=True)[:NEAR_CANDIDATES]
        if not ids and not (direct_out | direct_in | set(soft)):
            return []
        lookups = list(dict.fromkeys(ids + sorted(direct_out | direct_in) + sorted(soft)))
        rows = {x["id"]: x for x in self.db.execute(
            "SELECT * FROM symbols WHERE id IN (" + ",".join("?" * len(lookups)) + ")", lookups)}
        direct = sorted(direct_out | direct_in, key=lambda i: -scores.get(i, 0))
        far = [i for i in ids if i not in direct_out and i not in direct_in and i not in soft]
        picked, used = [], 0
        for i in direct + sorted(soft, key=lambda i: -scores.get(i, 0)) + far:
            x = rows.get(i)
            if x is None or (x["kind"] in ("module", "impl", "namespace") and i not in direct and i not in soft):
                continue
            c = toks(f"{x['file']}:{x['line_start']}  {x['sig'] or x['qualname']}\n")
            if used + c > budget:
                continue
            if i in direct_out:
                rel, note = "uses", ""
            elif i in direct_in:  # a test that calls it is a test of it, not just a user
                rel, note = ("tested by" if is_test_path(x["file"]) and not is_test_path(r["file"]) else "used by"), ""
            else:
                rel, note = soft.get(i, ("related", ""))
            picked.append((rel, x, note))
            used += c
        return picked

    @synchronized
    def related(self, symbol: str, max_tokens: int = 800) -> str:
        r, err = self._one(symbol)
        if err:
            return err
        picked = self._related(r, max(max_tokens, 0))
        return "\n".join(f"{rel:8} {x['file']}:{x['line_start']}  {x['sig'] or x['qualname']}"
                         + (f"  [{note}]" if note else "") for rel, x, note in picked) or "(isolated symbol)"

    @synchronized
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
            if p and p["kind"] not in ("module", "namespace"):
                parts.append(f"## inside\n{p['file']}:{p['line_start']}  {p['sig'] or p['qualname']}")
        remaining = max_tokens - toks("\n".join(parts)) - 20
        picked = self._related(r, max(remaining, 0))
        for rel, title in (("uses", "## uses"), ("used by", "## used by"), ("tested by", "## tested by"),
                           ("tests", "## tests"), ("key", "## shares a string key"),
                           ("similar", "## similar (by meaning, not linked by calls)"),
                           ("related", "## related (further out, ranked by closeness)")):
            group = [(x, note) for k, x, note in picked if k == rel]
            if group:
                parts.append(title)
                for x, note in group:
                    parts.append(f"{x['file']}:{x['line_start']}  {x['sig'] or x['qualname']}"
                                 + (f"  [{note}]" if note else ""))
                    if x["file"] != r["file"]:
                        used_files.append(x["file"])
        body = "\n".join(parts)
        naive = self._file_tokens(used_files)
        return (f"{body}\n[~{toks(body)} tokens for this symbol and its neighbours, vs ~{naive} "
                f"to read {len(set(used_files))} whole file(s) - the neighbours listed are "
                f"signatures only, not full bodies]")

    @synchronized
    def _resolve_file(self, term: str) -> str | None:
        """Prefer an exact path, then a unique suffix, then the shortest unique suffix match."""
        t = like(term.strip().lstrip("./"))
        exact = self.db.execute("SELECT path FROM files WHERE path=?", (t,)).fetchone()
        if exact:
            return exact[0]
        rows = self.db.execute("SELECT path FROM files WHERE path LIKE ? ESCAPE '\\' "
                               "ORDER BY LENGTH(path), path", (_suffix(t),)).fetchall()
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0][0]
        tail = "/" + t.rpartition("/")[2]
        narrowed = [r[0] for r in rows if r[0].endswith(tail)] if "/" in t else []
        if len(narrowed) == 1:
            return narrowed[0]
        return rows[0][0]

    @synchronized
    def deps(self, file: str) -> str:
        f = self._resolve_file(file)
        self.status = "ok" if f else "missing"
        if f is None:
            return f"No indexed file matches '{file}'."
        imports = [r[0] for r in self.db.execute("SELECT dst FROM imports WHERE file=?", (f,))]
        importers = [r[0] for r in self.db.execute("SELECT file FROM imports WHERE dst=?", (f,))]
        out = (f"{f}\nimports: {', '.join(sorted(imports)) or '-'}\n"
               f"imported by: {', '.join(sorted(importers)) or '-'}")
        partners = sorted(self._fl.get(f, ()), key=lambda t: (-t[1], t[0]))
        if partners:  # only when the repo has git history
            out += "\nco-changes with: " + ", ".join(f"{g} ({w:.2f})" for g, w in partners)
        return out

    @synchronized
    def stats(self) -> str:
        db = self.db

        def q(sql):
            return db.execute(sql).fetchone()[0]

        conf = dict(db.execute("SELECT conf, COUNT(*) FROM edges GROUP BY conf").fetchall())
        return (f"{q('SELECT COUNT(*) FROM files')} files, {q('SELECT COUNT(*) FROM symbols')} symbols, "
                f"{q('SELECT COUNT(*) FROM edges')} edges ({conf.get('extracted', 0)} extracted, "
                f"{conf.get('inferred', 0)} inferred, {conf.get('ambiguous', 0)} ambiguous), "
                f"{q('SELECT COUNT(*) FROM imports')} file imports, "
                f"{q('SELECT COUNT(DISTINCT community) FROM symbols WHERE community>=0')} communities")

    @synchronized
    def report(self, max_tokens: int = 1500) -> str:
        """Write .codegraph/GRAPH_REPORT.md and graph.html; return the report text."""
        from .report import short, write_all

        text, paths = write_all(self.db, self.root)
        cap = max_tokens * CHARS_PER_TOKEN
        if len(text) > cap:
            text = text[:cap].rsplit("\n", 1)[0] + "\n... truncated (full report in the file)"
        return text + "\nwritten: " + ", ".join(short(p, self.root) for p in paths)

    def review(self, base: str = "", max_tokens: int = 1500) -> str:
        """Impact of the current git diff (against `base`, default: where the branch left main)."""
        from .indexer import index_repo
        from .review import review

        with self._lock:
            index_repo(self.root)  # the diff is the working tree, so the index must be too
            self.reload()
            return review(self, base, max_tokens)
