"""`codegraph ui`: an interactive, local-only graph explorer served from the existing index.

Stdlib only. The page (ui.html) starts from the file-level network and lets you drag nodes, expand a
file into its symbols, pull in a symbol's neighbours and read source, all through the JSON API below.
"""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from collections import Counter, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .indexer import index_repo
from .queries import Graph

OVERVIEW_FILES = 500  # a force layout in a browser canvas stays smooth up to roughly this many nodes
IMPACT_LIMIT = 80
HIDDEN = ("module", "impl", "namespace")


def _sym(r) -> dict:
    return {"id": r["id"], "name": r["name"], "qn": r["qualname"], "kind": r["kind"], "file": r["file"],
            "l1": r["line_start"], "l2": r["line_end"], "rank": r["rank"], "c": r["community"]}


class Api:
    """Read-only queries over one Graph. Every method returns JSON-serialisable data."""

    def __init__(self, graph: Graph):
        self.g = graph

    def overview(self, limit: int = OVERVIEW_FILES) -> dict:
        with self.g._lock:
            db = self.g.db
            files = {}
            for r in db.execute(
                    f"SELECT file, COUNT(*) n, SUM(rank) rank FROM symbols WHERE kind NOT IN {HIDDEN!r} GROUP BY file"):
                files[r["file"]] = {"id": r["file"], "n": r["n"], "rank": r["rank"] or 0.0, "loc": 0, "c": -1}
            for r in db.execute("SELECT path, loc, lang FROM files"):
                if r["path"] in files:
                    files[r["path"]].update(loc=r["loc"] or 0, lang=r["lang"])
            votes: dict[str, Counter] = defaultdict(Counter)
            for r in db.execute(f"SELECT file, community FROM symbols WHERE community>=0 AND kind NOT IN {HIDDEN!r}"):
                votes[r["file"]][r["community"]] += 1
            for f, v in votes.items():
                if f in files:
                    files[f]["c"] = v.most_common(1)[0][0]
            keep = {f["id"] for f in sorted(files.values(), key=lambda f: -f["rank"])[:limit]}
            pair: Counter = Counter()
            for r in db.execute("SELECT a.file fa, b.file fb FROM edges e JOIN symbols a ON a.id=e.src "
                                "JOIN symbols b ON b.id=e.dst WHERE a.file != b.file"):
                if r["fa"] in keep and r["fb"] in keep:
                    pair[(r["fa"], r["fb"])] += 1
            links = [{"s": a, "t": b, "w": w, "k": "call"} for (a, b), w in pair.items()]
            seen = set(pair)
            for r in db.execute("SELECT file, dst FROM imports"):
                k = (r["file"], r["dst"])
                if r["file"] in keep and r["dst"] in keep and k not in seen and r["file"] != r["dst"]:
                    seen.add(k)
                    links.append({"s": k[0], "t": k[1], "w": 1, "k": "import"})
            return {"files": [files[f] for f in keep], "links": links, "total": len(files)}

    def file(self, path: str) -> dict:
        """Symbols of one file plus every edge that touches them (the far end is described too)."""
        with self.g._lock:
            db = self.g.db
            syms = [_sym(r) for r in db.execute(
                f"SELECT * FROM symbols WHERE file=? AND kind NOT IN {HIDDEN!r} ORDER BY line_start", (path,))]
            ids = [s["id"] for s in syms]
            return {"symbols": syms, **self._edges_of(ids)}

    def _edges_of(self, ids) -> dict:
        db = self.g.db
        edges, far = [], {}
        if not ids:
            return {"edges": edges, "far": []}
        mark = ",".join("?" * len(ids))
        mine = set(ids)
        for r in db.execute(f"SELECT src, dst, kind, conf FROM edges WHERE src IN ({mark}) OR dst IN ({mark})",
                            (*ids, *ids)):
            edges.append({"s": r["src"], "t": r["dst"], "k": r["kind"], "q": r["conf"]})
            for o in (r["src"], r["dst"]):
                if o not in mine:
                    far[o] = None
        if far:
            m2 = ",".join("?" * len(far))
            far_rows = db.execute(f"SELECT * FROM symbols WHERE id IN ({m2}) AND kind NOT IN {HIDDEN!r}",
                                  tuple(far)).fetchall()
            ok = {r["id"] for r in far_rows}
            edges = [e for e in edges if (e["s"] in mine or e["s"] in ok) and (e["t"] in mine or e["t"] in ok)]
            return {"edges": edges, "far": [_sym(r) for r in far_rows]}
        return {"edges": edges, "far": []}

    def symbol(self, sid: int) -> dict:
        with self.g._lock:
            db = self.g.db
            r = db.execute("SELECT * FROM symbols WHERE id=?", (sid,)).fetchone()
            if r is None:
                return {"error": "unknown symbol"}
            out = _sym(r)
            out.update(sig=r["sig"] or "", doc=r["doc"] or "", source=self.g._read(r)[:12000])
            nb = self._edges_of([sid])
            out["edges"], out["far"] = nb["edges"], nb["far"]
            return out

    def impact(self, sid: int, depth: int = 3) -> dict:
        """Transitive callers (what breaks if this changes), nearest first."""
        with self.g._lock:
            seen, q, order = {sid}, deque([(sid, 0)]), []
            while q and len(order) < IMPACT_LIMIT:
                u, d = q.popleft()
                if d >= depth:
                    continue
                for v in self.g._in.get(u, ()):
                    if v not in seen:
                        seen.add(v)
                        order.append(v)
                        q.append((v, d + 1))
            order = order[:IMPACT_LIMIT]
            if not order:
                return {"symbols": [], "edges": []}
            mark = ",".join("?" * len(order))
            rows = self.g.db.execute(f"SELECT * FROM symbols WHERE id IN ({mark}) AND kind NOT IN {HIDDEN!r}",
                                     tuple(order)).fetchall()
            ids = {r["id"] for r in rows} | {sid}
            edges = [{"s": u, "t": v, "k": "call", "q": "inferred"} for u in ids for v in self.g._out.get(u, ())
                     if v in ids]
            return {"symbols": [_sym(r) for r in rows], "edges": edges}

    def search(self, q: str) -> dict:
        with self.g._lock:
            rows = self.g._search_rows(q, 25) if q.strip() else []
            return {"symbols": [_sym(r) for r in rows if r["kind"] not in HIDDEN]}


class Handler(BaseHTTPRequestHandler):
    api: Api
    port: int
    page: bytes

    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # DNS-rebinding guard: the source of your repo must only reach pages served from this host:port
        if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
            return self._send(b"forbidden", "text/plain", 403)
        u = urlparse(self.path)
        a = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._send(self.page, "text/html; charset=utf-8")
            routes = {
                "/api/overview": lambda: self.api.overview(int(a.get("limit", OVERVIEW_FILES))),
                "/api/file": lambda: self.api.file(a["path"]),
                "/api/symbol": lambda: self.api.symbol(int(a["id"])),
                "/api/impact": lambda: self.api.impact(int(a["id"]), int(a.get("depth", 3))),
                "/api/search": lambda: self.api.search(a.get("q", "")),
            }
            if u.path not in routes:
                return self._send(b"not found", "text/plain", 404)
            return self._send(json.dumps(routes[u.path]()).encode(), "application/json")
        except (KeyError, ValueError):
            return self._send(b'{"error":"bad request"}', "application/json", 400)


def serve(root: str, port: int = 0, open_browser: bool = True) -> None:
    print(f"codegraph: indexing {root} ...", file=sys.stderr, flush=True)
    index_repo(root)
    graph = Graph(root)
    Handler.api = Api(graph)
    Handler.page = resources.files("codegraph").joinpath("ui.html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)  # loopback only: never expose source on the LAN
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"codegraph ui: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        graph.close()
