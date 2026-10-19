"""`codegraph ui`: an interactive, local-only graph explorer served from the existing index.

Stdlib only. The page (ui.html) starts from a community-level map and lets you open communities into
files and symbols, re-cluster at another resolution, colour the graph by a graph-theory "lens"
(bottlenecks, cycles, k-core, ...) and read source, all through the JSON API below.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from collections import Counter, defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from . import graphalgo as ga
from .community import CONF_WEIGHT
from .indexer import index_repo
from .queries import Graph
from .report import _area

OVERVIEW_PIECES = 700  # a force layout in a browser canvas stays smooth up to roughly this many nodes
IMPACT_LIMIT = 80
HIDDEN = ("module", "impl", "namespace")
CODE_KINDS = ("function", "method", "class", "struct", "trait", "interface", "type")
RES_MIN, RES_MAX = 0.1, 5.0
SNAPSHOT = "community_snapshot.json"

LENSES = {
    "importance": ("Importance", "PageRank: symbols many others lean on."),
    "bottlenecks": ("Bottlenecks", "Betweenness centrality: symbols that many shortest paths run through. "
                    "Changing or breaking these affects the most routes."),
    "cycles": ("Cycles", "Strongly connected components: symbols or files that depend on each other in a loop."),
    "layers": ("Layers", "Topological depth of what is on screen: callers on top, what they depend on below."),
    "coupling": ("Coupling", "Share of a symbol's links that leave its community. Marked symbols have more "
                 "links into another community than into their own: candidates to move."),
    "kcore": ("Core / periphery", "k-core number: the dense kernel everything is wired into versus leaf code."),
    "deadcode": ("Dead code", "Functions, methods and classes nothing calls and no unresolved reference "
                 "names. Candidates only: dynamic use and entry points are invisible to static analysis."),
    "hotspots": ("Hotspots", "Files that change together with many others (git co-change) weighted by "
                 "importance: central code that is also volatile."),
}


def _is_test(path: str) -> bool:
    p = path.lower()
    return ("/tests/" in "/" + p or "/test/" in "/" + p or "__tests__" in p or p.rsplit("/", 1)[-1].startswith("test_")
            or p.endswith(("_test.py", "_test.go", ".test.ts", ".test.tsx", ".test.js", ".spec.ts", ".spec.js")))


class Api:
    """Read-only queries over one Graph. Every method returns JSON-serialisable data."""

    def __init__(self, graph: Graph):
        self.g = graph
        self._parts: dict[float, dict[int, int]] = {}
        self._commmeta: dict[float, dict] = {}
        self._lens: dict[tuple, dict] = {}
        self._load()

    # ---- in-memory copy of the structural graph (small: ids and a few strings per symbol) -------------
    def _load(self):
        with self.g._lock:
            db = self.g.db
            self.S = {r["id"]: {"id": r["id"], "file": r["file"], "name": r["name"], "qn": r["qualname"],
                                "kind": r["kind"], "l1": r["line_start"], "l2": r["line_end"], "rank": r["rank"] or 0.0}
                      for r in db.execute(f"SELECT * FROM symbols WHERE kind NOT IN {HIDDEN!r}")}
            self.E = [(r["src"], r["dst"], r["kind"], r["conf"]) for r in db.execute("SELECT src, dst, kind, conf FROM edges")
                      if r["src"] in self.S and r["dst"] in self.S]
            self.files = {r["path"]: (r["loc"] or 0, r["lang"]) for r in db.execute("SELECT path, loc, lang FROM files")}
            self.imports = [(r["file"], r["dst"]) for r in db.execute("SELECT file, dst FROM imports")]
            self.raw_names = {r[0] for r in db.execute("SELECT DISTINCT name FROM raw_edges")}
            self.cochange = [(r["a"], r["b"], r["weight"]) for r in db.execute(
                "SELECT a, b, weight FROM file_links WHERE kind='cochange'")]
        self.out: dict[int, set[int]] = defaultdict(set)
        self.inn: dict[int, set[int]] = defaultdict(set)
        self.by_src: dict[int, list] = defaultdict(list)
        self.by_dst: dict[int, list] = defaultdict(list)
        for s, d, k, c in self.E:
            self.out[s].add(d)
            self.inn[d].add(s)
            self.by_src[s].append((d, k, c))
            self.by_dst[d].append((s, k, c))
        self.wedges = [(s, d, CONF_WEIGHT.get(c, 0.6)) for s, d, _, c in self.E]

    # ---- communities ---------------------------------------------------------------------------------
    @staticmethod
    def _res(res) -> float:
        try:
            return round(min(RES_MAX, max(RES_MIN, float(res))), 2)
        except (TypeError, ValueError):
            return 1.0

    def partition(self, res=1.0) -> dict[int, int]:
        """{symbol id: community} from Louvain at this resolution. Symbols without edges are absent (-1)."""
        res = self._res(res)
        if res not in self._parts:
            levels = ga.louvain(self.S, self.wedges, res)
            part = ga.fold_small(levels[-1], self.wedges, 3) if levels else {}
            if len(self._parts) >= 8:  # slider drags must not grow memory without bound
                self._parts.pop(next(iter(self._parts)))
                self._commmeta.pop(next(iter(self._commmeta)), None)
            self._parts[res] = part
        return self._parts[res]

    def comms(self, res=1.0) -> dict[int, dict]:
        res = self._res(res)
        if res in self._commmeta:
            return self._commmeta[res]
        part = self.partition(res)
        members: dict[int, list[int]] = defaultdict(list)
        for sid, c in part.items():
            members[c].append(sid)
        inside, touching = Counter(), Counter()
        for s, d, w in self.wedges:
            cs, cd = part.get(s, -1), part.get(d, -1)
            if cs >= 0:
                touching[cs] += w
            if cd >= 0 and cd != cs:
                touching[cd] += w
            if cs >= 0 and cs == cd:
                inside[cs] += w
        out = {}
        for c, ids in members.items():
            ids.sort(key=lambda i: (-self.S[i]["rank"], i))
            files = Counter(self.S[i]["file"] for i in ids)
            area = Counter(_area(self.S[i]["file"]) for i in ids).most_common(1)[0][0]
            names = [self.S[i]["name"] for i in ids if self.S[i]["kind"] in CODE_KINDS][:2] or \
                    [self.S[i]["name"] for i in ids[:2]]
            out[c] = {"id": c, "n": len(ids), "files": len(files), "area": area, "label": " · ".join(names),
                      "top": [self.S[i]["qn"] for i in ids[:5]], "rank": sum(self.S[i]["rank"] for i in ids),
                      "cohesion": round(inside[c] / touching[c], 3) if touching[c] else 1.0}
        self._commmeta[res] = out
        return out

    def _s(self, sid: int, res) -> dict:
        s = self.S[sid]
        return {**s, "c": self.partition(res).get(sid, -1)}

    # ---- views ---------------------------------------------------------------------------------------
    def overview(self, res=1.0, limit: int = OVERVIEW_PIECES) -> dict:
        """The graph of "pieces" (one file's symbols inside one community). The page derives community and
        file level views from it by merging, so one payload serves every zoom level."""
        part = self.partition(res)
        pieces: dict[tuple, dict] = {}
        for sid, s in self.S.items():
            p = pieces.setdefault((s["file"], part.get(sid, -1)), {"n": 0, "rank": 0.0})
            p["n"] += 1
            p["rank"] += s["rank"]
        keep = set(sorted(pieces, key=lambda k: (-pieces[k]["rank"], k))[:limit])
        agg: dict[tuple, list] = defaultdict(lambda: [0, 0])
        for s, d, _, conf in self.E:
            a, b = (self.S[s]["file"], part.get(s, -1)), (self.S[d]["file"], part.get(d, -1))
            if a != b and a in keep and b in keep:
                x = agg[(a, b)]
                x[0] += 1
                x[1] += conf == "extracted"
        key = lambda k: f"{k[0]}|{k[1]}"  # noqa: E731
        links = [{"s": key(a), "t": key(b), "w": w, "e": e, "k": "call"} for (a, b), (w, e) in agg.items()]
        main = {}  # imports are file-level: attach them to the biggest piece of each file
        for (f, c), p in pieces.items():
            if (f, c) in keep and (f not in main or p["n"] > pieces[main[f]]["n"]):
                main[f] = (f, c)
        seen = {(a, b) for (a, b) in agg}
        for f, dst in self.imports:
            if f in main and dst in main and f != dst and (main[f], main[dst]) not in seen:
                seen.add((main[f], main[dst]))
                links.append({"s": key(main[f]), "t": key(main[dst]), "w": 1, "e": 1, "k": "import"})
        return {"pieces": [{"k": key(k), "f": k[0], "c": k[1], "n": pieces[k]["n"], "rank": pieces[k]["rank"],
                            "loc": self.files.get(k[0], (0, None))[0]} for k in keep],
                "links": links, "comms": list(self.comms(res).values()),
                "total": len({f for f, _ in pieces}), "pieces_total": len(pieces), "res": self._res(res)}

    def _edges_of(self, ids, res) -> dict:
        mine = set(ids)
        edges, far = [], set()
        for sid in ids:
            for d, k, c in self.by_src.get(sid, ()):
                edges.append({"s": sid, "t": d, "k": k, "q": c})
                far.add(d)
            for s, k, c in self.by_dst.get(sid, ()):
                if s not in mine:  # edges between two of `ids` were already added from their source
                    edges.append({"s": s, "t": sid, "k": k, "q": c})
                    far.add(s)
        return {"edges": edges, "far": [self._s(i, res) for i in sorted(far - mine)]}

    def file(self, path: str, c=None, res=1.0) -> dict:
        """Symbols of one file (only those in community `c` if given) plus every edge that touches them."""
        part = self.partition(res)
        ids = sorted((i for i, s in self.S.items() if s["file"] == path
                      and (c is None or part.get(i, -1) == c)), key=lambda i: (self.S[i]["l1"], i))
        return {"symbols": [self._s(i, res) for i in ids], **self._edges_of(ids, res)}

    def symbol(self, sid: int, res=1.0) -> dict:
        if sid not in self.S:
            return {"error": "unknown symbol"}
        with self.g._lock:
            r = self.g.db.execute("SELECT * FROM symbols WHERE id=?", (sid,)).fetchone()
            out = self._s(sid, res)
            out.update(sig=r["sig"] or "", doc=r["doc"] or "", source=self.g._read(r)[:12000])
        out.update(self._edges_of([sid], res))
        cm = self.comms(res).get(out["c"])
        out["clabel"] = cm["label"] if cm else ""
        return out

    def impact(self, sid: int, depth: int = 3, res=1.0) -> dict:
        """Transitive callers (what breaks if this changes), nearest first."""
        seen, q, order = {sid}, deque([(sid, 0)]), []
        while q and len(order) < IMPACT_LIMIT:
            u, d = q.popleft()
            if d >= depth:
                continue
            for v in sorted(self.inn.get(u, ())):
                if v not in seen:
                    seen.add(v)
                    order.append(v)
                    q.append((v, d + 1))
        order = order[:IMPACT_LIMIT]
        ids = set(order) | {sid}
        return {"symbols": [self._s(i, res) for i in order],
                "edges": [{"s": u, "t": v, "k": "call", "q": "inferred"} for u in ids for v in self.out.get(u, ()) if v in ids]}

    def search(self, q: str, res=1.0) -> dict:
        with self.g._lock:
            rows = self.g._search_rows(q, 25) if q.strip() else []
        return {"symbols": [self._s(r["id"], res) for r in rows if r["id"] in self.S]}

    def community(self, cid: int, res=1.0) -> dict:
        """Everything worth knowing about one community: size, cohesion, top symbols, files, neighbours."""
        part, meta = self.partition(res), self.comms(res)
        if cid not in meta:
            return {"error": "unknown community"}
        ids = [i for i, c in part.items() if c == cid]
        ids.sort(key=lambda i: (-self.S[i]["rank"], i))
        files = Counter(self.S[i]["file"] for i in ids)
        inner = outer = 0
        nb: Counter = Counter()
        for s, d, _w in self.wedges:
            cs, cd = part.get(s, -1), part.get(d, -1)
            if cs == cid and cd == cid:
                inner += 1
            elif cs == cid and cd >= 0:
                outer += 1
                nb[cd] += 1
            elif cd == cid and cs >= 0:
                outer += 1
                nb[cs] += 1
        return {**meta[cid], "internal": inner, "external": outer,
                "symbols": [self._s(i, res) for i in ids[:12]],
                "file_list": [{"path": f, "n": n} for f, n in files.most_common(12)],
                "neighbours": [{"id": c, "label": meta[c]["label"], "w": w} for c, w in nb.most_common(8) if c in meta]}

    # ---- lenses --------------------------------------------------------------------------------------
    def lens(self, name: str, res=1.0) -> dict:
        res = self._res(res)
        if name not in LENSES:
            return {"error": "unknown lens"}
        key = (name, res if name == "coupling" else 0)
        if key not in self._lens:
            self._lens[key] = self._compute_lens(name, res)
        return self._lens[key]

    def _pack(self, name, sym: dict, res, agg="max", items=None, note="") -> dict:
        part = self.partition(res)
        piece: dict[str, float] = {}
        for sid, v in sym.items():
            k = f"{self.S[sid]['file']}|{part.get(sid, -1)}"
            piece[k] = piece.get(k, 0) + v if agg == "sum" else max(piece.get(k, 0), v)
        hi = max(sym.values(), default=0)
        return {"name": name, "title": LENSES[name][0], "desc": LENSES[name][1], "agg": agg, "hi": hi, "note": note,
                "sym": {i: round(v, 5) for i, v in sym.items() if v > 0}, "piece": {k: round(v, 5) for k, v in piece.items()},
                "items": items or []}

    def _item(self, sid, sub="") -> dict:
        s = self.S[sid]
        return {"t": s["qn"], "s": sub or s["file"], "id": sid}

    def _undirected(self) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = defaultdict(set)
        for s, d, _, _ in self.E:
            if s != d:
                adj[s].add(d)
                adj[d].add(s)
        return adj

    def _compute_lens(self, name, res) -> dict:
        S, part = self.S, self.partition(res)
        if name == "importance":
            sym = {i: s["rank"] for i, s in S.items()}
            top = sorted(sym, key=lambda i: -sym[i])[:12]
            return self._pack(name, sym, res, "sum", [self._item(i) for i in top])
        if name == "layers":
            return {"name": name, "title": LENSES[name][0], "desc": LENSES[name][1], "client": True,
                    "sym": {}, "piece": {}, "items": [], "hi": 0, "agg": "max"}
        if name == "bottlenecks":
            adj = self._undirected()
            n, m = len(adj), len(self.E)
            pivots = max(30, min(n, int(3_000_000 / (n + m + 1)))) if n else None
            bc = ga.betweenness(adj, pivots)
            top = sorted(bc, key=lambda i: -bc[i])[:12]
            note = f"estimated from {pivots} of {n} start points" if pivots and pivots < n else ""
            return self._pack(name, bc, res, "max", [self._item(i, f"{bc[i]:.0f} paths") for i in top if bc[i] > 0], note)
        if name == "cycles":
            comps = [c for c in ga.sccs(S, self.out) if len(c) > 1]
            sym = {i: len(c) for c in comps for i in c}
            fout: dict[str, set[str]] = defaultdict(set)
            for s, d, _, _ in self.E:
                if S[s]["file"] != S[d]["file"]:
                    fout[S[s]["file"]].add(S[d]["file"])
            fcomps = [c for c in ga.sccs({S[i]["file"] for i in S}, fout) if len(c) > 1]
            fsize = {f: len(c) for c in fcomps for f in c}
            items = [{"t": f"{len(c)} files call each other in a loop", "s": ", ".join(sorted(c)[:4]) + (" ..." if len(c) > 4 else ""),
                      "f": sorted(c)[0]} for c in sorted(fcomps, key=lambda c: -len(c))[:6]]
            items += [{"t": f"{len(c)} symbols in a call loop", "s": ", ".join(S[i]["qn"] for i in sorted(c)[:3]) + (" ..." if len(c) > 3 else ""),
                       "id": sorted(c)[0]} for c in sorted(comps, key=lambda c: -len(c))[:8]]
            lens = self._pack(name, sym, res, "max", items)
            for k in lens["piece"]:  # a file in a file-level loop is marked even if no single call loops
                f = k.rsplit("|", 1)[0]
                lens["piece"][k] = max(lens["piece"][k], fsize.get(f, 0))
            for f, size in fsize.items():
                for k in {f"{f}|{part.get(i, -1)}" for i, s in S.items() if s["file"] == f}:
                    lens["piece"][k] = max(lens["piece"].get(k, 0), size)
            lens["hi"] = max([lens["hi"], *fsize.values()])
            lens["note"] = "" if items else "No cycles found."
            return lens
        if name == "coupling":
            w_in: Counter = Counter()
            w_to: dict[int, Counter] = defaultdict(Counter)
            for s, d, w in self.wedges:
                cs, cd = part.get(s, -1), part.get(d, -1)
                if cs < 0 or cd < 0:
                    continue
                for a, ca, cb in ((s, cs, cd), (d, cd, cs)):
                    if ca == cb:
                        w_in[a] += w
                    else:
                        w_to[a][cb] += w
            sym, items = {}, []
            meta = self.comms(res)
            for i in set(w_in) | set(w_to):
                out_w = sum(w_to[i].values())
                tot = w_in[i] + out_w
                if tot < 2:
                    continue
                sym[i] = out_w / tot
                best = max(w_to[i], key=lambda c: (w_to[i][c], -c), default=None)
                if best is not None and w_to[i][best] > w_in[i]:
                    sym[i] = 1.0
                    items.append((w_to[i][best] - w_in[i], self._item(
                        i, f"belongs more with “{meta[best]['label']}” than with “{meta[part[i]]['label']}”")))
            items.sort(key=lambda t: -t[0])
            return self._pack(name, sym, res, "max", [it for _, it in items[:15]])
        if name == "kcore":
            core = ga.kcore(self._undirected())
            hi = max(core.values(), default=0)
            top = sorted((i for i in core if core[i] == hi), key=lambda i: -S[i]["rank"])[:12]
            return self._pack(name, {i: c for i, c in core.items()}, res, "max",
                              [self._item(i, f"{hi}-core: {S[i]['file']}") for i in top])
        if name == "deadcode":
            sym = {i: 1.0 for i, s in S.items()
                   if s["kind"] in ("function", "method", "class") and not self.inn.get(i) and not _is_test(s["file"])
                   and not s["name"].startswith(("__", "test", "main", "Test")) and s["name"] not in self.raw_names}
            items = [self._item(i) for i in sorted(sym, key=lambda i: (S[i]["file"], S[i]["l1"]))[:40]]
            return self._pack(name, sym, res, "sum", items, f"{len(sym)} candidates")
        if name == "hotspots":
            coch: Counter = Counter()
            for a, b, w in self.cochange:
                coch[a] += w
                coch[b] += w
            if not coch:
                return {**self._pack(name, {}, res), "note": "No git history indexed: co-change data is missing."}
            frank: Counter = Counter()
            for s in S.values():
                frank[s["file"]] += s["rank"]
            score = {f: coch[f] * frank[f] for f in coch if f in frank}
            top = max(score.values(), default=0) or 1.0
            score = {f: v / top for f, v in score.items()}
            sym = {i: score[s["file"]] for i, s in S.items() if s["file"] in score}
            items = [{"t": f, "s": f"changes with {coch[f]:.1f} weighted files", "f": f}
                     for f in sorted(score, key=lambda f: -score[f])[:15]]
            return self._pack(name, sym, res, "max", items)
        return {"error": "unknown lens"}

    # ---- overlays and tools --------------------------------------------------------------------------
    def path(self, a: int, b: int, res=1.0) -> dict:
        if a not in self.S or b not in self.S:
            return {"error": "unknown symbol"}
        ids = ga.shortest_path(self.out, self.inn, a, b)
        edges = []
        for u, v in zip(ids, ids[1:], strict=False):
            hit = next(((u, v, k, c) for d, k, c in self.by_src.get(u, ()) if d == v), None) or \
                next(((v, u, k, c) for d, k, c in self.by_src.get(v, ()) if d == u), None)
            if hit:
                edges.append({"s": hit[0], "t": hit[1], "k": hit[2], "q": hit[3]})
        return {"found": bool(ids), "symbols": [self._s(i, res) for i in ids], "edges": edges}

    def surprises(self, res=1.0) -> dict:
        """Certain edges between different communities *and* different folders, between low-degree symbols:
        the links a reader would not have guessed from the file layout."""
        part, S = self.partition(res), self.S
        scored = []
        for s, d, k, conf in self.E:
            if conf != "extracted" or part.get(s, -1) < 0 or part.get(d, -1) < 0 or part[s] == part[d]:
                continue
            if _area(S[s]["file"]) == _area(S[d]["file"]):
                continue
            scored.append((len(self.out[s]) + len(self.inn[s]) + len(self.out[d]) + len(self.inn[d]), s, d, k))
        scored.sort()
        top = scored[:12]
        ids = sorted({x for _, s, d, _ in top for x in (s, d)})
        return {"symbols": [self._s(i, res) for i in ids],
                "edges": [{"s": s, "t": d, "k": k, "q": "extracted"} for _, s, d, k in top]}

    def _snapshot_path(self):
        return self.g.root / ".codegraph" / SNAPSHOT

    def snapshot_save(self, res=1.0) -> dict:
        part, meta = self.partition(res), self.comms(res)
        data = {"res": self._res(res), "at": int(time.time()),
                "members": {f"{self.S[i]['file']}::{self.S[i]['qn']}": c for i, c in part.items()},
                "labels": {str(c): m["label"] for c, m in meta.items()}}
        self._snapshot_path().write_text(json.dumps(data))
        return {"saved": len(part), "at": data["at"]}

    def diff(self, res=1.0) -> dict:
        """Compare the current partition with the saved snapshot: which symbols changed community."""
        try:
            snap = json.loads(self._snapshot_path().read_text())
        except (OSError, ValueError):
            return {"error": "no snapshot saved yet"}
        part, meta = self.partition(res), self.comms(res)
        cur = {f"{self.S[i]['file']}::{self.S[i]['qn']}": (i, c) for i, c in part.items()}
        old = snap["members"]
        common = cur.keys() & old.keys()
        overlap: dict[int, Counter] = defaultdict(Counter)
        for k in common:
            overlap[cur[k][1]][old[k]] += 1
        match = {c: max(cnt, key=lambda o: (cnt[o], -o)) for c, cnt in overlap.items()}
        labels = snap["labels"]
        moved = [{"id": cur[k][0], "qn": self.S[cur[k][0]]["qn"], "file": self.S[cur[k][0]]["file"],
                  "from": labels.get(str(old[k]), "?"), "to": meta.get(cur[k][1], {}).get("label", "?")}
                 for k in sorted(common) if match.get(cur[k][1]) != old[k]]
        return {"at": snap["at"], "snap_res": snap["res"], "res": self._res(res), "new": len(cur.keys() - old.keys()),
                "gone": len(old.keys() - cur.keys()), "moved_count": len(moved), "moved": moved[:200]}


class Handler(BaseHTTPRequestHandler):
    api: Api
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

    def _guard(self) -> bool:
        """DNS-rebinding guard: the source of your repo must only reach pages served from this host."""
        host = self.headers.get("Host", "").rsplit(":", 1)[0]
        origin = self.headers.get("Origin")
        if host not in ("127.0.0.1", "localhost") or (origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost")):
            self._send(b"forbidden", "text/plain", 403)
            return False
        return True

    def _json(self, fn):
        try:
            return self._send(json.dumps(fn()).encode(), "application/json")
        except (KeyError, ValueError):
            return self._send(b'{"error":"bad request"}', "application/json", 400)

    def do_GET(self):
        if not self._guard():
            return None
        u = urlparse(self.path)
        a = {k: v[0] for k, v in parse_qs(u.query).items()}
        res = a.get("res", 1.0)
        api = self.api
        routes = {
            "/api/overview": lambda: api.overview(res, int(a.get("limit", OVERVIEW_PIECES))),
            "/api/file": lambda: api.file(a["path"], int(a["c"]) if "c" in a else None, res),
            "/api/symbol": lambda: api.symbol(int(a["id"]), res),
            "/api/impact": lambda: api.impact(int(a["id"]), int(a.get("depth", 3)), res),
            "/api/search": lambda: api.search(a.get("q", ""), res),
            "/api/community": lambda: api.community(int(a["id"]), res),
            "/api/lens": lambda: api.lens(a["name"], res),
            "/api/path": lambda: api.path(int(a["a"]), int(a["b"]), res),
            "/api/surprises": lambda: api.surprises(res),
            "/api/diff": lambda: api.diff(res),
        }
        if u.path == "/":
            return self._send(self.page, "text/html; charset=utf-8")
        if u.path not in routes:
            return self._send(b"not found", "text/plain", 404)
        return self._json(routes[u.path])

    def do_POST(self):
        if not self._guard():
            return None
        u = urlparse(self.path)
        a = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path != "/api/snapshot":
            return self._send(b"not found", "text/plain", 404)
        return self._json(lambda: self.api.snapshot_save(a.get("res", 1.0)))


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
