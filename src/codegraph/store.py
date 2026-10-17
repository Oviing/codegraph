"""SQLite graph store: symbols, resolved edges, file-level imports, FTS search."""
from __future__ import annotations

import posixpath
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .parser import Parsed
from .rank import pagerank

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, lang TEXT, hash TEXT, loc INT);
CREATE TABLE IF NOT EXISTS symbols(
  id INTEGER PRIMARY KEY, file TEXT, name TEXT, qualname TEXT, kind TEXT,
  line_start INT, line_end INT, sig TEXT, doc TEXT, parent INT, rank REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS raw_edges(src INT, name TEXT, kind TEXT);
CREATE TABLE IF NOT EXISTS edges(src INT, dst INT, kind TEXT);
CREATE TABLE IF NOT EXISTS raw_imports(file TEXT, raw TEXT);
CREATE TABLE IF NOT EXISTS imports(file TEXT, dst TEXT);
CREATE INDEX IF NOT EXISTS ix_sym_name ON symbols(name);
CREATE INDEX IF NOT EXISTS ix_sym_file ON symbols(file);
CREATE INDEX IF NOT EXISTS ix_sym_qual ON symbols(qualname);
CREATE INDEX IF NOT EXISTS ix_e_src ON edges(src);
CREATE INDEX IF NOT EXISTS ix_e_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS ix_re_src ON raw_edges(src);
CREATE INDEX IF NOT EXISTS ix_imp_file ON imports(file);
CREATE INDEX IF NOT EXISTS ix_imp_dst ON imports(dst);
CREATE VIRTUAL TABLE IF NOT EXISTS sym_fts USING fts5(name, words, qualname, doc);
"""

TARGET_KINDS_CALL = {"function", "method", "class", "struct", "type"}
TARGET_KINDS_INHERIT = {"class", "interface", "trait", "struct", "type"}
CLASS_KINDS = {"class", "struct", "trait", "interface", "impl"}
MAX_CANDIDATES = 4  # names defined in more places than this are too ambiguous to link


def split_words(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.migrated = False
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(symbols)")}
        if "rank" not in cols:  # database written by an older version
            self.db.execute("ALTER TABLE symbols ADD COLUMN rank REAL DEFAULT 0")
            self.migrated = True

    def commit(self):
        self.db.commit()

    # ---- files ----
    def file_hashes(self) -> dict[str, str]:
        return {r["path"]: r["hash"] for r in self.db.execute("SELECT path, hash FROM files")}

    def delete_file(self, path: str):
        ids = [r["id"] for r in self.db.execute("SELECT id FROM symbols WHERE file=?", (path,))]
        for i in ids:
            self.db.execute("DELETE FROM raw_edges WHERE src=?", (i,))
            self.db.execute("DELETE FROM sym_fts WHERE rowid=?", (i,))
        self.db.execute("DELETE FROM symbols WHERE file=?", (path,))
        self.db.execute("DELETE FROM raw_imports WHERE file=?", (path,))
        self.db.execute("DELETE FROM files WHERE path=?", (path,))

    def add_file(self, path: str, lang: str, h: str, p: Parsed, loc: int):
        self.db.execute("INSERT INTO files VALUES(?,?,?,?)", (path, lang, h, loc))
        idmap: dict[int, int] = {}
        for s in p.syms:
            parent = idmap.get(s.parent) if s.parent is not None else None
            cur = self.db.execute(
                "INSERT INTO symbols(file,name,qualname,kind,line_start,line_end,sig,doc,parent) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (path, s.name, s.qualname or s.name, s.kind, s.start, s.end, s.sig, s.doc, parent),
            )
            idmap[s.id] = cur.lastrowid
            self.db.execute(
                "INSERT INTO sym_fts(rowid,name,words,qualname,doc) VALUES(?,?,?,?,?)",
                (cur.lastrowid, s.name, split_words(s.name), s.qualname, s.doc),
            )
        for src, name, kind in p.refs:
            self.db.execute("INSERT INTO raw_edges VALUES(?,?,?)", (idmap[src], name, kind))
        for raw in set(p.imports):
            self.db.execute("INSERT INTO raw_imports VALUES(?,?)", (path, raw))

    # ---- resolution ----
    def _resolve_imports(self) -> dict[str, set[str]]:
        db = self.db
        db.execute("DELETE FROM imports")
        files = [r["path"] for r in db.execute("SELECT path FROM files")]
        fset = set(files)
        stems: dict[str, list[str]] = defaultdict(list)
        for f in files:
            stems[posixpath.splitext(posixpath.basename(f))[0]].append(f)
        irows = set()
        for r in db.execute("SELECT file, raw FROM raw_imports"):
            for dst in resolve_import(r["raw"], r["file"], fset, stems):
                if dst != r["file"]:
                    irows.add((r["file"], dst))
        db.executemany("INSERT INTO imports VALUES(?,?)", sorted(irows))
        imported: dict[str, set[str]] = defaultdict(set)
        for f, d in irows:
            imported[f].add(d)
        return imported

    def resolve(self):
        db = self.db
        db.execute("DELETE FROM edges")
        imported = self._resolve_imports()  # needed to disambiguate same-name targets
        by_name: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        file_of: dict[int, str] = {}
        name_of: dict[int, str] = {}
        kind_of: dict[int, str] = {}
        parent_of: dict[int, int | None] = {}
        for r in db.execute("SELECT id, name, file, kind, parent FROM symbols"):
            file_of[r["id"]] = r["file"]
            name_of[r["id"]], kind_of[r["id"]], parent_of[r["id"]] = r["name"], r["kind"], r["parent"]
            if r["kind"] not in ("module", "impl", "namespace"):
                by_name[r["name"]].append((r["id"], r["file"], r["kind"]))

        # Class families (name based): a class, everything it inherits from, and everything inheriting from it.
        bases: dict[str, set[str]] = defaultdict(set)
        subs: dict[str, set[str]] = defaultdict(set)
        for e in db.execute("SELECT src, name FROM raw_edges WHERE kind='inherits'"):
            child = name_of[e["src"]]
            bases[child].add(e["name"])
            subs[e["name"]].add(child)
        family_cache: dict[str, frozenset[str]] = {}

        def family(cn: str) -> frozenset[str]:
            if cn not in family_cache:
                seen_c = {cn}
                for graph in (bases, subs):
                    todo = [cn]
                    while todo:
                        for nxt in graph.get(todo.pop(), ()):
                            if nxt not in seen_c:
                                seen_c.add(nxt)
                                todo.append(nxt)
                family_cache[cn] = frozenset(seen_c)
            return family_cache[cn]

        class_cache: dict[int, str | None] = {}

        def class_of(i: int) -> str | None:
            if i not in class_cache:
                j, found = parent_of.get(i), None
                while j is not None:
                    if kind_of[j] in CLASS_KINDS:
                        found = name_of[j]
                        break
                    j = parent_of.get(j)
                class_cache[i] = found
            return class_cache[i]

        rows = []
        seen = set()
        for e in db.execute("SELECT src, name, kind FROM raw_edges"):
            cands = by_name.get(e["name"])
            if not cands:
                continue
            kind = e["kind"]
            self_cn = class_of(e["src"]) if kind == "calls_self" else None
            if kind == "calls_self" and self_cn is None:
                kind = "calls_attr"  # `self` outside any class: fall back to the generic attribute rule
            is_attr = kind == "calls_attr"
            out_kind = "calls" if kind.startswith("calls") else kind
            allowed = TARGET_KINDS_INHERIT if out_kind == "inherits" else TARGET_KINDS_CALL
            cands = [c for c in cands if c[2] in allowed]
            if not cands:
                continue
            sf = file_of[e["src"]]
            if self_cn is not None:
                # `self.name()` can only mean a method of this class, its bases, or its subclasses.
                # Nothing in that family => inherited from an external base (dict, Exception...) => no edge.
                fam = family(self_cn)
                cands = [c for c in cands if class_of(c[0]) in fam]
                if not cands:
                    continue
            # Narrow by graph evidence, strongest first: same file, imported file, same directory.
            same = [c for c in cands if c[1] == sf]
            if self_cn is not None:
                pass  # class family is already stronger evidence than any import
            elif same:
                cands = same
            else:
                imp = [c for c in cands if c[1] in imported.get(sf, ())]
                if imp:
                    cands = imp
                else:
                    sd = posixpath.dirname(sf)
                    near = [c for c in cands if posixpath.dirname(c[1]) == sd]
                    # `x.name()` with no same-file / imported / same-dir candidate is most likely a call
                    # on some library or builtin type (dict.items, str.get...), so it gets no edge.
                    cands = near if is_attr else (near or cands)
            if not cands:
                continue
            if len(cands) > MAX_CANDIDATES:
                continue
            for c in cands:
                if c[0] == e["src"]:
                    continue  # ignore direct self-recursion noise
                key = (e["src"], c[0], out_kind)
                if key not in seen:
                    seen.add(key)
                    rows.append(key)
        db.executemany("INSERT INTO edges VALUES(?,?,?)", rows)
        self._store_rank([r["id"] for r in db.execute("SELECT id FROM symbols")], [(r[0], r[1]) for r in rows])

    def _store_rank(self, ids, edges):
        scores = pagerank(ids, edges)
        self.db.executemany("UPDATE symbols SET rank=? WHERE id=?", [(v, k) for k, v in scores.items()])


JS_EXT = ["", ".ts", ".tsx", ".js", ".jsx", ".mjs", "/index.ts", "/index.tsx", "/index.js"]


def resolve_import(raw: str, src: str, files: set[str], stems: dict[str, list[str]]) -> list[str]:
    ext = posixpath.splitext(src)[1]
    sdir = posixpath.dirname(src)
    if ext == ".py":
        dots = len(raw) - len(raw.lstrip("."))
        rest = raw.lstrip(".").replace(".", "/")
        if dots:
            base = sdir
            for _ in range(dots - 1):
                base = posixpath.dirname(base)
            cands = [posixpath.normpath(posixpath.join(base, rest)) if rest else base]
            return [c + s for c in cands for s in (".py", "/__init__.py") if c + s in files]
        for f in files:
            if f.endswith("/" + rest + ".py") or f == rest + ".py" or f.endswith("/" + rest + "/__init__.py"):
                return [f]
        return []
    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        if raw.startswith("."):
            base = posixpath.normpath(posixpath.join(sdir, raw))
            for s in JS_EXT:
                if base + s in files:
                    return [base + s]
        return []
    if ext == ".go":
        tail = raw.split("/")[-1]
        return sorted(f for f in files if f.endswith(".go") and posixpath.basename(posixpath.dirname(f)) == tail)[:10]
    if ext == ".rs":
        segs = [s for s in re.split(r"[:{},\s*]+", raw) if s and s not in ("crate", "self", "super", "use")]
        for s in segs:
            hit = stems.get(s)
            if hit and len(hit) == 1:
                return hit
        return []
    if "/" in raw or "." in raw:  # c/c++ include
        return [f for f in files if f == raw or f.endswith("/" + raw)][:3]
    return []
