"""SQLite graph store: symbols, resolved edges, file-level imports, FTS search."""
from __future__ import annotations

import posixpath
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from .parser import Parsed
from .rank import pagerank

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, lang TEXT, hash TEXT, loc INT);
CREATE TABLE IF NOT EXISTS symbols(
  id INTEGER PRIMARY KEY, file TEXT, name TEXT, qualname TEXT, kind TEXT,
  line_start INT, line_end INT, sig TEXT, doc TEXT, parent INT, rank REAL DEFAULT 0,
  community INT DEFAULT -1);
CREATE TABLE IF NOT EXISTS raw_edges(src INT, name TEXT, kind TEXT);
-- conf: how the edge was resolved. extracted = unique target named by the source itself (same file,
-- import, self/this); inferred = name match by proximity; ambiguous = several equally plausible targets
CREATE TABLE IF NOT EXISTS edges(src INT, dst INT, kind TEXT, conf TEXT DEFAULT 'inferred');
CREATE TABLE IF NOT EXISTS raw_imports(file TEXT, raw TEXT);
CREATE TABLE IF NOT EXISTS imports(file TEXT, dst TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS ix_sym_name ON symbols(name);
CREATE INDEX IF NOT EXISTS ix_sym_file ON symbols(file);
CREATE INDEX IF NOT EXISTS ix_sym_qual ON symbols(qualname);
CREATE INDEX IF NOT EXISTS ix_e_src ON edges(src);
CREATE INDEX IF NOT EXISTS ix_e_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS ix_re_src ON raw_edges(src);
CREATE INDEX IF NOT EXISTS ix_imp_file ON imports(file);
CREATE INDEX IF NOT EXISTS ix_imp_dst ON imports(dst);
CREATE TABLE IF NOT EXISTS raw_keys(src INT, ns TEXT, key TEXT, role TEXT);
CREATE INDEX IF NOT EXISTS ix_rk_key ON raw_keys(ns, key);
CREATE INDEX IF NOT EXISTS ix_rk_src ON raw_keys(src);
-- non-structural symbol links (similar / tests / key). Kept apart from `edges` so callers, callees,
-- impact and global PageRank stay purely structural.
CREATE TABLE IF NOT EXISTS links(src INT, dst INT, kind TEXT, weight REAL, note TEXT);
CREATE INDEX IF NOT EXISTS ix_l_src ON links(src);
CREATE INDEX IF NOT EXISTS ix_l_dst ON links(dst);
CREATE TABLE IF NOT EXISTS file_links(a TEXT, b TEXT, kind TEXT, weight REAL);
CREATE INDEX IF NOT EXISTS ix_fl_a ON file_links(a);
CREATE INDEX IF NOT EXISTS ix_fl_b ON file_links(b);
-- embeddings: vectors are cached by content hash, so only changed symbols are ever re-embedded
CREATE TABLE IF NOT EXISTS sym_vec(sym_id INTEGER PRIMARY KEY, hash TEXT);
CREATE TABLE IF NOT EXISTS vec_cache(hash TEXT PRIMARY KEY, vec BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS sym_fts USING fts5(name, words, qualname, doc);
"""


def _migration_1(db: sqlite3.Connection) -> bool:
    """Add the rank column (absent in databases written before ranking) and speed up name lookups."""
    db.execute("CREATE INDEX IF NOT EXISTS ix_re_name ON raw_edges(name)")
    cols = {r[1] for r in db.execute("PRAGMA table_info(symbols)")}
    if "rank" in cols:
        return False
    db.execute("ALTER TABLE symbols ADD COLUMN rank REAL DEFAULT 0")
    return True


def _migration_2(db: sqlite3.Connection) -> bool:
    """Links, keys and embedding tables: created by SCHEMA, which is idempotent. Nothing to alter."""
    return False


def _migration_3(db: sqlite3.Connection) -> bool:
    """Edge confidence and symbol community. Old databases are re-indexed (parser version bump)."""
    changed = False
    for table, col, ddl in (("edges", "conf", "TEXT DEFAULT 'inferred'"),
                            ("symbols", "community", "INT DEFAULT -1")):
        if col not in {r[1] for r in db.execute(f"PRAGMA table_info({table})")}:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
            changed = True
    return changed


MIGRATIONS = {1: _migration_1, 2: _migration_2, 3: _migration_3}

TARGET_KINDS_CALL = {"function", "method", "class", "struct", "type", "interface", "trait", "enum",
                     "table", "view", "procedure"}  # SQL objects, referenced from docs and other SQL
TARGET_KINDS_INHERIT = {"class", "interface", "trait", "struct", "type", "enum"}
CLASS_KINDS = {"class", "struct", "trait", "interface", "impl"}
# kinds that are containers: a `function` inside one is really a method
CONTAINER_KINDS = {"class", "struct", "trait", "interface", "impl"}
CONF_RANK = {"ambiguous": 0, "inferred": 1, "extracted": 2}
MAX_CANDIDATES = 4  # names defined in more places than this are too ambiguous to link

# Python builtins called directly (`isinstance(x, T)`) are not calls to anything in the repo.
# Without this, a repo method that happens to be called `isinstance` becomes a hub with thousands
# of phantom in-edges and dominates the global PageRank. A same-file definition still wins, so
# code that deliberately shadows a builtin keeps working.
PY_BUILTINS = frozenset(
    "abs all any ascii bin bool breakpoint bytearray bytes callable chr classmethod compile complex "
    "dict dir divmod enumerate eval exec filter float format frozenset getattr globals hasattr hash "
    "help hex id input int isinstance issubclass iter len list locals map max memoryview min next "
    "object oct open ord pow print property range repr reversed round set setattr slice sorted "
    "staticmethod str sum super tuple type vars zip".split()
)


def split_words(name: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ")


def _ph(values) -> str:
    """Placeholder list for an IN clause. Safe for empty input."""
    vals = list(values)
    return ",".join("?" * len(vals)) if vals else "NULL"


def _chunks(items, n: int = 500):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _suffix_paths(path: str) -> list[str]:
    """Every trailing sub-path of `path`, each exactly once: 'a/b/c' -> ['a/b/c', 'b/c', 'c']."""
    parts = [p for p in path.split("/") if p]
    return list(dict.fromkeys("/".join(parts[i:]) for i in range(len(parts))))


def _module_keys(f: str) -> list[str]:
    """Import keys for a module file.

    'src/pkg/a.py' becomes '/src/pkg/a.py', '/pkg/a.py', '/pkg/a.py' -- the extra entries are
    what let `crate::model` reach `src/model.rs` and `from pkg.a import x` reach `src/pkg/a.py`,
    since import paths never repeat the repository's own leading directories. Package roots
    (`__init__.py`, `mod.rs`) additionally answer to their bare directory name.
    """
    d, base = posixpath.split(f)
    keys = ["/" + s for s in _suffix_paths(f)]
    if base in ("__init__.py", "mod.rs") and d:
        ext = posixpath.splitext(base)[1]
        keys += ["/" + s + ext for s in _suffix_paths(d)]
    return list(dict.fromkeys(keys))


class FileIndex:
    """Lookup structures over the indexed file set, built once per resolve pass.

    Import resolution is the hot path of a reindex; every lookup here is a dict hit
    instead of a scan over all files. Python is resolved positionally instead (see py_lookup).
    """

    def __init__(self, files):
        self.files = set(files)
        self.rs_mod: dict[str, list[str]] = defaultdict(list)
        self.go_dir: dict[str, list[str]] = defaultdict(list)
        self.basename: dict[str, list[str]] = defaultdict(list)
        self.stem: dict[str, list[str]] = defaultdict(list)
        # conventional Python source roots, present only if this repository uses them
        self.import_roots = [""] + [d for d in ("src", "lib") if any(f.startswith(d + "/") for f in self.files)]
        for f in sorted(self.files):
            base = posixpath.basename(f)
            self.stem[posixpath.splitext(base)[0]].append(f)
            self.basename[base].append(f)
            self._register(f, base)

    def py_lookup(self, dotted: str) -> list[str]:
        """Resolve an absolute dotted module against the repository's import roots.

        Python resolves absolute imports against sys.path, not against the importing file's own
        directory, so this deliberately does *not* walk up from the importer: that would make
        `from typing import cast` inside `pydantic/_internal/` find `pydantic/typing.py`. The
        import roots are the repository root plus the conventional source roots that exist here.
        """
        if not dotted:
            return []
        rel = dotted.replace(".", "/")
        for base in self.import_roots:
            target = posixpath.join(base, rel) if base else rel
            hit = next((target + suf for suf in (".py", "/__init__.py") if target + suf in self.files), None)
            if hit:
                return [hit]
        return []

    def _register(self, f: str, base: str):
        """Register a file under every import key that can refer to it. See _module_keys."""
        ext = posixpath.splitext(base)[1]
        if ext == ".rs":
            for k in _module_keys(f):
                self.rs_mod[k].append(f)
        elif ext == ".go":
            for k in _suffix_paths(posixpath.dirname(f)):
                self.go_dir[k].append(f)


JS_SUFFIXES = ["", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs",
               "/index.ts", "/index.tsx", "/index.js", "/index.jsx", "/index.mjs", "/index.cjs"]
CPP_EXTS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".inl", ".ipp", ".tpp"}


def resolve_import(raw: str, src: str, idx: FileIndex) -> list[str]:
    """Map one raw import string to the repo files it most likely refers to."""
    raw = raw.strip()
    if not raw:
        return []
    ext = posixpath.splitext(src)[1]
    sdir = posixpath.dirname(src)

    if ext == ".py":
        dots = len(raw) - len(raw.lstrip("."))
        rest = raw.lstrip(".").replace(".", "/")
        if dots:  # `from .mod import x`: relative to the importing package
            base = sdir
            for _ in range(dots - 1):
                base = posixpath.dirname(base)
            target = posixpath.normpath(posixpath.join(base, rest)) if rest else base
            return sorted(f for s in (".py", "/__init__.py") if (f := target + s) in idx.files)
        # absolute: search the import roots, not the importer's directory (see py_lookup)
        return idx.py_lookup(rest)

    if ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"):
        if raw.startswith("."):
            return _js_candidates(posixpath.normpath(posixpath.join(sdir, raw)), idx)
        # bare specifier: resolvable only inside this repo's own workspace layout
        return _js_candidates(posixpath.normpath(raw.lstrip("/")), idx)

    if ext == ".go":
        segs = [s for s in raw.strip('"`').split("/") if s and s != "."]
        for i in range(len(segs)):  # longest matching path suffix wins
            hit = idx.go_dir.get("/".join(segs[i:]))
            if hit:
                return sorted(hit)[:10]
        return []

    if ext == ".rs":
        segs = [s for s in re.split(r"[:{},\s*]+", raw) if s and s not in ("crate", "use", "self", "super")]
        for i in range(len(segs)):
            for j in range(len(segs), i, -1):
                hit = idx.rs_mod.get("/" + "/".join(segs[i:j]) + ".rs")
                if hit:
                    return sorted(hit)
        for s in segs:
            hit = idx.stem.get(s)
            if hit and len(hit) == 1:
                return list(hit)
        return []

    if ext in CPP_EXTS or ext in (".cc", ".cpp"):
        name = raw.strip('<>"\'')
        if not name:
            return []
        if "/" in name:  # "dir/foo.h": must match the trailing path as well
            return sorted(f for f in idx.files if f.endswith("/" + name) or f == name)[:3]
        base = posixpath.basename(name)
        hits = [f for f in idx.basename.get(base, []) if posixpath.splitext(f)[1] in CPP_EXTS]
        if not hits and "." not in name:  # extensionless: `foo` -> foo.h / foo.hpp / ...
            hits = [f for f in idx.files
                    if posixpath.splitext(f)[0].endswith("/" + name) and posixpath.splitext(f)[1] in CPP_EXTS]
        same_dir = [f for f in hits if posixpath.dirname(f) == sdir]
        return sorted(same_dir or hits)[:3]

    return []


def _js_candidates(base: str, idx: FileIndex) -> list[str]:
    for s in JS_SUFFIXES:
        if base + s in idx.files:
            return [base + s]
    return []


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        # WAL lets the long-lived server connection read while a reindex writes
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._resolved_raws: dict[str, set[str]] = {}
        self.migrated = self._migrate()
        # explicit ids let add_file insert a whole file's symbols in one executemany
        self._next_id = (self.db.execute("SELECT MAX(id) FROM symbols").fetchone()[0] or 0) + 1

    def _migrate(self) -> bool:
        """Apply outstanding schema migrations. Every step must be idempotent.

        Only writes when something actually changes, and commits immediately: leaving an open
        write transaction here would block the indexer's own connection for the lifetime of the
        MCP server (this connection is long-lived and `check_same_thread=False`).
        """
        db = self.db
        row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        current = int(row["value"]) if row else 0
        schema_changed = False
        for version in sorted(MIGRATIONS):
            if version <= current:  # already applied
                continue
            if MIGRATIONS[version](db):
                schema_changed = True
        if current != SCHEMA_VERSION:
            db.execute("INSERT INTO meta(key,value) VALUES('schema_version',?) "
                       "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(SCHEMA_VERSION),))
        if current != SCHEMA_VERSION or schema_changed:
            db.commit()
        # reports whether an existing schema was actually altered, not merely version-stamped
        return schema_changed

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.db.commit()  # never leave a write transaction open on a long-lived connection

    def close(self):
        self.db.close()

    def commit(self):
        self.db.commit()

    # ---- files ----
    def file_hashes(self) -> dict[str, str]:
        return {r["path"]: r["hash"] for r in self.db.execute("SELECT path, hash FROM files")}

    def names_in(self, files) -> set[str]:
        """Every symbol name defined in the given files (used to scope incremental resolution)."""
        files = list(files)
        if not files:
            return set()
        rows = self.db.execute(
            f"SELECT DISTINCT name FROM symbols WHERE file IN ({_ph(files)})", files)
        return {r[0] for r in rows}

    def importers_of(self, files) -> set[str]:
        """Files whose resolved imports point at any of `files`.

        Must be called *before* deleting those files: deleting a file removes the import rows
        that point at it, after which the affected importers can no longer be discovered.
        """
        files = list(files)
        if not files:
            return set()
        rows = self.db.execute(f"SELECT DISTINCT file FROM imports WHERE dst IN ({_ph(files)})",
                               files).fetchall()
        return {r[0] for r in rows}

    def clear(self):
        """Drop every derived row (not the embedding cache, which is keyed by content hash).
        A forced rebuild is far cheaper this way than deleting file by file."""
        for t in ("files", "symbols", "raw_edges", "edges", "raw_imports", "imports", "raw_keys",
                  "links", "file_links", "sym_vec", "sym_fts"):
            self.db.execute(f"DELETE FROM {t}")

    def delete_file(self, path: str):
        """Remove a file and everything derived from it, so the store is never left inconsistent."""
        db = self.db
        ids = [r[0] for r in db.execute("SELECT id FROM symbols WHERE file=?", (path,))]
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            key = _ph(chunk)
            for table, col in (("raw_edges", "src"), ("sym_fts", "rowid"), ("raw_keys", "src"),
                               ("sym_vec", "sym_id")):
                db.execute(f"DELETE FROM {table} WHERE {col} IN ({key})", chunk)
            db.execute(f"DELETE FROM edges WHERE src IN ({key}) OR dst IN ({key})", chunk + chunk)
            db.execute(f"DELETE FROM links WHERE src IN ({key}) OR dst IN ({key})", chunk + chunk)
        db.execute("DELETE FROM file_links WHERE a=? OR b=?", (path, path))
        db.execute("DELETE FROM imports WHERE file=? OR dst=?", (path, path))
        db.execute("DELETE FROM raw_imports WHERE file=?", (path,))
        db.execute("DELETE FROM symbols WHERE file=?", (path,))
        db.execute("DELETE FROM files WHERE path=?", (path,))

    def add_file(self, path: str, lang: str, h: str, p: Parsed, loc: int):
        db = self.db
        db.execute("INSERT INTO files VALUES(?,?,?,?)", (path, lang, h, loc))
        idmap: dict[int, int] = {}
        sym_rows, fts_rows = [], []
        for s in p.syms:
            sid = self._next_id
            self._next_id += 1
            idmap[s.id] = sid
            parent = idmap.get(s.parent) if s.parent is not None else None
            q = s.qualname or s.name
            sym_rows.append((sid, path, s.name, q, s.kind, s.start, s.end, s.sig, s.doc, parent))
            fts_rows.append((sid, s.name, split_words(s.name), q, s.doc or ""))
        db.executemany("INSERT INTO symbols(id,file,name,qualname,kind,line_start,line_end,sig,doc,parent) "
                       "VALUES(?,?,?,?,?,?,?,?,?,?)", sym_rows)
        db.executemany("INSERT INTO sym_fts(rowid,name,words,qualname,doc) VALUES(?,?,?,?,?)", fts_rows)
        db.executemany("INSERT INTO raw_edges VALUES(?,?,?)",
                       [(idmap[src], name, kind) for src, name, kind in p.refs])
        db.executemany("INSERT INTO raw_keys VALUES(?,?,?,?)",
                       [(idmap[src], ns, key, role) for src, ns, key, role in dict.fromkeys(p.keys)])
        db.executemany("INSERT INTO raw_imports VALUES(?,?)",
                       [(path, raw) for raw in dict.fromkeys(p.imports)])  # de-duped, order preserved

    # ---- import resolution ----
    def _imported_map(self) -> dict[str, set[str]]:
        imported: dict[str, set[str]] = defaultdict(set)
        for f, d in self.db.execute("SELECT file, dst FROM imports"):
            imported[f].add(d)
        return imported

    @staticmethod
    def _base(raw: str) -> str:
        return raw.strip().lstrip(".").split(".", 1)[0]

    @staticmethod
    def _leaf(raw: str) -> str:
        return raw.strip().lstrip(".").replace("::", ".").rsplit("/", 1)[-1].rsplit(".", 1)[-1]

    def _external_imports(self, rows) -> dict[str, set[str]]:
        """Leaf names a file imports from modules that are not in the repo.

        `from typing import cast` in a file that has nothing to do with SQLAlchemy must not make a
        bare `cast(...)` call resolve to `ColumnElement.cast`. A name bound to a module we cannot
        see is treated as living outside the graph.

        A dotted raw whose *base* module did resolve is only a speculative submodule guess
        (`from .util import shout` also emits `util.shout`), so it is not evidence of anything
        external.
        """
        per_file: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            per_file[r["file"]].append(r["raw"])
        out: dict[str, set[str]] = defaultdict(set)
        for file, raws in per_file.items():
            known_bases = set()
            for raw in raws:
                if raw in self._resolved_raws.get(file, ()):
                    known_bases.add(self._base(raw))
            for raw in raws:
                if raw in self._resolved_raws.get(file, ()):
                    continue
                if self._base(raw) in known_bases:
                    continue  # speculative submodule of a module we do know
                leaf = self._leaf(raw)
                if leaf:
                    out[file].add(leaf)
        return out

    def _resolve_imports(self, dirty: set[str] | None = None) -> dict[str, set[str]]:
        """(Re)resolve file-level imports. With `dirty` set, only those files and their
        importers are recomputed; otherwise the whole table is rebuilt.
        Returns (imported, external) maps."""
        db = self.db
        files = [r["path"] for r in db.execute("SELECT path FROM files")]
        if dirty is None:
            db.execute("DELETE FROM imports")
            targets = files
        else:
            affected = set(dirty)
            for r in db.execute(f"SELECT DISTINCT file FROM imports WHERE dst IN ({_ph(affected)})",
                                list(affected)).fetchall():
                affected.add(r[0])
            db.execute(f"DELETE FROM imports WHERE file IN ({_ph(affected)})", list(affected))
            targets = [f for f in files if f in affected]
        # Resolution runs over every file, but only the affected files' rows are written. The
        # external-import map must cover untouched files too: edges are re-resolved for sources in
        # any file, and a missing entry would turn `from typing import cast` into a phantom edge.
        idx = FileIndex(files)
        rows = db.execute("SELECT file, raw FROM raw_imports").fetchall()
        affected_set = set(targets)
        found, resolved = set(), defaultdict(set)
        for r in rows:
            hits = [d for d in resolve_import(r["raw"], r["file"], idx) if d != r["file"]]
            if hits:
                resolved[r["file"]].add(r["raw"])
                if r["file"] in affected_set:
                    found.update((r["file"], d) for d in hits)
        self._resolved_raws = resolved
        external = self._external_imports(rows)
        db.executemany("INSERT INTO imports VALUES(?,?)", sorted(found))
        return self._imported_map(), external

    # ---- edge resolution ----
    def resolve(self, *, full: bool = False, dirty_files=None, dirty_names=None) -> dict:
        """Resolve raw references into edges.

        `full=True` (or no dirty set) rebuilds everything. Otherwise only raw edges whose
        name, or whose source file, is affected are recomputed; everything else keeps its
        edges. Returns a small stats dict.
        """
        db = self.db
        incremental = not full and dirty_files is not None
        imported, external = self._resolve_imports(
            None if full or not incremental else set(dirty_files))

        file_of, name_of, kind_of, parent_of = {}, {}, {}, {}
        by_name: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        for r in db.execute("SELECT id, name, file, kind, parent FROM symbols"):
            file_of[r["id"]] = r["file"]
            name_of[r["id"]], kind_of[r["id"]], parent_of[r["id"]] = r["name"], r["kind"], r["parent"]
            if r["kind"] not in ("module", "impl", "namespace"):
                by_name[r["name"]].append((r["id"], r["file"], r["kind"]))

        bases: dict[str, set[str]] = defaultdict(set)
        subs: dict[str, set[str]] = defaultdict(set)
        for e in db.execute("SELECT src, name FROM raw_edges WHERE kind='inherits'").fetchall():
            child = name_of.get(e["src"])
            if child is None:
                continue
            bases[child].add(e["name"])
            subs[e["name"]].add(child)

        class_cache: dict[int, str | None] = {}

        def class_of(i: int) -> str | None:
            if i not in class_cache:
                j, found = parent_of.get(i), None
                while j is not None:
                    if kind_of.get(j) in CLASS_KINDS:
                        found = name_of.get(j)
                        break
                    j = parent_of.get(j)
                class_cache[i] = found
            return class_cache[i]

        def resolve_one(src: int, name: str, kind: str) -> list[tuple[int, int, str, str]]:
            cands = by_name.get(name)
            if not cands:
                return []
            self_cn = class_of(src) if kind == "calls_self" else None
            if kind == "calls_self" and self_cn is None:
                kind = "calls_attr"  # `self` outside any class: fall back to the generic attribute rule
            is_attr = kind == "calls_attr"
            out_kind = "calls" if kind.startswith("calls") else kind
            allowed = TARGET_KINDS_INHERIT if out_kind == "inherits" else TARGET_KINDS_CALL
            tier = "far"  # the strongest evidence that picked the candidates, for the edge's confidence
            cands = [c for c in cands if c[2] in allowed]
            if not cands:
                return []
            sf = file_of[src]
            if kind == "calls" and name in external.get(sf, ()):
                return []  # bound to a module outside the repo, not to something in here
            if kind == "calls" and name in PY_BUILTINS:
                # a bare builtin call can only ever mean a module-level binding of that name;
                # `Util.isinstance(...)` is a method and is spelled with a receiver
                shadowed = [c for c in cands if c[1] == sf and c[2] != "method"]
                if not shadowed:
                    return []
                cands = shadowed
            if self_cn is not None:
                # `self.name()` can only mean a method of this class, its bases, or its subclasses.
                # Nothing in that family => inherited from an external base (dict, Exception...) => no edge.
                fam = _family(self_cn, bases, subs)
                cands = [c for c in cands if class_of(c[0]) in fam]
                if not cands:
                    return []
                tier = "self"
            else:
                same = [c for c in cands if c[1] == sf]
                if same:
                    cands, tier = same, "same"
                else:
                    imp = [c for c in cands if c[1] in imported.get(sf, ())]
                    if imp:
                        cands, tier = imp, "imp"
                    else:
                        sd = posixpath.dirname(sf)
                        near = [c for c in cands if posixpath.dirname(c[1]) == sd]
                        # `x.name()` with no same-file / imported / same-dir candidate is most likely a
                        # call on some library or builtin type (dict.items, str.get...), so it gets no edge.
                        cands = near if is_attr else (near or cands)
                        tier = "near" if near else "far"
            if not cands or len(cands) > MAX_CANDIDATES:
                return []
            cands = [c for c in cands if c[0] != src]  # drop direct self-recursion
            if kind == "mentions" and len(cands) != 1:
                return []  # a word in prose or SQL names a symbol only when it is unambiguous
            if len(cands) > 1:
                conf = "ambiguous"
            elif tier == "self" or (tier in ("same", "imp") and not is_attr):
                conf = "extracted"
            else:
                conf = "inferred"
            return [(src, c[0], out_kind, conf) for c in cands]

        if incremental:
            names = set(dirty_names or ())
            names |= _family_closure(names, bases, subs)  # a hierarchy change moves whole families
            hit = db.execute(
                f"SELECT DISTINCT r.src FROM raw_edges r JOIN symbols s ON s.id=r.src "
                f"WHERE r.name IN ({_ph(names)}) OR s.file IN ({_ph(dirty_files)})",
                list(names) + list(dirty_files)).fetchall()
            srcs = {r[0] for r in hit}
            # a touched source is recomputed from *all* of its raw references, not just the ones
            # that matched: its edges are deleted wholesale, so anything not re-resolved would be lost
            raw = []
            for chunk in _chunks(sorted(srcs)):
                raw += db.execute(f"SELECT src, name, kind FROM raw_edges WHERE src IN ({_ph(chunk)})",
                                  chunk).fetchall()
                db.execute(f"DELETE FROM edges WHERE src IN ({_ph(chunk)})", chunk)
        else:
            db.execute("DELETE FROM edges")
            raw = db.execute("SELECT src, name, kind FROM raw_edges").fetchall()
            srcs = set()

        best: dict[tuple[int, int, str], str] = {}
        for e in raw:
            for src, dst, ek, conf in resolve_one(e["src"], e["name"], e["kind"]):
                key = (src, dst, ek)
                if key not in best or CONF_RANK[conf] > CONF_RANK[best[key]]:
                    best[key] = conf
        db.executemany("INSERT INTO edges VALUES(?,?,?,?)", [(*k, c) for k, c in best.items()])

        total = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return {"raw_edges_considered": len(raw), "edges_total": total,
                "edges_changed": bool(srcs) or not incremental}


def _family(cn: str, bases: dict[str, set[str]], subs: dict[str, set[str]]) -> frozenset[str]:
    seen = {cn}
    for graph in (bases, subs):
        todo = [cn]
        while todo:
            for nxt in graph.get(todo.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
    return frozenset(seen)


def _family_closure(names, bases: dict[str, set[str]], subs: dict[str, set[str]]) -> set[str]:
    out = set()
    for n in names:
        out |= _family(n, bases, subs)
    return out


def recompute_rank(store: Store) -> dict:
    """Global PageRank over the whole graph. This is the one step that is always a full pass,
    so convergence is reported rather than assumed."""
    ids = [r[0] for r in store.db.execute("SELECT id FROM symbols")]
    edges = [(r[0], r[1]) for r in store.db.execute("SELECT src, dst FROM edges")]
    stats: dict = {}
    prev = {r[0]: r[1] for r in store.db.execute("SELECT id, rank FROM symbols")}
    scores = pagerank(ids, edges, stats=stats, init=prev)
    store.db.executemany("UPDATE symbols SET rank=? WHERE id=?",
                         [(v, k) for k, v in scores.items()])
    return stats
