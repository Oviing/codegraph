"""Incremental repository indexer."""
from __future__ import annotations

import fnmatch
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import community, embed, links
from .docs import DOC_EXT, DOC_LANGS, parse_doc
from .parser import EXT, parse
from .store import SCHEMA_VERSION, Store, recompute_rank

SKIP_DIRS = {"node_modules", "venv", "__pycache__", "dist", "build", "target", "vendor", "site-packages"}
MAX_BYTES = 1_000_000
# bump when parser output changes; forces a full re-index
PARSER_VERSION = 6  # 6: edge confidence, docs/SQL/PDF symbols
IGNORE_FILE = ".codegraphignore"
PARALLEL_MIN = 48  # below this, starting worker processes costs more than it saves
ALL_EXT = {**EXT, **DOC_EXT}


def db_path(root: Path) -> Path:
    return root / ".codegraph" / "graph.db"


def load_ignore(root: Path) -> list[tuple[str, bool, bool, bool]]:
    """Parse `.codegraphignore` (gitignore-style, minus `**` and escapes) into
    (pattern, negated, dir_only, anchored) rules. Later rules win, as in gitignore."""
    try:
        text = (root / IGNORE_FILE).read_text(errors="replace")
    except OSError:
        return []
    rules = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        line = line.lstrip("!")
        dir_only = line.endswith("/")
        line = line.strip("/") if dir_only else line
        anchored = line.startswith("/") or "/" in line
        if line:
            rules.append((line.lstrip("/"), neg, dir_only, anchored))
    return rules


def is_ignored(rel: str, is_dir: bool, rules) -> bool:
    segs = rel.split("/")
    hit = False
    for pat, neg, dir_only, anchored in rules:
        if anchored:
            # the path itself, or any directory prefix of it
            cands = ["/".join(segs[:i]) for i in range(1, len(segs) + 1)]
            kinds = [True] * (len(segs) - 1) + [is_dir]
        else:
            cands, kinds = segs, [True] * (len(segs) - 1) + [is_dir]
        if any(fnmatch.fnmatchcase(c, pat) and (k or not dir_only) for c, k in zip(cands, kinds, strict=True)):
            hit = not neg
    return hit


def _scan(root: Path) -> dict[str, tuple[Path, str]]:
    seen: dict[str, tuple[Path, str]] = {}
    rules = load_ignore(root)
    for dirpath, dirs, files in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
                   and not (rules and is_ignored(rel_dir + d, True, rules))]
        for f in files:
            lang = ALL_EXT.get(Path(f).suffix.lower())
            if not lang or (rules and is_ignored(rel_dir + f, False, rules)):
                continue
            p = Path(dirpath) / f
            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
            except OSError:
                continue
            seen[p.relative_to(root).as_posix()] = (p, lang)
    return seen


def _parse_one(job):
    """Worker: parse one file. Module level so it can be pickled into a process pool."""
    data, lang, stem = job
    try:
        return (parse_doc if lang in DOC_LANGS else parse)(data, lang, stem), None
    except Exception as e:  # a broken file must not kill the index
        return None, str(e)


def _parse_all(jobs: list) -> list:
    """Parse in parallel across processes: tree-sitter plus the Python tree walk is CPU-bound, so
    threads would not help. Falls back to serial if a pool cannot be started."""
    if len(jobs) >= PARALLEL_MIN and (os.cpu_count() or 1) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, 8)) as pool:
                return list(pool.map(_parse_one, jobs, chunksize=16))
        except Exception as e:  # sandboxes without fork/semaphores, broken pool, ...
            print(f"codegraph: parallel parse unavailable ({e}); parsing serially", file=sys.stderr)
    return [_parse_one(j) for j in jobs]


def index_repo(root: str | Path, force: bool = False, progress=None) -> dict:
    """Update the index for `root`. Only changed files are re-parsed, and only the edges that
    those changes can affect are re-resolved."""
    root = Path(root).resolve()
    store = Store(db_path(root))

    def note(msg: str):
        if progress is not None:
            progress(msg)

    # a database written by an older parser, or by an older schema, is rebuilt from scratch
    if store.get_meta("parser_version") != str(PARSER_VERSION):
        note("parser version changed, rebuilding index")
        force = True

    if force:
        store.clear()  # one bulk delete instead of one per file
    seen = _scan(root)
    old = store.file_hashes()
    removed = set(old) - set(seen)

    # hash first, so we know the full dirty set before mutating anything
    to_parse: list[tuple[str, Path, str, bytes, str]] = []
    for rel, (p, lang) in sorted(seen.items()):
        try:
            data = p.read_bytes()
        except OSError:
            continue
        h = hashlib.sha1(data).hexdigest()
        if not force and old.get(rel) == h:
            continue
        to_parse.append((rel, p, lang, data, h))

    dirty_files = set(removed) | {rel for rel, *_ in to_parse}
    # names are captured before deletion: a definition that disappears can invalidate edges
    # elsewhere that referred to it by name
    dirty_names = store.names_in(dirty_files) if dirty_files else set()
    # ...and importers must be captured before delete_file drops the rows that point at them,
    # otherwise their import edges would never be recomputed and would stay missing
    if dirty_files:
        dirty_files |= store.importers_of(dirty_files)

    for r in sorted(removed):
        store.delete_file(r)
    for rel, *_ in to_parse:
        if rel in old:  # a new file has nothing to delete
            store.delete_file(rel)
    changed = 0
    parsed_all = _parse_all([(data, lang, Path(rel).stem) for rel, _p, lang, data, _h in to_parse])
    for (rel, _p, lang, data, h), (parsed, err) in zip(to_parse, parsed_all, strict=True):
        if parsed is None:
            print(f"codegraph: skipped {rel}: {err}", file=sys.stderr)  # stdout is the MCP channel
            continue
        store.add_file(rel, lang, h, parsed, data.count(b"\n") + 1)
        dirty_names |= {s.name for s in parsed.syms}
        changed += 1
        if changed % 200 == 0:
            note(f"parsed {changed} files")

    dirty_names -= {None}
    resolve_stats: dict = {}
    rank_stats: dict = {}
    link_stats: dict = {}
    if changed or removed or force or store.migrated:
        resolve_stats = store.resolve(full=force or not dirty_names,
                                      dirty_files=dirty_files,
                                      dirty_names=dirty_names)
        store.set_meta("parser_version", str(PARSER_VERSION))
        store.set_meta("schema_version", str(SCHEMA_VERSION))
        rank_stats = recompute_rank(store)
        link_stats.update(links.link_static(store.db))
        link_stats["communities"] = community.detect(store.db)
    link_stats["embed"] = embed.update(store, note)
    link_stats["cochange"] = links.link_cochange(store, root, force)  # -1: unchanged, 0: no git
    store.commit()
    db = store.db
    n_sym = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    n_edge = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    store.close()
    return {"files": len(seen), "changed": changed, "removed": len(removed),
            "symbols": n_sym, "edges": n_edge,
            "resolve": resolve_stats, "rank": rank_stats, "links": link_stats}
