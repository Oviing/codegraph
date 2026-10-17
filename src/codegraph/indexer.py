"""Incremental repository indexer."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from .parser import EXT, parse
from .store import Store

SKIP_DIRS = {"node_modules", "venv", "__pycache__", "dist", "build", "target", "vendor", "site-packages"}
MAX_BYTES = 1_000_000
PARSER_VERSION = 3  # bump when parser output changes; forces a full re-index


def db_path(root: Path) -> Path:
    return root / ".codegraph" / "graph.db"


def index_repo(root: str | Path, force: bool = False) -> dict:
    root = Path(root).resolve()
    store = Store(db_path(root))
    if store.db.execute("PRAGMA user_version").fetchone()[0] != PARSER_VERSION:
        force = True
    seen: dict[str, tuple[Path, str]] = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            lang = EXT.get(Path(f).suffix.lower())
            if not lang:
                continue
            p = Path(dirpath) / f
            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
            except OSError:
                continue
            seen[p.relative_to(root).as_posix()] = (p, lang)

    old = store.file_hashes()
    removed = set(old) - set(seen)
    for r in removed:
        store.delete_file(r)
    changed = 0
    for rel, (p, lang) in seen.items():
        try:
            data = p.read_bytes()
        except OSError:
            continue
        h = hashlib.sha1(data).hexdigest()
        if not force and old.get(rel) == h:
            continue
        store.delete_file(rel)
        try:
            parsed = parse(data, lang, p.stem)
        except Exception as e:  # a broken file must not kill the index
            print(f"codegraph: skipped {rel}: {e}", file=sys.stderr)  # stdout is the MCP channel
            continue
        store.add_file(rel, lang, h, parsed, data.count(b"\n") + 1)
        changed += 1
    if changed or removed or force or store.migrated:
        store.resolve()
    store.db.execute(f"PRAGMA user_version={PARSER_VERSION}")
    store.commit()
    db = store.db
    n_sym = db.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    n_edge = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    store.db.close()
    return {"files": len(seen), "changed": changed, "removed": len(removed), "symbols": n_sym, "edges": n_edge}
