"""Optional semantic layer: local embeddings of symbols, a content-hash vector cache and "similar" links.

Everything here is optional. Without numpy and an embedding provider (`pip install codegraph[embed]`),
`load_embedder()` returns None and the rest of codegraph behaves exactly as it does without this module.

No network at query time: the default provider runs a small local ONNX model through fastembed (the
model files are downloaded once, on first indexing). Providers sit behind the tiny `Embedder` protocol,
so another backend is a class with three members, selected with CODEGRAPH_EMBED=module:factory.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from typing import Protocol

from .links import is_test_path
from .store import TARGET_KINDS_CALL, Store, split_words

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
TEXT_VERSION = "1"  # bump when symbol_text() changes: every cached vector is then recomputed
EMBED_KINDS = tuple(sorted(TARGET_KINDS_CALL))  # variables / fields / modules are noise, not concepts
BATCH = 128
SIM_K = 5  # neighbours per symbol
SIM_THRESHOLD = 0.85  # cosine; tuned on this repo's own code (bge-small): below it, mostly generic noise
SIM_CHUNK = 256  # rows of the similarity matrix computed at once: bounds memory at chunk x N floats


class Embedder(Protocol):
    """What codegraph needs from an embedding backend. Vectors must be L2-normalised float32."""

    name: str  # identifies model + settings; part of every cache key, so changing it re-embeds

    def embed_passages(self, texts: list[str]): ...  # -> ndarray (len(texts), dim)

    def embed_query(self, text: str): ...  # -> ndarray (dim,)


class FastEmbedder:
    """fastembed / ONNX. The model is loaded on first use, so a no-change reindex never pays for it."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.name = f"fastembed:{model}"
        self._model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed_passages(self, texts):
        import numpy as np
        return np.asarray(list(self._load().passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text):
        import numpy as np
        return np.asarray(next(iter(self._load().query_embed([text]))), dtype=np.float32)


def load_embedder() -> Embedder | None:
    """The configured provider, or None when semantic search is unavailable or switched off.

    CODEGRAPH_EMBED: unset / "fastembed" (default), "off", or "package.module:factory".
    CODEGRAPH_EMBED_MODEL picks another fastembed model.
    """
    spec = os.environ.get("CODEGRAPH_EMBED", "fastembed").strip()
    if spec.lower() in ("off", "0", "none", "false", ""):
        return None
    if importlib.util.find_spec("numpy") is None:
        return None
    if ":" in spec:
        mod, _, attr = spec.partition(":")
        return getattr(importlib.import_module(mod), attr)()
    if importlib.util.find_spec("fastembed") is None:
        return None
    return FastEmbedder(os.environ.get("CODEGRAPH_EMBED_MODEL", DEFAULT_MODEL))


def symbol_text(r) -> str:
    """What gets embedded: the name as words, the qualified name, the signature and the doc line."""
    parts = [r["kind"], split_words(r["name"]), r["qualname"], r["sig"] or "", r["doc"] or ""]
    return ". ".join(p for p in parts if p)[:400]


def text_hash(embedder: Embedder, text: str) -> str:
    return hashlib.sha1(f"{TEXT_VERSION}|{embedder.name}|{text}".encode()).hexdigest()


def _ph(n: int) -> str:
    return ",".join("?" * n)


def embed_pending(store: Store, embedder: Embedder, progress=None) -> list[int]:
    """Embed every symbol that has no vector yet; vectors already cached by content are reused.
    Returns the ids of the symbols that were newly linked to a vector."""
    import numpy as np
    db = store.db
    if store.get_meta("embed_model") != embedder.name:  # other model: nothing cached is comparable
        db.execute("DELETE FROM sym_vec")
        db.execute("DELETE FROM links WHERE kind='similar'")
        store.set_meta("embed_model", embedder.name)
    rows = db.execute(
        f"SELECT s.id, s.name, s.qualname, s.kind, s.sig, s.doc FROM symbols s "
        f"LEFT JOIN sym_vec v ON v.sym_id = s.id WHERE v.sym_id IS NULL AND s.kind IN ({_ph(len(EMBED_KINDS))})",
        EMBED_KINDS).fetchall()
    if not rows:
        return []
    hashes = [text_hash(embedder, symbol_text(r)) for r in rows]
    known: set[str] = set()
    uniq = list(dict.fromkeys(hashes))
    for i in range(0, len(uniq), 500):
        chunk = uniq[i:i + 500]
        known |= {r[0] for r in db.execute(f"SELECT hash FROM vec_cache WHERE hash IN ({_ph(len(chunk))})", chunk)}
    todo = {h: symbol_text(r) for r, h in zip(rows, hashes, strict=True) if h not in known}
    items = list(todo.items())
    for i in range(0, len(items), BATCH):
        batch = items[i:i + BATCH]
        vecs = embedder.embed_passages([t for _, t in batch]).astype(np.float32)
        db.executemany("INSERT OR REPLACE INTO vec_cache VALUES(?,?)",
                       [(h, vecs[j].tobytes()) for j, (h, _) in enumerate(batch)])
        if progress and (i // BATCH) % 10 == 0:
            progress(f"embedded {min(i + BATCH, len(items))}/{len(items)} symbols")
    db.executemany("INSERT OR REPLACE INTO sym_vec VALUES(?,?)", [(r["id"], h) for r, h in zip(rows, hashes, strict=True)])
    db.execute("DELETE FROM vec_cache WHERE hash NOT IN (SELECT hash FROM sym_vec)")
    store.commit()
    return [r["id"] for r in rows]


def load_matrix(db):
    """(ids, unit-norm float32 matrix) of every embedded symbol, or (None, None) when there are none."""
    import numpy as np
    rows = db.execute("SELECT v.sym_id, c.vec FROM sym_vec v JOIN vec_cache c ON c.hash = v.hash "
                      "ORDER BY v.sym_id").fetchall()
    if not rows:
        return None, None
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, mat


def build_similar(store: Store, k: int = SIM_K, threshold: float = SIM_THRESHOLD,
                  only: set[int] | None = None) -> int:
    """Top-k nearest neighbours above `threshold`, as `similar` links. Brute-force cosine, chunked.

    `only` restricts the queries to those symbols (their links are replaced, everyone else's kept),
    which is what an edit to one file needs; None rebuilds every link.

    Skipped: pairs already tied together by a structural edge (a call or an inheritance says more than
    "looks alike"), a symbol's own ancestors, same-named overloads, and test-to-test pairs (test names
    all resemble each other, which tells you nothing).
    """
    import numpy as np
    db = store.db
    if only is None:
        db.execute("DELETE FROM links WHERE kind='similar'")
    ids, mat = load_matrix(db)
    if ids is None or len(ids) < 2:
        store.set_meta("similar_n", "0")
        return 0
    linked = {(a, b) for a, b in db.execute("SELECT src, dst FROM edges")}
    linked |= {(b, a) for a, b in linked}
    info = {r["id"]: (r["parent"], r["file"], r["name"]) for r in db.execute("SELECT id, parent, file, name FROM symbols")}

    def ancestors(i: int) -> set[int]:
        out, p = set(), info[i][0]
        while p is not None and p not in out:
            out.add(p)
            p = info.get(p, (None,))[0]
        return out

    seen = {(a, b) if a < b else (b, a) for a, b in db.execute("SELECT src, dst FROM links WHERE kind='similar'")}
    queries = np.arange(len(ids)) if only is None else np.flatnonzero(np.isin(ids, list(only)))
    kk = min(k + 1, len(ids))
    rows = []
    for start in range(0, len(queries), SIM_CHUNK):
        qi = queries[start:start + SIM_CHUNK]
        sims = mat[qi] @ mat.T
        sims[np.arange(len(qi)), qi] = -1.0  # not its own neighbour
        top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for r, i in enumerate(qi):
            a = int(ids[i])
            for j in top[r]:
                s = float(sims[r, j])
                b = int(ids[j])
                key = (a, b) if a < b else (b, a)
                if s < threshold or key in seen or (a, b) in linked:
                    continue
                if info[a][1:] == info[b][1:] or b in ancestors(a) or a in ancestors(b):
                    continue  # overloads of one name in one file / a class and its own member
                if is_test_path(info[a][1]) and is_test_path(info[b][1]):
                    continue
                seen.add(key)
                rows.append((key[0], key[1], "similar", s, ""))
    db.executemany("INSERT INTO links VALUES(?,?,?,?,?)", rows)
    store.set_meta("similar_n", str(len(ids)))
    return len(rows)


FULL_REBUILD_FRACTION = 0.2  # past this much churn a full pass is cheaper than patching


def update(store: Store, progress=None) -> dict:
    """Bring vectors and `similar` links up to date. Instant when nothing is pending."""
    embedder = load_embedder()
    if embedder is None:
        return {"enabled": False}
    new_ids = embed_pending(store, embedder, progress)
    n = store.db.execute("SELECT COUNT(*) FROM sym_vec").fetchone()[0]
    built = store.get_meta("similar_n")
    if built is None or len(new_ids) > FULL_REBUILD_FRACTION * max(n, 1):
        similar = build_similar(store)
    elif new_ids:
        similar = build_similar(store, only=set(new_ids))
    else:
        store.set_meta("similar_n", str(n))  # deletions only: their links went with the symbols
        similar = -1  # -1: unchanged
    store.commit()
    return {"enabled": True, "model": embedder.name, "embedded": len(new_ids), "vectors": n, "similar": similar}
