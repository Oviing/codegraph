"""Derived symbol roles: a ranking prior for search, computed from data already in the index.

A role says how a symbol is meant to be used (`public` API, `private` helper, `generated`, `native`
code in another language), not what it is. Nothing is stored: roles come from the name, qualified name
and path, so they need no reindex and cannot go stale. Structural tools (callers, callees, impact)
ignore them; only `_hybrid_rows` uses them to break ties between equally relevant symbols.
"""
from __future__ import annotations

import os
import posixpath
import re
from collections import Counter

from .parser import EXT

# how much of a symbol's relevance score survives its role; public code is the reference
ROLE_WEIGHT = {"public": 1.0, "private": 0.6, "generated": 0.5, "native": 0.7}
GENERATED = re.compile(r"(_pb2(_grpc)?\.py$|_generated\.|\.generated\.|(^|/)generated/|\.pb\.(cc|h)$)")
NATIVE_LANGS = {"cpp"}


def enabled() -> bool:
    return os.environ.get("CODEGRAPH_ROLES", "on").lower() not in ("off", "0", "false")


def lang_of(path: str) -> str | None:
    return EXT.get(posixpath.splitext(path)[1].lower())


def dominant_lang(file_paths) -> str | None:
    """The language most files are written in; code in a native language is only 'foreign' next to it."""
    counts = Counter(lang_of(p) for p in file_paths)
    counts.pop(None, None)
    return counts.most_common(1)[0][0] if counts else None


def _private(qualname: str) -> bool:
    parts = qualname.split(".")
    return any(p.startswith("_") and not (p.startswith("__") and p.endswith("__")) for p in parts if p)


def role_of(file: str, qualname: str, repo_lang: str | None) -> str:
    if GENERATED.search(file):
        return "generated"
    if lang_of(file) in NATIVE_LANGS and repo_lang not in NATIVE_LANGS:
        return "native"
    if _private(qualname):
        return "private"
    return "public"


def prior(file: str, qualname: str, repo_lang: str | None, query: str) -> float:
    """Multiplier for a symbol's relevance. Asking for an underscore name by its spelling (`_split`)
    means the user wants the private symbol, so it is not demoted."""
    role = role_of(file, qualname, repo_lang)
    if role == "private" and re.search(r"(^|\W)_[A-Za-z0-9]", query):
        return 1.0
    return ROLE_WEIGHT[role]
