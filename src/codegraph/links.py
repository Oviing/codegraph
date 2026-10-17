"""Hidden links that are not explicit in code: tests <-> subject, shared string keys, git co-change.

Everything here is derived from the index plus (optionally) `git log`; none of it needs a model.
Symbol-level results go to the `links` table and file-level ones to `file_links`, never to `edges`,
so callers / callees / impact stay purely structural.
"""
from __future__ import annotations

import posixpath
import re
import shutil
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from .store import TARGET_KINDS_CALL, Store, split_words

# ---------------------------------------------------------------------------------- tests <-> subject
TEST_DIR = re.compile(r"(^|/)(tests?|__tests__|spec|specs)/")
# (pattern, group holding the subject stem) per naming convention
TEST_NAMES = (
    re.compile(r"^test_(.+)\.py$"), re.compile(r"^(.+)_test\.py$"),
    re.compile(r"^(.+)\.(?:test|spec)\.[cm]?[jt]sx?$"),
    re.compile(r"^(.+)_test\.(?:go|rs|cpp|cc|cxx)$"), re.compile(r"^test_(.+)\.(?:cpp|cc|cxx|c)$"),
)
TEST_SYM = re.compile(r"^(?:test_?|Test_?)(.+?)(?:_?[Tt]ests?)?$")


def is_test_path(path: str) -> bool:
    return bool(TEST_DIR.search(path)) or subject_stem(path) is not None


def subject_stem(path: str) -> str | None:
    """`tests/test_store.py` -> `store`, `a/foo.test.ts` -> `foo`, `foo_test.go` -> `foo`."""
    base = posixpath.basename(path)
    for pat in TEST_NAMES:
        m = pat.match(base)
        if m:
            return m.group(1)
    return None


def _subject_file(test: str, by_stem: dict[str, list[str]]) -> str | None:
    """The non-test file a test file is named after; the nearest one wins when several share a stem."""
    stem = subject_stem(test)
    cands = [f for f in by_stem.get(stem or "", ()) if not is_test_path(f)]
    if not cands:
        return None

    def shared(f: str) -> int:
        n = 0
        for a, b in zip(test.split("/"), f.split("/"), strict=False):
            if a != b:
                break
            n += 1
        return n

    best = max(shared(f) for f in cands)
    top = [f for f in cands if shared(f) == best]
    return top[0] if len(top) == 1 else None


def _name_words(name: str) -> list[str]:
    return [w.lower() for w in split_words(name).split() if w]


def _subject_symbol(test_name: str, candidates: dict[str, int]) -> int | None:
    """`test_resolve_import_roots` -> the longest prefix that names a symbol: resolve_import_roots,
    resolve_import, resolve. `candidates` maps a normalised name to a symbol id."""
    m = TEST_SYM.match(test_name)
    words = _name_words(m.group(1) if m else test_name)
    for n in range(len(words), 0, -1):
        hit = candidates.get("".join(words[:n]))
        if hit is not None and (n > 1 or len(words[0]) >= 3):  # a lone tiny word is too weak a signal
            return hit
    return None


def link_tests(db) -> int:
    """Test file <-> subject file by naming convention, and `test_x` -> `x` inside that pair."""
    db.execute("DELETE FROM links WHERE kind='tests'")
    files = [r[0] for r in db.execute("SELECT path FROM files")]
    by_stem: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_stem[posixpath.splitext(posixpath.basename(f))[0]].append(f)
    syms: dict[str, list] = defaultdict(list)
    for r in db.execute("SELECT id, file, name, kind FROM symbols WHERE kind != 'impl'"):
        syms[r["file"]].append(r)
    rows = []
    for t in files:
        subject = _subject_file(t, by_stem)
        if subject is None:
            continue
        tmod = next((s["id"] for s in syms[t] if s["kind"] == "module"), None)
        smod = next((s["id"] for s in syms[subject] if s["kind"] == "module"), None)
        if tmod is not None and smod is not None:
            rows.append((tmod, smod, "tests", 1.0, "file"))
        cands = {"".join(_name_words(s["name"])): s["id"] for s in syms[subject]
                 if s["kind"] in TARGET_KINDS_CALL}
        for s in syms[t]:
            if s["kind"] in ("function", "method", "class") and (s["name"].lower().startswith("test") or
                                                                  s["name"].lower().endswith("test")):
                hit = _subject_symbol(s["name"], cands)
                if hit is not None:
                    rows.append((s["id"], hit, "tests", 1.0, "name"))
    db.executemany("INSERT INTO links VALUES(?,?,?,?,?)", rows)
    return len(rows)


# ---------------------------------------------------------------------------------- string keys
# namespace -> (role on one side, role on the other). Same role twice: everything sharing the key is related.
KEY_ROLES = {"route": ("provides", "uses"), "event": ("emits", "listens"), "config": ("reads", "reads")}
MAX_KEY_GROUP = 8  # a key shared by more symbols than this is a convention, not a link


def link_keys(db) -> int:
    """Link symbols in different files that use the same string key in matching roles."""
    db.execute("DELETE FROM links WHERE kind='key'")
    groups: dict[tuple[str, str], dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for r in db.execute("SELECT src, ns, key, role FROM raw_keys"):
        groups[(r["ns"], r["key"])][r["role"]].add(r["src"])
    file_of = {r[0]: r[1] for r in db.execute("SELECT id, file FROM symbols")}
    rows = []
    for (ns, key), roles in groups.items():
        a_role, b_role = KEY_ROLES.get(ns, (None, None))
        if a_role is None:
            continue
        a, b = roles.get(a_role, set()), roles.get(b_role, set())
        if a_role == b_role:
            pairs = combinations(sorted(a), 2) if 2 <= len(a) <= MAX_KEY_GROUP else ()
        elif a and b and len(a) <= MAX_KEY_GROUP and len(b) <= MAX_KEY_GROUP:
            pairs = ((x, y) for x in sorted(a) for y in sorted(b))
        else:
            pairs = ()
        note = f"{ns}:{key}"[:60]
        rows += [(x, y, "key", 1.0, note) for x, y in pairs if file_of.get(x) != file_of.get(y)]
    db.executemany("INSERT INTO links VALUES(?,?,?,?,?)", rows)
    return len(rows)


# ---------------------------------------------------------------------------------- git co-change
GIT_COMMITS = 1000  # history window
MAX_COMMIT_FILES = 30  # bigger commits (renames, reformatting, vendoring) say nothing about coupling
MIN_COMMITS = 3  # a pair must co-change at least this often: twice is a coincidence
MIN_WEIGHT = 0.3
MAX_PARTNERS = 8


def _git(root: Path, *args: str) -> str | None:
    """stdout of a git command in `root`, or None when git is missing / this is not a repository."""
    if shutil.which("git") is None:
        return None
    try:
        r = subprocess.run(["git", "-C", str(root), "-c", "core.quotepath=off", *args],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def cochange_pairs(log: str, known: set[str]) -> dict[tuple[str, str], float]:
    """Parse `git log --name-only` output (commits separated by a \\x1e line) into pair weights.

    A pair's weight is its co-change count over sqrt(commits of a * commits of b) (Ochiai), so a file
    touched in every commit does not become everyone's partner, shrunk by n/(n+2) so that pairs seen
    only a few times rank below pairs with a long shared history.
    """
    touched: dict[str, int] = defaultdict(int)
    pair_n: dict[tuple[str, str], int] = defaultdict(int)
    for commit in log.split("\x1e"):
        names = [f for f in commit.split("\n") if f]
        files = sorted({f for f in names if f in known})
        if len(files) < 2 or len(names) > MAX_COMMIT_FILES:  # size counts every file, indexed or not
            continue
        for f in files:
            touched[f] += 1
        for a, b in combinations(files, 2):
            pair_n[(a, b)] += 1
    out = {}
    for (a, b), n in pair_n.items():
        if n >= MIN_COMMITS:
            weight = n / (touched[a] * touched[b]) ** 0.5 * n / (n + 2)
            if weight >= MIN_WEIGHT:
                out[(a, b)] = weight
    return out


def link_cochange(store: Store, root: Path, force: bool = False) -> int:
    """Refresh file-level co-change edges when HEAD moved. No git, no history or no repo: no edges."""
    head = _git(root, "rev-parse", "HEAD")
    head = head.strip() if head else ""
    if not head:
        if store.get_meta("git_head"):  # was a repo, no longer is: drop the stale edges
            store.db.execute("DELETE FROM file_links WHERE kind='cochange'")
            store.set_meta("git_head", "")
        return 0
    if not force and store.get_meta("git_head") == head:
        return -1  # unchanged
    log = _git(root, "log", f"-n{GIT_COMMITS}", "--no-merges", "--relative", "--name-only",
               "--pretty=format:%x1e")
    known = {r[0] for r in store.db.execute("SELECT path FROM files")}
    pairs = cochange_pairs(log or "", known)
    # keep each file's strongest partners only, so a hub file cannot fan out to the whole repo
    per_file: dict[str, list[tuple[float, tuple[str, str]]]] = defaultdict(list)
    for pair, w in pairs.items():
        per_file[pair[0]].append((w, pair))
        per_file[pair[1]].append((w, pair))
    keep = {p for lst in per_file.values() for _, p in sorted(lst, reverse=True)[:MAX_PARTNERS]}
    store.db.execute("DELETE FROM file_links WHERE kind='cochange'")
    store.db.executemany("INSERT INTO file_links VALUES(?,?,?,?)",
                         [(a, b, "cochange", pairs[(a, b)]) for a, b in sorted(keep)])
    store.set_meta("git_head", head)
    return len(keep)


def link_static(db) -> dict:
    """The cheap, model-free symbol links; rebuilt wholesale whenever the index changed."""
    return {"tests": link_tests(db), "key": link_keys(db)}

