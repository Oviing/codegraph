"""Store-level behaviour: referential integrity of delete, schema migrations, meta handling."""
import sqlite3

from codegraph.parser import parse
from codegraph.store import SCHEMA_VERSION, Store


def make_store(tmp_path, name="graph.db"):
    return Store(tmp_path / name)


def seed(store):
    p = parse(b"class A:\n    def run(self):\n        return b()\n\n\ndef b():\n    return a()\n", "python", "m")
    store.add_file("m.py", "python", "h1", p, 20)
    store.add_file("other.py", "python", "h2",
                   parse(b"from m import b\n\n\ndef c():\n    return b()\n", "python", "other"), 20)
    store.resolve(full=True)
    store.commit()


def test_delete_file_removes_all_derived_rows(tmp_path):
    s = make_store(tmp_path)
    seed(s)
    ids = [r[0] for r in s.db.execute("SELECT id FROM symbols WHERE file='m.py'")]
    assert ids
    s.delete_file("m.py")
    # every row that referred to m.py must be gone...
    assert s.db.execute("SELECT COUNT(*) FROM symbols WHERE file='m.py'").fetchone()[0] == 0
    assert s.db.execute("SELECT COUNT(*) FROM files WHERE path='m.py'").fetchone()[0] == 0
    assert s.db.execute("SELECT COUNT(*) FROM raw_imports WHERE file='m.py'").fetchone()[0] == 0
    assert s.db.execute("SELECT COUNT(*) FROM imports WHERE file='m.py' OR dst='m.py'").fetchone()[0] == 0
    assert s.db.execute("SELECT COUNT(*) FROM edges WHERE src=? OR dst=?",
                       (ids[0], ids[0])).fetchone()[0] == 0
    # ...and other.py must be untouched
    assert s.db.execute("SELECT COUNT(*) FROM symbols WHERE file='other.py'").fetchone()[0] == 2
    assert s.db.execute("SELECT COUNT(*) FROM sym_fts").fetchone()[0] == 2
    # the importing file survives, but the import edge pointing at the deleted file is gone
    assert s.db.execute("SELECT COUNT(*) FROM imports WHERE dst='m.py'").fetchone()[0] == 0
    s.commit()
    s.close()


def test_delete_file_is_idempotent(tmp_path):
    s = make_store(tmp_path)
    seed(s)
    s.delete_file("m.py")
    s.delete_file("m.py")  # must not raise
    s.commit()
    s.close()


def test_no_dangling_edges_after_delete(tmp_path):
    s = make_store(tmp_path)
    seed(s)
    s.delete_file("m.py")
    s.resolve(full=True)
    dangling = s.db.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN symbols s ON s.id=e.dst WHERE s.id IS NULL"
    ).fetchone()[0]
    assert dangling == 0
    s.commit()
    s.close()


def test_wal_is_enabled(tmp_path):
    s = make_store(tmp_path)
    assert s.db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    s.close()


def test_schema_version_is_recorded(tmp_path):
    s = make_store(tmp_path)
    assert s.get_meta("schema_version") == str(SCHEMA_VERSION)
    s.close()


def test_migration_adds_rank_to_an_old_database(tmp_path):
    """A database from before the rank column must be upgraded, not silently mis-read."""
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE symbols(id INTEGER PRIMARY KEY, file TEXT, name TEXT, qualname TEXT,"
        " kind TEXT, line_start INT, line_end INT, sig TEXT, doc TEXT, parent INT);"
    )
    db.commit()
    db.close()

    s = Store(path)
    cols = {r[1] for r in s.db.execute("PRAGMA table_info(symbols)")}
    assert "rank" in cols
    assert s.migrated is True
    s.close()


def test_migration_is_idempotent(tmp_path):
    s = make_store(tmp_path)
    assert s.migrated is False  # already current
    s.close()
    s2 = Store(tmp_path / "graph.db")
    assert s2.migrated is False
    s2.close()


def test_meta_roundtrip(tmp_path):
    s = make_store(tmp_path)
    s.set_meta("parser_version", "7")
    assert s.get_meta("parser_version") == "7"
    s.set_meta("parser_version", "8")
    assert s.get_meta("parser_version") == "8"
    assert s.get_meta("absent", "fallback") == "fallback"
    s.commit()
    s.close()


def test_ph_helper_handles_empty_input():
    from codegraph.store import _ph
    assert _ph([]) == "NULL"
    assert _ph(["a", "b"]) == "?,?"


def test_add_file_deduplicates_imports(tmp_path):
    s = make_store(tmp_path)
    p = parse(b"import os\nimport os\n", "python", "m")
    s.add_file("m.py", "python", "h", p, 2)
    rows = s.db.execute("SELECT raw, COUNT(*) FROM raw_imports GROUP BY raw").fetchall()
    assert all(c == 1 for _, c in rows)
    s.commit()
    s.close()
