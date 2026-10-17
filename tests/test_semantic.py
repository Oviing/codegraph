"""The semantic layer, driven by a deterministic fake model (tests/fake_embed.py)."""
import json
import sqlite3

from codegraph.embed import load_embedder
from codegraph.indexer import PARSER_VERSION, db_path, index_repo
from codegraph.queries import Graph
from tests.conftest import link_set, make_repo

FILES = {
    "shop/auth.py": 'def authenticate_user(name, password):\n    """Check credentials and open a session."""\n'
                    "    return True\n\ndef hash_password(p):\n    return p\n",
    "shop/billing.py": 'def charge_card(amount):\n    """Bill the customer."""\n    return amount\n\n'
                       "def parse_invoice(text):\n    return text\n",
    "shop/mail.py": 'def send_receipt(addr):\n    """Email the customer."""\n    return addr\n',
}


def vectors(root) -> int:
    return sqlite3.connect(db_path(root)).execute("SELECT COUNT(*) FROM sym_vec").fetchone()[0]


# ---- degradation: the extra is optional ----
def test_without_the_extra_nothing_changes(tmp_path):
    root = make_repo(tmp_path, FILES)  # conftest sets CODEGRAPH_EMBED=off
    stats = index_repo(root)
    assert stats["links"]["embed"] == {"enabled": False}
    assert vectors(root) == 0
    g = Graph(root)
    out = g.search("authenticate")
    assert "authenticate_user" in out and "[" not in out.splitlines()[0].split("  ")[-1]  # no signal tag
    assert g.search("login") == "No matches."  # purely lexical, as before
    assert "Semantic index not available" in g.similar("charge_card")


def test_missing_numpy_or_fastembed_means_no_embedder(monkeypatch):
    monkeypatch.delenv("CODEGRAPH_EMBED", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    assert load_embedder() is None


# ---- hybrid search ----
def test_hybrid_finds_a_symbol_that_shares_no_word_with_the_query(tmp_path, semantic):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    g = Graph(root)
    assert g._search_rows("login", 10) == []  # lexical alone cannot do it
    out = g.search("login")
    assert out.splitlines()[0].startswith("function authenticate_user") and out.splitlines()[0].endswith("[semantic]")


def test_signals_are_reported_per_result(tmp_path, semantic):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    rows = dict((r["name"], sig) for r, sig in Graph(root)._hybrid_rows("authenticate", 10))
    assert rows["authenticate_user"] == "both"  # matched by name and by meaning
    g = Graph(root)
    assert "[both]" in g.search("authenticate")


def test_a_query_unrelated_to_everything_returns_no_semantic_noise(tmp_path, semantic):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    assert Graph(root).search("zebra quantum") == "No matches."


def test_focus_uses_hybrid_seeds(tmp_path, semantic):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    out = Graph(root).repo_map(focus="login")
    assert "authenticate_user" in out and "no symbols matched" not in out


# ---- vector cache ----
def test_only_changed_symbols_are_reembedded(tmp_path, semantic):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    first = semantic.calls
    assert first > 0 and vectors(root) == 5
    index_repo(root)
    assert semantic.calls == first  # nothing changed: the model is not even asked
    (root / "shop/mail.py").write_text(FILES["shop/mail.py"] + "\ndef send_reminder(addr):\n    return addr\n")
    index_repo(root)
    assert semantic.calls == first + 1  # one new symbol, everything else came from the cache
    assert vectors(root) == 6
    (root / "shop/mail.py").write_text(FILES["shop/mail.py"])  # the old text again: still cached by content
    before = semantic.calls
    index_repo(root)
    assert semantic.calls == before and vectors(root) == 5


def test_switching_model_reembeds_everything(tmp_path, semantic, monkeypatch):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    monkeypatch.setattr(semantic, "name", "fake:concepts-2")
    before = semantic.calls
    index_repo(root)
    assert semantic.calls == before + 5


# ---- similar links ----
def test_similar_links_connect_same_meaning_and_stay_soft(tmp_path, semantic):
    root = make_repo(tmp_path, {
        **FILES,
        "legacy/login.py": 'def signin_account(u, p):\n    """Authenticate credentials."""\n    return True\n',
    })
    index_repo(root)
    g = Graph(root)
    sim = {(a.split(":")[1], b.split(":")[1]) for a, b, _ in link_set(g, "similar")}
    assert any({a, b} == {"authenticate_user", "signin_account"} for a, b in sim)
    assert not any("charge_card" in pair and "authenticate_user" in pair for pair in sim)
    # soft: visible in related/context/similar, invisible to the structural tools
    assert "similar" in g.related("authenticate_user")
    assert "signin_account" not in g.callers("authenticate_user") + g.callees("authenticate_user")
    assert "signin_account" not in g.impact("authenticate_user")
    assert g.db.execute("SELECT COUNT(*) FROM edges WHERE kind='similar'").fetchone()[0] == 0
    assert "signin_account" in g.similar("authenticate_user")


def test_pairs_already_linked_by_a_call_get_no_similar_link(tmp_path, semantic):
    root = make_repo(tmp_path, {
        "a.py": 'def authenticate_user():\n    """Check credentials."""\n    return signin_check()\n\n'
                'def signin_check():\n    """Check credentials."""\n    return True\n',
    })
    index_repo(root)
    g = Graph(root)
    assert link_set(g, "similar") == set()  # the call already relates them
    assert "authenticate_user" in g.callers("signin_check")


def test_incremental_update_links_only_what_changed(tmp_path, semantic):
    root = make_repo(tmp_path, FILES)
    index_repo(root)
    (root / "legacy.py").write_text('def signin_account():\n    """Authenticate credentials."""\n    return 1\n')
    stats = index_repo(root)["links"]["embed"]
    assert stats["embedded"] == 1 and stats["similar"] >= 1
    g = Graph(root)
    assert any("signin_account" in a + b for a, b, _ in link_set(g, "similar"))
    (root / "legacy.py").unlink()
    index_repo(root)
    assert not any("signin_account" in a + b for a, b, _ in link_set(Graph(root), "similar"))


# ---- schema / parser versions ----
def test_old_index_rebuilds_automatically(tmp_path):
    root = make_repo(tmp_path, {"server.py": '@app.get("/u")\ndef a():\n    pass\n',
                                "client.py": 'def b():\n    return requests.get("/u")\n'})
    index_repo(root)
    db = sqlite3.connect(db_path(root))
    db.execute("UPDATE meta SET value='4' WHERE key='parser_version'")  # as written by the previous release
    db.execute("DELETE FROM raw_keys")
    db.execute("DELETE FROM links")
    db.commit()
    db.close()
    stats = index_repo(root)
    assert stats["changed"] == 2  # re-parsed from scratch
    assert len(link_set(Graph(root), "key")) == 1
    assert sqlite3.connect(db_path(root)).execute(
        "SELECT value FROM meta WHERE key='parser_version'").fetchone()[0] == str(PARSER_VERSION)


# ---- bench ----
def test_bench_reports_recall_and_tokens(tmp_path, semantic, capsys):
    from codegraph.bench import run
    root = make_repo(tmp_path, FILES)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"task": "login", "expect": ["authenticate_user"]},
                                 {"task": "billing", "expect": ["charge_card", "nope_missing"]}]))
    out = run(root, tasks, k=5)
    assert "semantic index: yes" in out
    assert "recall   lexical 0/2 (0%)   hybrid 2/2 (100%)" in out
    assert "not in index: nope_missing" in out
    assert "tokens   graph" in out


def test_bench_without_embeddings_says_so(tmp_path):
    from codegraph.bench import run
    root = make_repo(tmp_path, FILES)
    tasks = tmp_path / "t.json"
    tasks.write_text(json.dumps([{"task": "authenticate", "expect": ["authenticate_user"]}]))
    out = run(root, tasks)
    assert "NO (hybrid == lexical)" in out and "lexical 1/1" in out and "hybrid 1/1" in out
