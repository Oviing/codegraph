import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from codegraph import ui

from .conftest import FIXTURES, build_graph


@pytest.fixture(scope="module")
def base():
    graph = build_graph(FIXTURES / "py")
    ui.Handler.api = ui.Api(graph)
    ui.Handler.page = b"<html>page</html>"
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    graph.close()


def get(base, path, host=None, method="GET", headers=None):
    h = {"Host": host} if host else {}
    req = urllib.request.Request(base + path, headers={**h, **(headers or {})}, method=method)
    with urllib.request.urlopen(req) as r:
        return r.read()


def j(base, path, **kw):
    return json.loads(get(base, path, **kw))


def test_page_and_overview(base):
    assert b"page" in get(base, "/")
    d = j(base, "/api/overview")
    assert d["pieces"] and d["comms"] and d["total"] >= 1
    assert any(p["f"].endswith("service.py") for p in d["pieces"])
    keys = {p["k"] for p in d["pieces"]}
    assert d["links"] and all(x["s"] in keys and x["t"] in keys for x in d["links"])
    assert sum(p["n"] for p in d["pieces"]) == sum(c["n"] for c in d["comms"]) + sum(
        p["n"] for p in d["pieces"] if p["c"] < 0)


def test_resolution_changes_granularity_and_is_clamped(base):
    coarse = len(j(base, "/api/overview?res=0.2")["comms"])
    fine = len(j(base, "/api/overview?res=5")["comms"])
    assert coarse <= fine
    assert j(base, "/api/overview?res=999")["res"] == 5.0
    assert j(base, "/api/overview?res=oops")["res"] == 1.0


def test_file_symbol_impact_search_community(base):
    overview = j(base, "/api/overview")
    p = next(p for p in overview["pieces"] if p["f"].endswith("core.py"))
    d = j(base, f"/api/file?path={p['f']}&c={p['c']}")
    assert d["symbols"] and all(s["c"] == p["c"] for s in d["symbols"])
    ids = {s["id"] for s in d["symbols"]} | {s["id"] for s in d["far"]}
    assert all(e["s"] in ids and e["t"] in ids for e in d["edges"])
    sid = d["symbols"][0]["id"]
    sym = j(base, f"/api/symbol?id={sid}")
    assert sym["qn"] and "source" in sym and "clabel" in sym
    assert "symbols" in j(base, f"/api/impact?id={sid}")
    assert j(base, f"/api/search?q={d['symbols'][0]['name']}")["symbols"]
    cid = next(c["id"] for c in overview["comms"])
    c = j(base, f"/api/community?id={cid}")
    assert c["symbols"] and 0 <= c["cohesion"] <= 1 and c["internal"] >= 0
    assert "error" in j(base, "/api/community?id=9999")


@pytest.mark.parametrize("name", list(ui.LENSES))
def test_every_lens_answers(base, name):
    d = j(base, f"/api/lens?name={name}")
    assert "error" not in d and d["title"]
    assert set(d) >= {"sym", "piece", "items", "agg"}
    assert all(v >= 0 for v in d["sym"].values())
    assert "error" in j(base, "/api/lens?name=nope")


def test_path_surprises_and_snapshot_diff(base):
    syms = [s for p in j(base, "/api/overview")["pieces"] for s in j(base, f"/api/file?path={p['f']}&c={p['c']}")["symbols"]]
    a, b = syms[0]["id"], syms[-1]["id"]
    p = j(base, f"/api/path?a={a}&b={b}")
    assert p["found"] == bool(p["symbols"]) and (not p["found"] or p["symbols"][0]["id"] == a)
    assert "edges" in j(base, "/api/surprises")
    assert "error" in j(base, "/api/diff")  # nothing saved yet
    assert j(base, "/api/snapshot", method="POST")["saved"] > 0
    d = j(base, "/api/diff")
    assert d["moved_count"] == 0 and d["new"] == 0 and d["gone"] == 0  # same clustering vs itself


def test_rejects_foreign_host_origin_and_bad_input(base):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, "/", host="evil.example")
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:  # a web page elsewhere must not trigger writes
        get(base, "/api/snapshot", method="POST", headers={"Origin": "https://evil.example"})
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, "/api/symbol?id=abc")
    assert e.value.code == 400
