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


def get(base, path, host=None):
    req = urllib.request.Request(base + path, headers={"Host": host} if host else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def test_page_and_overview(base):
    assert b"page" in get(base, "/")
    d = json.loads(get(base, "/api/overview"))
    assert d["files"] and d["total"] >= len(d["files"])
    assert any(f["id"].endswith("service.py") for f in d["files"])
    assert d["links"]


def test_file_symbol_impact_search(base):
    path = next(f["id"] for f in json.loads(get(base, "/api/overview"))["files"] if f["id"].endswith("core.py"))
    d = json.loads(get(base, f"/api/file?path={path}"))
    assert d["symbols"]
    ids = {s["id"] for s in d["symbols"]} | {s["id"] for s in d["far"]}
    assert all(e["s"] in ids and e["t"] in ids for e in d["edges"])
    sid = d["symbols"][0]["id"]
    sym = json.loads(get(base, f"/api/symbol?id={sid}"))
    assert sym["qn"] and "source" in sym
    assert "symbols" in json.loads(get(base, f"/api/impact?id={sid}"))
    hits = json.loads(get(base, f"/api/search?q={d['symbols'][0]['name']}"))["symbols"]
    assert hits


def test_rejects_foreign_host_and_bad_input(base):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, "/", host="evil.example")
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base, "/api/symbol?id=abc")
    assert e.value.code == 400
