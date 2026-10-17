"""Import resolution unit tests. Each case is a (raw import, source file, files-on-disk) triple."""
import pytest

from codegraph.store import FileIndex, resolve_import

PY = ["pkg/__init__.py", "pkg/a.py", "pkg/sub/__init__.py", "pkg/sub/b.py", "other/c.py"]
TS = ["src/index.ts", "src/dep.ts", "src/dir/index.ts", "src/dir/thing.tsx", "lib/util.mts"]
GO = ["internal/http/client.go", "internal/http/server.go", "pkg/thing/service.go"]
RS = ["src/lib.rs", "src/model.rs", "src/model/inner.rs", "src/api/v1.rs"]
CPP = ["src/util.h", "src/util.cpp", "include/thing.hpp", "src/main.c"]


def resolve(raw, src, files):
    return resolve_import(raw, src, FileIndex(files))


@pytest.mark.parametrize("raw,expected", [
    ("pkg.a", "pkg/a.py"),
    ("pkg", "pkg/__init__.py"),
    ("pkg.sub.b", "pkg/sub/b.py"),
    ("pkg.sub", "pkg/sub/__init__.py"),
    ("other.c", "other/c.py"),
])
def test_python_absolute(raw, expected):
    assert expected in resolve(raw, "pkg/x.py", PY)


@pytest.mark.parametrize("src,raw,expected", [
    ("pkg/x.py", ".a", "pkg/a.py"),          # same package
    ("pkg/x.py", "..other.c", "other/c.py"),  # up one level
    ("pkg/sub/x.py", ".b", "pkg/sub/b.py"),
    ("pkg/sub/x.py", "..a", "pkg/a.py"),
    ("pkg/sub/x.py", ".", "pkg/sub/__init__.py"),
])
def test_python_relative(src, raw, expected):
    assert expected in resolve(raw, src, PY)


def test_python_relative_missing_module_resolves_to_nothing():
    assert resolve(".nope", "pkg/x.py", PY) == []


def test_python_src_layout_is_found():
    """`from pkg.a import x` must find src/pkg/a.py even though the import omits the src/ prefix."""
    files = ["src/pkg/__init__.py", "src/pkg/a.py"]
    assert resolve("pkg.a", "src/pkg/x.py", files) == ["src/pkg/a.py"]
    assert resolve("pkg", "src/pkg/x.py", files) == ["src/pkg/__init__.py"]


def test_python_unknown_module_resolves_to_nothing():
    assert resolve("pkg.nope", "pkg/x.py", PY) == []


def test_python_does_not_match_wrong_package():
    # `a.py` must not be found by the module path `other.a`
    assert resolve("other.a", "pkg/x.py", PY) == []


@pytest.mark.parametrize("raw,expected", [
    ("./dep", "src/dep.ts"),
    ("./dir", "src/dir/index.ts"),
    ("./dir/thing", "src/dir/thing.tsx"),
    ("../lib/util", "lib/util.mts"),
])
def test_js_relative(raw, expected):
    assert resolve(raw, "src/index.ts", TS) == [expected]


def test_js_bare_specifier_resolves_inside_the_workspace():
    # a monorepo workspace package imported by its path within the repo
    assert resolve("lib/util", "src/index.ts", TS) == ["lib/util.mts"]


def test_js_unknown_specifier_resolves_to_nothing():
    assert resolve("react", "src/index.ts", TS) == []


def test_go_matches_longest_directory_suffix():
    files = GO
    assert "internal/http/client.go" in resolve("github.com/x/y/internal/http", "main.go", files)
    # a suffix that does not exist must not drag in an unrelated directory
    assert resolve("corp/other", "main.go", files) == []


def test_go_stdlib_import_collision_is_a_documented_limitation():
    """`net/http` matches a local `internal/http`: without go.mod the two are indistinguishable.

    Kept as an explicit test so the limitation stays visible rather than being rediscovered.
    """
    assert resolve("net/http", "main.go", GO) == GO[:2]


def test_go_single_segment_directory():
    files = ["thing/service.go"]
    assert resolve("corp/thing", "main.go", files) == ["thing/service.go"]


def test_rust_module_paths():
    assert "src/model.rs" in resolve("crate::model", "src/lib.rs", RS)
    assert "src/model/inner.rs" in resolve("crate::model::inner", "src/lib.rs", RS)
    assert "src/api/v1.rs" in resolve("crate::api::v1::Thing", "src/lib.rs", RS)


def test_rust_group_use():
    assert "src/model.rs" in resolve("use crate::model::{A, B}", "src/lib.rs", RS)


@pytest.mark.parametrize("raw,expected", [
    ('"util.h"', "src/util.h"),
    ("<util.h>", "src/util.h"),
    ("thing.hpp", "include/thing.hpp"),
])
def test_cpp_include(raw, expected):
    assert expected in resolve(raw, "src/main.cpp", CPP)


def test_cpp_prefers_same_directory():
    # src/util.h and include/util.h both exist; the one next to the source wins
    files = CPP + ["include/util.h"]
    assert resolve("util.h", "src/main.cpp", files)[0] == "src/util.h"


def test_cpp_standard_library_header_is_not_matched():
    assert resolve("vector", "src/main.cpp", CPP) == []


def test_resolution_is_not_a_full_scan():
    """Every lookup must be a dict hit; a large file set must not slow resolution down."""
    import time
    big = [f"pkg/mod{i}/file{i}.py" for i in range(4000)] + ["pkg/a.py"]
    idx = FileIndex(big)
    t0 = time.perf_counter()
    for _ in range(200):
        resolve_import("pkg.a", "pkg/x.py", idx)
        resolve_import("pkg.mod7.file7", "pkg/x.py", idx)
    assert (time.perf_counter() - t0) < 0.5


# ---------- collisions that must NOT resolve ----------

def test_stdlib_name_does_not_resolve_to_a_repo_file():
    """`from typing import cast` must not find `pydantic/typing.py`.

    Absolute imports are resolved against import roots, never against the importer's own
    directory, so a same-named module in the importing package cannot capture them.
    """
    files = ["pydantic/_internal/_config.py", "pydantic/typing.py", "sqlalchemy/util/typing.py"]
    idx = FileIndex(files)
    assert resolve_import("typing", "pydantic/_internal/_config.py", idx) == []


def test_import_root_layout_resolves():
    files = ["src/pkg/__init__.py", "src/pkg/a.py"]
    idx = FileIndex(files)
    assert resolve_import("pkg.a", "src/pkg/x.py", idx) == ["src/pkg/a.py"]


def test_sibling_package_module_is_reachable_by_full_path():
    files = ["pydantic/_internal/x.py", "pydantic/typing.py"]
    idx = FileIndex(files)
    assert resolve_import("pydantic.typing", "pydantic/_internal/x.py", idx) == ["pydantic/typing.py"]
