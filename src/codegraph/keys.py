"""String-keyed hidden links: routes, event names and config keys.

Two symbols that mention the same string literal in a recognisable role (a route path registered
here and requested there, an event emitted here and listened to there) are related even though
no call edge connects them. Recognition is table-driven: add a `Pattern` row to support a new
framework. The parser feeds this module the callee text and the first string argument of every
call; resolution (matching roles across files) lives in links.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PY = frozenset({"python"})
JS = frozenset({"javascript", "typescript", "tsx"})
ANY = PY | JS

_HTTP = "get|post|put|patch|delete|head|options"
_STRING = re.compile(r"""^[rbfuRBFU]{0,2}(\"\"\"|'''|"|'|`)(.*)\1$""", re.DOTALL)
# events every Node/DOM program emits and listens to: linking on them would connect unrelated code
COMMON_EVENTS = frozenset(
    "error data end close exit open message connect finish drain readable line abort timeout "
    "click change input submit keydown keyup load resize sigint sigterm uncaughtexception "
    "unhandledrejection".split())


def _route(k: str) -> bool:
    return k.startswith("/") and len(k) > 1 and " " not in k


def _event(k: str) -> bool:
    return len(k) >= 3 and k.lower() not in COMMON_EVENTS and re.fullmatch(r"[\w.:/@-]+", k) is not None


def _config(k: str) -> bool:
    return len(k) >= 3 and re.fullmatch(r"[\w.:-]+", k) is not None


@dataclass(frozen=True)
class Pattern:
    langs: frozenset
    callee: re.Pattern  # matched against the whole callee text, e.g. `app.get`
    ns: str  # namespace of the key: keys only match inside one namespace
    role: str  # see links.KEY_ROLES for how roles pair up
    valid: object  # str -> bool, rejects keys that are too generic to mean anything
    decorator: bool | None = None  # True: only as a decorator, False: only as a plain call


def _p(langs, callee, ns, role, valid, decorator=None) -> Pattern:
    return Pattern(langs, re.compile(callee), ns, role, valid, decorator)


# Add rows here to teach codegraph a new framework. Roles per namespace: see links.KEY_ROLES.
PATTERNS: tuple[Pattern, ...] = (
    # routes: registered (provides) vs requested (uses)
    _p(PY, rf"^\w+\.({_HTTP}|route|websocket|api_route)$", "route", "provides", _route, decorator=True),
    _p(PY, rf"^(?:\w+\.)*(client|requests|httpx|session|http|api)\.({_HTTP})$", "route", "uses", _route,
       decorator=False),
    _p(JS, rf"^(app|router|server|\w+Router)\.({_HTTP}|all)$", "route", "provides", _route),
    _p(JS, rf"^(fetch|axios|ky|\$http)$|^(?:\w+\.)*(axios|client|http|api|request|supertest)\.({_HTTP})$",
       "route", "uses", _route),
    # events: emitted vs listened to
    _p(ANY, r"(?:^|\.)(emit|dispatch|dispatchEvent|trigger|publish)$", "event", "emits", _event),
    _p(ANY, r"(?:^|\.)(on|once|addListener|addEventListener|subscribe|prependListener)$",
       "event", "listens", _event),
    # config keys: everything that reads the same key is related
    _p(PY, r"(?:^|\.)(getenv|environ\.get|environ\.setdefault)$", "config", "reads", _config),
    _p(ANY, r"^(?:self\.|this\.)?(config|settings|cfg|conf|env)\.(get|getint|getboolean|require)$",
       "config", "reads", _config),
)


def string_value(text: str) -> str | None:
    """The contents of a plain string literal, or None for anything computed or interpolated."""
    m = _STRING.match(text.strip())
    if not m:
        return None
    body = m.group(2)
    if m.group(1) == "`" and "${" in body:
        return None
    if text.lstrip()[:1] in "fF" and "{" in body:  # f-string: not a constant
        return None
    return body


def match(lang: str, callee: str, key: str, decorator: bool) -> list[tuple[str, str, str]]:
    """Every (namespace, key, role) a call `callee("key", ...)` contributes."""
    out = []
    for p in PATTERNS:
        if lang not in p.langs or (p.decorator is not None and p.decorator != decorator):
            continue
        if p.callee.search(callee) and p.valid(key):
            out.append((p.ns, key, p.role))
    return out
