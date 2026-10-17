"""Non-code inputs: Markdown, SQL schemas and (with the `pdf` extra) PDFs.

Static like the code parser: no LLM, nothing leaves the machine. Each document becomes symbols the
rest of the graph already understands, so search, outline, PageRank and communities work on them:

  - Markdown heading  -> `section`, nested by heading level; `code spans` in its body become
                         `mentions` edges to the code symbols they name
  - SQL CREATE ...    -> `table` / `view` / `function` / `procedure`; FROM / JOIN / REFERENCES
                         inside the statement become `mentions` edges to other SQL objects
  - PDF page          -> `section` carrying the page text (line numbers are page numbers)
"""
from __future__ import annotations

import importlib.util
import io
import re

from .parser import Parsed, Sym

DOC_EXT = {".md": "markdown", ".markdown": "markdown", ".sql": "sql"}
if importlib.util.find_spec("pypdf"):
    DOC_EXT[".pdf"] = "pdf"
DOC_LANGS = frozenset(DOC_EXT.values())

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
DOTTED = re.compile(r"[A-Za-z_]\w*(?:(?:\.|::|->)[A-Za-z_]\w*)*")
MD_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")

SQL_CREATE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:(?:GLOBAL\s+|LOCAL\s+)?(?:TEMP|TEMPORARY)\s+|MATERIALIZED\s+)?"
    r"(TABLE|VIEW|FUNCTION|PROCEDURE)\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"`\[\]]+)", re.I)
SQL_REF = re.compile(r"\b(?:FROM|JOIN|REFERENCES|INTO|UPDATE)\s+([\w.\"`\[\]]+)", re.I)
SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
SQL_KEYWORDS = frozenset("select set values lateral unnest generate_series".split())
MIN_MENTION = 3  # shorter names are noise (`i`, `db`) and would link to anything
PDF_TEXT = 2000  # characters of a page kept as its searchable doc


def _first_line(text: str, n: int = 200) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:n]
    return ""


def _mentions(text: str) -> list[str]:
    """Identifiers named in `code spans`. `Store.resolve()` yields Store and resolve."""
    out: dict[str, None] = {}
    for span in CODE_SPAN.findall(text):
        for dotted in DOTTED.findall(span):
            for part in re.split(r"\.|::|->", dotted):
                if len(part) >= MIN_MENTION:
                    out[part] = None
    return list(out)


def parse_markdown(src: bytes, stem: str) -> Parsed:
    lines = src.decode("utf-8", errors="replace").splitlines()
    heads: list[tuple[int, int, str]] = []  # (line, level, title)
    in_fence = False
    for n, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := HEADING.match(line)):
            heads.append((n, len(m.group(1)), m.group(2)))
    p = Parsed()
    stack: list[tuple[int, Sym]] = []  # (level, symbol) of the open ancestors
    for k, (n, level, title) in enumerate(heads):
        while stack and stack[-1][0] >= level:
            stack.pop()
        # a section runs until the next heading of the same or a higher level
        end = next((m - 1 for m, lv, _ in heads[k + 1:] if lv <= level), len(lines))
        parent = stack[-1][1] if stack else None
        body = "\n".join(lines[n:end])
        sym = Sym(id=k, name=title, qualname=f"{parent.qualname} > {title}" if parent else title,
                  kind="section", start=n, end=max(end, n), sig="#" * level + " " + title,
                  doc=_first_line(body), parent=parent.id if parent else None)
        p.syms.append(sym)
        stack.append((level, sym))
        p.refs.extend((k, name, "mentions") for name in _mentions(body))
    # relative links to other files: resolved against the file set like an import
    p.imports.extend(dict.fromkeys(MD_LINK.findall("\n".join(lines))))
    return p


def _strip_sql_comments(text: str) -> str:
    """Blank comments but keep newlines, so line numbers still match the file."""
    return SQL_COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _sql_name(raw: str) -> str:
    return re.sub(r"[\"`\[\]]", "", raw).rsplit(".", 1)[-1]


def parse_sql(src: bytes, stem: str) -> Parsed:
    text = _strip_sql_comments(src.decode("utf-8", errors="replace"))
    total = text.count("\n") + 1
    found = list(SQL_CREATE.finditer(text))
    p = Parsed()
    for k, m in enumerate(found):
        name = _sql_name(m.group(2))
        start = text.count("\n", 0, m.start()) + 1
        stop = found[k + 1].start() if k + 1 < len(found) else len(text)
        # ponytail: a statement runs to the next CREATE; `;` inside $$ bodies makes `;` unusable.
        # Trailing ALTER / INSERT statements attach to the object above them.
        body = text[m.end():stop]
        end = max(start, start + text.count("\n", m.start(), stop) - (1 if stop < len(text) else 0))
        kind = m.group(1).lower()
        p.syms.append(Sym(id=k, name=name, qualname=name, kind=kind, start=start, end=min(end, total),
                          sig=" ".join(m.group(0).split()), doc="", parent=None))
        seen = {name}
        for r in SQL_REF.finditer(body):
            ref = _sql_name(r.group(1))
            if ref.lower() not in SQL_KEYWORDS and ref not in seen and len(ref) >= 2:
                seen.add(ref)
                p.refs.append((k, ref, "mentions"))
    return p


def parse_pdf(src: bytes, stem: str) -> Parsed:
    from pypdf import PdfReader  # optional extra, see DOC_EXT

    p = Parsed()
    for n, page in enumerate(PdfReader(io.BytesIO(src)).pages, 1):
        text = " ".join((page.extract_text() or "").split())
        if not text:
            continue
        title = f"{stem} p{n}: {text[:60]}"
        p.syms.append(Sym(id=len(p.syms), name=title, qualname=title, kind="section", start=n, end=n,
                          sig=f"page {n}", doc=text[:PDF_TEXT], parent=None))
    return p


PARSERS = {"markdown": parse_markdown, "sql": parse_sql, "pdf": parse_pdf}


def parse_doc(src: bytes, lang: str, stem: str) -> Parsed:
    return PARSERS[lang](src, stem)
