"""Word documents from Markdown, with python-docx.

The model writes Markdown (easy for any LLM) and this tool turns it into a real .docx.
Supported: ``#``/``##``/``###`` headings, paragraphs, ``-``/``*`` bullets, ``1.`` numbered
items, ``**bold**`` and ``*italic*`` runs, and pipe tables (``| a | b |``).
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt

from harness_lab import config

_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def _safe_path(filename: str, workspace: Path | None = None) -> Path:
    """Resolve ``filename`` inside the workspace; refuse anything that escapes it."""
    root = (workspace or config.WORKSPACE_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = (root / filename).resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"{filename!r} is outside the workspace {root}")
    return path


def _add_runs(paragraph, text: str) -> None:
    for piece in _INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            paragraph.add_run(piece[2:-2]).bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = paragraph.add_run(piece[1:-1])
            run.font.name = "Courier New"
        elif piece.startswith("*") and piece.endswith("*"):
            paragraph.add_run(piece[1:-1]).italic = True
        else:
            paragraph.add_run(piece)


def _table(document, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]  # drop |---|
    width = max(len(r) for r in cells)
    table = document.add_table(rows=len(cells), cols=width)
    table.style = "Table Grid"
    for i, row in enumerate(cells):
        for j in range(width):
            cell = table.cell(i, j)
            cell.text = ""
            _add_runs(cell.paragraphs[0], row[j] if j < len(row) else "")
            if i == 0:
                for run in cell.paragraphs[0].runs:
                    run.bold = True


def markdown_to_docx(markdown: str, title: str | None = None) -> Document:
    document = Document()
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(11)
    if title:
        document.add_heading(title, level=0)
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            _table(document, block)
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)", line)
        bullet = re.match(r"^\s*[-*]\s+(.*)", line)
        number = re.match(r"^\s*\d+[.)]\s+(.*)", line)
        if heading:
            document.add_heading(heading.group(2).strip(), level=len(heading.group(1)))
        elif bullet:
            _add_runs(document.add_paragraph(style="List Bullet"), bullet.group(1))
        elif number:
            _add_runs(document.add_paragraph(style="List Number"), number.group(1))
        else:
            _add_runs(document.add_paragraph(), line.strip())
        i += 1
    return document


def create_word_document(filename: str, markdown: str, title: str = "") -> str:
    """Create a .docx file in the workspace from Markdown. Returns the saved path."""
    if not filename.endswith(".docx"):
        filename += ".docx"
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_to_docx(markdown, title or None).save(path)
    return str(path)


def read_word_document(filename: str) -> str:
    """Return the text of a .docx in the workspace (headings marked with #, tables with |)."""
    document = Document(str(_safe_path(filename)))
    out = []
    for p in document.paragraphs:
        if not p.text.strip():
            continue
        style = p.style.name if p.style is not None else ""
        if style == "Title":
            out.append(f"TITLE: {p.text}")
        elif style.startswith("Heading"):
            level = style.split()[-1]
            out.append("#" * int(level) + " " + p.text if level.isdigit() else p.text)
        elif style.startswith("List"):
            out.append(f"- {p.text}")
        else:
            out.append(p.text)
    for t in document.tables:
        for row in t.rows:
            out.append("| " + " | ".join(c.text for c in row.cells) + " |")
    return "\n".join(out)


def list_workspace() -> str:
    """List the files in the workspace."""
    root = config.WORKSPACE_DIR
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    return "\n".join(files) or "(empty)"
