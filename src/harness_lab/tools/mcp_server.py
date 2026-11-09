"""Our own tools, exposed through the Model Context Protocol (MCP).

Any MCP-capable harness can use them: DeepSeek Harness, Codex and Claude Code all speak
MCP. The harness starts this program and talks to it over stdin/stdout:

    uv run harness-mcp

Inside the harness the tools appear as ``mcp__lab__create_word_document`` etc.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from harness_lab import config
from harness_lab.tools import docx_tools

server = MCPServer(
    "lab",
    instructions=(
        "Tools from the harness lab. Word documents are written in Markdown and saved in the "
        f"workspace ({config.WORKSPACE_DIR}). Use create_word_document to produce .docx files."
    ),
)


@server.tool()
def create_word_document(filename: str, markdown: str, title: str = "") -> str:
    """Create a Word (.docx) document from Markdown text.

    Args:
        filename: file name inside the workspace, e.g. "report.docx".
        markdown: document body. Supports #/##/### headings, paragraphs, - bullets,
            1. numbered lists, **bold**, *italic* and | pipe | tables |.
        title: optional title shown at the top of the document.
    """
    path = docx_tools.create_word_document(filename, markdown, title)
    return f"Saved {path}"


@server.tool()
def read_word_document(filename: str) -> str:
    """Read the text of a Word (.docx) document in the workspace."""
    return docx_tools.read_word_document(filename)


@server.tool()
def list_workspace() -> str:
    """List the files in the workspace."""
    return docx_tools.list_workspace()


@server.tool()
def compare_checkpoints(base: str, edited: str, top: int = 10) -> str:
    """Weight-space programming: list which tensors differ between two checkpoints.

    Args:
        base: Hugging Face repo id or local folder of the original model.
        edited: repo id or folder of the edited model.
        top: how many changed tensors to report.
    """
    from harness_lab.wsp.diff import compare_checkpoints as compare
    from harness_lab.wsp.diff import format_report

    return format_report(compare(base, edited)[:top])


def main() -> None:
    # The harness may start us from any directory: keep paths anchored to the repo.
    os.chdir(Path(config.REPO_ROOT))
    server.run("stdio")


if __name__ == "__main__":
    main()
