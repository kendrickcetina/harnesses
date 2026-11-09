"""End to end, offline: real harness binaries talking to the server with a scripted model.

DeepSeek Harness ships with the Python package, so its test always runs. Codex and Claude
Code are separate programs: their tests run only if installed and HARNESS_TEST_CLIS=1.
"""

import os
import shutil

import pytest
from docx import Document

from harness_lab import clis, dsh

from .conftest import call

DOC = "# Harness\n\n- bucle\n- herramientas\n\n| capa | ejemplo |\n|---|---|\n| modelo | Qwen |"


def doc_script(request):
    """Call our MCP Word tool once, then answer with the tool's output."""
    names = [t["function"]["name"] for t in request.tools or []]
    last = request.messages[-1]
    if last["role"] != "tool":
        target = next(n for n in names if n.endswith("create_word_document"))
        return "Voy a crear el documento.\n\n" + call(target, filename="harness.docx", markdown=DOC)
    return "Listo: " + last["content"]


def test_deepseek_harness_uses_mcp_word_tool(workspace, live_server, tmp_path):
    engine, _ = live_server(doc_script)
    result = dsh.run_task(
        "Crea harness.docx", workspace=workspace, dsh_home=tmp_path / "dsh_home", verbose=False, session_id="test-001"
    )
    assert result.finish_reason == "completed", result
    assert result.final_response.startswith("Listo: Saved")
    doc = Document(str(workspace / "harness.docx"))
    assert doc.tables[0].cell(1, 1).text == "Qwen"
    # What the harness sent: its own tools + ours, via the Anthropic Messages API
    tools = [t["function"]["name"] for t in engine.calls[0].tools]
    assert {"bash", "read_image", "working_directory", "mcp__lab__create_word_document"} <= set(tools)
    assert "DeepSeek Harness" in engine.calls[0].messages[0]["content"]


def test_deepseek_harness_without_tools_answers_directly(workspace, live_server, tmp_path):
    live_server(["2 + 2 = 4"])
    result = dsh.run_task(
        "¿Cuánto es 2+2?",
        workspace=workspace,
        dsh_home=tmp_path / "home2",
        mcp=False,
        verbose=False,
        session_id="test-002",
    )
    assert result.final_response == "2 + 2 = 4"


clis_enabled = pytest.mark.skipif(os.environ.get("HARNESS_TEST_CLIS") != "1", reason="set HARNESS_TEST_CLIS=1")


@clis_enabled
@pytest.mark.skipif(not shutil.which("codex"), reason="codex not installed")
def test_codex_uses_mcp_word_tool(workspace, live_server):
    live_server(doc_script)
    out = clis.run_codex("Crea harness.docx", timeout=300)
    assert (workspace / "harness.docx").exists(), out


@clis_enabled
@pytest.mark.skipif(not shutil.which("claude"), reason="claude not installed")
def test_claude_code_uses_mcp_word_tool(workspace, live_server):
    live_server(doc_script)
    out = clis.run_claude("Crea harness.docx", timeout=300)
    assert (workspace / "harness.docx").exists(), out
