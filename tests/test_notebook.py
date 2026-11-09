"""Run the whole tutorial notebook end to end.

* ``test_tutorial_runs_offline``: scripted model, no GPU, no downloads. Always runs.
* ``test_tutorial_runs_on_gpu``: the real Qwen3.5-4B on a GPU, exactly what students run
  (~3 min, downloads ~9 GB the first time). Opt in with ``HARNESS_TEST_GPU=1``; add
  ``HARNESS_TEST_CLIS=1`` to also run the Codex / Claude Code cells if they are installed:

      HARNESS_TEST_GPU=1 uv run pytest tests/test_notebook.py -k gpu -s
      HARNESS_TEST_GPU=1 HARNESS_TEST_CLIS=1 uv run pytest tests/test_notebook.py -k gpu -s
"""

import os
import shutil
from pathlib import Path

import nbformat
import pytest
from docx import Document
from nbclient import NotebookClient

from harness_lab.server.launch import free_port

NOTEBOOK = Path(__file__).resolve().parents[1] / "tutorial.ipynb"


def run_notebook(tmp_path, monkeypatch, scripted: bool, cell_timeout: int, clis: bool = False):
    """Execute tutorial.ipynb with every generated file under ``tmp_path``. Returns the cells."""
    port = free_port()
    env = {
        "HARNESS_SKIP_CLIS": "0" if clis else "1",
        "HARNESS_PORT": str(port),
        "HARNESS_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "HARNESS_WORKSPACE": str(tmp_path / "workspace"),
        "HARNESS_OUTPUTS": str(tmp_path / "outputs"),
        "HARNESS_DSH_HOME": str(tmp_path / "dsh_home"),
    }
    if scripted:
        env["HARNESS_SCRIPTED"] = "1"
    else:
        monkeypatch.delenv("HARNESS_SCRIPTED", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    resources = {"metadata": {"path": str(NOTEBOOK.parent)}}
    try:
        NotebookClient(nb, timeout=cell_timeout, kernel_name="python3", resources=resources).execute()
    finally:
        # Keep the executed notebook for inspection, whatever happened.
        nbformat.write(nb, tmp_path / "tutorial.executed.ipynb")
    return nb.cells


def cell_text(cell) -> str:
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if "text" in o)


def find_cell(cells, snippet: str):
    return next(c for c in cells if c.cell_type == "code" and snippet in c.source)


def all_text(cells) -> str:
    return "\n".join(cell_text(c) for c in cells if c.cell_type == "code")


def errors(cells) -> list[str]:
    return [
        f"{o.get('ename')}: {o.get('evalue')}"
        for c in cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]


def test_tutorial_runs_offline(tmp_path, monkeypatch):
    cells = run_notebook(tmp_path, monkeypatch, scripted=True, cell_timeout=300)
    outputs = all_text(cells)
    assert not errors(cells)
    assert "Server ready" in outputs
    assert "[step 1] create_word_document" in outputs  # mini harness
    assert "mcp__lab__create_word_document" in outputs  # dsh traffic summary
    assert (tmp_path / "workspace" / "harness.docx").exists()
    assert (tmp_path / "workspace" / "datos" / "uso_llm.xlsx").exists()  # data for the Codex example
    assert "--- Cálculo directo con pandas ---" in outputs


def _gpu_available() -> bool:
    import torch

    return torch.cuda.is_available()


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("HARNESS_TEST_GPU") != "1", reason="set HARNESS_TEST_GPU=1 (real model, ~6 min)")
@pytest.mark.skipif(not _gpu_available(), reason="needs a CUDA GPU")
def test_tutorial_runs_on_gpu(tmp_path, monkeypatch):
    clis = os.environ.get("HARNESS_TEST_CLIS") == "1"
    cells = run_notebook(tmp_path, monkeypatch, scripted=False, cell_timeout=3600, clis=clis)
    check_real_run(cells, tmp_path)
    if clis and shutil.which("codex"):
        check_codex_report(cells, tmp_path)


def tools_count(text: str) -> int:
    """Number after 'tools: ' in a traffic.show() printout."""
    return int(text.split("tools: ")[1].split()[0])


def check_real_run(cells, run_dir: Path) -> None:
    """What every section of the tutorial must show with the real model."""
    workspace = run_dir / "workspace"
    assert not errors(cells), errors(cells)

    # 0. Setup: the real model, not the scripted one, on the GPU
    setup = cell_text(find_cell(cells, "config.MODEL_ID)"))
    assert "Simulado:     False" in setup and "GPU:          no" not in setup
    server_log = (run_dir / "outputs" / "server.log").read_text()
    assert "Loaded Qwen/Qwen3.5-4B" in server_log and "on cuda" in server_log

    # 2. One call: real text back, and a structured tool call parsed from Qwen's markup
    assert len(cell_text(find_cell(cells, "Explica en una frase")).strip()) > 10
    tool_cell = cell_text(find_cell(cells, "tools = [{"))
    assert "finish_reason: tool_calls" in tool_cell
    assert "create_word_document {" in tool_cell

    # 3. Mini harness: the loop executed the Word tool and produced a real .docx
    assert "[step 1] create_word_document" in cell_text(find_cell(cells, "MiniHarness()"))
    assert any(workspace.glob("*.docx"))

    # 4. DeepSeek Harness: bash with the minimal profile (no MCP tools), then our MCP Word tool
    assert "-> tool bash(" in cell_text(find_cell(cells, "demo-bash"))
    minimal = cell_text(find_cell(cells, 'traffic.show(api="messages");'))
    assert "bash" in minimal and "mcp__lab" not in minimal
    assert "-> tool mcp__lab__create_word_document(" in cell_text(find_cell(cells, "demo-word"))
    doc = Document(str(workspace / "harness.docx"))
    assert len("".join(p.text for p in doc.paragraphs)) > 50
    # the full profile sends many more tools than the minimal one
    assert tools_count(cell_text(find_cell(cells, 'profile="sdk"'))) > tools_count(minimal)


def check_codex_report(cells, run_dir: Path) -> None:
    """Codex read the Excel by running Python and wrote the Word report with real numbers."""
    from harness_lab.data import make_llm_usage

    commands = cell_text(find_cell(cells, "show_tool_calls"))
    assert "python" in commands, commands  # it ran code, it did not guess
    report = run_dir / "workspace" / "informe_uso_llm.docx"
    assert report.exists(), cell_text(find_cell(cells, "run_codex"))
    doc = Document(str(report))
    text = "\n".join(p.text for p in doc.paragraphs) + "\n".join(
        cell.text for table in doc.tables for row in table.rows for cell in row.cells
    )
    assert "Ingeniería" in text and doc.tables, text[:500]
    # The real total cost appears, in any usual format: 1677.60, 1,677.60 or 1.677,60
    whole = int(make_llm_usage()["coste_eur"].sum())
    assert any(f in text for f in (str(whole), f"{whole:,}", f"{whole:,}".replace(",", "."))), text[:800]
