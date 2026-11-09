"""Run the whole tutorial notebook with the scripted model (no GPU, no downloads)."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

from harness_lab.server.launch import free_port

NOTEBOOK = Path(__file__).resolve().parents[1] / "tutorial.ipynb"


def test_tutorial_runs_offline(tmp_path, monkeypatch):
    port = free_port()
    env = {
        "HARNESS_SCRIPTED": "1",
        "HARNESS_SKIP_CLIS": "1",
        "HARNESS_PORT": str(port),
        "HARNESS_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "HARNESS_WORKSPACE": str(tmp_path / "workspace"),
        "HARNESS_OUTPUTS": str(tmp_path / "outputs"),
        "HARNESS_DSH_HOME": str(tmp_path / "dsh_home"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    resources = {"metadata": {"path": str(NOTEBOOK.parent)}}
    NotebookClient(nb, timeout=300, kernel_name="python3", resources=resources).execute()
    outputs = "\n".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if "text" in o
    )
    assert "Server ready" in outputs
    assert "[step 1] create_word_document" in outputs  # mini harness
    assert "mcp__lab__create_word_document" in outputs  # dsh traffic summary
    assert (tmp_path / "workspace" / "harness.docx").exists()
    assert not any(o.get("output_type") == "error" for c in nb.cells for o in c.get("outputs", []))
