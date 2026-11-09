import pytest
from docx import Document

from harness_lab.mini_harness import MiniHarness, tool_schema
from harness_lab.tools import docx_tools

from .conftest import call

MARKDOWN = """# Qué es un harness

Un **harness** es el bucle alrededor del modelo.

- Herramientas
- Memoria

1. Primero
2. Segundo

| Capa | Ejemplo |
|------|---------|
| Modelo | Qwen3.5-4B |
"""


def test_markdown_to_docx_roundtrip(workspace):
    path = docx_tools.create_word_document("informe", MARKDOWN, title="Informe")
    assert path.endswith("informe.docx")
    doc = Document(path)
    styles = [p.style.name for p in doc.paragraphs]
    assert "Title" in styles and "Heading 1" in styles and "List Bullet" in styles and "List Number" in styles
    assert doc.tables[0].cell(1, 1).text == "Qwen3.5-4B"
    bold = [r.text for p in doc.paragraphs for r in p.runs if r.bold]
    assert "harness" in bold
    text = docx_tools.read_word_document("informe.docx")
    assert "# Qué es un harness" in text and "| Modelo | Qwen3.5-4B |" in text
    assert "informe.docx" in docx_tools.list_workspace()


def test_path_traversal_is_rejected(workspace):
    with pytest.raises(ValueError):
        docx_tools.create_word_document("../escape.docx", "x")


def test_tool_schema_from_signature():
    schema = tool_schema(docx_tools.create_word_document)["function"]
    assert schema["name"] == "create_word_document"
    assert schema["parameters"]["required"] == ["filename", "markdown"]
    assert schema["parameters"]["properties"]["title"] == {"type": "string"}


def test_mini_harness_loop_creates_document(workspace, live_server):
    script = [
        call("create_word_document", filename="nota.docx", markdown="# Hola\n\nTexto."),
        call("read_word_document", filename="nota.docx"),
        "He creado nota.docx.",
    ]
    engine, _ = live_server(script)
    harness = MiniHarness(model="scripted", verbose=False)
    answer = harness.run("Crea un documento")
    assert answer == "He creado nota.docx."
    assert (workspace / "nota.docx").exists()
    roles = [m["role"] for m in harness.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert "# Hola" in harness.messages[5]["content"]  # read_word_document result
    assert len(engine.calls) == 3


def test_mini_harness_reports_tool_errors(workspace, live_server):
    live_server([call("read_word_document", filename="missing.docx"), "No existe."])
    harness = MiniHarness(model="scripted", verbose=False)
    assert harness.run("lee missing.docx") == "No existe."
    assert harness.messages[3]["content"].startswith("Error:")
