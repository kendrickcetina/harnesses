"""The fictional dataset of the Codex example, and reading tool calls back from traffic."""

import json

import pandas as pd

from harness_lab import traffic
from harness_lab.data import make_llm_usage


def test_dataset_hides_the_stories_of_the_report(tmp_path):
    path = tmp_path / "uso_llm.xlsx"
    df = make_llm_usage(path)
    assert pd.read_excel(path).shape == df.shape  # a real Excel file, readable with pandas
    by_model = df.groupby("modelo")[["peticiones", "coste_eur"]].sum()
    share = by_model / by_model.sum()
    assert share.loc["modelo-frontera", "peticiones"] < 0.2 < 0.8 < share.loc["modelo-frontera", "coste_eur"]
    assert df.groupby("departamento").coste_eur.sum().idxmax() == "Ingeniería"
    month = pd.to_datetime(df.fecha).dt.month
    soporte = df[df.departamento == "Soporte"].groupby(month).peticiones.sum()
    assert soporte[4] > 1.8 * soporte[3]  # chatbot from April
    mk = df[df.departamento == "Marketing"]
    errors = mk.groupby(month).errores.sum() / mk.groupby(month).peticiones.sum()
    assert errors.idxmax() == 3 and errors[3] > 0.1  # broken integration in March
    assert make_llm_usage()["coste_eur"].sum() == df["coste_eur"].sum()  # deterministic


def test_tool_calls_from_codex_traffic(tmp_path):
    saved = {
        "api": "responses",
        "request": {
            "input": [
                {"type": "message", "role": "user", "content": "analiza"},
                {"type": "function_call", "name": "exec_command", "arguments": json.dumps({"cmd": "python3 a.py"})},
                {"type": "function_call_output", "output": "ok"},
                {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch"},
            ]
        },
        "response": {},
    }
    path = tmp_path / "x-responses.json"
    path.write_text(json.dumps(saved))
    assert traffic.tool_calls(path) == [
        ("exec_command", json.dumps({"cmd": "python3 a.py"})),
        ("apply_patch", "*** Begin Patch"),
    ]


def test_show_tool_calls_prints_commands(tmp_path, capsys):
    saved = {
        "api": "messages",
        "request": {
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "name": "bash", "input": {"command": "ls"}}]}
            ]
        },
        "response": {},
    }
    path = tmp_path / "x-messages.json"
    path.write_text(json.dumps(saved))
    traffic.show_tool_calls(path)
    assert capsys.readouterr().out.strip() == "$ ls"
