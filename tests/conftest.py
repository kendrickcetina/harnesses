"""Offline fixtures: a scripted "model" behind the real server. No GPU, no downloads."""

from __future__ import annotations

import pytest

from harness_lab import config
from harness_lab.server.app import create_app
from harness_lab.server.engine import ScriptedEngine
from harness_lab.server.launch import free_port, serve_in_thread


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE_DIR", ws)
    monkeypatch.setenv("HARNESS_WORKSPACE", str(ws))
    return ws


@pytest.fixture
def live_server(monkeypatch):
    """``live_server(script)`` -> (engine, base_url). Points config.BASE_URL at it."""
    servers = []

    def start(script):
        engine = ScriptedEngine(script)
        port = free_port()
        servers.append(serve_in_thread(create_app(engine), port))
        base_url = f"http://127.0.0.1:{port}/v1"
        monkeypatch.setattr(config, "BASE_URL", base_url)
        return engine, base_url

    yield start
    for s in servers:
        s.should_exit = True


def call(name: str, **params) -> str:
    """Raw Qwen3.5 tool-call markup, as the real model writes it."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"
