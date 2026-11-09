"""Look at what a harness actually sends to the model (saved by the server in outputs/traffic)."""

from __future__ import annotations

import json
from pathlib import Path

from harness_lab import config

TRAFFIC_DIR = config.OUTPUTS_DIR / "traffic"


def files(api: str | None = None, directory: Path | None = None) -> list[Path]:
    paths = sorted((directory or TRAFFIC_DIR).glob("*.json"))
    return [p for p in paths if api is None or p.stem.endswith(api)]


def summary(path: Path) -> dict:
    """API, number of messages, tool names and system prompt of one saved request."""
    data = json.loads(path.read_text())
    req = data["request"]
    api = data["api"]
    if api == "messages":
        system = req.get("system") or ""
        system = system if isinstance(system, str) else " ".join(b.get("text", "") for b in system)
        tools = [t.get("name") for t in req.get("tools") or []]
        n_messages = len(req.get("messages") or [])
    elif api == "responses":
        system = req.get("instructions") or ""
        tools = []
        for t in req.get("tools") or []:
            if t.get("type") == "namespace":
                tools += [f"{t['name']}__{x['name']}" for x in t.get("tools", [])]
            else:
                tools.append(t.get("name") or t.get("type"))
        n_messages = len(req.get("input") or [])
    else:
        msgs = req.get("messages") or []
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        tools = [t["function"]["name"] for t in req.get("tools") or []]
        n_messages = len(msgs)
    return {
        "file": path.name,
        "api": api,
        "messages": n_messages,
        "tools": tools,
        "system_chars": len(system),
        "system": system,
    }


def show(path: Path | None = None, api: str | None = None, system_preview: int = 600) -> dict:
    """Print a readable summary of a request (the latest one by default)."""
    path = path or files(api)[-1]
    s = summary(path)
    print(f"{s['file']}\nAPI: {s['api']}   messages: {s['messages']}   tools: {len(s['tools'])}")
    print("tools:", ", ".join(s["tools"]))
    print(f"system prompt ({s['system_chars']} chars):\n{s['system'][:system_preview]}")
    return s


def tool_calls(path: Path | None = None, api: str | None = None) -> list[tuple[str, str]]:
    """(tool, arguments) the model asked for, in order, from the history of a saved request
    (the latest one by default). For Codex, the commands it ran with ``exec_command``."""
    path = path or files(api)[-1]
    data = json.loads(path.read_text())
    req, calls = data["request"], []
    if data["api"] == "responses":
        for item in req.get("input") or []:
            if item.get("type") == "function_call":
                calls.append((item.get("name", ""), item.get("arguments", "")))
            elif item.get("type") == "custom_tool_call":
                calls.append((item.get("name", ""), item.get("input", "")))
    elif data["api"] == "messages":
        for m in req.get("messages") or []:
            for b in m.get("content") if isinstance(m.get("content"), list) else []:
                if b.get("type") == "tool_use":
                    calls.append((b.get("name", ""), json.dumps(b.get("input") or {}, ensure_ascii=False)))
    else:
        for m in req.get("messages") or []:
            for c in m.get("tool_calls") or []:
                calls.append((c["function"]["name"], c["function"].get("arguments", "")))
    return calls


def show_tool_calls(path: Path | None = None, api: str | None = None, width: int = 400) -> None:
    """Print each tool call; shell commands (``cmd``/``command``) are shown as ``$ command``."""
    for name, args in tool_calls(path, api):
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            parsed = None
        command = parsed.get("cmd") or parsed.get("command") if isinstance(parsed, dict) else None
        if isinstance(command, list):
            command = " ".join(command)
        print(f"$ {command[:width]}" if command else f"{name}({str(args)[:width]})")
