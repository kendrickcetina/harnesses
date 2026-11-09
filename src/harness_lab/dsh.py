"""Run DeepSeek Harness (dsh) against the local Qwen server, with our MCP tools.

    uv run harness-dsh "Create a Word document summarising what a harness is."
    uv run harness-dsh --profile sdk "List the files and explain the repo."
    uv run harness-dsh --no-mcp "What is 2 + 2?"

dsh is a TypeScript harness; ``pip install deepseek-harness-sdk`` ships its native runtime,
so no Node.js is needed. Its DeepSeek adapter speaks the Anthropic Messages API, so we
point ``DEEPSEEK_BASE_URL`` at our server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from harness_lab import config

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant running inside DeepSeek Harness. Work in the current "
    "directory. Use your tools, check your results, and answer briefly."
)


def write_patch(path: Path, mcp: bool = True) -> Path:
    """Profile patch: switch off DeepSeek telemetry rows and register our MCP server."""
    rows: list[dict[str, Any]] = [
        # These rows upload session logs / package inventory to DeepSeek. Not wanted here.
        {"id": "session-log-deepseek", "disabled": True},
        {"id": "plugin-package-inventory-deepseek", "disabled": True},
    ]
    if mcp:
        rows.append(
            {
                "insert": [
                    {
                        "id": "mcp-lab",
                        "name": "@deepseek-ai/dsh-mcp-client",
                        "config": {
                            "serverName": "lab",
                            "transport": "stdio",
                            "command": sys.executable,
                            "args": ["-m", "harness_lab.tools.mcp_server"],
                            "env": {"HARNESS_WORKSPACE": str(config.WORKSPACE_DIR)},
                            "cwd": str(config.REPO_ROOT),
                            "startupTimeoutMs": 60000,
                            "toolCallTimeoutMs": 300000,
                        },
                    }
                ]
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rows, sort_keys=False))
    return path


def _texts(message: dict) -> str:
    return " ".join(b.get("text", "") for b in message.get("content", []) if isinstance(b, dict)).strip()


def describe(notification: Any, width: int = 240) -> str | None:
    """One readable line per interesting harness event: tool calls, results, answers, errors."""
    payload = getattr(notification, "payload", None) or {}
    event = payload.get("event") or {}
    kind, data = event.get("type"), event.get("data") or {}
    if kind == "tool/call":
        args = data.get("arguments", data.get("input", ""))
        args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return f"  -> tool {data.get('name')}({args[:width]})"
    if kind == "tool/result":
        return f"  <- {_texts(data.get('message') or {})[:width]}"
    if kind == "assistant/message":
        text = _texts(data.get("message") or {})
        return f"model: {text[:width]}" if text else None
    if kind == "turn/end" and (data.get("reason") or {}).get("kind") == "error":
        return f"ERROR: {data['reason'].get('error')}"
    return None


def run_task(
    prompt: str,
    *,
    profile: str = "sdk-minimal",
    model: str | None = None,
    mcp: bool = True,
    thinking: bool = False,
    workspace: Path | None = None,
    dsh_home: Path | None = None,
    session_id: str | None = None,
    max_tokens: int = 4096,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    verbose: bool = True,
    fresh_home: bool = False,
):
    """Run one task in DeepSeek Harness and return its ``RunResult``."""
    from deepseek_harness import DeepSeekHarness

    workspace = (workspace or config.WORKSPACE_DIR).resolve()
    # One home per profile: sdk and sdk-minimal store sessions in different formats.
    dsh_home = (dsh_home or config.DSH_HOME / profile).resolve()
    if fresh_home and dsh_home.exists():
        shutil.rmtree(dsh_home)
    workspace.mkdir(parents=True, exist_ok=True)
    dsh_home.mkdir(parents=True, exist_ok=True)
    patch = write_patch(dsh_home / "harness-lab.patch.yml", mcp=mcp)

    # The runtime reads these from its environment.
    os.environ["DSH_SYSTEM_PROMPT"] = system_prompt
    os.environ["DSH_CONTEXT_WINDOW"] = os.environ.get("DSH_CONTEXT_WINDOW", "32768")
    os.environ["HARNESS_WORKSPACE"] = str(workspace)

    def on_notification(n):
        line = describe(n)
        if verbose and line:
            print(line)

    with DeepSeekHarness(
        provider="deepseek-official",
        model=model or config.MODEL_ID,
        reasoning_effort=None if thinking else "off",
        max_tokens=max_tokens,
        base_url=config.BASE_URL.removesuffix("/v1"),
        api_key=config.API_KEY,
        cwd=str(workspace),
        dsh_home=str(dsh_home),
        profile=profile,
        patches=(str(patch),),
        initialize_timeout_seconds=120,
    ) as harness:
        return harness.run(prompt, session_id=session_id or f"lab-{os.getpid()}", on_notification=on_notification)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--profile", default="sdk-minimal", help="sdk-minimal (3 tools) or sdk (full tool set)")
    parser.add_argument("--model", default=config.MODEL_ID)
    parser.add_argument("--no-mcp", action="store_true", help="do not load our MCP tools")
    parser.add_argument("--thinking", action="store_true", help="let the model think (slower)")
    parser.add_argument("--session-id", default=None, help="reuse to continue a conversation")
    args = parser.parse_args()
    result = run_task(
        args.prompt,
        profile=args.profile,
        model=args.model,
        mcp=not args.no_mcp,
        thinking=args.thinking,
        session_id=args.session_id,
    )
    print(f"\n--- finish: {result.finish_reason}\n{result.final_response}")


if __name__ == "__main__":
    main()
