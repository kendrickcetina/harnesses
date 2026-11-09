"""A harness in ~100 lines: the loop that turns a model into an agent.

The model only returns text (or a request to call a tool). Everything else lives here:
the tool registry, executing the calls, feeding results back, and deciding when to stop.
Real harnesses (DeepSeek Harness, Codex, Claude Code) add permissions, sandboxes,
context compaction, sessions, subagents... but the core is this loop.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from openai import OpenAI

from harness_lab import config
from harness_lab.tools import docx_tools

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def tool_schema(fn: Callable) -> dict:
    """Build the JSON schema the model sees from a Python function signature + docstring."""
    hints = get_type_hints(fn)
    params = inspect.signature(fn).parameters
    properties = {name: {"type": _JSON_TYPES.get(hints.get(name, str), "string")} for name in params}
    required = [n for n, p in params.items() if p.default is inspect.Parameter.empty]
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def run_shell(command: str) -> str:
    """Run a shell command in the workspace and return its output."""
    config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, shell=True, cwd=config.WORKSPACE_DIR, capture_output=True, text=True, timeout=60)
    return (proc.stdout + proc.stderr)[-4000:] or f"(exit code {proc.returncode})"


DEFAULT_TOOLS: list[Callable] = [
    run_shell,
    docx_tools.create_word_document,
    docx_tools.read_word_document,
    docx_tools.list_workspace,
]


@dataclass
class MiniHarness:
    tools: list[Callable] = field(default_factory=lambda: list(DEFAULT_TOOLS))
    model: str = config.MODEL_ID
    system: str = "You are a helpful assistant. Use the tools when they help. Answer briefly."
    max_steps: int = 8
    client: Any = None
    verbose: bool = True
    messages: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.client = self.client or OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
        self.registry = {fn.__name__: fn for fn in self.tools}
        self.schemas = [tool_schema(fn) for fn in self.tools]

    def call_tool(self, name: str, arguments: str) -> str:
        if name not in self.registry:
            return f"Error: unknown tool {name}"
        try:
            return str(self.registry[name](**json.loads(arguments or "{}")))
        except Exception as exc:  # the model sees the error and can retry
            return f"Error: {type(exc).__name__}: {exc}"

    def run(self, task: str) -> str:
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        for step in range(1, self.max_steps + 1):
            # 1. One model call: the "LLM call" layer.
            reply = self.client.chat.completions.create(model=self.model, messages=self.messages, tools=self.schemas)
            msg = reply.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))
            # 2. No tool calls: the model is done.
            if not msg.tool_calls:
                return msg.content or ""
            # 3. Execute every requested tool and append the results.
            for call in msg.tool_calls:
                result = self.call_tool(call.function.name, call.function.arguments)
                if self.verbose:
                    print(f"[step {step}] {call.function.name}({call.function.arguments[:120]}) -> {result[:200]}")
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        return "Stopped: step limit reached."
