"""Turn raw Qwen text into reasoning, visible text and tool calls.

A language model only produces text. "Calling a tool" means writing a block like this,
which the server (not the model) recognises and converts into structured JSON:

    <tool_call>
    <function=write_file>
    <parameter=path>
    notes.md
    </parameter>
    </function>
    </tool_call>

Qwen3.5 uses that XML form (vLLM calls it ``qwen3_coder``). Qwen3 and Qwen2.5 use the
older JSON form (``hermes``): ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``.
Both are accepted here.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"  # how the template shows tool results to the model
MAX_TEXT_BETWEEN_CALLS = 256  # characters after a call, without a new one, that end the turn

_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)(?:\n?</parameter>|(?=<parameter=)|$)", re.DOTALL)


def new_call_id(prefix: str = "call") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=new_call_id)


@dataclass
class Event:
    """One piece of parsed output: ``kind`` is ``reasoning``, ``text`` or ``tool_call``."""

    kind: str
    text: str = ""
    call: ToolCall | None = None


@dataclass
class ParsedOutput:
    reasoning: str
    text: str
    tool_calls: list[ToolCall]


def _schema_types(tools: list[dict] | None) -> dict[str, dict[str, Any]]:
    """Map tool name -> {parameter: JSON schema} from OpenAI-style tool definitions."""
    out: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        props = (fn.get("parameters") or {}).get("properties") or {}
        out[fn.get("name", "")] = props
    return out


def _coerce(value: str, schema: dict[str, Any] | None) -> Any:
    """XML parameters arrive as strings; convert them using the declared JSON schema type."""
    kind = (schema or {}).get("type")
    if kind == "string" or (kind is None and schema is not None and "enum" in schema):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_tool_block(block: str, tools: list[dict] | None = None) -> ToolCall | None:
    """Parse the inside of one ``<tool_call>...</tool_call>`` block."""
    block = block.strip()
    if not block:
        return None
    if block.startswith("{"):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            return None
        args = data.get("arguments", data.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        return ToolCall(name=str(data.get("name", "")), arguments=args or {})
    match = _FUNCTION_RE.search(block)
    if not match:
        return None
    name = match.group(1).strip()
    schema = _schema_types(tools).get(name, {})
    args = {}
    for p_name, p_value in _PARAM_RE.findall(match.group(2)):
        p_name = p_name.strip()
        args[p_name] = _coerce(p_value, schema.get(p_name) if schema else None)
    return ToolCall(name=name, arguments=args)


def _holdback(buffer: str, tag: str) -> int:
    """Length of the longest suffix of ``buffer`` that could be the start of ``tag``."""
    for size in range(min(len(tag) - 1, len(buffer)), 0, -1):
        if tag.startswith(buffer[-size:]):
            return size
    return 0


class StreamParser:
    """Incremental parser: feed text deltas as they are generated, get events back.

    Visible text is streamed immediately. Once ``<tool_call>`` appears, the rest is held
    and converted into tool calls when each block closes.
    """

    def __init__(self, thinking: bool, tools: list[dict] | None = None):
        # With thinking on, the Qwen chat template already opened <think>, so the model
        # starts inside the reasoning block.
        self.state = "reasoning" if thinking else "text"
        self.tools = tools
        self.buffer = ""
        self.started_text = False

    def feed(self, delta: str) -> list[Event]:
        self.buffer += delta
        events: list[Event] = []
        while True:
            if self.state == "reasoning":
                idx = self.buffer.find(THINK_CLOSE)
                if idx == -1:
                    keep = _holdback(self.buffer, THINK_CLOSE)
                    out, self.buffer = self.buffer[: len(self.buffer) - keep], self.buffer[len(self.buffer) - keep :]
                    if out:
                        events.append(Event("reasoning", out))
                    return events
                if idx:
                    events.append(Event("reasoning", self.buffer[:idx]))
                self.buffer = self.buffer[idx + len(THINK_CLOSE) :]
                self.state = "text"
            elif self.state == "text":
                if not self.started_text:
                    stripped = self.buffer.lstrip()
                    if not stripped:
                        return events
                    if THINK_OPEN.startswith(stripped[: len(THINK_OPEN)]) and len(stripped) < len(THINK_OPEN):
                        return events  # could still become "<think>"
                    if stripped.startswith(THINK_OPEN):
                        self.buffer = stripped[len(THINK_OPEN) :]
                        self.state = "reasoning"
                        continue
                    self.buffer = stripped
                idx = self.buffer.find(CALL_OPEN)
                if idx == -1:
                    keep = _holdback(self.buffer, CALL_OPEN)
                    out, self.buffer = self.buffer[: len(self.buffer) - keep], self.buffer[len(self.buffer) - keep :]
                    if out:
                        self.started_text = True
                        events.append(Event("text", out))
                    return events
                if idx:
                    self.started_text = True
                    events.append(Event("text", self.buffer[:idx].rstrip()))
                self.buffer = self.buffer[idx + len(CALL_OPEN) :]
                self.state = "tool"
            elif self.state == "after_call":
                # Only another (parallel) <tool_call> should follow a call: the template says
                # "NO suffix", so text in between is ignored.
                idx = self.buffer.find(CALL_OPEN)
                if idx != -1:
                    self.buffer = self.buffer[idx + len(CALL_OPEN) :]
                    self.state = "tool"
                    continue
                # A made-up tool result, or a lot of text without a new call, ends the turn.
                # Without this a small model can ramble until max_tokens (32k for Claude Code,
                # ~15 min) while the client receives nothing and times out.
                if TOOL_RESPONSE_OPEN in self.buffer or len(self.buffer.strip()) > MAX_TEXT_BETWEEN_CALLS:
                    self.buffer = ""
                    self.state = "done"
                return events
            elif self.state == "done":
                self.buffer = ""
                return events
            else:  # inside <tool_call>
                idx = self.buffer.find(CALL_CLOSE)
                if idx == -1:
                    return events
                call = parse_tool_block(self.buffer[:idx], self.tools)
                if call:
                    events.append(Event("tool_call", call=call))
                self.buffer = self.buffer[idx + len(CALL_CLOSE) :]
                self.state = "after_call"

    @property
    def finished(self) -> bool:
        """True once the turn is over: the rest of the generation can be cancelled."""
        return self.state == "done"

    def finish(self) -> list[Event]:
        events: list[Event] = []
        rest, self.buffer = self.buffer, ""
        if self.state == "reasoning" and rest:
            events.append(Event("reasoning", rest))
        elif self.state == "text" and rest.strip():
            events.append(Event("text", rest.rstrip()))
        elif self.state == "tool":
            call = parse_tool_block(rest, self.tools)  # unterminated block at EOS
            if call:
                events.append(Event("tool_call", call=call))
        return events


def stream_events(generation: Iterable[str], parser: StreamParser) -> Iterator[Event]:
    """Feed a generation's text deltas to ``parser`` and yield the events.

    Stops the generation as soon as the parser says the turn is over, instead of letting
    the model write (invisible) text until max_tokens.
    """
    deltas = iter(generation)
    try:
        for delta in deltas:
            yield from parser.feed(delta)
            if parser.finished:
                break
        yield from parser.finish()
    finally:
        close = getattr(deltas, "close", None)
        if close:
            close()  # a generator: runs its cleanup, which cancels model.generate()


def parse_output(text: str, thinking: bool = False, tools: list[dict] | None = None) -> ParsedOutput:
    """Parse a complete generation in one go."""
    parser = StreamParser(thinking=thinking, tools=tools)
    events = parser.feed(text) + parser.finish()
    return ParsedOutput(
        reasoning="".join(e.text for e in events if e.kind == "reasoning").strip(),
        text="".join(e.text for e in events if e.kind == "text").strip(),
        tool_calls=[e.call for e in events if e.kind == "tool_call" and e.call],
    )
