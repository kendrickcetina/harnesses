"""OpenAI Responses API (``POST /v1/responses``).

Codex CLI only speaks this format. Instead of ``messages`` it sends a list of typed
``input`` items (messages, ``function_call``, ``function_call_output``...) and expects
typed ``output`` items back, streamed as ``response.*`` events.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from harness_lab.server.engine import Engine, GenerationRequest
from harness_lab.server.toolcalls import StreamParser, new_call_id, stream_events


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict))


def _flat(name: str, namespace: str | None) -> str:
    return f"{namespace}__{name}" if namespace else name


def _convert_tools(tools: list[dict]) -> tuple[list[dict], set[str], dict[str, tuple[str, str]]]:
    """Function tools pass through; freeform ``custom`` tools get one string parameter.

    Codex groups MCP tools in ``namespace`` tools (e.g. ``mcp__lab``). The model sees them
    flattened as ``mcp__lab__create_word_document``; ``namespaces`` maps them back.
    """
    out, custom, namespaces = [], set(), {}
    for t in tools or []:
        if t.get("type") == "namespace":
            inner, inner_custom, _ = _convert_tools(t.get("tools") or [])
            for fn in inner:
                raw = fn["function"]["name"]
                fn["function"]["name"] = _flat(raw, t["name"])
                namespaces[fn["function"]["name"]] = (t["name"], raw)
            out += inner
            custom |= {_flat(n, t["name"]) for n in inner_custom}
        elif t.get("type") == "function":
            params = t.get("parameters") or {"type": "object", "properties": {}}
            out.append(
                {
                    "type": "function",
                    "function": {"name": t["name"], "description": t.get("description", ""), "parameters": params},
                }
            )
        elif t.get("type") == "custom":
            custom.add(t["name"])
            params = {
                "type": "object",
                "properties": {"input": {"type": "string", "description": "Raw tool input"}},
                "required": ["input"],
            }
            out.append(
                {
                    "type": "function",
                    "function": {"name": t["name"], "description": t.get("description", ""), "parameters": params},
                }
            )
    return out, custom, namespaces


def to_request(body: dict[str, Any], default_thinking: bool) -> tuple[GenerationRequest, set[str], dict]:
    messages: list[dict[str, Any]] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    items = body.get("input") or []
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role = {"developer": "system"}.get(item["role"], item["role"])
            messages.append({"role": role, "content": _text(item.get("content"))})
        elif kind in ("function_call", "custom_tool_call"):
            args = item.get("arguments")
            args = json.loads(args or "{}") if kind == "function_call" else {"input": item.get("input", "")}
            name = _flat(item["name"], item.get("namespace"))
            call = {"id": item.get("call_id"), "type": "function", "function": {"name": name, "arguments": args}}
            if messages and messages[-1]["role"] == "assistant" and not messages[-1].get("_closed"):
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": output if isinstance(output, str) else _text(output),
                }
            )
        # "reasoning" and other item types are not replayed to the local model
    tools, custom, namespaces = _convert_tools(body.get("tools") or [])
    effort = (body.get("reasoning") or {}).get("effort")
    thinking = default_thinking if effort is None else effort not in ("none", "minimal")
    req = GenerationRequest(
        messages=messages,
        tools=tools or None,
        max_tokens=body.get("max_output_tokens") or 4096,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        thinking=thinking,
    )
    return req, custom, namespaces


def respond(engine: Engine, body: dict[str, Any], default_thinking: bool) -> dict | Iterator[str]:
    req, custom, namespaces = to_request(body, default_thinking)
    gen = engine.generate(req)
    parser = StreamParser(thinking=req.thinking, tools=req.tools)
    resp_id, created, model = new_call_id("resp"), int(time.time()), body.get("model", engine.model_id)

    def call_item(call) -> dict:
        namespace, name = namespaces.get(call.name, (None, call.name))
        extra = {"namespace": namespace} if namespace else {}
        if call.name in custom:
            return {
                "type": "custom_tool_call",
                "id": new_call_id("ctc"),
                "call_id": call.id,
                "name": name,
                "input": str(call.arguments.get("input", "")),
                "status": "completed",
                **extra,
            }
        return {
            "type": "function_call",
            "id": new_call_id("fc"),
            "call_id": call.id,
            "name": name,
            "arguments": json.dumps(call.arguments),
            "status": "completed",
            **extra,
        }

    def message_item(text: str, item_id: str) -> dict:
        return {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }

    def envelope(status: str, output: list[dict]) -> dict:
        usage = {
            "input_tokens": gen.prompt_tokens,
            "output_tokens": gen.completion_tokens,
            "total_tokens": gen.prompt_tokens + gen.completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        return {
            "id": resp_id,
            "object": "response",
            "created_at": created,
            "model": model,
            "status": status,
            "output": output,
            "usage": usage if status == "completed" else None,
        }

    if not body.get("stream"):
        events = list(stream_events(gen, parser))
        output = []
        text = "".join(e.text for e in events if e.kind == "text").strip()
        if text:
            output.append(message_item(text, new_call_id("msg")))
        output += [call_item(e.call) for e in events if e.kind == "tool_call"]
        result = envelope("completed", output)
        result["output_text"] = text
        return result

    def stream() -> Iterator[str]:
        seq = 0

        def sse(event: str, data: dict) -> str:
            nonlocal seq
            seq += 1
            return f"event: {event}\ndata: {json.dumps({'type': event, 'sequence_number': seq, **data})}\n\n"

        yield sse("response.created", {"response": envelope("in_progress", [])})
        output: list[dict] = []
        msg_id, text = None, ""

        def close_message():
            nonlocal msg_id, text
            if msg_id is None:
                return
            idx = len(output)
            yield sse(
                "response.output_text.done", {"item_id": msg_id, "output_index": idx, "content_index": 0, "text": text}
            )
            item = message_item(text, msg_id)
            yield sse("response.output_item.done", {"output_index": idx, "item": item})
            output.append(item)
            msg_id, text = None, ""

        def emit(events):
            nonlocal msg_id, text
            for e in events:
                if e.kind == "text" and e.text:
                    if msg_id is None:
                        msg_id = new_call_id("msg")
                        item = {
                            "type": "message",
                            "id": msg_id,
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        }
                        yield sse("response.output_item.added", {"output_index": len(output), "item": item})
                        part = {"type": "output_text", "text": "", "annotations": []}
                        yield sse(
                            "response.content_part.added",
                            {"item_id": msg_id, "output_index": len(output), "content_index": 0, "part": part},
                        )
                    text += e.text
                    yield sse(
                        "response.output_text.delta",
                        {"item_id": msg_id, "output_index": len(output), "content_index": 0, "delta": e.text},
                    )
                elif e.kind == "tool_call":
                    yield from close_message()
                    item = call_item(e.call)
                    yield sse(
                        "response.output_item.added",
                        {"output_index": len(output), "item": {**item, "status": "in_progress"}},
                    )
                    yield sse("response.output_item.done", {"output_index": len(output), "item": item})
                    output.append(item)

        yield from emit(stream_events(gen, parser))
        yield from close_message()
        yield sse("response.completed", {"response": envelope("completed", output)})

    return stream()
