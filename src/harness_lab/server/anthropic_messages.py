"""Anthropic Messages API (``POST /v1/messages``).

Two of our harnesses speak this format: **DeepSeek Harness** (its DeepSeek adapter posts
Anthropic-style requests to ``$DEEPSEEK_BASE_URL/v1/messages``) and **Claude Code**
(``$ANTHROPIC_BASE_URL/v1/messages``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from harness_lab.server.engine import Engine, GenerationRequest
from harness_lab.server.toolcalls import StreamParser, new_call_id


def _blocks(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _plain(content: Any) -> str:
    """Flatten a block list into text (images become a short placeholder)."""
    parts = []
    for b in _blocks(content):
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "image":
            parts.append("[image omitted: this local model only reads text]")
        elif b.get("type") == "document":
            parts.append("[document omitted]")
    return "\n".join(p for p in parts if p)


def to_request(body: dict[str, Any], default_thinking: bool) -> GenerationRequest:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _plain(system)})
    for m in body.get("messages", []):
        role = m["role"]
        if role == "system":  # DeepSeek Harness may send mid-conversation system updates
            text = _plain([b for b in _blocks(m["content"]) if b.get("type") == "text"])
            if text:
                messages.append({"role": "system", "content": text})
            continue
        if role == "assistant":
            text, reasoning, calls = [], [], []
            for b in _blocks(m["content"]):
                kind = b.get("type")
                if kind == "text":
                    text.append(b.get("text", ""))
                elif kind == "thinking":
                    reasoning.append(b.get("thinking", ""))
                elif kind == "tool_use":
                    calls.append(
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {"name": b["name"], "arguments": b.get("input") or {}},
                        }
                    )
            msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text)}
            if reasoning:
                msg["reasoning_content"] = "\n".join(reasoning)
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
            continue
        # user turn: tool results first (as "tool" messages), then the rest of the text
        user_text = []
        for b in _blocks(m["content"]):
            if b.get("type") == "tool_result":
                content = _plain(b.get("content"))
                if b.get("is_error"):
                    content = f"[error] {content}"
                messages.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": content})
            else:
                user_text.append(b)
        text = _plain(user_text)
        if text:
            messages.append({"role": "user", "content": text})
    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in body.get("tools") or []
        if "input_schema" in t or t.get("type") in (None, "custom")
    ]
    thinking = default_thinking
    if isinstance(body.get("thinking"), dict):
        thinking = body["thinking"].get("type") in ("enabled", "adaptive")
    return GenerationRequest(
        messages=messages,
        tools=tools or None,
        max_tokens=body.get("max_tokens") or 2048,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        thinking=thinking,
        stop=list(body.get("stop_sequences") or []),
    )


def count_tokens(engine: Engine, body: dict[str, Any]) -> dict:
    return {"input_tokens": engine.count_tokens(to_request(body, False))}


def respond(engine: Engine, body: dict[str, Any], default_thinking: bool) -> dict | Iterator[str]:
    req = to_request(body, default_thinking)
    gen = engine.generate(req)
    parser = StreamParser(thinking=req.thinking, tools=req.tools)
    msg_id, model = new_call_id("msg"), body.get("model", engine.model_id)

    def stop_reason(has_calls: bool) -> str:
        if has_calls:
            return "tool_use"
        return "max_tokens" if gen.finish_reason == "length" else "end_turn"

    if not body.get("stream"):
        events = [e for d in gen for e in parser.feed(d)] + parser.finish()
        content: list[dict] = []
        reasoning = "".join(e.text for e in events if e.kind == "reasoning").strip()
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
        text = "".join(e.text for e in events if e.kind == "text").strip()
        if text:
            content.append({"type": "text", "text": text})
        calls = [e.call for e in events if e.kind == "tool_call"]
        for c in calls:
            content.append({"type": "tool_use", "id": "toolu_" + c.id[5:], "name": c.name, "input": c.arguments})
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason(bool(calls)),
            "stop_sequence": None,
            "usage": {"input_tokens": gen.prompt_tokens, "output_tokens": gen.completion_tokens},
        }

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps({'type': event, **data})}\n\n"

    def stream() -> Iterator[str]:
        yield sse(
            "message_start",
            {
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": gen.prompt_tokens, "output_tokens": 0},
                }
            },
        )
        index, open_kind, n_calls = -1, None, 0

        def close():
            nonlocal open_kind
            if open_kind is not None:
                if open_kind == "thinking":
                    yield sse(
                        "content_block_delta", {"index": index, "delta": {"type": "signature_delta", "signature": ""}}
                    )
                yield sse("content_block_stop", {"index": index})
                open_kind = None

        def emit(events):
            nonlocal index, open_kind, n_calls
            for e in events:
                kind = {"reasoning": "thinking", "text": "text"}.get(e.kind, "tool_use")
                if kind == "tool_use":
                    yield from close()
                    index += 1
                    n_calls += 1
                    block = {"type": "tool_use", "id": "toolu_" + e.call.id[5:], "name": e.call.name, "input": {}}
                    yield sse("content_block_start", {"index": index, "content_block": block})
                    delta = {"type": "input_json_delta", "partial_json": json.dumps(e.call.arguments)}
                    yield sse("content_block_delta", {"index": index, "delta": delta})
                    yield sse("content_block_stop", {"index": index})
                    continue
                if not e.text:
                    continue
                if open_kind != kind:
                    yield from close()
                    index += 1
                    open_kind = kind
                    start = (
                        {"type": "thinking", "thinking": "", "signature": ""}
                        if kind == "thinking"
                        else {"type": "text", "text": ""}
                    )
                    yield sse("content_block_start", {"index": index, "content_block": start})
                delta = (
                    {"type": "thinking_delta", "thinking": e.text}
                    if kind == "thinking"
                    else {"type": "text_delta", "text": e.text}
                )
                yield sse("content_block_delta", {"index": index, "delta": delta})

        for d in gen:
            yield from emit(parser.feed(d))
        yield from emit(parser.finish())
        yield from close()
        if index == -1:  # never return an empty message
            yield sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
            yield sse("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "(no output)"}})
            yield sse("content_block_stop", {"index": 0})
        yield sse(
            "message_delta",
            {
                "delta": {"stop_reason": stop_reason(n_calls > 0), "stop_sequence": None},
                "usage": {"input_tokens": gen.prompt_tokens, "output_tokens": gen.completion_tokens},
            },
        )
        yield sse("message_stop", {})

    return stream()
