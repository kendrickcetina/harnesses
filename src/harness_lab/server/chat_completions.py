"""OpenAI Chat Completions API (``POST /v1/chat/completions``).

The oldest and most common wire format. We use it for the raw model calls and the
hand-written harness in the tutorial.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from harness_lab.server.engine import Engine, GenerationRequest
from harness_lab.server.toolcalls import StreamParser, new_call_id, stream_events


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def to_request(body: dict[str, Any], default_thinking: bool) -> GenerationRequest:
    messages = []
    for m in body.get("messages", []):
        role = "system" if m["role"] == "developer" else m["role"]
        msg: dict[str, Any] = {"role": role, "content": _text(m.get("content"))}
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": c.get("id"),
                    "type": "function",
                    "function": {
                        "name": c["function"]["name"],
                        "arguments": _loads(c["function"].get("arguments")),
                    },
                }
                for c in m["tool_calls"]
            ]
        if m.get("reasoning_content"):
            msg["reasoning_content"] = m["reasoning_content"]
        messages.append(msg)
    thinking = default_thinking
    kwargs = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in kwargs:
        thinking = bool(kwargs["enable_thinking"])
    elif body.get("reasoning_effort"):
        thinking = body["reasoning_effort"] not in ("none", "off", "minimal")
    stop = body.get("stop") or []
    return GenerationRequest(
        messages=messages,
        tools=body.get("tools") or None,
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens") or 2048,
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        thinking=thinking,
        stop=[stop] if isinstance(stop, str) else list(stop),
    )


def _loads(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"input": arguments}


def respond(engine: Engine, body: dict[str, Any], default_thinking: bool) -> dict | Iterator[str]:
    req = to_request(body, default_thinking)
    gen = engine.generate(req)
    parser = StreamParser(thinking=req.thinking, tools=req.tools)
    rid, created, model = f"chatcmpl-{new_call_id('')[1:]}", int(time.time()), body.get("model", engine.model_id)

    if not body.get("stream"):
        events = list(stream_events(gen, parser))
        calls = [e.call for e in events if e.kind == "tool_call"]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(e.text for e in events if e.kind == "text") or None,
        }
        reasoning = "".join(e.text for e in events if e.kind == "reasoning").strip()
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in calls
            ]
        finish = "tool_calls" if calls else gen.finish_reason
        return {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": _usage(gen),
        }

    def stream() -> Iterator[str]:
        def chunk(delta: dict, finish: str | None = None) -> str:
            data = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(data)}\n\n"

        yield chunk({"role": "assistant", "content": ""})
        n_calls = 0

        def emit(events):
            nonlocal n_calls
            for e in events:
                if e.kind == "text":
                    yield chunk({"content": e.text})
                elif e.kind == "reasoning":
                    yield chunk({"reasoning_content": e.text})
                elif e.call:
                    call = {"name": e.call.name, "arguments": json.dumps(e.call.arguments)}
                    yield chunk(
                        {"tool_calls": [{"index": n_calls, "id": e.call.id, "type": "function", "function": call}]}
                    )
                    n_calls += 1

        yield from emit(stream_events(gen, parser))
        yield chunk({}, "tool_calls" if n_calls else gen.finish_reason)
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model}
            yield f"data: {json.dumps({**usage, 'choices': [], 'usage': _usage(gen)})}\n\n"
        yield "data: [DONE]\n\n"

    return stream()


def _usage(gen) -> dict[str, int]:
    return {
        "prompt_tokens": gen.prompt_tokens,
        "completion_tokens": gen.completion_tokens,
        "total_tokens": gen.prompt_tokens + gen.completion_tokens,
    }
