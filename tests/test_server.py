"""The three wire protocols, checked with a scripted model."""

import json

from fastapi.testclient import TestClient

from harness_lab.server.app import create_app
from harness_lab.server.engine import ScriptedEngine

from .conftest import call

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def client(script):
    engine = ScriptedEngine(script)
    return TestClient(create_app(engine)), engine


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def test_health_and_models():
    c, _ = client(["hi"])
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/v1/models").json()["data"][0]["id"] == "scripted"


def test_chat_completions_tool_call():
    c, engine = client([call("get_weather", city="Madrid")])
    r = c.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "weather?"}], "tools": [TOOL]},
    )
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    fn = choice["message"]["tool_calls"][0]["function"]
    assert fn["name"] == "get_weather" and json.loads(fn["arguments"]) == {"city": "Madrid"}
    assert engine.calls[0].tools[0]["function"]["name"] == "get_weather"


def test_chat_completions_history_roundtrip():
    c, engine = client(["It is sunny."])
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Madrid"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": messages, "tools": [TOOL]})
    assert r.json()["choices"][0]["message"]["content"] == "It is sunny."
    # arguments reach the chat template as a dict, as Qwen expects
    assert engine.calls[0].messages[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Madrid"}


def test_chat_completions_stream():
    c, _ = client(["Hello " + call("get_weather", city="Paris")])
    r = c.post(
        "/v1/chat/completions",
        json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}], "tools": [TOOL]},
    )
    events = sse_events(r.text)
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"])
    calls = [tc for e in events if e["choices"] for tc in e["choices"][0]["delta"].get("tool_calls", [])]
    assert text.strip() == "Hello"
    assert calls[0]["function"]["name"] == "get_weather"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_messages_stream_event_order():
    c, engine = client([call("get_weather", city="Rome")])
    body = {
        "model": "m",
        "max_tokens": 100,
        "stream": True,
        "system": "be brief",
        "thinking": {"type": "disabled"},
        "tools": [{"name": "get_weather", "description": "w", "input_schema": TOOL["function"]["parameters"]}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "weather?"}]}],
    }
    events = sse_events(c.post("/v1/messages", json=body).text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["content_block"]["type"] == "tool_use" and start["content_block"]["name"] == "get_weather"
    delta = next(e for e in events if e["type"] == "content_block_delta")
    assert json.loads(delta["delta"]["partial_json"]) == {"city": "Rome"}
    assert next(e for e in events if e["type"] == "message_delta")["delta"]["stop_reason"] == "tool_use"
    assert engine.calls[0].messages[0] == {"role": "system", "content": "be brief"}


def test_anthropic_messages_tool_result_and_thinking():
    c, engine = client(["Done."])
    body = {
        "model": "m",
        "max_tokens": 100,
        "thinking": {"type": "enabled"},
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Rome"}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "rain"}]}
                ],
            },
        ],
    }
    out = c.post("/v1/messages", json=body).json()
    assert [b["type"] for b in out["content"]] == ["thinking", "text"]
    assert out["stop_reason"] == "end_turn"
    sent = engine.calls[0].messages
    assert sent[-1] == {"role": "tool", "tool_call_id": "t1", "content": "rain"}
    assert engine.calls[0].thinking is True


def test_anthropic_count_tokens():
    c, _ = client(["x"])
    r = c.post("/v1/messages/count_tokens", json={"model": "m", "messages": [{"role": "user", "content": "hello"}]})
    assert r.json()["input_tokens"] > 0


def test_responses_api_with_codex_namespaces():
    c, engine = client([call("mcp__lab__list_workspace")])
    body = {
        "model": "m",
        "stream": True,
        "instructions": "You are Codex.",
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
        ],
        "tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object", "properties": {}}},
            {
                "type": "namespace",
                "name": "mcp__lab",
                "tools": [
                    {"type": "function", "name": "list_workspace", "parameters": {"type": "object", "properties": {}}}
                ],
            },
            {"type": "web_search"},
        ],
    }
    events = sse_events(c.post("/v1/responses", json=body).text)
    assert events[0]["type"] == "response.created" and events[-1]["type"] == "response.completed"
    item = next(e["item"] for e in events if e["type"] == "response.output_item.done")
    assert item["type"] == "function_call"
    assert item["name"] == "list_workspace" and item["namespace"] == "mcp__lab"
    names = [t["function"]["name"] for t in engine.calls[0].tools]
    assert names == ["exec_command", "mcp__lab__list_workspace"]
    assert engine.calls[0].messages[0]["role"] == "system"


def test_responses_api_history_with_function_output():
    c, engine = client(["Two files."])
    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "list"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "list_workspace",
                "namespace": "mcp__lab",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c1", "output": "a.txt\nb.txt"},
        ],
    }
    out = c.post("/v1/responses", json=body).json()
    assert out["output_text"] == "Two files."
    msgs = engine.calls[0].messages
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "mcp__lab__list_workspace"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"}


def test_thinking_policy_forces_mode():
    engine = ScriptedEngine(["ok"])
    engine.thinking_policy = "off"
    c = TestClient(create_app(engine))
    c.post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 10,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert engine.calls[0].thinking is False
