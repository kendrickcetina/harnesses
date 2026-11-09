from harness_lab.server.toolcalls import StreamParser, parse_output

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}, "tags": {"type": "array"}},
            },
        },
    }
]

XML = (
    "I'll write it.\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n"
    '<parameter=lines>\n3\n</parameter>\n<parameter=tags>\n["a", "b"]\n</parameter>\n</function>\n</tool_call>'
)


def test_xml_tool_call_with_types():
    out = parse_output(XML, tools=TOOLS)
    assert out.text == "I'll write it."
    assert len(out.tool_calls) == 1
    call = out.tool_calls[0]
    assert call.name == "write_file"
    assert call.arguments == {"path": "notes.md", "lines": 3, "tags": ["a", "b"]}


def test_hermes_json_tool_call():
    text = '<tool_call>\n{"name": "write_file", "arguments": {"path": "a.txt"}}\n</tool_call>'
    out = parse_output(text, tools=TOOLS)
    assert out.text == ""
    assert out.tool_calls[0].arguments == {"path": "a.txt"}


def test_multiline_string_parameter_is_kept():
    text = (
        "<tool_call>\n<function=write_file>\n<parameter=path>\n"
        "line one\nline two\n</parameter>\n</function>\n</tool_call>"
    )
    assert parse_output(text, tools=TOOLS).tool_calls[0].arguments["path"] == "line one\nline two"


def test_thinking_is_separated():
    out = parse_output("Let me think.\n</think>\n\nThe answer is 4.", thinking=True)
    assert out.reasoning == "Let me think."
    assert out.text == "The answer is 4."


def test_streaming_char_by_char_matches_one_shot():
    full = "plan\n</think>\n\n" + XML + "\n" + XML
    parser = StreamParser(thinking=True, tools=TOOLS)
    events = [e for ch in full for e in parser.feed(ch)] + parser.finish()
    assert "".join(e.text for e in events if e.kind == "reasoning") == "plan\n"
    assert "".join(e.text for e in events if e.kind == "text").strip() == "I'll write it."
    assert [e.call.name for e in events if e.kind == "tool_call"] == ["write_file", "write_file"]


def test_unterminated_call_at_end_of_generation():
    text = "<tool_call>\n<function=write_file>\n<parameter=path>\nx.md\n"
    assert parse_output(text, tools=TOOLS).tool_calls[0].arguments == {"path": "x.md"}


def test_plain_text_with_angle_brackets():
    out = parse_output("Use a < b and <tag> freely.")
    assert out.text == "Use a < b and <tag> freely." and not out.tool_calls
