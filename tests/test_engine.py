"""TransformersEngine streaming, with a fake model and tokenizer (no GPU, no download).

The scripted engine used elsewhere never touches this code path, so a bug here would
only show up with the real model: e.g. a generation that never returns.
"""

import threading
import time

import pytest
import torch
from transformers import BatchEncoding

from harness_lab.server.engine import GenerationRequest, TransformersEngine


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)})

    def __call__(self, text):
        return {"input_ids": text.split()}


class FakeModel:
    """Streams words like transformers' generate(): text pieces, then the end-of-stream signal."""

    def __init__(self, words):
        self.words = words
        self.calls = 0
        self.emitted = 0

    def generate(self, streamer, stopping_criteria, **kwargs):
        self.calls += 1
        for word in self.words:
            if stopping_criteria[0](torch.zeros(1, 1), None).all():
                break
            streamer.on_finalized_text(word)
            self.emitted += 1
            time.sleep(0.01)  # like a real model: tokens take time
        streamer.on_finalized_text("", stream_end=True)


def fake_engine(words) -> TransformersEngine:
    engine = TransformersEngine.__new__(TransformersEngine)  # skip loading a real model
    engine.model_id, engine.device, engine.max_context = "fake", "cpu", 1024
    engine.tokenizer, engine.model, engine.lock = FakeTokenizer(), FakeModel(words), threading.Lock()
    engine.stop_ids = [0]
    return engine


def consume(engine, request, timeout=10):
    """Run the generation in a daemon thread so a hang fails the test instead of freezing pytest."""
    gen = engine.generate(request)
    out = []
    worker = threading.Thread(target=lambda: out.append("".join(gen)), daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        pytest.fail("generation never finished (streamer not closed properly)")
    return out[0], gen


def test_generation_finishes_when_the_model_stops():
    engine = fake_engine(["Un ", "agente ", "usa ", "tools."])
    text, gen = consume(engine, GenerationRequest(messages=[{"role": "user", "content": "hola"}]))
    assert text == "Un agente usa tools."
    assert gen.finish_reason == "stop" and gen.completion_tokens == 4 and gen.prompt_tokens == 3


def test_generation_cuts_at_stop_sequence():
    engine = fake_engine(["uno ", "dos ", "STOP ", "tres "])
    request = GenerationRequest(messages=[{"role": "user", "content": "x"}], stop=["STOP"])
    text, _ = consume(engine, request)
    assert text == "uno dos "


def test_engine_can_serve_several_requests():
    engine = fake_engine(["ok"])
    for _ in range(3):
        assert consume(engine, GenerationRequest(messages=[{"role": "user", "content": "x"}]))[0] == "ok"
    assert engine.model.calls == 3


def test_generation_is_cancelled_when_the_turn_is_over():
    """Text after a tool call ends the turn: the model must stop instead of running to max_tokens."""
    from harness_lab.server.toolcalls import StreamParser, stream_events

    call = "<tool_call>\n<function=ls>\n</function>\n</tool_call>"
    words = [call, "\n<tool_response>", " fake"] + [" bla"] * 500
    engine = fake_engine(words)
    gen = engine.generate(GenerationRequest(messages=[{"role": "user", "content": "x"}]))
    out = []
    worker = threading.Thread(target=lambda: out.extend(stream_events(gen, StreamParser(thinking=False))), daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "stream_events did not return"
    assert [e.kind for e in out] == ["tool_call"]
    assert engine.model.emitted < 10  # cancelled right after the turn ended, not after 500 words
    assert gen.completion_tokens > 0  # usage still filled in after an early stop


def test_stop_tokens_include_end_of_turn():
    """Qwen3.5 has no generation_config.json: its config only stops at <|endoftext|>."""
    from types import SimpleNamespace

    from harness_lab.server.engine import stop_token_ids

    class QwenLikeTokenizer:
        eos_token_id = 248046  # <|im_end|>

        def get_vocab(self):
            return {"<|im_end|>": 248046, "<|endoftext|>": 248044}

        def convert_tokens_to_ids(self, token):
            return self.get_vocab()[token]

    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
    assert stop_token_ids(QwenLikeTokenizer(), model) == [248044, 248046]
