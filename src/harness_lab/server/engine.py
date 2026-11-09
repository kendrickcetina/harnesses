"""The model itself: messages in, text out.

Every API in ``protocols/`` is translated into one canonical request (OpenAI-style
messages + tool schemas) and handed to an :class:`Engine`. Two engines exist:

* :class:`TransformersEngine` runs a real Hugging Face model on CUDA, MPS or CPU.
* :class:`ScriptedEngine` replays fixed answers. It needs no GPU and no download, which
  makes it ideal for offline tests of the whole harness stack.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    max_tokens: int = 2048
    temperature: float | None = None
    top_p: float | None = None
    thinking: bool = False
    stop: list[str] = field(default_factory=list)


@dataclass
class Generation:
    """Iterate to receive text deltas. Token counts are filled in while iterating."""

    deltas: Iterator[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"  # "stop" or "length"
    seconds: float = 0.0

    def __iter__(self) -> Iterator[str]:
        return self.deltas


class Engine(ABC):
    model_id: str = "local"
    max_context: int = 32768
    # "auto": follow what the client asks; "on"/"off": force thinking for every request.
    thinking_policy: str = "auto"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> Generation: ...

    def prepare(self, request: GenerationRequest) -> GenerationRequest:
        if self.thinking_policy in ("on", "off"):
            request.thinking = self.thinking_policy == "on"
        return request

    def count_tokens(self, request: GenerationRequest) -> int:
        text = str(request.messages) + str(request.tools or "")
        return len(text) // 4


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make messages acceptable to the Qwen chat template.

    The template wants exactly one system message, first. Harnesses sometimes send
    several (or send system updates mid-conversation), so they are merged.
    """
    systems = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
    rest = [m for m in messages if m["role"] != "system"]
    if not any(m["role"] == "user" for m in rest):
        rest.insert(0, {"role": "user", "content": "Continue."})
    if systems:
        return [{"role": "system", "content": "\n\n".join(systems)}, *rest]
    return rest


class ScriptedEngine(Engine):
    """Replays scripted answers. ``script`` is a list of raw outputs or a function.

    Raw outputs use the same markup the real model writes, e.g.
    ``'<tool_call>\\n<function=ls>\\n</function>\\n</tool_call>'``.
    """

    def __init__(self, script: list[str] | Callable[[GenerationRequest], str], model_id: str = "scripted"):
        self.script = script
        self.model_id = model_id
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> Generation:
        self.prepare(request)
        self.calls.append(request)
        if callable(self.script):
            text = self.script(request)
        else:
            text = self.script[min(len(self.calls), len(self.script)) - 1]
        if request.thinking and "</think>" not in text:
            # A real model would reason first; the template already opened <think>.
            text = "(scripted reasoning)\n</think>\n\n" + text
        gen = Generation(deltas=iter(()))
        gen.prompt_tokens = self.count_tokens(request)

        def deltas() -> Iterator[str]:
            for i in range(0, len(text), 7):
                gen.completion_tokens += 1
                yield text[i : i + 7]

        gen.deltas = deltas()
        return gen


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersEngine(Engine):
    """A Hugging Face model behind a lock (one generation at a time)."""

    def __init__(self, model_id: str, device: str | None = None, max_context: int = 32768):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_context = max_context
        self.device = device or pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Qwen3.5 checkpoints are multimodal; AutoModelForCausalLM loads the text part only.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.lock = threading.Lock()
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"Loaded {model_id} ({n_params:.2f}B params) on {self.device} in {time.time() - t0:.0f}s")

    def _prompt_ids(self, request: GenerationRequest):
        return self.tokenizer.apply_chat_template(
            normalize_messages(request.messages),
            tools=request.tools or None,
            add_generation_prompt=True,
            enable_thinking=request.thinking,
            return_tensors="pt",
            return_dict=True,
        )

    def count_tokens(self, request: GenerationRequest) -> int:
        return int(self._prompt_ids(request)["input_ids"].shape[-1])

    def generate(self, request: GenerationRequest) -> Generation:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        self.prepare(request)

        class Cancel(StoppingCriteria):
            # Lets us stop generate() early: client disconnected or a stop sequence matched.
            flag = False

            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), self.flag, dtype=torch.bool, device=input_ids.device)

        cancel = Cancel()
        inputs = self._prompt_ids(request).to(self.device)
        n_prompt = int(inputs["input_ids"].shape[-1])
        max_new = max(1, min(request.max_tokens, self.max_context - n_prompt))
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        temperature = request.temperature if request.temperature is not None else (0.6 if request.thinking else 0.7)
        kwargs = dict(
            **inputs,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([cancel]),
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=request.top_p or (0.95 if request.thinking else 0.8), top_k=20)
        gen = Generation(deltas=iter(()), prompt_tokens=n_prompt)

        def run() -> None:
            with self.lock:
                try:
                    self.model.generate(**kwargs)
                except Exception as exc:  # surface errors in the HTTP stream
                    streamer.on_finalized_text(f"\n[generation error: {exc}]", stream_end=True)

        def deltas() -> Iterator[str]:
            t0 = time.time()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            produced = ""
            try:
                for piece in streamer:
                    if not piece:
                        continue
                    produced += piece
                    cut = _stop_index(produced, request.stop)
                    if cut is not None:
                        yield piece[: len(piece) - (len(produced) - cut)]
                        break
                    yield piece
            finally:
                cancel.flag = True
                for _ in streamer:  # drain so generate() can finish
                    pass
                thread.join()
            gen.completion_tokens = len(self.tokenizer(produced)["input_ids"])
            gen.finish_reason = "length" if gen.completion_tokens >= max_new else "stop"
            gen.seconds = time.time() - t0

        gen.deltas = deltas()
        return gen


def _stop_index(text: str, stops: list[str]) -> int | None:
    hits = [text.find(s) for s in stops if s and s in text]
    return min(hits) if hits else None
