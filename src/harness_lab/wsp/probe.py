"""Measure the failure an edit should fix: answers that loop until the token limit.

Practice problems come from the GSM8K *train* split, like in the post (the test split
stays untouched for the final evaluation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ProbeResult:
    accuracy: float
    truncated_rate: float  # hit max_new_tokens without finishing
    loop_rate: float  # repeats itself (see looks_looping)
    outputs: list[str] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"accuracy={self.accuracy:.1%}  truncated={self.truncated_rate:.1%}  "
            f"looping={self.loop_rate:.1%}  (n={len(self.outputs)})"
        )


def load_practice(n: int = 64, split: str = "train", seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=seed).select(range(n))
    return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for r in data]


def extract_number(text: str) -> str | None:
    """Last number in the visible answer (after </think> if present)."""
    text = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    candidates = boxed or re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not candidates:
        return None
    return candidates[-1].replace(",", "").rstrip(".").strip()


def looks_looping(text: str, n: int = 8, threshold: float = 0.3) -> bool:
    """True if a large share of the word n-grams are repeats (the model is going in circles)."""
    words = text.split()
    if len(words) < 4 * n:
        return False
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) > threshold


def run_probe(
    model, tokenizer, problems: list[dict], max_new_tokens: int = 2048, thinking: bool = True, batch_size: int = 8
) -> ProbeResult:
    import torch

    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    outputs, truncated = [], []
    for start in range(0, len(problems), batch_size):
        batch = problems[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p["question"] + "\nPut the final answer in \\boxed{}."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            for p in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for row in out[:, enc["input_ids"].shape[1] :]:
            n_tokens = int((row != (tokenizer.pad_token_id or tokenizer.eos_token_id)).sum())
            truncated.append(n_tokens >= max_new_tokens)
            outputs.append(tokenizer.decode(row, skip_special_tokens=True))
        print(f"  probe {min(start + batch_size, len(problems))}/{len(problems)}")
    correct = [extract_number(o) == p["answer"] for o, p in zip(outputs, problems, strict=True)]
    n = max(len(problems), 1)
    return ProbeResult(
        accuracy=sum(correct) / n,
        truncated_rate=sum(truncated) / n,
        loop_rate=sum(looks_looping(o) for o in outputs) / n,
        outputs=outputs,
        truncated=truncated,
    )
