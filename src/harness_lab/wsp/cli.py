"""Weight-space programming lab.

# 1. Read: what did the published edit change?
uv run harness-wsp diff Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming

# 2. Measure: how often does a model loop on practice problems?
uv run harness-wsp probe --model Qwen/Qwen3.5-0.8B --n 32

# 3. Write: find a "looping" direction and project it out of a few MLP columns
uv run harness-wsp edit --model Qwen/Qwen3.5-0.8B --layers 15 19 --columns 64

# Your ARCA model (base + GRPO LoRA adapter):
uv run harness-wsp probe --model Qwen/Qwen3-0.6B --adapter ../dgm-arca/rlm/weights/final_rlm_lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_lab import config


def load_model(model_id: str, adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from harness_lab.server.engine import pick_device

    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if adapter:  # merge the LoRA so the edit acts on the weights the model really uses
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to(device).eval(), tokenizer


def cmd_diff(args) -> None:
    from harness_lab.wsp.diff import compare_checkpoints, format_report

    print(format_report(compare_checkpoints(args.base, args.edited)))


def cmd_probe(args):
    from harness_lab.wsp.probe import load_practice, run_probe

    model, tok = load_model(args.model, args.adapter)
    result = run_probe(
        model,
        tok,
        load_practice(args.n, seed=args.seed),
        args.max_new_tokens,
        thinking=not args.no_thinking,
        batch_size=args.batch_size,
    )
    print(result.summary())
    return model, tok, result


def cmd_edit(args) -> None:
    from harness_lab.wsp.edit import apply_edit, find_direction
    from harness_lab.wsp.probe import load_practice, looks_looping, run_probe

    model, tok, before = cmd_probe(args)
    bad = [o for o, t in zip(before.outputs, before.truncated, strict=True) if t or looks_looping(o)]
    good = [o for o, t in zip(before.outputs, before.truncated, strict=True) if not (t or looks_looping(o))]
    if len(bad) < 2 or len(good) < 2:
        print(
            f"Need at least 2 looping and 2 clean answers (got {len(bad)} / {len(good)}). "
            "Raise --n or lower --max-new-tokens."
        )
        return
    direction = find_direction(model, tok, bad, good, layer=args.direction_layer or args.layers[0])
    edited = apply_edit(model, direction, args.layers, args.columns)
    n_weights = sum(len(c) for c in edited.values()) * direction.numel()
    print(f"Edited {n_weights:,} weights in layers {list(edited)}")
    after = run_probe(
        model,
        tok,
        load_practice(args.n, seed=args.seed + 1),
        args.max_new_tokens,
        thinking=not args.no_thinking,
        batch_size=args.batch_size,
    )
    print("before:", before.summary())
    print("after: ", after.summary())
    out = Path(args.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "wsp_edit.json").write_text(
        json.dumps(
            {
                "base": args.model,
                "adapter": args.adapter,
                "layers": args.layers,
                "columns": edited,
                "before": before.summary(),
                "after": after.summary(),
            },
            indent=2,
        )
    )
    print(f"Saved edited model to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="compare two checkpoints tensor by tensor")
    d.add_argument("base", nargs="?", default=config.WSP_BASE_MODEL_ID)
    d.add_argument("edited", nargs="?", default=config.WSP_EDITED_MODEL_ID)
    d.set_defaults(func=cmd_diff)

    for name, func in (("probe", cmd_probe), ("edit", cmd_edit)):
        p = sub.add_parser(name)
        p.add_argument("--model", default=config.WSP_BASE_MODEL_ID)
        p.add_argument("--adapter", default=None, help="LoRA adapter folder to merge first (ARCA)")
        p.add_argument("--n", type=int, default=32, help="practice problems (GSM8K train)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-new-tokens", type=int, default=2048)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--no-thinking", action="store_true")
        p.set_defaults(func=func)
        if name == "edit":
            p.add_argument("--layers", type=int, nargs="+", default=[15, 19])
            p.add_argument("--direction-layer", type=int, default=None)
            p.add_argument("--columns", type=int, default=64)
            p.add_argument("--out", default=str(config.OUTPUTS_DIR / "wsp" / "edited"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
