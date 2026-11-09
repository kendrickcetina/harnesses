"""Write an edit into the weights: remove one direction from a few MLP output columns.

This mirrors the Qwen3.5-0.8B edit described in the weight-space programming post
(one direction subtracted from 64 ``down_proj`` columns in layers 15 and 19), but the
direction here comes from a simple recipe instead of a frontier model's reasoning:

1. Generate answers to practice problems and split them into *looping* (hit the token
   limit, repeat themselves) and *clean* ones.
2. Run both groups through the model and record what each layer's MLP writes into the
   residual stream. The difference of the means is the candidate direction ``u``.
3. In ``down_proj`` pick the columns that write most strongly along ``u`` and project
   ``u`` out of them: ``W[:, cols] <- W[:, cols] - u (u^T W[:, cols])``.

No gradients, no optimizer, no loss: compare with SFT/GRPO from the ARCA practice.
"""

from __future__ import annotations

import torch


def find_down_proj(model, layer: int) -> torch.nn.Linear:
    """The MLP output projection of decoder layer ``layer`` (Qwen2/3/3.5 naming)."""
    suffix = f"layers.{layer}.mlp.down_proj"
    for name, module in model.named_modules():
        if name.endswith(suffix):
            return module
    raise KeyError(f"no module ending in {suffix!r}")


@torch.no_grad()
def mlp_outputs(model, tokenizer, texts: list[str], layer: int, max_length: int = 1024) -> torch.Tensor:
    """Mean MLP output (residual-stream space) per text, shape [len(texts), hidden]."""
    proj = find_down_proj(model, layer)
    captured: list[torch.Tensor] = []
    hook = proj.register_forward_hook(lambda _m, _i, out: captured.append(out.float().mean(dim=1).cpu()))
    try:
        device = next(model.parameters()).device
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            model(**ids)
    finally:
        hook.remove()
    return torch.cat(captured)


def find_direction(model, tokenizer, looping: list[str], clean: list[str], layer: int) -> torch.Tensor:
    """Unit vector: mean MLP output on looping texts minus the mean on clean texts."""
    diff = mlp_outputs(model, tokenizer, looping, layer).mean(0) - mlp_outputs(model, tokenizer, clean, layer).mean(0)
    return diff / diff.norm()


@torch.no_grad()
def top_columns(weight: torch.Tensor, direction: torch.Tensor, k: int) -> list[int]:
    """Columns of ``weight`` [hidden, intermediate] that write most along ``direction``."""
    scores = (direction.to(weight) @ weight).abs()
    return torch.topk(scores.float(), k).indices.sort().values.tolist()


@torch.no_grad()
def remove_direction(weight: torch.Tensor, direction: torch.Tensor, columns: list[int] | None = None) -> int:
    """In place: ``W[:, cols] -= u (u^T W[:, cols])``. Returns the number of weights touched."""
    u = (direction / direction.norm()).to(device=weight.device, dtype=torch.float32)
    cols = list(range(weight.shape[1])) if columns is None else columns
    block = weight[:, cols].float()
    block -= torch.outer(u, u @ block)
    weight[:, cols] = block.to(weight.dtype)
    return weight.shape[0] * len(cols)


def apply_edit(model, direction: torch.Tensor, layers: list[int], n_columns: int = 64) -> dict[int, list[int]]:
    """Project ``direction`` out of the top ``n_columns`` columns of each layer's down_proj."""
    edited = {}
    for layer in layers:
        w = find_down_proj(model, layer).weight
        cols = top_columns(w, direction, n_columns)
        remove_direction(w, direction, cols)
        edited[layer] = cols
    return edited
