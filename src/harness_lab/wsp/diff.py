"""Read an edit straight from the weights: which tensors changed, where, and how.

Works on raw ``.safetensors`` files, so it needs no GPU and never builds the model.

    uv run harness-wsp diff Qwen/Qwen3.5-0.8B RikiyaT/Qwen3.5-0.8B-weight-space-programming
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass
class TensorDiff:
    name: str
    shape: tuple[int, ...]
    n_changed: int
    changed_columns: list[int]
    rank: int | None = None  # numerical rank of the change (2-D tensors)
    top_singular_share: float | None = None  # how much of the change is one direction
    projection_match: float | None = None  # cosine with "remove direction u from these columns"
    direction: torch.Tensor | None = None  # the u in W <- W - u u^T W (unit vector)

    @property
    def fraction(self) -> float:
        total = 1
        for s in self.shape:
            total *= s
        return self.n_changed / total


def resolve(model: str) -> Path:
    """Local folder for a repo id (downloaded if needed) or a path."""
    path = Path(model)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors"]))


def _tensors(folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for file in sorted(folder.glob("*.safetensors")):
        with safe_open(str(file), framework="pt") as f:
            for key in f.keys():
                index[key] = file
    return index


def analyse(name: str, base: torch.Tensor, edited: torch.Tensor) -> TensorDiff | None:
    delta = edited.float() - base.float()
    changed = delta != 0
    n = int(changed.sum())
    if n == 0:
        return None
    if delta.ndim != 2:
        return TensorDiff(name, tuple(base.shape), n, [])
    cols = torch.nonzero(changed.any(dim=0)).flatten().tolist()
    sub = delta[:, cols]
    s = torch.linalg.svdvals(sub)
    rank = int((s > s[0] * 1e-3).sum())
    share = float(s[0] ** 2 / (s**2).sum())
    u = torch.linalg.svd(sub, full_matrices=False).U[:, 0]
    # If the edit removed direction u from those columns, delta == -u u^T W.
    expected = -torch.outer(u, u @ base.float()[:, cols])
    match = float(torch.nn.functional.cosine_similarity(expected.flatten(), sub.flatten(), dim=0))
    return TensorDiff(name, tuple(base.shape), n, cols, rank, share, match, u)


def compare_checkpoints(base: str, edited: str) -> list[TensorDiff]:
    """All tensors that differ between two checkpoints, largest change first."""
    a, b = _tensors(resolve(base)), _tensors(resolve(edited))
    diffs = []
    for key in sorted(set(a) & set(b)):
        with safe_open(str(a[key]), framework="pt") as fa, safe_open(str(b[key]), framework="pt") as fb:
            ta, tb = fa.get_tensor(key), fb.get_tensor(key)
        if ta.shape != tb.shape:
            diffs.append(TensorDiff(key, tuple(ta.shape), -1, []))
            continue
        d = analyse(key, ta, tb)
        if d:
            diffs.append(d)
    return sorted(diffs, key=lambda d: -d.n_changed)


def format_report(diffs: list[TensorDiff]) -> str:
    if not diffs:
        return "The checkpoints are identical."
    total = sum(max(d.n_changed, 0) for d in diffs)
    lines = [f"{len(diffs)} tensors changed, {total:,} weights in total", ""]
    for d in diffs:
        lines.append(f"{d.name}  shape={list(d.shape)}  changed={d.n_changed:,} ({d.fraction:.3%})")
        if d.rank is not None:
            cols = d.changed_columns
            shown = ", ".join(map(str, cols[:8])) + (" ..." if len(cols) > 8 else "")
            lines.append(
                f"    columns={len(cols)} [{shown}]  rank={d.rank}  "
                f"top-direction share={d.top_singular_share:.3f}  "
                f"matches 'remove u from columns'={d.projection_match:+.3f}"
            )
    return "\n".join(lines)
