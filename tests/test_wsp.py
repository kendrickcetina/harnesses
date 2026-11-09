"""Weight-space programming utilities on tiny synthetic checkpoints."""

import torch
from safetensors.torch import save_file

from harness_lab.wsp.diff import compare_checkpoints, format_report
from harness_lab.wsp.edit import remove_direction, top_columns
from harness_lab.wsp.probe import extract_number, looks_looping


def make_pair(tmp_path, cols=(1, 4, 6)):
    torch.manual_seed(0)
    hidden, inter = 16, 32
    tensors = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(hidden, inter) for i in range(3)}
    tensors["model.embed_tokens.weight"] = torch.randn(10, hidden)
    base, edited = tmp_path / "base", tmp_path / "edited"
    base.mkdir()
    edited.mkdir()
    save_file(tensors, base / "model.safetensors")
    u = torch.randn(hidden)
    u /= u.norm()
    changed = {k: v.clone() for k, v in tensors.items()}
    remove_direction(changed["model.layers.2.mlp.down_proj.weight"], u, list(cols))
    save_file(changed, edited / "model.safetensors")
    return base, edited, u


def test_diff_reads_a_direction_edit(tmp_path):
    base, edited, u = make_pair(tmp_path)
    diffs = compare_checkpoints(str(base), str(edited))
    assert len(diffs) == 1
    d = diffs[0]
    assert d.name == "model.layers.2.mlp.down_proj.weight"
    assert d.changed_columns == [1, 4, 6]
    assert d.n_changed == 16 * 3
    assert d.rank == 1
    assert d.projection_match > 0.999  # it is exactly "remove u from these columns"
    assert abs(float(d.direction @ u)) > 0.999  # and we recover u (up to sign)
    assert "rank=1" in format_report(diffs)


def test_identical_checkpoints(tmp_path):
    base, _, _ = make_pair(tmp_path)
    assert format_report(compare_checkpoints(str(base), str(base))) == "The checkpoints are identical."


def test_remove_direction_makes_columns_orthogonal():
    torch.manual_seed(1)
    w = torch.randn(8, 20)
    u = torch.randn(8)
    cols = top_columns(w, u, 5)
    scores = (u / u.norm() @ w).abs()
    assert set(cols) == set(torch.topk(scores, 5).indices.tolist())
    touched = remove_direction(w, u, cols)
    assert touched == 8 * 5
    assert torch.allclose(u / u.norm() @ w[:, cols], torch.zeros(5), atol=1e-5)


def test_probe_helpers():
    assert extract_number("<think>2+2</think>So the answer is \\boxed{1,234}.") == "1234"
    assert extract_number("Total: 42.") == "42"
    assert looks_looping("wait let me check again " * 50)
    assert not looks_looping("A short and perfectly normal answer with no repetition at all here today.")


def test_split_answers_loops_or_errors():
    from harness_lab.wsp.cli import split_answers
    from harness_lab.wsp.probe import ProbeResult

    loop = "wait let me check again " * 50
    result = ProbeResult(
        accuracy=0.5,
        truncated_rate=0.25,
        loop_rate=0.25,
        outputs=["right", "wrong", "cut", loop],
        truncated=[False, False, True, False],
        correct=[True, False, True, False],
    )
    assert split_answers(result, "loops") == (["cut", loop], ["right", "wrong"])
    assert split_answers(result, "errors") == (["wrong", loop], ["right", "cut"])
