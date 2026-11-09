"""Codex configuration: the sandbox is on by default and only an explicit opt-out turns it off."""

from harness_lab import clis


def _sandbox_line(tmp_path, monkeypatch) -> str:
    monkeypatch.setattr(clis, "CODEX_HOME", tmp_path)
    clis.codex_env(mcp=False)
    return next(line for line in (tmp_path / "config.toml").read_text().splitlines() if "sandbox_mode" in line)


def test_codex_sandbox_is_on_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_CODEX_SANDBOX", raising=False)
    assert _sandbox_line(tmp_path, monkeypatch) == 'sandbox_mode = "workspace-write"'


def test_codex_sandbox_explicit_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_CODEX_SANDBOX", "off")
    assert _sandbox_line(tmp_path, monkeypatch) == 'sandbox_mode = "danger-full-access"'


def test_run_codex_refuses_when_the_sandbox_cannot_start(monkeypatch):
    """Instead of letting the model retry failing commands until the timeout."""
    monkeypatch.delenv("HARNESS_CODEX_SANDBOX", raising=False)
    monkeypatch.setattr(clis.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(clis, "codex_sandbox_works", lambda: False)
    monkeypatch.setattr(clis, "_run", lambda *a, **k: pytest_fail())
    assert "HARNESS_CODEX_SANDBOX=off" in clis.run_codex("hola")


def pytest_fail():
    raise AssertionError("codex must not be started")


def test_codex_and_claude_find_the_project_python(monkeypatch):
    import sys
    from pathlib import Path

    project_bin = str(Path(sys.executable).parent)
    assert clis.claude_env()["PATH"].split(":")[0] == project_bin
