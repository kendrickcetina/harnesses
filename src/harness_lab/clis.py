"""Codex CLI and Claude Code on the same local model.

Both are standalone programs (``codex`` and ``claude``). We do not touch your personal
configuration: each gets its own config folder inside the repo (``.codex_home/`` and
``.claude_home/``, both git-ignored) and is pointed at the local server.

* Codex speaks the OpenAI **Responses** API  -> ``$BASE_URL/responses``
* Claude Code speaks the Anthropic **Messages** API -> ``$BASE/v1/messages``

Both also load our MCP tools (the same ones DeepSeek Harness uses).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

from harness_lab import config

CODEX_HOME = config.REPO_ROOT / ".codex_home"
CLAUDE_HOME = config.REPO_ROOT / ".claude_home"


def _mcp_command() -> dict:
    return {
        "command": sys.executable,
        "args": ["-m", "harness_lab.tools.mcp_server"],
        "env": {"HARNESS_WORKSPACE": str(config.WORKSPACE_DIR)},
    }


def _env_with_project_python() -> dict[str, str]:
    """Our environment, with the project's Python first on PATH: when a harness runs
    ``python3`` in its shell it gets pandas, openpyxl and python-docx from ``.venv``."""
    path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    return {**os.environ, "PATH": path}


def codex_sandbox_enabled() -> bool:
    """Codex runs commands in its sandbox unless you opt out with HARNESS_CODEX_SANDBOX=off."""
    return os.environ.get("HARNESS_CODEX_SANDBOX", "on").lower() != "off"


@cache
def codex_sandbox_works() -> bool:
    """Can Codex's Linux sandbox run a command here?

    It needs bubblewrap (``bwrap``) and permission to create mounts. Inside some containers
    (e.g. code-server on the DGX) bwrap cannot mount its new root: every command the model
    runs fails, and a small model just retries until the timeout. Checked once per session.
    """
    if sys.platform != "linux" or not shutil.which("codex"):
        return True
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["codex", "sandbox", "-c", 'sandbox_mode="workspace-write"', "--", "true"],
            env={**os.environ, "CODEX_HOME": str(CODEX_HOME)},
            capture_output=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def codex_env(model: str | None = None, mcp: bool = True) -> dict[str, str]:
    """Write ``.codex_home/config.toml`` and return the environment that selects it."""
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    mcp_cfg = _mcp_command()
    sandbox = "workspace-write" if codex_sandbox_enabled() else "danger-full-access"
    lines = [
        f'model = "{model or config.MODEL_ID}"',
        'model_provider = "local"',
        "model_context_window = 32768",
        'approval_policy = "never"',
        f'sandbox_mode = "{sandbox}"',
        "",
        "[model_providers.local]",
        'name = "Local Qwen (harness-lab)"',
        f'base_url = "{config.BASE_URL}"',
        'wire_api = "responses"',
        'env_key = "HARNESS_API_KEY"',
    ]
    if mcp:
        lines += [
            "",
            "[mcp_servers.lab]",
            f"command = {json.dumps(mcp_cfg['command'])}",
            f"args = {json.dumps(mcp_cfg['args'])}",
            f"env = {{ HARNESS_WORKSPACE = {json.dumps(mcp_cfg['env']['HARNESS_WORKSPACE'])} }}",
            'default_tools_approval_mode = "approve"',
        ]
    (CODEX_HOME / "config.toml").write_text("\n".join(lines) + "\n")
    return {**_env_with_project_python(), "CODEX_HOME": str(CODEX_HOME), "HARNESS_API_KEY": config.API_KEY}


def claude_env(model: str | None = None) -> dict[str, str]:
    """Environment that sends Claude Code to the local server instead of Anthropic."""
    CLAUDE_HOME.mkdir(parents=True, exist_ok=True)
    model = model or config.MODEL_ID
    return {
        **_env_with_project_python(),
        "CLAUDE_CONFIG_DIR": str(CLAUDE_HOME),
        "ANTHROPIC_BASE_URL": config.BASE_URL.removesuffix("/v1"),
        "ANTHROPIC_AUTH_TOKEN": config.API_KEY,
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_AUTOUPDATER": "1",
    }


def claude_mcp_config() -> Path:
    path = CLAUDE_HOME / "mcp.json"
    CLAUDE_HOME.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {"lab": _mcp_command()}}, indent=2))
    return path


def _run(cmd: list[str], env: dict[str, str], timeout: int) -> str:
    config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=config.WORKSPACE_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    return (proc.stdout + ("\n" + proc.stderr if proc.returncode else "")).strip()


def run_codex(prompt: str, model: str | None = None, mcp: bool = True, timeout: int = 900) -> str:
    """``codex exec``: one non-interactive task in the workspace."""
    if not shutil.which("codex"):
        return "codex not found: run scripts/install_clis.sh (or npm i -g @openai/codex)"
    if codex_sandbox_enabled() and not codex_sandbox_works():
        return (
            "El sandbox de Codex (bubblewrap) no funciona en este entorno: cada comando fallaría y el "
            "modelo reintentaría hasta el timeout. Si este entorno ya está aislado (p. ej. un contenedor) "
            "y aceptas que Codex ejecute comandos sin sandbox, define HARNESS_CODEX_SANDBOX=off."
        )
    if not codex_sandbox_enabled():
        print("Aviso: HARNESS_CODEX_SANDBOX=off, Codex ejecuta los comandos sin sandbox.")
    cmd = ["codex", "exec", "--skip-git-repo-check", "-C", str(config.WORKSPACE_DIR), prompt]
    return _run(cmd, codex_env(model, mcp), timeout)


def run_claude(
    prompt: str, model: str | None = None, tools: str = "Bash,Read,Write,Edit", mcp: bool = True, timeout: int = 900
) -> str:
    """``claude -p``: one non-interactive task. Few tools = shorter prompt for a small model."""
    if not shutil.which("claude"):
        return "claude not found: run scripts/install_clis.sh"
    cmd = [
        "claude",
        "-p",
        prompt,
        "--tools",
        tools,
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        tools + (",mcp__lab" if mcp else ""),
    ]
    if mcp:
        cmd += ["--mcp-config", str(claude_mcp_config()), "--strict-mcp-config"]
    return _run(cmd, claude_env(model), timeout)
