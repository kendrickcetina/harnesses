"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default model for the harness. Qwen3.5-4B in bf16 is ~9.3 GB: it fits in a 16 GB
# DGX MIG slice and leaves room for the KV cache.
MODEL_ID = os.environ.get("HARNESS_MODEL", "Qwen/Qwen3.5-4B")

# Small model for quick tests on a laptop CPU/MPS.
SMALL_MODEL_ID = "Qwen/Qwen3.5-0.8B"

HOST = os.environ.get("HARNESS_HOST", "127.0.0.1")
# Not 8000: on the DGX a shared vLLM already listens there.
PORT = int(os.environ.get("HARNESS_PORT", "8765"))
BASE_URL = os.environ.get("HARNESS_BASE_URL", f"http://{HOST}:{PORT}/v1")

# The local server does not check keys, but every client wants one.
API_KEY = os.environ.get("HARNESS_API_KEY", "sk-local")

# Where generated files and harness state live (all git-ignored).
WORKSPACE_DIR = Path(os.environ.get("HARNESS_WORKSPACE", REPO_ROOT / "workspace"))
DSH_HOME = Path(os.environ.get("HARNESS_DSH_HOME", REPO_ROOT / ".dsh_home"))
OUTPUTS_DIR = Path(os.environ.get("HARNESS_OUTPUTS", REPO_ROOT / "outputs"))
HARNESS_CODEX_SANDBOX = "off"

# HARNESS_SCRIPTED=1 runs everything with a scripted fake model: no GPU, no downloads.
# Useful to rehearse the tutorial offline; the answers are fixed.
SCRIPTED = os.environ.get("HARNESS_SCRIPTED") == "1"
