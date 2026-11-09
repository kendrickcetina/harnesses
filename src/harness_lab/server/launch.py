"""Start the model server from Python (the notebook and the tests use this)."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from harness_lab import config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(base_url: str, timeout: float = 900, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see the log file")
        try:
            if httpx.get(base_url.removesuffix("/v1") + "/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"server at {base_url} not ready after {timeout}s")


def start_server(
    model: str | None = None, port: int | None = None, extra: list[str] | None = None, log_file: Path | None = None
) -> subprocess.Popen:
    """Run ``harness-serve`` as a background process and wait until it answers."""
    port = port or config.PORT
    log_file = log_file or config.OUTPUTS_DIR / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "harness_lab.server.app", "--port", str(port), "--model", model or config.MODEL_ID]
    if config.SCRIPTED:
        cmd.append("--scripted")
    proc = subprocess.Popen(cmd + (extra or []), stdout=log_file.open("w"), stderr=subprocess.STDOUT)
    wait_ready(f"http://127.0.0.1:{port}/v1", proc=proc)
    print(f"Server ready at http://127.0.0.1:{port}/v1 (pid {proc.pid}, log {log_file})")
    return proc


def serve_in_thread(app, port: int):
    """Run a FastAPI app in a daemon thread (used with ScriptedEngine in tests)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    wait_ready(f"http://127.0.0.1:{port}/v1", timeout=30)
    return server
