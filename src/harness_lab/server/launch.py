"""Start the model server from Python (the notebook and the tests use this)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import psutil

from harness_lab import config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_in_use(port: int) -> bool:
    """True if some process is listening on ``port``.

    SO_REUSEADDR, like uvicorn uses: without it, a server stopped less than ~60 s ago (e.g. a
    restarted kernel) still blocks bind() through its TIME_WAIT connections, although the
    new server could start fine. A real listener is still detected.
    """
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def stop_server(process: psutil.Process, timeout: float = 15) -> None:
    """Terminate a server process (SIGTERM, then SIGKILL) and wait until it is gone."""
    try:
        process.terminate()
        process.wait(timeout)
    except psutil.TimeoutExpired:
        process.kill()
        process.wait(5)
    except psutil.NoSuchProcess:
        pass


def _server_port(cmdline: list[str]) -> int:
    if "--port" in cmdline and cmdline.index("--port") + 1 < len(cmdline):
        return int(cmdline[cmdline.index("--port") + 1])
    return config.PORT


def find_servers() -> list[psutil.Process]:
    """Harness servers of the current user, both serving and still loading the model."""
    me = psutil.Process().username()
    found = []
    for p in psutil.process_iter(["cmdline", "username"]):
        cmd = p.info["cmdline"] or []
        if p.info["username"] == me and p.pid != os.getpid():
            if "harness_lab.server.app" in cmd or any(Path(c).name == "harness-serve" for c in cmd[:2]):
                found.append(p)
    return found


def _log_tail(log_file: Path | None, lines: int = 5) -> str:
    if log_file is None or not log_file.exists():
        return ""
    return "\n".join(log_file.read_text(errors="replace").strip().splitlines()[-lines:])


def wait_ready(
    base_url: str, timeout: float = 900, proc: subprocess.Popen | None = None, log_file: Path | None = None
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode}. Last lines of {log_file or 'the log'}:\n"
                + _log_tail(log_file)
            )
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
    """Run ``harness-serve`` as a background process and wait until it answers.

    * One of our servers already on that port (e.g. left by a restarted kernel): it is
      stopped and relaunched, so the model always runs the current code.
    * Starting a real model: our other servers with a real model are stopped too (they
      hold the GPU memory). Scripted servers, which use no GPU, are left alone.
    * The port belongs to another program: we leave it alone and take a free port.
      ``config.PORT`` and ``config.BASE_URL`` follow, so the rest of the notebook too.
    """
    requested = port
    port = port or config.PORT
    model = model or config.MODEL_ID

    for p in find_servers():
        try:
            cmdline = p.cmdline()
        except psutil.Error:
            continue
        other = _server_port(cmdline)
        if other == port:
            print(f"Stopping the previous harness server on port {port} (pid {p.pid})...")
            stop_server(p)
        elif not config.SCRIPTED and "--scripted" not in cmdline:
            # A real model on another port (e.g. from before a kernel restart): two copies do not
            # fit in a 16 GB GPU. Scripted servers (tests) use no GPU and are left alone.
            print(f"Stopping the previous harness server with a model on port {other} (pid {p.pid})...")
            stop_server(p)

    deadline = time.time() + 10  # a stopped server releases its port in a moment
    while port_in_use(port) and time.time() < deadline:
        time.sleep(0.5)
    if port_in_use(port):
        if requested:
            raise RuntimeError(f"port {port} is already in use by another program")
        new = free_port()
        print(f"Port {port} is used by another program (not a harness server): using port {new} instead.")
        port = new

    config.PORT = port
    config.BASE_URL = f"http://{config.HOST}:{port}/v1"
    log_file = log_file or config.OUTPUTS_DIR / "server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "harness_lab.server.app", "--host", config.HOST, "--port", str(port), "--model", model]
    if config.SCRIPTED:
        cmd.append("--scripted")
    # Unbuffered, so the log shows "Loaded ..." and every request as it happens.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd + (extra or []), stdout=log_file.open("w"), stderr=subprocess.STDOUT, env=env)
    wait_ready(config.BASE_URL, proc=proc, log_file=log_file)
    print(f"Server ready at {config.BASE_URL} (pid {proc.pid}, log {log_file})")
    return proc


def serve_in_thread(app, port: int):
    """Run a FastAPI app in a daemon thread (used with ScriptedEngine in tests)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    wait_ready(f"http://127.0.0.1:{port}/v1", timeout=30)
    return server
