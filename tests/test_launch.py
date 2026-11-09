"""start_server's port check: detect real listeners, not leftovers of a stopped server."""

import socket
import time

from harness_lab.server.launch import port_in_use


def listener() -> socket.socket:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    return srv


def test_port_with_a_listener_is_in_use():
    srv = listener()
    try:
        assert port_in_use(srv.getsockname()[1])
    finally:
        srv.close()


def test_port_of_a_stopped_server_is_free():
    """A server that closed its connections and stopped leaves TIME_WAIT sockets for ~60 s
    (e.g. after restarting the notebook kernel). Restarting on that port must be allowed."""
    srv = listener()
    port = srv.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = srv.accept()
    conn.close()  # the server closes first -> its side goes to TIME_WAIT
    time.sleep(0.1)
    client.close()
    srv.close()
    time.sleep(0.1)
    assert not port_in_use(port)


def _scripted(monkeypatch, tmp_path, port):
    from harness_lab import config

    monkeypatch.setattr(config, "SCRIPTED", True)
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(config, "PORT", port)
    monkeypatch.setattr(config, "BASE_URL", f"http://127.0.0.1:{port}/v1")
    return config


def test_port_taken_by_another_program_uses_a_free_port(tmp_path, monkeypatch, capsys):
    """E.g. another web app on 8765: leave it alone, start on a free port, and point config at it."""
    import httpx

    from harness_lab.server.launch import start_server

    foreign = listener()
    port = foreign.getsockname()[1]
    config = _scripted(monkeypatch, tmp_path, port)
    proc = start_server()
    try:
        assert config.PORT != port and config.BASE_URL == f"http://127.0.0.1:{config.PORT}/v1"
        assert httpx.get(config.BASE_URL.removesuffix("/v1") + "/health").json()["model"] == "scripted"
        assert "used by another program" in capsys.readouterr().out
    finally:
        proc.terminate()
        proc.wait(10)
        foreign.close()


def test_previous_harness_server_on_the_port_is_restarted(tmp_path, monkeypatch, capsys):
    """E.g. a server left by a restarted kernel: stop it and start a fresh one on the same port."""
    from harness_lab.server.launch import free_port, start_server

    port = free_port()
    config = _scripted(monkeypatch, tmp_path, port)
    old = start_server()
    new = start_server()
    try:
        assert old.wait(10) is not None  # the previous one was stopped...
        assert new.poll() is None and config.PORT == port  # ...and the new one serves the same port
        assert f"Stopping the previous harness server on port {port} (pid {old.pid})" in capsys.readouterr().out
    finally:
        new.terminate()
        new.wait(10)


def test_explicit_port_taken_by_another_program_fails(tmp_path, monkeypatch):
    import pytest

    from harness_lab.server.launch import start_server

    foreign = listener()
    port = foreign.getsockname()[1]
    _scripted(monkeypatch, tmp_path, port)
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            start_server(port=port)
    finally:
        foreign.close()


def test_scripted_start_leaves_servers_on_other_ports_alone(tmp_path, monkeypatch):
    """The notebook runs the offline tests while its real server is up: they must not stop it."""
    from harness_lab.server.launch import free_port, start_server

    config = _scripted(monkeypatch, tmp_path, free_port())
    first = start_server()
    monkeypatch.setattr(config, "PORT", free_port())
    second = start_server()
    try:
        assert first.poll() is None and second.poll() is None
    finally:
        for proc in (first, second):
            proc.terminate()
            proc.wait(10)
