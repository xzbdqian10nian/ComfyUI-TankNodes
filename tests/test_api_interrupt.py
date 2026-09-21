import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class Interrupted(BaseException):
    """ComfyUI's interrupt is a BaseException, not an ordinary API error."""


@pytest.fixture
def interrupt(module, monkeypatch):
    flag = threading.Event()

    def check():
        if flag.is_set():
            flag.clear()
            raise Interrupted()

    monkeypatch.setattr(module("api_transport"), "check_interrupted", check)
    monkeypatch.setattr(module("api_nodes"), "check_interrupted", check)
    return flag


@pytest.fixture
def slow_api():
    state = {
        "requests": [],
        "waiting": threading.Event(),
        "disconnected": threading.Event(),
        "stop": threading.Event(),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            model = body["model"]
            if model == "error":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"synthetic failure"}}')
                return
            if model != "headers":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if body["stream"] else "application/json")
                self.end_headers()
                if body["stream"]:
                    self.wfile.write(b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n')
                else:
                    self.wfile.write(b'{"choices":[')
                self.wfile.flush()
            state["waiting"].set()
            # The provider intentionally never finishes. Verify that the client
            # actually disconnects rather than merely hiding a running request.
            self.connection.settimeout(0.05)
            while not state["stop"].is_set():
                try:
                    if not self.connection.recv(1):
                        state["disconnected"].set()
                        return
                except socket.timeout:
                    continue
                except ConnectionError:
                    state["disconnected"].set()
                    return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        state["stop"].set()
        server.shutdown()
        server.server_close()
        thread.join()


def force_fallback(module, monkeypatch):
    backends = module("backends")

    def unavailable(self):
        raise backends.BackendError("The openai package is unavailable")

    monkeypatch.setattr(backends.OpenAICompatibleBackend, "_ensure_client", unavailable)


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("stage", ["headers", "body"])
@pytest.mark.parametrize("node_id", ["VisionAPIEnv", "VisionAPIDirect"])
def test_interrupt_closes_stalled_connection(module, plugin, monkeypatch, interrupt, slow_api,
                                             fallback, stage, node_id):
    if fallback:
        force_fallback(module, monkeypatch)
    url, state = slow_api
    outcome = {}
    status = []
    monkeypatch.setattr(module("api_nodes").StatusTicker, "send", lambda self, text, **kw: status.append(text))

    def run():
        try:
            outcome["result"] = plugin.NODE_CLASS_MAPPINGS[node_id]().request(
                base_url=url, model=stage, system_prompt="", prompt="synthetic",
                max_tokens=0, temperature=0,
            )
            outcome["downstream"] = True
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert state["waiting"].wait(5), "Request did not reach the local server"
        if stage == "body" and not fallback:
            deadline = time.monotonic() + 2
            while not any("generating" in text for text in status) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert any("generating" in text for text in status)
        started = time.monotonic()
        interrupt.set()
        worker.join(timeout=1.0)
        assert not worker.is_alive(), "Interrupt waited for the HTTP timeout"
        assert time.monotonic() - started < 1.0
        assert isinstance(outcome.get("error"), Interrupted)
        assert "result" not in outcome and "downstream" not in outcome
        assert not any("complete" in text for text in status)
        assert state["disconnected"].wait(1.0), "The old HTTP request is still open"
        assert len(state["requests"]) == 1, "The cancelled request was retried"
        assert not any(t.name == "TankNodes-API" for t in threading.enumerate())
        assert not interrupt.is_set(), "The flag would interrupt the next run"
    finally:
        interrupt.set()
        worker.join(timeout=2)
        interrupt.clear()


@pytest.mark.parametrize("fallback", [False, True])
def test_api_errors_remain_errors(module, plugin, monkeypatch, slow_api, fallback):
    if fallback:
        force_fallback(module, monkeypatch)
    url, state = slow_api
    with pytest.raises(module("backends").BackendError, match="synthetic failure"):
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url=url, model="error", system_prompt="", prompt="synthetic", max_tokens=0, temperature=0,
        )
    assert len(state["requests"]) == 1
    assert not any(t.name == "TankNodes-API" for t in threading.enumerate())


def test_interrupt_before_request_sends_nothing(plugin, interrupt, slow_api):
    url, state = slow_api
    interrupt.set()
    with pytest.raises(Interrupted):
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url=url, model="headers", system_prompt="", prompt="synthetic", max_tokens=0, temperature=0,
        )
    assert state["requests"] == []
