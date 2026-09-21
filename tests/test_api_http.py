import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def api_server():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, self.headers.get("Authorization")))
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "test reasoning"}}]},
                    {"choices": [{"index": 0, "delta": {"content": "test answer"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                ]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "test answer", "reasoning_content": "test reasoning"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("node_id", ["VisionAPIEnv", "VisionAPIDirect"])
def test_api_nodes_over_real_http(module, plugin, monkeypatch, api_server, fallback, node_id):
    base_url, requests = api_server
    backends = module("backends")
    if fallback:
        def unavailable(self):
            raise backends.BackendError("The openai package is unavailable")
        monkeypatch.setattr(backends.OpenAICompatibleBackend, "_ensure_client", unavailable)
    monkeypatch.setenv("TANK_TEST_API_KEY", "synthetic-key")
    kwargs = {"api_key_env": "TANK_TEST_API_KEY"} if node_id == "VisionAPIEnv" else {"api_key": "synthetic-key"}
    result = plugin.NODE_CLASS_MAPPINGS[node_id]().request(base_url=base_url, model="test-model", system_prompt="test persona",
        prompt="test prompt", max_tokens=0, temperature=0, thinking_mode="medium", **kwargs)["result"]
    assert result[0] == "test answer" and result[3] == "test reasoning"
    assert json.loads(result[1]) == {"prompt_tokens": "10", "completion_tokens": "5"}
    path, request, authorization = requests[0]
    assert path == "/v1/chat/completions"
    assert "max_tokens" not in request and "temperature" not in request
    assert request["reasoning_effort"] == "medium"
    assert request["messages"][0] == {"role": "system", "content": "test persona"}
    assert authorization == "Bearer synthetic-key"
    assert request["stream"] is not fallback


def test_environment_key_host_boundary(module, monkeypatch):
    backends = module("backends")
    monkeypatch.delenv("COMFYUI_API_ALLOWED_HOSTS", raising=False)
    with pytest.raises(backends.BackendError):
        backends.OpenAICompatibleBackend(base_url="https://untrusted.example/v1", model="test", api_key="", api_key_env="TANK_TEST_API_KEY", timeout=1, restrict_endpoint=True)


def test_key_source_info(module, monkeypatch):
    backends = module("backends")
    monkeypatch.setenv("TANK_TEST_API_KEY", "synthetic-key")
    backend = backends.OpenAICompatibleBackend(base_url="http://localhost/v1", model="test", api_key="", api_key_env="TANK_TEST_API_KEY", timeout=1)
    assert "key_source=TANK_TEST_API_KEY" in backend.info()
    assert "synthetic-key" not in backend.info()
