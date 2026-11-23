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


def test_back_to_back_non_stream_requests_do_not_reuse_a_closed_client(module, api_server, monkeypatch):
    # Regression: the non-stream path cancels the worker task as soon as the single
    # result arrives. If that cancellation lands while the finally block is still
    # awaiting client.close(), the backend used to keep a reference to the closed
    # client and the next request failed with "Cannot send a request, as the client
    # has been closed". Make close() slow so the cancellation reliably lands there.
    import asyncio
    import openai

    original_close = openai.AsyncOpenAI.close

    async def slow_close(self):
        await original_close(self)
        await asyncio.sleep(0.3)

    monkeypatch.setattr(openai.AsyncOpenAI, "close", slow_close)
    base_url, requests = api_server
    backends = module("backends")
    chat = module("chat")
    api = backends.OpenAICompatibleBackend(base_url, "synthetic-key", "", "test-model", 30)
    for _ in range(3):
        result = api.complete(messages=[{"role": "user", "content": "hi"}], max_tokens=16, stream=False)
        message, _usage = chat._extract_completion(result)
        assert chat._split_message(message)[0] == "test answer"
    assert len(requests) == 3


def test_api_direct_consumes_h3_image_t2i_and_edit_templates(plugin, monkeypatch):
    from pathlib import Path
    import torch

    replies = iter([
        json.dumps({"expanded_caption": "a warm ceramic coffee cup on wood", "aspect_ratio": "1:1"}),
        json.dumps({"expanded_caption": "the person wears a bright red hat outdoors", "target_aspect_ratio": "3:2"}),
    ])
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            content = next(replies)
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in (
                    {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                ):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        root = Path(__file__).resolve().parents[1]
        t2i_tpl = (root / "prompt_templates/h3_image_t2i_official.txt").read_text("utf-8") + "\n\nAdditional rules from the workflow owner:\nUse {\"projection\": \"360\"}"
        edit_tpl = (root / "prompt_templates/h3_image_edit_official.txt").read_text("utf-8")

        out_t2i = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url=base_url,
            model="test-model",
            system_prompt=t2i_tpl,
            prompt="a coffee cup",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key="synthetic-key",
        )["result"]
        assert out_t2i[0] == "a warm ceramic coffee cup on wood"
        sent_t2i = requests[-1][1]["messages"]
        assert len(sent_t2i) == 2 and sent_t2i[0]["role"] == "system"
        assert "a coffee cup" in sent_t2i[1]["content"]
        assert '{"projection": "360"}' in sent_t2i[1]["content"]

        img = torch.rand((1, 32, 48, 3), dtype=torch.float32)
        out_edit = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url=base_url,
            model="test-model",
            system_prompt=edit_tpl,
            prompt="change hat color",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key="synthetic-key",
            image=img,
        )["result"]
        assert out_edit[0] == "the person wears a bright red hat outdoors"
        sent_edit = requests[-1][1]["messages"]
        assert len(sent_edit) == 1 and sent_edit[0]["role"] == "user"
        assert any(part.get("type") == "image_url" for part in sent_edit[0]["content"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_api_direct_h3_image_retries_and_rejects_malformed_json(plugin):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    t2i_tpl = (root / "prompt_templates/h3_image_t2i_official.txt").read_text("utf-8")
    replies = iter([
        ('{"expanded_caption": "truncated', "stop"),
        (json.dumps({"expanded_caption": "recovered caption", "aspect_ratio": "16:9"}), "stop"),
        (json.dumps({"expanded_caption": "cut short despite valid json", "aspect_ratio": "1:1"}), "length"),
        (json.dumps({"expanded_caption": "retried after length", "aspect_ratio": "1:1"}), "stop"),
        ('{"expanded_caption": "ok", "extra": NaN}', "stop"),
        ('still {broken json', "stop"),
        ('still {truncated on second try', "length"),
        ('still {truncated again', "length"),
    ])
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            content, finish = next(replies)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4}},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        node = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]()
        out = node.request(
            base_url=base_url,
            model="test-model",
            system_prompt=t2i_tpl,
            prompt="a coffee cup",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key="synthetic-key",
        )["result"]
        assert out[0] == "recovered caption"
        assert len(requests) == 2

        out_len = node.request(
            base_url=base_url,
            model="test-model",
            system_prompt=t2i_tpl,
            prompt="a coffee cup",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key="synthetic-key",
        )["result"]
        assert out_len[0] == "retried after length"
        assert len(requests) == 4

        with pytest.raises(ValueError, match="H3 Image PE 返回格式不合格"):
            node.request(
                base_url=base_url,
                model="test-model",
                system_prompt=t2i_tpl,
                prompt="a coffee cup",
                max_tokens=4096,
                temperature=0,
                thinking_mode="medium",
                api_key="synthetic-key",
            )

        with pytest.raises(ValueError, match="finish_reason=length"):
            node.request(
                base_url=base_url,
                model="test-model",
                system_prompt=t2i_tpl,
                prompt="a coffee cup",
                max_tokens=4096,
                temperature=0,
                thinking_mode="medium",
                api_key="synthetic-key",
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_api_direct_redacts_secret_in_chained_traceback(module, plugin, monkeypatch):
    from pathlib import Path
    import traceback
    api_nodes = module("api_nodes")

    def boom(*a, **kw):
        # Set BOTH __cause__ and __context__ (without relying solely on 3.11+ add_note)
        # so Python 3.10+ still exercises dual-branch exception chain traversal.
        ctx_err = RuntimeError("context leak sk-chain-leak-9999")
        cause_err = RuntimeError("upstream cause error")
        if hasattr(ctx_err, "add_note"):
            ctx_err.add_note("note leak sk-chain-leak-9999")
        try:
            raise ctx_err
        except RuntimeError:
            raise ValueError("outer parser error") from cause_err

    monkeypatch.setattr(api_nodes, "_complete_direct_messages", boom)
    monkeypatch.setattr(api_nodes, "OpenAICompatibleBackend", lambda **kw: type("B", (), {"backend_kind": "openai_compatible", "model": kw["model"], "unload": lambda self: None})())
    root = Path(__file__).resolve().parents[1]
    t2i_tpl = (root / "prompt_templates/h3_image_t2i_official.txt").read_text("utf-8")
    try:
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url="https://api.example.com/v1",
            model="m",
            system_prompt=t2i_tpl,
            prompt="coffee",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key="sk-chain-leak-9999",
        )
        pytest.fail("expected exception")
    except Exception as exc:
        tb = traceback.format_exc()
        assert "sk-chain-leak-9999" not in tb
        assert exc.__cause__ is None and exc.__context__ is None

    # Multi-arg exceptions (e.g. JSONDecodeError carrying key only in .doc) must sanitize cleanly without TypeError.
    def boom_json(*a, **kw):
        raise json.JSONDecodeError("bad json", '{"k":"sk-chain-leak-9999"}', 0)

    monkeypatch.setattr(api_nodes, "_complete_direct_messages", boom_json)
    try:
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url="https://api.example.com/v1",
            model="m",
            system_prompt=t2i_tpl,
            prompt="coffee",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key="sk-chain-leak-9999",
        )
        pytest.fail("expected exception")
    except Exception as exc:
        assert "sk-chain-leak-9999" not in traceback.format_exc()
        assert exc.__cause__ is None and exc.__context__ is None

    # Duplicate JSON key carrying secret also leaves no secret anywhere on exception chain,
    # and provider usage echoing a key (including quotes/backslashes) is redacted before JSON serialization.
    key2 = 'sk-"dup\\secret"'
    monkeypatch.setattr(
        api_nodes,
        "_complete_direct_messages",
        lambda *a, **kw: (f'{{{json.dumps(key2)}: 1, {json.dumps(key2)}: 2}}', "", "", {}, "stop"),
    )
    try:
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url="https://api.example.com/v1",
            model="m",
            system_prompt=t2i_tpl,
            prompt="coffee",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key=key2,
        )
        pytest.fail("expected exception")
    except Exception as exc:
        assert key2 not in traceback.format_exc()
        assert exc.__cause__ is None and exc.__context__ is None

    class UsageEchoBackend:
        backend_kind = "openai_compatible"
        model = "m"
        def __init__(self, echoed):
            self.echoed = echoed
        def unload(self):
            pass
        def complete(self, **kw):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": self.echoed, "completion_tokens": 1},
            }

    monkeypatch.setattr(api_nodes, "OpenAICompatibleBackend", lambda **kw: UsageEchoBackend(kw["api_key"]))
    out = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="https://api.example.com/v1",
        model="m",
        system_prompt="plain",
        prompt="hi",
        max_tokens=0,
        temperature=0,
        api_key=key2,
    )
    decoded_usage = json.loads(out["result"][1])
    assert key2 not in out["result"][1]
    assert decoded_usage["prompt_tokens"] == "***"

    # A key whose literal text looks like a JSON property name ('"prompt_tokens"') must not break JSON structure.
    literal_key = '"prompt_tokens"'
    out_lit = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="https://api.example.com/v1",
        model="m",
        system_prompt="plain",
        prompt="hi",
        max_tokens=0,
        temperature=0,
        api_key=literal_key,
    )
    decoded_lit = json.loads(out_lit["result"][1])
    assert decoded_lit == {"prompt_tokens": "***", "completion_tokens": "1"}

    # Attribute names in vars(exc) containing a secret trigger sanitization too.
    def boom_attr(*a, **kw):
        err = RuntimeError("clean msg")
        err.__dict__[key2] = "benign"
        raise err

    monkeypatch.setattr(api_nodes, "_complete_direct_messages", boom_attr)
    try:
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url="https://api.example.com/v1",
            model="m",
            system_prompt=t2i_tpl,
            prompt="coffee",
            max_tokens=4096,
            temperature=0,
            thinking_mode="medium",
            api_key=key2,
        )
        pytest.fail("expected exception")
    except Exception as exc:
        assert key2 not in vars(exc)
        assert exc.__cause__ is None and exc.__context__ is None


def test_api_direct_consumes_h3_video_template_and_exposes_ui_report(module, plugin, monkeypatch, api_server):
    from pathlib import Path

    base_url, requests = api_server
    root = Path(__file__).resolve().parents[1]
    h3_tpl = (root / "prompt_templates/h3_official.txt").read_text("utf-8")
    out = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url=base_url,
        model="test-model",
        system_prompt=h3_tpl,
        prompt="a cat jumps onto a table",
        max_tokens=4096,
        temperature=0,
        thinking_mode="medium",
        api_key="synthetic-key",
    )
    assert isinstance(out["result"][0], str) and out["result"][0]
    assert "\n\n---\n" not in out["result"][0]
    assert "\n\n---\n" in out["ui"]["text"][0]
    assert out["ui"]["tank_prompt_meta"] == [{"has_report": True, "analysis_failed": False}]
    sent_sys = requests[0][1]["messages"][0]["content"]
    assert "OFFICIAL GUIDE" in sent_sys and "MiniMax H3" in sent_sys

    # Plain non-H3 chat omits tank_prompt_meta
    plain = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url=base_url,
        model="test-model",
        system_prompt="plain persona",
        prompt="hello",
        max_tokens=0,
        temperature=0,
        thinking_mode="medium",
        api_key="synthetic-key",
    )
    assert "tank_prompt_meta" not in plain["ui"]

    # Simulate H3 edit analysis_failed report -> ui.text and tank_prompt_meta both signal fallback
    h3_opt = module("h3_optimizer")
    monkeypatch.setattr(h3_opt, "run_h3", lambda *a, **kw: ("fallback draft prompt", "template=H3 video-edit\nanalysis=failed, generic fallback\n- analysis_failed: check by hand"))
    out_fail = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url=base_url,
        model="test-model",
        system_prompt=h3_tpl,
        prompt="replace person",
        max_tokens=4096,
        temperature=0,
        thinking_mode="medium",
        api_key="synthetic-key",
    )
    assert out_fail["result"][0] == "fallback draft prompt"
    assert "analysis=failed, generic fallback" in out_fail["ui"]["text"][0]
    assert out_fail["ui"]["tank_prompt_meta"] == [{"has_report": True, "analysis_failed": True}]


def test_api_http_sends_zero_requests_when_saved_key_endpoint_mismatches(module, plugin, monkeypatch, tmp_path, api_server):
    base_url, requests = api_server
    api_nodes = module("api_nodes")
    cfg_file = tmp_path / "tank_runtime_api.json"
    cfg_file.write_text(json.dumps({"base_url": "https://configured.example/v1", "model": "m", "api_key": "sk-secret"}), "utf-8")
    monkeypatch.setattr(api_nodes, "RUNTIME_API_CONFIG_PATH", str(cfg_file))
    with pytest.raises(ValueError, match="自定义 Base URL"):
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url=base_url,
            model=api_nodes.API_MODEL_PLACEHOLDER,
            api_key=api_nodes.API_KEY_PLACEHOLDER,
            system_prompt="persona",
            prompt="hello",
            max_tokens=0,
            temperature=0,
        )
    assert requests == []


def _write_mirror(path, payload, seed):
    import base64, hashlib
    mask = hashlib.sha256(seed.encode("utf-8")).digest()
    raw = json.dumps(payload).encode("utf-8")
    path.write_text(base64.b64encode(bytes(b ^ mask[i % len(mask)] for i, b in enumerate(raw))).decode("ascii"), "utf-8")


@pytest.fixture
def model_server():
    """Answers empty content for model 'empty-model', 'good answer' otherwise."""
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body["model"])
            text = "" if body["model"] == "empty-model" else "good answer"
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in ({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]},
                              {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()
    server.server_close()
    thread.join()


def test_mirror_default_falls_back_when_primary_returns_empty(module, plugin, monkeypatch, tmp_path, model_server):
    base_url, seen = model_server
    api_nodes = module("api_nodes")
    mirror = tmp_path / ".mirror_api"
    _write_mirror(mirror, {"base_url": base_url, "model": "empty-model", "api_key": "k1"}, api_nodes._MIRROR_KEY_SEED)
    _write_mirror(tmp_path / ".mirror_api_fallbacks", [{"base_url": base_url, "model": "good-model", "api_key": "k2"}],
                  api_nodes._MIRROR_KEY_SEED)
    monkeypatch.setattr(api_nodes, "MIRROR_API_CONFIG_PATH", str(mirror))
    result = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="", model="", api_key="", system_prompt="persona", prompt="p", max_tokens=0, temperature=0)["result"]
    assert result[0] == "good answer"
    assert seen[0] == "empty-model" and seen[-1] == "good-model"
    assert "备用1" in result[2]


def test_empty_api_response_raises_instead_of_returning_blank_prompt(module, plugin, monkeypatch, tmp_path, model_server):
    base_url, _ = model_server
    api_nodes = module("api_nodes")
    mirror = tmp_path / ".mirror_api"
    _write_mirror(mirror, {"base_url": base_url, "model": "empty-model", "api_key": "k1"}, api_nodes._MIRROR_KEY_SEED)
    monkeypatch.setattr(api_nodes, "MIRROR_API_CONFIG_PATH", str(mirror))
    with pytest.raises(Exception, match="为空"):
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url="", model="", api_key="", system_prompt="persona", prompt="p", max_tokens=0, temperature=0)
