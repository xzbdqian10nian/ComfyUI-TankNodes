from pathlib import Path

import pytest


@pytest.mark.parametrize("model,variant", [
    ("Ternary-Bonsai-2-27B-PQ2_0.gguf", "stock"),
    ("Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf", "abliterated_v2"),
])
def test_loader_selects_prism_backend(module, tmp_path, monkeypatch, model, variant):
    catalog = module("model_catalog")
    nodes = module("local_nodes")
    monkeypatch.setenv("QWEN38_MODEL_DIR", str(tmp_path))
    for filename in (model, catalog.BONSAI_MMPROJ):
        with (tmp_path / filename).open("wb") as file:
            file.truncate(2 * 1024 * 1024)
    assert model in nodes.Qwen38VLLoader._model_choices()
    backend, info = nodes.Qwen38VLLoader().load_model(model, catalog.BONSAI_MMPROJ, 1024, 512, -1, True)
    assert backend.variant == variant
    assert backend.backend_kind == "local_bonsai_server"
    assert "PrismML" in info
    with pytest.raises(nodes.BackendError, match="matching projector"):
        nodes.Qwen38VLLoader().load_model(model, model, 1024, 512, -1, True)
    backend.unload()


def test_bonsai_fast_defaults_and_reasoning(module, tmp_path, monkeypatch):
    backends = module("backends")
    bonsai = module("bonsai_backend")
    settings = backends.LocalRuntimeSettings(
        tmp_path / "model.gguf", tmp_path / "mmproj.gguf", 1024, 512, -1, True,
    )
    backend = bonsai.LocalBonsaiBackend(settings, "abliterated_v2")
    backend.configure_chat(8192, "auto")
    assert backend.n_ctx == 16384
    command = backend._server_command(Path("/prism/llama-server"), 12345)
    assert command[command.index("--ctx-size") + 1] == "16384"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q8_0"
    assert "--spec-type" not in command
    api = bonsai._BonsaiHTTPBackend("http://127.0.0.1:12345/v1", "", "", "bonsai2-27b", 30)
    body = {}
    api._apply_reasoning(body, "auto")
    assert body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning_effective=off" in api.reasoning_info()
    api._auto_visual = True
    body = {}
    api._apply_reasoning(body, "auto")
    assert body == {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"}
    assert "reasoning_effective=low" in api.reasoning_info()
    body = {}
    api._apply_reasoning(body, "medium")
    assert body == {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "medium"}
    monkeypatch.setattr(backends.OpenAICompatibleBackend, "complete", lambda self, **kwargs: self._auto_visual)
    assert api.complete(messages=[{"role": "user", "content": "polish this prompt"}]) is False
    assert api.complete(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]) is True
    backend.unload()


def test_empty_eos_retries_once_without_hiding_reasoning_exhaustion(module, tmp_path, monkeypatch):
    backends = module("backends")
    bonsai = module("bonsai_backend")
    settings = backends.LocalRuntimeSettings(
        tmp_path / "model.gguf", tmp_path / "mmproj.gguf", 1024, 512, -1, False,
    )
    backend = bonsai.LocalBonsaiBackend(settings, "stock")
    monkeypatch.setattr(backend, "ensure_loaded", lambda: None)

    class Client:
        def __init__(self):
            self.seeds = []
            self.tokens = 1

        def complete(self, **kwargs):
            self.seeds.append(kwargs["seed"])
            if len(self.seeds) == 1:
                return iter([{"choices": [{"delta": {"content": ""}}], "usage": {"completion_tokens": self.tokens}}])
            return iter([{"choices": [{"delta": {"content": "red square"}}], "usage": {"completion_tokens": 3}}])

        def unload(self):
            pass

    client = Client()
    backend._client = client
    chunks = list(backend.complete(stream=True, messages=[], seed=1))
    assert client.seeds == [1, 42]
    assert chunks[-1]["choices"][0]["delta"]["content"] == "red square"
    assert backend.empty_response_retries == 1

    client.seeds.clear()
    client.tokens = 512
    with pytest.raises(backends.BackendError, match="Increase max_tokens"):
        list(backend.complete(stream=True, messages=[], seed=1))
    assert client.seeds == [1]
    backend.unload()
