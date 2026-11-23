import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class Backend:
    backend_kind = "local_llama_cpp"
    model = "test-model"

    def __init__(self, **kwargs):
        self.unloaded = 0
        self.kwargs = kwargs

    def complete(self, **kwargs):
        self.request = kwargs
        return {"choices": [{"message": {"content": "answer", "reasoning_content": "test reasoning"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7}}

    def unload(self):
        self.unloaded += 1


def chat_kwargs(cls, backend):
    values = {key: value[1]["default"] for key, value in cls.INPUT_TYPES()["required"].items() if key != "backend"}
    return {**values, "backend": backend, "prompt": "test", "system_prompt": "persona"}


def test_legacy_node_contract(plugin):
    old = json.loads((ROOT / "tests/fixtures/v060_contract.json").read_text())
    assert set(old) <= set(plugin.NODE_CLASS_MAPPINGS)
    for key, contract in old.items():
        cls = plugin.NODE_CLASS_MAPPINGS[key]
        schema = cls.INPUT_TYPES()
        for group in ("required", "optional"):
            fields = schema.get(group, {})
            assert list(fields)[:len(contract[group])] == list(contract[group])
            for field, previous in contract[group].items():
                kind, *options = fields[field]
                options = options[0] if options else {}
                assert ("COMBO" if isinstance(kind, list) else kind) == previous["type"]
                if field not in {"model_file", "mmproj_file"}:
                    assert options.get("default") == previous["default"]
                assert options.get("control_after_generate", False) == previous["control_after_generate"]
        assert list(cls.RETURN_TYPES[:len(contract["return_types"])]) == contract["return_types"]
        assert list(cls.RETURN_NAMES[:len(contract["return_names"])]) == contract["return_names"]


def test_names_locales_and_native_advanced(plugin):
    for lang in ("en", "zh"):
        definitions = json.loads((ROOT / f"locales/{lang}/nodeDefs.json").read_text())
        assert set(definitions) == set(plugin.NODE_CLASS_MAPPINGS)
        for key, cls in plugin.NODE_CLASS_MAPPINGS.items():
            node = definitions[key]
            assert node["display_name"].endswith("· Tank")
            assert cls.CATEGORY.startswith("TankNodes/")
            schema = cls.INPUT_TYPES()
            assert set(node["inputs"]) == set(schema.get("required", {})) | set(schema.get("optional", {}))
            assert len(node["outputs"]) == len(cls.RETURN_TYPES)
            if lang == "en":
                assert node["display_name"] == plugin.NODE_DISPLAY_NAME_MAPPINGS[key]
    fields = plugin.NODE_CLASS_MAPPINGS["VisionChat"].INPUT_TYPES()["required"]
    assert fields["top_p"][1]["advanced"] is True
    assert "advanced" not in fields["prompt"][1]


@pytest.mark.parametrize("legacy", ["backend_default", "thinking", "instruct", "none", "disabled"])
def test_legacy_reasoning_validation(plugin, legacy):
    for key in ("VisionChat", "VisionAPIEnv", "VisionAPIDirect"):
        cls = plugin.NODE_CLASS_MAPPINGS[key]
        assert cls.VALIDATE_INPUTS(thinking_mode=legacy) is True
        assert cls.VALIDATE_INPUTS(thinking_mode="invalid") is not True


def test_model_discovery_and_pairing(module, tmp_path, monkeypatch):
    models = module("model_catalog")
    monkeypatch.setenv("QWEN38_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("TANKNODES_AUTO_DOWNLOAD", "0")
    # With an empty folder the catalogue files are still offered for download.
    assert models._choices("model")[0] == models.DEFAULT_MODEL
    assert set(models._choices("model")) == set(models.KNOWN_FILES["model"])
    assert models._choices("mmproj")[0] == models.DEFAULT_MMPROJ
    with pytest.raises(FileNotFoundError, match="huggingface.co/unsloth"):
        models._resolve_file(models.DEFAULT_MODEL, "model")
    with pytest.raises(FileNotFoundError, match="No model"):
        models._resolve_file(models.MISSING_FILES["model"], "model")
    for filename in (models.DEFAULT_MODEL, models.DEFAULT_MMPROJ, "other/alt.GGUF", "other/MMPROJ.GGUF", "partial.gguf"):
        file = tmp_path / filename
        file.parent.mkdir(exist_ok=True)
        with file.open("wb") as f:
            f.truncate(2 * 1024 * 1024)
    (tmp_path / "partial.gguf.aria2").touch()
    choices = models._choices("model")
    assert choices[:2] == [models.DEFAULT_MODEL, "other/alt.GGUF"]
    assert "partial.gguf" not in choices
    assert models._choices("mmproj")[:2] == [models.DEFAULT_MMPROJ, "other/MMPROJ.GGUF"]
    assert models._resolve_file("other/alt.GGUF", "model") == tmp_path / "other/alt.GGUF"
    assert "projector_filename_match=True" in models.selection_info(models.DEFAULT_MODEL, models.DEFAULT_MMPROJ)
    with pytest.raises(RuntimeError, match="downloading"):
        models._resolve_file("partial.gguf", "model")
    # The abliterated Bonsai release reuses the official projector repository.
    assert models.KNOWN_FILES["mmproj"]["Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"] == "prism-ml/Ternary-Bonsai-2-27B-gguf"


def test_local_chat_outputs_media_and_unload(plugin):
    cls = plugin.NODE_CLASS_MAPPINGS["VisionChat"]
    backend = Backend()
    result = cls().generate(**chat_kwargs(cls, backend), image=torch.zeros((3, 8, 8, 3)), video_frames=torch.zeros((12, 8, 8, 3)))
    assert len(result["result"]) == 4
    assert result["result"][:3] == ("answer", "test reasoning", "answer")
    assert "images=3\nvideo_frames=8" in result["result"][3]
    assert backend.request["messages"][0] == {"role": "system", "content": "persona"}
    assert len([x for x in backend.request["messages"][1]["content"] if x["type"] == "image_url"]) == 11
    assert backend.unloaded == 0


@pytest.mark.parametrize("phase", ["configure", "load", "generate"])
def test_cleanup_for_all_failures(plugin, phase):
    cls = plugin.NODE_CLASS_MAPPINGS["VisionChat"]
    backend = Backend()
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    setattr(backend, {"configure": "configure_chat", "load": "ensure_loaded", "generate": "complete"}[phase], fail)
    kwargs = chat_kwargs(cls, backend)
    kwargs["unload_after"] = True
    with pytest.raises(RuntimeError, match="synthetic"):
        cls().generate(**kwargs)
    assert backend.unloaded == 1


@pytest.mark.parametrize("kind", ["VisionAPIEnv", "VisionAPIDirect"])
def test_api_preserves_outputs_and_appends_reasoning(module, plugin, monkeypatch, kind):
    created = []
    def make(**kwargs):
        backend = Backend(**kwargs)
        backend.backend_kind = "openai_compatible"
        created.append(backend)
        return backend
    monkeypatch.setattr(module("api_nodes"), "OpenAICompatibleBackend", make)
    node = plugin.NODE_CLASS_MAPPINGS[kind]()
    result = node.request(base_url="http://127.0.0.1/v1", model="test", system_prompt="", prompt="test", max_tokens=0, temperature=0, thinking_mode="medium")
    outputs = result["result"]
    assert outputs[0] == "answer"
    assert json.loads(outputs[1]) == {"prompt_tokens": "11", "completion_tokens": "7"}
    assert "api_elapsed=" in outputs[2]
    assert outputs[3:] == ("test reasoning", "answer")
    assert created[0].unloaded == 1
    assert created[0].kwargs["restrict_endpoint"] == (kind == "VisionAPIEnv")


def test_stream_cleanup_on_error(module, monkeypatch):
    chat = module("chat")
    closed = []
    def stream():
        try:
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise RuntimeError("disconnected")
        finally:
            closed.append(True)
    backend = Backend()
    backend.complete = lambda **kwargs: stream()
    kwargs = chat_kwargs(module("local_nodes").VisionChat, backend)
    kwargs.pop("context_length")
    kwargs.pop("unload_after")
    with pytest.raises(RuntimeError, match="disconnected"):
        chat._run_chat(**kwargs, image=None, video_frames=None, video=None, tools_json=None, progress_callback=lambda _: None)
    assert closed == [True]


def test_unload_dependency_and_repeat(plugin):
    cls = plugin.NODE_CLASS_MAPPINGS["VisionUnload"]
    assert cls.INPUT_TYPES()["optional"]["after"][1]["forceInput"] is True
    assert math.isnan(cls.IS_CHANGED())
    backend = Backend()
    assert cls().unload(backend, after="answer")["result"] == ("Model released · Tank",)
    assert backend.unloaded == 1


def test_missing_stream_usage_does_not_claim_zero_speed(plugin):
    cls = plugin.NODE_CLASS_MAPPINGS["VisionChat"]
    backend = Backend()
    backend.complete = lambda **kwargs: iter([{"choices": [{"delta": {"content": "hello"}}]}])
    result = cls().generate(**chat_kwargs(cls, backend))["result"]
    assert result[0] == "hello"
    assert "completion_tokens=n/a\nspeed=n/a\n" in result[3]


def test_failed_weight_load_releases_projector(module, monkeypatch, tmp_path):
    backends = module("backends")
    backend = backends.LocalQwen38Backend(backends.LocalRuntimeSettings(tmp_path / "model.gguf", tmp_path / "mmproj.gguf", 1024, 512, -1, False))
    closed = []
    handler = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(backend, "_make_handler", lambda: handler)
    def fail(**kwargs):
        raise RuntimeError("synthetic out of memory")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=fail))
    with pytest.raises(backends.BackendError, match="Unable to load"):
        backend.ensure_loaded()
    assert closed == [True]
    assert backend.llm is None and backend.chat_handler is None
    assert backends._ACTIVE_LOCAL_BACKEND is None


def test_reasoning_changes_do_not_reload(module, tmp_path):
    backends = module("backends")
    backend = backends.LocalQwen38Backend(backends.LocalRuntimeSettings(tmp_path / "model.gguf", tmp_path / "mmproj.gguf", 1024, 512, -1, False))
    llm = SimpleNamespace(close=lambda: None)
    backend.llm = llm
    backend.chat_handler = SimpleNamespace(extra_template_arguments={})
    backend.configure_chat(8192, "high")
    assert backend.llm is llm
    assert backend.chat_handler.extra_template_arguments["reasoning_effort"] == "medium"
    backend.configure_chat(8192, "max")
    assert backend.llm is llm
    assert backend.chat_handler.extra_template_arguments["reasoning_effort"] == "xhigh"
    backend.unload()


def test_direct_api_placeholders_resolve_and_allow_partial_override(module, plugin, monkeypatch, tmp_path):
    api_nodes = module("api_nodes")
    cfg_file = tmp_path / "tank_runtime_api.json"
    cfg_file.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "gpt-4o",
        "api_key": "sk-test-secret-value",
    }), encoding="utf-8")
    monkeypatch.setattr(api_nodes, "RUNTIME_API_CONFIG_PATH", str(cfg_file))

    created = []
    def make(**kwargs):
        backend = Backend(**kwargs)
        backend.backend_kind = "openai_compatible"
        created.append(backend)
        return backend
    monkeypatch.setattr(api_nodes, "OpenAICompatibleBackend", make)

    node = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]()
    node.request(
        base_url=api_nodes.API_BASE_URL_PLACEHOLDER,
        model=api_nodes.API_MODEL_PLACEHOLDER,
        api_key=api_nodes.API_KEY_PLACEHOLDER,
        system_prompt="",
        prompt="hello",
        max_tokens=0,
        temperature=0,
    )
    assert created[-1].kwargs["base_url"] == "https://api.example.com/v1"
    assert created[-1].kwargs["model"] == "gpt-4o"
    assert created[-1].kwargs["api_key"] == "sk-test-secret-value"

    # Per-workflow partial override keeps global base_url + api_key while overriding model.
    node.request(
        base_url=api_nodes.API_BASE_URL_PLACEHOLDER,
        model="deepseek-chat",
        api_key=api_nodes.API_KEY_PLACEHOLDER,
        system_prompt="",
        prompt="hello",
        max_tokens=0,
        temperature=0,
    )
    assert created[-1].kwargs["base_url"] == "https://api.example.com/v1"
    assert created[-1].kwargs["model"] == "deepseek-chat"
    assert created[-1].kwargs["api_key"] == "sk-test-secret-value"

    # Unresolved placeholders fail clearly without leaking secrets.
    cfg_file.unlink()
    with pytest.raises(ValueError, match="未配置外部 API 参数"):
        node.request(
            base_url=api_nodes.API_BASE_URL_PLACEHOLDER,
            model=api_nodes.API_MODEL_PLACEHOLDER,
            api_key=api_nodes.API_KEY_PLACEHOLDER,
            system_prompt="",
            prompt="hello",
            max_tokens=0,
            temperature=0,
        )


def test_direct_api_rejects_endpoint_mismatch_for_saved_keys(module, plugin, monkeypatch, tmp_path):
    api_nodes = module("api_nodes")
    cfg_file = tmp_path / "tank_runtime_api.json"
    cfg_file.write_text(json.dumps({
        "base_url": "https://api.example.com/v1",
        "model": "gpt-4o",
        "api_key": "sk-runtime-secret",
    }), encoding="utf-8")
    mirror_file = tmp_path / ".mirror_api"
    import base64, hashlib
    mask = hashlib.sha256(api_nodes._MIRROR_KEY_SEED.encode("utf-8")).digest()
    raw_mirror = json.dumps({"base_url": "https://mirror.example.com/v1", "model": "mirror-m", "api_key": "sk-mirror-secret"}).encode("utf-8")
    mirror_file.write_text(base64.b64encode(bytes(b ^ mask[i % len(mask)] for i, b in enumerate(raw_mirror))).decode("ascii"), "utf-8")

    monkeypatch.setattr(api_nodes, "RUNTIME_API_CONFIG_PATH", str(cfg_file))
    monkeypatch.setattr(api_nodes, "MIRROR_API_CONFIG_PATH", str(mirror_file))

    created = []
    def make(**kwargs):
        backend = Backend(**kwargs)
        backend.backend_kind = "openai_compatible"
        created.append(backend)
        return backend
    monkeypatch.setattr(api_nodes, "OpenAICompatibleBackend", make)

    node = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]()
    # Matching normalized runtime URL succeeds (e.g. trailing slash / chat/completions).
    node.request(
        base_url="https://API.example.com:443/v1/chat/completions/",
        model=api_nodes.API_MODEL_PLACEHOLDER,
        api_key=api_nodes.API_KEY_PLACEHOLDER,
        system_prompt="",
        prompt="hi",
        max_tokens=0,
        temperature=0,
    )
    assert created[-1].kwargs["base_url"] == "https://api.example.com/v1"
    assert created[-1].kwargs["api_key"] == "sk-runtime-secret"
    count_before = len(created)

    # Tier 2 runtime key with different base_url must fail before instantiating backend.
    with pytest.raises(ValueError, match="自定义 Base URL"):
        node.request(
            base_url="https://attacker.example/v1",
            model=api_nodes.API_MODEL_PLACEHOLDER,
            api_key=api_nodes.API_KEY_PLACEHOLDER,
            system_prompt="",
            prompt="hi",
            max_tokens=0,
            temperature=0,
        )
    # Tier 2 mirror fallback with different base_url also fails before backend creation.
    cfg_file.unlink()
    with pytest.raises(ValueError, match="自定义 Base URL"):
        node.request(
            base_url="https://attacker.example/v1",
            model="mirror-m",
            api_key=api_nodes.API_KEY_PLACEHOLDER,
            system_prompt="",
            prompt="hi",
            max_tokens=0,
            temperature=0,
        )
    # Tier 3 empty api_key with custom URL or default openai.com + explicit model also fails.
    with pytest.raises(ValueError, match="自定义 Base URL"):
        node.request(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            api_key="",
            system_prompt="",
            prompt="hi",
            max_tokens=0,
            temperature=0,
        )
    assert len(created) == count_before


def test_prompt_template_is_changed_invalidates_on_edit(module, plugin, monkeypatch, tmp_path):
    prompt_nodes = module("prompt_nodes")
    user_file = tmp_path / "tank_prompt_templates.json"
    monkeypatch.setattr(prompt_nodes, "_user_store_path", lambda: str(user_file))
    prompt_nodes._save_user_templates([{"label": "我的模板", "text": "v1 rules", "aliases": [], "updated": 1}])

    cls = plugin.NODE_CLASS_MAPPINGS["TankPromptTemplate"]
    fp1 = cls.IS_CHANGED("我的模板", extra_rules="rule a")
    assert cls().run("我的模板", extra_rules="rule a")[0].startswith("v1 rules")

    prompt_nodes._save_user_templates([{"label": "我的模板", "text": "v2 rules", "aliases": [], "updated": 2}])
    fp2 = cls.IS_CHANGED("我的模板", extra_rules="rule a")
    assert fp2 != fp1
    assert cls().run("我的模板", extra_rules="rule a")[0].startswith("v2 rules")
    assert cls.IS_CHANGED("我的模板", extra_rules="rule b") != fp2


def test_deepseek_official_uses_thinking_object_without_reasoning_effort(module):
    backends = module("backends")
    ds = backends.OpenAICompatibleBackend(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        api_key_env="",
        model="deepseek-chat",
        timeout=60.0,
    )
    body_off = {}
    ds._apply_reasoning(body_off, "off")
    assert body_off == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in body_off
    assert ds.last_reasoning_effort is None

    body_on = {}
    ds._apply_reasoning(body_on, "medium")
    assert body_on == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in body_on
    assert ds.last_reasoning_effort == "medium"
