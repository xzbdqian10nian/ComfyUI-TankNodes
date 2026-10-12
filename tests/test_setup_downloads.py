"""First-use setup: resumable downloads, runtime install and backend choice."""
import hashlib
import io
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def file_server():
    files = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            name = self.path.lstrip("/").split("/")[-1]
            data = files.get(name)
            if data is None:
                self.send_error(404)
                return
            start = 0
            if self.headers.get("Range"):
                start = int(self.headers["Range"].split("=")[1].split("-")[0])
                self.send_response(206)
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield files, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.mark.parametrize("system,machine,cuda,build", [
    ("Linux", "x86_64", "13.0", "linux-cuda-13.3"),
    ("Linux", "x86_64", "12.8", "linux-cuda-12.8"),
    ("Linux", "x86_64", "12.1", "linux-cuda-12.4"),
    ("Linux", "x86_64", None, "linux-cpu"),
    ("Windows", "AMD64", "12.6", "windows-cuda-12.4"),
    ("Windows", "AMD64", "13.0", "windows-cuda-13.3"),
    ("Darwin", "arm64", None, "macos-arm64"),
])
def test_runtime_build_selection(module, system, machine, cuda, build):
    runtime = module("runtime_manager")
    assert runtime.select_build(system, machine, cuda) == build
    for name, digest in runtime.ASSETS[build]:
        assert name.endswith((".tar.gz", ".zip")) and len(digest) == 64


def test_runtime_unsupported_platform_explains_manual_install(module):
    runtime = module("runtime_manager")
    with pytest.raises(runtime.RuntimeUnavailable, match="BONSAI_LLAMA_SERVER"):
        runtime.select_build("Linux", "aarch64", "12.8")


def test_fetch_resumes_and_verifies(module, tmp_path, file_server):
    downloads = module("downloads")
    files, base = file_server
    data = bytes(range(256)) * 4096
    files["model.gguf"] = data
    dest = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.part").write_bytes(data[:1000])
    digest = hashlib.sha256(data).hexdigest()
    assert downloads.fetch(f"{base}/model.gguf", dest, sha256=digest, progress=lambda *a: None) == dest
    assert dest.read_bytes() == data and not (tmp_path / "model.gguf.part").exists()
    other = tmp_path / "bad.gguf"
    files["bad.gguf"] = data
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        downloads.fetch(f"{base}/bad.gguf", other, sha256="0" * 64, progress=lambda *a: None)
    assert not other.exists() and not (tmp_path / "bad.gguf.part").exists()


def test_runtime_lookup_order_and_install(module, tmp_path, file_server, monkeypatch):
    runtime = module("runtime_manager")
    files, base = file_server
    monkeypatch.setattr(runtime, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.delenv("BONSAI_LLAMA_SERVER", raising=False)
    assert runtime.find_server() is None
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for name, payload in (("llama-x/llama-server", b"#!/bin/sh\n"), ("llama-x/libllama.so.0.2.0", b"lib")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        link = tarfile.TarInfo("llama-x/libllama.so.0")
        link.type, link.linkname = tarfile.SYMTYPE, "libllama.so.0.2.0"
        bundle.addfile(link)
    files["runtime.tar.gz"] = archive.getvalue()
    monkeypatch.setattr(runtime, "RELEASE_URL", base + "/")
    monkeypatch.setitem(runtime.ASSETS, "linux-cpu", [("runtime.tar.gz", hashlib.sha256(archive.getvalue()).hexdigest())])
    monkeypatch.setattr(runtime, "select_build", lambda: "linux-cpu")
    monkeypatch.setattr(runtime, "_executable_name", lambda system=None: "llama-server")
    monkeypatch.setenv("TANKNODES_AUTO_DOWNLOAD", "0")
    with pytest.raises(runtime.RuntimeUnavailable, match="TANKNODES_AUTO_DOWNLOAD=0"):
        runtime.ensure_server()
    monkeypatch.delenv("TANKNODES_AUTO_DOWNLOAD")
    installed = runtime.ensure_server()
    assert installed == tmp_path / "runtime" / runtime.RELEASE / "llama-x/llama-server"
    assert installed.stat().st_mode & 0o111
    assert not (tmp_path / "runtime" / ".downloads").exists()
    custom = tmp_path / "custom-server"
    custom.write_text("x")
    monkeypatch.setenv("BONSAI_LLAMA_SERVER", str(custom))
    assert runtime.find_server() == custom
    monkeypatch.setenv("BONSAI_LLAMA_SERVER", str(tmp_path / "missing"))
    with pytest.raises(runtime.RuntimeUnavailable, match="missing file"):
        runtime.find_server()


def test_listed_model_downloads_on_first_use(module, tmp_path, file_server, monkeypatch):
    models = module("model_catalog")
    files, base = file_server
    files[models.DEFAULT_MMPROJ] = b"g" * (2 * 1024 * 1024)
    monkeypatch.setenv("QWEN38_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HF_ENDPOINT", base)
    path = models._resolve_file(models.DEFAULT_MMPROJ, "mmproj")
    assert path == tmp_path / models.DEFAULT_MMPROJ and path.stat().st_size == 2 * 1024 * 1024
    assert models.DEFAULT_MMPROJ in models._choices("mmproj")


def test_qwen_runs_through_llama_server_without_llama_cpp(module, tmp_path, monkeypatch):
    catalog = module("model_catalog")
    nodes = module("local_nodes")
    monkeypatch.setenv("QWEN38_MODEL_DIR", str(tmp_path))
    for filename in (catalog.DEFAULT_MODEL, catalog.DEFAULT_MMPROJ):
        with (tmp_path / filename).open("wb") as file:
            file.truncate(2 * 1024 * 1024)
    monkeypatch.setattr(nodes.importlib.util, "find_spec", lambda name: None)
    backend, info = nodes.Qwen38VLLoader().load_model(catalog.DEFAULT_MODEL, catalog.DEFAULT_MMPROJ, 1024, 512, -1, True)
    assert backend.backend_kind == "local_llama_server" and backend.variant == "qwen"
    assert "context_minimum=2048" in info
    backend.configure_chat(4096, "auto")
    assert backend.n_ctx == 4096
    command = backend._server_command(tmp_path / "llama-server", 1234)
    assert command[command.index("--alias") + 1] == "qwen38-27b"
    backend.unload()
    monkeypatch.setenv("TANKNODES_LOCAL_RUNTIME", "llama-cpp-python")
    backend, _ = nodes.Qwen38VLLoader().load_model(catalog.DEFAULT_MODEL, catalog.DEFAULT_MMPROJ, 1024, 512, -1, True)
    assert backend.backend_kind == "local_llama_cpp"
    backend.unload()


def test_qwen_server_auto_reasoning_stays_off_for_images(module):
    bonsai = module("bonsai_backend")
    api = bonsai._BonsaiHTTPBackend("http://127.0.0.1:1/v1", "", "", "qwen38-27b", 30)
    api.auto_visual_effort = None
    api._auto_visual = True
    body = {}
    api._apply_reasoning(body, "auto")
    assert body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_bonsai_projector_mismatch_fails_before_download(module, tmp_path, monkeypatch):
    nodes = module("local_nodes")
    monkeypatch.setenv("QWEN38_MODEL_DIR", str(tmp_path))
    called = []
    monkeypatch.setattr(nodes, "_resolve_file", lambda *a: called.append(a))
    with pytest.raises(nodes.BackendError, match="matching projector"):
        nodes.Qwen38VLLoader().load_model("Ternary-Bonsai-2-27B-PQ2_0.gguf", "mmproj-BF16.gguf", 1024, 512, -1, True)
    assert called == []


def test_cuda_libraries_from_torch_wheels_are_linked(module, tmp_path, monkeypatch):
    runtime = module("runtime_manager")
    home = tmp_path / "rt"
    home.mkdir()
    server = home / "llama-server"
    server.write_text("x")
    (home / "libggml-cuda.so").write_text("x")
    wheel_lib = tmp_path / "nvidia" / "cu13" / "lib"
    wheel_lib.mkdir(parents=True)
    for name in ("libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"):
        (wheel_lib / name).write_text(name)
    rounds = iter([{"libcudart.so.13", "libcublas.so.13"}, {"libcublasLt.so.13"}, set()])
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/ldd")
    monkeypatch.setattr(runtime, "_missing_libraries", lambda files: next(rounds))
    monkeypatch.setattr(runtime, "_cuda_library_dirs", lambda: [wheel_lib])
    assert sorted(runtime.link_cuda_libraries(server)) == ["libcublas.so.13", "libcublasLt.so.13", "libcudart.so.13"]
    assert (home / "libcudart.so.13").resolve() == wheel_lib / "libcudart.so.13"
