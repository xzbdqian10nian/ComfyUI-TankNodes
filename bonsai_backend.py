"""Run PrismML llama-server beside an idle ComfyUI process.

llama-cpp-python cannot load PrismML PQ2_0 weights, and is often not installed
at all. This adapter starts the bundled-on-demand llama-server on loopback for
Bonsai, and for Qwen GGUF when llama-cpp-python is unavailable, then reuses the
plugin's cancellable OpenAI-compatible transport and multimodal chat path.
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .backends import (
    BackendError,
    LocalRuntimeSettings,
    OpenAICompatibleBackend,
    _free_comfy_vram,
    claim_local_backend,
    release_local_backend,
)
from .reasoning import resolve_qwen38_effort
from .runtime_manager import RuntimeUnavailable, ensure_server


MIN_CONTEXT = 16384
QWEN_MIN_CONTEXT = 2048
VARIANTS = {
    # variant: (alias, minimum context, display name, auto effort with images)
    "stock": ("bonsai2-27b-pq2", MIN_CONTEXT, "Bonsai", "low"),
    "abliterated_v2": ("bonsai2-27b-abliterated-v2", MIN_CONTEXT, "Bonsai", "low"),
    "qwen": ("qwen38-27b", QWEN_MIN_CONTEXT, "Qwen", None),
}
STARTUP_TIMEOUT = 180.0


class _BonsaiHTTPBackend(OpenAICompatibleBackend):
    """Pass Qwen-family thinking controls to the Prism chat template."""

    # Bonsai auto thinks at low effort for images; local Qwen auto stays off.
    auto_visual_effort: str | None = "low"

    def complete(self, **kwargs):
        # Short text polishing benefits most from no thinking; images and
        # sampled video frames get low effort so visible details are less
        # likely to be dropped. Explicit user choices always win.
        with self._lock:
            self._auto_visual = any(
                isinstance(part, dict) and part.get("type") in {"image_url", "video_url"}
                for message in kwargs.get("messages", [])
                for part in (
                    message.get("content", [])
                    if isinstance(message, dict) and isinstance(message.get("content"), list)
                    else []
                )
            )
            return super().complete(**kwargs)

    def _apply_reasoning(self, extra_body, value):
        choice, enabled, effort = resolve_qwen38_effort(value)
        self.last_reasoning_choice = choice
        if enabled is None:
            effort = self.auto_visual_effort if getattr(self, "_auto_visual", False) else None
            enabled = effort is not None
        self.last_reasoning_effort = effort if enabled else "off"
        template = extra_body.setdefault("chat_template_kwargs", {})
        template["enable_thinking"] = enabled
        if effort is not None:
            extra_body["reasoning_effort"] = effort


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


def _response_parts(value: Any) -> tuple[bool, int | None]:
    """Return whether a completion/chunk has an answer and its token count."""
    data = _plain(value)
    if not isinstance(data, dict):
        return False, None
    usage = _plain(data.get("usage"))
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    choices = data.get("choices") or []
    if not choices:
        return False, tokens
    choice = _plain(choices[0])
    if not isinstance(choice, dict):
        return False, tokens
    message = _plain(choice.get("delta") or choice.get("message") or {})
    if not isinstance(message, dict):
        return False, tokens
    return bool(message.get("content") or message.get("tool_calls")), tokens


def _retry_seed(seed: Any) -> int:
    return (int(seed) + 41) % (2**32) if seed is not None else 42


class LocalBonsaiBackend:
    backend_kind = "local_bonsai_server"

    def __init__(self, settings: LocalRuntimeSettings, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Bonsai variant: {variant}")
        self.settings = settings
        self.variant = variant
        self.alias, self.min_context, self.label, self.auto_visual_effort = VARIANTS[variant]
        if variant == "qwen":
            self.backend_kind = "local_llama_server"
        self.n_ctx = max(self.min_context, 8192)
        self.thinking_mode = "auto"
        self.load_seconds = 0.0
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._client: _BonsaiHTTPBackend | None = None
        self._log = None
        self._port: int | None = None
        self.empty_response_retries = 0
        atexit.register(self.unload)

    def configure_chat(self, context_length: int, thinking_mode: str) -> None:
        with self._lock:
            requested = max(self.min_context, int(context_length))
            if self._process is not None and self.n_ctx != requested:
                self.unload()
            self.n_ctx = requested
            self.thinking_mode = resolve_qwen38_effort(thinking_mode)[0]

    def _server_command(self, binary: Path, port: int) -> list[str]:
        alias = self.alias
        gpu_layers = 999 if self.settings.n_gpu_layers < 0 else self.settings.n_gpu_layers
        args = [
            str(binary), "--model", str(self.settings.model_path),
            "--mmproj", str(self.settings.mmproj_path),
            "--n-gpu-layers", str(gpu_layers),
            "--ctx-size", str(self.n_ctx), "--parallel", "1",
            "--batch-size", str(self.settings.n_batch),
            "--ubatch-size", str(self.settings.n_ubatch),
            "--flash-attn", "on", "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0", "--host", "127.0.0.1",
            "--port", str(port), "--alias", alias, "--no-ui",
            "--reasoning-effort", "medium", "--temp", "1.0",
            "--top-p", "0.95", "--top-k", "20", "--min-p", "0.05",
            "--repeat-penalty", "1.0", "--presence-penalty", "0.0",
        ]
        if self.n_ctx > 262144:
            args.extend([
                "--rope-scaling", "yarn", "--rope-scale", str(self.n_ctx / 262144),
                "--yarn-orig-ctx", "262144",
                "--override-kv", f"qwen35.context_length=int:{self.n_ctx}",
            ])
        # MTP2 was slower on the measured Chinese creative tasks. Keep it off
        # for prompt optimization and image/video reverse prompting.
        return args

    def ensure_loaded(self, progress_callback=None) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                if callable(progress_callback):
                    progress_callback("ready", 1.0)
                return
            if self._process is not None:
                self.unload()
            if callable(progress_callback):
                progress_callback("runtime", 0.02)
            try:
                binary = ensure_server()
            except (RuntimeUnavailable, OSError) as error:
                raise BackendError(f"PrismML llama-server is unavailable: {error}") from None
            if not os.access(binary, os.X_OK):
                raise BackendError(f"PrismML llama-server is not executable: {binary}")
            for path in (self.settings.model_path, self.settings.mmproj_path):
                if not path.is_file():
                    raise BackendError(f"{self.label} model file is missing: {path}")
            claim_local_backend(self)
            try:
                if self.settings.free_comfy_vram:
                    if callable(progress_callback):
                        progress_callback("free_vram", 0.05)
                    _free_comfy_vram()
                self._port = _free_loopback_port()
                self._log = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
                command = self._server_command(binary, self._port)
                started = time.monotonic()
                if callable(progress_callback):
                    progress_callback("weights", 0.25)
                self._process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name("bonsai_exec.py")),
                     str(os.getpid()), *command],
                    stdin=subprocess.DEVNULL, stdout=self._log,
                    stderr=subprocess.STDOUT, close_fds=True,
                )
                health_url = f"http://127.0.0.1:{self._port}/health"
                while time.monotonic() - started < STARTUP_TIMEOUT:
                    if self._process.poll() is not None:
                        break
                    try:
                        with urlopen(health_url, timeout=2) as response:
                            if response.status == 200:
                                self.load_seconds = time.monotonic() - started
                                client = _BonsaiHTTPBackend(
                                    base_url=f"http://127.0.0.1:{self._port}/v1",
                                    api_key="", api_key_env="", model=command[command.index("--alias") + 1],
                                    timeout=300.0,
                                    extra_body_json='{"stream_options":{"include_usage":true}}',
                                )
                                client.auto_visual_effort = self.auto_visual_effort
                                self._client = client
                                if callable(progress_callback):
                                    progress_callback("ready", 1.0)
                                return
                    except (OSError, HTTPError, URLError):
                        pass
                    if callable(progress_callback):
                        progress_callback(f"weights_wait:{time.monotonic() - started:.0f}", 0.25)
                    time.sleep(0.5)
                detail = ""
                if self._log is not None:
                    self._log.flush()
                    self._log.seek(0)
                    detail = self._log.read()[-1200:].strip()
                raise BackendError(f"PrismML llama-server ({self.label}) did not become ready. {detail}")
            except BaseException:
                self.unload()
                raise

    def complete(self, **kwargs):
        if kwargs.get("stream"):
            return self._stream_completion(kwargs)
        with self._lock:
            self.ensure_loaded()
            assert self._client is not None
            self.empty_response_retries = 0
            result = self._client.complete(**kwargs)
            answered, tokens = _response_parts(result)
            if answered:
                return result
            if tokens is not None and tokens > 1:
                raise BackendError(f"{self.label} used the output budget without a final answer. Increase max_tokens or lower thinking effort.")
            retry = dict(kwargs, seed=_retry_seed(kwargs.get("seed")))
            self.empty_response_retries = 1
            result = self._client.complete(**retry)
            answered, _ = _response_parts(result)
            if not answered:
                raise BackendError(f"{self.label} returned no answer after one retry. Try another seed or model.")
            return result

    def _stream_completion(self, kwargs):
        with self._lock:
            self.ensure_loaded()
            assert self._client is not None
            self.empty_response_retries = 0
            for attempt in range(2):
                request = kwargs if attempt == 0 else dict(kwargs, seed=_retry_seed(kwargs.get("seed")))
                if attempt:
                    self.empty_response_retries = 1
                    print(f"[TankNodes] {self.label} returned an empty response; retrying once with another seed")
                stream = self._client.complete(**request)
                answered = False
                completion_tokens = None
                try:
                    for chunk in stream:
                        has_answer, tokens = _response_parts(chunk)
                        answered = answered or has_answer
                        if tokens is not None:
                            completion_tokens = tokens
                        yield chunk
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if answered:
                    return
                if completion_tokens is not None and completion_tokens > 1:
                    raise BackendError(f"{self.label} used the output budget without a final answer. Increase max_tokens or lower thinking effort.")
            raise BackendError(f"{self.label} returned no answer after one retry. Try another seed or model.")

    def reasoning_info(self) -> str:
        if self._client is not None:
            return f"{self._client.reasoning_info()}\nempty_response_retries={self.empty_response_retries}"
        return f"reasoning_choice={self.thinking_mode}\nreasoning_effective=server_default_medium"

    def info(self) -> str:
        return (
            f"backend={self.backend_kind}\nvariant={self.variant}\n"
            f"model={self.settings.model_path.name}\nmmproj={self.settings.mmproj_path.name}\n"
            f"runtime=PrismML llama-server\ncontext_minimum={self.min_context}\n"
            "flash_attention=on\nkv_cache=q8_0\nmtp=off\n"
            f"state={'loaded' if self._process is not None else 'prepared'}\n"
            f"load_seconds={self.load_seconds:.1f}"
        )

    def unload(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.unload()
                self._client = None
            process = self._process
            self._process = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if self._log is not None:
                self._log.close()
                self._log = None
            self._port = None
            release_local_backend(self)
