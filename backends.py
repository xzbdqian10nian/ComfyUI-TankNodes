"""Backend adapters for TankNodes.

The node layer deliberately talks to a very small interface (``complete`` and
``unload``).  This keeps local llama.cpp models and OpenAI-compatible APIs
reusable by the local and direct API node entry points.
"""

from __future__ import annotations

import gc
import json
import os
import re
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from .api_transport import APIResponse
from .reasoning import (
    effective_reasoning_effort,
    is_qwen38_model,
    normalize_reasoning_choice,
    resolve_qwen38_effort,
)


def _free_comfy_vram() -> None:
    """Release ComfyUI models before loading a large local VLM."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:
        print(f"[TankNodes] ComfyUI VRAM cleanup skipped: {exc}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BackendError(RuntimeError):
    """A user-facing backend configuration or request error."""


API_ALLOWED_HOSTS_ENV = "COMFYUI_API_ALLOWED_HOSTS"
# This is deliberately a host allow-list, not a URL prefix check.  The
# administrator can replace this default with exact hosts through
# COMFYUI_API_ALLOWED_HOSTS; a workflow cannot change that process setting.
DEFAULT_API_ALLOWED_HOSTS = frozenset({
    "api.openai.com",
    "localhost",
    "127.0.0.1",
    "::1",
})
API_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalise_allowed_host(value: str) -> str | None:
    """Normalise one administrator-supplied host[:port] allow-list entry."""
    value = value.strip()
    if not value:
        return None
    parsed = urlsplit(value if "://" in value else f"//{value}")
    if parsed.username or parsed.password or parsed.path not in {"", "/"}:
        return None
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not host:
        return None
    host = host.rstrip(".").lower()
    return f"{host}:{port}" if port is not None else host


def _api_allowed_hosts() -> set[str]:
    """Return the server-side allow-list for environment-key API requests."""
    configured = os.getenv(API_ALLOWED_HOSTS_ENV, "")
    configured_hosts = set()
    for value in configured.split(","):
        normalised = _normalise_allowed_host(value)
        if normalised:
            configured_hosts.add(normalised)
    return configured_hosts or set(DEFAULT_API_ALLOWED_HOSTS)


def _validate_env_api_endpoint(base_url: str) -> None:
    """Prevent a workflow from exfiltrating a server-side API key.

    The API environment-variable node is intentionally restricted to an
    administrator-controlled host allow-list.  Direct-key nodes do not call
    this function because their key is supplied by the workflow itself rather
    than read from the ComfyUI server environment.
    """
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise BackendError("Environment-key API URL is invalid.") from exc
    if parsed.username or parsed.password or not host:
        raise BackendError("Environment-key API URL must be complete and must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise BackendError("Environment-key API URL must not contain a query or fragment.")

    scheme = parsed.scheme.lower()
    host = host.rstrip(".").lower()

    if scheme not in {"https", "http"}:
        raise BackendError("Environment-key mode only allows HTTPS; HTTP is allowed for loopback services.")
    is_loopback = host in {"localhost", "127.0.0.1", "::1"}
    if scheme == "http" and not is_loopback:
        raise BackendError("Environment-key mode will not send a key to a non-loopback HTTP address.")

    allowed = _api_allowed_hosts()
    host_match = host in allowed
    port_match = f"{host}:{port}" in allowed if port is not None else False
    default_port = 443 if scheme == "https" else 80
    host_port_match = host_match and (port is None or port == default_port or is_loopback)
    if not host_port_match and not port_match:
        raise BackendError(
            f"Environment-key mode rejected the unallowlisted API host: {host}. "
            f"The default is api.openai.com; set {API_ALLOWED_HOSTS_ENV} before starting ComfyUI to allow exact additional hosts."
        )


@dataclass(frozen=True)
class LocalRuntimeSettings:
    model_path: Path
    mmproj_path: Path
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    free_comfy_vram: bool


_ACTIVE_LOCAL_BACKEND_LOCK = threading.RLock()
_ACTIVE_LOCAL_BACKEND: weakref.ReferenceType | None = None


class LocalQwen38Backend:
    """Qwen3.8 GGUF + image/video projector through CUDA llama.cpp."""

    backend_kind = "local_llama_cpp"

    def __init__(self, settings: LocalRuntimeSettings):
        self.settings = settings
        self.n_ctx = 8192
        self.thinking_mode = "auto"
        self.thinking = False
        self.reasoning_effort = None
        self.reasoning_effort_supported = True
        self.llm = None
        self.chat_handler = None
        self._lock = threading.RLock()
        self.load_seconds = 0.0

    def configure_chat(self, context_length: int, thinking_mode: str) -> None:
        """Apply chat-owned settings before lazy model loading.

        llama.cpp allocates its KV cache while loading, so a context change
        requires a reload. Reasoning effort is a template setting and can be
        updated without reloading model weights.
        """
        n_ctx = max(2048, int(context_length))
        choice, enabled, effort = resolve_qwen38_effort(thinking_mode)
        # Preserve the plugin's previous local default: old workflows using
        # backend_default/auto remain non-thinking until the user picks a tier.
        thinking = bool(enabled) if enabled is not None else False
        with self._lock:
            if self.llm is not None and self.n_ctx != n_ctx:
                print(
                    "[TankNodes] Chat context changed; "
                    "reloading the local model"
                )
                self.unload()
            self.n_ctx = n_ctx
            self.thinking_mode = choice
            self.thinking = thinking
            self.reasoning_effort = effort
            # Reasoning is a chat-template setting, so changing it does not
            # need to reload model weights. Update an existing handler in
            # place for fast switching between effort levels.
            handler = self.chat_handler
            if handler is not None:
                handler.enable_thinking = thinking
                arguments = getattr(handler, "extra_template_arguments", None)
                if isinstance(arguments, dict):
                    arguments["enable_thinking"] = thinking
                    if effort is None:
                        arguments.pop("reasoning_effort", None)
                    else:
                        arguments["reasoning_effort"] = effort

    def _claim_active_slot(self) -> None:
        """Unload another local VLM before this one allocates GPU memory."""
        global _ACTIVE_LOCAL_BACKEND
        with _ACTIVE_LOCAL_BACKEND_LOCK:
            previous = _ACTIVE_LOCAL_BACKEND() if _ACTIVE_LOCAL_BACKEND is not None else None
            if previous is not None and previous is not self:
                print(
                    "[TankNodes] Model selection changed; "
                    "unloading the previous local model first"
                )
                previous.unload()
            _ACTIVE_LOCAL_BACKEND = weakref.ref(self)

    def _make_handler(self):
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter, Qwen35ChatHandler
        except Exception as exc:
            raise BackendError(
                "The installed llama-cpp-python does not provide Qwen35ChatHandler; the local vision model cannot be loaded."
            ) from exc

        handler_kwargs = dict(
            mmproj_path=str(self.settings.mmproj_path),
            enable_thinking=self.thinking,
            preserve_thinking=False,
            add_vision_id=True,
            image_min_tokens=256,
            image_max_tokens=4096,
            batch_max_tokens=1024,
            use_gpu=self.settings.n_gpu_layers != 0,
            verbose=False,
        )
        if self.reasoning_effort is not None:
            handler_kwargs["extra_template_arguments"] = {
                "reasoning_effort": self.reasoning_effort,
            }
        try:
            self.reasoning_effort_supported = True
            handler = Qwen35ChatHandler(**handler_kwargs)
        except TypeError as exc:
            # Older compatible wheels may not expose extra template
            # arguments. Keep on/off thinking functional instead of making
            # the entire local model unloadable.
            if (
                "extra_template_arguments" not in handler_kwargs
                or "extra_template_arguments" not in str(exc)
            ):
                raise
            handler_kwargs.pop("extra_template_arguments", None)
            self.reasoning_effort_supported = False
            print(
                "[TankNodes] This llama-cpp-python build does not "
                f"accept reasoning effort; using thinking on/off only: {exc}",
                flush=True,
            )
            handler = Qwen35ChatHandler(**handler_kwargs)
        # llama-cpp-python's current MTMD formatter builds its own
        # sandboxed Jinja environment but, unlike Jinja2ChatFormatter, some
        # releases omit the standard Hugging Face template helpers. Qwen3.8's
        # thinking template calls raise_exception, so register the two helpers
        # when that environment is exposed. This is harmless on releases that
        # already provide them and keeps the plugin portable across wheels.
        environment = getattr(getattr(handler, "chat_template", None), "environment", None)
        if environment is not None:
            environment.globals.setdefault("raise_exception", Jinja2ChatFormatter.raise_exception)
            environment.globals.setdefault("strftime_now", Jinja2ChatFormatter.strftime_now)
        return handler

    def ensure_loaded(self, progress_callback=None) -> None:
        """Own cleanup for every failed load, including cancellation."""
        with self._lock:
            try:
                self._load_model(progress_callback)
            except BaseException:
                self.unload()
                raise

    def _load_model(self, progress_callback=None) -> None:
        with self._lock:
            if self.llm is not None:
                if callable(progress_callback):
                    progress_callback("ready", 1.0)
                return
            self._claim_active_slot()
            if self.settings.free_comfy_vram:
                if callable(progress_callback):
                    progress_callback("free_vram", 0.05)
                _free_comfy_vram()

            try:
                from llama_cpp import Llama
            except Exception as exc:
                raise BackendError(
                    "A CUDA-enabled llama-cpp-python build is not installed; install a wheel matching the existing ComfyUI runtime."
                ) from exc

            started = time.perf_counter()
            print(
                "[TankNodes] Loading "
                f"{self.settings.model_path.name}, ctx={self.n_ctx}, "
                f"gpu_layers={self.settings.n_gpu_layers}"
            )
            if callable(progress_callback):
                progress_callback("projector", 0.18)
            self.chat_handler = self._make_handler()
            if callable(progress_callback):
                # llama-cpp-python performs this call synchronously and does
                # not expose llama.cpp's weight-loading callback.  Keep the
                # node status on this phase until Llama() returns instead of
                # pretending that a byte-accurate percentage is available.
                progress_callback("weights", 0.25)
            # Llama() is synchronous and the Python binding does not expose a
            # byte-level weight callback.  Emit a low-frequency heartbeat so
            # long loads do not look like a hung ComfyUI process in the log.
            load_heartbeat_done = threading.Event()

            def _load_heartbeat():
                while not load_heartbeat_done.wait(5.0):
                    elapsed = time.perf_counter() - started
                    if callable(progress_callback):
                        # Refresh the same stage bar/status rather than adding
                        # a new log line every five seconds.
                        progress_callback(f"weights_wait:{elapsed:.0f}", 0.25)
                    else:
                        print(
                            "[TankNodes] Loading weights still in progress "
                            f"({elapsed:.0f}s elapsed; llama.cpp has no byte-level callback)",
                            flush=True,
                        )

            heartbeat = threading.Thread(target=_load_heartbeat, name="vision-llm-load-progress", daemon=True)
            heartbeat.start()
            try:
                try:
                    self.llm = Llama(
                        model_path=str(self.settings.model_path),
                        n_ctx=int(self.n_ctx),
                        n_batch=int(self.settings.n_batch),
                        n_ubatch=int(self.settings.n_ubatch),
                        n_gpu_layers=int(self.settings.n_gpu_layers),
                        chat_handler=self.chat_handler,
                        use_mmap=True,
                        use_mlock=False,
                        offload_kqv=True,
                        no_perf=False,
                        verbose=False,
                    )
                except Exception as exc:
                    if self.chat_handler is not None:
                        try:
                            self.chat_handler.close()
                        except Exception:
                            pass
                    self.chat_handler = None
                    gc.collect()
                    memory_hint = ""
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            free_bytes, total_bytes = torch.cuda.mem_get_info()
                            memory_hint = (
                                f" Current CUDA memory: {free_bytes / 2**30:.1f} GiB free / "
                                f"{total_bytes / 2**30:.1f} GiB total."
                            )
                        except Exception:
                            pass
                    raise BackendError(
                        f"Unable to load GGUF model '{self.settings.model_path.name}'."
                        f"{memory_hint} Check that the download is complete, the GGUF is "
                        "supported by the installed llama-cpp-python build, and the selected "
                        "GPU offload/context settings fit in memory."
                    ) from exc
            finally:
                load_heartbeat_done.set()
                heartbeat.join(timeout=1.0)
            self.load_seconds = time.perf_counter() - started
            if callable(progress_callback):
                progress_callback("ready", 1.0)
            print(f"[TankNodes] Local model loaded in {self.load_seconds:.1f}s")

    def complete(self, **kwargs):
        if kwargs.get("stream"):
            return self._stream_completion(kwargs)
        with self._lock:
            # Internal ComfyUI execution metadata must never be forwarded to
            # llama.cpp as a generation option.
            kwargs.pop("_comfy_node_id", None)
            self.ensure_loaded()
            return self.llm.create_chat_completion(**kwargs)

    def _stream_completion(self, kwargs):
        # A llama.cpp stream is lazy. Hold the model lock while consuming it,
        # not only while constructing the generator.
        with self._lock:
            kwargs.pop("_comfy_node_id", None)
            self.ensure_loaded()
            stream = self.llm.create_chat_completion(**kwargs)
            try:
                yield from stream
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

    def info(self) -> str:
        reasoning = self.reasoning_info()
        return (
            f"backend=local_llama_cpp\nmodel={self.settings.model_path.name}\n"
            f"mmproj={self.settings.mmproj_path.name}\n"
            f"gpu_layers={self.settings.n_gpu_layers}\n"
            f"{reasoning}\n"
            f"state={'loaded' if self.llm is not None else 'prepared'}\n"
            f"load_seconds={self.load_seconds:.1f}"
        )

    def reasoning_info(self) -> str:
        effective = self.reasoning_effort or ("off" if not self.thinking else "on")
        if self.thinking and not self.reasoning_effort_supported:
            effective = "on (runtime supports on/off only)"
        return f"reasoning_choice={self.thinking_mode}\nreasoning_effective={effective}"

    def unload(self) -> None:
        global _ACTIVE_LOCAL_BACKEND
        with self._lock:
            if self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
            self.llm = None
            if self.chat_handler is not None:
                try:
                    self.chat_handler.close()
                except Exception:
                    pass
            self.chat_handler = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with _ACTIVE_LOCAL_BACKEND_LOCK:
                active = _ACTIVE_LOCAL_BACKEND() if _ACTIVE_LOCAL_BACKEND is not None else None
                if active is self:
                    _ACTIVE_LOCAL_BACKEND = None
            print("[TankNodes] Local model unloaded")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


def _parse_json_object(value: str, field_name: str) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise BackendError(f"{field_name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BackendError(f"{field_name} must be a JSON object.")
    return parsed


class OpenAICompatibleBackend:
    """Lazy OpenAI-compatible chat-completions client.

    This follows the standard OpenAI-compatible chat-completions contract,
    while keeping the client creation lazy so merely loading ComfyUI never makes a
    network request.
    """

    backend_kind = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_key_env: str,
        model: str,
        timeout: float,
        organization: str = "",
        headers_json: str = "",
        extra_body_json: str = "",
        restrict_endpoint: bool = False,
    ):
        self.base_url = base_url.strip().rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")].rstrip("/")
        self.api_key_env = api_key_env.strip()
        if self.api_key_env and not API_ENV_NAME_RE.fullmatch(self.api_key_env):
            raise BackendError("API key environment-variable name is invalid.")
        if restrict_endpoint:
            _validate_env_api_endpoint(self.base_url)
        self.api_key = api_key.strip() or os.getenv(self.api_key_env, "")
        self.key_source = "direct" if api_key.strip() else (self.api_key_env or "none")
        self.model = model.strip()
        self.timeout = max(1.0, float(timeout))
        self.organization = organization.strip()
        self.headers = _parse_json_object(headers_json, "headers_json")
        self.extra_body = _parse_json_object(extra_body_json, "extra_body_json")
        self.client = None
        self._response = None
        self._lock = threading.RLock()
        self.last_reasoning_choice = "auto"
        self.last_reasoning_effort = None

        if not self.base_url:
            raise BackendError("API base URL cannot be empty.")
        if not self.model:
            raise BackendError("API model cannot be empty.")

    def _ensure_client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import AsyncOpenAI
        except Exception as exc:
            raise BackendError(
                "The openai package is unavailable; using ComfyUI's HTTP client."
            ) from exc

        # Some local OpenAI-compatible servers do not require a key.  The SDK
        # still requires a non-empty value, so use a harmless placeholder.
        kwargs: dict[str, Any] = {
            "api_key": self.api_key or "EMPTY",
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
        if self.organization:
            kwargs["organization"] = self.organization
        if self.headers:
            kwargs["default_headers"] = self.headers
        self.client = AsyncOpenAI(**kwargs)
        return self.client

    def _apply_reasoning(self, extra_body: dict[str, Any], value: str | None) -> None:
        """Translate the unified selector to common Chat Completions shapes."""
        choice = normalize_reasoning_choice(value)
        self.last_reasoning_choice = choice
        self.last_reasoning_effort = None
        if choice == "auto":
            return

        host = (urlsplit(self.base_url).hostname or "").lower()
        if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
            if choice == "off":
                extra_body.setdefault("reasoning", {"enabled": False})
            else:
                extra_body.setdefault("reasoning", {"effort": choice})
                self.last_reasoning_effort = choice
            return

        if is_qwen38_model(self.model):
            effort = effective_reasoning_effort(choice, self.model)
            qwen_cloud = (
                host == "qwencloud.com"
                or host.endswith(".qwencloud.com")
                or host == "dashscope.aliyuncs.com"
                or host.endswith(".dashscope.aliyuncs.com")
            )
            if qwen_cloud:
                extra_body.setdefault("enable_thinking", choice != "off")
            else:
                template_kwargs = extra_body.get("chat_template_kwargs")
                if not isinstance(template_kwargs, dict):
                    template_kwargs = {}
                    extra_body["chat_template_kwargs"] = template_kwargs
                template_kwargs.setdefault("enable_thinking", choice != "off")
            if effort is not None:
                extra_body.setdefault("reasoning_effort", effort)
                self.last_reasoning_effort = effort
            return

        # Flat reasoning_effort is the most common Chat Completions shape for
        # OpenAI, DeepSeek, GLM and compatible gateways. Keep it in extra_body
        # so older OpenAI Python SDK versions still pass it through.
        wire_effort = "none" if choice == "off" else choice
        extra_body.setdefault("reasoning_effort", wire_effort)
        self.last_reasoning_effort = wire_effort

    def complete(self, **kwargs):
        with self._lock:
            request: dict[str, Any] = {
                "model": self.model,
                "messages": kwargs["messages"],
                "stream": bool(kwargs.get("stream", False)),
            }
            # The simple API nodes use 0 as "leave this parameter out" so the
            # provider can apply its own default.  This is also useful for
            # endpoints that reject temperature=0 or max_tokens=0.
            max_tokens = int(kwargs.get("max_tokens", 1024))
            if max_tokens > 0:
                request["max_tokens"] = max_tokens
            temperature = float(kwargs.get("temperature", 0.6))
            if temperature > 0:
                request["temperature"] = temperature
            top_p = float(kwargs.get("top_p", 0.95))
            if top_p > 0:
                request["top_p"] = top_p
            seed = kwargs.get("seed")
            if seed is not None and int(seed) >= 0:
                request["seed"] = int(seed)
            tools = kwargs.get("tools")
            if tools:
                request["tools"] = tools
            extra_body = dict(self.extra_body)
            self._apply_reasoning(extra_body, kwargs.get("thinking_mode", "auto"))

            if extra_body:
                request["extra_body"] = extra_body

            self._response = APIResponse(lambda: self._complete_async(request, extra_body))
            if request["stream"]:
                return self._response
            try:
                return next(self._response)
            finally:
                self._response.close()

    async def _complete_async(self, request, extra_body):
        try:
            client = self._ensure_client()
        except BackendError as exc:
            if "openai package is unavailable" not in str(exc):
                raise
            # ComfyUI already depends on aiohttp. Keep the fallback as one JSON
            # response, but make its connection and body reads cancellable too.
            request["stream"] = False
            yield await self._complete_with_aiohttp(request, extra_body)
            return

        try:
            result = await client.chat.completions.create(**request)
            if request["stream"]:
                async with result:
                    async for chunk in result:
                        yield chunk
            else:
                yield result
        except Exception as exc:
            raise BackendError(f"OpenAI-compatible API request failed: {exc}") from exc
        finally:
            await client.close()
            self.client = None

    async def _complete_with_aiohttp(self, request: dict[str, Any], extra_body: dict[str, Any]):
        """Use ComfyUI's async HTTP dependency when the OpenAI SDK is absent."""
        import aiohttp

        body = dict(request)
        body.pop("extra_body", None)
        body.update(extra_body)
        endpoint = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.timeout, sock_read=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.post(endpoint, json=body, headers=headers) as response:
                    if response.status >= 400:
                        detail = (await response.content.read(1000)).decode("utf-8", errors="replace")
                        raise BackendError(f"OpenAI-compatible API returned HTTP {response.status}: {detail}")
                    return await response.json(content_type=None)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"OpenAI-compatible API request failed: {exc}") from exc

    def info(self) -> str:
        return (
            f"backend=openai_compatible\nbase_url={self.base_url}\n"
            f"model={self.model}\ntimeout={self.timeout:.1f}s\nkey_source={self.key_source}\n"
            f"{self.reasoning_info()}"
        )

    def reasoning_info(self) -> str:
        effective = self.last_reasoning_effort or (
            "provider_default" if self.last_reasoning_choice == "auto" else "off"
        )
        return (
            f"reasoning_choice={self.last_reasoning_choice}\n"
            f"reasoning_effective={effective}"
        )

    def unload(self) -> None:
        with self._lock:
            if self._response is not None:
                self._response.close()
                self._response = None
