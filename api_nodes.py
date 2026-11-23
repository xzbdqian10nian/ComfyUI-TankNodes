"""Direct API chat nodes sharing one implementation for both key sources."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch

from . import h3_optimizer as _h3_optimizer
from .backends import OpenAICompatibleBackend
from .api_transport import check_interrupted
from .chat import _collect_stream, _extract_completion, _run_chat, _split_message
from .h3_image_prompt import (
    _build_messages as _build_h3_image_messages,
    _image_geometry as _h3_image_geometry,
    _parse_result as _parse_h3_image_result,
    _parse_template as _parse_h3_image_template,
    _reduced_ratio as _h3_reduced_ratio,
)
from .media import (
    extract_video_frames,
    image_batch_frames,
)
from .progress import ConsoleProgressBar, StatusTicker, make_progress, update_progress
from .reasoning import REASONING_CHOICES, validate_reasoning_choice

_H3_MARKER = "[[tank:h3]]"
_H3_IMAGE_MARKER_PREFIX = "[[tank:h3-image:"
_JSON_PROMPT_KEYS = ("rewritten_prompt", "expanded_caption", "prompt", "edit_prompt")


def _extract_json_prompt_field(text: str, keys: tuple[str, ...] = _JSON_PROMPT_KEYS) -> str | None:
    """Extract a prompt field when an API model wraps its answer in a JSON object."""
    cleaned = (text or "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned, start)
    except ValueError:
        return None
    if isinstance(data, dict):
        for key in keys:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _complete_direct_messages(
    backend: OpenAICompatibleBackend,
    messages: list[dict[str, Any]],
    *,
    requested_max_tokens: int,
    temperature: float,
    seed: int,
    thinking_mode: str,
    progress_callback,
):
    stream = backend.complete(
        messages=messages,
        max_tokens=int(requested_max_tokens),
        temperature=float(temperature),
        top_p=0.0,
        top_k=0,
        min_p=0.0,
        repeat_penalty=1.0,
        seed=int(seed),
        thinking_mode=thinking_mode,
        stream=callable(progress_callback),
    )
    try:
        raw_result = _collect_stream(stream, progress_callback)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    message, usage = _extract_completion(raw_result)
    response, reasoning, raw = _split_message(message)
    finish_reason = None
    if isinstance(raw_result, dict):
        choices = raw_result.get("choices") or []
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
    return response, reasoning, raw, usage, finish_reason


def _report_analysis_failed(report: str) -> bool:
    for raw_line in str(report or "").splitlines():
        line = raw_line.strip()
        if line == "analysis=failed, generic fallback" or line.startswith("- analysis_failed:"):
            return True
    return False


def _exception_chain_has_secret(exc: BaseException, *secrets: str) -> bool:
    tokens = [str(s or "").strip() for s in secrets]
    tokens = [t for t in tokens if t and t not in _API_PLACEHOLDERS.values()]
    if not tokens:
        return False
    seen: set[int] = set()
    queue: list[BaseException] = [exc]
    while queue:
        cur = queue.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        parts = [
            str(cur),
            *(str(a) for a in getattr(cur, "args", ())),
            *(str(n) for n in (getattr(cur, "__notes__", None) or ())),
            *(str(item) for pair in (getattr(cur, "__dict__", None) or {}).items() for item in pair),
        ]
        if any(t in part for t in tokens for part in parts):
            return True
        if cur.__cause__ is not None:
            queue.append(cur.__cause__)
        if cur.__context__ is not None:
            queue.append(cur.__context__)
        for sub in getattr(cur, "exceptions", None) or ():
            if isinstance(sub, BaseException):
                queue.append(sub)
    return False

API_AUTO_PROGRESS_CHUNKS_PER_PERCENT = 2000
API_BASE_URL_PLACEHOLDER = "{{API_BASE_URL}}"
API_MODEL_PLACEHOLDER = "{{API_MODEL}}"
API_KEY_PLACEHOLDER = "{{API_KEY}}"
_DEFAULT_RUNTIME_API_CONFIG_PATH = "/tmp/tank_runtime_api.json"
RUNTIME_API_CONFIG_PATH = _DEFAULT_RUNTIME_API_CONFIG_PATH
_DEFAULT_MIRROR_API_CONFIG_PATH = str(Path(__file__).resolve().parent / "private" / ".mirror_api")
MIRROR_API_CONFIG_PATH = _DEFAULT_MIRROR_API_CONFIG_PATH
_MIRROR_KEY_SEED = "TankNodes·mirror·v1"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_NO_API_CONFIG_ERROR = "没有可用的 API 配置：请在节点或前端「API 管理」填写自己的 Key（申请地址见说明）（未配置外部 API 参数）"
_API_PLACEHOLDERS = {
    "base_url": API_BASE_URL_PLACEHOLDER,
    "model": API_MODEL_PLACEHOLDER,
    "api_key": API_KEY_PLACEHOLDER,
}


def _redact_api_secret(text: str, *secrets: str) -> str:
    out = str(text or "")
    for secret in secrets:
        token = str(secret or "").strip()
        if token and token not in _API_PLACEHOLDERS.values() and token in out:
            out = out.replace(token, "***")
    return out


def _is_loopback_base_url(base_url: str) -> bool:
    try:
        host = (urlsplit(str(base_url or "").strip()).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


def _normalize_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/")
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".").lower()
    if not scheme or not host:
        return raw.rstrip("/")
    port = parsed.port
    default_port = 80 if scheme == "http" else (443 if scheme == "https" else None)
    host_part = f"[{host}]" if ":" in host else host
    if port is not None and port != default_port:
        host_part = f"{host_part}:{port}"
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username + (f":{parsed.password}" if parsed.password is not None else "") + "@"
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")].rstrip("/")
    suffix = f"?{parsed.query}" if parsed.query else ""
    return f"{scheme}://{userinfo}{host_part}{path}{suffix}"


def _resolve_bound_url(
    raw_url: str,
    configured_url: str,
    *,
    allow_default_openai_when_model_unset: bool = False,
    raw_model: str = "",
) -> str:
    if not raw_url or raw_url == API_BASE_URL_PLACEHOLDER:
        return configured_url
    if (
        allow_default_openai_when_model_unset
        and raw_url == _DEFAULT_OPENAI_BASE_URL
        and (not raw_model or raw_model == API_MODEL_PLACEHOLDER)
    ):
        return configured_url
    if _normalize_base_url(raw_url) == _normalize_base_url(configured_url):
        return configured_url
    raise ValueError("自定义 Base URL 必须同时填写对应的 API Key，不能自动复用已保存或镜像默认密钥。")


def _load_mirror_api_config(mirror_path: str | None) -> dict[str, str] | None:
    if not mirror_path:
        return None
    try:
        raw_line = Path(mirror_path).read_text("utf-8").strip()
        if not raw_line:
            return None
        cipher = base64.b64decode(raw_line.encode("ascii"), validate=True)
        mask = hashlib.sha256(_MIRROR_KEY_SEED.encode("utf-8")).digest()
        plain = bytes(byte ^ mask[idx % len(mask)] for idx, byte in enumerate(cipher))
        data = json.loads(plain.decode("utf-8"))
        if not isinstance(data, dict):
            return None
        resolved = {k: str(data.get(k) or "").strip() for k in ("base_url", "model", "api_key")}
        if all(resolved.values()):
            return resolved
    except Exception:
        return None
    return None


class EmptyAPIResponseError(RuntimeError):
    """The provider answered but the final prompt text is empty."""


def _load_mirror_api_fallbacks(mirror_path: str | None) -> list[dict[str, str]]:
    """Optional `<mirror>_fallbacks` file (same encoding, JSON list): backups tried in order when the
    mirror default fails, times out, or returns an empty prompt."""
    if not mirror_path:
        return []
    try:
        raw_line = Path(str(mirror_path) + "_fallbacks").read_text("utf-8").strip()
        if not raw_line:
            return []
        cipher = base64.b64decode(raw_line.encode("ascii"), validate=True)
        mask = hashlib.sha256(_MIRROR_KEY_SEED.encode("utf-8")).digest()
        data = json.loads(bytes(b ^ mask[i % len(mask)] for i, b in enumerate(cipher)).decode("utf-8"))
    except Exception:
        return []
    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict):
            resolved = {k: str(item.get(k) or "").strip() for k in ("base_url", "model", "api_key")}
            if all(resolved.values()):
                out.append(resolved)
    return out


def _effective_mirror_path(config_path: str | None, mirror_path: str | None) -> str | None:
    if mirror_path is not None:
        return mirror_path
    if MIRROR_API_CONFIG_PATH != _DEFAULT_MIRROR_API_CONFIG_PATH:
        return MIRROR_API_CONFIG_PATH
    effective_runtime = RUNTIME_API_CONFIG_PATH if config_path is None else config_path
    if effective_runtime and effective_runtime != _DEFAULT_RUNTIME_API_CONFIG_PATH:
        return str(Path(effective_runtime).parent / ".mirror_api")
    return MIRROR_API_CONFIG_PATH


def resolve_direct_api_inputs(
    base_url: str,
    model: str,
    api_key: str,
    config_path: str | None = None,
    mirror_path: str | None = None,
    return_source: bool = False,
) -> tuple[str, str, str] | tuple[str, str, str, str]:
    """Three-tier direct API resolution:
    1. Real `api_key` on the node -> 节点填写 (may still resolve {{API_BASE_URL}} / {{API_MODEL}} if used).
    2. Placeholder `{{API_KEY}}` -> `/tmp/tank_runtime_api.json` (前端配置), else mirror default (镜像默认).
    3. Empty `api_key` -> directly mirror default (镜像默认; never reads `/tmp`), except explicit keyless loopback.
    """
    raw = {"base_url": (base_url or "").strip(), "model": (model or "").strip(), "api_key": (api_key or "").strip()}
    resolved_runtime_path = RUNTIME_API_CONFIG_PATH if config_path is None else config_path
    resolved_mirror_path = _effective_mirror_path(config_path, mirror_path)

    def _finish(url_val: str, model_val: str, key_val: str, source_label: str):
        return (url_val, model_val, key_val, source_label) if return_source else (url_val, model_val, key_val)

    def _load_runtime() -> dict[str, Any]:
        if not resolved_runtime_path:
            return {}
        try:
            data = json.loads(Path(resolved_runtime_path).read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    # Tier 1: Node has a real API key filled in.
    if raw["api_key"] and raw["api_key"] != API_KEY_PLACEHOLDER:
        runtime_cfg = _load_runtime() if any(raw[k] == _API_PLACEHOLDERS[k] for k in ("base_url", "model")) else {}
        out_url = str(runtime_cfg.get("base_url") or "").strip() if raw["base_url"] == API_BASE_URL_PLACEHOLDER else raw["base_url"]
        out_model = str(runtime_cfg.get("model") or "").strip() if raw["model"] == API_MODEL_PLACEHOLDER else raw["model"]
        if not out_url or out_url == API_BASE_URL_PLACEHOLDER or not out_model or out_model == API_MODEL_PLACEHOLDER:
            raise ValueError(_NO_API_CONFIG_ERROR)
        return _finish(out_url, out_model, raw["api_key"], "节点填写")

    # Tier 2: Frontend placeholder `{{API_KEY}}` -> try `/tmp/tank_runtime_api.json` first, else mirror default.
    if raw["api_key"] == API_KEY_PLACEHOLDER:
        runtime_cfg = _load_runtime()
        rt_key = str(runtime_cfg.get("api_key") or "").strip()
        if rt_key and rt_key != API_KEY_PLACEHOLDER:
            rt_url = str(runtime_cfg.get("base_url") or "").strip()
            if not rt_url or rt_url == API_BASE_URL_PLACEHOLDER:
                raise ValueError(_NO_API_CONFIG_ERROR)
            out_url = _resolve_bound_url(raw["base_url"], rt_url)
            out_model = (
                str(runtime_cfg.get("model") or "").strip()
                if (not raw["model"] or raw["model"] == API_MODEL_PLACEHOLDER)
                else raw["model"]
            )
            if not out_url or out_url == API_BASE_URL_PLACEHOLDER or not out_model or out_model == API_MODEL_PLACEHOLDER:
                raise ValueError(_NO_API_CONFIG_ERROR)
            return _finish(out_url, out_model, rt_key, "前端配置")

        mirror_cfg = _load_mirror_api_config(resolved_mirror_path)
        if mirror_cfg is None:
            raise ValueError(_NO_API_CONFIG_ERROR)
        out_url = _resolve_bound_url(
            raw["base_url"],
            mirror_cfg["base_url"],
            allow_default_openai_when_model_unset=True,
            raw_model=raw["model"],
        )
        out_model = mirror_cfg["model"] if (not raw["model"] or raw["model"] == API_MODEL_PLACEHOLDER) else raw["model"]
        return _finish(out_url, out_model, mirror_cfg["api_key"], "镜像默认")

    # Tier 3: Empty `api_key` -> never read `/tmp`. Allow explicit local loopback server without key; otherwise mirror default.
    if _is_loopback_base_url(raw["base_url"]) and raw["model"] and raw["model"] != API_MODEL_PLACEHOLDER:
        return _finish(raw["base_url"], raw["model"], "", "节点填写")

    mirror_cfg = _load_mirror_api_config(resolved_mirror_path)
    if mirror_cfg is None:
        raise ValueError(_NO_API_CONFIG_ERROR)
    out_url = _resolve_bound_url(
        raw["base_url"],
        mirror_cfg["base_url"],
        allow_default_openai_when_model_unset=True,
        raw_model=raw["model"],
    )
    out_model = mirror_cfg["model"] if (not raw["model"] or raw["model"] == API_MODEL_PLACEHOLDER) else raw["model"]
    return _finish(out_url, out_model, mirror_cfg["api_key"], "镜像默认")


def _decode_video_for_h3(video: Any) -> torch.Tensor:
    """Decode all frames of a ComfyUI VIDEO input into a BHWC tensor for run_h3."""
    pils = extract_video_frames(video, 10000)
    arrays = [np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 for img in pils]
    stacked = np.stack(arrays, axis=0)
    from_numpy = getattr(torch, "from_numpy", None)
    return from_numpy(stacked) if callable(from_numpy) else torch.Tensor(stacked)

def _api_progress_value(streamed_chunks: int, requested_max_tokens: int) -> tuple[int, int]:
    """Map an API stream count onto a progress value without fake precision."""
    count = max(0, int(streamed_chunks))
    requested = max(0, int(requested_max_tokens))
    if requested > 0:
        # Reserve the final step for a confirmed completed response.
        return min(count, max(0, requested - 1)), requested
    # With a provider-controlled output limit there is no real denominator.
    # Advance slowly enough for long-context responses and stop at 99% until
    # the server emits a normal completion.
    percent = min(count // API_AUTO_PROGRESS_CHUNKS_PER_PERCENT, 99)
    return percent, 100

class _VisionAPINodeBase:
    """Shared implementation for the two deliberately simple API nodes."""

    auth_mode = "env"

    @classmethod
    def VALIDATE_INPUTS(cls, thinking_mode=None):
        return validate_reasoning_choice(thinking_mode)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "base_url": (
                "STRING",
                {
                    "default": "https://api.openai.com/v1",
                    "tooltip": (
                        "OpenAI-compatible API base URL; do not include /chat/completions. "
                        "The environment-key node only permits the administrator's host allow-list."
                        if cls.auth_mode == "env"
                        else "OpenAI-compatible API base URL; do not include /chat/completions."
                    ),
                },
            ),
            "model": (
                "STRING",
                {"default": "", "tooltip": "Exact model ID exposed by the API provider."},
            ),
        }
        if cls.auth_mode == "env":
            required["api_key_env"] = (
                "STRING",
                {
                    "default": "OPENAI_API_KEY",
                    "tooltip": "Environment variable name containing the API key. Its request host must be administrator-allow-listed.",
                },
            )
        else:
            required["api_key"] = (
                "STRING",
                {
                    "default": "",
                    "password": True,
                    "tooltip": "API key used for this request. It is not read from an environment variable.",
                },
            )
        required.update(
            {
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional persona and system-level instruction; leave blank for the provider default.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "User instruction sent to the API, together with any attached image or video input.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32768,
                        "step": 16,
                        "tooltip": "Maximum output token count. 0 = no limit (omits max_tokens and lets the provider decide).",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"advanced": True,
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. 0 means do not send this parameter; use the provider default.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 2**32 - 1,
                        "control_after_generate": True,
                        "tooltip": "Seed sent for sampling when the provider supports it; providers may ignore it.",
                    },
                ),
                "thinking_mode": (
                    list(REASONING_CHOICES),
                    {
                        "default": "auto",
                        "tooltip": "Reasoning control: auto leaves the provider unchanged, off disables reasoning when supported, and low/medium/high/xhigh/max are the five unified effort tiers. Unsupported tiers may be rejected or mapped by the provider.",
                    },
                ),
            }
        )
        return {
            "required": required,
            "optional": {
                "max_video_frames": (
                    "INT",
                    {"advanced": True,
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Sample at most this many frames from video or video frames. Every image in the image input is still sent.",
                    },
                ),
                "video_transport": (
                    ["frames", "video_url", "auto"],
                    {"advanced": True,
                        "default": "frames",
                        "tooltip": "Choose sampled image frames, native video_url, or automatic native-video fallback. The provider must support the selected format.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "One still image or an IMAGE batch; every item is sent in the same API request.",
                    },
                ),
                "video_frames": (
                    "IMAGE",
                    {
                        "tooltip": "Decoded video as an IMAGE batch; frames are sampled according to max_video_frames.",
                    },
                ),
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "ComfyUI VIDEO input; use frames for broad support or video_url only when the provider supports native video.",
                    },
                ),
                "seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 120.0,
                        "step": 0.01,
                        "tooltip": "H3 templates only: target duration in seconds. 0 uses source video length for edits (or 5.0s default).",
                    },
                ),
                "video_role": (
                    ["auto", "edit source", "motion reference", "continuation source"],
                    {
                        "default": "auto",
                        "tooltip": "H3 templates only: how the source video is used. Ignored by non-H3 templates.",
                    },
                ),
                "video_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": "H3 templates only: frame rate of video_frames / video for cut timing and duration.",
                    },
                ),
                "keep_source_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "H3 templates only: preserve the source soundtrack as <Audio 1> fully_copy for edits.",
                    },
                ),
                "source_audio": (
                    "AUDIO",
                    {
                        "tooltip": "H3 templates only: optional source audio used to detect silence; when silent/empty, <Audio 1> fully_copy is omitted.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # Append new outputs; the first three indices are used by old workflows.
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "usage", "stats", "reasoning", "raw_response")
    OUTPUT_TOOLTIPS = (
        "Final API response text.",
        "Usage JSON returned by the provider, when available.",
        "Request timing and backend information.",
        "Reasoning text when the provider returns it; otherwise empty.",
        "Original assistant content or complete tool-call message.",
    )
    FUNCTION = "request"
    CATEGORY = "TankNodes/API"
    DESCRIPTION = "Call an OpenAI-compatible API with a prompt and optional images or video."

    def request(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int = 1,
        thinking_mode: str = "auto",
        api_key_env: str = "",
        api_key: str = "",
        image: torch.Tensor | None = None,
        max_video_frames: int = 8,
        video_transport: str = "frames",
        video_frames: torch.Tensor | None = None,
        video: Any | None = None,
        seconds: float = 0.0,
        video_role: str = "auto",
        video_fps: float = 24.0,
        keep_source_audio: bool = True,
        source_audio: Any = None,
        unique_id: str | None = None,
    ):
        check_interrupted()
        raw_input_key = str(api_key or "").strip()
        api_source = "环境变量"
        if self.auth_mode != "env":
            base_url, model, api_key, api_source = resolve_direct_api_inputs(
                base_url, model, api_key, return_source=True
            )
        attempts = [(base_url, model, api_key)]
        if api_source == "镜像默认":
            attempts += [(f["base_url"], f["model"], f["api_key"])
                         for f in _load_mirror_api_fallbacks(_effective_mirror_path(None, None))]
        kwargs = dict(system_prompt=system_prompt, prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                      seed=seed, thinking_mode=thinking_mode, api_key_env=api_key_env, image=image,
                      max_video_frames=max_video_frames, video_transport=video_transport, video_frames=video_frames,
                      video=video, seconds=seconds, video_role=video_role, video_fps=video_fps,
                      keep_source_audio=keep_source_audio, source_audio=source_audio, unique_id=unique_id,
                      raw_input_key=raw_input_key)
        failures: list[str] = []
        for index, (try_url, try_model, try_key) in enumerate(attempts):
            check_interrupted()
            label = api_source if index == 0 else f"{api_source}·备用{index}"
            try:
                result = self._request_once(try_url, try_model, try_key, label,
                                            timeout=90.0 if len(attempts) > 1 else 120.0, **kwargs)
                if failures:
                    print(f"[TankNodes] API 主用失败，已切换 {label} / {try_model}：" + "；".join(failures))
                return result
            except Exception as exc:
                if type(exc).__name__ in ("InterruptProcessingException", "KeyboardInterrupt"):
                    raise
                msg = _redact_api_secret(str(exc), try_key, raw_input_key)[:200]
                failures.append(f"{try_model}: {msg}")
                print(f"[TankNodes] API 第 {index + 1}/{len(attempts)} 个配置失败（{try_model}）：{msg}")
                if index == len(attempts) - 1:
                    if len(attempts) > 1:
                        raise RuntimeError("API 提示词优化全部失败（已依次尝试 " + str(len(attempts))
                                           + " 个配置）：" + "；".join(failures)) from None
                    raise

    def _request_once(
        self,
        base_url: str,
        model: str,
        api_key: str,
        api_source: str,
        *,
        timeout: float,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        thinking_mode: str,
        api_key_env: str,
        image: torch.Tensor | None,
        max_video_frames: int,
        video_transport: str,
        video_frames: torch.Tensor | None,
        video: Any | None,
        seconds: float,
        video_role: str,
        video_fps: float,
        keep_source_audio: bool,
        source_audio: Any,
        unique_id: str | None,
        raw_input_key: str,
    ):
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            timeout=timeout,
            restrict_endpoint=self.auth_mode == "env",
        )
        secrets = (getattr(backend, "api_key", ""), str(api_key or ""), raw_input_key)
        requested_max_tokens = int(max_tokens)
        # Keep 0 intact in the request so the API applies its own default. The
        # helper then uses a deliberately slow approximate scale because the
        # provider has not exposed a real output denominator.
        _, progress_total = _api_progress_value(0, requested_max_tokens)
        progress = make_progress(progress_total, unique_id)
        log_progress = ConsoleProgressBar("API generation", progress_total)
        ticker = StatusTicker(unique_id)
        update_progress(progress, 0, progress_total)
        ticker.send("Preparing API request…", force=True)

        def report_token(count: int):
            visible, _ = _api_progress_value(count, requested_max_tokens)
            update_progress(progress, visible, progress_total)
            log_progress.update(visible, suffix=f"{count} streamed chunks")
            ticker.send(
                f"API generating… {count} streamed chunks",
                mirror_log=False,
            )

        started = time.perf_counter()
        ui_text: str | None = None
        ui_prompt_meta: list[dict[str, bool]] | None = None
        sanitized_exc: BaseException | None = None
        try:
            stripped_sys = (system_prompt or "").lstrip()
            expanded_sys = _h3_optimizer.expand_template_includes(system_prompt or "").lstrip()
            if stripped_sys.startswith(_H3_IMAGE_MARKER_PREFIX):
                mode, sys_msg, task_msg = _parse_h3_image_template(stripped_sys)
                refs = []
                if mode == "edit":
                    for idx, frm in enumerate(image_batch_frames(image), 1):
                        w, h = _h3_image_geometry(frm)
                        refs.append(
                            {
                                "port": idx,
                                "image": frm,
                                "width": w,
                                "height": h,
                                "aspect_ratio": _h3_reduced_ratio(w, h),
                            }
                        )
                messages = _build_h3_image_messages(mode, sys_msg, task_msg, prompt.strip(), refs, "adaptive")
                reasonings: list[str] = []
                raw_texts: list[str] = []
                total_prompt_tokens = 0
                total_completion_tokens = 0
                has_prompt_tokens = False
                has_completion_tokens = False
                response = ""
                for attempt in range(2):
                    raw_resp, step_reasoning, step_raw_text, step_usage, finish_reason = _complete_direct_messages(
                        backend,
                        messages,
                        requested_max_tokens=requested_max_tokens,
                        temperature=float(temperature),
                        seed=(int(seed) + attempt) % (2**31),
                        thinking_mode=thinking_mode,
                        progress_callback=report_token,
                    )
                    check_interrupted()
                    if step_reasoning:
                        reasonings.append(step_reasoning)
                    raw_texts.append(step_raw_text)
                    if isinstance(step_usage, dict):
                        if isinstance(step_usage.get("prompt_tokens"), int):
                            total_prompt_tokens += step_usage["prompt_tokens"]
                            has_prompt_tokens = True
                        if isinstance(step_usage.get("completion_tokens"), int):
                            total_completion_tokens += step_usage["completion_tokens"]
                            has_completion_tokens = True
                    final_parse_err: ValueError | None = None
                    try:
                        if str(finish_reason or "").strip().lower() == "length":
                            raise ValueError("模型输出达到长度上限被截断（finish_reason=length）。")
                        parsed_img = _parse_h3_image_result(raw_resp, mode, "adaptive")
                        response = parsed_img["expanded_caption"]
                        break
                    except ValueError as exc:
                        if attempt == 1:
                            final_parse_err = ValueError(
                                f"H3 Image PE 返回格式不合格（已调用 {attempt + 1} 次），停止生成，不回退原请求：{exc}"
                            )
                        else:
                            required_format = (
                                "必须包含非空字符串 expanded_caption；aspect_ratio 建议字段可选。"
                                if mode == "t2i"
                                else "恰好包含 expanded_caption 和 target_aspect_ratio 两个字符串字段，正文非空且保持单段。"
                            )
                            fix = (
                                f"上次回答的返回格式不合格：{exc}\n请仅修复格式并返回完整 JSON 对象，"
                                f"{required_format}"
                                "保留原任务和字面文字，正确进行 JSON 转义；不要推理、代码围栏或额外文字。"
                            )
                            messages = messages + [{"role": "assistant", "content": raw_resp}, {"role": "user", "content": fix}]
                    if final_parse_err is not None:
                        raise final_parse_err
                reasoning = "\n\n".join(reasonings)
                raw_text = raw_texts[-1] if raw_texts else ""
                usage_dict = {
                    "prompt_tokens": total_prompt_tokens if has_prompt_tokens else "n/a",
                    "completion_tokens": total_completion_tokens if has_completion_tokens else "n/a",
                }
                elapsed = time.perf_counter() - started
                stats = (
                    f"time={elapsed:.2f}s\nprompt_tokens={usage_dict.get('prompt_tokens', 'n/a')}\n"
                    f"completion_tokens={usage_dict.get('completion_tokens', 'n/a')}\n"
                    f"backend={backend.backend_kind}\nmodel={backend.model}\nh3_image_mode={mode}"
                )
            elif _h3_optimizer.is_h3_template(stripped_sys) or _h3_optimizer.is_h3_template(expanded_sys):
                h3_refs = [frm[None] if getattr(frm, "ndim", 0) == 3 else frm for frm in image_batch_frames(image)]
                is_mv_tpl = _h3_optimizer.is_h3_mv_template(expanded_sys)
                h3_video = (
                    video_frames
                    if video_frames is not None
                    else (_decode_video_for_h3(video) if (video is not None and not is_mv_tpl) else None)
                )
                extra_h3: dict[str, Any] = {}
                if source_audio is not None:
                    extra_h3["source_audio"] = source_audio
                elif video is not None:
                    extra_h3["video"] = video
                backend._h3_usages = []
                backend._h3_reasonings = []
                response, h3_report = _h3_optimizer.run_h3(
                    backend,
                    system_prompt,
                    prompt.strip(),
                    seconds=float(seconds or 0.0),
                    references=h3_refs,
                    video_frames=h3_video,
                    video_role=video_role,
                    video_fps=float(video_fps),
                    keep_source_audio=bool(keep_source_audio),
                    thinking_mode=thinking_mode,
                    max_tokens=requested_max_tokens,
                    seed=int(seed),
                    sampling={"temperature": float(temperature)},
                    **extra_h3,
                )
                check_interrupted()
                reasoning = "\n\n".join(getattr(backend, "_h3_reasonings", []))
                raw_text = response
                usages = getattr(backend, "_h3_usages", []) or []
                last_u = usages[-1] if usages and isinstance(usages[-1], dict) else {}
                usage_dict = {
                    "prompt_tokens": sum(int(u.get("prompt_tokens", 0)) for u in usages if isinstance(u, dict))
                    if usages and any("prompt_tokens" in u for u in usages if isinstance(u, dict))
                    else last_u.get("prompt_tokens", "n/a"),
                    "completion_tokens": sum(int(u.get("completion_tokens", 0)) for u in usages if isinstance(u, dict))
                    if usages and any("completion_tokens" in u for u in usages if isinstance(u, dict))
                    else last_u.get("completion_tokens", "n/a"),
                }
                elapsed = time.perf_counter() - started
                stats = (
                    f"time={elapsed:.2f}s\nprompt_tokens={usage_dict['prompt_tokens']}\n"
                    f"completion_tokens={usage_dict['completion_tokens']}\n"
                    f"backend={backend.backend_kind}\nmodel={backend.model}\n{h3_report}"
                )
                ui_text = f"{response}\n\n---\n{h3_report}" if h3_report else response
                ui_prompt_meta = [{"has_report": True, "analysis_failed": _report_analysis_failed(h3_report)}]
            else:
                result = _run_chat(
                    backend,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=requested_max_tokens,
                    temperature=float(temperature),
                    top_p=0.0,
                    top_k=0,
                    min_p=0.0,
                    repeat_penalty=1.0,
                    seed=int(seed),
                    max_video_frames=max(1, int(max_video_frames)),
                    video_transport=video_transport,
                    thinking_mode=thinking_mode,
                    image=image,
                    video_frames=video_frames,
                    video=video,
                    tools_json=None,
                    progress_callback=report_token,
                )
                check_interrupted()
                response, stats = result.response, result.stats
                reasoning, raw_text, usage_dict = result.reasoning, result.raw_response, result.usage
                if response.lstrip().startswith(("{", "```")):
                    extracted = _extract_json_prompt_field(response)
                    if extracted:
                        response = extracted

            if not str(response or "").strip():
                raise EmptyAPIResponseError(
                    f"API 返回的提示词为空（{getattr(backend, 'model', model)}），不能用空提示词生成视频")
            update_progress(progress, progress_total, progress_total)
            log_progress.finish("response complete")
            ticker.send("API request complete", force=True)
            # Keep the two legacy string-valued fields for existing consumers.
            usage = json.dumps(
                {
                    key: _redact_api_secret(str(usage_dict.get(key, "n/a")), *secrets)
                    for key in ("prompt_tokens", "completion_tokens")
                },
                ensure_ascii=False,
            )
            elapsed = time.perf_counter() - started
            api_line = f"api={api_source} / {getattr(backend, 'model', model)}"
            direct_stats = _redact_api_secret(f"{api_line}\n{stats}\napi_elapsed={elapsed:.2f}s", *secrets)
            response = _redact_api_secret(response, *secrets)
            ui_out = _redact_api_secret(ui_text if ui_text is not None else response, *secrets)
            reasoning = _redact_api_secret(reasoning, *secrets)
            raw_text = _redact_api_secret(raw_text, *secrets)
            return {
                "ui": {
                    "text": (ui_out,),
                    **({"tank_prompt_meta": ui_prompt_meta} if ui_prompt_meta is not None else {}),
                },
                "result": (response, usage, direct_stats, reasoning, raw_text),
            }
        except Exception as exc:
            redacted = _redact_api_secret(str(exc), *secrets)
            if redacted != str(exc) or _exception_chain_has_secret(exc, *secrets):
                try:
                    sanitized_exc = type(exc)(redacted)
                except Exception:
                    sanitized_exc = ValueError(redacted) if isinstance(exc, ValueError) else RuntimeError(redacted)
            else:
                raise
        finally:
            log_progress.close()
            backend.unload()
        if sanitized_exc is not None:
            raise sanitized_exc

class VisionAPIEnv(_VisionAPINodeBase):
    auth_mode = "env"

class VisionAPIDirect(_VisionAPINodeBase):
    auth_mode = "direct"
