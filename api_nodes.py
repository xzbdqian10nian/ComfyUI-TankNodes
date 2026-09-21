"""Direct API chat nodes sharing one implementation for both key sources."""
from __future__ import annotations

import json
import time
from typing import Any
import torch

from .backends import OpenAICompatibleBackend
from .chat import _run_chat
from .progress import ConsoleProgressBar, StatusTicker, make_progress, update_progress
from .reasoning import REASONING_CHOICES, validate_reasoning_choice

API_AUTO_PROGRESS_CHUNKS_PER_PERCENT = 2000

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
                        "tooltip": "Maximum output token count. 0 omits this parameter and uses the provider default.",
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
    OUTPUT_NODE = True

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
        unique_id: str | None = None,
    ):
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            timeout=120.0,
            restrict_endpoint=self.auth_mode == "env",
        )
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
        try:
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
            response, stats = result.response, result.stats
            update_progress(progress, progress_total, progress_total)
            log_progress.finish("response complete")
            ticker.send("API request complete", force=True)
            # Keep the two legacy string-valued fields for existing consumers.
            usage = json.dumps(
                {key: str(result.usage.get(key, "n/a")) for key in ("prompt_tokens", "completion_tokens")},
                ensure_ascii=False,
            )
            elapsed = time.perf_counter() - started
            direct_stats = f"{stats}\napi_elapsed={elapsed:.2f}s"
            return {
                "ui": {"text": (response,)},
                "result": (response, usage, direct_stats, result.reasoning, result.raw_response),
            }
        finally:
            log_progress.close()
            backend.unload()

class VisionAPIEnv(_VisionAPINodeBase):
    auth_mode = "env"

class VisionAPIDirect(_VisionAPINodeBase):
    auth_mode = "direct"
