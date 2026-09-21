"""Local model nodes. Public node IDs and widget order remain compatible."""
from __future__ import annotations

from typing import Any
import torch

from .backends import BackendError, LocalQwen38Backend, LocalRuntimeSettings
from .chat import _run_chat
from .model_catalog import _choices, _resolve_file, selection_info
from .progress import ConsoleProgressBar, StatusTicker, make_progress, send_status, update_progress
from .reasoning import REASONING_CHOICES, validate_reasoning_choice

class Qwen38VLLoader:
    """Load the configured local Qwen3.8 GGUF backend."""
    RETURN_TYPES = ("VISION_LLM_BACKEND", "STRING")
    RETURN_NAMES = ("backend", "backend_info")
    OUTPUT_TOOLTIPS = (
        "Prepared model configuration for Local Multimodal Chat · Tank. Weights load when chat runs.",
        "Resolved model, projector, GPU layers, and backend state.",
    )
    CATEGORY = "TankNodes/Local"
    DESCRIPTION = "Select a local Qwen3.8 GGUF and matching vision projector. Connect to Local Multimodal Chat · Tank; weights load when chat runs."
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        models, projectors = cls._model_choices(), cls._mmproj_choices()
        return {
            "required": {
                "model_file": (
                    models,
                    {
                        "default": models[0],
                        "tooltip": "Main Qwen3.8 GGUF model file. Q4_K_M is a practical starting point for 32 GB VRAM.",
                    },
                ),
                "mmproj_file": (
                    projectors,
                    {
                        "default": projectors[0],
                        "tooltip": "Vision projector GGUF required to understand images and video frames.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {"advanced": True,
                        "default": 1024,
                        "min": 128,
                        "max": 8192,
                        "step": 128,
                        "tooltip": "Number of prompt tokens processed in one llama.cpp batch. Higher can improve prompt processing speed but uses more VRAM.",
                    },
                ),
                "micro_batch_size": (
                    "INT",
                    {"advanced": True,
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Maximum tokens processed in one physical llama.cpp sub-batch. Reduce this first if prompt processing runs out of VRAM.",
                    },
                ),
                "gpu_layers": (
                    "INT",
                    {"advanced": True,
                        "default": -1,
                        "min": -1,
                        "max": 256,
                        "tooltip": "Number of transformer layers offloaded to the GPU. -1 offloads all layers; lower it for partial GPU offload.",
                    },
                ),
                "free_comfy_vram": (
                    "BOOLEAN",
                    {"advanced": True,
                        "default": True,
                        "tooltip": "Unload ComfyUI diffusion models before loading this large local vision-language model.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @staticmethod
    def _model_choices():
        return _choices("model")

    @staticmethod
    def _mmproj_choices():
        return _choices("mmproj")

    def load_model(
        self,
        model_file: str,
        mmproj_file: str,
        batch_size: int,
        micro_batch_size: int,
        gpu_layers: int,
        free_comfy_vram: bool,
        unique_id: str | None = None,
    ):
        send_status("Preparing local model backend…", unique_id)
        model_path = _resolve_file(model_file, "model")
        mmproj_path = _resolve_file(mmproj_file, "mmproj")
        backend = LocalQwen38Backend(
            LocalRuntimeSettings(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_batch=int(batch_size),
                n_ubatch=int(micro_batch_size),
                n_gpu_layers=int(gpu_layers),
                free_comfy_vram=bool(free_comfy_vram),
            )
        )
        send_status(f"Model configuration ready: {model_path.name}", unique_id)
        return backend, f"{backend.info()}\n{selection_info(model_file, mmproj_file)}"

class VisionChat:
    @classmethod
    def VALIDATE_INPUTS(cls, thinking_mode=None):
        return validate_reasoning_choice(thinking_mode)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": ("VISION_LLM_BACKEND", {"tooltip": "Connect Qwen3.8 Model Loader · Tank. The API Chat nodes return text directly and do not connect here."}),
                "system_prompt": (
                    "STRING",
                    {"default": "You are a professional visual-understanding assistant running in ComfyUI. Answer accurately and directly.", "multiline": True, "tooltip": "Persona and system-level behavior or response-format instruction."},
                ),
                "prompt": (
                    "STRING",
                    {"default": "Please describe the input image or video in detail.", "multiline": True, "tooltip": "User instruction sent with the attached image/video content."},
                ),
                "thinking_mode": (list(REASONING_CHOICES), {"default": "auto", "tooltip": "Reasoning control: auto preserves the old/default behavior, off disables thinking, and low/medium/high/xhigh/max are the five unified effort tiers. Local Qwen3.8 natively supports low/medium/xhigh, so high maps to medium and max maps to xhigh."}),
                "context_length": (
                    "INT",
                    {"advanced": True,
                        "default": 8192,
                        "min": 2048,
                        "max": 262144,
                        "step": 1024,
                        "tooltip": "Local llama.cpp context window for the prompt, media tokens, and response. API backends ignore it; changing it reloads a loaded local model.",
                    },
                ),
                "max_tokens": ("INT", {"default": 4096, "min": 16, "max": 32768, "step": 16, "tooltip": "Maximum number of new text tokens to generate. This is only a limit; generation still stops when the model finishes its answer."}),
                "temperature": ("FLOAT", {"advanced": True, "default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Sampling randomness. Lower values are steadier; higher values are more varied. With a local model, 0 uses deterministic greedy sampling when supported."}),
                "top_p": ("FLOAT", {"advanced": True, "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus-sampling probability threshold. Lower values keep only more likely tokens; 0.95 is a strong general default."}),
                "top_k": ("INT", {"advanced": True, "default": 40, "min": 0, "max": 200, "tooltip": "For local llama.cpp, restrict each token choice to the top K candidates. 0 disables top-K filtering; API backends ignore it."}),
                "min_p": ("FLOAT", {"advanced": True, "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "For local llama.cpp, discard tokens whose probability is below this fraction of the best token's probability. API backends ignore it."}),
                "repeat_penalty": ("FLOAT", {"advanced": True, "default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "For local llama.cpp, penalize repeated text. 1.0 disables the penalty; API backends ignore it."}),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 2**32 - 1,
                        "control_after_generate": True,
                        "tooltip": "Random seed used for reproducible sampling when all other inputs and backend settings match. API providers may ignore it.",
                    },
                ),
                "max_video_frames": ("INT", {"advanced": True, "default": 8, "min": 1, "max": 64, "tooltip": "Sample at most this many frames from video or video frames. Every image in the image input is still sent."}),
                "video_transport": (["auto", "frames", "video_url"], {"advanced": True, "default": "auto", "tooltip": "Compatibility option: local models always sample frames. Saved values are retained."}),
                "unload_after": ("BOOLEAN", {"default": False, "tooltip": "Release this backend after generation. Enable to recover VRAM, disable for faster repeated chats."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One still image or an IMAGE batch of still images."}),
                "video_frames": ("IMAGE", {"tooltip": "Decoded video frames as a ComfyUI IMAGE batch; the batch is sampled according to max video frames."}),
                "video": ("VIDEO", {"tooltip": "ComfyUI VIDEO object; local models sample it into frames, while API transport follows video transport."}),
                "tools_json": ("STRING", {"advanced": True, "default": "", "multiline": True, "tooltip": "Optional OpenAI function/tool schema as one JSON object or an array."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "raw_response", "stats")
    OUTPUT_TOOLTIPS = (
        "Final assistant answer without hidden reasoning tags.",
        "Reasoning text when the backend exposes it.",
        "Raw assistant content or complete tool-call message.",
        "Timing, token, speed, media-count, and backend statistics.",
    )
    FUNCTION = "generate"
    CATEGORY = "TankNodes/Local"
    DESCRIPTION = "One local text, image, image-batch or video request. Connect the model loader; no conversation history is stored between runs."
    OUTPUT_NODE = True

    def generate(
        self,
        backend: Any,
        system_prompt: str,
        prompt: str,
        thinking_mode: str,
        context_length: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repeat_penalty: float,
        seed: int,
        max_video_frames: int,
        video_transport: str,
        unload_after: bool,
        image: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        video: Any | None = None,
        tools_json: str | None = None,
        unique_id: str | None = None,
    ):
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise BackendError("Connect Qwen3.8 Model Loader · Tank to the model input.")
        total = max(1, int(max_tokens))
        progress = make_progress(total, unique_id)
        generation_log = ConsoleProgressBar("Local model generation", total)
        ticker = StatusTicker(unique_id)
        update_progress(progress, 0, total)
        ticker.send(f"Preparing image/video input… (max_tokens={total})", force=True)

        try:
            configure_chat = getattr(backend, "configure_chat", None)
            if callable(configure_chat):
                configure_chat(context_length, thinking_mode)

            ensure_loaded = getattr(backend, "ensure_loaded", None)
            if callable(ensure_loaded):
                load_log = ConsoleProgressBar("Local model loading", 100, interval=0.0)
                phase_text = {
                    "free_vram": "Freeing ComfyUI VRAM…",
                    "projector": "Loading vision projector…",
                    "weights": "Loading model weights…",
                    "ready": "Local model ready",
                }

                def report_load(phase: str, fraction: float):
                    phase_name, separator, phase_detail = phase.partition(":")
                    phase_label = phase_text.get(phase_name, phase_name)
                    if separator and phase_name == "weights_wait":
                        phase_label = f"Loading model weights… ({phase_detail}s elapsed)"
                    load_value = round(max(0.0, min(1.0, fraction)) * 100)
                    if phase_name == "ready":
                        load_log.finish("ready")
                    else:
                        load_log.update(
                            load_value,
                            suffix=phase_label,
                            force=True,
                        )
                    ticker.send(
                        f"{phase_label} [{load_value}%]",
                        force=True,
                        mirror_log=False,
                    )

                try:
                    ensure_loaded(progress_callback=report_load)
                finally:
                    load_log.close()

            def report_token(count: int):
                visible = min(max(1, int(count)), total - 1) if total > 1 else 1
                update_progress(progress, visible, total)
                percent = round(100 * visible / total)
                generation_log.update(visible, suffix=f"{count} streamed chunks")
                ticker.send(
                    f"Generating… {count} streamed chunks (token budget ~{percent}%)",
                    mirror_log=False,
                )

            result = _run_chat(
                backend,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
                max_video_frames=max_video_frames,
                video_transport=video_transport,
                thinking_mode=thinking_mode,
                image=image,
                video_frames=video_frames,
                video=video,
                tools_json=tools_json,
                progress_callback=report_token,
            )
            response, reasoning, raw, stats = result.response, result.reasoning, result.raw_response, result.stats
            update_progress(progress, total, total)
            generation_log.finish(stats.splitlines()[0])
            ticker.send(f"Generation complete · {stats.splitlines()[0]}", force=True)
            print(f"[TankNodes] Generation finished: {stats.splitlines()[0]}")
            return {
                "ui": {"text": (response,)},
                "result": (response, reasoning, raw, stats),
            }
        finally:
            generation_log.close()
            if unload_after and callable(getattr(backend, "unload", None)):
                backend.unload()

class VisionUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"backend": ("VISION_LLM_BACKEND", {"tooltip": "Model configuration from the local loader."})},
            "optional": {"after": ("STRING", {"forceInput": True, "tooltip": "Connect the chat response to release the model after that chat completes. Alternatively enable unload after on the chat node."})},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Releasing a cached model is a side effect, so repeat it each queue.
        return float("nan")

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_TOOLTIPS = ("Human-readable unload result.",)
    FUNCTION = "unload"
    CATEGORY = "TankNodes/Local"
    DESCRIPTION = "Release a local model. Connect the chat response to after to establish execution order."
    OUTPUT_NODE = True

    def unload(self, backend: Any, after: str | None = None):
        if callable(getattr(backend, "unload", None)):
            backend.unload()
        status = "Model released · Tank"
        return {"ui": {"text": (status,)}, "result": (status,)}
