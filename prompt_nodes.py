"""Prompt Template · Tank  +  Prompt Optimizer · Tank

A two-node, model-agnostic prompt-engineering pair:

* ``TankPromptTemplate`` picks a prompt-engineering template (official H3 format, official Qwen-Image 2.1 PE rules,
  official MOSS enhancer, generic ones) and outputs it as text, so it can be previewed or edited downstream.
* ``TankPromptOptimizer`` sends the template + the user's prompt + whatever media is connected to a Tank backend
  (e.g. Bonsai-2 v2) and returns the optimized prompt.  The task is inferred from the connections:
  first frame -> image-to-video, first + last -> first/last-frame video, reference batch / video -> reference video.
  ``seconds`` is used only when it is > 0 (connected or typed); 0 means "not used".

Templates whose text starts with ``[[tank:h3]]`` are routed to the H3 official-format engine (with format checks and
one correction pass); all others run as a plain system prompt.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from .h3_optimizer import Asset, Job, _complete, _frames_to_urls, optimize
from .reasoning import REASONING_CHOICES, validate_reasoning_choice

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_templates")
H3_MARKER = "[[tank:h3]]"


def _load_index() -> list[dict[str, str]]:
    try:
        with open(os.path.join(_TEMPLATE_DIR, "index.json"), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return [{"label": "通用 · 提示词扩写", "file": ""}]


def _template_text(label: str) -> str:
    for item in _load_index():
        if item["label"] == label:
            if not item.get("file"):
                return item.get("text", "")
            with open(os.path.join(_TEMPLATE_DIR, item["file"]), encoding="utf-8") as handle:
                return handle.read()
    raise ValueError(f"Unknown prompt template: {label}")


class TankPromptTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        labels = [item["label"] for item in _load_index()]
        return {
            "required": {
                "template": (labels, {"default": labels[0], "tooltip": "Prompt-engineering template for the target model/task."}),
            },
            "optional": {
                "extra_rules": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Optional house rules appended to the template (one per line)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("template",)
    FUNCTION = "run"
    CATEGORY = "TankNodes/Prompt"
    DESCRIPTION = "Outputs a prompt-engineering template (official H3 / Qwen-Image 2.1 PE / MOSS, or generic) for Prompt Optimizer · Tank."

    def run(self, template, extra_rules=""):
        text = _template_text(template)
        extra = str(extra_rules or "").strip()
        if extra:
            text = text.rstrip() + "\n\nAdditional rules from the workflow owner:\n" + extra
        return (text,)


def _clean(text: str) -> str:
    out = (text or "").strip()
    if "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    out = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", out).strip()
    # Official Qwen-Image PE templates answer with a JSON object; take its prompt field when present.
    start = out.find("{")
    if start >= 0:
        try:
            data, _ = json.JSONDecoder().raw_decode(out, start)
            if isinstance(data, dict):
                for key in ("rewritten_prompt", "prompt", "caption", "edit_prompt"):
                    if isinstance(data.get(key), str) and data[key].strip():
                        return data[key].strip()
        except ValueError:
            pass
    return out


class TankPromptOptimizer:
    @classmethod
    def VALIDATE_INPUTS(cls, thinking_mode=None):
        return validate_reasoning_choice(thinking_mode)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": ("VISION_LLM_BACKEND", {"tooltip": "Connect Qwen / Bonsai GGUF Loader · Tank (Bonsai-2 v2)."}),
                "template": ("STRING", {"forceInput": True, "tooltip": "Connect Prompt Template · Tank."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Connect your prompt (a multiline text node). Quoted lines are kept verbatim."}),
                "thinking_mode": (list(REASONING_CHOICES), {"default": "high",
                                  "tooltip": "Thinking effort. Kept as chosen: if the answer runs out of tokens the node "
                                             "retries with a bigger budget, never with less thinking."}),
                "unload_after": ("BOOLEAN", {"default": True, "tooltip": "Release the language model afterwards so sampling gets the VRAM."}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.01,
                                      "tooltip": "Target duration. 0 = not used; connect or type a value to use it."}),
                "first_frame": ("IMAGE", {"tooltip": "First frame. Alone: image-to-video; with last frame: first/last-frame video."}),
                "last_frame": ("IMAGE", {"tooltip": "Last frame."}),
                "reference_images": ("IMAGE", {"tooltip": "Reference pictures as one IMAGE batch -> reference task. For image "
                                                         "editing, picture 1 is the image being edited."}),
                "video_frames": ("IMAGE", {"tooltip": "Frames of a source video (sampled to 8) for editing / motion reference."}),
                "video_role": (["auto", "edit source", "motion reference"], {"default": "auto",
                               "tooltip": "auto: editing without reference pictures, motion reference with them."}),
                "max_tokens": ("INT", {"default": 10240, "min": 512, "max": 32768,
                                       "tooltip": "Output budget (thinking + answer)."}),
                "context_length": ("INT", {"default": 32768, "min": 8192, "max": 131072, "step": 1024,
                                           "tooltip": "Context window for the local model (pictures + template + answer)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "run"
    CATEGORY = "TankNodes/Prompt"
    DESCRIPTION = ("Optimizes a prompt with a local Tank model following the connected template. The task is inferred from "
                   "the connected media; seconds is used only when > 0.")

    def run(self, backend, template, prompt, thinking_mode, unload_after, seed, seconds=0.0, first_frame=None,
            last_frame=None, reference_images=None, video_frames=None, max_tokens=10240, video_role="auto",
            context_length=32768):
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise ValueError("Connect Qwen / Bonsai GGUF Loader · Tank to backend.")
        request = str(prompt or "").strip()
        tpl = str(template or "")
        secs = float(seconds or 0.0)
        started = time.perf_counter()
        try:
            budget = int(max_tokens)
            # Never lower the thinking effort (quality drops). If the answer ran out of tokens, give it a bigger budget
            # (and the context to hold it) and try again with the same thinking.
            for attempt in range(3):
                configure = getattr(backend, "configure_chat", None)
                if callable(configure):
                    configure(max(int(context_length), budget + 8192), thinking_mode)
                try:
                    if tpl.lstrip().startswith(H3_MARKER):
                        prompt_out, report = self._h3(backend, request, secs, first_frame, last_frame, reference_images,
                                                      video_frames, thinking_mode, budget, int(seed), video_role)
                    else:
                        prompt_out, report = self._generic(backend, tpl, request, secs, first_frame, last_frame,
                                                           reference_images, video_frames, thinking_mode, budget, int(seed))
                    if attempt:
                        report += "\n" + f"answer needed a bigger budget: max_tokens={budget}"
                    break
                except Exception as error:
                    if attempt == 2 or not re.search(r"budget|final answer|empty", str(error), re.I):
                        raise
                    budget = min(32768, int(budget * 1.6))
        finally:
            if unload_after and callable(getattr(backend, "unload", None)):
                backend.unload()
        report += f"\ntime={time.perf_counter() - started:.1f}s"
        print(f"[TankNodes] prompt optimizer: {report.splitlines()[0]} {report.splitlines()[-1]}")
        if not prompt_out.strip():
            prompt_out, report = request, report + "\nempty answer -> original prompt used"
        return {"ui": {"text": (prompt_out + "\n\n---\n" + report,)}, "result": (prompt_out, report)}

    # -------------------------------------------------------------- H3 official format
    def _h3(self, backend, request, secs, first, last, refs, video, thinking, max_tokens, seed, video_role="auto"):
        assets: list[Asset] = []
        for role, batch in (("first_frame", first), ("last_frame", last)):
            for urls in _frames_to_urls(batch)[:1]:
                assets.append(Asset("picture", role, "", urls))
        for urls in _frames_to_urls(refs):
            assets.append(Asset("picture", "reference", "", urls))
        if video is not None:
            role = {"edit source": "editing", "motion reference": "motion"}.get(video_role,
                                                                              "motion" if refs is not None else "editing")
            assets.append(Asset("video", role, "", [u[0] for u in _frames_to_urls(video, 8)]))
        duration = secs if secs > 0 else 5.0
        result = optimize(backend, Job(request, duration, assets, "auto"), thinking_mode=thinking, max_tokens=max_tokens,
                          seed=seed, correction=True)
        hard = [i for i in result["issues"] if i["hard"]]
        lines = [f"template=H3 mode={result['mode']}", f"seconds={'%.2f' % secs if secs > 0 else 'not used (5.00 assumed)'}",
                 f"corrected={result['corrected']} final_hard_issues={len(hard)}"]
        lines += [f"- {i['code']}: {i['msg']}" for i in result["issues"]]
        return result["prompt"], "\n".join(lines)

    # -------------------------------------------------------------- any other template
    def _generic(self, backend, tpl, request, secs, first, last, refs, video, thinking, max_tokens, seed):
        content: list[dict[str, Any]] = []
        roles: list[str] = []
        n = 0

        def add(label: str, batch, limit=None):
            nonlocal n
            for urls in _frames_to_urls(batch, limit) if batch is not None else []:
                n += 1
                content.append({"type": "text", "text": f"<image{n}> ({label}):"})
                content.append({"type": "image_url", "image_url": {"url": urls[0]}})
                roles.append(label)

        add("first frame", first)
        add("last frame", last)
        add("reference / input picture", refs)
        add("source video frame", video, 8)
        task = ("first/last-frame video" if first is not None and last is not None else "image-to-video" if first is not None
                else "reference" if refs is not None or video is not None else "text-only")
        context = []
        if secs > 0:
            context.append(f"Target duration: {secs:.2f} seconds.")
        if n:
            context.append(f"{n} picture(s) are attached, numbered <image1>…<image{n}> in the order shown.")
        user = (("\n".join(context) + "\n\n") if context else "") + "User request:\n" + request
        content.append({"type": "text", "text": user})
        messages = ([{"role": "system", "content": tpl}] if tpl.strip() else []) + [{"role": "user", "content": content}]
        raw, _usage = _complete(backend, messages, max_tokens=max_tokens, thinking_mode=thinking, seed=seed)
        out = _clean(raw)
        report = "\n".join([f"template=generic task={task}", f"pictures={n}",
                            f"seconds={'%.2f' % secs if secs > 0 else 'not used'}"])
        return out, report


class TankImageBatch:
    """Batch up to 9 optional pictures for Prompt Optimizer · Tank. Missing / bypassed inputs are skipped; with no
    picture at all the output is None (downstream optional inputs just see nothing). Sizes are matched to the first."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {f"image{i}": ("IMAGE",) for i in range(1, 10)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "TankNodes/Prompt"
    DESCRIPTION = "Batches optional pictures (skips empty inputs, outputs nothing when none are connected)."

    def run(self, **kwargs):
        import torch
        import comfy.utils
        frames = []
        for i in range(1, 10):
            img = kwargs.get(f"image{i}")
            if img is None:
                continue
            frames.extend(img[j:j + 1] for j in range(img.shape[0]))
        if not frames:
            return (None,)
        h, w = frames[0].shape[1], frames[0].shape[2]
        out = []
        for f in frames:
            if f.shape[1] != h or f.shape[2] != w:
                f = comfy.utils.common_upscale(f.movedim(-1, 1), w, h, "lanczos", "center").movedim(1, -1)
            out.append(f)
        return (torch.cat(out, dim=0),)
