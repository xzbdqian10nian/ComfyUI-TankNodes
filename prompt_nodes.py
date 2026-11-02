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


def _builtin_index() -> list[dict[str, Any]]:
    try:
        with open(os.path.join(_TEMPLATE_DIR, "index.json"), encoding="utf-8") as handle:
            return [dict(item, builtin=True) for item in json.load(handle)]
    except (OSError, ValueError):
        return [{"label": "通用 · 提示词扩写", "file": "", "builtin": True}]


def _user_store_path() -> str:
    try:
        import folder_paths
        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user")
    return os.path.join(base, "default", "tank_prompt_templates.json")


def _user_templates() -> list[dict[str, Any]]:
    try:
        with open(_user_store_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return [dict(t, builtin=False) for t in data.get("templates", []) if t.get("label") and isinstance(t.get("text"), str)]
    except (OSError, ValueError):
        return []


def _save_user_templates(items: list[dict[str, Any]]) -> None:
    path = _user_store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"templates": [{"label": t["label"], "text": t["text"], "updated": t.get("updated", int(time.time()))} for t in items]},
                  handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _load_index() -> list[dict[str, Any]]:
    return _builtin_index() + _user_templates()


def _template_text(label: str) -> str:
    for item in _load_index():
        if item["label"] == label:
            if not item.get("file"):
                return item.get("text", "")
            with open(os.path.join(_TEMPLATE_DIR, item["file"]), encoding="utf-8") as handle:
                text = handle.read()
            if item.get("append") == "qwen21_plain":
                text = text.rstrip() + QWEN21_PLAIN_OUTPUT
            return text
    raise ValueError(f"Unknown prompt template: {label}")


def _register_template_routes() -> None:
    """Template manager for the 提示词模板 node: list / save (create, copy, edit) / delete. Built-in templates are read-only."""
    try:
        from aiohttp import web
        from server import PromptServer
        routes = PromptServer.instance.routes
    except Exception:
        return

    def listing():
        out = []
        for item in _load_index():
            try:
                text = _template_text(item["label"])
            except Exception:
                text = ""
            out.append({"label": item["label"], "builtin": bool(item.get("builtin")), "text": text})
        return out

    @routes.get("/tank/prompt_templates")
    async def _list(request):
        return web.json_response({"templates": listing()})

    @routes.post("/tank/prompt_templates")
    async def _save(request):
        body = await request.json()
        label = str(body.get("label") or "").strip()
        text = str(body.get("text") or "")
        old = str(body.get("old_label") or "").strip()
        if not label or not text.strip():
            return web.json_response({"error": "模板名称和内容都不能为空。"}, status=400)
        if len(label) > 80 or len(text) > 200000:
            return web.json_response({"error": "模板名称或内容太长。"}, status=400)
        if label in {i["label"] for i in _builtin_index()}:
            return web.json_response({"error": "这是内置模板的名字，请换一个名称（内置模板不能被覆盖，可以复制后修改）。"}, status=400)
        items = _user_templates()
        items = [t for t in items if t["label"] not in (label, old)]
        items.append({"label": label, "text": text, "updated": int(time.time())})
        _save_user_templates(items)
        return web.json_response({"saved": label, "templates": listing()})

    @routes.post("/tank/prompt_templates/delete")
    async def _delete(request):
        body = await request.json()
        label = str(body.get("label") or "").strip()
        items = _user_templates()
        if label not in {t["label"] for t in items}:
            return web.json_response({"error": "只能删除自己新建的模板。"}, status=400)
        _save_user_templates([t for t in items if t["label"] != label])
        return web.json_response({"deleted": label, "templates": listing()})


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
    DESCRIPTION = ("The single prompt-template node: official rules for H3 / Qwen-Image 2.1 / FLUX3 / MOSS, generic templates, and your own "
                   "templates (use the 管理模板 button on the node to edit, copy, add or delete). Feeds Prompt Optimizer · Tank or the "
                   "system prompt of Local Multimodal Chat · Tank.")

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
        refs = {f"reference_image_{i}": ("IMAGE", {"tooltip": f"Reference picture {i} (kept at its own size). "
                                                             "Any reference picture makes this a reference task."})
                for i in range(1, 10)}
        return {
            "required": {
                "backend": ("VISION_LLM_BACKEND", {"tooltip": "Connect Qwen / Bonsai GGUF Loader · Tank (Bonsai-2 v2)."}),
                "template": ("STRING", {"forceInput": True, "tooltip": "Connect Prompt Template · Tank."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Connect your prompt (a multiline text node). Quoted lines are kept verbatim."}),
                "thinking_mode": (list(REASONING_CHOICES), {"default": "high",
                                  "tooltip": "Thinking effort. Kept as chosen: if the answer runs out of tokens the node "
                                             "retries with a bigger budget, never with less thinking."}),
                "context_length": ("INT", {"default": 32768, "min": 8192, "max": 131072, "step": 1024, "advanced": True,
                                           "tooltip": "Context window for the local model (pictures + template + answer)."}),
                "max_tokens": ("INT", {"default": 10240, "min": 512, "max": 32768, "step": 16,
                                       "tooltip": "Output budget (thinking + answer)."}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05, "advanced": True,
                                          "tooltip": "Randomness. Lower is steadier, higher is more varied. 0.6 is the tested default."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True,
                                    "tooltip": "Nucleus sampling: only the most likely words covering this share are considered."}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200, "advanced": True,
                                  "tooltip": "Only the K most likely words are considered. 0 = no limit."}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True,
                                    "tooltip": "Drop words whose probability is below this share of the best word."}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01, "advanced": True,
                                             "tooltip": "Above 1 discourages repeating the same words."}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1,
                                 "tooltip": "Language-model seed. Same inputs + same seed = same prompt."}),
                "unload_after": ("BOOLEAN", {"default": True, "tooltip": "Release the language model afterwards so sampling gets the VRAM."}),
            },
            "optional": {
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.01,
                                      "tooltip": "Target duration. 0 = not used; connect or type a value to use it."}),
                "first_frame": ("IMAGE", {"tooltip": "First frame. Alone: image-to-video; with last frame: first/last-frame video."}),
                "last_frame": ("IMAGE", {"tooltip": "Last frame."}),
                **refs,
                "video_frames": ("IMAGE", {"tooltip": "Frames of a source video (sampled to 8) for editing / motion reference."}),
                "video_role": (["auto", "edit source", "motion reference"], {"default": "auto",
                               "tooltip": "auto: editing without reference pictures, motion reference with them."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "report")
    FUNCTION = "run"
    CATEGORY = "TankNodes/Prompt"
    DESCRIPTION = ("Optimizes a prompt with a local Tank model following the connected template. The task is inferred from "
                   "the connected media (first/last frame, up to 9 reference pictures, video); seconds is used only when > 0.")

    def run(self, backend, template, prompt, thinking_mode, context_length=32768, max_tokens=10240, temperature=0.6,
            top_p=0.95, top_k=40, min_p=0.05, repeat_penalty=1.05, seed=1234, unload_after=True, seconds=0.0,
            first_frame=None, last_frame=None, video_frames=None, video_role="auto", **kwargs):
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise ValueError("Connect Qwen / Bonsai GGUF Loader · Tank to backend.")
        refs = [kwargs.get(f"reference_image_{i}") for i in range(1, 10)]
        refs = [r for r in refs if r is not None]
        sampling = {"temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k), "min_p": float(min_p),
                    "repeat_penalty": float(repeat_penalty)}
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
                        prompt_out, report = self._h3(backend, request, secs, first_frame, last_frame, refs, video_frames,
                                                      thinking_mode, budget, int(seed), video_role, sampling)
                    else:
                        prompt_out, report = self._generic(backend, tpl, request, secs, first_frame, last_frame, refs,
                                                           video_frames, thinking_mode, budget, int(seed), sampling)
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
        report += "\n" + f"time={time.perf_counter() - started:.1f}s"
        print(f"[TankNodes] prompt optimizer: {report.splitlines()[0]} {report.splitlines()[-1]}")
        if not prompt_out.strip():
            prompt_out, report = request, report + "\n" + "empty answer -> original prompt used"
        return {"ui": {"text": (prompt_out + "\n\n---\n" + report,)}, "result": (prompt_out, report)}

    # -------------------------------------------------------------- H3 official format
    def _h3(self, backend, request, secs, first, last, refs, video, thinking, max_tokens, seed, video_role, sampling):
        assets: list[Asset] = []
        for role, batch in (("first_frame", first), ("last_frame", last)):
            for urls in _frames_to_urls(batch)[:1]:
                assets.append(Asset("picture", role, "", urls))
        for ref in refs:
            for urls in _frames_to_urls(ref)[:1]:
                assets.append(Asset("picture", "reference", "", urls))
        if video is not None:
            role = {"edit source": "editing", "motion reference": "motion"}.get(video_role, "motion" if refs else "editing")
            assets.append(Asset("video", role, "", [u[0] for u in _frames_to_urls(video, 8)]))
        duration = secs if secs > 0 else 5.0
        result = optimize(backend, Job(request, duration, assets, "auto"), thinking_mode=thinking, max_tokens=max_tokens,
                          seed=seed, correction=True, sampling=sampling)
        hard = [i for i in result["issues"] if i["hard"]]
        secs_text = ("%.2f" % secs) if secs > 0 else "not used (5.00 assumed)"
        lines = [f"template=H3 mode={result['mode']} references={len(refs)}", "seconds=" + secs_text,
                 f"corrected={result['corrected']} final_hard_issues={len(hard)}"]
        lines += [f"- {i['code']}: {i['msg']}" for i in result["issues"]]
        return result["prompt"], "\n".join(lines)

    # -------------------------------------------------------------- any other template
    def _generic(self, backend, tpl, request, secs, first, last, refs, video, thinking, max_tokens, seed, sampling):
        content: list[dict[str, Any]] = []
        n = 0

        def add(label: str, batch, limit=None):
            nonlocal n
            for urls in _frames_to_urls(batch, limit) if batch is not None else []:
                n += 1
                content.append({"type": "text", "text": f"<image{n}> ({label}):"})
                content.append({"type": "image_url", "image_url": {"url": urls[0]}})

        add("first frame", first)
        add("last frame", last)
        for ref in refs:
            add("reference / input picture", ref)
        add("source video frame", video, 8)
        task = ("first/last-frame video" if first is not None and last is not None else "image-to-video" if first is not None
                else "reference" if refs or video is not None else "text-only")
        context = []
        if secs > 0:
            context.append("Target duration: %.2f seconds." % secs)
        if n:
            context.append(f"{n} picture(s) are attached, numbered <image1>…<image{n}> in the order shown.")
        user = (("\n".join(context) + "\n\n") if context else "") + "User request:\n" + request
        content.append({"type": "text", "text": user})
        messages = ([{"role": "system", "content": tpl}] if tpl.strip() else []) + [{"role": "user", "content": content}]
        raw, _usage = _complete(backend, messages, max_tokens=max_tokens, thinking_mode=thinking, seed=seed, sampling=sampling)
        out = _clean(raw)
        secs_text = ("%.2f" % secs) if secs > 0 else "not used"
        report = "\n".join([f"template=generic task={task}", f"pictures={n}", "seconds=" + secs_text])
        return out, report


QWEN21_PLAIN_OUTPUT = (
    "\n\n---\nOutput format for this workflow: reply with the rewritten prompt text only - exactly what you would put in "
    "rewritten_prompt - with no JSON, no field names, no explanation.\n")


class TankQwen21PromptEngineering:
    """Official Qwen-Image 2.1 prompt-enhancer system prompt (T2I or edit), for Local Multimodal Chat · Tank."""
    DEPRECATED = True  # use 提示词模板 · Tank instead; kept so old workflows and published cards still run

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"task": (["text-to-image", "image edit"], {"default": "image edit",
                                 "tooltip": "Which official Qwen-Image 2.1 prompt enhancer to use."})},
                "optional": {"extra_rules": ("STRING", {"multiline": True, "default": "",
                                                        "tooltip": "Optional house rules appended after the official text."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = "TankNodes/Prompt"
    DESCRIPTION = ("Outputs the official Qwen-Image 2.1 prompt-enhancer instructions (text-to-image or edit). Connect it to "
                   "Local Multimodal Chat · Tank as the system prompt; the user prompt and pictures go into the chat node.")

    def run(self, task, extra_rules=""):
        name = "qwen21_t2i_official_pe.txt" if task == "text-to-image" else "qwen21_edit_official_pe.txt"
        with open(os.path.join(_TEMPLATE_DIR, name), encoding="utf-8") as handle:
            text = handle.read().rstrip() + QWEN21_PLAIN_OUTPUT
        extra = str(extra_rules or "").strip()
        return (text + ("\nAdditional rules:\n" + extra if extra else ""),)


class TankMossPromptEngineering:
    """Official MOSS-Image prompt-enhancer system prompt, for Local Multimodal Chat · Tank."""
    DEPRECATED = True  # use 提示词模板 · Tank instead; kept so old workflows and published cards still run

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"extra_rules": ("STRING", {"multiline": True, "default": "",
                                                        "tooltip": "Optional house rules appended after the official text."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = "TankNodes/Prompt"
    DESCRIPTION = ("Outputs the official MOSS-Image prompt-enhancer instructions. Connect it to Local Multimodal Chat · Tank "
                   "as the system prompt; the user prompt (and optional pictures) go into the chat node.")

    def run(self, extra_rules=""):
        with open(os.path.join(_TEMPLATE_DIR, "moss_t2i_official.txt"), encoding="utf-8") as handle:
            text = handle.read().rstrip()
        extra = str(extra_rules or "").strip()
        return (text + ("\n\nAdditional rules:\n" + extra if extra else ""),)


class TankFlux3PromptEngineering:
    """Official FLUX 3 Image Klein prompt-upsampler system prompt (text-to-image / image edit), for Local Multimodal Chat · Tank."""
    DEPRECATED = True  # use 提示词模板 · Tank instead; kept so old workflows and published cards still run

    FILES = {"text-to-image": "flux3_t2i_official_upsampler.txt", "image edit": "flux3_edit_official_upsampler.txt"}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"task": (list(cls.FILES), {"default": "text-to-image",
                                                        "tooltip": "text-to-image or image edit: the two official upsampler prompts."})},
                "optional": {"extra_rules": ("STRING", {"multiline": True, "default": "",
                                                        "tooltip": "Optional house rules appended after the official text."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = "TankNodes/Prompt"
    DESCRIPTION = ("Outputs the official FLUX 3 Image Klein prompt-upsampler instructions. Connect it to Local Multimodal Chat · Tank "
                   "as the system prompt; the user prompt (and reference pictures for editing) go into the chat node.")

    def run(self, task, extra_rules=""):
        with open(os.path.join(_TEMPLATE_DIR, self.FILES[task]), encoding="utf-8") as handle:
            text = handle.read().rstrip()
        extra = str(extra_rules or "").strip()
        return (text + ("\n\nAdditional rules:\n" + extra if extra else ""),)


class TankImageBatch:
    """Batch up to 9 optional pictures for Prompt Optimizer · Tank. Missing / bypassed inputs are skipped; with no
    picture at all the output is None (downstream optional inputs just see nothing). Sizes are matched to the first."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {f"image{i}": ("IMAGE", {"tooltip": f"Picture {i} (optional; skipped when empty or bypassed)."})
                                             for i in range(1, 10)}}

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


class TankOptionalImageDownscale:
    """Resize a picture to at most N megapixels (optionally also enlarge, snap to a multiple). The picture is optional: with nothing
    connected (loader bypassed / not uploaded) the outputs are empty, so optional reference inputs downstream just see nothing."""

    METHODS = ["lanczos", "bicubic", "area", "bilinear", "nearest-exact"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"max_megapixels": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 16.0, "step": 0.05,
                                                          "tooltip": "Target size in megapixels. Larger pictures are shrunk to it."})},
                "optional": {"image": ("IMAGE", {"tooltip": "Optional picture; empty input = no output."}),
                             "upscale_method": (cls.METHODS, {"default": "lanczos", "tooltip": "Resampling filter."}),
                             "multiple_of": ("INT", {"default": 0, "min": 0, "max": 128, "step": 8,
                                                     "tooltip": "Snap width and height to a multiple of this (0 = keep the exact size)."}),
                             "allow_upscale": ("BOOLEAN", {"default": False, "tooltip": "Also enlarge smaller pictures up to the target size."})}}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "run"
    CATEGORY = "TankNodes/Image"
    DESCRIPTION = "Optional picture → resized to about N megapixels (shrink only by default); empty outputs when no picture is connected."

    def run(self, max_megapixels, image=None, upscale_method="lanczos", multiple_of=0, allow_upscale=False):
        import math
        import comfy.utils
        if image is None:
            return (None, 0, 0)
        h, w = image.shape[1], image.shape[2]
        limit = max_megapixels * 1_000_000
        scale = math.sqrt(limit / (h * w))
        if scale >= 1 and not allow_upscale:
            scale = 1.0
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        if multiple_of:
            nw, nh = max(multiple_of, nw // multiple_of * multiple_of), max(multiple_of, nh // multiple_of * multiple_of)
        if (nw, nh) == (w, h):
            return (image, w, h)
        out = comfy.utils.common_upscale(image.movedim(-1, 1), nw, nh, upscale_method, "disabled").movedim(1, -1)
        return (out, nw, nh)
