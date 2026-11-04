"""MOSS-Image swap_text helpers: a local VL model reads + translates a manga page, these nodes turn its answer
into the layout-JSON prompt the MOSS i2i swap_text LoRA was trained on.

Layout format (ComfyUI-MOSS-Toolkit / anime_tools): one ``{"label":"text","caption":"text","text":…,"bbox":[y0,x0,y1,x1]}``
item per line to letter, bbox in ``abs_centralized`` pixels (origin at the centre) of the canvas the i2i nodes generate
at: the page at its own aspect, sides multiples of 16, area capped at ``resolution²`` and never larger than the page.
"""
from __future__ import annotations

import json
import math
import re

STRIDE = 16
AREA_TOLERANCE = 1.10
LANGS = {"english": ("English", "comic, english text"), "korean": ("Korean (한국어)", "comic, korean text")}
_HANGUL = re.compile(r"[가-힣]")


def target_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """Same rule as ComfyUI-MOSS-Toolkit ``nodes.target_size``."""
    cap = resolution * resolution
    if width % STRIDE == 0 and height % STRIDE == 0 and width * height <= cap * AREA_TOLERANCE:
        return width, height
    area = min(cap, width * height)
    aspect = width / height
    out_w = round(math.sqrt(area * aspect) / STRIDE) * STRIDE
    out_h = round(math.sqrt(area / aspect) / STRIDE) * STRIDE
    out_w = max(STRIDE, min(out_w, width // STRIDE * STRIDE))
    out_h = max(STRIDE, min(out_h, height // STRIDE * STRIDE))
    return out_w, out_h


def centered_box(box, size) -> list[int]:
    """``(x0, y0, x1, y1)`` canvas pixels -> ``abs_centralized`` ``yxyx`` (same as anime_tools)."""
    w, h = size
    x0, y0, x1, y1 = (float(v) for v in box)

    def at(v: float, side: int) -> int:
        return round(min(max(v, 0.0), float(side)) - side / 2)

    return [at(y0, h), at(x0, w), at(y1, h), at(x1, w)]


def parse_detections(text: str) -> list[dict]:
    """Tolerant parse of the VL answer: code fences, prose around the JSON, {"items": [...]} or a bare array."""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()
    for candidate in (s, s[s.find("["): s.rfind("]") + 1] if "[" in s else "", s[s.find("{"): s.rfind("}") + 1] if "{" in s else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            data = data.get("items") or data.get("texts") or data.get("blocks") or [data]
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    raise ValueError("Could not find a JSON list of text blocks in the model answer:\n" + s[:400])


def _box_of(d: dict):
    for key in ("bbox_2d", "bbox", "box", "bounding_box"):
        b = d.get(key)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                return [float(v) for v in b]
            except (TypeError, ValueError):
                return None
    return None


def _text_of(d: dict) -> str:
    for key in ("text", "translation", "en", "english", "ko", "korean", "target"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_layout(detections: list[dict], page_w: int, page_h: int, resolution: int, coord_mode: str = "auto",
                 uppercase_english: bool = True, min_box: int = 8, language: str = "auto") -> tuple[str, list[dict]]:
    """Return (layout JSON, kept items with page-pixel boxes for the preview)."""
    boxes = [(d, _box_of(d), _text_of(d)) for d in detections]
    boxes = [(d, b, t) for d, b, t in boxes if b and t]
    if coord_mode == "auto":
        top = max((max(b) for _, b, _ in boxes), default=0)
        coord_mode = "pixels" if top > 1000 else "0-1000"
    w, h = target_size(page_w, page_h, resolution)
    items, kept = [], []
    for d, (x0, y0, x1, y1), t in boxes:
        if coord_mode == "0-1000":
            x0, x1 = x0 * page_w / 1000, x1 * page_w / 1000
            y0, y1 = y0 * page_h / 1000, y1 * page_h / 1000
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < min_box or y1 - y0 < min_box:
            continue
        if uppercase_english and not _HANGUL.search(t):
            t = t.upper()
        canvas_box = [x0 * w / page_w, y0 * h / page_h, x1 * w / page_w, y1 * h / page_h]
        items.append({"label": "text", "caption": "text", "text": t, "bbox": centered_box(canvas_box, (w, h))})
        kept.append({"text": t, "source": str(d.get("ja") or d.get("source") or d.get("original") or ""), "box": [x0, y0, x1, y1]})
    if language == "auto":
        language = "korean" if any(_HANGUL.search(i["text"]) for i in items) else "english"
    layout = {"summary": {"caption": LANGS[language][1]}, "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":")), kept


SYSTEM_PROMPT = (
    "You are a professional manga translator and letterer. You look at one manga page and list every block of "
    "Japanese text a reader would read: speech balloons, thought balloons, narration / caption boxes and signs that "
    "carry dialogue. Skip page numbers, watermarks, signatures, and sound effects drawn as artwork outside balloons. "
    "Read vertical Japanese top-to-bottom, right-to-left. For each block give its bounding box covering the area inside "
    "the balloon or box where the lettering sits (not the whole panel), as relative coordinates from 0 to 1000 "
    "[x1, y1, x2, y2] with x to the right and y downwards. Translate naturally and briefly into {lang}, as it would be "
    "lettered in a published translation; keep names, honorifics and tone; keep it short enough to fit the balloon. "
    "Answer with ONLY a JSON array, no prose, no code fence: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "ja": "original Japanese", "text": "translation"}}]'
)


class TankMangaTranslatePrompt:
    """Prompts for Local Multimodal Chat: find + translate the page's Japanese text blocks as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "language": (list(LANGS), {"default": "english", "tooltip": "Target language of the lettering."}),
            "notes": ("STRING", {"default": "", "multiline": True,
                                 "tooltip": "Optional extra instructions, e.g. character names or a glossary (光太郎 = Kotaro)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "prompt")
    FUNCTION = "build"
    CATEGORY = "TankNodes/MOSS"
    DESCRIPTION = "System + user prompt that makes a local VL model list the page's Japanese text blocks with boxes and translations."

    def build(self, language, notes=""):
        system = SYSTEM_PROMPT.format(lang=LANGS[language][0])
        prompt = f"List and translate every Japanese text block on this manga page into {LANGS[language][0]}."
        if notes and notes.strip():
            prompt += "\nNotes: " + notes.strip()
        return system, prompt


class TankMossSwapTextLayout:
    """VL detections (JSON) + page -> MOSS swap_text layout JSON, plus a preview of the boxes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The manga page (the same image the i2i Reference node gets)."}),
            "detections": ("STRING", {"forceInput": True, "tooltip": "The VL model's JSON answer: [{bbox_2d, ja, text}]."}),
            "resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16,
                                   "tooltip": "Same value as on the MOSS i2i Reference node: output area ≈ resolution²."}),
            "coord_mode": (["auto", "0-1000", "pixels"], {"default": "auto", "tooltip": "auto: boxes above 1000 are pixels, else relative 0-1000."}),
            "uppercase_english": ("BOOLEAN", {"default": True, "tooltip": "Manga lettering is usually upper case."}),
            "min_box": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip": "Drop boxes thinner than this (page pixels)."}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("layout_json", "preview", "lines")
    FUNCTION = "convert"
    CATEGORY = "TankNodes/MOSS"
    DESCRIPTION = "Turn a VL model's text-block boxes + translations into the layout-JSON prompt of the MOSS swap_text LoRA."

    def convert(self, image, detections, resolution, coord_mode="auto", uppercase_english=True, min_box=8):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw

        page = image[0]
        h, w = int(page.shape[0]), int(page.shape[1])
        layout, kept = build_layout(parse_detections(detections), w, h, resolution, coord_mode, uppercase_english, min_box)
        pil = Image.fromarray((page.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(pil)
        for i, k in enumerate(kept, 1):
            x0, y0, x1, y1 = k["box"]
            draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=max(2, w // 300))
            draw.text((x0 + 3, y0 + 2), str(i), fill=(255, 40, 40))
        preview = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)[None, ...]
        lines = "\n".join(f"{i}. {k['source']} → {k['text']}" for i, k in enumerate(kept, 1)) or "(no text blocks found)"
        return layout, preview, lines
