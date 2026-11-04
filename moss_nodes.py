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
LANGS = {"english": ("English", "comic, english text"), "korean": ("Korean (한국어)", "comic, korean text"),
         "chinese": ("Simplified Chinese (简体中文)", "comic, chinese text")}
_HANGUL = re.compile(r"[가-힣]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]")


def detect_language(texts) -> str:
    joined = "".join(texts)
    if _HANGUL.search(joined):
        return "korean"
    if _HAN.search(joined) and not _KANA.search(joined):
        return "chinese"
    return "english"


_NO_LINE_START = set("，。！？、；：…）」』】》〉,.!?;:)]}～~·—")


def wrap_cjk(text: str, box_w: float, box_h: float) -> str:
    """Break a Chinese line into rows fitting the box's shape; punctuation never opens a row."""
    t = text.replace("\n", "")
    if len(t) <= 2:
        return t
    per = max(2, math.ceil(math.sqrt(len(t) * max(box_w, 1.0) / max(box_h, 1.0))))
    rows, cur = [], ""
    for ch in t:
        if len(cur) >= per and ch not in _NO_LINE_START:
            rows.append(cur); cur = ""
        cur += ch
    if cur:
        rows.append(cur)
    return "\n".join(rows)


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
                 uppercase_english: bool = True, min_box: int = 8, language: str = "auto",
                 merge: bool = True, cjk_wrap: bool = True) -> tuple[str, list[dict]]:
    """Return (layout JSON, kept items with page-pixel boxes for the preview)."""
    boxes = [(d, _box_of(d), _text_of(d)) for d in detections]
    boxes = [(d, b, t) for d, b, t in boxes if b and t]
    if coord_mode == "auto":
        top = max((max(b) for _, b, _ in boxes), default=0)
        coord_mode = "pixels" if top > 1000 else "0-1000"
    w, h = target_size(page_w, page_h, resolution)
    entries = []
    for d, (x0, y0, x1, y1), t in boxes:
        if coord_mode == "0-1000":
            x0, x1 = (min(max(v, 0.0), 1000.0) * page_w / 1000 for v in (x0, x1))
            y0, y1 = (min(max(v, 0.0), 1000.0) * page_h / 1000 for v in (y0, y1))
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < min_box or y1 - y0 < min_box:
            continue
        entries.append({"box": [x0, y0, x1, y1], "text": t,
                        "source": str(d.get("ja") or d.get("source") or d.get("original") or "")})
    entries = dedupe_entries(entries, page_w, page_h)
    if merge:
        entries = merge_columns(entries)
    items, kept = [], []
    for e in entries:
        t = e["text"]
        if uppercase_english and not _HANGUL.search(t) and not _HAN.search(t):
            t = t.upper()
        x0, y0, x1, y1 = e["box"]
        if cjk_wrap and _HAN.search(t) and not _HANGUL.search(t) and not _KANA.search(t):
            t = wrap_cjk(t, x1 - x0, y1 - y0)
        canvas_box = [x0 * w / page_w, y0 * h / page_h, x1 * w / page_w, y1 * h / page_h]
        items.append({"label": "text", "caption": "text", "text": t, "bbox": centered_box(canvas_box, (w, h))})
        kept.append({"text": t.replace(chr(10), ""), "source": e["source"], "box": e["box"]})
    if language == "auto":
        language = detect_language(i["text"] for i in items)
    layout = {"summary": {"caption": LANGS[language][1]}, "items": items,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":")), kept


SYSTEM_PROMPT = (
    "You are a professional manga translator and letterer. Look at the manga page and find every speech balloon, "
    "thought balloon and narration / caption box that contains Japanese text. Return ONE entry per balloon or box: "
    "a balloon often holds several vertical columns - read all of them (top to bottom, columns right to left) and "
    "translate them together as one line of dialogue. The bbox_2d covers the whole inside of that balloon or box "
    "(where the lettering sits, not the whole panel), as relative coordinates from 0 to 1000 [x1, y1, x2, y2], "
    "x to the right and y downwards, measured on this image. Skip page numbers, watermarks, signatures and sound "
    "effects drawn as artwork outside balloons. List each balloon exactly once; never invent text and never output "
    "empty entries. Translate "
    "naturally and briefly into {lang}, as it would be lettered in a published translation; keep names and tone. "
    "Answer with ONLY a JSON array, no prose: "
    '[{{"bbox_2d": [x1, y1, x2, y2], "ja": "original Japanese", "text": "translation"}}]'
)


def _union(group):
    return [min(e["box"][0] for e in group), min(e["box"][1] for e in group),
            max(e["box"][2] for e in group), max(e["box"][3] for e in group)]


def dedupe_entries(entries: list[dict], page_w: float, page_h: float) -> list[dict]:
    """Drop repeats a VL model sometimes emits: identical entries, and an entry whose Japanese is already part of a
    nearby entry's Japanese (its box is folded into that entry, its translation dropped)."""
    out = []
    for e in entries:
        key = (tuple(round(v) for v in e["box"]), e["text"].strip().lower())
        if any((tuple(round(v) for v in o["box"]), o["text"].strip().lower()) == key for o in out):
            continue
        out.append(e)
    diag = (page_w ** 2 + page_h ** 2) ** 0.5

    def near(a, b):
        ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 < 0.15 * diag

    def core(t):
        return "".join(ch for ch in t if ch.isalnum())

    keep = []
    for i, e in enumerate(out):
        src = core(e["source"])
        host = next((o for j, o in enumerate(out) if j != i and src and len(core(o["source"])) > len(src)
                     and src in core(o["source"]) and near(e["box"], o["box"])), None)
        if host is not None:
            hb = host["box"]
            host["box"] = [min(hb[0], e["box"][0]), min(hb[1], e["box"][1]), max(hb[2], e["box"][2]), max(hb[3], e["box"][3])]
            continue
        keep.append(e)
    return keep


def merge_columns(entries: list[dict]) -> list[dict]:
    """Merge side-by-side narrow vertical boxes of one balloon (a VL model sometimes returns one entry per column).
    entries: {"box": [x0, y0, x1, y1] page px, "text": str, "source": str}. Text joins right-to-left."""
    def tall(b):
        return (b[3] - b[1]) > 1.5 * (b[2] - b[0])

    def same_balloon(a, b):
        if not (tall(a) and tall(b)):
            return False
        gap = max(a[0], b[0]) - min(a[2], b[2])
        ov = min(a[3], b[3]) - max(a[1], b[1])
        return gap <= 1.0 * min(a[2] - a[0], b[2] - b[0]) + 8 and ov >= 0.5 * min(a[3] - a[1], b[3] - b[1])

    groups = [[e] for e in entries]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if same_balloon(_union(groups[i]), _union(groups[j])):
                    groups[i] += groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for g in groups:
        g = sorted(g, key=lambda e: -e["box"][2])          # vertical Japanese: right column first
        out.append({"box": _union(g), "text": " ".join(e["text"] for e in g), "source": "".join(e["source"] for e in g)})
    return out


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
            "merge_columns": ("BOOLEAN", {"default": True, "tooltip": "Merge side-by-side vertical column boxes of one balloon into one item."}),
            "chinese_line_breaks": ("BOOLEAN", {"default": True, "tooltip": "Pre-break Chinese lines to the balloon's shape (the LoRA drops or doubles characters where it breaks Chinese lines itself)."}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("layout_json", "preview", "lines")
    FUNCTION = "convert"
    CATEGORY = "TankNodes/MOSS"
    DESCRIPTION = "Turn a VL model's text-block boxes + translations into the layout-JSON prompt of the MOSS swap_text LoRA."

    def convert(self, image, detections, resolution, coord_mode="auto", uppercase_english=True, min_box=8, merge_columns=True, chinese_line_breaks=True):
        import numpy as np
        import torch
        from PIL import Image, ImageDraw

        page = image[0]
        h, w = int(page.shape[0]), int(page.shape[1])
        layout, kept = build_layout(parse_detections(detections), w, h, resolution, coord_mode, uppercase_english, min_box, merge=merge_columns, cjk_wrap=chinese_line_breaks)
        pil = Image.fromarray((page.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(pil)
        for i, k in enumerate(kept, 1):
            x0, y0, x1, y1 = k["box"]
            draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=max(2, w // 300))
            draw.text((x0 + 3, y0 + 2), str(i), fill=(255, 40, 40))
        preview = torch.from_numpy(np.asarray(pil).astype(np.float32) / 255.0)[None, ...]
        lines = "\n".join(f"{i}. {k['source']} → {k['text']}" for i, k in enumerate(kept, 1)) or "(no text blocks found)"
        return layout, preview, lines
