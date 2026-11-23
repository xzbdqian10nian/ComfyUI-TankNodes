"""H3 Prompt Optimizer · Tank

Turns a plain (often Chinese) request plus optional keyframes / reference pictures / source-video
frames into a MiniMax H3 prompt that follows the official H3 prompt-writing guide
(T2VA / I2VA / FL2VA / L2VA and full-reference Ref2VA), using a local Tank backend
(e.g. Ternary Bonsai-2 v2). Deterministic parts (alignment instruction, shot numbering in it,
timecode format, label hygiene) are produced in code; the model writes the creative body.
A validator checks the official contract and, if needed, asks the model for one correction
that is only accepted when it reduces the hard issues.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image, ImageDraw

from .backends import BackendError
from .chat import _collect_stream, _extract_completion, _split_message
from .media import image_batch_frames, pil_to_data_url, sample_image_batch, tensor_frame_to_pil
from .reasoning import REASONING_CHOICES, validate_reasoning_choice

_SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_skill")
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_templates")
H3_MARKER = "[[tank:h3]]"
H3_EDIT_MARKER = "[[tank:h3-edit]]"
H3_MV_MARKER = "[[tank:h3-mv]]"
H3_DH_MARKER = "[[tank:h3-dh]]"   # 数字人专用模板：单镜头固定机位；接了音频就对口型
_INCLUDE_DIRECTIVE = re.compile(r"^\s*\[\[include:([A-Za-z0-9_.-]+)\]\]\s*$", re.I)
_H3_TEMPLATE_LINE_RE = re.compile(r"(?m)^\s*\[\[tank:h3(?:-(?:edit|mv))?\]\]\s*$", re.I)
_H3_MV_LINE_RE = re.compile(r"(?m)^\s*\[\[tank:h3-mv\]\]\s*$", re.I)


class OutputTruncatedError(BackendError):
    """Raised when the model stops due to token budget exhaustion before finishing the answer."""


def expand_template_includes(text: str, _seen: frozenset[str] = frozenset()) -> str:
    out_lines: list[str] = []
    for line in str(text or "").splitlines():
        match = _INCLUDE_DIRECTIVE.fullmatch(line)
        if not match:
            out_lines.append(line)
            continue
        inc_name = os.path.basename(match.group(1))
        if inc_name in _seen:
            continue
        inc_path = os.path.join(_TEMPLATE_DIR, inc_name)
        if os.path.isfile(inc_path):
            with open(inc_path, encoding="utf-8") as handle:
                out_lines.append(expand_template_includes(handle.read().rstrip("\n"), _seen | {inc_name}))
    return "\n".join(out_lines)


def h3_edit_template_options(template: str) -> tuple[str, str]:
    """Return the last render directive and handbook text without engine-only marker lines."""
    expanded = expand_template_includes(template)
    lines = str(expanded or "").splitlines()
    directives = []
    for line in lines:
        match = re.fullmatch(r"\s*\[\[render:(code|llm)\]\]\s*", line, flags=re.I)
        if match:
            directives.append(match.group(1).lower())
    render = directives[-1] if directives else "llm"
    kept = []
    for line in lines:
        if re.fullmatch(r"\s*\[\[tank:h3(?:-edit)?\]\]\s*", line, flags=re.I):
            continue
        if re.fullmatch(r"\s*\[\[render:(?:code|llm)\]\]\s*", line, flags=re.I):
            continue
        if re.fullmatch(r"\s*\[\[style:(?:picture|video)\]\]\s*", line, flags=re.I):
            continue
        kept.append(line)
    return render, "\n".join(kept).strip()


def is_h3_template(template: str) -> bool:
    """Return True when the template (before or after include expansion) carries an H3 engine marker."""
    raw = str(template or "")
    if raw.lstrip().startswith((H3_MARKER, H3_EDIT_MARKER, H3_MV_MARKER)):
        return True
    expanded = expand_template_includes(raw)
    return bool(
        expanded.lstrip().startswith((H3_MARKER, H3_EDIT_MARKER, H3_MV_MARKER))
        or _H3_TEMPLATE_LINE_RE.search(expanded)
    )


def is_h3_mv_template(template: str) -> bool:
    """Return True when the template contains a standalone [[tank:h3-mv]] marker line."""
    expanded = expand_template_includes(str(template or ""))
    return bool(_H3_MV_LINE_RE.search(expanded))


def _strip_h3_mv_markers(text: str) -> str:
    kept = [
        line for line in str(text or "").splitlines()
        if not re.fullmatch(r"\s*\[\[tank:h3(?:-mv)?\]\]\s*", line, flags=re.I)
    ]
    return "\n".join(kept).strip()


def h3_mv_template_handbook(template: str) -> str:
    """Return the MV handbook text without engine marker lines, falling back to prompt_templates/h3_mv.txt."""
    expanded = expand_template_includes(str(template or ""))
    body = _strip_h3_mv_markers(expanded)
    if body:
        return body
    mv_path = os.path.join(_TEMPLATE_DIR, "h3_mv.txt")
    if os.path.isfile(mv_path):
        with open(mv_path, encoding="utf-8") as handle:
            return _strip_h3_mv_markers(expand_template_includes(handle.read()))
    return ""


def _guide(name: str) -> str:
    with open(os.path.join(_SKILL_DIR, name), encoding="utf-8") as handle:
        return handle.read()


BASE_SECTIONS = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]
REF_SECTIONS = ["subject_definitions", "summary", "retention_analysis", "detailed_description",
                "overall_soundscape", "non_diegetic_music"]
MODES = ("T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA")
VIDEO_ROLES = {
    "editing source / 编辑源视频": "editing",
    "continuation source / 续拍源视频": "continuation",
    "motion & camera reference / 动作运镜参考": "motion",
    "auto / 按需求文字判断": "auto",
}
# 视频用途无法只凭「接了什么素材」判断（官方指南：有视频不等于编辑）。auto 按需求文字判断，判断不出再按素材兜底。
_CONT_WORDS = re.compile(r"续拍|接着拍|接着往下|往后续|延续|继续生成|continu|extend", re.I)
_EDIT_WORDS = re.compile(r"替换|换成|换掉|换人|换脸|换头|换装|换衣|改成|改为|变成|去掉|删掉|删除|移除|抹掉|"
                         r"replace|swap|turn\s+\S+(?:\s+\S+){0,3}\s+into|change\s+.{1,40}?\s+(?:into|to)\b|remove|erase|\bedit", re.I)
_MOTION_WORDS = re.compile(r"参考.{0,6}(?:动作|运镜|镜头|节奏|走位)|照着.{0,8}(?:跳|动作|做|演)|模仿|跟着.{0,6}(?:跳|做|动|演)|"
                           r"(?:跳|学|做|演|复刻|按|照)[^，。,.;；]{0,8}(?:视频|原片|这段|这支|这个)[^，。,.;；]{0,6}(?:舞|动作|姿势|动)|"
                           r"动作参考|运镜参考|motion reference|camera reference|follow(?:s|ing)? the (?:motion|camera|choreography)|"
                           r"(?:dance|move)s?\s+like\s+the\s+video", re.I)
# 明确的动作迁移：说了「动作迁移」，或要保留照片自己的背景（编辑原视频时背景只能来自视频，所以这只能是动作迁移）
_STRONG_MOTION_WORDS = re.compile(r"动作迁移|姿势迁移|动作复刻|motion\s+transfer|pose\s+transfer|"
                                  r"(?:保持|保留|沿用)[^，。,.;；]{0,6}(?:原图|照片|图片|参考图|图\s*[1-9]|人物图)[^，。,.;；]{0,6}的?(?:背景|场景|环境)|"
                                  r"keep\s+(?:the\s+)?(?:photo|picture|image)'?s?\s+(?:own\s+)?(?:background|scene)", re.I)


_OUTFIT_CLAUSE_RE = re.compile(r"衣服|服装|穿|裙|裤|鞋|换装|换衣|道具|手里|\b(?:outfit|clothes|clothing|dress|shoes|prop)\b", re.I)


def _role_by_words(text: str, *, continuation: bool = False) -> tuple[str, str] | None:
    """明确动作迁移 > 编辑措辞 > 「跳/学/做 + 视频里的动作」。否定的说法不算；只改背景的分句不算编辑；
    已经说了跟视频动作时，只改衣服/道具的分句也不算编辑（那是动作迁移里顺带换装）。"""
    text = str(text or "")
    if continuation:
        hit = affirmed_hit(_CONT_WORDS, text)
        if hit:
            return "continuation", f"request says '{hit.group(0)}'"
    hit = affirmed_hit(_STRONG_MOTION_WORDS, text)
    if hit:
        return "motion", f"request says '{hit.group(0)}'"
    motion = affirmed_hit(_MOTION_WORDS, text)
    clauses = [c for c in _clauses(text) if not _setting_only_clause(c)
               and not (motion and _OUTFIT_CLAUSE_RE.search(c) and not _REQ_PERSON_NOUN_RE.search(c))]
    hit = affirmed_hit(_EDIT_WORDS, "，".join(clauses))
    if hit:
        return "editing", f"request says '{hit.group(0)}'"
    if motion:
        return "motion", f"request says '{motion.group(0)}'"
    return None


def infer_edit_role(request: str) -> tuple[str, str]:
    """视频编辑模板的 auto 判断。"""
    return _role_by_words(request) or ("editing", "default edit preset")


STYLE_SOURCES = {"video", "picture"}
_STYLE_WORDS_CN = r"(?:绘画风格|美术风格|渲染风格|画风|风格|插画风(?:格)?|二次元风(?:格)?|3d\s*风(?:格)?|真人风(?:格)?)"
_STYLE_EXPLICIT_CN = r"(?:绘画风格|美术风格|渲染风格|画风|插画风(?:格)?|二次元风(?:格)?|3d\s*风(?:格)?|真人风(?:格)?)"
_STYLE_PICTURE_WORDS = re.compile(
    rf"(?:保持|保留|沿用|按(?:照)?|用)\s*(?:参考)?\s*图(?:片|像|\s*\d+)?\s*(?:里|中)?\s*(?:的)?\s*{_STYLE_WORDS_CN}|"
    rf"(?:保持|保留|沿用)\s*{_STYLE_EXPLICIT_CN}|"
    r"(?:keep|preserve|use)\s+(?:the\s+)?(?:reference\s+)?picture(?:'s)?\s+(?:art\s+|visual\s+|rendering\s+|drawing\s+)?style",
    re.I,
)
_STYLE_LOOK_TRAP = re.compile(r"保持(?:参考)?图(?:片|像)?(?:的)?样子|按图(?:片|像)?(?:的)?样子", re.I)
_STYLE_DIRECTIVE = re.compile(r"^\s*\[\[style:(picture|video)\]\]\s*$", re.I)


def infer_style_source(request: str) -> str:
    """Return the user-request style source from explicit style keywords (excluding '样子')."""
    return "picture" if affirmed_hit(_STYLE_PICTURE_WORDS, str(request or "")) else "video"


def resolve_style_source(request: str, template: str = "", analysis_style_source: str | None = None) -> str:
    """模板指令 > 模型分析给的 style_source > 原话里明确的画风说法 > 跟原片。"""
    directives = []
    for line in str(template or "").splitlines():
        match = _STYLE_DIRECTIVE.fullmatch(line)
        if match:
            directives.append(match.group(1).lower())
    if directives:
        return directives[-1]
    if isinstance(analysis_style_source, str) and analysis_style_source.lower() in STYLE_SOURCES:
        return analysis_style_source.lower()
    if infer_style_source(request) == "picture" and not _STYLE_LOOK_TRAP.search(str(request or "")):
        return "picture"
    return "video"


def infer_video_role(request: str, has_refs: bool) -> tuple[str, str]:
    """返回 (role, 理由)。优先级：续拍 > 明确动作迁移 > 编辑 > 动作参考 > 按素材兜底。"""
    text = str(request or "")
    found = None
    motion = affirmed_hit(_MOTION_WORDS, text)
    edit_text = "，".join(c for c in _clauses(text)
                         if not (motion and _OUTFIT_CLAUSE_RE.search(c) and not _REQ_PERSON_NOUN_RE.search(c)
                                 and not re.search(r"原视频|视频里|视频中|原片|\bvideo'?s\b|\boriginal\b", c, re.I)))
    for role, pattern, where in (("continuation", _CONT_WORDS, text), ("motion", _STRONG_MOTION_WORDS, text),
                                 ("editing", _EDIT_WORDS, edit_text), ("motion", _MOTION_WORDS, text)):
        hit = affirmed_hit(pattern, where)
        if hit:
            found = (role, f"request says '{hit.group(0)}'")
            break
    if found:
        return found
    return ("motion", "no keyword; reference pictures attached") if has_refs else ("editing", "no keyword; no reference pictures")


def detect_cuts(batch: torch.Tensor | None, fps: float, min_shot: float = 0.4) -> list[float]:
    """在源视频帧里找硬切（灰度帧差的突跃），返回切点秒数。"""
    if batch is None or batch.ndim != 4 or int(batch.shape[0]) < 3 or fps <= 0:
        return []
    h, w = int(batch.shape[1]), int(batch.shape[2])
    step = max(1, min(h, w) // 64)
    gray = batch[:, ::step, ::step, :3].float().mean(-1)
    diff = (gray[1:] - gray[:-1]).abs().mean((1, 2))
    thr = max(0.07, float(diff.median()) * 6.0)
    cuts: list[float] = []
    for i in torch.nonzero(diff > thr).flatten().tolist():
        t = (i + 1) / fps
        if t >= min_shot and (not cuts or t - cuts[-1] >= min_shot) and (int(batch.shape[0]) - (i + 1)) / fps >= min_shot:
            cuts.append(round(t, 3))
    return cuts


def shot_sheets(batch: torch.Tensor | None, fps: float, cuts: list[float] | None, max_sheets: int = 8,
                tiles: int = 4, tile_w: int = 384) -> list[dict[str, Any]]:
    """Build compact, labeled horizontal contact sheets from source-video shots.

    Hard-cut shots longer than four seconds are split into roughly four-second spans. If
    that still creates too many images, adjacent shortest spans are merged without changing
    their playback order.
    """
    if batch is None or getattr(batch, "ndim", 0) != 4 or int(batch.shape[0]) <= 0 or fps <= 0:
        return []
    n = int(batch.shape[0])
    max_sheets = max(1, int(max_sheets))
    tiles = max(1, int(tiles))
    tile_w = max(1, int(tile_w))
    cut_frames = sorted({max(1, min(n - 1, int(round(float(c) * fps))))
                         for c in (cuts or []) if 0 < float(c) < n / fps})
    shot_bounds = [0, *cut_frames, n]
    groups: list[dict[str, Any]] = []
    for shot_no, (shot_start, shot_end) in enumerate(zip(shot_bounds, shot_bounds[1:]), 1):
        if shot_end <= shot_start:
            continue
        step_frames = max(1, int(round(4.0 * fps)))
        part_bounds = list(range(shot_start, shot_end, step_frames)) + [shot_end]
        for a, b in zip(part_bounds, part_bounds[1:]):
            if b > a:
                groups.append({"start_frame": a, "end_frame": b, "shots": [shot_no],
                               "segments": [{"start_frame": a, "end_frame": b, "shot": shot_no}]})

    while len(groups) > max_sheets:
        pair = min(range(len(groups) - 1),
                   key=lambda i: (groups[i + 1]["end_frame"] - groups[i]["start_frame"], i))
        left, right = groups[pair], groups[pair + 1]
        merged = {"start_frame": left["start_frame"], "end_frame": right["end_frame"],
                  "shots": list(dict.fromkeys(left["shots"] + right["shots"])),
                  "segments": left["segments"] + right["segments"]}
        groups[pair:pair + 2] = [merged]

    sheets: list[dict[str, Any]] = []
    for group in groups:
        a, b = group["start_frame"], group["end_frame"]
        count = min(tiles, b - a)
        picks = [min(b - 1, a + int((b - a) * (i + 0.5) / count)) for i in range(count)]
        shot_ids = group["shots"]
        shot_label = str(shot_ids[0]) if len(shot_ids) == 1 else f"{shot_ids[0]}-{shot_ids[-1]}"
        tiles_pil = []
        for frame_index in picks:
            frame = batch[frame_index]
            image = tensor_frame_to_pil(frame)
            height = max(1, int(round(image.height * tile_w / image.width)))
            image = image.resize((tile_w, height), getattr(Image, "Resampling", Image).LANCZOS)
            draw = ImageDraw.Draw(image)
            timestamp = frame_index / fps
            source_shot = next((segment["shot"] for segment in group["segments"]
                                if segment["start_frame"] <= frame_index < segment["end_frame"]), shot_ids[0])
            label = f"S{source_shot} {timestamp:.1f}s"
            draw.text((6, 5), label, font=None, fill="white", stroke_width=2, stroke_fill="black")
            tiles_pil.append((image, timestamp))
        sheet_h = max(image.height for image, _ in tiles_pil)
        sheet = Image.new("RGB", (tile_w * len(tiles_pil), sheet_h), "black")
        for index, (image, _timestamp) in enumerate(tiles_pil):
            sheet.paste(image, (index * tile_w, 0))
        start, end = a / fps, b / fps
        shot_value = shot_ids[0] if len(shot_ids) == 1 else shot_label
        sheets.append({"label": f"shot {shot_label} ({start:.1f}-{end:.1f} s)",
                       "url": pil_to_data_url(sheet, 88), "shot": shot_value, "start": start, "end": end})
    return sheets


def video_asset(batch: torch.Tensor, role: str, fps: float, limit: int = 12) -> tuple[Asset, list[float], float]:
    """源视频 → Asset：每个镜头至少取一帧（取中间），其余名额按镜头时长分配；帧旁标注镜头号和秒数。"""
    n = int(batch.shape[0]) if batch.ndim == 4 else 1
    duration = n / fps if fps > 0 else 0.0
    cuts = detect_cuts(batch, fps) if n > 1 else []
    bounds = [0] + [int(round(c * fps)) for c in cuts] + [n]
    spans = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
    budget = max(limit, len(spans))
    per = [1] * len(spans)
    extra = budget - len(spans)
    for _ in range(extra):  # 把剩余名额给「每帧代表时长」最长的镜头
        k = max(range(len(spans)), key=lambda k: (spans[k][1] - spans[k][0]) / (per[k] + 1))
        per[k] += 1
    picks: list[tuple[int, int]] = []
    for k, ((a, b), c) in enumerate(zip(spans, per)):
        for j in range(c):
            picks.append((k + 1, min(b - 1, a + int((b - a) * (j + 0.5) / c))))
    frames = [batch[i] if batch.ndim == 4 else batch for _, i in picks]
    urls = [pil_to_data_url(tensor_frame_to_pil(f), 88) for f in frames]
    labels = [f"shot {k}, {i / fps:.2f}s" if fps > 0 else f"shot {k}" for k, i in picks]
    asset = Asset("video", role, "", urls)
    asset.frame_labels = labels
    return asset, cuts, duration


def _waveform_has_sound(waveform: Any, threshold: float = 1e-5) -> bool | None:
    """True = 有声，False = 确认无声，None = 读不出来（调用方按「默认有声」处理）。"""
    if waveform is None:
        return False
    try:
        arr = waveform.detach().float().cpu().numpy() if hasattr(waveform, "detach") else (
            waveform.numpy() if hasattr(waveform, "numpy") else waveform
        )
        import numpy as np
        np_arr = np.asarray(arr, dtype=np.float32)
        if np_arr.size == 0:
            return False
        return bool(float(np.abs(np_arr).max()) > threshold)
    except Exception:
        return None


def inspect_source_audio(keep_source_audio: bool, source_audio: Any = None, video: Any = None) -> tuple[bool, str]:
    """Determine whether the source video actually has usable audio.

    Returns ``(has_audio, state)`` where ``state`` is one of ``off``, ``present``,
    ``silent``, ``none``, or ``assumed`` (when no audio metadata/waveform is supplied).
    """
    if not keep_source_audio:
        return False, "off"
    if isinstance(source_audio, bool):
        return (True, "present") if source_audio else (False, "none")
    if source_audio is not None:
        # ComfyUI AUDIO 是 {"waveform", "sample_rate"}；VHS_LoadVideo 给的是按需加载的 Mapping（不是 dict）
        wf = source_audio
        try:
            if hasattr(source_audio, "__getitem__") and not hasattr(source_audio, "shape"):
                wf = source_audio["waveform"]
        except Exception:
            wf = None
            return True, "assumed"
        if wf is None:
            return False, "none"
        sound = _waveform_has_sound(wf)
        if sound is None:
            return True, "assumed"
        return (True, "present") if sound else (False, "silent")
    if video is not None:
        get_components = getattr(video, "get_components", None)
        if callable(get_components):
            try:
                comps = get_components()
                comp_audio = getattr(comps, "audio", None)
                if comp_audio is None:
                    return False, "none"
                return inspect_source_audio(True, source_audio=comp_audio)
            except Exception:
                pass
        try:
            from .media import _video_source
            import av
            import io

            src = _video_source(video)
            if src is not None:
                if isinstance(src, io.BytesIO):
                    src.seek(0)
                with av.open(src, mode="r") as container:
                    if not container.streams.audio:
                        return False, "none"
                    peak = 0.0
                    checked = 0
                    for frame in container.decode(container.streams.audio[0]):
                        arr = frame.to_ndarray()
                        if arr.size:
                            denom = float(np_iinfo_max := (32768.0 if arr.dtype.kind in ("i", "u") else 1.0))
                            peak = max(peak, float( abs(arr).max() ) / denom)
                            if peak > 1e-5:
                                return True, "present"
                        checked += 1
                        if checked >= 200:
                            break
                    return (True, "present") if peak > 1e-5 else (False, "silent")
        except Exception:
            pass
    return True, "assumed"


def source_video_assets(batch: torch.Tensor | None, role_choice: str, request: str, has_refs: bool, fps: float,
                        keep_source_audio: bool, source_audio: Any = None,
                        video: Any = None) -> tuple[list[Asset], list[float] | None, float, list[str]]:
    """两个优化节点共用：定视频用途（auto 按需求文字）、找硬切、按镜头取帧；编辑且确认/假定有原声时补 <Audio 1>。"""
    if batch is None:
        return [], None, 0.0, []
    role, why = (infer_video_role(request, has_refs) if role_choice == "auto" else (role_choice, "chosen on the node"))
    asset, cuts, duration = video_asset(batch, role, fps)
    assets = [asset]
    lines = [f"video_role={role} ({why})", f"source={duration:.2f}s cuts=" + (", ".join(f"{c:.2f}" for c in cuts) or "none")]
    if role == "editing":
        has_audio, audio_state = inspect_source_audio(keep_source_audio, source_audio=source_audio, video=video)
        if has_audio:
            assets.append(Asset("audio", "source_audio", ""))
            lines.append(f"source audio kept as <Audio 1> (fully_copy) audio={audio_state}")
        else:
            lines.append(f"audio={audio_state}")
    return assets, (cuts if role == "editing" else None), duration, lines


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"
CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯　-〿！-／：-＠［-｀｛-･ｦ-ﾟ\U00020000-\U0003134F]")


@dataclass
class Asset:
    kind: str                 # picture | video | audio
    role: str                 # first_frame | last_frame | reference | editing | continuation | motion | audio
    note: str = ""
    data_urls: list[str] = field(default_factory=list)
    frame_labels: list[str] = field(default_factory=list)


@dataclass
class Job:
    request: str
    duration: float
    assets: list[Asset] = field(default_factory=list)
    mode: str = "auto"
    dialogue: list[str] | None = None
    cuts: list[float] | None = None      # 源视频硬切（秒），编辑任务里目标视频沿用
    generic_edit: bool = False
    edit_analysis: dict[str, Any] | None = None
    mv_spec: dict[str, Any] | None = None
    mv_handbook: str = ""
    dh: bool = False        # 数字人模板：单镜头固定机位、背景不变、不拿道具
    lipsync: bool = False   # 数字人模板 + 接了音频：对口型，<Audio 1> 原声 1:1，不编台词


_MV_FIELD_RE = re.compile(
    r"^\s*(segment_duration|lyrics|section|persona(?:_[1-9])?|scene|prev_end_scale|customer_request)\s*[:：]\s*(.*?)\s*$",
    re.I,
)
_MV_DISTINCT_KEYS = frozenset(
    {"segment_duration", "lyrics", "section", "prev_end_scale", "customer_request", *(f"persona_{i}" for i in range(1, 10))}
)
_MV_SCALE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("extreme_close_up", re.compile(r"\bextreme\s+close[- ]?up\b|\bECU\b|大特写|极特写", re.I)),
    ("medium_close_up", re.compile(r"\bmedium[- ]close[- ]?up\b|\bMCU\b|中近景|近景", re.I)),
    ("close_up", re.compile(r"\bclose[- ]?up\b|\bCU\b|特写", re.I)),
    ("extreme_wide", re.compile(r"\bextreme\s+(?:wide|long)(?:\s+(?:shot|view|angle))?\b|\bEWS\b|大远景|极远景", re.I)),
    ("medium_wide", re.compile(r"\bmedium[- ](?:wide|long|full)(?:\s+(?:shot|view))?\b|\bMWS\b|中远景|中全景", re.I)),
    ("wide_shot", re.compile(r"\b(?:wide|long)\s+(?:shot|view|frame|framing|tableau)\b|\bwide[- ]angle\b|远景", re.I)),
    ("full_shot", re.compile(r"\bfull(?:-body|\s+body)?\s+(?:shot|view|frame|framing)\b|全景|全身(?:景|镜头)?", re.I)),
    ("medium_shot", re.compile(r"\b(?:medium|mid)(?:[- ]two|[- ]three)?\s+(?:shot|view|frame|framing)\b|中景", re.I)),
)


def _clean_mv_field_value(raw: str) -> str:
    val = str(raw or "").strip()
    if val in {'""', "''", "“”", "「」"} or val.lower() in {"null", "none"}:
        return ""
    return val


def _split_mv_lyrics(lyrics: str) -> list[str]:
    raw = _clean_mv_field_value(lyrics)
    if not raw:
        return []
    if len(raw) >= 2 and (raw[0], raw[-1]) in {('"', '"'), ("'", "'"), ("“", "”"), ("「", "」")}:
        inner = raw[1:-1].strip()
        if inner and raw[0] not in inner and raw[-1] not in inner:
            raw = inner
    if not raw:
        return []
    clauses: list[str] = []
    for line in re.split(r"[\r\n|｜/]+", raw):
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"[，。！？；;]+", line):
            part = part.strip()
            if not part:
                continue
            if CJK.search(part):
                subparts = re.split(
                    r"(?<=[\u3400-\u9fffA-Za-z0-9])\s+(?=[\u3400-\u9fff])|"
                    r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fffA-Za-z0-9])",
                    part,
                )
                clauses.extend(s.strip() for s in subparts if s.strip())
            else:
                clauses.append(part)
    return clauses or [raw]


def _extract_mv_sung_text(blocks: list[str]) -> str:
    parts: list[str] = []
    prev_trailing_trans = False
    for raw_block in blocks:
        cleaned = re.sub(r"^\s*\[[^\]]+\]\s*", "", str(raw_block or "")).strip()
        starts_trans = bool(re.match(r"^<scenetrans>\s*", cleaned, flags=re.I))
        ends_trans = bool(re.search(r"\s*<scenetrans>$", cleaned, flags=re.I))
        cleaned = re.sub(r"<scenetrans>", "", cleaned, flags=re.I).strip()
        if not cleaned:
            prev_trailing_trans = ends_trans
            continue
        if parts and not (prev_trailing_trans and starts_trans):
            parts.append(" ")
        parts.append(cleaned)
        prev_trailing_trans = ends_trans
    return "".join(parts)


def _mv_shot_scale_buckets(text: str) -> set[str]:
    remaining = str(text or "")
    buckets: set[str] = set()
    for bucket, pattern in _MV_SCALE_PATTERNS:
        if pattern.search(remaining):
            buckets.add(bucket)
            remaining = pattern.sub(" ", remaining)
    return buckets


def parse_mv_request(request: str, default_seconds: float = 0.0) -> dict[str, Any]:
    """Parse an MV request as either a structured 'field: value' block or plain free text."""
    cleaned_req = _strip_h3_mv_markers(str(request or ""))
    raw_lines = cleaned_req.splitlines()
    non_empty = [line for line in raw_lines if line.strip()]

    fields: dict[str, str] = {
        "segment_duration": "",
        "section": "",
        "scene": "",
        "prev_end_scale": "",
        "customer_request": "",
    }
    lyrics_parts: list[str] = []
    personas: dict[int, str] = {}
    seen_keys: set[str] = set()
    is_field_block = bool(non_empty and _MV_FIELD_RE.match(non_empty[0]))

    if is_field_block:
        current_key: str | None = None
        for line in raw_lines:
            if not line.strip():
                continue
            match = _MV_FIELD_RE.match(line)
            if match:
                key = match.group(1).lower()
                if key == "persona":
                    key = "persona_1"
                seen_keys.add(key)
                val = _clean_mv_field_value(match.group(2))
                current_key = key
                if key == "lyrics":
                    if val:
                        lyrics_parts.append(val)
                elif key.startswith("persona_"):
                    idx = int(key.split("_", 1)[1])
                    if val:
                        personas[idx] = val
                    else:
                        personas.pop(idx, None)
                else:
                    fields[key] = val
            elif current_key is not None:
                cont = line.strip()
                if current_key == "lyrics":
                    lyrics_parts.append(cont)
                elif current_key.startswith("persona_"):
                    idx = int(current_key.split("_", 1)[1])
                    personas[idx] = (personas.get(idx, "") + " " + cont).strip()
                else:
                    sep = "\n" if current_key == "customer_request" else " "
                    fields[current_key] = (fields[current_key] + sep + cont).strip() if fields[current_key] else cont
        lyrics = "\n".join(lyrics_parts).strip()
    else:
        fields["customer_request"] = cleaned_req.strip()
        explicit_sung: list[str] = []
        for grp in re.findall(
            r"(?:唱(?:着|道|词)?|歌词|lyrics?|sing(?:s|ing)?)\s*[:：]?\s*(?:“([^”]+)”|「([^」]+)」|\"([^\"]+)\")",
            cleaned_req,
            flags=re.I,
        ):
            hit = next((item.strip() for item in grp if item and item.strip()), "")
            if hit:
                explicit_sung.append(hit)
        lyrics = "\n".join(explicit_sung).strip()

    seg_dur: float | None = None
    dur_str = fields["segment_duration"].strip()
    if dur_str:
        dur_match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds|秒)?$", dur_str, flags=re.I)
        if dur_match and float(dur_match.group(1)) > 0:
            seg_dur = float(dur_match.group(1))
    fallback_dur = float(default_seconds) if float(default_seconds or 0.0) > 0 else 5.0
    resolved_duration = seg_dur if seg_dur is not None else fallback_dur

    return {
        "is_field_block": is_field_block,
        "seen_keys": sorted(seen_keys),
        "segment_duration": seg_dur,
        "duration": resolved_duration,
        "lyrics": lyrics,
        "lyric_lines": _split_mv_lyrics(lyrics),
        "section": fields["section"],
        "personas": dict(sorted(personas.items())),
        "scene": fields["scene"],
        "prev_end_scale": fields["prev_end_scale"],
        "customer_request": fields["customer_request"],
    }


# ------------------------------------------------------------------ fixed protocol pieces
def resolve_mode(job: Job) -> str:
    if job.mode in MODES:
        return job.mode
    pics = [a for a in job.assets if a.kind == "picture"]
    if any(a.kind in ("video", "audio") for a in job.assets) or any(a.role == "reference" for a in pics):
        return "Ref2VA"
    roles = {a.role for a in pics}
    if {"first_frame", "last_frame"} <= roles:
        return "FL2VA"
    if "first_frame" in roles:
        return "I2VA"
    if "last_frame" in roles:
        return "L2VA"
    return "T2VA"


def _sec(duration: float) -> str:
    return f"{duration:.2f}"


def instruction_line(mode: str, duration: float, last_shot: int) -> str:
    if mode == "I2VA":
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if mode == "FL2VA":
        return ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
                f"mark of the target video; Picture 2 (from Shot {last_shot}) aligns with the {_sec(duration)}-second mark of "
                "the target video.")
    if mode == "L2VA":
        return (f"How the reference pictures align with the target video — <Picture 1> (from [Shot {last_shot}]) aligns with "
                f"the {_sec(duration)}-second mark of the target video.")
    return ""


def extract_dialogue(text: str) -> list[str]:
    found: list[str] = []
    for pattern in (r"“([^”]+)”", r"「([^」]+)」", r"\"([^\"]+)\""):
        found += re.findall(pattern, text)
    return [s.strip() for s in found if s.strip()]


# ------------------------------------------------------------------ prompt construction
def build_messages(job: Job, mode: str) -> list[dict[str, Any]]:
    pics = [a for a in job.assets if a.kind == "picture"]
    videos = [a for a in job.assets if a.kind == "video"]
    audios = [a for a in job.assets if a.kind == "audio"]
    is_ref = mode == "Ref2VA"
    dialogue = job.dialogue if job.dialogue is not None else extract_dialogue(job.request)
    allowed = ([f"<Picture {i + 1}>" for i in range(len(pics))] + [f"<Video {i + 1}>" for i in range(len(videos))]
               + [f"<Audio {i + 1}>" for i in range(len(audios))])
    sections = REF_SECTIONS if is_ref else BASE_SECTIONS
    fixed = instruction_line(mode, job.duration, 1)
    rules = [
        f"Task mode: {mode}. Target video duration: {_sec(job.duration)} seconds.",
        f"Output exactly these sections in this order, each starting with its field name and a colon: {', '.join(sections)}.",
    ]
    if fixed:
        rules.append("The first line must be this alignment instruction copied verbatim (its shot number is adjusted "
                     "automatically if you use more shots), followed by one blank line:\n" + fixed)
        if mode == "FL2VA":
            rules.append("FL2VA: prefer one continuous [Shot 1] that travels from Picture 1 to Picture 2; describe the path, "
                         "not two static images.")
    else:
        rules.append("Do not write an alignment instruction line; begin directly with the first section.")
    is_mv = job.mv_spec is not None
    mv = job.mv_spec or {}
    mv_lyrics = list(mv.get("lyric_lines") or [])
    if is_mv and job.duration > 2.2:
        shot_rule = (
            "[Shot 1] MUST NOT have a timestamp. Use 2-3 shots total. Each later shot starts with '[Shot N] At MM:SS.mmm,' "
            f"with strictly increasing times at least 1.20s from 00:00.000 and at least 1.00s before {_sec(job.duration)}s "
            f"(between 00:01.200 and {_mmss(job.duration - 1.0)})."
        )
    else:
        shot_rule = (
            "[Shot 1] has no timestamp. Each later shot starts with '[Shot N] At MM:SS.mmm,' with strictly increasing times "
            f"strictly below {_sec(job.duration)} seconds. Leave time for every spoken line (about 4 Chinese characters per second)."
        )
    if is_mv and not mv_lyrics:
        speaker_rule = (
            "Lyrics are empty for this MV segment: NEVER output <d>...</d> blocks or speaker IDs (S1), and NEVER invent lyrics."
        )
    else:
        speaker_rule = (
            "Every vocal source gets a stable speaker ID (S1), (S2), ...; attach it together with a short voice description "
            "(age, gender, timbre, delivery) in the sentence where that person actually speaks, not at an earlier silent appearance."
        )
    if is_mv and audios:
        audio_fields_rule = (
            "overall_soundscape: 1-2 sentences stating the vocal and performance audio from <Audio 1> plays cleanly throughout "
            "without added competing sound effects (no <d>, no N/A). non_diegetic_music: 1 sentence stating <Audio 1> is reused "
            "1:1 as the complete original song track synced to the cuts, unchanged."
        )
    else:
        audio_fields_rule = (
            "overall_soundscape: 1-4 sentences of ambience and physical sounds, no dialogue. non_diegetic_music: 1-3 sentences "
            "on instrumentation, tempo and dynamics, or N/A."
        )
    if job.dh:
        shot_rule = (
            "Write exactly ONE continuous shot: [Shot 1] with no timestamp, no later shots and no cuts. Keep one fixed camera "
            "position and one shot scale for the whole video (the framing the user asked for; if none, a medium shot from the "
            "waist up), named with ONE unambiguous framing term that is never mixed with another (not 'medium close-up from "
            "the waist up'); at most a very slow, small push-in."
        )
        voice = "<Audio 1>" if job.lipsync else "her speech"
        rules.append(
            "This is a digital-human video: one person talking or singing to the camera. The person is the picture the user "
            "names for the person (if unspecified, the first picture not named as the background). Describe ONE concrete background: the picture the user names as "
            "the background or scene if there is one, otherwise the scene behind the person in the person's picture (if that "
            "background is plain, a plain studio backdrop of that color); state that it stays identical in every frame, from "
            "the first frame to the last. The character holds nothing in her hands: no microphone, no props. Her appearance "
            "(face, hair, outfit) comes from the person's picture but NOT its pose: she "
            "stands in a relaxed natural stance; gestures stay small and natural and her expression follows the mood of "
            f"{voice}. Never describe anything that contradicts the pictures (do not move, remove or add objects that the "
            "pictures show)."
        )
    if job.lipsync:
        speaker_rule = (
            "The character's voice is <Audio 1>. NEVER invent dialogue or lyrics and NEVER output <d>...</d> blocks: write that "
            "<Subject 1> speaks (or sings) the content of <Audio 1> to the camera with lip movements precisely synced to "
            "<Audio 1>, and give her a speaker ID (S1) with a short voice description."
        )
        audio_fields_rule = (
            "overall_soundscape: 1-2 sentences: <Audio 1> (the voice and any music inside it) plays unchanged as the complete "
            "soundtrack, with nothing added. non_diegetic_music: if <Audio 1> contains song accompaniment, one sentence saying "
            "it is the original music inside <Audio 1>, unchanged; otherwise N/A."
        )
        rules.append(
            "The summary starts with '[reference generation + audio reuse]'. <Audio 1> is the provided voice track; there is "
            "no <Video 1>, so never call it the soundtrack of a video."
        )
    elif job.dh:
        rules.append(
            "There is no voice track: she speaks the lines the user quoted, word for word inside <d>[Language] ...</d>; if the "
            "user quoted none, write short natural lines that fit the request, at about 4 Chinese characters per second, and "
            "keep her talking for most of the video."
        )
    rules += [
        shot_rule,
        "All descriptive prose is English. Dialogue and lyrics stay in their original language inside <d>[Chinese] ...</d> "
        "(or the matching language tag); quoted on-screen text stays verbatim inside English double quotes.",
        speaker_rule,
        "Camera movement uses the guide's vocabulary in natural sentences with amplitude and speed when meaningful; a big "
        "framing change (for example close-up to full body) is 'with large amplitude'.",
        "On-screen text: only quote text the user wrote. Signs, couplets or labels in the pictures that the user did not spell "
        "out are described without quoting or guessing their characters; never invent names from unreadable signage.",
        audio_fields_rule,
    ]
    if allowed:
        rules.append("The only reference labels that exist are: " + ", ".join(allowed) + ". Never invent other "
                     "<Picture>/<Video>/<Audio> labels." + (" Use <Subject N> for reusable visible content (people, places, "
                                                          "props, styles) defined from these assets." if is_ref else ""))
    else:
        rules.append("There are no reference assets: use no <Picture>, <Video>, <Audio> or <Subject> labels.")
    if is_ref:
        editing = any(v.role == "editing" for v in videos)
        word_target_text = "." if editing else (" and normally has 350-480 English words." if is_mv else " and normally has 350-500 English words.")
        rules.append("<Video N> is only for a whole source video that is edited, continued, or used for camera/pacing "
                     "structure; people, places and props are always <Subject N>. detailed_description opens with one or two "
                     "style sentences before [Shot 1]" + word_target_text)
        task_types = []
        generic_motion = bool(job.generic_edit and job.edit_analysis
                              and job.edit_analysis.get("edit_type") == "motion_transfer")
        if is_mv:
            if pics:
                task_types.append("reference generation")
            if any(a.role == "source_audio" for a in audios):
                task_types.append("audio reuse")
            elif audios:
                task_types.append("audio reference")
            if not task_types:
                task_types.append("reference generation")
            if any(a.role == "source_audio" for a in audios):
                rules.append("In retention_analysis, include '<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target "
                             "video's complete final audio track.'")
            for subj_no, p_text in sorted((mv.get("personas") or {}).items()):
                if not p_text:
                    continue
                if subj_no <= len(pics):
                    rules.append(f"In subject_definitions, the <Subject {subj_no}> line MUST be '<Subject {subj_no}> "
                                 f"(from <Picture {subj_no}>): {p_text}' (copying persona_{subj_no} verbatim character-for-character).")
                else:
                    rules.append(f"In subject_definitions, the <Subject {subj_no}> line MUST contain persona_{subj_no} verbatim: {p_text}")
            if mv_lyrics:
                rules.append("Every sung lyric line must appear verbatim and in order inside <d>[Language] ...</d> blocks: "
                             + " | ".join(mv_lyrics))
            prev_scale_val = str(mv.get("prev_end_scale") or "").strip()
            if prev_scale_val:
                rules.append(f"Previous segment ended on '{prev_scale_val}': [Shot 1] MUST switch to a clearly different shot "
                             "scale and angle, and must not repeat that same shot-scale term.")
        elif job.generic_edit:
            if generic_motion:
                task_types.append("reference generation")
                rules.append("This is generic motion transfer, not an edit of the source video: the summary describes a new target "
                             "generated from the motion reference, and does not say it is an edited version of <Video 1>. "
                             "In retention_analysis, <Video 1> is partially_preserved only as the source of motion, rhythm, "
                             "camera and framing.")
            else:
                task_types.extend(["video editing", "reference generation"])
                rules.append("This is a generic video edit: the summary begins with 'The target video is an edited version of "
                             "<Video 1>.' and only the analyzed targets change. Keep exactly one [Shot 1] and do not write timestamps "
                             "or shot-by-shot narration.")
                rules.append("In retention_analysis, <Video 1> is partially_preserved: preserve all unrequested content, shots, cuts, "
                             "camera, framing, motion and timing frame by frame. Do not enumerate source cuts as separate prompt shots.")
                active_for_rules = _active_edits(job.edit_analysis or {})
                text_outfits = [e for e in active_for_rules if e.get("kind") == "outfit" and _edit_picture(e) is None]
                if text_outfits:
                    rules.append("Outfit described only in words: the sentence subject is the person, who keeps her/his own "
                                 "face, hair, body and motion from <Video 1> and now wears the requested clothing; this edit has "
                                 "no picture, so write no <Picture N> or <Subject j> for it and no picture negative.")
                picture_outfits_r = [e for e in active_for_rules if e.get("kind") == "outfit" and _edit_picture(e) is not None]
                if len(picture_outfits_r) > 1:
                    for e in picture_outfits_r:
                        ow_e = _outfit_wording(_outfit_choice(e))
                        rules.append(f"Outfit from <Picture {_edit_picture(e)}>: include '{ow_e['do_not_copy']} "
                                     f"<Picture {_edit_picture(e)}>; take {ow_e['take']}.' in BOTH summary and [Shot 1].")
                if picture_outfits_r:
                    outfit_edit = picture_outfits_r[0]
                    choice = _outfit_choice(outfit_edit)
                    ow = _outfit_wording(choice)
                    rules.append(
                        "Outfit edit rules (CRITICAL): (1) The sentence subject in summary and [Shot 1] MUST be the person "
                        f"(The <target_person> keeps her/his own {ow['keeps']} from <Video 1> "
                        "and now wears the <clothing> shown in <Picture N> as <Subject j> (from <Picture N>).), NEVER the clothes "
                        "themselves. (2) In subject_definitions write '<Subject j>: the outfit (<clothing description>) "
                        f"worn by the character in <Picture N>; {ow['subject_not']}' "
                        f"(3) In retention_analysis write '<Picture N>: partially_preserved - only {ow['take']} is "
                        f"referenced; {ow['picture_ignored']} <Picture N> are ignored.' (do NOT mark <Subject j> or "
                        f"<Picture N> fully_preserved). (4) Include the exact hard constraint '{ow['do_not_copy']} <Picture N>; "
                        f"take {ow['take']}.' in BOTH summary and [Shot 1]."
                        + (f" The user asked for the {_join_words(choice['wanted'])} too, so describe them with the clothing."
                           if choice["wanted"] else "")
                        + (f" The user does not want the {_join_words(choice['declined'])}; never describe them."
                           if choice["declined"] else "")
                    )
                if any(e.get("kind") == "prop" for e in active_for_rules):
                    rules.append(
                        "Prop edit rules: use the person (<target_person>) as the sentence subject, define <Subject j> as "
                        "'the prop only (<prop description>) shown in <Picture N>; NOT any character's face, hair, head accessories, "
                        "outfit or body in <Picture N>.', mark <Picture N> partially_preserved in retention_analysis, and forbid "
                        "copying any person, face, hair, head accessories or outfit from <Picture N>."
                    )
                if any(e.get("kind") == "face_swap" for e in active_for_rules):
                    rules.append(
                        "Face-swap rules: use The <target_person> as sentence subject, define <Subject j> as facial features and "
                        "face shape only (NOT hairstyle, head accessories, outfit, body or held items), mark <Picture N> "
                        "partially_preserved in retention_analysis, and state in summary and [Shot 1]: 'Do not copy the hairstyle, "
                        "head accessories, outfit, body or held items of <Picture N>; take only the facial features and face shape.'"
                    )
                for head_edit in [e for e in active_for_rules if e.get("kind") == "head_swap"]:
                    opts = _head_swap_opts(head_edit)
                    parts = ("face, facial features, hairstyle and head accessories" if opts["picture_accessories"]
                             else "face, facial features and hairstyle, without the picture's head accessories")
                    keep_hat = opts["keep_source_headwear"]
                    rules.append(
                        "Head-swap rules: use The <target_person> as sentence subject; replace the entire head ("
                        + parts + ") from <Subject j> (from <Picture N>) while keeping the outfit, body and motion"
                        + (" and her/his own original headwear" if keep_hat else "") + " from <Video 1>. Identify that person "
                        "ONLY by clothing, body and position; NEVER describe the original hair or hair color of <Video 1>"
                        + ("; the user asked to keep the original headwear, so say that she/he keeps her/his own headwear and "
                           "do not bring in the picture's head accessories (this overrides the handbook example)."
                           if keep_hat else ", and never its cap, hat or headwear (not even to say it disappears).")
                    )
            if job.edit_analysis and job.edit_analysis.get("style_source") == "picture":
                rules.append("Visual style source is picture: each replacing subject keeps the art style and rendering of its own "
                             "picture while the rest of the video keeps the style of <Video 1>; mention 'and its art style' in each "
                             "replacing Subject's or Picture's retention_analysis line.")
        else:
            if any(a.role in ("first_frame", "last_frame") for a in pics):
                task_types.append("keyframe completion")
            if any(a.role == "reference" for a in pics) or any(v.role == "motion" for v in videos):
                task_types.append("reference generation")
            if editing:
                task_types.insert(0, "video editing")
                rules.append("This is a video edit: after the task prefix the summary begins with 'The target video is an edited "
                             "version of <Video 1>.' and the edit keeps everything the user did not ask to change.")
                rules.append("retention_analysis must contain a <Video 1> line marked partially_preserved: only what the request "
                             "changes is changed; shots, cuts and their timing, camera, backgrounds, every other character and all "
                             "motion are retained frame by frame.")
                rules.append("For an edit, detailed_description defines each changed or replacing subject precisely (identity, hair, "
                             "outfit, style from its picture) and says where it appears; for the unchanged action, camera and timing "
                             "state that they follow <Video 1> exactly instead of re-narrating them beat by beat (an inaccurate "
                             "action description misleads the edit). Normally 150-350 English words.")
                if job.cuts is not None:
                    if job.cuts:
                        stamps = ", ".join(f"[Shot {k + 2}] At {_mmss(c)}" for k, c in enumerate(job.cuts))
                        rules.append(f"<Video 1> has hard cuts at {', '.join(_mmss(c) for c in job.cuts)} (detected in code). The target "
                                     f"keeps them: write exactly {len(job.cuts) + 1} shots - [Shot 1], then {stamps} - one per source "
                                     "shot, using these exact timestamps. Video frames below are labelled with their shot number.")
                    else:
                        rules.append("<Video 1> is one continuous shot without cuts: write a single [Shot 1].")
        if not job.generic_edit and any(v.role == "continuation" for v in videos):
            task_types.insert(0, "video continuation")
        if any(a.role != "source_audio" for a in audios):
            task_types.append("audio reference")
        if task_types:
            rules.append("Use the summary prefix [" + " + ".join(dict.fromkeys(task_types)) + "] unless the request clearly "
                         "needs a different official task type.")
    else:
        rules.append("integrated_multimodal_description is vivid and concrete (about 150-320 English words for 5-10 seconds).")
    if dialogue:
        rules.append("These spoken lines must appear verbatim (same characters and punctuation) inside <d> blocks: "
                     + " | ".join(dialogue))
    rules += ["Describe only what the request and the attached media support; ground appearance, setting and props in what is "
              "actually visible. Do not add plot beyond the request.",
              "Output only the final prompt text: no preface, no notes, no markdown, no reasoning."]

    guide = (_guide("ref-en.txt") + "\n\n---\n\n" + _guide("base-en.txt")) if is_ref else _guide("base-en.txt")
    system = ("You are a professional MiniMax H3 prompt writer. Follow the official H3 prompt-writing guide below exactly, "
              "including field names, section order, label rules, timing notation and camera vocabulary.\n\n"
              "===== OFFICIAL GUIDE =====\n" + guide + "\n===== END OF GUIDE =====\n\nHard requirements for this job:\n- "
              + "\n- ".join(rules))
    if is_mv and job.mv_handbook.strip():
        system += "\n\n===== MV SEGMENT HANDBOOK =====\n" + job.mv_handbook.strip() + "\n===== END MV SEGMENT HANDBOOK ====="

    role_text = {"first_frame": "first frame of the target video", "last_frame": "last frame of the target video",
                 "reference": "reference picture", "editing": "source video to edit", "continuation": "source video to continue",
                 "motion": "motion and camera reference video", "audio": "reference audio",
                 "source_audio": "original soundtrack of Video 1, kept as the final audio"}
    if job.generic_edit and job.edit_analysis and job.edit_analysis.get("edit_type") == "motion_transfer":
        role_text["editing"] = "motion and camera reference video"
    content: list[dict[str, Any]] = []
    if is_mv:
        mv_lines = [
            f"- segment_duration: {_sec(job.duration)}",
            f"- section: {mv.get('section') or '(infer from request)'}",
            f"- lyrics: {mv.get('lyrics') if mv.get('lyrics') else '(empty — instrumental / no scripted <d> lyrics)'}",
            f"- scene: {mv.get('scene') or '(infer from customer_request and section)'}",
            f"- prev_end_scale: {mv.get('prev_end_scale') or '(none)'}",
            f"- customer_request: {mv.get('customer_request') or '(expand from section, scene, and performers)'}",
        ]
        for subj_no, p_text in sorted((mv.get("personas") or {}).items()):
            pic_tag = f" (paired with <Picture {subj_no}>)" if subj_no <= len(pics) else ""
            mv_lines.append(f"- persona_{subj_no}{pic_tag}: {p_text}")
        content.append({"type": "text", "text": "MV segment structured inputs:\n" + "\n".join(mv_lines)})
    content.append({"type": "text", "text": "User request (may be Chinese):\n" + job.request.strip()})
    manifest: list[str] = []
    for i, a in enumerate(pics, 1):
        manifest.append(f"Picture {i}: {role_text[a.role]}" + (f" — {a.note}" if a.note else ""))
    for i, a in enumerate(videos, 1):
        manifest.append(f"Video {i}: {role_text[a.role]} ({len(a.data_urls)} sampled frames)" + (f" — {a.note}" if a.note else ""))
    for i, a in enumerate(audios, 1):
        manifest.append(f"Audio {i}: {a.note or role_text.get(a.role, 'reference audio')} (you cannot hear it; use the note)")
    if manifest:
        content.append({"type": "text", "text": "Attached assets:\n" + "\n".join(manifest)})
    for i, a in enumerate(pics, 1):
        content.append({"type": "text", "text": f"Picture {i}:"})
        content.append({"type": "image_url", "image_url": {"url": a.data_urls[0]}})
    for i, a in enumerate(videos, 1):
        for k, url in enumerate(a.data_urls, 1):
            tag = f" ({a.frame_labels[k - 1]})" if k - 1 < len(a.frame_labels) else ""
            content.append({"type": "text", "text": f"Video {i} frame {k}/{len(a.data_urls)}{tag}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


# ------------------------------------------------------------------ validation and repair
def split_sections(text: str, names: list[str]) -> dict[str, Any]:
    marks = []
    for name in names:
        match = re.search(r"(?m)^\s*" + re.escape(name) + r"\s*:", text)
        if match:
            marks.append((match.start(), match.end(), name))
    marks.sort()
    out: dict[str, Any] = {}
    for i, (_, end, name) in enumerate(marks):
        out[name] = text[end: marks[i + 1][0] if i + 1 < len(marks) else len(text)].strip()
    out["__order"] = [name for _, _, name in marks]
    out["__preamble"] = text[: marks[0][0]].strip() if marks else text.strip()
    return out


def shots(body: str) -> list[tuple[int, float | None]]:
    found = []
    for m in re.finditer(r"\[Shot (\d+)\](\s*At (\d{2}):(\d{2})\.(\d{3}))?", body):
        stamp = int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1000 if m.group(2) else None
        found.append((int(m.group(1)), stamp))
    return found


def main_body(text: str, mode: str) -> str:
    names = REF_SECTIONS if mode == "Ref2VA" else BASE_SECTIONS
    return split_sections(text, names).get("detailed_description" if mode == "Ref2VA" else BASE_SECTIONS[0], "")


def _prompt_segments(text: str) -> list[str]:
    """提示词按句号、分号、换行切成小段（校验时按段判断来源和肯否）。"""
    return [seg for seg in re.split(r"(?<=[.;])\s+|\n", str(text or "")) if seg.strip()]


def validate(
    text: str,
    mode: str,
    job: Job,
    generic_edit: bool = False,
    mv_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def add(code: str, msg: str, hard: bool = True):
        issues.append({"code": code, "msg": msg, "hard": hard})

    is_ref = mode == "Ref2VA"
    generic_edit = bool(generic_edit or job.generic_edit)
    effective_mv = mv_spec if mv_spec is not None else job.mv_spec
    generic_motion = bool(generic_edit and job.edit_analysis
                          and job.edit_analysis.get("edit_type") == "motion_transfer")
    names = REF_SECTIONS if is_ref else BASE_SECTIONS
    sec = split_sections(text, names)
    if sec["__order"] != names:
        add("sections", f"sections must be exactly {names} in this order; found {sec['__order']}")
    body = sec.get("detailed_description" if is_ref else BASE_SECTIONS[0], "")
    found = shots(body)
    if not found or found[0][0] != 1:
        add("shot1", "the main description must start with [Shot 1]")
    else:
        if found[0][1] is not None or re.search(
            r"\[Shot 1\]\s*(?:At\s+\d+(?::\d+(?:\.\d+)?)?(?:\s*s(?:ec(?:onds)?)?)?|[\(\[]?\d{1,2}:\d{2}(?:\.\d{1,3})?[\)\]]?)",
            body,
            re.I,
        ):
            add("shot1_time", "[Shot 1] must not carry a timestamp")
        last = -1.0
        for position, (number, stamp) in enumerate(found[1:], start=2):
            if number != position:
                add("shot_order", f"shots must be numbered sequentially; got [Shot {number}] at position {position}")
            if stamp is None:
                add("shot_time", f"[Shot {number}] needs 'At MM:SS.mmm,'")
            elif stamp <= last or stamp >= job.duration:
                add("shot_time", f"[Shot {number}] at {stamp:.3f}s must increase and stay below {job.duration:.2f}s")
            else:
                last = stamp
            if effective_mv is not None and stamp is not None:
                if stamp < 1.2 - 1e-9:
                    add("mv_shot_time", f"[Shot {number}] at {stamp:.3f}s must be at least 1.20s after 00:00.000")
                if (job.duration - stamp) < 1.0 - 1e-9:
                    add("mv_shot_time", f"[Shot {number}] at {stamp:.3f}s must leave at least 1.00s before {job.duration:.2f}s")
    if job.dh and len(found) > 1:
        add("dh_shots", "a digital-human video must be exactly one continuous [Shot 1] with no cuts")
    if job.lipsync:
        if "<d>" in text:
            add("lipsync_dialogue", "do not invent dialogue or lyrics: the words come from <Audio 1>; remove every <d>...</d> block")
        if "<Audio 1>" not in body:
            add("lipsync_audio", "[Shot 1] must say the character speaks or sings <Audio 1> with lips synced to it")
        if not sec.get("summary", "").lstrip().startswith("[reference generation + audio reuse]"):
            add("lipsync_summary", "the summary must start with '[reference generation + audio reuse]'")
        if re.search(r"\bVideo 1\b", text) and not any(a.kind == "video" for a in job.assets):
            add("lipsync_video_label", "there is no source video: never mention Video 1; <Audio 1> is the provided voice track")
    if effective_mv is not None and found and not (2 <= len(found) <= 3):
        add("mv_shots", f"MV segment recommends 2-3 shots; found {len(found)}", hard=False)
    if generic_edit:
        later_shots = sorted({int(n) for n in re.findall(r"\[Shot (\d+)\]", text) if int(n) >= 2})
        if later_shots:
            add("generic_shots", "generic video edits must contain only [Shot 1]; found " + ", ".join(
                f"[Shot {n}]" for n in later_shots))
        if re.search(r"\bAt\s+\d{1,2}:\d{2}(?:\.\d{1,3})?\b", text, re.I):
            add("generic_timestamp", "generic video edits must not contain timestamps such as 'At MM:SS'")
    fixed = instruction_line(mode, job.duration, found[-1][0] if found else 1)
    if fixed and sec["__preamble"].strip() != fixed:
        add("instruction", "the first line must be the exact alignment instruction", hard=False)
    elif not fixed and sec["__preamble"]:
        add("preamble", "text before the first section", hard=False)
    counts = {"Picture": sum(a.kind == "picture" for a in job.assets), "Video": sum(a.kind == "video" for a in job.assets),
              "Audio": sum(a.kind == "audio" for a in job.assets)}
    for kind, limit in counts.items():
        for n in sorted({int(x) for x in re.findall(rf"<{kind} (\d+)>", text)}):
            if n > limit:
                add("label", f"<{kind} {n}> does not exist (only {limit} {kind.lower()} asset(s))")
    if not is_ref and re.search(r"<Subject \d+>", text):
        add("label", "base modes must not use <Subject N> labels")
    if is_ref:
        used = set(re.findall(r"<Subject \d+>", text))
        defined = set(re.findall(r"(?m)^\s*(<Subject \d+>)", sec.get("subject_definitions", "")))
        for label in sorted(used - defined):
            add("label", f"{label} is used but not defined in subject_definitions")
        narrative_text = sec.get("summary", "") + "\n" + sec.get("detailed_description", "")
        for label in sorted(defined, key=lambda s: int(re.search(r"\d+", s).group(0))):
            if label not in narrative_text:
                add("dangling_subject", f"{label} is defined in subject_definitions but never used in summary or detailed_description")
        if not re.match(r"\s*\[[a-z +]+\]", sec.get("summary", "")):
            add("summary", "summary must start with a [task type] prefix")
        if any(a.role == "editing" for a in job.assets) and not generic_motion and "edited version of <Video 1>" not in sec.get("summary", ""):
            add("summary", "a video edit summary must contain 'The target video is an edited version of <Video 1>.'")
        if generic_edit:
            audio_kept = any(a.role == "source_audio" for a in job.assets)
            expected_prefix = ("[reference generation" if generic_motion else
                               "[video editing + reference generation")
            if audio_kept:
                expected_prefix += " + audio reuse]"
            else:
                expected_prefix += "]"
            if not sec.get("summary", "").lstrip().startswith(expected_prefix):
                add("summary", f"generic edit summary must start with {expected_prefix!r}")
        editing = any(a.role == "editing" for a in job.assets)
        retention = sec.get("retention_analysis", "")
        if editing and not re.search(r"<Video 1>[^\n]*partially_preserved", retention):
            add("retention", "retention_analysis needs a '<Video 1> ...: partially_preserved' line for the edited source video")
        if any(a.role == "source_audio" for a in job.assets) and not re.search(r"<Audio 1>[^\n]*fully_copy", retention):
            add("retention", "retention_analysis needs '<Audio 1>: fully_copy' (the source soundtrack is kept)")
        if editing and not generic_edit and job.cuts is not None and found:
            if len(found) != len(job.cuts) + 1:
                add("cuts", f"<Video 1> has {len(job.cuts) + 1} shots (cuts at {', '.join(_mmss(c) for c in job.cuts) or 'none'}); "
                            f"the prompt has {len(found)}")
            else:
                for (number, stamp), cut in zip(found[1:], job.cuts):
                    if stamp is not None and abs(stamp - cut) > 0.06:
                        add("cuts", f"[Shot {number}] must be At {_mmss(cut)} (the source cut), not {_mmss(stamp)}")
        if generic_edit:
            analysis = job.edit_analysis or {}
            active_edits = _active_edits(analysis)
            summary_lower = sec.get("summary", "").lower()
            for edit in active_edits:
                if edit.get("kind") in PERSON_PART_KINDS:
                    person_kw_source, _part_kw = _resolve_person_and_part(edit)
                    target = person_kw_source or str(edit.get("target") or "")
                else:
                    target = str(edit.get("target") or "")
                tokens = re.findall(r"[a-z0-9]+|[\u3400-\u9fff]+", target.lower())
                stop = {"the", "a", "an", "in", "on", "at", "of", "with", "from", "to", "that", "this", "person", "thing"}
                keywords = [word for word in tokens if word not in stop][:4]
                if keywords and not any(word in summary_lower for word in keywords):
                    add("summary_target", f"summary should mention target keywords from: {target}", hard=False)
            if any(edit.get("kind") == "background" for edit in active_edits):
                # “No part of the original setting remains …”否定的是原背景，不算矛盾；其余照查
                bg_text = "\n".join(seg for seg in _prompt_segments(narrative_text + "\n" + retention)
                                    if not re.match(r"\s*(?:no\s+part|nothing|none)\s+of\s+the\s+(?:original|old|source)\b", seg, re.I))
                if _BG_CONTRADICTION_RE.search(bg_text):
                    add("background_contradiction", "this edit replaces the background, so never say the background/setting "
                        "stays as in <Video 1>")
                shot_text = sec.get("detailed_description", "")
                if not re.search(r"\b(?:nothing|no part) of the\b", shot_text, re.I):
                    add("background_guard", "[Shot 1] must end by stating that nothing of the original setting (its walls, "
                        "ceiling, floor, furniture or lights) appears in any frame, from the first frame to the last")
                if not re.search(r"\b(?:foreground|ground|track|floor|sky|ceiling|behind)\b", shot_text, re.I):
                    add("background_layout", "[Shot 1] must describe the new setting's layout: the ground under the "
                        "characters, what is behind them, and the sky or ceiling")
            segments = _prompt_segments(text)
            # 模型已经定下来的东西，成稿不能丢：多人里选中的那位舞者、客户要保留的原帽子
            if analysis.get("edit_type") == "motion_transfer":
                for edit in active_edits:
                    if edit.get("kind") == "character_swap":
                        who = _motion_performer(edit)
                        if who != "the main performer" and who.lower() not in text.lower():
                            add("performer_lost", f"refer to the copied dancer of <Video 1> exactly as '{who}'")
            for edit in active_edits:
                if edit.get("kind") == "head_swap" and _head_swap_opts(edit)["keep_source_headwear"] and not re.search(
                        r"\b(?:own|original)\b[^.;\n]{0,60}\b(?:headwear|hats?|caps?|head\s+accessor\w*|beanies?|headbands?)\b",
                        text, re.I):
                    add("keep_headwear_lost", "the user asked to keep the original headwear; say that the person keeps her/his own "
                        "headwear")
            for edit in active_edits:
                if edit.get("kind") != "head_swap":
                    continue
                phrases = edit.get("source_head_traits") or _head_trait_phrases(edit)
                keep_hat = _head_swap_opts(edit)["keep_source_headwear"]
                becomes_low = str(edit.get("becomes") or "").lower()
                leaked = []
                for phrase in phrases:
                    for seg in segments:
                        for hit in re.finditer(re.escape(phrase), seg, re.I):
                            lead = seg[:hit.start()]
                            if phrase.lower() in becomes_low and re.search(r"<Subject \d+>[^.;]*$|<Picture \d+>[^.;]*$", lead) \
                                    and not re.search(r"\b(?:dancer|woman|man|girl|boy|person|performer)\s+(?:\S+\s+){0,2}$", lead, re.I):
                                continue          # 参考图那颗头的描述（同样是 white cap 也没关系），且不是在描述原人物
                            if keep_hat and _HEAD_ACCESSORY_RE.search(phrase) and \
                                    re.search(r"\b(?:own|original)\b(?:\s+\S+){0,3}\s*$", lead, re.I):
                                continue          # 客户要保留原来的帽子：“her own / original ... cap”可以写
                            leaked.append(phrase)
                if leaked:
                    add("head_swap_source_head", "a head swap must not describe the original head of <Video 1> ("
                        + "; ".join(sorted(set(leaked))[:3]) + "); identify that person only by clothing, body and position",
                        hard=False)
            # 只有拿参考图换装时才要求“别抄图里的人”；纯文字换装没有图可抄
            picture_outfits = [e for e in active_edits if e.get("kind") == "outfit" and _edit_picture(e)]
            missing_neg = [e for e in picture_outfits if not all(
                re.search(r"\b(?:do\s+not|don't|never)\s+copy\b[^\n.;]{0,80}\bface\b[^\n.;]{0,120}<Picture "
                          + str(_edit_picture(e)) + ">", sec.get(part, ""), re.I) for part in ("summary", "detailed_description"))]
            if missing_neg:
                picture_outfits = missing_neg
                outfit_pic = _edit_picture(picture_outfits[0])
                add(
                    "outfit_negative",
                    "outfit edit prompts must include a hard negative constraint such as "
                    f"'{_outfit_wording(_outfit_choice(picture_outfits[0]))['do_not_copy']} <Picture {outfit_pic}>; take "
                    f"{_outfit_wording(_outfit_choice(picture_outfits[0]))['take']}.' in summary and [Shot 1]",
                )
            # 客户不要的部件不能出现在参考图那一侧的正向描述里（否定列表、ignored、原视频/别的角色那句不算）
            for edit in active_edits:
                declined = []
                if edit.get("kind") == "outfit" and _edit_picture(edit):
                    declined = _outfit_choice(edit, job.request).get("declined", [])
                    declined = [p for p in declined if p not in _outfit_choice(edit, job.request).get("kept", [])]
                elif edit.get("kind") == "head_swap" and not _head_swap_opts(edit, job.request)["picture_accessories"]:
                    declined = ["head accessories"]
                if not declined:
                    continue
                pattern = re.compile("|".join(_PART_EN_RE[p] for p in declined), re.I)
                pic = _edit_picture(edit)
                several = sum(e.get("kind") == edit.get("kind") for e in active_edits) > 1
                shown = []
                for seg in segments:
                    if re.search(r"\bignored\b", seg, re.I):
                        continue
                    if several and pic is not None and f"<Picture {pic}>" not in seg:
                        continue                  # 有多个同类编辑时只看提到这张图的句子
                    cut = re.search(r"\bNOT\s+the\b|\bDo\s+not\b|\bnever\b", seg)
                    positive_part = seg[:cut.start()] if cut else seg
                    for hit in pattern.finditer(positive_part):
                        lead = positive_part[:hit.start()]
                        if re.search(r"\b(?:without|no|not)\s+(?:\S+\s+){0,2}$", lead, re.I):
                            continue
                        if re.search(r"\b(?:own|original|second|other)\s+(?:\S+\s+){0,2}$", lead, re.I):
                            continue              # “her original fox tail”说的是原视频或别的角色
                        shown.append(hit.group(0))
                if shown:
                    add("declined_part_described", "the user does not want " + _join_words(declined)
                        + f" from the picture, but the prompt still describes it ({shown[0]}); remove it from the positive description",
                        hard=False)
            subject_to_picture = {}
            subject_no = 0
            for edit in active_edits:
                if edit.get("kind") == "style":
                    continue
                source = edit.get("source") if isinstance(edit, dict) else None
                picture = source.get("picture") if isinstance(source, dict) else None
                if picture is not None:
                    subject_no += 1
                    subject_to_picture[subject_no] = int(picture)
            definition_text = sec.get("subject_definitions", "")
            definition_lines = {int(n): int(p) for n, p in re.findall(
                r"(?m)^\s*<Subject (\d+)>[^\n]*?<Picture (\d+)>[^\n]*$", definition_text)}
            for subject_no, picture_no in subject_to_picture.items():
                if definition_lines.get(subject_no) != picture_no:
                    add("subject_picture", f"<Subject {subject_no}> must be defined from <Picture {picture_no}>")
            for n in sorted({int(x) for x in re.findall(r"<Subject (\d+)>", text)}):
                if n not in subject_to_picture:
                    add("subject_picture", f"<Subject {n}> has no matching picture-backed edit")
            for unused_pic in _explicit_unused_pictures(analysis):
                if re.search(rf"<Picture {unused_pic}>", text):
                    add("unused_picture", f"<Picture {unused_pic}> is marked unused and must not be referenced")
        else:
            words = len(re.findall(r"[A-Za-z']+", body))
            low, target = (120, "150-350") if editing else ((300, "350-480") if effective_mv is not None else (300, "350-500"))
            if words < low:
                add("length", f"detailed_description has {words} English words; target {target}", hard=False)
    blocks = re.findall(r"<d>(.*?)</d>", text, re.S)
    if effective_mv is not None:
        subj_defs = sec.get("subject_definitions", "")
        for subj_no, persona_text in sorted((effective_mv.get("personas") or {}).items()):
            clean_p = str(persona_text or "").strip()
            if not clean_p:
                continue
            subj_match = re.search(rf"(?m)^\s*<Subject {int(subj_no)}>[^\n]*$", subj_defs)
            if not subj_match:
                add("mv_persona", f"subject_definitions must define <Subject {int(subj_no)}> with persona_{int(subj_no)} verbatim")
            elif clean_p not in subj_match.group(0):
                add("mv_persona", f"<Subject {int(subj_no)}> in subject_definitions must contain persona_{int(subj_no)} verbatim")

        prev_scale = str(effective_mv.get("prev_end_scale") or "").strip()
        if prev_scale and found and found[0][0] == 1:
            shot1_match = re.search(r"\[Shot 1\](.*?)(?=\[Shot 2\]|\Z)", body, re.S)
            shot1_raw = shot1_match.group(1) if shot1_match else ""
            shot1_clean = re.sub(
                r"(?:switching|breaking|contrasting|cutting|shifting|moving|transitioning)\b[^,.;]*?"
                r"\b(?:previous|prior|preceding|last|earlier)\b[^,.;]*[,.;]?",
                " ",
                shot1_raw,
                flags=re.I,
            ).strip()
            first_sent = re.split(r"[.!?\n]", shot1_clean, maxsplit=1)[0]
            prev_buckets = _mv_shot_scale_buckets(prev_scale)
            shot1_buckets = _mv_shot_scale_buckets(first_sent) or _mv_shot_scale_buckets(shot1_clean)
            if (prev_buckets and (prev_buckets & shot1_buckets)) or (
                not prev_buckets and prev_scale.lower() in first_sent.lower()
            ):
                add("mv_prev_scale", f"[Shot 1] should switch away from prev_end_scale ({prev_scale})", hard=False)

        mv_lyric_lines = list(effective_mv.get("lyric_lines") or _split_mv_lyrics(str(effective_mv.get("lyrics") or "")))
        if mv_lyric_lines:
            if not blocks:
                add("mv_lyrics", "non-empty lyrics must appear verbatim inside <d> blocks: " + " | ".join(mv_lyric_lines))
            else:
                sung_text = _extract_mv_sung_text(blocks)
                cursor = 0
                for phrase in mv_lyric_lines:
                    pos = sung_text.find(phrase, cursor)
                    if pos < 0:
                        add("mv_lyrics", f"lyric line must appear verbatim and in order inside <d>: {phrase}")
                    else:
                        cursor = pos + len(phrase)
        elif blocks or re.search(r"</?d>", text, flags=re.I):
            add("mv_no_lyrics", "lyrics is empty; <d> blocks must not appear")
    else:
        dialogue = job.dialogue if job.dialogue is not None else extract_dialogue(job.request)
        for line in dialogue:
            if not any(line in block for block in blocks):
                add("dialogue", f"this spoken line must appear verbatim inside <d>: {line}")
    for block in blocks:
        if not re.match(r"\s*\[[A-Za-z ]+\]", block):
            add("dialogue_tag", "each <d> block must start with a [Language] tag", hard=False)
    if blocks and not re.search(r"\(S\d+\)", text):
        add("speaker", "vocal events need speaker IDs such as (S1)")
    without_dialogue = re.sub(r"<d>.*?</d>", "", text.replace("“", '"').replace("”", '"').replace("「", '"').replace("」", '"'), flags=re.S)
    for quoted in re.findall(r"\"([^\"]*)\"", without_dialogue):
        if CJK.search(quoted) and quoted not in job.request:
            add("onscreen_text", f"quoted on-screen text \"{quoted}\" is not in the user request; describe that sign without quoting it")
    if len(CJK.findall(re.sub(r"\"[^\"]*\"", "", without_dialogue))) > (0 if generic_edit else 6):
        add("language", "descriptive prose must be English (Chinese only inside <d> or quoted on-screen text)")
    if "<d>" in sec.get("overall_soundscape", ""):
        add("soundscape", "overall_soundscape must not contain dialogue")
    return issues


def repair(text: str, mode: str, job: Job) -> str:
    out = (text or "").strip()
    if "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    out = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", out).strip()
    names = REF_SECTIONS if mode == "Ref2VA" else BASE_SECTIONS
    first = re.search(r"(?m)^\s*" + re.escape(names[0]) + r"\s*:", out)
    if first:
        body = out[first.start():].strip()
        found = shots(main_body(body, mode))
        fixed = instruction_line(mode, job.duration, found[-1][0] if found else 1)
        out = f"{fixed}\n\n{body}" if fixed else body

    def timecode(match: re.Match) -> str:
        seconds = float(match.group(1)) if match.group(1) else int(match.group(2)) * 60 + float(match.group(3))
        return f"At {int(seconds // 60):02d}:{seconds % 60:06.3f}"

    out = re.sub(r"At (?:(\d+(?:\.\d+)?)\s*s(?:econds)?|(\d{1,2}):(\d{1,2}(?:\.\d+)?))(?=,)", timecode, out)
    if job.cuts and any(a.role == "editing" for a in job.assets):
        stamped = list(re.finditer(r"\[Shot (\d+)\]\s*At \d{2}:\d{2}\.\d{3}", out))
        if len(stamped) == len(job.cuts):  # 镜头数对得上 → 时间戳一律对齐到源视频切点
            for m, cut in reversed(list(zip(stamped, job.cuts))):
                out = out[:m.start()] + f"[Shot {m.group(1)}] At {_mmss(cut)}" + out[m.end():]
    if mode == "FL2VA":
        out = re.sub(r"<Picture (\d)>", r"Picture \1", out)   # official FL2VA wording
    return out


# ------------------------------------------------------------------ backend call
_BUDGET_ERR_RE = re.compile(r"output budget|without a final answer|truncated|finish_reason=length", re.I)


def _complete(backend: Any, messages: list[dict[str, Any]], *, max_tokens: int, thinking_mode: str, seed: int,
              sampling: dict[str, Any] | None = None) -> tuple[str, dict]:
    backend_kind = getattr(backend, "backend_kind", "")
    is_openai_api = backend_kind == "openai_compatible"
    requested = int(max_tokens)
    if requested <= 0:
        if not is_openai_api:
            raise ValueError("max_tokens <= 0 (provider default / unlimited) is only supported for OpenAI-compatible API backends.")
        requested = 0

    if is_openai_api:
        base_kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "seed": int(seed),
            "stream": False,
        }
    else:
        base_kwargs = {
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.05,
            "seed": int(seed),
            "stream": False,
        }
    base_kwargs.update({k: v for k, v in (sampling or {}).items() if v is not None})
    # context_length is for local backends only; never forward it to OpenAI-compatible APIs.
    base_kwargs.pop("context_length", None)
    if backend_kind in {"openai_compatible", "local_bonsai_server", "local_llama_server"}:
        base_kwargs["thinking_mode"] = thinking_mode

    budget = requested
    max_attempts = 1 if budget <= 0 else 3
    for attempt in range(max_attempts):
        if attempt > 0 and not is_openai_api:
            configure = getattr(backend, "configure_chat", None)
            if callable(configure):
                base_ctx = int(getattr(backend, "n_ctx", 32768) or 32768)
                configure(max(base_ctx, budget + 8192), thinking_mode)
        call_kwargs = dict(base_kwargs, max_tokens=budget, seed=int(seed) + attempt * 17) if attempt else dict(base_kwargs, max_tokens=budget)
        try:
            raw_stream = backend.complete(**call_kwargs)
            result = _collect_stream(raw_stream)
            plain = result if isinstance(result, dict) else {}
            choices = plain.get("choices") or []
            first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            finish_reason = str(first_choice.get("finish_reason") or "").lower()
            message, usage = _extract_completion(result)
            response, reasoning, _raw = _split_message(message)
            if finish_reason == "length" or (not response.strip() and bool(reasoning.strip())):
                if budget <= 0:
                    raise OutputTruncatedError(
                        "API 响应在服务商默认输出上限处被截断（finish_reason=length 或思考耗尽预算且回答为空）；"
                        "请显式设置更大的 max_tokens 或调低 thinking_mode。"
                    )
                raise OutputTruncatedError(f"Output truncated at max_tokens={budget} (finish_reason={finish_reason or 'empty_content'}).")
            if attempt > 0:
                retries = getattr(backend, "_h3_retry_budgets", None)
                if isinstance(retries, list):
                    retries.append(budget)
            usages_sink = getattr(backend, "_h3_usages", None)
            if isinstance(usages_sink, list) and isinstance(usage, dict):
                usages_sink.append(usage)
            reasonings_sink = getattr(backend, "_h3_reasonings", None)
            if isinstance(reasonings_sink, list) and reasoning.strip():
                reasonings_sink.append(reasoning.strip())
            return response or "", usage
        except Exception as error:
            if budget <= 0 or attempt + 1 >= max_attempts or budget >= 32768 or not _BUDGET_ERR_RE.search(str(error)):
                raise
            budget = min(32768, max(budget + 1024, int(budget * 1.6)))
    raise OutputTruncatedError(f"Output truncated after budget retries (max_tokens={budget}).")


def optimize(backend: Any, job: Job, *, thinking_mode: str = "medium", max_tokens: int = 6000, seed: int = 1234,
             correction: bool = True, sampling: dict[str, Any] | None = None) -> dict[str, Any]:
    mode = resolve_mode(job)
    messages = build_messages(job, mode)
    started = time.perf_counter()
    raw, usage = _complete(backend, messages, max_tokens=max_tokens, thinking_mode=thinking_mode, seed=seed,
                           sampling=sampling)
    draft = repair(raw, mode, job)
    issues = validate(draft, mode, job)
    report = {"mode": mode, "first_issues": issues, "corrected": False, "usage": [usage]}
    hard = [i for i in issues if i["hard"]]
    if correction and hard:
        fix = ("Your draft violates these hard requirements:\n- " + "\n- ".join(i["msg"] for i in hard)
               + "\n\nRewrite the complete prompt fixing only these problems and keep everything else. Output only the final prompt.")
        raw2, usage2 = _complete(backend, messages + [{"role": "assistant", "content": draft}, {"role": "user", "content": fix}],
                                 max_tokens=max_tokens, thinking_mode=thinking_mode, seed=seed + 1, sampling=sampling)
        candidate = repair(raw2, mode, job)
        candidate_issues = validate(candidate, mode, job)
        report["usage"].append(usage2)
        if sum(i["hard"] for i in candidate_issues) < len(hard):
            draft, issues, report["corrected"] = candidate, candidate_issues, True
    report.update({"prompt": draft, "issues": issues, "seconds": time.perf_counter() - started})
    return report


EDIT_TYPES = {"character_swap", "face_swap", "head_swap", "outfit", "prop", "background", "style", "add", "remove",
              "attribute", "motion_transfer", "mixed"}
EDIT_KINDS = {"character_swap", "face_swap", "head_swap", "outfit", "prop", "background", "style", "add", "remove", "attribute"}
PERSON_PART_KINDS = frozenset({"outfit", "face_swap", "head_swap", "prop"})
_OUTFIT_NO_COPY_FACE_RE = re.compile(r"\b(?:do\s+not|don't|never)\s+copy\b[^\n.;]{0,60}\bface\b", re.I)
_EDIT_KIND_ZH = {
    "character_swap": "人物",
    "face_swap": "换脸",
    "head_swap": "换头",
    "outfit": "服装",
    "prop": "道具",
    "background": "背景",
    "style": "画风",
    "add": "新增物体",
    "remove": "移除",
    "attribute": "属性",
}
_EDIT_ANALYSIS_RULES = """Analyze only the requested changes. Output the JSON object matching the schema below, with no markdown.
Schema:
{"edit_type":"character_swap|face_swap|head_swap|outfit|prop|background|style|add|remove|attribute|motion_transfer|mixed",
 "style_source":"video|picture",
 "edits":[{"kind":"character_swap|face_swap|head_swap|outfit|prop|background|style|add|remove|attribute",
            "target":"the thing in <Video 1> that changes, as a noun phrase with stable distinguishing traits (for character_swap/outfit/face_swap: the person in <Video 1> with gender, hair, top, bottom, colors; for head_swap: gender, top, bottom and position only, never hair or headwear; for props: shape/color/where it is held); include a position cue only when two targets look alike",
            "target_person":"REQUIRED for outfit, face_swap, head_swap and prop (empty \"\" for other kinds): the person in <Video 1> whose outfit/face/head/prop is modified, described with stable visible traits (e.g. 'young woman with brown hair in a bun', 'main female dancer'); NEVER put clothing, face or prop alone here",
            "target_part":"REQUIRED for outfit, face_swap, head_swap and prop (empty \"\" for other kinds): the specific clothing, face, head or held prop on that person in <Video 1> being replaced (e.g. 'light-blue crop top and white shorts', 'facial features and face shape', 'head and brown bun hairstyle', 'wooden cane')",
            "source":{"picture":1,"text":""},
            "becomes":"a noun phrase describing ONLY what transfers from its picture or from the text — NEVER include held objects (books, cups, phones, etc.), standing pose/gestures, facial expression, or white/studio background from the reference picture unless the user explicitly requested that prop",
            "keep_from_target":"what stays from the target in <Video 1> (position, motion, pose, expression, gaze, held props, and for outfit/face/head/prop the unchanged face/hair/headwear/outfit/body)",
            "performer":"motion_transfer only, else \"\": which dancer in <Video 1> is copied, by position only, never clothes or hair (e.g. 'the second dancer from the left in the front row'); \"the main performer\" when there is one",
            "drop_parts":"outfit/head_swap only, else []: the parts of the picture the user explicitly does NOT want, from [head accessories, ears, tail, wings]; [] when the user says nothing about them",
            "add_parts":"outfit only, else []: the parts of the picture the user explicitly asks to bring along, from [head accessories, ears, tail, wings]",
            "keep_original_parts":"outfit/head_swap only, else []: parts the user explicitly wants to keep from the original person in <Video 1>, from [head accessories, ears, tail, wings]"}],
 "pictures":[{"picture":1,"use":"used|unused","note":""}],
 "unchanged":["noun phrases for every other visible character / important object that must stay as in <Video 1>"],
 "style":"ONLY the rendering of <Video 1>, one of: live-action photorealistic | 3D CG render | 2D anime illustration | stop-motion | other (<=6 words); never mention costumes, dance, culture, people or setting",
 "source_motion":"one sentence with ONLY the camera movement (fixed / handheld drift / tracking, cuts) and the main actions and interactions of the people, e.g. 'handheld wide shot, no cuts; the two women hold hands, spin, raise their arms and swap sides'; never mention the place, floor, furniture, lighting, costumes or colors",
 "notes":"<=30 words of anything ambiguous or ignored held items in pictures, may be empty"}
Use source.picture as a 1-based attached Picture number and source.text as the requested replacement only when no picture is used; with a picture, source.text is empty. Default rule: one picture serves one edit. Fallbacks: (1) only when the user explicitly asks for the same picture to serve two different roles (e.g. character + background), allow that picture in two edits of different kinds; never assign one picture to two character_swap edits; (2) if extra attached pictures are not mentioned in the request, mark them {"picture":N,"use":"unused"} in pictures and do not invent edits for them; (3) if the request asks to change more targets than pictures supplied, use source.text where described in words and leave any character without a picture or text replacement unchanged (note it in notes, e.g. "second lead has no reference picture; kept unchanged"), never reusing another character's picture; (4) if a reference picture contains multiple people or a cluttered background, pick the main/central character and note "Picture N has multiple people; picked the central one" in notes.
Distinguish face_swap vs head_swap: "换脸" / "swap face" -> face_swap (replace ONLY facial features and face shape; preserve <Video 1>'s hairstyle, headwear and outfit); "换头" / "连发型一起" / "swap head" -> head_swap (replace face + hairstyle together; preserve <Video 1>'s outfit and body); when uncertain, default to face_swap.
Becomes hygiene: in character_swap, face_swap, head_swap and outfit edits, omit any held objects (book, cup, bag, umbrella, phone, etc.), pose/gesture, expression, or white/studio background visible in the reference picture unless the user asked for that prop; if a reference picture shows a held item, note "Picture N held item ignored" in notes.
Set style_source="video" by default; set style_source="picture" only when the user explicitly asks to keep the reference picture's art/drawing/rendering style (phrases about keeping character likeness such as "保持参考图的样子" stay "video").
For a person target (except head_swap, where hair and headwear are never named), always give hair color and style, top and bottom (garment type and color), e.g. "the woman in a dark red wide-sleeved robe and red trousers with black hair in a bun"; describe what you actually see in the sheets, and if two people swap sides during the video, identify them by clothing and hair, not by left/right. Keep target and becomes non-empty; for outfit, face_swap, head_swap and prop also fill non-empty target_person (the person in <Video 1>) and target_part (their clothes/face/head/prop). Prefer the user's explicit location or clothing cue; “main character” means the most frequent / centered character, and “second lead” the next most frequent. Recognize the same target across shots by stable traits, not position. A wolf can be replaced by a human when the request maps them by character identity, not appearance. An avatar-only picture supplies only visible traits.
When the source-video role is motion (chosen on the node, or decided by you when the role is auto), set edit_type=motion_transfer and include one character_swap edit targeting the source-video performer, with source.picture = the picture the user names for the person (when the request does not say, the first picture that is not named as the background). Auto role selection already uses the user's motion-transfer wording; an explicit manual video-role choice takes precedence over wording. For motion transfer, the new pictured person performs the source performer's motion; do not describe a shot sequence. The setting of a motion transfer comes from exactly one of: (a) the place of <Video 1> — the default when the request names no setting; add no background edit; (b) the scene behind the person in the person's own picture, when the user asks to keep that picture's/photo's own background or scene (e.g. "保持图1原来的背景", "保留照片里的场景", "keep the photo's background") — add a background edit whose source.picture is that same picture (this is the allowed two-role case) and whose becomes describes only that scene, never the person; (c) another attached picture the user names as the background or scene, whatever its number — a background edit from that picture; (d) a place described only in words — a background edit with source.text.
For outfit edits, target_person is the person in <Video 1>, target_part is their original clothing in <Video 1>, and becomes describes the garments, footwear and outfit ornaments from <Picture N> as worn there, and any head accessories, ears, tail or wings that belong to the costume may stay in it — leave one out only when the user says they do not want it (and list it in drop_parts; list parts the user explicitly asks for in add_parts, and parts the user wants to keep from the original person in keep_original_parts); NEVER include the reference character's face, hair, body or held items. For face_swap, becomes describes ONLY facial features and face shape. For head_swap, becomes describes ONLY the head, face, hairstyle and (unless the user declines them) head accessories from <Picture N>, and target / target_person identify the person in <Video 1> ONLY by clothing, body and position — NEVER by hair color, hairstyle, cap, hat or other headwear, because those are being replaced and naming them pulls them back; follow the user when they name which head parts to take or leave (picture's head accessories declined → drop_parts ["head accessories"]; keep the original hat or headwear → keep_original_parts ["head accessories"]). For prop, target_person is the holder in <Video 1>, target_part is the original prop, and becomes describes ONLY the replacement prop. For style edits, the picture is a style reference only, not a subject. Keep every unrequested visible character and important object in unchanged.
For background edits, target names the whole original setting in <Video 1> as "the <kind of place> with <its 3-5 main visible parts: walls, floor, ceiling or sky, furniture, lights>" read from <Video 1> itself, and becomes describes the new setting as a complete scene layout in 30-70 words: the ground surface the characters stand on, the midground, the far background, the sky or ceiling, the time of day and light, and its art style; with a picture, read all of these from that picture. Never list the background among unchanged items when it is being replaced."""

_HELD_PROP_NOUNS = (
    "book", "books", "notebook", "cup", "mug", "teacup", "coffee", "tea", "drink", "bottle", "glass",
    "phone", "smartphone", "cellphone", "tablet", "ipad", "laptop", "camera", "umbrella", "parasol",
    "bag", "handbag", "backpack", "purse", "briefcase", "sword", "blade", "spear", "staff", "wand",
    "gun", "bow", "shield", "pen", "pencil", "brush", "folder", "clipboard", "scroll", "letter",
    "flower", "flowers", "bouquet", "rose", "microphone", "mic", "fan", "lantern", "balloon", "toy",
)
_HELD_PROP_NOUN_CN = {
    "book": "书", "books": "书", "notebook": "本", "cup": "杯", "mug": "杯", "teacup": "杯",
    "coffee": "咖啡", "tea": "茶", "bottle": "瓶", "glass": "杯", "phone": "手机", "smartphone": "手机",
    "umbrella": "伞", "parasol": "伞", "bag": "包", "handbag": "包", "backpack": "背包", "sword": "剑",
    "staff": "杖", "wand": "魔杖", "pen": "笔", "pencil": "笔", "flower": "花", "flowers": "花",
    "bouquet": "花", "microphone": "话筒", "mic": "麦克风", "fan": "扇", "lantern": "灯笼",
}
_HELD_CLAUSE_RE = re.compile(
    r"(?:,\s*|\s+and\s+|\s+while\s+|\s+)"
    r"(?:holding|carrying|clutching|gripping|cradling|reading|with)\s+"
    r"(?:a|an|the|her|his|their|one|two|some)?\s*"
    r"(?:[a-z0-9-]+\s+){0,3}"
    r"\b(" + "|".join(_HELD_PROP_NOUNS) + r")\b"
    r"(?:\s+(?:in|with|under)\s+(?:her|his|their|one|both)?\s*(?:left\s+|right\s+)?(?:hand|hands|arm|arms))?",
    re.I,
)
_STUDIO_BG_CLAUSE_RE = re.compile(
    r"(?:,\s*|\s+)(?:against|on|in\s+front\s+of|over|with)\s+"
    r"(?:a|an|the)?\s*(?:plain\s+|clean\s+|pure\s+|solid\s+|simple\s+|empty\s+)*"
    r"(?:white|gray|grey|black|neutral|studio|blank|monochrome)\s+"
    r"(?:studio\s+)?(?:background|backdrop)\b",
    re.I,
)
_POSE_CLAUSE_RE = re.compile(
    r"(?:,\s*|\s+)(?:standing|posing|posed|seated|sitting)\s+"
    r"(?:straight|still|frontally|upright|in\s+(?:a|an)\s+[a-z -]{1,24}\bpose)",
    re.I,
)
_STRIPPED_CLUTTER_BY_ANALYSIS: dict[int, list[str]] = {}


_HOLD_ANY_RE = re.compile(r"手里|手上|手中|手持(?!运镜|拍摄|镜头|跟拍|摄影|相机|摄像|手机|稳定器)|拿着|拿的|道具|\bprops?\b|\bholding\b|\bheld\s+items?\b", re.I)


def _request_mentions_prop(request: str, noun: str) -> bool:
    req = str(request or "")
    if affirmed_hit(_HOLD_ANY_RE, req):          # “手里拿的东西也要”：模型写进去的手持物都保留
        return True
    if re.search(rf"\b{re.escape(noun.lower())}\b", req.lower()):
        return True
    cn = _HELD_PROP_NOUN_CN.get(noun.lower())
    return bool(cn and cn in req)


# ───────────────────────── 自动判断视频用途用的分句/否定小工具 ─────────────────────────
# 客户原话的细节（人物/背景用哪张图、要不要尾巴帽子）由模型在分析里读，代码不二次解读。
_CLAUSE_SPLIT_RE = re.compile(r"[，。;；!！?？\n]+|(?<!\d)\.(?!\d)|,(?!\d)|(?:但是|但|不过|并且|同时|然后|另外|而是|算了)"
                              r"|并(?=把|将|给|让|且)|再(?=加|把|给|换|用)|\s+(?:but|then|and\s+then|while|instead)\s+", re.I)
_NEG_IN_CLAUSE_RE = re.compile(r"(?:不要|不想|不用|不需要|无需|别|不是|并非|忽略|去掉|去除|删掉|拿掉|没有|没|不带|不保留|不留|不加|除了|"
                               r"\bdon'?t\b|\bdo\s+not\b|\bnot\b|\bno\b|\bnever\b|\bwithout\b|\bignore\b|\bremove\b|\bdrop\b|"
                               r"\bexcept\b)", re.I)
_NEG_TAIL_RE = re.compile(r"(?:不|别|没|勿)\s*$")
_NEG_AFTER_RE = re.compile(r"^[^，。,.;；!！?？\n]{0,4}(?:不要|别要|去掉|不用|不需要|不留|拿掉|删掉|就不要|也不要|\s+removed|\s+not\s+needed)", re.I)
_REQ_PERSON_NOUN_RE = re.compile(r"人物|人|女生|女孩|女人|男生|男人|女|男|角色|主角|舞者|演员|"
                             r"\b(?:person|people|character|girl|boy|woman|man|dancer|performer|lady|guy|actor|actress)\b", re.I)
_PERSON_WORD_RE = re.compile(_REQ_PERSON_NOUN_RE.pattern + r"|她|他", re.I)
_SETTING_WORD_RE = re.compile(r"背景|场景|环境|地点|\b(?:background|scene|setting|place|backdrop)\b", re.I)
def _norm_request(text: str) -> str:
    """统一弯引号/弯撇号、全角标点，去掉 “No. 2” 这类图号里的句点。"""
    text = str(text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("（", "(").replace("）", ")")
    return re.sub(r"\b(no)\.\s*(?=\d)", r"\1 ", text, flags=re.I)


def _clauses(text: str) -> list[str]:
    return [c for c in _CLAUSE_SPLIT_RE.split(_norm_request(text)) if c and c.strip()]


def _negated(clause: str, start: int, end: int) -> bool:
    """命中词在本分句里前面有否定词（含紧贴的单字“不/别/没”），或紧跟着“不要/去掉”。"""
    before = clause[:start]
    return bool(_NEG_IN_CLAUSE_RE.search(before) or _NEG_TAIL_RE.search(before) or _NEG_AFTER_RE.search(clause[end:]))


def affirmed_hit(pattern: re.Pattern, text: str) -> re.Match | None:
    """第一个没被否定的命中（「不要给我做动作迁移」「Don't do motion transfer」都不算）。"""
    for clause in _clauses(text):
        for hit in pattern.finditer(clause):
            if not _negated(clause, hit.start(), hit.end()):
                return hit
    return None


def _setting_only_clause(clause: str) -> bool:
    """这个分句改的只是背景（「背景改成海边」「replace the background with picture 2」）。"""
    s = _SETTING_WORD_RE.search(clause)
    if not s:
        return False
    person = _PERSON_WORD_RE.search(clause)
    edit = _EDIT_WORDS.search(clause)
    return not person or (edit is not None and s.start() < edit.start() and person.start() > edit.start())


_PART_NAMES = ("head accessories", "ears", "tail", "wings")
_PART_ALIASES = (
    ("head accessories", r"\bhead\s*(?:accessor\w*|band|piece|dress|wear)s?\b|\b(?:hats?|caps?|beanies?|helmets?|crowns?|tiaras?|"
                         r"bonnets?|berets?|veils?|hoods?)\b|\bhair\s*(?:pin|clip|bow|ribbon)s?\b|头饰|帽"),
    ("ears", r"\bears?\b|耳朵"), ("tail", r"\btails?\b|尾巴|鱼尾"), ("wings", r"\bwings?\b|翅膀"),
)


def _norm_parts(value: Any) -> list[str]:
    """模型填的部件列表 → 固定的四个名字（一项里写了几样都认，认不出的丢掉）。"""
    items = value if isinstance(value, list) else ([value] if isinstance(value, str) and value.strip() else [])
    found = {n for item in items for n, pat in _PART_ALIASES if re.search(pat, str(item), re.I)}
    return [n for n in _PART_NAMES if n in found]


def _apply_part_fields(edit: dict[str, Any]) -> None:
    """把模型填的 drop_parts / add_parts / keep_original_parts 换成渲染用的选择。"""
    if not isinstance(edit.get("kind"), str) or edit.get("kind") not in {"outfit", "head_swap"}:
        for key in ("drop_parts", "add_parts", "keep_original_parts"):
            edit.pop(key, None)
        return
    drop, add, keep = (_norm_parts(edit.get(k)) for k in ("drop_parts", "add_parts", "keep_original_parts"))
    edit["drop_parts"], edit["add_parts"], edit["keep_original_parts"] = drop, add, keep
    if edit.get("kind") == "outfit":
        declined = drop + [n for n in keep if n not in drop and n not in add]
        edit["extras_choice"] = {"wanted": [n for n in add if n not in declined], "declined": declined, "kept": keep}
    if edit.get("kind") == "head_swap":
        edit["head_options"] = {"picture_accessories": "head accessories" not in drop and "head accessories" not in keep,
                                "keep_source_headwear": "head accessories" in keep}


def _outfit_choice(edit: dict[str, Any], request: str = "") -> dict[str, list[str]]:
    choice = edit.get("extras_choice") if isinstance(edit, dict) else None
    if isinstance(choice, dict) and isinstance(choice.get("wanted"), list) and isinstance(choice.get("declined"), list):
        order = list(_PART_NAMES)
        return {k: [n for n in order if n in choice.get(k, [])] for k in ("wanted", "declined", "kept")}
    return {"wanted": [], "declined": [], "kept": []}


def _join_words(words: list[str], last: str = "and") -> str:
    words = [w for w in words if w]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + f" {last} " + words[-1]


def _outfit_wording(choice: dict[str, list[str]]) -> dict[str, str]:
    """换装各处要用的措辞。默认只排除脸、头发、体型、手持物；客户不要的附件加进排除，点名要的写明带上。"""
    wanted, declined = list(choice.get("wanted", [])), list(choice.get("declined", []))
    excluded = ["face", "hair", *declined, "body", "held items"]
    take = "the outfit as worn there" + (f", including the {_join_words(wanted)}" if wanted else "")
    kept = list(choice.get("kept", []))
    keeps = ["face", "hair", *kept, "body", "motion"]
    return {
        "take": take,
        "do_not_copy": f"Do not copy the {_join_words(excluded, 'or')} of",
        "subject_not": f"NOT the character's {_join_words(excluded, 'or')}.",
        "picture_ignored": f"the person, {_join_words(excluded)} in",
        "keeps": _join_words(keeps),
        "carried": (f" The {_join_words(wanted)} of the character in that picture "
                    f"{'are' if len(wanted) > 1 or wanted[0].endswith('s') else 'is'} carried over too, as the user asked."
                    if wanted else ""),
    }


def _sanitize_becomes(becomes: str, kind: str, request: str = "") -> tuple[str, list[str]]:
    """Strip unrequested reference-picture held props, poses, and studio backgrounds from `becomes`."""
    text = str(becomes or "").strip()
    if kind not in {"character_swap", "face_swap", "head_swap", "outfit"} or not text:
        return text, []
    stripped: list[str] = []

    def _strip_held(match: re.Match) -> str:
        noun = match.group(1)
        if _request_mentions_prop(request, noun):
            return match.group(0)
        stripped.append(match.group(0).strip(" ,"))
        return ""

    cleaned = _HELD_CLAUSE_RE.sub(_strip_held, text)
    for pattern in (_STUDIO_BG_CLAUSE_RE, _POSE_CLAUSE_RE):
        for found in pattern.finditer(cleaned):
            stripped.append(found.group(0).strip(" ,"))
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;")
    return (cleaned or text), stripped


def _active_edits(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    edits = analysis.get("edits", [])
    if not isinstance(edits, list):
        return []
    return [edit for edit in edits if isinstance(edit, dict) and str(edit.get("use") or "used").lower() != "unused"]


def _explicit_unused_pictures(analysis: dict[str, Any]) -> list[int]:
    unused: list[int] = []
    for item in analysis.get("pictures", []) if isinstance(analysis.get("pictures"), list) else []:
        if isinstance(item, dict) and str(item.get("use") or "").lower() == "unused":
            pic = item.get("picture")
            if isinstance(pic, int) and not isinstance(pic, bool):
                unused.append(pic)
    for edit in analysis.get("edits", []) if isinstance(analysis.get("edits"), list) else []:
        if isinstance(edit, dict) and str(edit.get("use") or "").lower() == "unused":
            source = edit.get("source")
            pic = source.get("picture") if isinstance(source, dict) else edit.get("picture")
            if isinstance(pic, int) and not isinstance(pic, bool):
                unused.append(pic)
    return sorted(set(unused))


def _first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    source = str(text or "")
    for index, char in enumerate(source):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no complete JSON object found")


_STYLE_CONTENT_RE = re.compile(
    r"danc|costume|outfit|dress|robe|chinese|japanese|korean|traditional|folk|performance|perform|hall|stage|room|"
    r"woman|women|girl|man\b|men\b|people|dancer|singer|舞|服|表演|中国|古风|汉服", re.I)


def _normalize_style(style: Any) -> str:
    """风格栏只留渲染方式：短且不含内容词就原样保留，否则归到几种渲染方式。"""
    text = str(style or "").strip()
    low = text.lower()
    if text and len(text.split()) <= 6 and not _STYLE_CONTENT_RE.search(text):
        return text
    if re.search(r"live|photo|real|film|cinemat|真人|实拍", low):
        return "live-action photorealistic"
    if re.search(r"anime|2d|cartoon|illustrat|manga|二次元|动漫|动画", low):
        return "2D anime illustration"
    if re.search(r"3d|cg|render|三维", low):
        return "3D CG render"
    if re.search(r"stop.?motion|clay", low):
        return "stop-motion"
    return " ".join(text.split()[:6]) or "visual style"


_DEFAULT_KEEP = {
    "character_swap": "position, route, scale, pose, every movement, expression, gaze and held props",
    "background": "the characters, their positions and motion, the camera and framing",
    "outfit": "face, hair, head accessories, body and motion",
    "face_swap": "hairstyle, headwear, outfit, body and motion",
    "head_swap": "outfit, body and motion",
    "prop": "face, hair, outfit, grip, position and motion",
}


def parse_edit_analysis(text: str, picture_count: int, request: str = "", video_role: str = "") -> dict[str, Any]:
    """Parse the first complete JSON object and validate the two-pass edit-analysis contract."""
    value = _first_json_object(text)
    errors: list[str] = []
    edit_type = value.get("edit_type")
    if not isinstance(edit_type, str) or edit_type not in EDIT_TYPES:
        errors.append(f"edit_type must be one of {sorted(EDIT_TYPES)}")
    if isinstance(value, dict):
        value["style"] = _normalize_style(value.get("style"))
        if "source_motion" in value and not isinstance(value["source_motion"], str):
            value["source_motion"] = ""
    if "style_source" in value and value["style_source"] is not None:
        style_source = value["style_source"]
        if not isinstance(style_source, str) or style_source.lower() not in STYLE_SOURCES:
            errors.append(f"style_source must be one of {sorted(STYLE_SOURCES)}")
        else:
            value["style_source"] = style_source.lower()
    edits = value.get("edits")
    if not isinstance(edits, list) or not edits:
        errors.append("edits must be a non-empty array")
        edits = []
    used_picture_kinds: dict[int, list[str]] = {}
    active_count = 0
    stripped_details: list[str] = []
    for index, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            errors.append(f"edits[{index}] must be an object")
            continue
        use_flag = edit.get("use")
        if use_flag is not None and (not isinstance(use_flag, str) or use_flag.lower() not in {"used", "unused"}):
            errors.append(f"edits[{index}].use must be 'used' or 'unused'")
            continue
        if isinstance(use_flag, str) and use_flag.lower() == "unused":
            source = edit.get("source")
            pic = source.get("picture") if isinstance(source, dict) else edit.get("picture")
            if pic is not None:
                if isinstance(pic, bool) or not isinstance(pic, int) or pic < 1 or pic > int(picture_count):
                    errors.append(f"edits[{index}] unused picture={pic} exceeds {picture_count} attached picture(s)")
            continue
        active_count += 1
        kind = edit.get("kind")
        if not isinstance(kind, str) or kind not in EDIT_KINDS:
            errors.append(f"edits[{index}].kind is invalid")
            kind = ""
        for pfield in ("target_person", "target_part"):
            if edit.get(pfield) is not None and not isinstance(edit.get(pfield), str):
                errors.append(f"edits[{index}].{pfield} must be a string")
        if kind in PERSON_PART_KINDS and not (isinstance(edit.get("target"), str) and edit["target"].strip()):
            tp = str(edit.get("target_person") or "").strip() if isinstance(edit.get("target_person"), str) else ""
            tpart = str(edit.get("target_part") or "").strip() if isinstance(edit.get("target_part"), str) else ""
            if kind == "prop" and tpart:
                edit["target"] = tpart
            elif tp:
                edit["target"] = tp
            elif tpart:
                edit["target"] = tpart
        for key in ("target", "becomes"):
            if not isinstance(edit.get(key), str) or not edit[key].strip():
                errors.append(f"edits[{index}].{key} must be a non-empty string")
        if kind == "head_swap":
            # 清理之前先记下原视频的头发/帽子描述，校验时用来查成稿有没有写回去
            edit["source_head_traits"] = _head_trait_phrases(edit)
            if isinstance(edit.get("target"), str):
                edit["target"] = _strip_head_traits(edit["target"])
        if kind in PERSON_PART_KINDS and isinstance(edit.get("target"), str) and edit["target"].strip():
            if isinstance(edit.get("target_person"), (str, type(None))) and isinstance(edit.get("target_part"), (str, type(None))):
                resolved_person, resolved_part = _resolve_person_and_part(edit)
                edit["target_person"] = resolved_person
                edit["target_part"] = resolved_part
        _apply_part_fields(edit)
        if kind in {"outfit", "head_swap"} and isinstance(edit.get("becomes"), str):
            edit["becomes"] = _drop_declined_parts(edit)
        if kind and isinstance(edit.get("becomes"), str) and edit["becomes"].strip():
            cleaned_becomes, removed = _sanitize_becomes(edit["becomes"], kind, request)
            edit["becomes"] = cleaned_becomes
            stripped_details.extend(removed)
        if not isinstance(edit.get("keep_from_target"), str) or not edit["keep_from_target"].strip():
            # 小模型常把这一栏留空：按类型给默认含义，不算失败
            edit["keep_from_target"] = _DEFAULT_KEEP.get(kind, _DEFAULT_KEEP["character_swap"])
        source = edit.get("source")
        if isinstance(source, dict):
            pic_raw = source.get("picture")
            if isinstance(pic_raw, str) and pic_raw.strip().isdigit():
                source["picture"] = int(pic_raw.strip())
            elif isinstance(pic_raw, str) and not pic_raw.strip():
                source["picture"] = None
            if source.get("picture") == 0 and isinstance(source.get("text"), str) and source["text"].strip():
                source["picture"] = None    # 小模型用 0 表示「没用图、用文字」
            if source.get("text") is None and source.get("picture") is not None:
                source["text"] = ""
        if not isinstance(source, dict):
            errors.append(f"edits[{index}].source must be an object")
            continue
        if "picture" not in source or "text" not in source:
            errors.append(f"edits[{index}].source must contain picture and text fields")
        picture = source.get("picture")
        replacement_text = source.get("text")
        if picture is not None:
            if isinstance(picture, bool) or not isinstance(picture, int):
                errors.append(f"edits[{index}].source.picture must be an integer or null")
            elif picture < 1 or picture > int(picture_count):
                errors.append(f"edits[{index}].source.picture={picture} exceeds {picture_count} attached picture(s)")
            else:
                prior_kinds = used_picture_kinds.setdefault(picture, [])
                if kind and kind in prior_kinds:
                    errors.append(f"Picture {picture} cannot be assigned to multiple {kind} edits")
                elif kind:
                    prior_kinds.append(kind)
            if replacement_text != "":
                errors.append(f"edits[{index}].source.text must be empty when a picture is used")
        elif not isinstance(replacement_text, str) or not replacement_text.strip():
            errors.append(f"edits[{index}].source.text must describe a text-only replacement")
    if edits and active_count == 0:
        errors.append("edits must contain at least one active edit")
    if "pictures" in value and value["pictures"] is not None:
        pictures_meta = value["pictures"]
        if not isinstance(pictures_meta, list):
            errors.append("pictures must be an array when provided")
        else:
            for idx, item in enumerate(pictures_meta, 1):
                if not isinstance(item, dict):
                    errors.append(f"pictures[{idx}] must be an object")
                    continue
                pic = item.get("picture")
                use_val = item.get("use", "used")
                if isinstance(pic, bool) or not isinstance(pic, int) or pic < 1 or pic > int(picture_count):
                    errors.append(f"pictures[{idx}].picture={pic} exceeds {picture_count} attached picture(s)")
                if not isinstance(use_val, str) or use_val.lower() not in {"used", "unused"}:
                    errors.append(f"pictures[{idx}].use must be 'used' or 'unused'")
    used_pics = {_edit_picture(e) for e in _active_edits(value) if isinstance(e, dict)} - {None}
    marked_unused = [item.get("picture") for item in (value.get("pictures") if isinstance(value.get("pictures"), list) else [])
                     if isinstance(item, dict) and str(item.get("use") or "").lower() == "unused"]
    marked_unused += [_edit_picture(e) for e in (edits if isinstance(edits, list) else [])
                      if isinstance(e, dict) and str(e.get("use") or "").lower() == "unused"]
    for pic in marked_unused:
        if isinstance(pic, int) and not isinstance(pic, bool) and pic in used_pics:
            errors.append(f"Picture {pic} is used by an edit but marked unused")
    unchanged = value.get("unchanged")
    if not isinstance(unchanged, list) or any(not isinstance(item, str) or not item.strip() for item in unchanged):
        errors.append("unchanged must be an array of non-empty noun phrases")
    style = value.get("style")
    if not isinstance(style, str) or len(style.split()) > 15:
        errors.append("style must be a string of at most 15 words")
    notes = value.get("notes")
    if not isinstance(notes, str) or len(notes.split()) > 30:
        errors.append("notes must be a string of at most 30 words")
    if edit_type == "motion_transfer":
        active = _active_edits(value)
        motion_source = next((edit for edit in active if edit.get("kind") == "character_swap"), None)
        source = motion_source.get("source", {}) if motion_source else {}
        if not isinstance(source, dict) or not isinstance(source.get("picture"), int):
            errors.append("motion_transfer needs a character_swap sourced from the person's picture")
    motion_text = str(value.get("source_motion") or "")
    for edit in _active_edits(value):
        if edit.get("kind") == "head_swap" and motion_text:
            extra = _head_phrases_in_text(motion_text)
            edit["source_head_traits"] = sorted(set(edit.get("source_head_traits", []) + extra), key=len, reverse=True)
    if not errors:                          # 先过格式，再查跟客户原话对不对得上
        errors.extend(_request_conflicts(value, request, picture_count, video_role))
    if errors:
        raise ValueError("; ".join(errors))
    if stripped_details:
        _STRIPPED_CLUTTER_BY_ANALYSIS[id(value)] = stripped_details
    return value


def _request_conflicts(analysis: dict[str, Any], request: str, picture_count: int, video_role: str = "") -> list[str]:
    """只查结构性矛盾：视频被选成动作参考时，分析必须是动作迁移。客户原话由模型读，代码不二次解读。"""
    role = str(video_role).lower()
    if role == "motion" and analysis.get("edit_type") != "motion_transfer":
        return ["the source video is the motion reference, so edit_type must be motion_transfer"]
    if role == "editing" and analysis.get("edit_type") == "motion_transfer":
        return ["the source video was chosen as the video to edit, so do not use motion_transfer"]
    return []


def _edit_picture(edit: dict[str, Any]) -> int | None:
    source = edit.get("source", {})
    picture = source.get("picture") if isinstance(source, dict) else None
    if picture is None and isinstance(edit, dict):
        picture = edit.get("picture")
    if isinstance(picture, str) and picture.strip().isdigit():
        return int(picture.strip())
    return picture if isinstance(picture, int) and not isinstance(picture, bool) else None


def _edit_subjects(analysis: dict[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    next_subject = 0
    for edit_index, edit in enumerate(_active_edits(analysis)):
        if edit.get("kind") == "style":
            continue
        picture = _edit_picture(edit)
        if picture is not None:
            next_subject += 1
            result[edit_index] = next_subject
    return result


def _without_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", str(text or "").strip(), flags=re.I)


_PERSON_NOUN_RE = re.compile(
    r"\b(?:woman|women|man|men|girl|girls|boy|boys|person|people|character|characters|dancer|dancers|singer|singers|"
    r"performer|performers|actor|actress|lady|gentleman|guy|kid|child|children|lead|protagonist|hero|heroine|"
    r"swordsman|warrior|knight|maid|bride|groom|student|teacher|doctor|worker|chef|driver|rider|player|speaker|host|"
    r"model|skater|boxer|fighter|runner|swimmer|climber|cyclist|musician|guitarist|drummer|pianist|violinist|rapper|"
    r"idol|streamer|vlogger|presenter|anchor|officer|soldier|guard|nurse|waiter|waitress|clerk|monk|priest|elder|"
    r"teenager|teen|youth|baby|toddler|human|adult|wolf|sheep|dog|cat|bear|rabbit|fox|creature|animal|figure)\b",
    re.I,
)
_CLOTHING_OR_PART_ONLY_RE = re.compile(
    r"\b(?:clothes|clothing|outfit|garment|garments|attire|dress|gown|shirt|t-shirt|tee|top|blouse|jacket|coat|"
    r"hoodie|sweater|cardigan|vest|suit|uniform|costume|apron|skirt|shorts|pants|trousers|jeans|leggings|socks|"
    r"stockings|shoes|boots|heels|sneakers|footwear|hat|cap|headband|headpiece|headwear|face|features|head|"
    r"hair|hairstyle|prop|cane|sword|blade|staff|wand|baton|book|books|cup|mug|umbrella|parasol|fan|microphone|"
    r"mic|bag|handbag|backpack|lantern|flower|bouquet)\s*$",
    re.I,
)
_POSSESSIVE_TARGET_RE = re.compile(r"^(?P<owner>.+?)(?:'s|’s)\s+(?P<part>.+)$")
_PART_OF_OWNER_RE = re.compile(
    r"^(?:the\s+)?(?P<part>(?:clothes|clothing|outfit|dress|shirt|top|blouse|jacket|coat|suit|uniform|costume|"
    r"skirt|shorts|pants|trousers|shoes|footwear|face|facial\s+features(?:\s+and\s+face\s+shape)?|"
    r"head(?:\s+and\s+hair(?:style)?)?|hair(?:style)?|prop|item|object|book|cup|sword|cane|baton|staff|wand|"
    r"umbrella|fan|microphone|mic|bag|hat|weapon)(?:\s+and\s+[a-z -]+)?)\s+"
    r"(?:of|on|worn\s+by|held\s+by|carried\s+by)\s+(?P<owner>.+)$",
    re.I,
)
_PLURAL_PERSON_RE = re.compile(
    r"\b(?:women|men|girls|boys|people|characters|dancers|singers|performers|actors|kids|children|warriors|"
    r"soldiers|players|students|wolves|animals|creatures|both|two|three)\b",
    re.I,
)
_MALE_PERSON_RE = re.compile(
    r"\b(?:man|boy|male|gentleman|guy|father|dad|brother|son|husband|groom|king|prince|swordsman|actor|hero|"
    r"waiter|monk|priest|he|him|his)\b",
    re.I,
)
_NEUTRAL_CREATURE_RE = re.compile(
    r"\b(?:wolf|sheep|dog|cat|bear|rabbit|fox|creature|animal|robot|mascot)\b",
    re.I,
)
_DEFAULT_PART_BY_KIND = {
    "outfit": "outfit",
    "face_swap": "facial features and face shape",
    "head_swap": "head and hairstyle",
    "prop": "held prop",
}


def _split_person_and_part(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    if not text:
        return "", ""
    m_pos = _POSSESSIVE_TARGET_RE.match(text)
    if m_pos:
        owner, rest = m_pos.group("owner").strip(), m_pos.group("part").strip()
        if owner and rest and not _PERSON_NOUN_RE.search(rest):
            return owner, rest
    m_of = _PART_OF_OWNER_RE.match(text)
    if m_of:
        owner, part = m_of.group("owner").strip(), m_of.group("part").strip()
        if owner and part:
            return owner, part
    return text, ""


# 换头：原视频人物的头发、帽子、头饰都是要被换掉的东西，提示词里一旦描述它们，模型就会照着画回来。
# 所以认人只用衣服、体型和位置；头部部件默认整颗换，客户说了哪部分要/不要再按客户的来。
_HEAD_TRAIT_WORD_RE = re.compile(
    r"\b(?:hair|haired|hairstyle|bun|ponytail|braids?|pigtails?|bangs|fringe|curls|curly|wavy|locks|afro|dreadlocks|updo|"
    r"mohawk|crew\s*cut|buzz\s*cut|cap|hat|beanie|helmet|hood|headband|headpiece|"
    r"headwear|bandana|headscarf|hairpin|hair\s*clip|crown|tiara|ribbon|bow)\b", re.I)
_HEAD_ADJ_RE = re.compile(r"\b(?:blonde|blond|brunette|redheaded|redhead|auburn|[a-z]+-haired|bald|capped|curly|wavy)\s+", re.I)
def _strip_head_traits(person: str) -> str:
    """去掉人物描述里的头发、帽子、头饰（“the dancer's head with brown hair and a white cap” → “the dancer”）。"""
    text = re.sub(r"(?:'s|’s)\s+head\b", "", str(person or ""), flags=re.I)
    text = _HEAD_ADJ_RE.sub("", text)
    pieces = re.split(r"(\s*,\s*|\s+(?:with|in|wearing|and|who\s+has|having)\s+)", text, flags=re.I)
    out, pending = pieces[0], None
    if _HEAD_TRAIT_WORD_RE.search(out):
        return "the main performer"
    for sep, chunk in zip(pieces[1::2], pieces[2::2]):
        if _HEAD_TRAIT_WORD_RE.search(chunk):
            pending = pending or sep
            continue
        if pending and sep.strip().lower() in {"and", ","}:
            sep = pending
        pending = None
        out += sep + chunk
    out = re.sub(r"\s{2,}", " ", out).strip(" ,;")
    return out if _PERSON_NOUN_RE.search(out) else "the main performer"


def _strip_head_words(text: str) -> str:
    """整句里去掉描述头发/帽子的短语（“the woman with short gray hair spins” → “the woman spins”）。"""
    text = _HEAD_ADJ_RE.sub("", str(text or ""))
    text = re.sub(r"\s*,?\s*\b(?:with|in|wearing|who\s+has)\s+(?:a|an|the|her|his|their)?\s*(?:[\w-]+\s+){0,3}?"
                  r"(?:hair|hairstyle|bun|ponytail|braids?|pigtails?|bangs|curls|locks|cap|hat|beanie|helmet|hood|headband|"
                  r"headpiece|bandana)\b", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def _head_phrases_in_text(text: str) -> list[str]:
    """一段英文里描述头发/帽子的短语（“short gray hair”“a white cap”）。"""
    found = re.findall(r"(?:[\w-]+\s+){0,3}(?:hair|hairstyle|bun|ponytail|braids?|pigtails?|bangs|curls|locks|cap|hat|beanie|"
                       r"helmet|hood|headband|headpiece|bandana)\b", str(text or ""), re.I)
    out = []
    for phrase in found:
        phrase = re.sub(r"^(?:with|in|wearing|has|a|an|the|her|his|their|and)\s+", "", phrase.strip(), flags=re.I)
        phrase = re.sub(r"^(?:with|in|wearing|has|a|an|the|her|his|their|and)\s+", "", phrase, flags=re.I)
        # 只记带颜色或帽饰的具体说法；“long hair”这种泛称新头发也会用，不记
        specific = re.search(r"\b(?:black|brown|blonde?|gr[ae]y|white|red|blue|pink|purple|green|silver|golden|auburn|dark|"
                             r"light|ginger|cap|hat|beanie|helmet|hood|headband|headpiece|bandana)\b", phrase, re.I)
        if len(phrase) >= 6 and specific:
            out.append(phrase)
    return out


def _head_trait_phrases(edit: dict[str, Any]) -> list[str]:
    """分析里原视频人物的头发、帽子描述（“brown hair”“a white cap”），用来检查成稿有没有写回去。"""
    found = []
    for key in ("target", "target_person", "target_part"):
        text = str(edit.get(key) or "")
        for chunk in re.split(r"\s*,\s*|\s+(?:with|in|wearing|and|who\s+has|having)\s+|(?:'s|’s)\s+", text, flags=re.I):
            chunk = re.sub(r"^(?:a|an|the|her|his|their)\s+", "", chunk.strip(), flags=re.I)
            if _HEAD_TRAIT_WORD_RE.search(chunk) and len(chunk) >= 6 and not re.fullmatch(r"(?:head|hair|hairstyle)(?:\s+and\s+hairstyle)?", chunk, re.I):
                found.append(chunk)
    return sorted(set(found), key=len, reverse=True)


_HEAD_ACC_EN = (r"\b(?:head\s*(?:band|piece|dress|wear|scarf)s?|hats?|caps?(?![\s-]+sleeves?)|beanies?|bonnets?|berets?|crowns?|tiaras?|veils?|"
                r"helmets?|hoods?|bandanas?|hair\s*(?:ribbon|bow|pin|clip|band|ornament|accessor\w*)s?|frilly\s+band|maid\s+headpiece|"
                r"head\s+accessor\w*)\b")
_PART_EN_RE = {
    "head accessories": _HEAD_ACC_EN,
    "ears": r"\b(?:[a-z-]+\s+)?ears?\b",
    "tail": r"\b(?:[a-z-]+\s+)?tails?\b",
    "wings": r"\b(?:[a-z-]+\s+)?wings?\b",
}


def _motion_performer(edit: dict[str, Any]) -> str:
    """动作迁移里称呼源舞者：用模型给的位置说法（performer 字段），不描述衣服帽子（会被画回来）。"""
    locator = str(edit.get("performer") or "").strip().rstrip(".")
    if locator and not re.search(r"\b(?:hair|cap|hat|jacket|shirt|dress|skirt|pants|wearing|in\s+a)\b|\btop\b(?![\s-]+(?:left|right))(?<!-top)",
                                 locator, re.I):
        return locator if re.match(r"(?:the|a)\b", locator, re.I) else "the " + locator
    return "the main performer"


def _drop_parts(desc: str, parts: list[str]) -> str:
    """从正向描述里删掉客户不要的部件：只删那个短语（连同前面的 a/the 和最多两个修饰词），
    「without a tail」本来就是不要，不删。删完只剩连接词时收拾干净。"""
    text = str(desc or "")
    if not parts:
        return text
    bare = {"head accessories": _HEAD_ACC_EN, "ears": r"\bears?\b", "tail": r"\btails?\b", "wings": r"\bwings?\b"}
    core = "|".join(f"(?:{bare[p]})" for p in parts if p in bare)
    word = r"\b(?!(?:with|and|without|no|not|under|in|wearing|plus|or)\b)[\w-]+\s+"
    phrase = re.compile(r"(?P<lead>\s*(?:,|\band\b|\bwith\b|\bplus\b|\bwearing\b|\bunder\b|\bin\b)?\s*)"
                        r"(?P<body>\b(?:(?:a|an|the|her|his|its)\s+)?(?:" + word + r"){0,2}?(?:" + core + r"))", re.I)

    def cut(m: re.Match) -> str:
        if re.search(r"\b(?:without|no|not)\s+(?:\S+\s+){0,2}$", text[:m.start("body")], re.I) or \
                re.search(r"\b(?:without|no|not|with|and)\b", m.group("body"), re.I):
            return m.group(0)
        return ""

    out = phrase.sub(cut, text)
    out = re.sub(r"\s+(?:,|and|with|plus|wearing|under|in)\s*$", "", out.strip(), flags=re.I)
    out = re.sub(r"^\s*(?:,|and|with|plus)\s+", "", out, flags=re.I)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+,", ",", out)).strip(" ,;")


def _drop_declined_parts(edit: dict[str, Any]) -> str:
    """换装去掉客户不要的附件；换头客户不要图里头饰时去掉头饰。删空时给不带附件的默认说法。"""
    becomes = str(edit.get("becomes") or "")
    picture = _edit_picture(edit)
    where = f" shown in <Picture {picture}>" if picture else ""
    if edit.get("kind") == "outfit":
        declined = _outfit_choice(edit).get("declined", [])
        return _drop_parts(becomes, declined) or f"the outfit{where}" if declined else becomes
    if edit.get("kind") == "head_swap" and not _head_swap_opts(edit)["picture_accessories"]:
        return _drop_parts(becomes, ["head accessories"]) or f"the face and hairstyle{where}"
    return becomes


_HEAD_ACCESSORY_RE = re.compile(_HEAD_ACC_EN, re.I)


def _head_swap_opts(edit: dict[str, Any], request: str = "") -> dict[str, bool]:
    opts = edit.get("head_options") if isinstance(edit, dict) else None
    if isinstance(opts, dict) and {"picture_accessories", "keep_source_headwear"} <= set(opts):
        return {k: bool(opts[k]) for k in ("picture_accessories", "keep_source_headwear")}
    return {"picture_accessories": True, "keep_source_headwear": False}


def _resolve_person_and_part(edit: dict[str, Any]) -> tuple[str, str]:
    """Return `(target_person, target_part)` ensuring the subject is a person, never clothing or a prop."""
    kind = str(edit.get("kind") or "").strip()
    default_part = _DEFAULT_PART_BY_KIND.get(kind, "part")
    raw_person = str(edit.get("target_person") or "").strip() if isinstance(edit.get("target_person"), str) else ""
    raw_part = str(edit.get("target_part") or "").strip() if isinstance(edit.get("target_part"), str) else ""
    raw_target = str(edit.get("target") or "").strip() if isinstance(edit.get("target"), str) else ""

    person, split_from_person = _split_person_and_part(raw_person)
    if split_from_person and not raw_part:
        raw_part = split_from_person

    target_owner, split_from_target = _split_person_and_part(raw_target)
    if not person:
        if split_from_target:
            person = target_owner
            if not raw_part:
                raw_part = split_from_target
        elif raw_target and _PERSON_NOUN_RE.search(raw_target):
            person = raw_target
        elif kind == "prop" and (raw_target or raw_part):
            part_clean = _without_article(raw_part or raw_target)
            person = f"the character holding the {part_clean}"
            if not raw_part:
                raw_part = raw_target
        else:
            if raw_target and not raw_part:
                raw_part = raw_target
            person = "the main character"
    elif not _PERSON_NOUN_RE.search(person) and _CLOTHING_OR_PART_ONLY_RE.search(person):
        if target_owner and _PERSON_NOUN_RE.search(target_owner):
            if not raw_part:
                raw_part = person
            person = target_owner
        elif kind == "prop":
            part_clean = _without_article(raw_part or person)
            if not raw_part:
                raw_part = person
            person = f"the character holding the {part_clean}"
        else:
            if not raw_part:
                raw_part = person
            person = "the main character"

    part = raw_part or split_from_target or default_part
    if kind == "head_swap":
        person, part = _strip_head_traits(person), "head"
    return person.strip(), part.strip()


def _possessive_pronoun(person: str) -> str:
    text = str(person or "")
    if _PLURAL_PERSON_RE.search(text):
        return "their"
    if _MALE_PERSON_RE.search(text):
        return "his"
    if _NEUTRAL_CREATURE_RE.search(text) and not re.search(r"\b(?:woman|girl|female|lady|man|boy|male)\b", text, re.I):
        return "its"
    return "her"


def _strip_picture_provenance(text: str) -> str:
    cleaned = re.sub(
        r"(?:,\s*)?(?:as\s+)?(?:shown\s+in|seen\s+in|from|in|of)\s+<Picture\s+\d+>\.?\s*$",
        "",
        str(text or "").strip(),
        flags=re.I,
    ).strip(" ,;.")
    return cleaned or str(text or "").strip()


def _changed_target_phrase(edit: dict[str, Any]) -> str:
    if edit.get("kind") in PERSON_PART_KINDS:
        person, part = _resolve_person_and_part(edit)
        p_clean = _without_article(person)
        part_clean = _without_article(part)
        if p_clean.lower() in part_clean.lower():
            return part_clean
        if re.search(r"\s(?:in|with|wearing|holding|on|at|from)\s", p_clean, re.I):
            return f"the {part_clean} of the {p_clean}"
        return f"{p_clean}'s {part_clean}"
    return _without_article(str(edit.get("target", "")))


def _edit_sentence(edit: dict[str, Any], subject: int | None, request: str = "") -> str:
    kind = edit["kind"]
    target = str(edit["target"]).strip()
    target_no_article = _without_article(target)
    becomes, _ = _sanitize_becomes(str(edit["becomes"]).strip(), kind, request)
    picture = _edit_picture(edit)
    picture_ref = f"<Picture {picture}>" if picture is not None else ""
    subject_ref = f"<Subject {subject}>" if subject is not None else becomes
    subject_with_pic = f"<Subject {subject}> (from {picture_ref})" if (subject is not None and picture_ref) else subject_ref
    if kind == "character_swap":
        replacement = f"{subject_ref} ({becomes})" if subject is not None else becomes
        return (f"{target} is replaced by {replacement}, who takes over that character's position, route, scale, pose, "
                "every movement, expression, gaze and any props the original character holds.")
    if kind in PERSON_PART_KINDS:
        person, part = _resolve_person_and_part(edit)
        person_no_article = _without_article(person)
        part_no_article = _without_article(part)
        pronoun = _possessive_pronoun(person)
        clean_desc = _strip_picture_provenance(becomes)
        if kind == "face_swap":
            if subject is not None and picture_ref:
                return (
                    f"The {person_no_article} keeps {pronoun} original hairstyle, headwear, head accessories, outfit, body "
                    f"and motion from <Video 1>. Only the facial features and face shape of the {person_no_article} are "
                    f"replaced by those of {subject_with_pic}. Do not copy the hairstyle, head accessories, outfit, body or "
                    f"held items of {picture_ref}; take only the facial features and face shape."
                )
            return (
                f"The {person_no_article} keeps {pronoun} original hairstyle, headwear, head accessories, outfit, body and "
                f"motion from <Video 1>. Only the facial features and face shape of the {person_no_article} are replaced by "
                f"{becomes}; do not change the original hairstyle, headwear, outfit or body."
            )
        if kind == "head_swap":
            opts = _head_swap_opts(edit, request)
            replacement = f"those of {subject_with_pic}" if (subject is not None and picture_ref) else becomes
            parts = ("face, facial features and hairstyle" if not opts["picture_accessories"]
                     else "face, facial features, hairstyle and head accessories")
            keeps = "outfit, headwear, body and motion" if opts["keep_source_headwear"] else "outfit, body and motion"
            tail = ""
            if not opts["picture_accessories"] and picture_ref and not opts["keep_source_headwear"]:
                tail = f" Do not copy the head accessories or headwear of {picture_ref}."
            return (
                f"The {person_no_article} keeps {pronoun} own {keeps} from <Video 1>. "
                f"The entire head — {parts} — of the {person_no_article} is replaced by "
                f"{replacement}, the whole hairstyle down to its tips; the {keeps.replace(' and motion', '')} and everything "
                f"else of that character stay as in <Video 1>.{tail}"
            )
        if kind == "outfit":
            clothing_noun = _without_article(clean_desc)
            if subject is not None and picture_ref:
                ow = _outfit_wording(_outfit_choice(edit, request))
                return (
                    f"The {person_no_article} keeps {pronoun} own {ow['keeps']} from "
                    f"<Video 1> and now wears the {clothing_noun} shown in {picture_ref} as {subject_with_pic}."
                    f"{ow['carried']} {ow['do_not_copy']} {picture_ref}; take {ow['take']}."
                )
            ow = _outfit_wording(_outfit_choice(edit, request))
            choice = _outfit_choice(edit, request)
            declined = [n for n in choice.get("declined", []) if n not in choice.get("kept", [])]
            return (
                f"The {person_no_article} keeps {pronoun} own {ow['keeps']} from <Video 1> "
                f"and now wears {becomes} as requested. Do not change the face, hair, body or held items; only the outfit "
                "changes." + (f" Do not add any {_join_words(declined, 'or')}." if declined else "")
            )
        if kind == "prop":
            if subject is not None and picture_ref:
                return (
                    f"The {person_no_article} keeps {pronoun} own face, hair, head accessories, outfit, body and motion "
                    f"from <Video 1>, while the {part_no_article} is replaced by {subject_with_pic}, {clean_desc}, held and "
                    f"moved exactly as the original prop. Do not copy any person, face, hair, head accessories or outfit of "
                    f"{picture_ref}; take only the prop."
                )
            return (
                f"The {person_no_article} keeps {pronoun} own face, hair, head accessories, outfit, body and motion from "
                f"<Video 1>, while the {part_no_article} is replaced by {becomes} as requested, held and moved exactly as "
                "the original prop."
            )
    if kind == "background":
        place = subject_with_pic if (subject is not None and picture_ref) else becomes
        return (f"The whole video now takes place in {place} instead of the {target_no_article}. The characters keep "
                "their original positions and motion and stand in the foreground of this new setting, lit by its own light.")
    if kind == "style":
        return (f"The whole video is redrawn in {becomes}" + (f" (style reference {picture_ref}, not a subject)" if picture_ref else " as requested")
                + "; composition, motion, timing and cuts stay as in <Video 1>.")
    if kind == "add":
        where = (target_no_article if re.match(r"^(?:at|in|on|near|beside|above|below|behind|inside|outside)\b", target_no_article, re.I)
                 else f"at {target_no_article}") if target.strip() else "in the indicated area"
        if subject is not None and picture_ref:
            return f"{subject_with_pic}, {becomes}, is added {where}, consistent across every shot where that area is visible."
        return f"{becomes} is added {where}, consistent across every shot where that area is visible."
    if kind == "remove":
        return f"{target} is removed from every shot; the area shows plausible background."
    if kind == "attribute":
        if subject is not None and picture_ref:
            return f"Only {target} changes to {becomes}, matching {subject_with_pic}."
        return f"Only {target} changes to {becomes}."
    return f"Only {target} changes to {becomes}."


def _motion_summary_sentence(edit: dict[str, Any], subject: int | None, request: str = "") -> str:
    if edit.get("kind") == "character_swap":
        clean_becomes, _ = _sanitize_becomes(str(edit["becomes"]).strip(), "character_swap", request)
        performer = f"<Subject {subject}> ({clean_becomes})" if subject is not None else clean_becomes
        return (f"{performer} performs the same movements and choreography as {edit['target']} in <Video 1>, "
                "preserving the source motion rather than editing its performer.")
    return _edit_sentence(edit, subject, request)


_BG_CONTRADICTION_RE = re.compile(
    r"(?:(?<!new\s)(?<!this\s)\b(?:all\s+)?(?:unchanged\s+)?(?:the\s+)?(?:backgrounds?|background\s+elements|setting|environment)\b"
    r"(?:(?!character|camera|people|dancer|\bnot\b|\bnever\b|n't\b|\bno\b)[^.;\n<]){0,80}"
    r"\b(?:stays?|remains?|is\s+kept|are\s+kept|follows?)\b[^.;\n]{0,40}<Video 1>)"
    r"|(?:Everything else\s*[—-][^.\n]{0,240}\b(?:the\s+)?(?:backgrounds?|background elements)\b[^.\n]{0,240}<Video 1>)",
    re.I,
)


def _background_guard(edits: list[dict[str, Any]], subject_map: dict[int, int]) -> dict[str, str] | None:
    """换背景时必须写进提示词的三句：风格开头 / summary 收尾 / [Shot 1] 收尾（不靠模型自觉）。"""
    for index, edit in enumerate(edits):
        if edit.get("kind") != "background":
            continue
        subject = subject_map.get(index)
        picture = _edit_picture(edit)
        becomes = _strip_picture_provenance(str(edit.get("becomes") or "").strip()).rstrip(".")
        old = _without_article(str(edit.get("target") or "the original setting"))
        place = f"<Subject {subject}>" if (subject is not None and picture is not None) else "the new setting"
        return {
            "open": f"The target video is set entirely in {place}, with that setting's own light, colors and atmosphere.",
            "summary": f"From the first frame to the last the background is always {place}; no part of the {old} remains.",
            "shot": (f"In every frame, from the first frame to the last, the background is {place}: {becomes}; the characters stand "
                     f"in its foreground with light and shadows matching it. Nothing of the {old} — its walls, ceiling, floor, "
                     "furniture or lights — appears anywhere."),
            "marker": "Nothing of the",
        }
    return None


def render_edit_prompt(analysis: dict[str, Any], *, duration: float, has_audio: bool = True, n_shots: int = 1,
                       request: str = "") -> str:
    """Render the authoritative edit analysis as one generic, timestamp-free Ref2VA prompt."""
    edits = _active_edits(analysis)
    subject_map = _edit_subjects(analysis)
    if analysis.get("edit_type") == "motion_transfer" and sum(e.get("kind") == "character_swap" for e in edits) == 1:
        # 单人动作迁移：源舞者只是动作来源，不描述其长相（写了帽子、衣服会漏到新人物身上）
        edits = [dict(e, target=_motion_performer(e)) if e.get("kind") == "character_swap" else e for e in edits]
    style_source = str(analysis.get("style_source") or "video").lower()
    if style_source not in STYLE_SOURCES:
        style_source = "video"
    # 同一张图既给人物又给背景（动作迁移「保留照片原背景」）时，人物行不能再说「图里的背景忽略」
    setting_pictures = {_edit_picture(edit) for edit in edits if edit.get("kind") == "background"} - {None}
    person_pictures = {_edit_picture(edit) for edit in edits if edit.get("kind") == "character_swap"} - {None}

    def _picture_ignored(picture: int) -> str:
        if picture in setting_pictures:
            return (f"Only the standing pose of <Picture {picture}> is ignored; the scene behind the person in "
                    f"<Picture {picture}> is used separately as the setting.")
        return f"The background and standing pose of <Picture {picture}> are ignored."

    subject_lines = []
    for edit_index, edit in enumerate(edits):
        source = edit.get("source", {})
        picture = _edit_picture(edit)
        kind = str(edit.get("kind") or "")
        if kind == "style":
            if picture is not None:
                subject_lines.append(f"<Picture {picture}>: style reference only (not a subject).")
            continue
        subject_no = subject_map.get(edit_index)
        if subject_no is not None and picture is not None:
            becomes, _ = _sanitize_becomes(str(edit["becomes"]).strip(), kind, request)
            clean_desc = _strip_picture_provenance(becomes)
            if kind == "outfit":
                ow = _outfit_wording(_outfit_choice(edit, request))
                subject_lines.append(
                    f"<Subject {subject_no}>: the outfit ({clean_desc}) worn by the character in "
                    f"<Picture {picture}>; {ow['subject_not']}{ow['carried']}"
                )
            elif kind == "prop":
                subject_lines.append(
                    f"<Subject {subject_no}>: the prop only ({clean_desc}) shown in <Picture {picture}>; "
                    f"NOT any character's face, hair, head accessories, outfit or body in <Picture {picture}>."
                )
            elif kind == "face_swap":
                subject_lines.append(
                    f"<Subject {subject_no}>: the facial features and face shape only ({clean_desc}) of the character in "
                    f"<Picture {picture}>; NOT the character's hairstyle, head accessories, outfit, body or held items."
                )
            elif kind == "head_swap":
                opts = _head_swap_opts(edit, request)
                if not opts["picture_accessories"]:
                    clean_desc = _drop_parts(clean_desc, ["head accessories"]) or "the face and hairstyle"
                head_parts = "head, face, hairstyle and head accessories" if opts["picture_accessories"] else "head, face and hairstyle"
                ignored = "outfit, body and held items" if opts["picture_accessories"] else "head accessories, outfit, body and held items"
                subject_lines.append(
                    f"<Subject {subject_no}>: the {head_parts} ({clean_desc}) shown in <Picture {picture}>; "
                    f"only these are transferred, while the {ignored} in <Picture {picture}> are ignored."
                )
            elif kind == "character_swap":
                subject_lines.append(
                    f"<Subject {subject_no}>: the person from <Picture {picture}> — {clean_desc}; identity, face, hair and "
                    f"outfit come from <Picture {picture}>, while pose, position and every movement come from "
                    f"{_without_article(str(edit.get('target') or 'the replaced character'))} in <Video 1>. "
                    + _picture_ignored(picture)
                )
            elif kind == "background" and picture in person_pictures:
                subject_lines.append(
                    f"<Subject {subject_no}>: the setting behind the person in <Picture {picture}> — {clean_desc}; only the "
                    "place is used, not the person standing in it."
                )
            elif kind == "background":
                subject_lines.append(f"<Subject {subject_no}>: the setting from <Picture {picture}> — {clean_desc}.")
            elif f"<Picture {picture}>" in becomes:
                subject_lines.append(f"<Subject {subject_no}>: {becomes}.")
            else:
                subject_lines.append(f"<Subject {subject_no}>: {becomes}, as shown in <Picture {picture}>.")
    motion = analysis.get("edit_type") == "motion_transfer"
    video_desc = ("the motion reference video" if motion else "the source video being edited")
    if int(n_shots) > 1:
        video_desc += f", {int(n_shots)} shots with hard cuts"
    source_motion = str(analysis.get("source_motion") or "").strip().rstrip(".")
    if any(edit.get("kind") == "head_swap" for edit in edits):   # 换头时动作描述里也不能带原来的头发/帽子
        source_motion = _strip_head_words(source_motion)
    subject_lines.append(f"<Video 1>: {video_desc}.")
    if has_audio:
        subject_lines.append("<Audio 1>: the original soundtrack of <Video 1>.")

    audio_suffix = " + audio reuse" if has_audio else ""
    prefix = ("[reference generation" if motion else "[video editing + reference generation") + audio_suffix + "]"
    edit_sentences = [_edit_sentence(edit, subject_map.get(i), request) for i, edit in enumerate(edits)]
    unchanged = "; ".join(str(item).strip() for item in analysis.get("unchanged", []) if str(item).strip())
    if not unchanged:
        unchanged = "no other visible characters or important objects are specified"
    has_background_edit = any(edit.get("kind") == "background" for edit in edits)
    background_retention = "the characters' positions" if has_background_edit else "the backgrounds"
    bg_guard = _background_guard(edits, subject_map)
    n_swaps = sum(1 for edit in edits if edit.get("kind") == "character_swap")
    summary_start = f"{prefix} "
    if not motion:
        summary_start += "The target video is an edited version of <Video 1>. "
    summary_edits = ([_motion_summary_sentence(edit, subject_map.get(i), request) for i, edit in enumerate(edits)]
                     if motion else edit_sentences)
    summary = (summary_start + " ".join(summary_edits)
               + f" Everything else — {unchanged}, {background_retention}, the camera, every shot and cut, all motion and timing — "
                 "stays exactly as in <Video 1>." + (f" {bg_guard['summary']}" if bg_guard else ""))
    if motion:
        summary = f"{prefix} " + " ".join(summary_edits) + (
            f" Everything else — {unchanged}, {background_retention}, the camera and framing — follows <Video 1>."
        ) + (f" {bg_guard['summary']}" if bg_guard else "")

    edit_names = "; ".join(_changed_target_phrase(edit) for edit in edits)
    retention = []
    if motion:
        retention.append("<Video 1>: partially_preserved - only the motion, rhythm, camera and framing"
                         + ("" if has_background_edit else " and its setting") + " are reused.")
    else:
        retention.append(f"<Video 1>: partially_preserved - only {edit_names} change"
                         + (" (the original setting is removed entirely)" if has_background_edit else "")
                         + f"; shots, cuts, camera, framing, {background_retention}, {unchanged}, motion, expressions and "
                           "timing are retained frame by frame.")
    style_clause = " and its art style" if style_source == "picture" else ""
    for edit_index, edit in enumerate(edits):
        source = edit.get("source", {})
        picture = _edit_picture(edit)
        kind = str(edit.get("kind") or "")
        if kind == "style":
            if picture is not None:
                retention.append(f"<Picture {picture}>: reference - visual style only.")
            continue
        subject_no = subject_map.get(edit_index)
        if subject_no is None or picture is None:
            continue
        if kind == "outfit":
            ow = _outfit_wording(_outfit_choice(edit, request))
            retention.append(
                f"<Picture {picture}>: partially_preserved - only {ow['take']}{style_clause} is referenced; "
                f"{ow['picture_ignored']} <Picture {picture}> are ignored."
            )
            continue
        if kind == "prop":
            retention.append(
                f"<Picture {picture}>: partially_preserved - only the prop{style_clause} is referenced; "
                f"any person, face, hair and outfit in <Picture {picture}> are ignored."
            )
            continue
        if kind == "face_swap":
            retention.append(
                f"<Picture {picture}>: partially_preserved - only the facial features and face shape{style_clause} are "
                f"referenced; the hairstyle, head accessories, outfit, body and held items in <Picture {picture}> are ignored."
            )
            continue
        trait = {
            "character_swap": "identity, appearance and outfit",
            "head_swap": ("face, head, hairstyle and head accessories"
                          if _head_swap_opts(edit, request)["picture_accessories"] else "face, head and hairstyle"),
            "background": "setting",
            "add": "appearance",
            "attribute": "the requested appearance",
        }.get(kind, "appearance")
        if style_source == "picture" and kind != "style":
            trait = f"{trait} and its art style"
        if kind == "background":
            retention.append(f"<Subject {subject_no}> (appears in [Shot 1]): fully_preserved - the {trait} from <Picture {picture}> "
                             "is the only background in every frame and every shot; no part of the original setting remains.")
            continue
        retention.append(f"<Subject {subject_no}>: fully_preserved - the {trait} from <Picture {picture}> stays identical in every frame and every shot.")
        if kind == "character_swap":
            if picture in setting_pictures:
                retention.append(f"<Picture {picture}>: partially_preserved - the person and outfit are referenced, and the "
                                 "scene behind the person becomes the setting; only its standing pose is ignored.")
            else:
                retention.append(f"<Picture {picture}>: partially_preserved - only the person and outfit are referenced; the "
                                 f"background and standing pose of <Picture {picture}> are ignored.")
    if has_audio:
        retention.append("<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.")
    retention = list(dict.fromkeys(retention))

    style_edit = next((edit for edit in edits if edit.get("kind") == "style"), None)
    if style_edit:
        style_picture = style_edit.get("source", {}).get("picture")
        style_open = f"The video is redrawn in {style_edit['becomes']}" + (
            f" from <Picture {style_picture}>, a style reference rather than a subject." if style_picture is not None else ".")
        if style_picture is None:
            style_open = f"The video is redrawn in {style_edit['becomes']} as requested."
    else:
        style = str(analysis.get("style") or "visual style")
        if style_source == "picture":
            style_open = (
                "Each replacing subject keeps the art style and rendering of its own picture "
                f"(e.g. a 3D-rendered or live-action look), while the rest of the video keeps the {style} of <Video 1>."
            )
        else:
            style_open = f"The video keeps the {style} look of <Video 1>; replacements are rendered in that look."
        if bg_guard:
            style_open = bg_guard["open"] + (
                " Each replacing subject keeps the art style and rendering of its own picture."
                if style_source == "picture"
                else f" The characters keep the {str(analysis.get('style') or 'visual style')} look of <Video 1>.")

    if motion:
        motion_edit_index = next((i for i, edit in enumerate(edits) if edit.get("kind") == "character_swap"), None)
        if motion_edit_index is not None:
            motion_edit = edits[motion_edit_index]
            subject_no = subject_map.get(motion_edit_index)
            performer = f"<Subject {subject_no}>" if subject_no is not None else str(motion_edit["becomes"])
            bg_index = next((i for i, edit in enumerate(edits) if edit.get("kind") == "background"), None)
            if bg_index is None:
                setting = "the same setting as <Video 1>"
            elif bg_index in subject_map:
                setting = f"the setting shown by <Subject {subject_map[bg_index]}>"
            else:
                setting = "the new setting described below"
            detailed = (f"[Shot 1] {performer} performs exactly the same movements, steps, rhythm, facing and camera as "
                        f"{motion_edit['target']} in <Video 1>, in {setting}. The camera, framing, cuts and timing follow "
                        "<Video 1> exactly.")
        else:
            detailed = "[Shot 1] The referenced performance follows the source video's motion and camera exactly."
        if bg_guard:
            detailed += f" {bg_guard['shot']}"
    else:
        anchor_items: list[str] = []
        for edit in edits:
            if edit.get("kind") == "background":
                continue
            if edit.get("kind") in PERSON_PART_KINDS:
                person, _part = _resolve_person_and_part(edit)
                anchor = f"the {_without_article(person)}"
            else:
                anchor = str(edit["target"]).strip()
            if anchor and anchor not in anchor_items:
                anchor_items.append(anchor)
        targets = "; ".join(anchor_items)
        verb = "appear" if (len(anchor_items) > 1 or any(_PLURAL_PERSON_RE.search(item) for item in anchor_items)) else "appears"
        where = f" and at every moment where {targets} {verb}" if targets else ""
        detailed = (f"[Shot 1] Throughout <Video 1>, from the very first frame to the last, in every shot{where}, "
                    + " ".join(edit_sentences) + f" {unchanged} stay exactly as in <Video 1>. The camera, framing, cuts and "
                    "the timing of every movement follow <Video 1> exactly."
                    + (f" In <Video 1>, {source_motion[0].lower() + source_motion[1:]}; the replacement characters perform exactly these "
                       "movements and interactions." if source_motion else "")
                    + (" Each replacement character is matched to its original character by clothing and hair from the very "
                       "first frame and keeps following that same character through every turn, crossing or side swap; the "
                       "original characters never reappear in any frame." if n_swaps >= 2 else "")
                    + (f" {bg_guard['shot']}" if bg_guard else ""))
    details = style_open + "\n" + detailed
    soundscape = (
        "<Audio 1> continues unchanged as the full soundtrack; nothing is added or removed."
        if has_audio
        else "The video is quiet, with no added sound."
    )
    music = "Any music is the original music inside <Audio 1>, unchanged." if has_audio else "N/A"
    return ("subject_definitions:\n" + "\n".join(subject_lines)
            + "\n\nsummary:\n" + summary
            + "\n\nretention_analysis:\n" + "\n".join(retention)
            + "\n\ndetailed_description:\n" + details
            + "\n\noverall_soundscape:\n" + soundscape
            + "\n\nnon_diegetic_music:\n" + music)


def _analysis_messages(job: Job) -> list[dict[str, Any]]:
    system = ("You are a video-edit analyst; read the request, the pictures and the source-video sheets; output only JSON.")
    pictures = [asset for asset in job.assets if asset.kind == "picture"]
    videos = [asset for asset in job.assets if asset.kind == "video"]
    selected_role = videos[0].note if videos else "editing"
    selected_kind = str(selected_role or "editing").split()[0].lower()
    if "(auto)" in str(selected_role or ""):
        selected_kind = "auto"
    role_rule = ("The selected role is editing, so keep edit_type in the editing family even if the request mentions movement."
                 if selected_kind == "editing" else
                 "The video role was not chosen by hand (a first guess from the wording only: "
                 + str(selected_role).split()[0] + "). Decide from the request, the pictures and the video: when the user wants "
                 "the pictured person to perform the video's motion (dance or move like the video, keep the photo's own scene, "
                 "only the motion comes from the video), set edit_type=motion_transfer; otherwise edit the video."
                 if selected_kind == "auto" else
                 "The selected role is motion, so set edit_type=motion_transfer; the performer comes from the picture the "
                 "user names for the person (if unspecified, the first picture not named as the background)."
                 if selected_kind == "motion" else "")
    content: list[dict[str, Any]] = [{"type": "text", "text": "User request:\n" + job.request.strip()
        + "\n\nSelected source-video role: " + (selected_role or "editing")
        + ("\n" + role_rule if role_rule else "")
        + "\n\n" + _EDIT_ANALYSIS_RULES + f"\n\nThere are {len(pictures)} attached picture(s)."}]
    for number, asset in enumerate(pictures, 1):
        content.append({"type": "text", "text": f"Picture {number}:"})
        if asset.data_urls:
            content.append({"type": "image_url", "image_url": {"url": asset.data_urls[0]}})
    for number, asset in enumerate(videos, 1):
        for index, url in enumerate(asset.data_urls, 1):
            label = asset.frame_labels[index - 1] if index - 1 < len(asset.frame_labels) else f"sheet {index}"
            content.append({"type": "text", "text": f"Source video, {label}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _analysis_english(analysis: dict[str, Any]) -> str:
    """Render the structured reverse-analysis as compact, authoritative English instructions."""
    lines = []
    for index, edit in enumerate(_active_edits(analysis), 1):
        source = edit.get("source", {})
        picture = _edit_picture(edit)
        source_text = f"reference <Picture {picture}>" if picture is not None else f"text instruction: {source.get('text', '')}"
        kind = str(edit.get("kind") or "")
        if kind in PERSON_PART_KINDS:
            person, part = _resolve_person_and_part(edit)
            lines.append(
                f"Edit {index} ({kind}): Target person in <Video 1> is {person} (sentence subject MUST be The {_without_article(person)}, "
                f"never the {part} itself); only their {part} is replaced by {edit['becomes']} ({source_text}). "
                f"Preserve from {person} in <Video 1>: {edit['keep_from_target']}. "
                + (
                    f"Ignore {_outfit_wording(_outfit_choice(edit))['picture_ignored']} <Picture {picture}>; "
                    f"take {_outfit_wording(_outfit_choice(edit))['take']}."
                    if (kind == "outfit" and picture is not None)
                    else f"Ignore the hairstyle, head accessories, outfit, body and held items in <Picture {picture}>; take only the facial features and face shape."
                    if (kind == "face_swap" and picture is not None)
                    else f"Ignore any person, face, hair, head accessories and outfit in <Picture {picture}>; take only the prop."
                    if (kind == "prop" and picture is not None)
                    else (f"Transfer the whole head — face, hairstyle"
                          f"{' and head accessories' if _head_swap_opts(edit)['picture_accessories'] else ''} — from <Picture {picture}>; "
                          f"keep outfit{', headwear' if _head_swap_opts(edit)['keep_source_headwear'] else ''} and body from <Video 1>.")
                    if (kind == "head_swap" and picture is not None)
                    else ""
                )
            )
        elif kind == "character_swap" and analysis.get("edit_type") == "motion_transfer":
            lines.append(f"Edit {index} (motion transfer): the pictured person ({source_text}) performs the motion of "
                         f"{_motion_performer(edit)} in <Video 1>; refer to that dancer only as '{_motion_performer(edit)}', "
                         "never by clothes or hair.")
        else:
            lines.append(f"Edit {index} ({kind}): In <Video 1>, {edit['target']} becomes {edit['becomes']} ({source_text}). "
                         f"Preserve from the target: {edit['keep_from_target']}.")
    unchanged = "; ".join(analysis.get("unchanged", [])) or "none specified"
    style_source = str(analysis.get("style_source") or "video").lower()
    lines.extend([
        f"Unchanged visible content: {unchanged}.",
        f"Source-video visual style: {analysis.get('style', '')}.",
        f"Visual style source: {style_source}.",
    ])
    if analysis.get("notes"):
        lines.append("Ambiguity notes: " + analysis["notes"])
    if analysis.get("edit_type") == "motion_transfer":
        lines.append("Task type: motion transfer / reference generation; <Video 1> is a motion reference, not an edited source.")
    else:
        lines.append("Task type: generic video editing; change only the listed targets.")
    return "\n".join(lines)


_CLUTTER_HINT_RE = re.compile(
    r"held\s+(?:item|prop|object)|holding|carrying|手持|道具.{0,6}(?:未带|忽略)|白底|棚拍|studio\s+background|"
    r"\b(?:" + "|".join(_HELD_PROP_NOUNS) + r")\b[^\n.;]{0,24}(?:ignored|omitted|dropped|not\s+carried)",
    re.I,
)


def _has_ignored_picture_clutter(analysis: dict[str, Any], stripped_extra: list[str] | None = None) -> bool:
    if stripped_extra or _STRIPPED_CLUTTER_BY_ANALYSIS.get(id(analysis)):
        return True
    if str(analysis.get("ignored_from_picture") or "").strip():
        return True
    for edit in _active_edits(analysis):
        if str(edit.get("ignored_from_picture") or "").strip():
            return True
    for item in analysis.get("pictures", []) if isinstance(analysis.get("pictures"), list) else []:
        if isinstance(item, dict) and _CLUTTER_HINT_RE.search(str(item.get("note") or "")):
            return True
    if _CLUTTER_HINT_RE.search(str(analysis.get("notes") or "")):
        return True
    return False


def _format_edit_report(analysis: dict[str, Any], render: str, picture_count: int,
                        stripped_extra: list[str] | None = None) -> str:
    active = _active_edits(analysis)
    style_source = str(analysis.get("style_source") or "video").lower()
    report_lines = [f"edit=two-pass render={render} analysis=ok type={analysis['edit_type']} style_source={style_source}"]
    picture_kinds: dict[int, list[str]] = {}
    for edit in active:
        picture = _edit_picture(edit)
        if picture is not None:
            picture_kinds.setdefault(int(picture), []).append(str(edit.get("kind") or ""))
        tag = f"图{picture}" if picture is not None else "文字"
        if edit.get("kind") in PERSON_PART_KINDS:
            person, part = _resolve_person_and_part(edit)
            shown_target = f"{person}（{part}）" if part and part.lower() not in person.lower() else person
        else:
            shown_target = str(edit["target"])
        report_lines.append(f"{tag} → 替换: {shown_target}\n      换成: {edit['becomes']}")
    if _has_ignored_picture_clutter(analysis, stripped_extra):
        report_lines.append("提示: 参考图中的手持物/姿势/背景未带入")
    report_lines.append("不变: " + ("; ".join(analysis.get("unchanged", [])) or "无"))
    report_lines.append("画风: " + ("保持参考图" if style_source == "picture" else "跟原片"))
    if str(analysis.get("style") or "").strip():
        report_lines.append("原片画风: " + str(analysis["style"]).strip())

    remarks: list[str] = []
    for pic in sorted(picture_kinds):
        kinds = picture_kinds[pic]
        if len(kinds) >= 2:
            kinds_zh = "+".join(_EDIT_KIND_ZH.get(k, k) for k in kinds)
            remarks.append(f"图{pic} 同时用于 {kinds_zh}")
    unused_pics = sorted(
        (set(range(1, max(0, int(picture_count)) + 1)) - set(picture_kinds)) | set(_explicit_unused_pictures(analysis))
    )
    for pic in unused_pics:
        remarks.append(f"图{pic} 未使用（需求里没提到）")
    raw_notes = str(analysis.get("notes") or "").strip()
    for item in analysis.get("pictures", []) if isinstance(analysis.get("pictures"), list) else []:
        if not isinstance(item, dict):
            continue
        pic = item.get("picture")
        note_text = str(item.get("note") or "").strip()
        if isinstance(pic, int) and re.search(r"multiple\s+people|several\s+people|multi[_-]?person|多个人|多人", note_text, re.I):
            remarks.append(f"图{pic} 里有多个人，取了中间的那个")
    for match in re.finditer(r"Picture\s*(\d+)[^\n.;]*(?:multiple|several)\s+people", raw_notes, re.I):
        msg = f"图{match.group(1)} 里有多个人，取了中间的那个"
        if msg not in remarks:
            remarks.append(msg)
    if re.search(r"(?:second\s+lead|第二主角)[^\n.;]*(?:no\s+(?:reference\s+)?picture|没有参考图)", raw_notes, re.I):
        remarks.append("第二主角没有参考图，按文字/保持原样")
    if raw_notes and raw_notes not in remarks:
        remarks.append(raw_notes)
    report_lines.append("备注: " + "；".join(remarks))
    return "\n".join(report_lines)


def _sample_sheets_evenly(urls: list[str], labels: list[str], limit: int = 4) -> tuple[list[str], list[str]]:
    """Select up to `limit` sheets evenly across time, always including the first and last."""
    total = len(urls)
    limit = max(1, int(limit))
    if total <= limit:
        return list(urls), list(labels)
    if limit == 1:
        picks = [0]
    else:
        picks = sorted({int(round(i * (total - 1) / (limit - 1))) for i in range(limit)})
        picks[0] = 0
        picks[-1] = total - 1
    picked_urls = [urls[i] for i in picks]
    picked_labels = [labels[i] if i < len(labels) else f"sheet {i + 1}" for i in picks]
    return picked_urls, picked_labels


def _video_role_note(job: Job, *, manual_only: bool = False) -> str:
    """视频用途：motion / editing / continuation。manual_only=True 时，auto 猜的不算（返回 ""）。"""
    videos = [a for a in job.assets if a.kind == "video"]
    note = str((videos[0].note if videos else "") or "editing")
    if manual_only and "(auto)" in note:
        return ""
    parts = note.split()
    return parts[0].lower() if parts else "editing"


_FALLBACK_KIND_WORDS = (
    ("head_swap", r"换头|头换成|连发型|head\s*swap|swap\s+(?:her|his|their|the)\s+head"),
    ("face_swap", r"换脸|脸换成|face\s*swap|swap\s+(?:her|his|their|the)\s+face"),
    ("outfit", r"换装|换衣|衣服|服装|\boutfit\b|\bclothes\b"),
    ("background", r"换背景|背景换成|背景改成|背景用|换个背景|\b(?:change|replace|swap)\s+the\s+background\b"),
)


def _conservative_fallback_analysis(job: Job, picture_count: int, style_source: str) -> dict[str, Any]:
    """模型两轮都没给出合法分析时的最简单默认稿：不去猜客户原话的细节。
    动作参考 → 图1的人做动作迁移，场景沿用原视频；编辑 → 按卡片措辞（换头/换脸/换装，否则换人）用图1；没图就只写泛称。
    报告里会写明“分析失败，用了默认稿，建议人工检查”。"""
    req = str(job.request or "")
    role_note = _video_role_note(job)
    pic = 1 if picture_count >= 1 else None
    if role_note == "motion":
        edit_type, kind = "motion_transfer", "character_swap"
    else:
        kind = next((k for k, pattern in _FALLBACK_KIND_WORDS if affirmed_hit(re.compile(pattern, re.I), req)), "character_swap")
        edit_type = kind
    if pic is None:
        edits = [{"kind": "attribute", "target": "the main character", "source": {"picture": None, "text": "the requested change"},
                  "becomes": "the requested appearance", "keep_from_target": "position, motion, camera and timing"}]
        edit_type = "attribute"
    elif kind == "background":
        edits = [{"kind": "background", "target": "the original setting of <Video 1>", "source": {"picture": pic, "text": ""},
                  "becomes": f"the setting shown in <Picture {pic}>", "keep_from_target": _DEFAULT_KEEP["background"]}]
    elif kind == "outfit":
        edits = [{"kind": "outfit", "target": "the main character", "target_person": "the main character", "target_part": "outfit",
                  "source": {"picture": pic, "text": ""}, "becomes": f"the outfit and shoes shown in <Picture {pic}>",
                  "keep_from_target": "face, hair, body and motion"}]
    else:
        noun = {"character_swap": "character", "face_swap": "facial features and face shape", "head_swap": "head and hairstyle"}[kind]
        edits = [{"kind": kind, "target": "the main character",
                  **({"target_person": "the main character", "target_part": noun} if kind in {"face_swap", "head_swap"} else {}),
                  "source": {"picture": pic, "text": ""}, "becomes": f"the {noun} shown in <Picture {pic}>",
                  "keep_from_target": "position, route, scale, pose, movement, expression, gaze and held props"}]
    for edit in edits:
        _apply_part_fields(edit)
    out: dict[str, Any] = {"edit_type": edit_type, "edits": edits, "unchanged": ["all other visible characters and objects"],
                           "style": "visual style", "notes": "conservative generic fallback; please check the result by hand"}
    if style_source == "picture":
        out["style_source"] = "picture"
    return out


def optimize_edit(backend: Any, job: Job, *, render: str = "llm", handbook: str = "", template: str = "",
                  style_source: str | None = None, max_tokens: int = 6000, thinking_mode: str = "medium",
                  seed: int = 1234, correction: bool = True, sampling: dict[str, Any] | None = None,
                  fallback_job: Job | None = None) -> dict[str, Any]:
    """Analyze a generic video edit, then render deterministically or with the H3 writer."""
    started = time.perf_counter()
    render = "code" if str(render).lower() == "code" else "llm"
    picture_count = sum(a.kind == "picture" for a in job.assets)
    analysis_messages = _analysis_messages(job)
    req_tokens = int(max_tokens)
    analysis_budget = 0 if req_tokens <= 0 else min(req_tokens, 6144)
    usage: list[dict[str, Any]] = []
    errors: list[str] = []
    analysis = None
    raw = ""
    for attempt in range(2):
        raw, item_usage = _complete(backend, analysis_messages, max_tokens=analysis_budget, thinking_mode=thinking_mode,
                                    seed=int(seed) + attempt, sampling=sampling)
        usage.append(item_usage)
        try:
            analysis = parse_edit_analysis(raw, picture_count, request=job.request,
                                           video_role=_video_role_note(job, manual_only=True))
            break
        except ValueError as error:
            errors.append(str(error))
            if attempt == 0:
                analysis_messages = analysis_messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Your previous response failed JSON/schema validation: " + str(error)
                     + "\nRewrite it once as a single valid JSON object matching the required schema. Output JSON only."},
                ]
    has_audio = any(a.kind == "audio" and a.role == "source_audio" for a in job.assets)
    n_shots = len(job.cuts) + 1 if job.cuts is not None else 1
    if analysis is None:
        fallback_style = (
            style_source.lower()
            if isinstance(style_source, str) and style_source.lower() in STYLE_SOURCES
            else resolve_style_source(job.request, template)
        )
        fallback_analysis = _conservative_fallback_analysis(job, picture_count, fallback_style)
        fallback_generic = Job(job.request, job.duration, list(job.assets), job.mode, job.dialogue, None,
                               generic_edit=True, edit_analysis=fallback_analysis)
        mode = resolve_mode(fallback_generic)
        prompt_out = render_edit_prompt(
            fallback_analysis, duration=job.duration, has_audio=has_audio, n_shots=n_shots, request=job.request
        )
        issues = validate(prompt_out, mode, fallback_generic, generic_edit=True)
        issues.append({"code": "analysis_failed", "hard": True,
                       "msg": "the edit analysis failed twice; this is only a generic default draft and may not match the "
                              "request — check it by hand or run again"})
        return {
            "mode": mode,
            "prompt": prompt_out,
            "issues": issues,
            "first_issues": issues,
            "corrected": False,
            "usage": usage,
            "analysis": fallback_analysis,
            "render": render,
            "edit_report": "\n".join(["analysis=failed, generic fallback", "提示: 模型两次都没给出合法分析，用了最简单的默认稿，建议人工检查",
                                     *[f"analysis_error={e}" for e in errors]]),
            "seconds": time.perf_counter() - started,
        }

    stripped_extra: list[str] = []
    for edit in _active_edits(analysis):
        kind = str(edit.get("kind") or "")
        if kind and isinstance(edit.get("becomes"), str):
            cleaned_b, rem = _sanitize_becomes(edit["becomes"], kind, job.request)
            edit["becomes"] = cleaned_b
            stripped_extra.extend(rem)

    effective_style_source = (
        style_source.lower()
        if isinstance(style_source, str) and style_source.lower() in STYLE_SOURCES
        else resolve_style_source(job.request, template, analysis.get("style_source"))
    )
    if effective_style_source != "video" or "style_source" in analysis:
        analysis["style_source"] = effective_style_source

    generic_job = Job(job.request, job.duration, list(job.assets), job.mode, job.dialogue, None,
                      generic_edit=True, edit_analysis=analysis)
    draft = render_edit_prompt(analysis, duration=job.duration, has_audio=has_audio, n_shots=n_shots, request=job.request)
    mode = resolve_mode(generic_job)
    first_issues: list[dict[str, Any]] = []
    corrected = False
    used_draft = False
    if render == "code":
        final = draft
        issues = validate(final, mode, generic_job, generic_edit=True)
        first_issues = issues
    else:
        final_assets = []
        for asset in generic_job.assets:
            if asset.kind == "video":
                sampled_urls, sampled_labels = _sample_sheets_evenly(asset.data_urls, asset.frame_labels, 4)
                final_assets.append(Asset(asset.kind, asset.role, asset.note, sampled_urls, sampled_labels))
            else:
                final_assets.append(asset)
        final_job = Job(generic_job.request, generic_job.duration, final_assets, generic_job.mode, generic_job.dialogue,
                        None, generic_edit=True, edit_analysis=analysis)
        messages = build_messages(final_job, mode)
        subject_json = json.dumps(analysis, ensure_ascii=False, indent=2)
        extra_system = ("\n\n===== VIDEO EDIT HANDBOOK =====\n" + handbook.strip()
                        + "\n===== END VIDEO EDIT HANDBOOK =====\n\nEdit analysis (authoritative):\n" + _analysis_english(analysis)
                        + "\n\nStructured analysis JSON:\n" + subject_json
                        + "\n\nWrite the prompt in the generic style: exactly one [Shot 1], no timestamps, no shot-by-shot narration; "
                          "describe each target and its replacement once, precisely, then state what follows <Video 1> exactly as the draft "
                          "does (for motion transfer only the motion, rhythm, camera and framing, plus the setting of <Video 1> when the "
                          "draft keeps it). "
                          "Use the code-rendered draft below as the skeleton; you may enrich the target/replacement descriptions from "
                          "the pictures but keep its structure. Keep every subject definition's identity sentence (identity, face, "
                          "hair and outfit from the picture; pose and motion from the replaced character; the picture's standing pose "
                          "ignored, and its background ignored unless the draft uses that picture's scene as the setting). Refer to each original character only with the short identification already in the "
                          "draft and never describe the source video's costumes, place, floor, furniture or lighting anywhere "
                          "else: every extra word about how <Video 1> looks pulls the result back to the original. In [Shot 1] you "
                          "may add one sentence on the actual movements and interactions you see (actions only) and say how the "
                          "new characters' hair and clothing move."
                        + ("\nTHIS EDIT REPLACES THE BACKGROUND. The model can only paint the new setting if you describe it as a "
                           "whole picture, so the draft already contains three required sentences; keep all three, in your own "
                           "words if you like: (1) the opening line 'The target video is set entirely in <Subject N> ...' (or, for a setting described only in "
                           "words, in that described setting — never invent a Subject for it); "
                           "(2) in [Shot 1], the new setting's full layout — the ground the characters stand on, what is behind "
                           "them, the far background, the sky or ceiling, the light and shadows; (3) the last sentence of "
                           "[Shot 1]: nothing of the original setting (its walls, ceiling, floor, furniture or lights) appears "
                           "in any frame, from the first frame to the last. Never write that the background, setting or "
                           "'remaining background elements' stay as in or follow <Video 1>; only the characters' positions, "
                           "motion, camera, cuts and timing follow <Video 1>."
                           if any(e.get("kind") == "background" for e in _active_edits(analysis)) else "")
                        + "\n\nCODE-RENDERED DRAFT:\n" + draft)
        messages[0]["content"] += extra_system
        raw_prompt, item_usage = _complete(backend, messages, max_tokens=int(max_tokens), thinking_mode=thinking_mode,
                                           seed=int(seed) + len(usage), sampling=sampling)
        usage.append(item_usage)
        final = repair(raw_prompt, mode, final_job)
        first_issues = validate(final, mode, final_job, generic_edit=True)
        issues = first_issues
        hard = [item for item in issues if item["hard"]]
        if correction and hard:
            fix = ("Your draft violates these hard requirements:\n- " + "\n- ".join(item["msg"] for item in hard)
                   + "\n\nRewrite the complete prompt fixing only these problems. Keep exactly one [Shot 1], no timestamps, "
                     "and preserve the authoritative edit analysis and skeleton. Output only the final prompt.")
            raw2, usage2 = _complete(backend, messages + [{"role": "assistant", "content": final},
                                                          {"role": "user", "content": fix}],
                                     max_tokens=int(max_tokens), thinking_mode=thinking_mode, seed=int(seed) + len(usage),
                                     sampling=sampling)
            usage.append(usage2)
            candidate = repair(raw2, mode, final_job)
            candidate_issues = validate(candidate, mode, final_job, generic_edit=True)
            if sum(item["hard"] for item in candidate_issues) < len(hard):
                final, issues, corrected = candidate, candidate_issues, True
        # 模型两轮都没写对（格式、换背景、角色……任何硬问题）而代码草稿是干净的：用代码草稿，不把坏提示词交给生成
        if any(item["hard"] for item in issues):
            draft_issues = validate(draft, mode, final_job, generic_edit=True)
            if not any(item["hard"] for item in draft_issues):
                used_draft = ",".join(sorted({item["code"] for item in issues if item["hard"]}))
                final, issues = draft, draft_issues
    return {"mode": mode, "prompt": final, "issues": issues, "first_issues": first_issues,
            "corrected": corrected, "usage": usage, "analysis": analysis, "render": render,
            "edit_report": _format_edit_report(analysis, render, picture_count, stripped_extra)
                           + (f"\ndraft_fallback=模型写的稿有硬问题（{used_draft}），改用代码草稿" if used_draft else "")
                           + ("\nfirst_pass_hard=" + ",".join(sorted({i["code"] for i in first_issues if i["hard"]}))
                              if any(i["hard"] for i in first_issues) else ""),
            "seconds": time.perf_counter() - started}


_VIDEO_ROLE_MAP = {
    **VIDEO_ROLES,
    "auto": "auto",
    "editing": "editing",
    "edit source": "editing",
    "motion": "motion",
    "motion reference": "motion",
    "continuation": "continuation",
    "continuation source": "continuation",
}


def _frames_to_urls(batch: torch.Tensor | None, limit: int | None = None) -> list[list[str]]:
    if batch is None:
        return []
    frames = sample_image_batch(batch, limit) if limit else image_batch_frames(batch)
    return [[pil_to_data_url(tensor_frame_to_pil(frame), 88)] for frame in frames]


def _should_run_h3_mv(expanded_tpl: str, req_text: str, mode: str) -> bool:
    if is_h3_mv_template(expanded_tpl) or bool(_H3_MV_LINE_RE.search(req_text)):
        return True
    if str(mode or "").strip().lower() == "mv":
        return True
    if str(expanded_tpl or "").strip() == H3_MARKER:
        parsed = parse_mv_request(req_text)
        if parsed["is_field_block"] and (set(parsed["seen_keys"]) & _MV_DISTINCT_KEYS):
            return True
    return False


def format_mv_report(prompt: str, mv_spec: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    """Format the concise MV engine report summary line."""
    section_val = str(mv_spec.get("section") or "").strip() or "auto"
    n_shots = len(shots(main_body(prompt, "Ref2VA")))
    issue_codes = {issue["code"] for issue in issues}
    lyrics_ok = not (issue_codes & {"mv_lyrics", "mv_no_lyrics", "dialogue", "speaker"})
    lyrics_val = "ok" if lyrics_ok else "fail"
    personas = mv_spec.get("personas") or {}
    if personas:
        persona_val = "verbatim" if "mv_persona" not in issue_codes else "fail"
    else:
        persona_val = "none"
    prev_val = str(mv_spec.get("prev_end_scale") or "").strip() or "none"
    return (
        f"mode=mv section={section_val} shots={n_shots} "
        f"lyrics={lyrics_val} persona={persona_val} prev_scale={prev_val}"
    )


def run_h3(
    backend: Any,
    template_text: str,
    request: str,
    *,
    seconds: float = 0.0,
    first_frame: torch.Tensor | None = None,
    last_frame: torch.Tensor | None = None,
    references: Any = (),
    video_frames: torch.Tensor | None = None,
    video_role: str = "auto",
    video_fps: float = 24.0,
    keep_source_audio: bool = True,
    thinking_mode: str = "high",
    max_tokens: int = 10240,
    seed: int = 1234,
    sampling: dict[str, Any] | None = None,
    mode: str = "auto",
    correction: bool = True,
    reference_notes: str = "",
    reference_audio_note: str = "",
    source_audio: Any = None,
    video: Any = None,
) -> tuple[str, str]:
    """Unified H3 prompt-generation engine shared by local, dedicated H3, and API nodes."""
    expanded_tpl = expand_template_includes(str(template_text or ""))
    req_text = str(request or "").strip()
    secs = float(seconds or 0.0)
    fps = float(video_fps)
    refs_list = [r for r in (references or ()) if r is not None]
    notes = [n.strip() for n in str(reference_notes or "").splitlines() if n.strip()]
    audio_note = str(reference_audio_note or "").strip()
    effective_keep_audio = bool(keep_source_audio) and not audio_note
    choice = _VIDEO_ROLE_MAP.get(str(video_role or "auto").strip(), "auto")

    prev_retries = getattr(backend, "_h3_retry_budgets", None)
    retry_budgets: list[int] = []
    try:
        setattr(backend, "_h3_retry_budgets", retry_budgets)
    except Exception:
        pass

    try:
        if re.search(r"(?m)^\s*\[\[tank:h3-edit\]\]\s*$", expanded_tpl):
            if video_frames is None:
                raise ValueError("视频编辑预设需要接 video_frames（源视频）")
            if fps <= 0:
                raise ValueError("video_fps must be greater than 0")
            render, handbook = h3_edit_template_options(expanded_tpl)
            if choice == "motion" and not refs_list and first_frame is None and last_frame is None:
                raise ValueError("动作迁移需要接人物图片（参考图），现在一张都没有")
            if choice == "auto":
                role, why = infer_edit_role(req_text)
            else:
                role = choice
                why = "chosen on the node"

            picture_assets: list[Asset] = []
            for picture_role, batch in (("first_frame", first_frame), ("last_frame", last_frame)):
                for urls in _frames_to_urls(batch)[:1]:
                    picture_assets.append(Asset("picture", picture_role, "", urls))
            for idx, ref in enumerate(refs_list):
                for urls in _frames_to_urls(ref)[:1]:
                    picture_assets.append(Asset("picture", "reference", notes[idx] if idx < len(notes) else "", urls))

            cuts = detect_cuts(video_frames, fps)
            sheets = shot_sheets(video_frames, fps, cuts, max_sheets=8, tiles=4, tile_w=384)
            if not sheets:
                raise ValueError("视频编辑预设无法从 video_frames 提取源视频帧")
            frame_urls = [sheet["url"] for sheet in sheets]
            frame_labels = [f"shot {sheet['shot']} ({sheet['start']:.1f}-{sheet['end']:.1f} s)" for sheet in sheets]
            role_note = f"{role} ({'manual' if choice != 'auto' else 'auto'})"
            sheet_asset = Asset("video", "editing", role_note, frame_urls, frame_labels)
            n_frames = int(video_frames.shape[0]) if getattr(video_frames, "ndim", 0) == 4 else 1
            source_duration = n_frames / fps
            duration = secs if secs > 0 else source_duration
            has_audio, audio_state = inspect_source_audio(effective_keep_audio, source_audio=source_audio, video=video)
            main_assets = [*picture_assets, sheet_asset]
            if has_audio:
                main_assets.append(Asset("audio", "source_audio", ""))
            if audio_note:
                main_assets.append(Asset("audio", "audio", audio_note))
            edit_job = Job(req_text, duration, main_assets, mode, cuts=cuts, generic_edit=True)

            result = optimize_edit(
                backend,
                edit_job,
                render=render,
                handbook=handbook,
                template=expanded_tpl,
                max_tokens=int(max_tokens),
                thinking_mode=thinking_mode,
                seed=int(seed),
                correction=bool(correction),
                sampling=sampling,
            )
            hard = [issue for issue in result["issues"] if issue["hard"]]
            if choice == "auto" and isinstance(result.get("analysis"), dict) and "analysis=failed" not in str(result.get("edit_report")):
                final = "motion" if result["analysis"].get("edit_type") == "motion_transfer" else "editing"
                why = f"auto: decided by the analysis; wording guess was {role}"
                role = final
            secs_text = f"{secs:.2f}" if secs > 0 else f"source length {source_duration:.2f}"
            lines = [
                f"template=H3 video-edit render={render} mode={result['mode']}",
                "seconds=" + secs_text,
                f"corrected={result['corrected']} final_hard_issues={len(hard)}",
                f"video_role={role} ({why})",
                f"source={source_duration:.2f}s cuts=" + (", ".join(f"{cut:.2f}" for cut in cuts) or "none"),
                f"audio={audio_state}",
            ]
            if result.get("edit_report"):
                lines.append(result["edit_report"])
            lines += [f"- {issue['code']}: {issue['msg']}" for issue in result["issues"]]
            if retry_budgets:
                lines.append(f"answer needed a bigger budget: max_tokens={retry_budgets[-1]}")
            return result["prompt"], "\n".join(lines)

        if _should_run_h3_mv(expanded_tpl, req_text, mode):
            mv_handbook = h3_mv_template_handbook(expanded_tpl if is_h3_mv_template(expanded_tpl) else "")
            mv_spec = parse_mv_request(req_text, default_seconds=secs)
            duration = mv_spec["duration"]
            raw_pics: list[list[str]] = []
            for batch in (first_frame, last_frame):
                raw_pics.extend(_frames_to_urls(batch)[:1])
            for ref in refs_list:
                raw_pics.extend(_frames_to_urls(ref)[:1])
            picture_assets: list[Asset] = []
            personas = mv_spec.get("personas") or {}
            for pic_idx, urls in enumerate(raw_pics, 1):
                user_note = notes[pic_idx - 1] if pic_idx - 1 < len(notes) else ""
                if pic_idx in personas:
                    pic_note = f"character reference for <Subject {pic_idx}> (use exact persona_{pic_idx} verbatim)"
                    if user_note:
                        pic_note += f"; {user_note}"
                else:
                    pic_note = user_note or "character or optional scene reference picture"
                picture_assets.append(Asset("picture", "reference", pic_note, urls))

            video_ignored = (video_frames is not None) or (video is not None)
            has_mv_audio = (source_audio is not None and source_audio is not False) or bool(keep_source_audio)
            audio_assets: list[Asset] = []
            if has_mv_audio:
                audio_assets.append(
                    Asset(
                        "audio",
                        "source_audio",
                        audio_note or "original song segment reused 1:1 as the complete final soundtrack (fully_copy)",
                    )
                )
                if source_audio is not None:
                    _ok, audio_state = inspect_source_audio(True, source_audio=source_audio)
                else:
                    audio_state = "fully_copy"
            else:
                audio_state = "off"
                if audio_note:
                    audio_assets.append(Asset("audio", "audio", audio_note))

            mv_job = Job(
                _strip_h3_mv_markers(req_text),
                duration,
                [*picture_assets, *audio_assets],
                mode="Ref2VA",
                dialogue=[],
                cuts=None,
                mv_spec=mv_spec,
                mv_handbook=mv_handbook,
            )
            result = optimize(
                backend,
                mv_job,
                thinking_mode=thinking_mode,
                max_tokens=int(max_tokens),
                seed=int(seed),
                correction=bool(correction),
                sampling=sampling,
            )
            hard = [issue for issue in result["issues"] if issue["hard"]]
            mv_summary = format_mv_report(result["prompt"], mv_spec, result["issues"])
            lines = [
                mv_summary,
                f"template=H3 MV task_mode={result['mode']} references={len(picture_assets)} seconds={duration:.2f} audio={audio_state}",
                f"corrected={result['corrected']} final_hard_issues={len(hard)}",
            ]
            if video_ignored:
                lines.append("video_frames=ignored (MV 模式无源视频，已忽略接入的 video_frames)")
            lines += [f"- {issue['code']}: {issue['msg']}" for issue in result["issues"]]
            if retry_budgets:
                lines.append(f"answer needed a bigger budget: max_tokens={retry_budgets[-1]}")
            return result["prompt"], "\n".join(lines)

        assets: list[Asset] = []
        for role_name, batch in (("first_frame", first_frame), ("last_frame", last_frame)):
            for urls in _frames_to_urls(batch)[:1]:
                assets.append(Asset("picture", role_name, "", urls))
        for idx, ref in enumerate(refs_list):
            for urls in _frames_to_urls(ref)[:1]:
                assets.append(Asset("picture", "reference", notes[idx] if idx < len(notes) else "", urls))
        video_assets, cuts, src_len, video_lines = source_video_assets(
            video_frames, choice, req_text, bool(refs_list), fps, effective_keep_audio,
            source_audio=source_audio, video=video,
        )
        dh_mode = bool(re.search(r"(?m)^\s*\[\[tank:h3-dh\]\]\s*$", expanded_tpl))
        if dh_mode:
            if video_assets:
                video_lines.append("video_frames=ignored (数字人模板不用源视频)")
            video_assets, cuts = [], None
            video_lines.append("template=H3 digital human (one continuous shot, fixed camera)")
        assets += video_assets
        lipsync = False
        if audio_note:
            assets.append(Asset("audio", "audio", audio_note))
        elif dh_mode and refs_list and source_audio is not None:
            has_audio, audio_state = inspect_source_audio(effective_keep_audio, source_audio=source_audio)
            if has_audio:
                assets.append(Asset("audio", "source_audio", ""))
                lipsync = True
                video_lines.append(f"lip-sync performance: <Audio 1> kept 1:1, one continuous shot audio={audio_state}")
        editing = cuts is not None
        duration = secs if secs > 0 else (src_len if editing and src_len > 0 else 5.0)
        result = optimize(
            backend,
            Job(req_text, duration, assets, mode, dialogue=[] if lipsync else None, cuts=cuts, dh=dh_mode, lipsync=lipsync),
            thinking_mode=thinking_mode,
            max_tokens=int(max_tokens),
            seed=int(seed),
            correction=bool(correction),
            sampling=sampling,
        )
        hard = [i for i in result["issues"] if i["hard"]]
        secs_text = (f"{secs:.2f}" if secs > 0 else (f"source length {duration:.2f}" if editing and src_len > 0
                                                     else "not used (5.00 assumed)"))
        lines = [
            f"template=H3 mode={result['mode']} references={len(refs_list)}",
            "seconds=" + secs_text,
            f"corrected={result['corrected']} final_hard_issues={len(hard)}",
        ] + video_lines
        lines += [f"- {i['code']}: {i['msg']}" for i in result["issues"]]
        if retry_budgets:
            lines.append(f"answer needed a bigger budget: max_tokens={retry_budgets[-1]}")
        return result["prompt"], "\n".join(lines)
    finally:
        try:
            if prev_retries is None:
                delattr(backend, "_h3_retry_budgets")
            else:
                setattr(backend, "_h3_retry_budgets", prev_retries)
        except Exception:
            pass


# ------------------------------------------------------------------ ComfyUI node
class TankH3PromptOptimizer:
    @classmethod
    def VALIDATE_INPUTS(cls, thinking_mode=None):
        return validate_reasoning_choice(thinking_mode)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": ("VISION_LLM_BACKEND", {"tooltip": "Connect Qwen / Bonsai GGUF Loader · Tank (Bonsai-2 v2 recommended)."}),
                "request": ("STRING", {"multiline": True, "default": "", "tooltip": "What should happen, in plain Chinese or English. Put spoken lines in “quotes”; they are kept verbatim."}),
                "mode": (["auto", *MODES, "mv"], {"default": "auto", "tooltip": "auto picks the official H3 mode from the connected media; mv runs the H3 MV segment mode."}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5, "tooltip": "Target video length; connect the output to the H3 node so both agree."}),
                "thinking_mode": (list(REASONING_CHOICES), {"default": "medium", "tooltip": "medium is the tested default for H3 prompts; off is fastest but weaker."}),
                "correction": ("BOOLEAN", {"default": True, "tooltip": "If the draft breaks the official format, ask the model once to fix it (accepted only if it improves)."}),
                "unload_after": ("BOOLEAN", {"default": True, "tooltip": "Release the language model afterwards so H3 sampling gets the VRAM."}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "First frame (I2VA / FL2VA, or keyframe in a reference task)."}),
                "last_frame": ("IMAGE", {"tooltip": "Last frame (L2VA / FL2VA, or keyframe in a reference task)."}),
                **{f"reference_image_{i}": ("IMAGE", {"tooltip": f"Reference picture {i} for Ref2VA. Separate inputs keep each picture's own size."})
                   for i in range(1, 7)},
                "reference_images": ("IMAGE", {"tooltip": "Extra reference pictures as one IMAGE batch (same size), numbered after reference image 1-6."}),
                "reference_notes": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional, one line per reference picture, e.g. '男主' / '80年代院子' / '断绝关系书'."}),
                "source_video_frames": ("IMAGE", {"tooltip": "Frames of a source video (sampled) for editing / continuation / motion reference."}),
                "source_video_role": (list(VIDEO_ROLES), {"default": "editing source / 编辑源视频",
                                      "tooltip": "auto decides from the request text (replace/换成 → editing, 续拍 → continuation, "
                                                 "参考动作/跟着跳 → motion)."}),
                "source_video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001,
                                               "tooltip": "Frame rate of source_video_frames, used to time the detected hard cuts."}),
                "keep_source_audio": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Video edit: the source soundtrack is the final audio (<Audio 1> fully_copy)."}),
                "reference_audio_note": ("STRING", {"default": "", "tooltip": "If an audio reference is used, describe its role (the model cannot hear it), e.g. '男主音色参考'."}),
                "max_tokens": ("INT", {"default": 6000, "min": 1024, "max": 16384}),
                "context_length": ("INT", {"default": 32768, "min": 16384, "max": 131072, "step": 1024}),
                "source_audio": ("AUDIO", {"tooltip": "Optional source-video audio used to detect silence; when silent/empty, <Audio 1> fully_copy is omitted."}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("h3_prompt", "duration_seconds", "duration_int", "report")
    OUTPUT_TOOLTIPS = ("Official-format H3 prompt for the H3 conditioning node.", "Same duration as the input.",
                       "Duration rounded to whole seconds.", "Mode, checks, correction and timing.")
    FUNCTION = "run"
    CATEGORY = "TankNodes/H3"
    DESCRIPTION = ("Writes MiniMax H3 prompts that follow the official H3 prompt-writing guide, with a local model and format checks. "
                   "与 Prompt Optimizer + H3 模板等价，保留用于旧工作流。")

    def run(self, backend, request, mode, duration_seconds, thinking_mode, correction, unload_after, seed,
            first_frame=None, last_frame=None, reference_images=None, reference_notes="", source_video_frames=None,
            source_video_role="editing source / 编辑源视频", reference_audio_note="", max_tokens=6000, context_length=32768,
            source_video_fps=24.0, keep_source_audio=True, source_audio=None, **separate_refs):
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise ValueError("Connect Qwen / Bonsai GGUF Loader · Tank to backend.")
        refs: list[torch.Tensor] = []
        for i in range(1, 7):
            ref = separate_refs.get(f"reference_image_{i}")
            if ref is not None:
                refs.append(ref[:1] if getattr(ref, "ndim", 0) == 4 else ref)
        for frm in image_batch_frames(reference_images):
            refs.append(frm[None] if getattr(frm, "ndim", 0) == 3 else frm)
        extra_kwargs: dict[str, Any] = {}
        if mode != "auto":
            extra_kwargs["mode"] = mode
        if not bool(correction):
            extra_kwargs["correction"] = False
        if str(reference_notes or "").strip():
            extra_kwargs["reference_notes"] = str(reference_notes)
        if str(reference_audio_note or "").strip():
            extra_kwargs["reference_audio_note"] = str(reference_audio_note)
        if source_audio is not None:
            extra_kwargs["source_audio"] = source_audio
        started = time.perf_counter()
        try:
            configure = getattr(backend, "configure_chat", None)
            if callable(configure):
                configure(max(int(context_length), int(max_tokens) + 8192), thinking_mode)
            prompt, report = run_h3(
                backend,
                H3_MARKER,
                str(request or ""),
                seconds=float(duration_seconds),
                first_frame=first_frame,
                last_frame=last_frame,
                references=refs,
                video_frames=source_video_frames,
                video_role=VIDEO_ROLES.get(source_video_role, source_video_role),
                video_fps=float(source_video_fps),
                keep_source_audio=bool(keep_source_audio),
                thinking_mode=thinking_mode,
                max_tokens=int(max_tokens),
                seed=int(seed),
                **extra_kwargs,
            )
        finally:
            if unload_after and callable(getattr(backend, "unload", None)):
                backend.unload()
        report = report + f"\ntime={time.perf_counter() - started:.1f}s"
        print(f"[TankNodes] H3 prompt optimizer: {report.splitlines()[0]} {report.splitlines()[-1]}")
        analysis_failed = any(
            ln.strip() == "analysis=failed, generic fallback" or ln.strip().startswith("- analysis_failed:")
            for ln in report.splitlines()
        )
        return {
            "ui": {
                "text": (prompt + "\n\n---\n" + report,),
                "tank_prompt_meta": [{"has_report": True, "analysis_failed": analysis_failed}],
            },
            "result": (prompt, float(duration_seconds), int(round(float(duration_seconds))), report),
        }
