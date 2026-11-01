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

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .chat import _collect_stream, _extract_completion, _split_message
from .media import image_batch_frames, pil_to_data_url, sample_image_batch, tensor_frame_to_pil
from .reasoning import REASONING_CHOICES, validate_reasoning_choice

_SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_skill")


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
}
CJK = re.compile(r"[㐀-鿿＀-￯　-〿]")


@dataclass
class Asset:
    kind: str                 # picture | video | audio
    role: str                 # first_frame | last_frame | reference | editing | continuation | motion | audio
    note: str = ""
    data_urls: list[str] = field(default_factory=list)


@dataclass
class Job:
    request: str
    duration: float
    assets: list[Asset] = field(default_factory=list)
    mode: str = "auto"
    dialogue: list[str] | None = None


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
    rules += [
        "[Shot 1] has no timestamp. Each later shot starts with '[Shot N] At MM:SS.mmm,' with strictly increasing times "
        f"strictly below {_sec(job.duration)} seconds. Leave time for every spoken line (about 4 Chinese characters per second).",
        "All descriptive prose is English. Dialogue and lyrics stay in their original language inside <d>[Chinese] ...</d> "
        "(or the matching language tag); quoted on-screen text stays verbatim inside English double quotes.",
        "Every vocal source gets a stable speaker ID (S1), (S2), ...; attach it together with a short voice description "
        "(age, gender, timbre, delivery) in the sentence where that person actually speaks, not at an earlier silent appearance.",
        "Camera movement uses the guide's vocabulary in natural sentences with amplitude and speed when meaningful; a big "
        "framing change (for example close-up to full body) is 'with large amplitude'.",
        "On-screen text: only quote text the user wrote. Signs, couplets or labels in the pictures that the user did not spell "
        "out are described without quoting or guessing their characters; never invent names from unreadable signage.",
        "overall_soundscape: 1-4 sentences of ambience and physical sounds, no dialogue. non_diegetic_music: 1-3 sentences "
        "on instrumentation, tempo and dynamics, or N/A.",
    ]
    if allowed:
        rules.append("The only reference labels that exist are: " + ", ".join(allowed) + ". Never invent other "
                     "<Picture>/<Video>/<Audio> labels." + (" Use <Subject N> for reusable visible content (people, places, "
                                                          "props, styles) defined from these assets." if is_ref else ""))
    else:
        rules.append("There are no reference assets: use no <Picture>, <Video>, <Audio> or <Subject> labels.")
    if is_ref:
        rules.append("<Video N> is only for a whole source video that is edited, continued, or used for camera/pacing "
                     "structure; people, places and props are always <Subject N>. detailed_description opens with one or two "
                     "style sentences before [Shot 1] and normally has 350-500 English words.")
        task_types = []
        if any(a.role in ("first_frame", "last_frame") for a in pics):
            task_types.append("keyframe completion")
        if any(a.role == "reference" for a in pics) or any(v.role == "motion" for v in videos):
            task_types.append("reference generation")
        if any(v.role == "editing" for v in videos):
            task_types.insert(0, "video editing")
            rules.append("This is a video edit: after the task prefix the summary begins with 'The target video is an edited "
                         "version of <Video 1>.' and the edit keeps everything the user did not ask to change.")
        if any(v.role == "continuation" for v in videos):
            task_types.insert(0, "video continuation")
        if audios:
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

    role_text = {"first_frame": "first frame of the target video", "last_frame": "last frame of the target video",
                 "reference": "reference picture", "editing": "source video to edit", "continuation": "source video to continue",
                 "motion": "motion and camera reference video", "audio": "reference audio"}
    content: list[dict[str, Any]] = [{"type": "text", "text": "User request (may be Chinese):\n" + job.request.strip()}]
    manifest: list[str] = []
    for i, a in enumerate(pics, 1):
        manifest.append(f"Picture {i}: {role_text[a.role]}" + (f" — {a.note}" if a.note else ""))
    for i, a in enumerate(videos, 1):
        manifest.append(f"Video {i}: {role_text[a.role]} ({len(a.data_urls)} sampled frames)" + (f" — {a.note}" if a.note else ""))
    for i, a in enumerate(audios, 1):
        manifest.append(f"Audio {i}: {a.note or 'reference audio'} (you cannot hear it; use the note)")
    if manifest:
        content.append({"type": "text", "text": "Attached assets:\n" + "\n".join(manifest)})
    for i, a in enumerate(pics, 1):
        content.append({"type": "text", "text": f"Picture {i}:"})
        content.append({"type": "image_url", "image_url": {"url": a.data_urls[0]}})
    for i, a in enumerate(videos, 1):
        for k, url in enumerate(a.data_urls, 1):
            content.append({"type": "text", "text": f"Video {i} frame {k}/{len(a.data_urls)}:"})
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


def validate(text: str, mode: str, job: Job) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    def add(code: str, msg: str, hard: bool = True):
        issues.append({"code": code, "msg": msg, "hard": hard})

    is_ref = mode == "Ref2VA"
    names = REF_SECTIONS if is_ref else BASE_SECTIONS
    sec = split_sections(text, names)
    if sec["__order"] != names:
        add("sections", f"sections must be exactly {names} in this order; found {sec['__order']}")
    body = sec.get("detailed_description" if is_ref else BASE_SECTIONS[0], "")
    found = shots(body)
    if not found or found[0][0] != 1:
        add("shot1", "the main description must start with [Shot 1]")
    else:
        if found[0][1] is not None:
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
        if not re.match(r"\s*\[[a-z +]+\]", sec.get("summary", "")):
            add("summary", "summary must start with a [task type] prefix")
        if any(a.role == "editing" for a in job.assets) and "edited version of <Video 1>" not in sec.get("summary", ""):
            add("summary", "a video edit summary must contain 'The target video is an edited version of <Video 1>.'")
        words = len(re.findall(r"[A-Za-z']+", body))
        if words < 300:
            add("length", f"detailed_description has {words} English words; target 350-500", hard=False)
    dialogue = job.dialogue if job.dialogue is not None else extract_dialogue(job.request)
    blocks = re.findall(r"<d>(.*?)</d>", text, re.S)
    for line in dialogue:
        if not any(line in block for block in blocks):
            add("dialogue", f"this spoken line must appear verbatim inside <d>: {line}")
    for block in blocks:
        if not re.match(r"\s*\[[A-Za-z ]+\]", block):
            add("dialogue_tag", "each <d> block must start with a [Language] tag", hard=False)
    if blocks and not re.search(r"\(S\d+\)", text):
        add("speaker", "vocal events need speaker IDs such as (S1)")
    without_dialogue = re.sub(r"<d>.*?</d>", "", text, flags=re.S)
    for quoted in re.findall(r"\"([^\"]*)\"", without_dialogue):
        if CJK.search(quoted) and quoted not in job.request:
            add("onscreen_text", f"quoted on-screen text \"{quoted}\" is not in the user request; describe that sign without quoting it")
    if len(CJK.findall(re.sub(r"\"[^\"]*\"", "", without_dialogue))) > 6:
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
    if mode == "FL2VA":
        out = re.sub(r"<Picture (\d)>", r"Picture \1", out)   # official FL2VA wording
    return out


# ------------------------------------------------------------------ backend call
def _complete(backend: Any, messages: list[dict[str, Any]], *, max_tokens: int, thinking_mode: str, seed: int,
              sampling: dict[str, Any] | None = None) -> tuple[str, dict]:
    kwargs: dict[str, Any] = {"messages": messages, "max_tokens": int(max_tokens), "temperature": 0.6, "top_p": 0.95,
                              "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.05, "seed": int(seed), "stream": False}
    kwargs.update({k: v for k, v in (sampling or {}).items() if v is not None})
    # Same rule as VisionChat: only server/API backends take the unified thinking selector.
    if getattr(backend, "backend_kind", "") in {"openai_compatible", "local_bonsai_server", "local_llama_server"}:
        kwargs["thinking_mode"] = thinking_mode
    result = backend.complete(**kwargs)
    result = _collect_stream(result)
    message, usage = _extract_completion(result)
    response, _reasoning, _raw = _split_message(message)
    return response or "", usage


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


def _frames_to_urls(batch: torch.Tensor | None, limit: int | None = None) -> list[list[str]]:
    if batch is None:
        return []
    frames = sample_image_batch(batch, limit) if limit else image_batch_frames(batch)
    return [[pil_to_data_url(tensor_frame_to_pil(frame), 88)] for frame in frames]


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
                "mode": (["auto", *MODES], {"default": "auto", "tooltip": "auto picks the official H3 mode from the connected media."}),
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
                "source_video_role": (list(VIDEO_ROLES), {"default": "editing source / 编辑源视频"}),
                "reference_audio_note": ("STRING", {"default": "", "tooltip": "If an audio reference is used, describe its role (the model cannot hear it), e.g. '男主音色参考'."}),
                "max_tokens": ("INT", {"default": 6000, "min": 1024, "max": 16384}),
                "context_length": ("INT", {"default": 32768, "min": 16384, "max": 131072, "step": 1024}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("h3_prompt", "duration_seconds", "duration_int", "report")
    OUTPUT_TOOLTIPS = ("Official-format H3 prompt for the H3 conditioning node.", "Same duration as the input.",
                       "Duration rounded to whole seconds.", "Mode, checks, correction and timing.")
    FUNCTION = "run"
    CATEGORY = "TankNodes/H3"
    DESCRIPTION = "Writes MiniMax H3 prompts that follow the official H3 prompt-writing guide, with a local model and format checks."
    OUTPUT_NODE = True

    def run(self, backend, request, mode, duration_seconds, thinking_mode, correction, unload_after, seed,
            first_frame=None, last_frame=None, reference_images=None, reference_notes="", source_video_frames=None,
            source_video_role="editing source / 编辑源视频", reference_audio_note="", max_tokens=6000, context_length=32768,
            **separate_refs):
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise ValueError("Connect Qwen / Bonsai GGUF Loader · Tank to backend.")
        notes = [n.strip() for n in str(reference_notes or "").splitlines() if n.strip()]
        assets: list[Asset] = []
        for role, batch in (("first_frame", first_frame), ("last_frame", last_frame)):
            for urls in _frames_to_urls(batch)[:1]:
                assets.append(Asset("picture", role, "", urls))
        ref_urls: list[list[str]] = []
        for i in range(1, 7):
            ref_urls += _frames_to_urls(separate_refs.get(f"reference_image_{i}"))[:1]
        ref_urls += _frames_to_urls(reference_images)
        for i, urls in enumerate(ref_urls):
            assets.append(Asset("picture", "reference", notes[i] if i < len(notes) else "", urls))
        if source_video_frames is not None:
            frames = [u[0] for u in _frames_to_urls(source_video_frames, 8)]
            assets.append(Asset("video", VIDEO_ROLES.get(source_video_role, "editing"), "", frames))
        if str(reference_audio_note or "").strip():
            assets.append(Asset("audio", "audio", str(reference_audio_note).strip()))
        job = Job(str(request or ""), float(duration_seconds), assets, mode)
        try:
            configure = getattr(backend, "configure_chat", None)
            if callable(configure):
                configure(int(context_length), thinking_mode)
            result = optimize(backend, job, thinking_mode=thinking_mode, max_tokens=int(max_tokens), seed=int(seed),
                              correction=bool(correction))
        finally:
            if unload_after and callable(getattr(backend, "unload", None)):
                backend.unload()
        hard = [i for i in result["issues"] if i["hard"]]
        lines = [f"mode={result['mode']}", f"time={result['seconds']:.1f}s", f"corrected={result['corrected']}",
                 f"first_draft_hard_issues={sum(i['hard'] for i in result['first_issues'])}",
                 f"final_hard_issues={len(hard)}"]
        lines += [f"- {i['code']}: {i['msg']}" for i in result["issues"]]
        report = "\n".join(lines)
        print(f"[TankNodes] H3 prompt optimizer: {lines[0]} {lines[1]} final_hard_issues={len(hard)}")
        prompt = result["prompt"]
        return {"ui": {"text": (prompt + "\n\n---\n" + report,)},
                "result": (prompt, float(duration_seconds), int(round(float(duration_seconds))), report)}
