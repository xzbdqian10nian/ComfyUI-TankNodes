"""MiniMax H3 long-video helper nodes.

The window math deliberately mirrors the GPU-validated h3lab reference.  Heavy
ComfyUI/Torch imports stay inside execution methods so the planning node remains
usable in lightweight test environments.
"""

from __future__ import annotations

import json
from typing import Any, Iterable


FPS = 24
AUDIO_SR = 32000
AUDIO_LATENT_FPS = 40


def lat_t(frames: int) -> int:
    """Convert an H3-valid pixel-frame count to video latent time."""
    assert frames >= 5 and (frames - 5) % 17 == 0, frames
    return (frames - 5) // 17 * 5 + 2


def px_start(latent_idx: int) -> int:
    """Convert a latent index on a five-latent block boundary to pixel frames."""
    assert latent_idx % 5 == 0, latent_idx
    return latent_idx // 5 * 17


def plan_windows(total_frames: int, win_frames: int, ov_frames: int) -> list[int]:
    """Mirror ComfyUI's static-standard context-window math for H3."""
    total_latents, win_latents = lat_t(total_frames), lat_t(win_frames)
    overlap_latents = min(lat_t(ov_frames), win_latents - 5)
    if total_latents <= win_latents:
        return [0]

    starts: list[int] = []
    step = win_latents - overlap_latents
    for start in range(0, total_latents, step):
        end = start + win_latents
        if end >= total_latents:
            starts.append(start - (end - total_latents))
            break
        starts.append(start)

    count = len(starts)
    regions = [
        min(max(int(((s + s + win_latents - 1) / (2 * total_latents)) * count), 0), count - 1)
        for s in starts
    ]
    assert regions == list(range(count)), f"窗口与条件区域不能一一对应：{regions}"
    return [px_start(s) for s in starts]


def pad17(frames: int) -> int:
    """Round up to the H3 frame grid 17*n+5."""
    n = max(0, -(-(frames - 5) // 17))
    return 17 * n + 5


def fit_total(
    needed: int,
    win: int = 226,
    overlaps: Iterable[int] = (56, 39, 73),
    max_extra: int = 17 * 6,
) -> tuple[int, int]:
    """Find the smallest valid total and overlap with one condition region/window."""
    overlaps = tuple(overlaps)
    base = pad17(needed)
    for extra in range(0, max_extra + 1, 17):
        for overlap in overlaps:
            try:
                plan_windows(base + extra, win, overlap)
                return base + extra, overlap
            except AssertionError:
                pass
    raise ValueError(f"找不到可用窗口规划：needed={needed} win={win}")


def _parse_overlaps(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError("重叠候选 overlap_candidates 必须是逗号分隔的整数，例如 56,39,73") from exc
    else:
        values = tuple(int(item) for item in value)
    if not values:
        raise ValueError("重叠候选 overlap_candidates 不能为空")
    return values


def _window_plan(frames_needed: int, context_length: int, overlap_candidates: str | Iterable[int]):
    if frames_needed < 5:
        raise ValueError(f"frames_needed 必须至少为 5，当前为 {frames_needed}")
    if context_length < 5 or (context_length - 5) % 17 != 0:
        raise ValueError(f"context_length 必须满足 17n+5，当前为 {context_length}")

    overlaps = _parse_overlaps(overlap_candidates)
    try:
        total, overlap = fit_total(frames_needed, context_length, overlaps)
        starts = plan_windows(total, context_length, overlap)
    except (AssertionError, ValueError) as exc:
        raise ValueError(
            f"找不到可用窗口规划：需要 {frames_needed} 帧，上下文 {context_length} 帧，重叠候选 {list(overlaps)}"
        ) from exc

    return {
        "total_frames": total,
        "context_length": context_length,
        "context_overlap": overlap,
        "window_starts": starts,
        "windows": len(starts),
        "frames_needed": frames_needed,
        "pad_frames": total - frames_needed,
    }


def _plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


class TankH3WindowPlan:
    """Calculate a valid H3 context-window plan without importing Torch."""

    DESCRIPTION = "计算 MiniMax H3 长视频的 17n+5 总帧数、上下文重叠和窗口起点。"
    CATEGORY = "TankNodes/H3 Long Video"
    RETURN_TYPES = ("INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("total_frames", "context_length", "context_overlap", "window_starts", "plan_json")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_needed": ("INT", {"default": 736, "min": 5, "step": 1}),
                "context_length": ("INT", {"default": 226, "min": 22, "step": 17}),
                "overlap_candidates": ("STRING", {"default": "56,39,73"}),
            }
        }

    def plan(self, frames_needed: int, context_length: int, overlap_candidates: str):
        plan = _window_plan(frames_needed, context_length, overlap_candidates)
        return (
            plan["total_frames"],
            plan["context_length"],
            plan["context_overlap"],
            json.dumps(plan["window_starts"], ensure_ascii=False),
            _plan_json(plan),
        )


class TankH3WindowedReferenceToVideo:
    """Encode one MiniMax H3 reference condition per planned context window."""

    DESCRIPTION = (
        "按 H3 长视频窗口分别编码参考条件；下游连接 MiniMaxH3ContextWindows，使用本节点输出的 "
        "context_overlap；context_length 与本节点输入保持一致（可由 TankH3WindowPlan 同时连接），"
        "并开启 split_conds_to_windows=True。split_conds=False 时下游也应关闭该开关。"
        "采样后用 ImageFromBatch 裁回 frames_needed。"
    )
    CATEGORY = "TankNodes/H3 Long Video"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "latent", "total_frames", "context_overlap", "plan_json")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "width": ("INT", {"default": 544, "min": 16, "step": 16}),
                "height": ("INT", {"default": 960, "min": 16, "step": 16}),
                "frames_needed": ("INT", {"default": 736, "min": 5, "step": 1}),
                "context_length": ("INT", {"default": 226, "min": 22, "step": 17}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "ref_image_size": (["match", "short_edge"], {"default": "match"}),
            },
            "optional": {
                "source_video": ("IMAGE", {"tooltip": "源视频帧；有连接时每个窗口将其对应片段作为 <Video 1>。"}),
                "image_1": ("IMAGE", {"tooltip": "<Picture 1> 参考图。"}),
                "image_2": ("IMAGE", {"tooltip": "<Picture 2> 参考图。"}),
                "image_3": ("IMAGE", {"tooltip": "<Picture 3> 参考图。"}),
                "image_4": ("IMAGE", {"tooltip": "<Picture 4> 参考图。"}),
                "image_5": ("IMAGE", {"tooltip": "<Picture 5> 参考图。"}),
                "image_6": ("IMAGE", {"tooltip": "<Picture 6> 参考图。"}),
                "image_7": ("IMAGE", {"tooltip": "<Picture 7> 参考图。"}),
                "image_8": ("IMAGE", {"tooltip": "<Picture 8> 参考图。"}),
                "image_9": ("IMAGE", {"tooltip": "<Picture 9> 参考图（H3 最多 9 张）。一个口接入多张（批量）时按顺序拆成连续的 Picture。"}),
                "window_prompts": ("STRING", {"default": "", "multiline": True, "tooltip": "JSON 字符串数组，长度必须等于窗口数。"}),
                "audio": ("AUDIO", {"tooltip": "锁定的目标音轨；编码后音频噪声遮罩为 0。"}),
                "split_conds": ("BOOLEAN", {"default": True, "tooltip": "按窗口分别编码条件；关闭时只编码一个条件供所有窗口使用（长视频带 source_video 时需开启，否则会报错避免静默截断）。"}),
            },
        }

    @staticmethod
    def _node_result(value: Any, index: int = 0):
        result = getattr(value, "result", value)
        return result[index]

    @staticmethod
    def _pad_source(source_video, total: int):
        if source_video.shape[0] == 0:
            raise ValueError("source_video 不能是空视频")
        if source_video.shape[0] >= total:
            return source_video[:total]
        import torch

        pad = source_video[-1:].repeat(total - source_video.shape[0], 1, 1, 1)
        return torch.cat((source_video, pad), dim=0)

    @staticmethod
    def _encode_held_audio(audio, audio_vae, total_frames: int, samples_video):
        import torch
        import torch.nn.functional as F
        import comfy.audio
        import comfy.nested_tensor

        n = round((total_frames + 1) / FPS * AUDIO_SR)
        wav = torch.zeros(2, n)
        if audio is not None:
            waveform = audio["waveform"][0].float()
            sr = audio["sample_rate"]
            waveform = waveform.repeat(2, 1) if waveform.shape[0] == 1 else waveform[:2]
            waveform = waveform[:, : round(total_frames / FPS * sr)]
            if sr != AUDIO_SR:
                waveform = comfy.audio.resample(waveform, sr, AUDIO_SR)
            wav = waveform
        wav = F.pad(wav, (0, max(0, n - wav.shape[1])))[:, :n]
        wav = F.pad(wav, (0, -n % 800))
        audio_latent = audio_vae.encode(wav[None].movedim(1, -1))
        audio_latent = audio_latent[..., : round(total_frames / FPS * AUDIO_LATENT_FPS)]
        latent = {
            "samples": comfy.nested_tensor.NestedTensor((samples_video, audio_latent)),
            "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(samples_video), torch.zeros_like(audio_latent))),
        }
        return latent

    def execute(
        self,
        clip,
        vae,
        audio_vae,
        width,
        height,
        frames_needed,
        context_length,
        prompt,
        ref_image_size,
        source_video=None,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        window_prompts="",
        audio=None,
        split_conds=True,
    ):
        from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, MiniMaxH3ReferenceToVideo

        plan = _window_plan(frames_needed, context_length, (56, 39, 73))
        starts = plan["window_starts"] if split_conds else [0]
        total = plan["total_frames"]
        plan = {**plan, "window_starts": starts, "windows": len(starts), "split_conds": split_conds}
        length = min(context_length, total)
        if not split_conds and source_video is not None and total > length:
            raise ValueError(
                f"长视频带 source_video（总帧数 {total} > 单窗 {length}）时必须开启 split_conds，"
                "否则后续源视频帧会被截断。"
            )

        prompts = [prompt] * len(starts)
        if window_prompts not in (None, ""):
            try:
                prompts = json.loads(window_prompts)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("window_prompts 必须是 JSON 字符串数组") from exc
            if not isinstance(prompts, list) or len(prompts) != len(starts) or not all(isinstance(p, str) for p in prompts):
                raise ValueError(f"window_prompts 长度必须等于窗口数 {len(starts)}，且每项为字符串")

        source = self._pad_source(source_video, total) if source_video is not None else None
        ref_list = []
        for image in (image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9):
            if image is None:
                continue
            # 一个口接入批量（如 TankImageBatch）时拆成连续的 Picture，否则模型只看到 Picture 1。
            ref_list.extend(image[i : i + 1] for i in range(image.shape[0])) if getattr(image, "ndim", 0) == 4 and image.shape[0] > 1 else ref_list.append(image)
        if len(ref_list) > 9:
            raise ValueError(f"H3 参考图最多 9 张，当前 {len(ref_list)} 张")
        ref_images = {f"ref_image_{index}": image for index, image in enumerate(ref_list)}

        positive = None
        for start, window_prompt in zip(starts, prompts):
            ref_videos = {}
            if source is not None:
                ref_videos["ref_video_0"] = source[start : start + length]
            result = MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                prompt=window_prompt,
                width=width,
                height=height,
                length=length,
                ref_image_size=ref_image_size,
                vae=vae,
                audio_vae=audio_vae,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios={},
                ref_audios={},
            )
            condition = self._node_result(result, 0)
            positive = condition if positive is None else positive + condition

        empty = EmptyMiniMaxH3LatentAV.execute(width=width, height=height, length=total)
        latent = self._node_result(empty, 0)
        if audio is not None:
            samples = latent["samples"]
            video = samples.unbind()[0]
            latent = self._encode_held_audio(audio, audio_vae, total, video)

        return positive, latent, total, plan["context_overlap"], _plan_json(plan)


class TankH3MaskedSourceLatent:
    """遮罩内重绘、遮罩外锁定原内容；多人换人时每遍只替换一个人。"""

    DESCRIPTION = "遮罩内重绘、遮罩外锁定原内容；用于多人换人时每遍只替换一个人。"
    CATEGORY = "TankNodes/H3 Long Video"
    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "mask_preview")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "images": ("IMAGE", {"tooltip": "源视频帧，24fps，帧数 17n+5，尺寸即生成尺寸（16 的倍数）。"}),
                "mask": ("MASK", {"tooltip": "要重绘的区域（每帧一张，或单张静态）。"}),
                "dilate": ("INT", {"default": 24, "min": 0, "max": 256}),
                "invert": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "bbox_expand": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": ">0 时把每帧遮罩外接框四周各扩大该比例并并入遮罩。"}),
                "protect": ("MASK", {"tooltip": "其它角色等需要保持不变的区域，从重绘区域中扣除。"}),
                "temporal_dilate": ("INT", {"default": 0, "min": 0, "max": 48, "tooltip": "遮罩在时间上前后各扩展的帧数。"}),
            },
        }

    @staticmethod
    def _fit(m, nf, h, w):
        import torch
        import torch.nn.functional as F

        m = m.float()
        if m.dim() == 2:
            m = m[None]
        if m.shape[0] == 1:
            m = m.repeat(nf, 1, 1)
        m = m[:nf]
        if m.shape[0] < nf:
            m = torch.cat([m, m[-1:].repeat(nf - m.shape[0], 1, 1)])
        if (m.shape[1], m.shape[2]) != (h, w):
            m = F.interpolate(m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        return m

    def run(self, vae, audio_vae, images, mask, dilate, invert, audio=None, bbox_expand=0.0, protect=None, temporal_dilate=0):
        import torch
        import torch.nn.functional as F
        import comfy.audio
        import comfy.nested_tensor
        import comfy.ldm.minimax.model as mm

        nf, h, w = images.shape[0], images.shape[1], images.shape[2]
        assert (nf - 5) % 17 == 0, f"帧数需为 17n+5，当前 {nf}"
        assert h % 16 == 0 and w % 16 == 0, f"尺寸需为 16 的倍数，当前 {w}x{h}"
        video = vae.encode(images[..., :3])
        latent_t = video.shape[2]
        m = self._fit(mask, nf, h, w)
        if invert:
            m = 1.0 - m
        if dilate > 0:
            m = F.max_pool2d(m[:, None], kernel_size=2 * dilate + 1, stride=1, padding=dilate)[:, 0]
        m = (m > 0.5).float()
        if bbox_expand > 0:
            for i in range(nf):
                ys, xs = torch.nonzero(m[i] > 0, as_tuple=True)
                if len(ys) == 0:
                    continue
                y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()
                dy = int((y1 - y0 + 1) * bbox_expand)
                dx = int((x1 - x0 + 1) * bbox_expand)
                m[i, max(0, y0 - dy) : min(h, y1 + dy + 1), max(0, x0 - dx) : min(w, x1 + dx + 1)] = 1.0
        if temporal_dilate > 0:
            kernel = 2 * temporal_dilate + 1
            m = F.max_pool1d(
                m.permute(1, 2, 0).reshape(-1, 1, nf),
                kernel_size=kernel,
                stride=1,
                padding=temporal_dilate,
            ).reshape(h, w, nf).permute(2, 0, 1)
        if protect is not None:
            protected = (self._fit(protect, nf, h, w) > 0.5).float()
            m = m * (1.0 - protected)
        preview = m.clone()

        starts = [0]
        for index in range(latent_t):
            starts.append(starts[-1] + mm.FRAME_PER_TOKEN[index % 5])
        spans = [(starts[index], starts[index + 1]) for index in range(latent_t)]
        assert spans[-1][1] == nf, f"latent 时间跨度 {spans[-1][1]} 与帧数 {nf} 不一致"
        mask_time = torch.stack([m[start:end].amax(0) for start, end in spans])
        mask_latent = F.max_pool2d(mask_time[:, None], kernel_size=16, stride=16)[:, 0]
        video_mask = mask_latent[None, None].to(video.dtype).expand_as(video).contiguous()

        n = round((nf + 1) / FPS * AUDIO_SR)
        wav = torch.zeros(2, n)
        if audio is not None:
            wav, sr = audio["waveform"][0].float(), audio["sample_rate"]
            wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]
            wav = wav[:, : round(nf / FPS * sr)]
            if sr != AUDIO_SR:
                wav = comfy.audio.resample(wav, sr, AUDIO_SR)
        wav = F.pad(wav, (0, max(0, n - wav.shape[1])))[:, :n]
        wav = F.pad(wav, (0, -n % 800))
        audio_latent = audio_vae.encode(wav[None].movedim(1, -1))[..., : round(nf / FPS * AUDIO_LATENT_FPS)]
        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio_latent)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, torch.zeros_like(audio_latent))),
        }
        return latent, preview


NODE_CLASS_MAPPINGS = {
    "TankH3WindowPlan": TankH3WindowPlan,
    "TankH3WindowedReferenceToVideo": TankH3WindowedReferenceToVideo,
    "TankH3MaskedSourceLatent": TankH3MaskedSourceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TankH3WindowPlan": "H3 Window Plan · Tank",
    "TankH3WindowedReferenceToVideo": "H3 Windowed Reference to Video · Tank",
    "TankH3MaskedSourceLatent": "H3 Masked Source Latent · Tank",
}
