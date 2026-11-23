"""TankNodes Camera & 360° Equirectangular Panorama Suite (4 consolidated nodes):

1. TankMultiAngleCamera      — 3D 多角度相机 · Tank (Three.js 3-handle gizmo + V6 / <qma> / <sks> multi-dialect compiler)
2. TankPanoramaSetup         — 360° 全景构建 · Tank (direct/extension dual-mode prompt + strict 2:1 size + 21% narrow-FOV canvas & cosine mask)
3. TankPanoramaSeamlessPatch — 360° 全景无缝补丁 · Tank (Dynamic-VRAM-safe X-only circular + Y replicate Conv2d patch)
4. TankPanoramaViewer        — 360° 全景球面预览与切图 · Tank (Three.js 360° interactive sphere viewer + 58° anti-stretch view slicer)
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import uuid

try:
    import folder_paths
except ImportError:
    folder_paths = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .camera_prompt_builder import compile_multi_angle_prompts
from .panorama_prompt_builder import compile_panorama_prompts

CATEGORY_CAMERA = "TankNodes/Camera"


def _save_temp_preview(image_tensor: torch.Tensor, prefix: str, max_side: int = 2048) -> dict[str, object]:
    frame = image_tensor[0].detach().cpu().clamp(0.0, 1.0).numpy()
    arr = (frame * 255.0).round().astype(np.uint8)
    im = Image.fromarray(arr)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    temp_dir = folder_paths.get_temp_directory() if folder_paths is not None else tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    fname = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    im.save(os.path.join(temp_dir, fname), format="PNG", compress_level=3)
    return {"filename": fname, "subfolder": "", "type": "temp"}


class TankMultiAngleCamera:
    """3D 多角度交互舞台相机（内嵌 Three.js 粉环/青弧/黄球拖拽舞台 + 10 机位预设 + V6 / QMA / SKS 方言编译）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "horizontal_angle": ("INT", {"default": 0, "min": 0, "max": 345, "step": 15}),
                "vertical_angle": ("INT", {"default": 0, "min": -30, "max": 90, "step": 15}),
                "shot_scale": (["wide", "medium_wide", "medium", "medium_close_up", "close_up"], {"default": "medium"}),
                "prompt_dialect": (["structured_v6", "qma_lora", "sks_lora", "concise_edit"], {"default": "structured_v6"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "extra_style_hint": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "camera_json", "passthrough_image")
    FUNCTION = "compile"
    CATEGORY = CATEGORY_CAMERA

    def compile(
        self,
        horizontal_angle: int,
        vertical_angle: int,
        shot_scale: str,
        prompt_dialect: str,
        reference_image: torch.Tensor | None = None,
        extra_style_hint: str = "",
    ):
        res = compile_multi_angle_prompts(
            horizontal_angle=horizontal_angle,
            vertical_angle=vertical_angle,
            shot_scale=shot_scale,
            dialect=prompt_dialect,
        )
        pos = str(res["positive_prompt"])
        if extra_style_hint and extra_style_hint.strip():
            pos = f"{pos}\n\nAdditional Context: {extra_style_hint.strip()}"
        cam_meta = json.dumps(
            {
                "horizontal_angle": res["horizontal_angle"],
                "vertical_angle": res["vertical_angle"],
                "shot_scale": res["shot_scale"],
                "dialect": prompt_dialect,
            },
            ensure_ascii=False,
        )
        out_img = reference_image if reference_image is not None else torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        ui_data = {}
        if reference_image is not None:
            ui_data["tank_cam_ref"] = [_save_temp_preview(reference_image, "tank_cam_ref", max_side=768)]
        return {"ui": ui_data, "result": (pos, res["negative_prompt"], cam_meta, out_img)}


class TankPanoramaSetup:
    """360° 球面全景一体化构建节点（文生 direct / 图扩 extension 双模提示词 + 严格 2:1 尺寸 + 21% 赤道窄视窗画布与余弦羽化遮罩）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["auto", "direct", "extension"], {"default": "auto"}),
                "dialect": (["structured_equirect", "concise_qwen_edit", "flux_skybox_lora"], {"default": "structured_equirect"}),
                "pano_width": ("INT", {"default": 2880, "min": 1536, "max": 4096, "step": 64}),
                "indoor_anti_stretch": ("BOOLEAN", {"default": True}),
                "anchor_width_ratio": ("FLOAT", {"default": 0.21, "min": 0.15, "max": 0.35, "step": 0.01}),
                "feather_px": ("INT", {"default": 24, "min": 0, "max": 128, "step": 4}),
                "scene_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "lora_trigger": ("STRING", {"default": "TOK, equirectangular 360 panorama, vr skybox"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "IMAGE", "MASK")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "width_2to1", "height_2to1", "padded_canvas", "outpaint_mask")
    FUNCTION = "setup"
    CATEGORY = CATEGORY_CAMERA

    def setup(
        self,
        mode: str,
        dialect: str,
        pano_width: int,
        indoor_anti_stretch: bool,
        anchor_width_ratio: float,
        feather_px: int,
        scene_prompt: str,
        reference_image: torch.Tensor | None = None,
        lora_trigger: str = "TOK, equirectangular 360 panorama, vr skybox",
    ):
        w_out = max(1024, (int(pano_width) // 64) * 64)
        h_out = w_out // 2
        actual_mode = ("extension" if reference_image is not None else "direct") if mode == "auto" else mode

        prompts = compile_panorama_prompts(
            scene_prompt=scene_prompt,
            mode=actual_mode,
            dialect=dialect,
            indoor_anti_stretch=indoor_anti_stretch,
            lora_trigger=lora_trigger,
        )

        if reference_image is None:
            empty_canvas = torch.zeros((1, h_out, w_out, 3), dtype=torch.float32)
            empty_mask = torch.ones((1, h_out, w_out), dtype=torch.float32)
            return (prompts["positive"], prompts["negative"], w_out, h_out, empty_canvas, empty_mask)

        b, h, w, _ = reference_image.shape
        target_w = max(64, int(round(w_out * anchor_width_ratio)))
        target_h = min(int(round(h_out * 0.62)), max(64, int(round(target_w * h / w))))

        resized = (
            F.interpolate(reference_image.permute(0, 3, 1, 2), size=(target_h, target_w), mode="bicubic", align_corners=False)
            .permute(0, 2, 3, 1)
            .clamp(0.0, 1.0)
        )
        bg = F.interpolate(reference_image.permute(0, 3, 1, 2), size=(16, 32), mode="bilinear", align_corners=False)
        bg = F.interpolate(bg, size=(h_out, w_out), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).clamp(0.0, 1.0)
        mean_rgb = reference_image.mean(dim=(1, 2), keepdim=True)
        canvas = (0.65 * bg + 0.35 * mean_rgb).clamp(0.0, 1.0)

        x0 = (w_out - target_w) // 2
        y0 = (h_out - target_h) // 2
        canvas[:, y0 : y0 + target_h, x0 : x0 + target_w, :] = resized

        mask = torch.ones((b, h_out, w_out), dtype=reference_image.dtype, device=reference_image.device)
        mask[:, y0 : y0 + target_h, x0 : x0 + target_w] = 0.0
        if feather_px > 0:
            pad = min(feather_px, target_w // 4, target_h // 4)
            yy = torch.arange(target_h, device=reference_image.device).float()
            xx = torch.arange(target_w, device=reference_image.device).float()
            wy = (torch.minimum(yy, (target_h - 1) - yy).clamp(0, pad)) / max(1, pad)
            wx = (torch.minimum(xx, (target_w - 1) - xx).clamp(0, pad)) / max(1, pad)
            keep = 0.5 - 0.5 * torch.cos(math.pi * torch.minimum(wy[:, None], wx[None, :]))
            mask[:, y0 : y0 + target_h, x0 : x0 + target_w] = 1.0 - keep
        return (prompts["positive"], prompts["negative"], w_out, h_out, canvas, mask)


def _make_comfy_safe_x_circular_forward(orig_forward, orig_padding):
    """Wraps ComfyUI's native Conv2d.forward so comfy-aimdo Dynamic VRAM weight staging,
    FP8/INT8/W4A8 quantized ops, and LoRA hooks all execute intact while replacing X-padding with circular."""
    py, px = int(orig_padding[0]), int(orig_padding[1])

    def _forward(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        mod = orig_forward.__self__
        if isinstance(x, torch.Tensor) and x.ndim == 4 and px > 0 and x.shape[-1] > px:
            x = F.pad(x, (px, px, 0, 0), mode="circular")
            if py > 0 and x.shape[-2] > py:
                x = F.pad(x, (0, 0, py, py), mode="replicate")
                temp_pad = (0, 0)
            else:
                temp_pad = (py, 0)
            mod.padding = temp_pad
            try:
                return orig_forward(x, *args, **kwargs)
            finally:
                mod.padding = orig_padding
        return orig_forward(x, *args, **kwargs)

    return _forward


class TankPanoramaSeamlessPatch:
    """非对称 X 轴圆柱周期无缝卷积补丁（仅水平 X 轴 circular、垂直 Y 轴 replicate，完整兼容 ComfyUI 0.45+ Dynamic VRAM）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "patch"
    CATEGORY = CATEGORY_CAMERA

    def patch(self, model, enable: bool = True, vae=None):
        if not enable:
            return (model, vae)

        m_clone = model.clone()
        diff_model = getattr(m_clone.model, "diffusion_model", None)
        if diff_model is not None:
            for name, m in diff_model.named_modules():
                if isinstance(m, nn.Conv2d) and isinstance(m.padding, tuple) and len(m.padding) == 2:
                    py, px = int(m.padding[0]), int(m.padding[1])
                    if px > 0:
                        fn = _make_comfy_safe_x_circular_forward(m.forward, (py, px))
                        attr_path = f"diffusion_model.{name}.forward" if name else "diffusion_model.forward"
                        m_clone.add_object_patch(attr_path, fn)

        if vae is None:
            return (m_clone, None)

        vae_clone = copy.copy(vae)
        if hasattr(vae, "patcher") and vae.patcher is not None:
            vae_clone.patcher = vae.patcher.clone()
            fsm = getattr(vae, "first_stage_model", None)
            if fsm is not None:
                for name, m in fsm.named_modules():
                    if isinstance(m, nn.Conv2d) and isinstance(m.padding, tuple) and len(m.padding) == 2:
                        py, px = int(m.padding[0]), int(m.padding[1])
                        if px > 0:
                            fn = _make_comfy_safe_x_circular_forward(m.forward, (py, px))
                            attr_path = f"{name}.forward" if name else "forward"
                            vae_clone.patcher.add_object_patch(attr_path, fn)
        return (m_clone, vae_clone)


class TankPanoramaViewer:
    """360° 全景球面交互预览与多机位逆投影切图二合一节点（内嵌 Three.js 360° 球面查看器 + 黄金 58° VFOV 防拉伸切图）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "equirect_image": ("IMAGE",),
                "default_vfov": ("FLOAT", {"default": 55.0, "min": 35.0, "max": 90.0, "step": 1.0}),
                "slice_preset": (["4_cardinal_views", "12_orbit_views", "single_custom_view", "none"], {"default": "4_cardinal_views"}),
                "slice_vfov": ("FLOAT", {"default": 58.0, "min": 38.0, "max": 95.0, "step": 1.0}),
                "slice_width": ("INT", {"default": 1280, "min": 256, "max": 2560, "step": 16}),
                "slice_height": ("INT", {"default": 720, "min": 256, "max": 1440, "step": 16}),
                "custom_yaw_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 15.0}),
                "custom_pitch_deg": ("FLOAT", {"default": 0.0, "min": -85.0, "max": 85.0, "step": 5.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("equirect_image", "sliced_views")
    FUNCTION = "preview_and_slice"
    CATEGORY = CATEGORY_CAMERA
    OUTPUT_NODE = True

    def preview_and_slice(
        self,
        equirect_image: torch.Tensor,
        default_vfov: float = 55.0,
        slice_preset: str = "4_cardinal_views",
        slice_vfov: float = 58.0,
        slice_width: int = 1280,
        slice_height: int = 720,
        custom_yaw_deg: float = 0.0,
        custom_pitch_deg: float = 0.0,
    ):
        meta = _save_temp_preview(equirect_image, "tank_pano360", max_side=2880)
        meta["vfov"] = float(default_vfov)

        if slice_preset == "none":
            return {"ui": {"tank_pano": [meta]}, "result": (equirect_image, equirect_image)}

        angles = (
            [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0)]
            if slice_preset == "4_cardinal_views"
            else ([(i * 30.0, 0.0) for i in range(12)] if slice_preset == "12_orbit_views" else [(custom_yaw_deg, custom_pitch_deg)])
        )
        src = equirect_image[0:1].permute(0, 3, 1, 2).float()
        _, _, eh, ew = src.shape
        device, dtype = src.device, src.dtype
        fy = 0.5 * slice_height / math.tan(math.radians(slice_vfov) * 0.5)
        xs = (torch.arange(slice_width, device=device, dtype=dtype) - (slice_width - 1) * 0.5) / fy
        ys = ((slice_height - 1) * 0.5 - torch.arange(slice_height, device=device, dtype=dtype)) / fy
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gz = torch.ones_like(gx)
        norm = torch.sqrt(gx * gx + gy * gy + gz * gz)
        dx, dy, dz = gx / norm, gy / norm, gz / norm

        src_wrap = torch.cat([src, src[:, :, :, :1]], dim=-1)
        views = []
        for yaw_d, pitch_d in angles:
            p, y = math.radians(pitch_d), math.radians(yaw_d)
            y1 = dy * math.cos(p) + dz * math.sin(p)
            z1 = -dy * math.sin(p) + dz * math.cos(p)
            wx = dx * math.cos(y) + z1 * math.sin(y)
            wz = -dx * math.sin(y) + z1 * math.cos(y)
            u = (torch.atan2(wx, wz) / (2.0 * math.pi) + 0.5) % 1.0
            v = (0.5 - torch.asin(y1.clamp(-1.0, 1.0)) / math.pi).clamp(0.0, 1.0)
            grid_x = u * 2.0 - 1.0
            grid_y = v * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            sampled = F.grid_sample(src_wrap, grid, mode="bilinear", padding_mode="border", align_corners=True)
            views.append(sampled.permute(0, 2, 3, 1))

        sliced = torch.cat(views, dim=0).clamp(0.0, 1.0)
        return {"ui": {"tank_pano": [meta]}, "result": (equirect_image, sliced)}


NODE_CLASS_MAPPINGS = {
    "TankMultiAngleCamera": TankMultiAngleCamera,
    "TankPanoramaSetup": TankPanoramaSetup,
    "TankPanoramaSeamlessPatch": TankPanoramaSeamlessPatch,
    "TankPanoramaViewer": TankPanoramaViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TankMultiAngleCamera": "3D Multi-Angle Camera · Tank",
    "TankPanoramaSetup": "360 Panorama Setup · Tank",
    "TankPanoramaSeamlessPatch": "360 Panorama Seamless Patch · Tank",
    "TankPanoramaViewer": "360 Panorama Viewer & Slicer · Tank",
}
