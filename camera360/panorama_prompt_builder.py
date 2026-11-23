"""Dual-mode 360-degree Equirectangular (2:1) Panorama Prompt Compiler.
Supports Mode 1 (extension: Image-to-Panorama 21% compact forward anchor + 3m clearance)
and Mode 2 (direct: Text-to-Panorama with camera re-centering override + unique landmark rule).
"""
from __future__ import annotations
from typing import Literal

PanoramaMode = Literal["direct", "extension"]
PanoramaDialect = Literal["structured_equirect", "concise_qwen_edit", "flux_skybox_lora"]

PANORAMA_NEGATIVE_PROMPT = (
    "flat rectilinear wide banner, circular fisheye border, little planet, "
    "tilted horizon, black nadir hole, tripod legs, camera rig, photographer shadow, "
    "visible vertical seam, hard cut edge, duplicated main building, cloned bed, "
    "oversized stretched foreground sofa, split screen, collage, watermark, text overlay"
)


def compile_panorama_prompts(
    scene_prompt: str = "",
    mode: str = "direct",
    dialect: str = "structured_equirect",
    indoor_anti_stretch: bool = True,
    lora_trigger: str = "TOK, equirectangular 360 panorama, vr skybox",
) -> dict[str, str]:
    clean = (scene_prompt or "").strip()
    clearance_rule = (
        "Maintain a generous 2.8–3.5m clear central walkable floor circle; "
        "place sofas, tables, beds, and cabinets back along room walls/perimeter "
        "to prevent close-range spherical bulging, and omit wide-angle lens terms."
        if indoor_anti_stretch
        else "Maintain at least 2.5m clear space around the central standing camera (~1.6m height)."
    )

    if dialect == "flux_skybox_lora":
        if mode == "extension":
            pos = (
                f"{lora_trigger}, seamless 360x180 spherical equirectangular outpainting, strict 2:1, "
                f"keep central 21% reference scene intact, {clearance_rule} "
                f"Level horizon at 50% midline, smooth sinusoidal ceiling/floor arcs, "
                f"seamless left-right wraparound at 180-degree rear meridian. {clean}"
            ).strip()
        else:
            pos = (
                f"{lora_trigger}, full 360x180 spherical equirectangular environment map, strict 2:1. "
                f"Scene: {clean}. Override any single-corner framing and place camera in open center. "
                f"{clearance_rule} Level horizon at 50% height, unique landmarks without cloning, "
                f"clean zenith ceiling/sky and nadir floor, seamless periodic left-right edges."
            ).strip()
        return {"positive": pos, "negative": PANORAMA_NEGATIVE_PROMPT}

    if dialect == "concise_qwen_edit":
        if mode == "extension":
            pos = (
                "将画布正中央的参考场景向四周无缝扩展补全为一张严格的 2:1（360°×180°）球面等距柱状全景图（VR Skybox）。"
                "保持正中央现有景物、材质与主光源方向完全不变；家具向房间四周墙边退让留出 3 米中心空地；"
                "视平线严格水平居中（50%高度），顶部天花板/天空与底部地面呈现真实球面纬线弧度且无黑洞脚架；"
                f"图像最左边缘与最右边缘必须在身后 180° 经线处像素级无缝衔接闭合。{clean}"
            )
        else:
            pos = (
                f"生成一张严格的 2:1（360°×180°）无缝球面等距柱状全景图（VR Skybox 环境贴图）。场景内容：{clean}。"
                "忽略任何单向角落构图词，将相机置于场地中央人眼高度（周围留出 3 米空旷地面，室内家具退向四周墙边防止球面鼓包拉伸）；"
                "地平线严格锁定在画面垂直正中 50% 处，核心建筑/牌匾/主景物全图只出现一次绝不重复克隆；"
                "天顶天花板/天空与脚底地面连续闭合无黑洞无脚架，最左边缘与最右边缘在背后 180° 经线处严丝合缝连续拼接。"
            )
        return {"positive": pos, "negative": PANORAMA_NEGATIVE_PROMPT}

    if mode == "extension":
        note_line = f"\n- ADDITIONAL SCENE NOTE: {clean}" if clean else ""
        pos = "\n".join([
            "360-degree equirectangular panorama outpainting, VR skybox environment map, strict 2:1 aspect ratio, full spherical 360° × 180° lat-long projection (top edge = +90° zenith straight overhead, bottom edge = -90° nadir straight underfoot, center = 0° forward, left & right edges = ±180° behind camera seamlessly connected):",
            "Expand the input reference photograph into a complete, seamless 360° horizontal × 180° vertical equirectangular VR environment map while faithfully anchoring the reference scene.",
            "",
            "1. COMPACT CENTRAL FORWARD ANCHOR (~65°–70° WINDOW = ONLY 20%–22% OF PANORAMA WIDTH):",
            "- Map the input reference image strictly into a compact central 20%–22% horizontal window around the 0° forward equator so existing sofas, tables, and walls keep crisp, natural architectural proportions without wrapping or stretching wide!",
            f"- Preserve the exact architectural style, key furniture/structures, material textures, color grading, time of day, and 3D light direction of the reference image in the central forward zone — do NOT stretch the reference image across the entire 2:1 canvas, and do NOT invent a different room/place!{note_line}",
            f"- ANTI-BULGING INTERIOR CLEARANCE: {clearance_rule}",
            "",
            "2. 360° × 180° SPHERICAL OUTPAINTING & SEAM CONTINUITY:",
            "- HORIZON LOCK: Align a level horizontal horizon strictly along the exact vertical midline (50% height) of the 2:1 frame.",
            "- SURROUNDING 285° COMPLETION: Smoothly extend the left (-90°) and right (+90°) side walls, corridors, or landscape outward from the reference edges, and construct a coherent, physically logical rear environment (±180° behind the original camera) across the outer left and right quarters.",
            "- AUTHENTIC EQUIRECTANGULAR PROJECTION: Apply true spherical lat-long projection curvature toward the upper and lower latitudes so all lines appear straight inside a 3D spherical viewer (do NOT render a flat rectilinear banner or a circular fisheye).",
            "- CLEAN ZENITH & NADIR POLES: Complete the full overhead ceiling/roof/sky up to the top pole (0%–15%) and continuous floor/ground down to the bottom nadir pole (85%–100%) — NO black hole, NO tripod/equipment, NO pinched vortex smearing.",
            "- STRICT CYLINDRICAL PERIODIC BOUNDARY (SEAMLESS WRAPAROUND): Pixel column x=0 (left border) and pixel column x=width (right border) are the EXACT SAME 180° rear meridian sliced open — any wall, molding, floor plank, or road surface at the right edge MUST continue directly from the left edge with matching height and lighting, without duplicating the same piece of furniture near both borders!",
            "",
            "3. CONSISTENCY & CLEAN RENDER:",
            "- Keep world-space sunlight/lighting direction consistent across all 360 degrees. Do not add people if none exist in the reference, and omit all watermarks, UI labels, or borders.",
        ])
    else:
        scene_spec = (
            "\n".join([
                "1. SCENE ENVIRONMENT SPECIFICATION (Source Scene Description):",
                f'"""{clean}"""',
                f"- CAMERA RE-CENTERING OVERRIDE: Absorb 100% of the architecture, era, props, materials, weather, color palette, and lighting mood from the scene description above, BUT override any narrow single-direction framing phrases (such as \"looking from a corner\", \"viewed from the doorway\", \"close-up\", or \"tilted upward\"). Place a stationary tripod camera in an OPEN, UNOBSTRUCTED WALKABLE SPOT near the center of this environment at natural standing eye-level (~1.6m height). {clearance_rule}",
            ])
            if clean
            else f"1. CAMERA PLACEMENT: Place a stationary tripod camera in an open walkable center (~1.6m height). {clearance_rule}"
        )
        pos = "\n".join([
            "360-degree equirectangular panorama, VR skybox environment map, strict 2:1 aspect ratio, full spherical 360° × 180° lat-long projection (top edge = +90° zenith straight overhead, bottom edge = -90° nadir straight underfoot, center = 0° forward, left & right edges = ±180° behind camera seamlessly connected):",
            "Generate a single seamless, ultra-crisp, photorealistic 360-degree horizontal by 180-degree vertical equirectangular environment map for an interactive 3D spherical viewer.",
            "",
            scene_spec,
            "",
            "2. SPHERICAL EQUIRECTANGULAR GEOMETRY (CRITICAL):",
            "- HORIZON LOCK: Keep a perfectly level, flat horizontal horizon locked strictly along the exact vertical midline (50% height) of the 2:1 frame — zero camera pitch or roll.",
            "- 360° FOUR-DIRECTION SPATIAL LAYOUT: Arrange primary focal architecture around 0°, coherent side structures at ±90°, and complete the rear (±180°) with UNIQUE LANDMARK RULE (every named building, emblem, bed, or sign appears ONLY ONCE at its true azimuth — NEVER clone central motifs onto side/rear walls).",
            "- NATURAL EQUIRECTANGULAR CURVATURE: Horizontal ceiling beams, rooflines, and floor lines away from the equator must curve smoothly in natural sinusoidal arcs so they project straight inside a 3D sphere viewer.",
            "- ZENITH & NADIR: Continuous overhead sky/ceiling (0%–15%) and clean floor/ground (85%–100%) — NO black hole, NO tripod shadow, NO circular fisheye border.",
            "- STRICT CYLINDRICAL PERIODIC BOUNDARY: Pixel column x=0 and x=width are the EXACT SAME 180° rear meridian sliced open — structures touching the right edge MUST continue smoothly across the left edge!",
            "",
            "3. LIGHTING & CLEAN OUTPUT:",
            "- Single physically consistent 3D light source direction, rich photorealistic textures, no people unless requested, zero watermarks or borders.",
        ])
    return {"positive": pos, "negative": PANORAMA_NEGATIVE_PROMPT}
