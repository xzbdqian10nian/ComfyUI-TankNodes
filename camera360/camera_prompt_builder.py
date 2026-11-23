"""V6 Relative-Origin 3D Multi-Angle Prompt Compiler (Open & Closed Model Dialects).
Aligned with Lumet V6 specification: relative 0-degree camera origin, mandatory outer crop,
vanishing-point shift, world-locked subject gaze/anti-turntable, OTS pair stacking, and anti-mirroring.
"""
from __future__ import annotations
from typing import Literal

ShotScale = Literal["wide", "medium_wide", "medium", "medium_close_up", "close_up"]
PromptDialect = Literal["structured_v6", "concise_edit", "qma_lora", "sks_lora"]

ANGLE_STEP = 15
VERTICAL_MIN = -30
VERTICAL_MAX = 90

SHOT_TO_ZOOM: dict[str, float] = {
    "wide": 1.0,
    "medium_wide": 3.0,
    "medium": 5.0,
    "medium_close_up": 7.0,
    "close_up": 9.0,
}

SHOT_TO_LABEL: dict[str, str] = {
    "wide": "wide shot",
    "medium_wide": "medium-wide shot",
    "medium": "medium shot",
    "medium_close_up": "medium close-up",
    "close_up": "close-up",
}

SHOT_TO_ZH: dict[str, str] = {
    "wide": "远景",
    "medium_wide": "中远景",
    "medium": "中景",
    "medium_close_up": "中近景",
    "close_up": "特写",
}

DEFAULT_NEGATIVE_PROMPT = (
    "unchanged camera angle, original composition lock, 2D horizontal mirror flip, "
    "flat crop or zoom only, subject rotating with camera, person turning head to stare at camera, "
    "visible face in 180-degree back view, front grille in rear car view, "
    "cross-side wall transplanting, foreground doorframe border lock, distorted anatomy, "
    "extra furniture, watermark, text overlay, UI badge, border"
)


def snap_horizontal(angle: float) -> int:
    snapped = round(float(angle) / ANGLE_STEP) * ANGLE_STEP
    return ((snapped % 360) + 360) % 360


def snap_vertical(angle: float) -> int:
    snapped = round(float(angle) / ANGLE_STEP) * ANGLE_STEP
    return max(VERTICAL_MIN, min(VERTICAL_MAX, snapped))


def build_lora_prompt(h: int, v: int, shot_scale: str, token: str = "<sks>") -> str:
    if token == "<qma>":
        h_dir = (
            "front view" if (h >= 337.5 or h < 22.5) else
            "front-right quarter view" if h < 67.5 else
            "right side view" if h < 112.5 else
            "back-right quarter view" if h < 157.5 else
            "back view" if h < 202.5 else
            "back-left quarter view" if h < 247.5 else
            "left side view" if h < 292.5 else "front-left quarter view"
        )
        v_dir = (
            "low-angle shot" if v < -15 else
            "eye-level shot" if v < 15 else
            "elevated shot" if v < 45 else "high-angle shot"
        )
        s_lbl = (
            "wide shot"
            if shot_scale in ("wide", "medium_wide")
            else ("close-up" if shot_scale in ("medium_close_up", "close_up") else "medium shot")
        )
        return f"<qma> {h_dir} {v_dir} {s_lbl}"

    h_dir = (
        "front view" if (h >= 337.5 or h < 22.5) else
        "front-right view" if h < 67.5 else
        "right side view" if h < 112.5 else
        "back-right view" if h < 157.5 else
        "back view" if h < 202.5 else
        "back-left view" if h < 247.5 else
        "left side view" if h < 292.5 else "front-left view"
    )
    v_dir = (
        "low angle" if v < -15 else "eye level" if v < 15 else
        "high angle" if v < 45 else "bird's eye view" if v < 75 else "top-down view"
    )
    return f"{token} {h_dir}, {v_dir}, {SHOT_TO_LABEL.get(shot_scale, 'medium shot')}\n(horizontal: {h}, vertical: {v}, zoom: {SHOT_TO_ZOOM.get(shot_scale, 5.0):.1f})"


def _horizontal_block_v6(h: int) -> str:
    step_inside = "" if h == 0 else (
        "\n- LEAVE FOREGROUND FRAMES BEHIND: Physically walk INSIDE the scene past any foreground "
        "doorframe, entrance posts, stone stele, or framing branches that bordered the 0° reference photo "
        "— NEVER keep original foreground framing borders around the new shot!"
    )
    if h == 0:
        return (
            "1. HORIZONTAL CAMERA ORBIT — 0° (Keep Exact Original Horizontal Heading):\n"
            "- Do NOT orbit left or right. Maintain the exact horizontal viewing direction and left-to-right spatial alignment of the reference photo."
        )
    if h == 180:
        return (
            "1. HORIZONTAL CAMERA ORBIT — 180° TRUE REVERSE-ANGLE COUNTER-SHOT:\n"
            f"- Move the camera all the way past the central subject into the far background of the reference photo, turn a full 180° around, and shoot straight BACK toward where the original 0° photographer stood!{step_inside}\n"
            "- STRICT ANTI-MIRRORING & ANTI-TURNTABLE RULE (CRITICAL):\n"
            "  * NEVER horizontally flip or 2D-mirror the input image, and NEVER rotate people, chairs, desks, vehicles, or products to face the new camera!\n"
            "  * PEOPLE & CREATURES: Anyone who faced the 0° camera MUST now be seen strictly from BEHIND (showing ONLY the back of their head/hair, back of shoulders, and the back of their chair — their face MUST be 100% hidden facing away from the lens!). If two people sat on the left and right, their left/right screen positions swap.\n"
            "  * VEHICLES & PRODUCTS: Any vehicle or product showing its front/grille/lens in the 0° photo MUST now show its OPPOSITE REAR END (taillights/trunk/backplate — NEVER show the front grille or front lens again!).\n"
            "  * ENVIRONMENT & LIGHTING: The far wall/window/mountain/horizon from the 0° photo is 100% BEHIND THE CAMERA LENS (out of frame), replaced by the opposite wall/gate/space behind the original 0° camera; left and right sides swap; backlit scenes become front-lit."
        )
    is_right = h < 180
    deg = h if is_right else 360 - h
    cam, look = ("right", "left") if is_right else ("left", "right")
    if deg <= 20:
        return (
            f"1. HORIZONTAL CAMERA ORBIT — {deg}° Subtle Orbit into the {cam.upper()} side, Panning {look.upper()}:\n"
            f"- Step the camera {deg}° into the {cam} side of the space and pan {look} toward the center.{step_inside}\n"
            f"- RIGID SUBJECT YAW ({deg}°): Freeze all people, vehicles, and objects in world space so their viewing angle rotates {deg}° (a person who looked into the 0° lens now gazes {deg}° toward the {look} of the new lens).\n"
            f"- CROP & PARALLAX: The outermost 25% of the {cam} margin crops out behind the lens, shifting the former center background toward the {cam} half."
        )
    if deg <= 35:
        return (
            f"1. HORIZONTAL CAMERA ORBIT — {deg}° Oblique Orbit from the {cam.upper()} aiming {look.upper()}:\n"
            f"- Walk the camera {deg}° into the {cam} side of the reference scene and angle the lens across toward the {look} half.{step_inside}\n"
            f"- RIGID SUBJECT YAW ({deg}° — NO POSE LOCK): Freeze all people, vehicles, furniture, and products in world space so their visible angle rotates a full {deg}°! A vehicle/product angled ~25°–35° toward the {cam} now aligns DEAD-ON with its symmetrical front face; a person who faced 0° now faces {deg}° off-camera toward the {look} (do NOT turn their head to stare at the lens).\n"
            f"- MANDATORY CROP & SHIFT: Crop out the outer {cam} third of the reference background, shifting the former center background into the {cam} third of the frame."
        )
    if deg <= 55:
        return (
            f"1. HORIZONTAL CAMERA ORBIT — {deg}° Three-Quarter Diagonal View from the {cam.upper()} looking {look.upper()}:\n"
            f"- Walk the camera deep into the {cam} side of the scene ({deg}° orbit around the center) and shoot diagonally across toward the {look} side.{step_inside}\n"
            f"- WORLD-LOCKED SUBJECTS & NO GAZE-TRACKING: People, chairs, vehicles, and products DO NOT rotate with the camera! Anyone who faced the 0° camera is now seen in a true 45° three-quarter side view looking off-camera toward the {look.upper()} side of the frame — NEVER let them turn their neck/eyes to stare at the new camera! Vehicles/products rotate {deg}° in world space.\n"
            f"- MANDATORY CROP & SHIFT: The entire {cam} half of the reference background is OUT OF FRAME. Push the former center background into the outer {cam} quarter at a sharp {deg}° angle, swinging the reference {look} side into the DEAD CENTER of the frame."
        )
    if deg <= 70:
        return (
            f"1. HORIZONTAL CAMERA ORBIT — {deg}° Steep Side Orbit from the {cam.upper()} across toward the {look.upper()}:\n"
            f"- Orbit the camera {deg}° onto the {cam} flank of the scene and aim steeply across toward the {look} side.{step_inside}\n"
            f"- WORLD-LOCKED SUBJECTS ({deg}° YAW CHANGE): Freeze all people, vehicles, and products in 3D world space (a slightly angled vehicle/object turns into a pure horizontal side profile across the frame; a frontal person is seen in steep 60° profile looking toward the {look}, never looking at the lens).\n"
            f"- MANDATORY CROP: Crop out the entire {cam} side and most of the former center background (at most a narrow edge on the extreme {cam} margin), bringing the reference {look} side head-on into the center."
        )
    if deg <= 105:
        return (
            f"1. HORIZONTAL CAMERA ORBIT — {deg}° TRUE PERPENDICULAR CROSS-SHOT (Standing on the {cam.upper()} side, Shooting 90° Straight Across at the {look.upper()} side):\n"
            f"- CAMERA PLACEMENT: Stand inside the {cam} side of the space with your back directly against the reference {cam} wall/boundary, aiming the lens 90° perpendicular to the original photo's sightline straight across toward the {look} side!{step_inside}\n"
            "- WORLD-LOCKED PEOPLE, OVER-THE-SHOULDER PAIRS & RIGID OBJECTS (CRITICAL):\n"
            f"  * SINGLE PERSON / CHAIR / PRODUCT: Anything that faced the 0° camera is now seen in pure 90° SIDE PROFILE facing horizontally toward the {look.upper()} edge of the frame (never turning to look at the lens).\n"
            f"  * LEFT-RIGHT PAIR / TWO PEOPLE (OVER-THE-SHOULDER DEPTH STACKING): People or objects arranged LEFT-TO-RIGHT across the 0° photo now stack FRONT-TO-BACK in depth: the person/object on the reference {cam.upper()} side sits in the NEAR FOREGROUND seen from BEHIND (over-the-shoulder back of head/chair), looking across at the FRONT FACE of the person/object on the reference {look.upper()} side in the midground!\n"
            "  * FULL 90° ROTATION: Freestanding furniture/vehicles/structures running front-to-back in the 0° photo rotate a full 90° to run HORIZONTALLY left-to-right across the frame; long desks running left-to-right now run FRONT-TO-BACK in depth!\n"
            f"- ZERO FAR-WALL SNAPBACK & NO CROSS-SIDE TRANSPLANTING (CRITICAL): The original far center wall/window runs PARALLEL to the new lens ray — DO NOT show it facing the camera! Everything belonging to the {cam.upper()} wall is BEHIND THE CAMERA — NEVER copy or paste {cam}-side structures onto the {look} wall! What used to be on the {look.upper()} wall/side now faces the camera STRAIGHT-ON dead center!"
        )
    return (
        f"1. HORIZONTAL CAMERA ORBIT — {deg}° REAR-QUARTER COUNTER-SHOT (In the Far Rear past the {cam.upper()} side, Looking Back Outward):\n"
        f"- Walk past the {cam} side all the way into the far rear behind the subject ({deg}° orbit) and turn around to shoot diagonally BACK toward the original 0° camera area and the {look} side!{step_inside}\n"
        "- WORLD-LOCKED REAR 3/4 VIEW: People, chairs, vehicles, and products stay frozen in world space showing their REAR three-quarter back view (back of chair & back/side of head, rear taillights, or product backplate — NEVER turn them to face the lens!).\n"
        f"- STRICT OCCLUSION & CROP: Both the original far center background AND the {cam} side of the reference photo are BEHIND the camera's back and OUT OF FRAME; structures from the reference {look} side sit across the {cam}-center, while the space toward the original 0° camera opens up on the {look}-center."
    )


def _vertical_block_v6(v: int) -> str:
    if v <= -25:
        return f"2. CAMERA ELEVATION — {v}° Low-Angle Upward Shot: Place the camera low near ground/ankle height tilting UP ~30°. Foreground ground looms close, verticals converge upward, and ceilings/roofs/canopies/sky expand overhead while subjects keep their original head pitch."
    if v < 0:
        return f"2. CAMERA ELEVATION — {v}° Gentle Low-Angle Shot: Lower the camera to waist height tilting slightly UP ~15°, lowering the horizon and showing more upper structure/sky."
    if v == 0:
        return "2. CAMERA ELEVATION — 0° Level Horizontal Sightline: Keep a natural level horizontal camera pitch with upright vertical lines."
    if v <= 20:
        return f"2. CAMERA ELEVATION — +{v}° Slight High-Angle Shot: Raise the camera slightly above standing height tilting DOWN ~15°, revealing more floor/ground and top surfaces."
    if v <= 40:
        return f"2. CAMERA ELEVATION — +{v}° Elevated High-Angle Shot: Mount the camera high up tilting DOWN ~30° at the scene center, clearly spreading out ground/tabletop planes while people keep looking forward horizontally rather than looking up at the lens."
    if v <= 55:
        return f"2. CAMERA ELEVATION — +{v}° Steep 45° Overhead Angle: Position the camera high overhead tilting DOWN 45°; ground/tabletop layout and top surfaces of heads/shoulders/objects dominate with steep downward foreshortening (subjects do NOT crane their necks up to stare at the high camera)."
    if v <= 75:
        return f"2. CAMERA ELEVATION — +{v}° True Bird's-Eye Aerial View: Fly the camera high in the air ABOVE the tallest roofs/peaks/ceilings/subjects, looking steeply DOWNWARD at {v}° onto roofs, courtyards, vehicle tops, and ground layout below (never looking upward from below)!"
    return "2. CAMERA ELEVATION — +90° PURE VERTICAL TOP-DOWN PLAN / FLAT-LAY VIEW (Zero Horizon): Suspend the camera directly above the exact center pointing 90° STRAIGHT DOWN perpendicular to the floor/tabletop. Show ONLY horizontal ground/floor/tabletop and pure TOP SURFACES (roof/hood top, camera top-plate & dials without front glass, crown of head) — NO horizon or frontal walls."


SHOT_BLOCKS_V6: dict[str, str] = {
    "wide": "3. SHOT SCALE & LENS — Wide Establishing Shot (24mm wide-angle): Pull the camera significantly backward along the new viewing axis to capture a broader, more expansive view of the space.",
    "medium_wide": "3. SHOT SCALE & LENS — Medium-Wide Shot (28mm): Step moderately backward to frame the full main area/subject with generous surrounding context.",
    "medium": "3. SHOT SCALE & LENS — Medium Shot (35mm natural perspective): Maintain balanced standard viewing distance.",
    "medium_close_up": "3. SHOT SCALE & LENS — Medium Close-Up (50mm): Physically move the camera 2x closer forward toward the central focal subject (chest-up for a person, tight framing for an object/area), cropping out peripheral outer areas.",
    "close_up": "3. SHOT SCALE & LENS — True Close-Up Shot (85mm): Physically dolly the camera 3.5x–4x closer right up to the primary focal subject (tight head-and-shoulders portrait for a person, tight macro/detail framing for an object or scene center) so it fills 70%–85% of the frame, leaving legs/full furniture/outer walls out of frame with natural shallow depth of field.",
}


def build_concise_edit_prompt(h: int, v: int, shot_scale: str) -> str:
    if h == 0:
        orbit = "keep the exact original horizontal viewing direction (0° orbit)"
    elif h == 180:
        orbit = (
            "move the camera 180° behind the subject into the far background and shoot straight back toward the original camera spot; "
            "keep all people, chairs, vehicles, and objects frozen in world space (show ONLY the back of a frontal person's head/chair or the rear taillights/backplate of a frontal vehicle/product, NEVER mirror-flip the image or turn subjects to face the lens), "
            "crop out the original far background behind the camera, and swap left and right sides"
        )
    else:
        is_right = h < 180
        deg = h if is_right else 360 - h
        cam, look = ("right", "left") if is_right else ("left", "right")
        if 75 <= deg <= 105:
            orbit = (
                f"move the camera 90° onto the {cam} side of the space and shoot perpendicular straight across toward the {look} side; "
                f"crop out the reference {cam} wall behind the lens and keep the original far center wall parallel/out-of-frame; "
                f"freeze all subjects in world space so central objects rotate 90° horizontally, a frontal person shows a pure 90° side profile facing the {look} edge without looking at the lens, left-right pairs stack front-to-back over-the-shoulder, and the reference {look} side faces the camera dead center"
            )
        else:
            orbit = (
                f"orbit the camera {deg}° into the {cam} side of the scene aiming across toward the {look} side; "
                f"crop out the outer {cam} background behind the lens, shift the former center background toward the {cam} side, bring the reference {look} side toward the center, and freeze all people/vehicles/objects in world space rotated by {deg}° without turning to look at the camera"
            )
    pitch = (
        "at level eye-level height (0° pitch)" if v == 0 else
        "from directly overhead pointing 90° straight down (pure top-down flat-lay/plan view, zero horizon)" if v == 90 else
        f"from an elevated camera tilting {v}° downward" if v > 0 else
        f"from a low camera near ground level tilting {abs(v)}° upward"
    )
    shot_desc = {
        "wide": "wide-angle 24mm establishing distance pulled far back",
        "medium_wide": "medium-wide 28mm distance stepped slightly back",
        "medium": "balanced 35mm medium distance",
        "medium_close_up": "50mm medium close-up pushed 2x closer to the central subject",
        "close_up": "tight 85mm close-up pushed right up to the primary subject with shallow depth of field",
    }.get(shot_scale, "balanced 35mm medium distance")
    return (
        f"Re-photograph this exact frozen 3D scene and world-locked subjects by repositioning the camera: {orbit}, {pitch}, {shot_desc}. "
        "Preserve all architecture, object identities, signage, lighting direction, and art style with true 3D parallax and occlusion, and remove any watermarks."
    )


def compile_multi_angle_prompts(
    horizontal_angle: float,
    vertical_angle: float,
    shot_scale: str = "medium",
    dialect: str = "structured_v6",
) -> dict[str, object]:
    h, v = snap_horizontal(horizontal_angle), snap_vertical(vertical_angle)
    sks = build_lora_prompt(h, v, shot_scale, token="<sks>")
    qma = build_lora_prompt(h, v, shot_scale, token="<qma>")
    if dialect == "sks_lora":
        pos = sks
    elif dialect == "qma_lora":
        pos = qma
    elif dialect == "concise_edit":
        pos = build_concise_edit_prompt(h, v, shot_scale)
    else:
        pos = "\n\n".join([
            "3D SCENE & SUBJECT CAMERA REPOSITIONING (Relative to Input Reference Photo = 0° Origin):\n"
            "Treat the input image as a real 3D world frozen at the exact same instant. Every person (exact body pose, head direction, world-space eye-gaze vector, clothing), vehicle, product, piece of furniture, architectural structure, terrain, inscription, art style, and 3D world-space light direction is FROZEN LIKE A STATUE in world coordinates — ONLY the camera moves around them to re-photograph from the new pose below:",
            _horizontal_block_v6(h),
            _vertical_block_v6(v),
            SHOT_BLOCKS_V6.get(shot_scale, SHOT_BLOCKS_V6["medium"]),
            "4. STRICT 3D CONSISTENCY & CLEAN RENDER:\n"
            "- Compute true 3D geometric parallax, occlusion changes, and newly revealed side/rear faces. NEVER rotate people/vehicles/objects to follow the camera and NEVER return the unchanged input composition or a 2D horizontal mirror flip.\n"
            "- Preserve exact facial/garment/vehicle/product/architectural details without inventing unrelated items; naturally extend newly visible background areas in the exact same style.\n"
            "- Completely omit any corner watermarks, UI badges, or artificial borders.",
        ])
    return {
        "positive_prompt": pos,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "lora_prompt": qma if dialect == "qma_lora" else sks,
        "horizontal_angle": h,
        "vertical_angle": v,
        "shot_scale": shot_scale,
    }
