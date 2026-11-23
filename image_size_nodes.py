import math

FOLLOW_IMAGE_1 = "跟随图像1"
RATIO_PRESETS = ["1:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9", "9:21", "21:9", "4:5", "5:4", "2:1", "1:2", "3:1", "1:3"]
ASPECT_RATIOS = [FOLLOW_IMAGE_1, *RATIO_PRESETS]
ALIGNMENTS = [16, 32, 64]


def _resolve_ratio(aspect_ratio, image=None):
    if aspect_ratio == FOLLOW_IMAGE_1:
        if image is None:
            raise ValueError("已选择「跟随图像1」，请连接输入图像。")
        if not hasattr(image, "ndim") or image.ndim != 4 or int(image.shape[1]) <= 0 or int(image.shape[2]) <= 0:
            raise ValueError("输入图像维度无效，需要有效图像张量 [B, H, W, C]。")
        return int(image.shape[2]), int(image.shape[1])
    if aspect_ratio not in RATIO_PRESETS:
        raise ValueError("请选择支持的宽高比。")
    return map(int, aspect_ratio.split(":"))


def image_size(aspect_ratio, megapixels, multiple, image=None):
    if not math.isfinite(float(megapixels)) or not 0.1 <= float(megapixels) <= 32:
        raise ValueError("像素面积必须在 0.1 到 32 MP 之间。")
    if multiple not in ALIGNMENTS:
        raise ValueError("尺寸倍数必须为 16、32 或 64。")
    rw, rh = _resolve_ratio(aspect_ratio, image=image)
    # Match ComfyUI's 1024²-based MP convention and the frontend's rounding.
    area = float(megapixels) * 1024**2
    width = max(multiple, math.floor(math.sqrt(area * rw / rh) / multiple + 0.5) * multiple)
    height = max(multiple, math.floor(math.sqrt(area * rh / rw) / multiple + 0.5) * multiple)
    return width, height


class TankImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS, {"default": "3:4", "tooltip": "Target width-to-height ratio, or 跟随图像1 to match optional image 1. Alignment may slightly change the final ratio."}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 32.0, "step": 0.1,
                                          "tooltip": "Target pixel area: 1 MP = 1024 squared pixels. The preview shows the aligned size."}),
                "multiple": (ALIGNMENTS, {"default": 32, "tooltip": "Align width and height to this multiple; this does not multiply the dimensions."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                        "tooltip": "Number of empty latent images. Larger areas and batches use more VRAM."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional reference image 1 used when aspect_ratio is set to 跟随图像1."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("latent", "width", "height", "pixels", "actual_megapixels")
    OUTPUT_TOOLTIPS = (
        "Generic zero latent for standard samplers. Do not replace editing, masked, or H3 Image conditioning latents.",
        "Aligned width in pixels.", "Aligned height in pixels.", "Actual pixel count per image: width times height.",
        "Actual aligned area in MP, using 1024 squared pixels per MP.",
    )
    FUNCTION = "generate"
    CATEGORY = "TankNodes/Image"
    DESCRIPTION = "Create an empty latent from aspect ratio (or optional image 1) and pixel area, with width, height and pixel-count outputs. For H3 Image, use only width and height and keep its conditioning node."

    def generate(self, aspect_ratio="3:4", megapixels=2.0, multiple=32, batch_size=1, image=None):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 64:
            raise ValueError("批量必须是 1 到 64 的整数。")
        width, height = image_size(aspect_ratio, megapixels, multiple, image=image)
        from nodes import EmptyLatentImage

        latent = EmptyLatentImage().generate(width, height, batch_size)[0]
        pixels = width * height
        return latent, width, height, pixels, pixels / 1024**2


NODE_CLASS_MAPPINGS = {"TankImageSize": TankImageSize}
NODE_DISPLAY_NAME_MAPPINGS = {"TankImageSize": "Image Size (Ratio · Megapixels) · Tank"}
