"""TankNodes: local multimodal models, prompt workflows, manga/camera utilities, and OpenAI-compatible API tools."""

from .api_nodes import VisionAPIDirect, VisionAPIEnv
from .local_nodes import Qwen38VLLoader, VisionChat, VisionUnload
from .text_nodes import TankTextSplit
from .manga import (
    TankMangaCharacterPrompt,
    TankMangaTranslatePrompt,
    TankMossColorizeLayout,
    TankMossSwapTextLayout,
)
from .h3_optimizer import TankH3PromptOptimizer
from .prompt_nodes import (
    TankImageBatch,
    TankOptionalImageDownscale,
    TankPromptOptimizer,
    TankPromptTemplate,
    _register_template_routes,
)

_register_template_routes()

# These are persisted in existing workflows. Branding must not change them.
NODE_CLASS_MAPPINGS = {
    "Qwen38VLLoader": Qwen38VLLoader,
    "VisionAPIEnv": VisionAPIEnv,
    "VisionAPIDirect": VisionAPIDirect,
    "VisionChat": VisionChat,
    "VisionUnload": VisionUnload,
    "TankTextSplit": TankTextSplit,
    "TankH3PromptOptimizer": TankH3PromptOptimizer,
    "TankPromptTemplate": TankPromptTemplate,
    "TankPromptOptimizer": TankPromptOptimizer,
    "TankImageBatch": TankImageBatch,
    "TankOptionalImageDownscale": TankOptionalImageDownscale,
    "TankMangaTranslatePrompt": TankMangaTranslatePrompt,
    "TankMossSwapTextLayout": TankMossSwapTextLayout,
    "TankMangaCharacterPrompt": TankMangaCharacterPrompt,
    "TankMossColorizeLayout": TankMossColorizeLayout,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen38VLLoader": "Qwen / Bonsai GGUF Loader · Tank",
    "VisionAPIEnv": "API Chat (Environment Key) · Tank",
    "VisionAPIDirect": "API Chat (Direct Key) · Tank",
    "VisionChat": "Local Multimodal Chat · Tank",
    "VisionUnload": "Unload Model · Tank",
    "TankTextSplit": "Text Split · Tank",
    "TankH3PromptOptimizer": "H3 Prompt Optimizer · Tank",
    "TankPromptTemplate": "Prompt Template · Tank",
    "TankPromptOptimizer": "Prompt Optimizer · Tank",
    "TankImageBatch": "Image Batch (optional inputs) · Tank",
    "TankOptionalImageDownscale": "Reference Image Resize (optional input) · Tank",
    "TankMangaTranslatePrompt": "Manga Translate Prompt (VL) · Tank",
    "TankMossSwapTextLayout": "MOSS Swap-Text Layout · Tank",
    "TankMangaCharacterPrompt": "Manga Character Finder Prompt (VL) · Tank",
    "TankMossColorizeLayout": "MOSS Per-Character Colorize Layout · Tank",
}

from .h3_image_prompt import (
    NODE_CLASS_MAPPINGS as _H3_IMAGE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _H3_IMAGE_NAMES,
)

NODE_CLASS_MAPPINGS.update(_H3_IMAGE_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(_H3_IMAGE_NAMES)

from .camera360 import (
    NODE_CLASS_MAPPINGS as _CAMERA_360_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _CAMERA_360_NAMES,
)

NODE_CLASS_MAPPINGS.update(_CAMERA_360_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(_CAMERA_360_NAMES)

from .image_size_nodes import (
    NODE_CLASS_MAPPINGS as _IMAGE_SIZE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_SIZE_NAMES,
)

NODE_CLASS_MAPPINGS.update(_IMAGE_SIZE_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(_IMAGE_SIZE_NAMES)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

from .h3_long_video import (
    NODE_CLASS_MAPPINGS as _H3_LONG_VIDEO_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as _H3_LONG_VIDEO_NAMES,
)

NODE_CLASS_MAPPINGS.update(_H3_LONG_VIDEO_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(_H3_LONG_VIDEO_NAMES)
