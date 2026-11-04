"""TankNodes: local multimodal models and OpenAI-compatible API tools."""

from .api_nodes import VisionAPIDirect, VisionAPIEnv
from .local_nodes import Qwen38VLLoader, VisionChat, VisionUnload
from .text_nodes import TankTextSplit
from .moss_nodes import TankMangaTranslatePrompt, TankMossSwapTextLayout
from .h3_optimizer import TankH3PromptOptimizer
from .prompt_nodes import (TankFlux3PromptEngineering, TankImageBatch, TankMossPromptEngineering, TankOptionalImageDownscale, TankPromptOptimizer,
                           TankPromptTemplate,
                           TankQwen21PromptEngineering)

from .prompt_nodes import _register_template_routes

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
    "TankQwen21PromptEngineering": TankQwen21PromptEngineering,
    "TankMossPromptEngineering": TankMossPromptEngineering,
    "TankFlux3PromptEngineering": TankFlux3PromptEngineering,
    "TankOptionalImageDownscale": TankOptionalImageDownscale,
    "TankMangaTranslatePrompt": TankMangaTranslatePrompt,
    "TankMossSwapTextLayout": TankMossSwapTextLayout,
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
    "TankQwen21PromptEngineering": "Qwen-Image 2.1 Prompt Engineering · Tank",
    "TankMossPromptEngineering": "MOSS-Image Prompt Engineering · Tank",
    "TankFlux3PromptEngineering": "FLUX 3 Klein Prompt Engineering · Tank",
    "TankOptionalImageDownscale": "Reference Image Resize (optional input) · Tank",
    "TankMangaTranslatePrompt": "Manga Translate Prompt (VL) · Tank",
    "TankMossSwapTextLayout": "MOSS Swap-Text Layout · Tank",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
