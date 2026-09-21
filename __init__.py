"""TankNodes: local multimodal models and OpenAI-compatible API tools."""

from .api_nodes import VisionAPIDirect, VisionAPIEnv
from .local_nodes import Qwen38VLLoader, VisionChat, VisionUnload

# These are persisted in existing workflows. Branding must not change them.
NODE_CLASS_MAPPINGS = {
    "Qwen38VLLoader": Qwen38VLLoader,
    "VisionAPIEnv": VisionAPIEnv,
    "VisionAPIDirect": VisionAPIDirect,
    "VisionChat": VisionChat,
    "VisionUnload": VisionUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen38VLLoader": "Qwen3.8 Model Loader · Tank",
    "VisionAPIEnv": "API Chat (Environment Key) · Tank",
    "VisionAPIDirect": "API Chat (Direct Key) · Tank",
    "VisionChat": "Local Multimodal Chat · Tank",
    "VisionUnload": "Unload Model · Tank",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
