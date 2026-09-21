# TankNodes

English · [简体中文](README_zh-CN.md)

A collection of ComfyUI nodes by Tank. Use a local Qwen3.8 model for text, images and video, or call other models through an OpenAI-compatible API.

Search for **Tank** in the node menu.

## Nodes

| Node | Use |
| --- | --- |
| Qwen3.8 Model Loader · Tank | Select a main model and vision projector, then connect to local chat. |
| Local Multimodal Chat · Tank | Enter a prompt, optionally add images or video, and get a text response. |
| API Chat (Environment Key) · Tank | Read an API key from a server environment variable. |
| API Chat (Direct Key) · Tank | Enter an API key in the node. Clear it before sharing the workflow. |
| Unload Model · Tank | Release GPU memory. Connect the chat response to **after** to wait for generation to finish. |

Each run is a separate request, with no automatic conversation history. API nodes work on their own and do not need the local model loader.

## Install

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-TankNodes.git
```

Restart ComfyUI and refresh the browser. Use Python 3.10 or newer. Install the extra dependencies in [requirements.txt](requirements.txt) with ComfyUI's own Python environment.

Local inference also needs a `llama-cpp-python` build that supports Qwen3.8 and matches your CUDA runtime. API nodes do not need it. The plugin does not replace Torch, CUDA or GPU drivers.

## Quick start

1. Put the main GGUF and matching vision projector in `ComfyUI/models/LLM/Qwen3.8/`.
2. Open [01_text_chat.json](example_workflows/01_text_chat.json) and select both files in the loader.
3. Edit the user prompt and run. Model weights load when chat starts.

The tested pair is `Qwen3.8-27B-UD-Q4_K_M.gguf` and `mmproj-BF16.gguf` from [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Other catalogued files are listed in [models.json](models.json). Use a main model and projector from the same release.

Subfolders are supported. Set `QWEN38_MODEL_DIR` to use another location. The plugin does not download weights; if the menu says no model was found, check the path and whether the download finished.

## Images and video

- **image** accepts one image or a batch and sends every image in it.
- **video frames** accepts an IMAGE batch and samples up to **max video frames**.
- **video** accepts a VIDEO object. Local models sample frames across the whole clip.

For APIs, `frames` is the default. Use `video_url` if the provider accepts native video. `auto` tries to encode the video and falls back to frames if encoding fails; it does not retry a rejected API request with frames.

The [eight examples](example_workflows/) cover text, single and multiple images, both video inputs, both API key options, and model release.

## API setup

Enter the base URL, the provider's exact **model ID**, and the key source. For example, use `https://api.openai.com/v1` without `/chat/completions`.

For an environment key, set the variable before starting ComfyUI:

```bash
export OPENAI_API_KEY='your-api-key'
```

Enter only `OPENAI_API_KEY` in the node. This mode allows OpenAI's official endpoint and loopback services by default. An administrator can allow other providers:

```bash
export COMFYUI_API_ALLOWED_HOSTS='api.openai.com,api.example.com'
```

Environment-key requests require HTTPS except for loopback services. Direct-key mode uses the key entered in the node.

For API nodes, **max output tokens = 0** and **temperature = 0** omit those settings and use the provider defaults. Local temperature 0 uses deterministic sampling. API outputs are response, usage, stats, reasoning and raw response. Missing reasoning is empty; unavailable usage or throughput is shown as n/a.

## Useful settings

- Local `auto` keeps thinking off. `off` explicitly disables it. API `auto` keeps the provider's default.
- For local Qwen3.8, `high` runs as `medium` and `max` as `xhigh`. Stats show the effective setting. Changing reasoning effort keeps the model loaded; changing context length reloads it.
- Keep the model loaded for repeated chats. Enable **unload after** if you need the GPU memory for another model.
- Nodes 2.0 can collapse advanced inputs. The classic canvas shows them together.

## Development

See [tests/README.md](tests/README.md) for test commands. Licensed under [MIT](LICENSE).

Thanks to the Qwen team, llama.cpp and the community GGUF publishers.
