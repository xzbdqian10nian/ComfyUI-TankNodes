# TankNodes

English · [简体中文](README_zh-CN.md)

A collection of ComfyUI nodes by Tank. Use local Qwen3.8 or Ternary Bonsai for text, images and video, or call other models through an OpenAI-compatible API.

Search for **Tank** in the node menu.

## Nodes

| Node | Use |
| --- | --- |
| Qwen / Bonsai GGUF Loader · Tank | Select a main model and matching vision projector, then connect to local chat. |
| Local Multimodal Chat · Tank | Enter a prompt, optionally add images or video, and get a text response. |
| API Chat (Environment Key) · Tank | Read an API key from a server environment variable. |
| API Chat (Direct Key) · Tank | Enter an API key in the node. Clear it before sharing the workflow. |
| Unload Model · Tank | Release GPU memory. Connect the chat response to **after** to wait for generation to finish. |
| Text Split · Tank | Split text into a list by a delimiter (`\n` for new lines, Chinese punctuation works as-is). Connected nodes run once per item; advanced options trim items or use a regular expression such as `[,，]`. |

Each run is a separate request, with no automatic conversation history. API nodes work on their own and do not need the local model loader.

## Install

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-TankNodes.git
```

Restart ComfyUI and refresh the browser. Use Python 3.10 or newer. Install the extra dependencies in [requirements.txt](requirements.txt) with ComfyUI's own Python environment.

Local Qwen inference needs a `llama-cpp-python` build that supports Qwen3.8 and matches your CUDA runtime. Bonsai PQ2_0 instead needs PrismML's `llama-server`. API nodes do not need either local runtime. The plugin does not replace Torch, CUDA or GPU drivers.

## Quick start

1. Put the main GGUF and matching vision projector in `ComfyUI/models/LLM/Qwen3.8/`.
2. Open [01_text_chat.json](example_workflows/01_text_chat.json) and select both files in the loader.
3. Edit the user prompt and run. Model weights load when chat starts.

The tested pair is `Qwen3.8-27B-UD-Q4_K_M.gguf` and `mmproj-BF16.gguf` from [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Other catalogued files are listed in [models.json](models.json). Use a main model and projector from the same release.

Subfolders are supported. Set `QWEN38_MODEL_DIR` to use another location. The plugin does not download weights; if the menu says no model was found, check the path and whether the download finished.

## Ternary Bonsai for reverse prompting

The loader also recognizes the official `Ternary-Bonsai-2-27B-PQ2_0.gguf` and `Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf`. Pair either with `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` in the same model directory. The [official GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and [abliterated v2](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP-GGUF) are separate choices. Install a PQ2_0-capable [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp) binary at `/root/bonsai-test/runtime/llama-server`, or set `BONSAI_LLAMA_SERVER` before starting ComfyUI. The plugin does not download models or the runtime.

For a local Bonsai chat, the plugin starts a loopback server on demand without stopping ComfyUI. It uses at least 16K context, Flash Attention, Q8 KV cache, one concurrent slot and no MTP. In measured short Chinese creative tasks, MTP was slower; a filename containing MTP does not enable it. Bonsai `auto` disables thinking for text-only prompt polishing and uses `low` effort when images or video frames are attached. Select `medium` explicitly for harder requests and allow enough output tokens for the reasoning. A one-token empty answer gets one retry with another seed; reasoning that exhausts the output budget raises an explicit error. Keep the model loaded for repeated calls, or use **unload after** to free GPU memory for image generation. On one idle RTX 5090, either 16K variant used about 8.9 GiB and worked with ComfyUI still running. Single images, short text and a synthetic two-frame input were verified; full VIDEO objects and longer workflows need checks with actual material.

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

Use ComfyUI's interrupt button while waiting for an API response or receiving output. It stops the local request and closes the connection. Whether remote generation and billing stop depends on the provider.

## Useful settings

- Local Qwen `auto` keeps thinking off. Bonsai `auto` uses no thinking for text and `low` for images/frames. `off` explicitly disables it. External API `auto` keeps the provider's default.
- For local Qwen3.8, `high` runs as `medium` and `max` as `xhigh`. Stats show the effective setting. Changing reasoning effort keeps the model loaded; changing context length reloads it.
- Keep the model loaded for repeated chats. Enable **unload after** if you need the GPU memory for another model.
- Nodes 2.0 can collapse advanced inputs. The classic canvas shows them together.

## Development

See [tests/README.md](tests/README.md) for test commands. Licensed under [MIT](LICENSE).

Thanks to the Qwen team, llama.cpp and the community GGUF publishers.
