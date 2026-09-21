# TankNodes

[English](README.md) | [简体中文](README_zh-CN.md)

A personal ComfyUI toolkit by **Tank / xzbd**. It provides local Qwen3.8 multimodal inference and direct OpenAI-compatible API chat. All node titles end in **· Tank**, under **TankNodes** in the node menu.

Local workflows use **model loader → local chat**. The two API Chat nodes make requests directly and return text; they do not connect to the local chat node. Media conversion and response processing are shared internally.

## Quick start

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-Qwen3.8-VL.git
```

The repository URL, installation directory, Registry package ID `qwen38-vl`, and internal node IDs remain unchanged for upgrades. TankNodes is the display name; install one copy of the plugin.

1. Put a GGUF main model and its matching vision projector under `ComfyUI/models/LLM/Qwen3.8/`. Subfolders are supported. `QWEN38_MODEL_DIR` can override this directory.
2. Restart ComfyUI and refresh the browser.
3. Add **Qwen3.8 Model Loader · Tank** and select the two files. Empty directories show an explicit missing-model option.
4. Connect its **model configuration** output to **Local Multimodal Chat · Tank**.
5. Enter a persona and prompt, optionally connect images or video, and run. Weights load when chat executes.

For API use, add either API Chat node, enter a base URL, exact **model ID**, key source and prompt. Local GGUF weights are not required.

### Updating

```bash
git -C ComfyUI-Qwen3.8-VL pull --ff-only
```

Run this from the parent of the existing plugin folder, or run `git pull --ff-only` inside that folder. Restart ComfyUI and refresh the browser after updating.

## Nodes

| Node | Purpose |
| --- | --- |
| Qwen3.8 Model Loader · Tank | Select GGUF files and prepare a local model configuration. |
| Local Multimodal Chat · Tank | One text, image, image-batch or video request using the local model. |
| API Chat (Environment Key) · Tank | Direct API request using a server environment variable. |
| API Chat (Direct Key) · Tank | Direct API request using a key entered in the node. |
| Unload Model · Tank | Release the model; connect a chat response to **after** to run after that chat. |

Each run is a separate request, with no automatic conversation history. Common inputs stay visible. Sampling, context and transport controls are marked as native **advanced** inputs; ComfyUI Nodes 2.0 can expand/collapse them, while the classic canvas displays them together. Widget order and saved values are unchanged. Local `video_transport` remains as an advanced compatibility field; local models always sample frames.

For a single chat, **unload after** is the simplest way to release VRAM. The separate unload node's optional **after** connection establishes an explicit dependency when needed.

## Models and reasoning

Use a main model and projector from the same release. Selections retain real filenames, not marketing labels. The loader's **model info** includes known source, variant, recommended projector and filename-match status. Filename matching is informational, not a checksum or architecture check; arbitrary renamed files cannot be identified reliably.

| Source | Typical files | Difference |
| --- | --- | --- |
| [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | [UD Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf?download=true) + [mmproj-BF16](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf?download=true) | Recommended starting point; compact dynamic quantization with the original model alignment. |
| [Huihui AI Abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | [Q4_K](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf?download=true) or Q4_K_L + `mmproj-model-bf16.gguf` | A community low-refusal variant; larger Q4_K_L retains more high-precision tensors. |
| [Orcarouter Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | [Q4_K_M](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-Q4_K_M.gguf?download=true) + [mmproj f16](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/mmproj-Qwen3.8-27B-Uncensored-f16.gguf?download=true) | Another community low-refusal variant. Hugging Face access conditions may apply. |

These are the existing catalogue entries. Review each source's current files and model card before downloading. Model weights are not included or downloaded automatically.

The reasoning selector retains `auto`, `off`, `low`, `medium`, `high`, `xhigh`, `max`:

- Local `auto` preserves the old non-thinking default; `off` explicitly disables thinking.
- The local Qwen3.8 adapter maps `high` to `medium` and `max` to `xhigh`. Statistics report selected and effective values. Switching effort does not reload weights; changing context length does.
- API `auto` leaves provider reasoning defaults unchanged. Other settings follow the current provider adapter; support varies by endpoint. An older llama.cpp runtime that only supports on/off is reported as such.

## Images and video

The **image** input sends every image in the batch. The **video frames** input samples an IMAGE batch up to **max video frames**. The **video** input samples across the whole video, including containers without a frame count; those need a counting pass before sampling. Images are not silently resized.

API `frames` sends sampled images; `video_url` sends native video. `auto` falls back to frames when local encoding fails, not when a provider rejects the request. The provider must support the format. Native uploads preserve the actual container MIME type. Statistics distinguish image count, sampled frame count and native video count; when token usage is unavailable, throughput is `n/a`.

## API settings and outputs

- **base URL**: the API root, for example `https://api.openai.com/v1`, without `/chat/completions`.
- **model ID**: the provider's exact identifier, not a translated display name.
- **max output tokens = 0**: omit the limit and use the provider default.
- **temperature = 0**: omit the API parameter. Local temperature 0 retains its existing deterministic-sampling meaning.
- **seed**: sent when configured; provider support varies.

API outputs keep their existing first three positions: **response**, **usage**, **stats**. **reasoning** and **raw response** are appended at positions 4 and 5. Reasoning is empty when the provider does not return it. The usage JSON preserves the two legacy string-valued token fields. Local outputs remain **response / reasoning / raw response / stats**.

### Environment key

Set the variable before starting ComfyUI and enter only its name in the node:

```bash
export OPENAI_API_KEY='your-api-key'
```

The server-side allow-list defaults to `api.openai.com` and loopback hosts. Other providers need an administrator-configured exact host or `host:port`:

```bash
export COMFYUI_API_ALLOWED_HOSTS='api.openai.com,api.example.com'
```

HTTPS is required except for loopback services. Direct Key uses only the explicitly entered key; do not save real keys in shared workflows. No author key or private endpoint is included.

## Example workflows

Eight examples in [example_workflows](example_workflows/) cover text, a single image, multiple images, decoded video frames, native VIDEO, both API key modes and ordered model release. Local examples retain their original functional parameters. Instructions are bilingual; standard plugin-node titles follow the interface language.

## Existing workflows

The five internal node IDs, existing field names, widget order, original output indices, model paths and Registry package ID are retained. Old `backend_default / thinking / instruct` choices remain supported. The browser extension recognizes the previous 11-widget API layout and inserts the missing reasoning default without shifting video settings. User custom titles and links are preserved.

The extension also repairs evidenced v0.5.0 prompt-first layouts, with a marker preventing repeated swaps. It does not guess the meaning of text in ambiguous older layouts.

The rename is a display change, not a Registry or GitHub repository migration. Restart the server and refresh the browser together so the Python definitions and compatibility extension are from the same version.

## Dependencies and development

Python 3.10+; use the existing ComfyUI Torch environment. Local inference needs a compatible CUDA `llama-cpp-python` build. API mode uses the installed OpenAI SDK or the standard-library HTTP fallback. This plugin does not replace Torch, CUDA or GPU drivers. `requirements.txt` contains only the small additional runtime dependency.

`__init__.py` registers nodes. `local_nodes.py` and `api_nodes.py` hold node interfaces. `chat.py` shares content/response processing; `backends.py` owns runtime adapters; `model_catalog.py`, `media.py`, `reasoning.py` and `progress.py` each cover one concern. `web/` contains only old-workflow compatibility, with no custom widget serialization.

Development tests require pytest, Torch, NumPy, Pillow, PyAV and OpenAI in an isolated environment. See [tests/README.md](tests/README.md).

## Acknowledgements

Thanks to the Qwen team and the community GGUF publishers listed above. TankNodes is an independent project; it does not include model weights or imply endorsement by model providers.
