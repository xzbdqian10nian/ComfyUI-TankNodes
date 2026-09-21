# Development checks

Use an isolated Python environment with `pytest torch numpy pillow av openai`.
From the plugin directory:

```bash
python -m pytest tests -q
node --test tests/workflow_compat.test.mjs
```

The tests compare the five node interfaces against the 0.6.0 contract,
exercise real MP4/Matroska decoding (file and memory inputs), check cleanup
and reasoning changes, and call a local HTTP test server through both the
OpenAI SDK streaming path and the standard-library fallback. Test keys and
responses are synthetic. No model weights or paid API calls are used.

The JavaScript tests check historical widget migration and preservation of
custom titles, links and other configure hooks. Separate real ComfyUI browser
checks are still needed for display, loading, and graph execution. CUDA GGUF
inference and real provider compatibility require their respective runtimes.

`fixtures/v060_contract.json` records public input names, defaults and output
indices from commit `63d2136`; it contains no user prompts or keys.
