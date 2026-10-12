# TankNodes 更新记录 / Changelog

## Unreleased — 安装即用 / Works after install

- 模型首次使用时自动下载：加载器列出 models.json 收录的主模型和视觉投影，本地缺少时从 Hugging Face 断点续传下载到 `models/LLM/Qwen3.8`，支持 `HF_ENDPOINT` 镜像；`TANKNODES_AUTO_DOWNLOAD=0` 可关闭。
- PrismML llama-server 自动安装：按系统与 PyTorch CUDA 版本下载固定的官方 `prism-b10743-adfffbe` 构建，校验 SHA-256 后解压到插件 `runtime/`。查找顺序为 `BONSAI_LLAMA_SERVER` → `runtime/` → 自动下载；删除写死的实例路径。
- Qwen 未安装 llama-cpp-python 时改用同一 llama-server（`auto` 思考保持关闭）；已安装时行为不变，`TANKNODES_LOCAL_RUNTIME` 可强制选择。
- Bonsai 视觉投影不匹配时在下载前报错。README 去除实例专用路径。

Listed models now download on first use (resumable, mirror-aware) and the pinned PrismML llama-server installs itself into the plugin's `runtime/` folder after SHA-256 verification. Qwen falls back to llama-server when llama-cpp-python is missing. Instance-specific paths were removed.

## Unreleased — 文本分割 / Text split

- 新增“文本分割 · Tank”（`TankTextSplit`），可替代 mixlab 的 Text Split By Delimiter：同样支持开始索引、每隔跳过和最大数量，并额外输出实际项数。
- 分隔符只转换 `\n`、`\t`、`\r`、`\\`，中文逗号等非 ASCII 分隔符不再被错误解码。高级参数可去除首尾空白或启用正则；没有可输出的项时明确报错。

Added Text Split · Tank as a drop-in alternative to mixlab's Text Split By Delimiter, with a count output, safe handling of non-ASCII delimiters, optional trimming and regex mode, and an explicit error when nothing is output.

## Unreleased — Ternary Bonsai 本地反推

- 原有 `Qwen38VLLoader` 节点增加官方 Bonsai PQ2_0 与 Abliterated v2 选项，节点 ID、输入顺序和输出位置保持不变。
- Bonsai 使用 PrismML `llama-server` 在本机回环地址按需加载；16K 起步、Flash Attention、Q8 KV、单并发，中文创作任务默认不启用 MTP。卸载节点与生成后卸载均可释放显存。
- `auto` 对纯文字提示词优化关闭思考，对图片/视频帧使用 `low`。极少数随机种子导致的 1-token 空回答会换种子重试一次；思考耗尽输出预算时明确报错。
- 已在 RTX 5090 上通过 ComfyUI 的文字和图片流程验证，包括 20,480 上下文下的 Abliterated v2 加载与视觉投影。

The existing loader now supports official and abliterated-v2 Bonsai PQ2_0 through PrismML's local server while preserving saved node contracts. Short-context Flash Attention and Q8 KV are enabled; MTP stays off for the measured Chinese creative tasks. Auto reasoning uses off for text and low for images/frames. Empty one-token responses get one retry, while exhausted reasoning budgets produce an explicit error. Text and image workflows have been verified through ComfyUI on an RTX 5090, including abliterated-v2 loading and its vision projector at a 20,480-token context.

## 0.7.2 — API 中断 / API interruption

- 两种 API 节点支持 ComfyUI 中断按钮，等待首个响应和接收输出时均可取消。
- 取消时关闭请求连接，不返回半截结果，也不继续执行后续节点。
- 未安装 OpenAI SDK 时复用 ComfyUI 的 aiohttp，替代不能及时取消的同步请求。
- 服务商是否停止生成和计费，取决于对方接口。节点参数、ID 和输出位置不变。

Both API nodes now handle ComfyUI's interrupt button while waiting for a response or receiving output. Cancellation closes the connection and discards partial results. The SDK-free path uses ComfyUI's aiohttp dependency. Remote generation and billing remain provider-dependent. Node inputs, IDs and output positions are unchanged.

## 0.7.1 — 仓库更名 / Repository rename

- GitHub 仓库统一为 `xzbdqian10nian/ComfyUI-TankNodes`，更新简介、安装、文档与问题反馈链接。
- 已有安装可保留 `ComfyUI-Qwen3.8-VL` 目录，修改 origin 后直接更新；只保留一份插件。
- 作者署名统一为 Tank，改写 README、节点说明和示例注释。
- 节点功能、ID、控件与输出位置沿用 0.7.0。Registry 包 ID `qwen38-vl` 保留兼容，模型文件与目录不变。

The repository, installation links and project metadata now use **ComfyUI-TankNodes**. Existing installations can keep their folder and update the Git remote. Author credits now use Tank. The README, node descriptions and example notes have been rewritten. Node behavior, saved IDs and output positions are unchanged from 0.7.0.

## 0.7.0 — TankNodes 整理 / Toolkit cleanup

- 统一五个节点的中英文名称、Tank 后缀和分类；整理共用对话、模型目录与节点实现。
- 修复未知帧数视频的全段抽帧、媒体统计、原生视频 MIME 类型、加载失败清理与流式资源关闭。
- API 末尾追加思考/原始响应输出，保留旧端口位置；支持有执行依赖的模型释放。
- 兼容旧 API 控件和有明确元数据的旧提示词顺序，补全八份示例与双语说明。

Unified bilingual node names and categories; split shared chat and node modules. Fixed video sampling, media statistics and cleanup. Added API reasoning outputs and ordered model release while preserving old node IDs and output positions. Added historical workflow migration and eight examples.
