# TankNodes 更新记录 / Changelog

## 0.8.0 — H3 提示词引擎、长视频窗口、360° 全景与生图尺寸套件

- 新增 H3 视频编辑、MV 分段、人设反推、数字人口播/唱歌等提示词模板，以及 `H3 Image` 官方 PE（文生图/编辑）与画幅/尺寸节点（`TankH3ImageSize`、`TankH3ImageAspect`、`TankImageSize`）。
- 新增 3D 多角度相机与 360° 全景套件（`TankMultiAngleCamera`、`TankPanoramaSetup`、`TankPanoramaSeamlessPatch`、`TankPanoramaViewer`）。
- 新增 H3 长视频上下文窗口规划、9 图分窗口编码与遮罩锁定节点（`TankH3WindowPlan`、`TankH3WindowedReferenceToVideo`、`TankH3MaskedSourceLatent`）。
- 模板管理器名内「/」统一改用「、」并用 `aliases` 向后兼容旧工作流，自建模板重名与改名隔离，内置模板改为从 `prompt_templates/` 读取并由 `TankPromptTemplate` 统一选用。

### H3 视频编辑提示词 / H3 video-edit prompts

- 视频用途 auto 改为按需求文字判断：写了替换/换成/去掉 → 编辑原片，续拍 → 续拍，参考动作/跟着跳 → 动作参考；都没写才按素材兜底（以前「接了参考图」一律当动作参考，换人任务会写成参考生成）。`H3 Prompt Optimizer · Tank` 的源视频用途新增 auto 选项，默认值不变。
- 源视频自动找硬切，每个镜头至少给模型看一帧并标注镜头号和秒数；编辑任务要求镜头数与切点和原片一致，镜头数对上时时间戳自动对齐到原片切点。
- 编辑任务默认保留原声：补 `<Audio 1>`（原片音轨，fully_copy）与 `[… + audio reuse]`；retention 要求 `<Video 1>` partially_preserved。新增「视频帧率」「保留原声」输入。
- 编辑任务不再要求 350–500 词逐拍复述动作：只写清换上的角色，动作、镜头、时序写「沿用 <Video 1>」（复述不准会误导动作）。时长填 0 时按原片长度。
- 新增「H3 视频编辑 · 通用模板」：先让小模型依据需求、参考图和源视频拼图反推目标与替换内容，再用一条无时间戳、无逐镜头叙述的通用提示词覆盖整支视频；支持代码确定性渲染与官方规则 LLM 渲染，并在分析失败时输出通用默认兜底稿并在报告标注 `analysis_failed`。
- 视频编辑通用模板增加画风来源（默认跟原片 / 保持参考图画风，含 `[[style:picture|video]]` 指令与「保持参考图画风」独立预设），并补充一图多用途（如人物+背景）、多余参考图未使用、任务多于参考图及多人参考图的兜底规则与中文报告备注。
- 统一三个 H3 提示词入口（`Prompt Optimizer · Tank`、`H3 Prompt Optimizer · Tank`、`API Chat · Tank`）共用唯一引擎 `run_h3` 与超预算自动扩容重试；API 节点新增 H3 可选参数（`seconds`、`video_role`、`video_fps`、`keep_source_audio`），贯穿 `max_tokens=0`（不设限，由服务商决定）规则，并支持「节点填写 → 运行时配置 → 本地默认配置」三层密钥解析（已保存密钥与配置 endpoint 严格绑定）与密钥脱敏。
- 视频编辑通用模板与分工重构：客户原话与素材由模型结构化解析出 `source.picture`、背景编辑、`drop_parts` / `add_parts` / `keep_original_parts` 与动作迁移的 `performer`；`style` 仅作风格参考不建 `<Subject>`；换装默认只排除脸、头发、体型与手持物，尾巴、耳朵、翅膀、头饰默认随套装带出（除非客户明确排除）；换头默认整颗头替换并避写原头发与原帽子描述；保留原帽子或指定舞者丢失、分析二次失败兜底等情况会在报告给出明确提示。
- 新增 MV 分段提示词预设（`H3 MV · 分段提示词（一句话→分镜、运镜、特效，每段独立）`）与人设反推预设（`人设反推 · 参考图→固定人设句（MV、长视频复用）`）：`run_h3` 识别 `[[tank:h3-mv]]` 进入 Ref2VA MV 模式，校验镜头递增、歌词 `<d>` 与 `persona_N` 保留。

Added the 0.8.0 node suites (360° camera & panorama, H3 Image sizing/PE, H3 long-video windowing & masked latent, MOSS manga subpackage) and the unified model-driven H3 prompt optimizer with template alias compatibility.

### 安装即用 / Works after install

- 模型首次使用时自动下载：加载器列出 models.json 收录的主模型和视觉投影，本地缺少时从 Hugging Face 断点续传下载到 `models/LLM/Qwen3.8`，支持 `HF_ENDPOINT` 镜像；`TANKNODES_AUTO_DOWNLOAD=0` 可关闭。
- PrismML llama-server 自动安装：按系统与 PyTorch CUDA 版本下载固定的官方 `prism-b10743-adfffbe` 构建，校验 SHA-256 后解压到插件 `runtime/`。查找顺序为 `BONSAI_LLAMA_SERVER` → `runtime/` → 自动下载；删除写死的实例路径。
- Qwen 未安装 llama-cpp-python 时改用同一 llama-server（`auto` 思考保持关闭）；已安装时行为不变，`TANKNODES_LOCAL_RUNTIME` 可强制选择。
- Bonsai 视觉投影不匹配时在下载前报错。README 去除实例专用路径。

Listed models now download on first use (resumable, mirror-aware) and the pinned PrismML llama-server installs itself into the plugin's `runtime/` folder after SHA-256 verification. Qwen falls back to llama-server when llama-cpp-python is missing. Instance-specific paths were removed.

### 文本分割 / Text split

- 新增“文本分割 · Tank”（`TankTextSplit`），可替代 mixlab 的 Text Split By Delimiter：同样支持开始索引、每隔跳过和最大数量，并额外输出实际项数。
- 分隔符只转换 `\n`、`\t`、`\r`、`\\`，中文逗号等非 ASCII 分隔符不再被错误解码。高级参数可去除首尾空白或启用正则；没有可输出的项时明确报错。

Added Text Split · Tank as a drop-in alternative to mixlab's Text Split By Delimiter, with a count output, safe handling of non-ASCII delimiters, optional trimming and regex mode, and an explicit error when nothing is output.

### Ternary Bonsai 本地反推

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
