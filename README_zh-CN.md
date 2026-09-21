# TankNodes

[English](README.md) | 简体中文

**Tank / xzbd** 的 ComfyUI 日常节点工具箱。当前提供本地 Qwen3.8 多模态理解和 OpenAI 兼容 API 对话，所有节点名称以 **· Tank** 结尾，统一位于 **TankNodes** 菜单下。

本地使用 **模型加载 → 本地对话**；两个 API 对话节点直接发请求、输出结果，不连接本地对话节点。内部共用媒体处理和回答解析。

## 快速开始

在 `ComfyUI/custom_nodes` 目录执行：

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-Qwen3.8-VL.git
```

为兼容升级，仓库地址、安装目录、Registry 包 ID `qwen38-vl` 和内部节点 ID 继续保留。TankNodes 是显示名称，同一环境只安装一份本插件。

1. 将 GGUF 主模型及匹配的视觉投影文件放入 `ComfyUI/models/LLM/Qwen3.8/`，支持子目录；可用 `QWEN38_MODEL_DIR` 指定独立目录。
2. 重启 ComfyUI 并刷新浏览器。
3. 添加 **Qwen3.8 模型加载 · Tank**，选择两个文件。没有模型时会明确显示“未找到”，不会用推荐文件名冒充已安装模型。
4. 将 **模型配置** 输出连接到 **本地多模态对话 · Tank**。
5. 填写人设与用户输入，按需连接图片或视频并运行。执行对话时才加载权重。

使用 API 时，直接添加任一 API 对话节点，填写接口地址、准确的 **模型 ID**、密钥来源和提示词，无需本地 GGUF 模型。

### 更新

```bash
git -C ComfyUI-Qwen3.8-VL pull --ff-only
```

在现有插件目录的上级执行，或进入原目录运行 `git pull --ff-only`。更新后重启 ComfyUI 并刷新浏览器。

## 节点

| 节点 | 用途 |
| --- | --- |
| Qwen3.8 模型加载 · Tank | 选择 GGUF 文件，准备本地模型配置。 |
| 本地多模态对话 · Tank | 使用本地模型完成一次文本、单图、多图或视频请求。 |
| API 对话（环境变量密钥）· Tank | 使用服务器环境变量中的密钥直接请求 API。 |
| API 对话（直接密钥）· Tank | 使用节点中填写的密钥直接请求 API。 |
| 模型释放 · Tank | 释放模型；把对话回答接入“等待完成”，明确先生成后释放。 |

每次运行都是独立请求，不会自动保留多轮聊天历史。常用输入保持可见，采样、上下文和传输配置使用原生“高级参数”标记；在 ComfyUI 的 Nodes 2.0 界面中可以展开/收起，经典画布仍一起显示；不修改控件序列化。控件顺序与保存值保持不变。本地的视频传输设置保留为高级兼容项，本地模型始终抽帧。

单次对话可以直接开启 **生成后卸载**；需要独立释放节点时，连接其 **等待完成** 输入建立执行顺序。

## 模型与思考强度

主模型与视觉投影应来自同一发布版本。选项保留真实文件名，避免用商品名替换执行参数。加载器的 **模型信息** 会给出已知来源、版本、推荐投影及文件名是否匹配；这些是文件名层面的提示，不代替模型架构或文件哈希校验。自行改名的文件无法可靠识别来源。

| 来源 | 常用文件 | 区别 |
| --- | --- | --- |
| [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | [UD Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf?download=true) + [mmproj-BF16](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf?download=true) | 推荐作为起点；动态量化体积较小，保留原模型的对齐方式。 |
| [Huihui AI Abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | [Q4_K](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf?download=true) 或 Q4_K_L + `mmproj-model-bf16.gguf` | 社区低拒答版本；较大的 Q4_K_L 会保留更多高精度张量。 |
| [Orcarouter Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | [Q4_K_M](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-Q4_K_M.gguf?download=true) + [mmproj f16](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/mmproj-Qwen3.8-27B-Uncensored-f16.gguf?download=true) | 另一种社区低拒答版本；Hugging Face 可能要求登录或同意访问条件。 |

以上为原有模型目录资料，下载前请核对来源当前文件及模型说明。插件不包含或自动下载模型权重。

思考选择继续提供 `auto / off / low / medium / high / xhigh / max`：

- 本地 `auto` 沿用旧版本默认关闭思考，`off` 明确关闭。
- 当前 Qwen3.8 本地适配将 `high` 按 `medium`、`max` 按 `xhigh` 执行；运行统计同时显示选择值与实际值。切换思考强度不重载权重，改变上下文长度会重载。
- API `auto` 不修改服务商默认行为，其他选项按适配逻辑发送，实际支持情况由接口决定。旧 llama.cpp 若只支持开关，统计会明确说明。

## 图片与视频

**图片** 输入会发送整个图片批次；**视频帧** 输入会按“最多视频帧数”抽样 IMAGE 批次；**视频** 输入会覆盖整段视频均匀抽样。缺少总帧数信息的容器先计数再抽样，不再只取片头。插件不会默默缩小图片。

API `frames` 发送抽取的图片帧，`video_url` 发送原生视频。`auto` 仅在本地编码失败时回退抽帧，不会在服务商拒绝请求后自动重试。服务商仍须支持对应格式。原生上传保留真实容器 MIME 类型。统计分别记录图片数、实际抽帧数和原生视频数；后端未返回 token 用量时，速度显示 n/a。

## API 设置与输出

- **接口基础地址**：例如 `https://api.openai.com/v1`，不含 `/chat/completions`。
- **模型 ID**：填写服务商原始标识，不填写中文显示名。
- **最大输出 token 数 = 0**：省略限制，使用服务商默认值。
- **随机度 = 0**：API 不发送 temperature 参数；本地 0 保持原来的确定性采样语义。
- **随机种子**：按配置发送，是否支持取决于服务商。

API 前三个输出的位置保持 **回答、用量、运行统计**，末尾新增 **思考内容、原始响应**。服务商未返回思考内容时该输出为空。“用量”JSON 保持原有两个字符串类型 token 字段。本地仍为 **回答 / 思考内容 / 原始响应 / 运行统计**。

### 环境变量密钥

启动 ComfyUI 前设置环境变量，节点内只填写变量名称：

```bash
export OPENAI_API_KEY='your-api-key'
```

服务器允许列表默认包含 `api.openai.com` 和本机回环地址。其他服务商由管理员设置精确主机或 `host:port`：

```bash
export COMFYUI_API_ALLOWED_HOSTS='api.openai.com,api.example.com'
```

回环服务以外要求 HTTPS。直接密钥节点只使用填写的密钥；分享工作流前清空真实密钥。插件不携带作者密钥或私人接口地址。

## 示例工作流

[example_workflows](example_workflows/) 中有 8 份示例，覆盖纯文本、单图、多图、视频帧、原生 VIDEO、两种 API 密钥方式及有顺序的模型释放。原有本地示例的功能参数保留，操作说明统一中英对照，标准插件节点标题跟随界面语言。

## 旧工作流兼容

保留 5 个内部节点 ID、既有字段名、控件顺序、原输出端口位置、模型目录及 Registry 包 ID。`backend_default / thinking / instruct` 等旧思考值继续兼容；浏览器扩展识别旧的 11 控件 API 节点，补入缺失的思考默认值，不挪错视频设置。能从旧输入顺序确认的 0.5.0 提示词顺序也会修正，并标记为已迁移，避免重复交换。缺少版本/布局证据的更早工作流不猜测文字用途。用户自定义标题和连线保留。

本轮名称调整不代表 Registry 或 GitHub 仓库迁移。升级后应同时重启服务器和刷新浏览器，让节点定义与旧工作流兼容扩展使用同一版本。

## 依赖与开发

Python 3.10+，沿用 ComfyUI 原有 Torch 环境。本地推理需要适配当前 CUDA 的 `llama-cpp-python`；API 使用现有 OpenAI SDK，缺少时使用标准库 HTTP 实现。插件不会替换 Torch、CUDA 或驱动，`requirements.txt` 只保留少量额外依赖。

`__init__.py` 汇总注册；`local_nodes.py` 与 `api_nodes.py` 定义节点；`chat.py` 共用请求组装和回答处理；`backends.py` 管理后端；模型目录、媒体、思考参数和进度各自独立。`web/` 只处理旧工作流兼容，不自定义控件序列化。

开发测试在独立环境安装 pytest、Torch、NumPy、Pillow、PyAV、OpenAI，参见 [测试说明](tests/README.md)。

## 致谢

感谢 Qwen 团队及上述社区 GGUF 发布者。TankNodes 为独立项目，不包含模型权重，也不代表模型服务商对本插件的认可。
