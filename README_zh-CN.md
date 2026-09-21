# TankNodes

[English](README.md) · 简体中文

Tank 的 ComfyUI 节点合集。现在可以用本地 Qwen3.8 处理文字、图片和视频，也可以通过 API 调用其他模型。

在节点菜单里搜索 **Tank** 就能找到。

## 有哪些节点

| 节点 | 怎么用 |
| --- | --- |
| Qwen3.8 模型加载 · Tank | 选择主模型和视觉投影，接到本地对话节点。 |
| 本地多模态对话 · Tank | 输入提示词，按需接入图片、图片批次或视频，输出文字。 |
| API 对话（环境变量密钥）· Tank | 从服务器环境变量读取 API Key。 |
| API 对话（直接密钥）· Tank | 在节点里填写 API Key。分享工作流前记得清空。 |
| 模型释放 · Tank | 释放显存；把对话的回答接到“等待完成”，就会在生成结束后执行。 |

每次运行是一次独立请求，不会自动记住上一次的对话。API 节点可以单独使用，不需要本地模型加载器。

## 安装

在 `ComfyUI/custom_nodes` 下执行：

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-TankNodes.git
```

重启 ComfyUI，然后刷新浏览器。Python 需要 3.10 或更新版本；额外依赖见 [requirements.txt](requirements.txt)，使用 ComfyUI 自己的 Python 环境安装。

本地推理还需要支持 Qwen3.8、与现有 CUDA 匹配的 `llama-cpp-python`。用 API 不需要这项依赖。插件不会替换 Torch、CUDA 或显卡驱动。

## 先跑一个本地示例

1. 把 GGUF 主模型和对应的视觉投影放进 `ComfyUI/models/LLM/Qwen3.8/`。
2. 打开 [01_text_chat.json](example_workflows/01_text_chat.json)，在加载器里选好两个文件。
3. 修改“用户输入”，点击运行。权重会在对话开始时加载。

本次测试用的是 [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) 的 `Qwen3.8-27B-UD-Q4_K_M.gguf` 和 `mmproj-BF16.gguf`。其他已收录的文件见 [models.json](models.json)。主模型和投影要使用配套版本，不要只看文件名相似就混用。

模型目录支持子文件夹，也可以用 `QWEN38_MODEL_DIR` 指定其他位置。插件不下载模型；下拉框显示“未找到”时，先检查文件是否放对、下载是否完成。

## 图片和视频

- **图片**：接单张图片或图片批次，批次里的每张图都会发送。
- **视频帧**：接已经解码的 IMAGE 批次，按“最多视频帧数”抽样。
- **视频**：直接接 VIDEO，本地模型会从整段视频里均匀抽帧。

API 默认用 `frames` 发送图片帧。接口支持原生视频时，可以选 `video_url`；`auto` 会先编码视频，编码失败再抽帧。接口返回错误后不会自动改成抽帧重发。

[示例目录](example_workflows/) 里还有单图、多图、两种视频输入、两种 API 密钥方式和模型释放，共 8 份。

## API 怎么填

填写接口基础地址、服务商给出的**模型 ID**和密钥。例如基础地址是 `https://api.openai.com/v1`，不用加 `/chat/completions`。

如果选环境变量密钥，先在启动 ComfyUI 的环境里设置：

```bash
export OPENAI_API_KEY='your-api-key'
```

节点里只填 `OPENAI_API_KEY` 这个变量名。默认允许访问 OpenAI 官方和本机服务；接其他服务商时，管理员还需要配置允许的主机：

```bash
export COMFYUI_API_ALLOWED_HOSTS='api.openai.com,api.example.com'
```

除本机服务外，环境变量密钥模式要求 HTTPS。直接密钥模式使用节点里填写的 Key。

API 的“最大输出 token 数”和“随机度”设为 0 时，不发送这两项参数，交给服务商决定。本地的随机度 0 则表示确定性采样。API 返回回答、用量、运行统计、思考内容和原始响应；服务商没有返回思考或 token 用量时，对应内容为空或显示 n/a。

等待 API 响应或接收输出时，可以点击 ComfyUI 的中断按钮停止等待并关闭连接。服务商是否停止生成和计费，取决于对方接口。

## 几个常用设置

- 本地 `auto` 沿用默认的不思考模式，`off` 明确关闭思考。API 的 `auto` 使用服务商默认设置。
- 本地 Qwen3.8 的 `high` 按 `medium` 执行，`max` 按 `xhigh` 执行；运行统计里能看到实际档位。切换思考强度不重新加载权重，修改上下文长度会重新加载。
- 连续对话可以保留模型；后面还要跑其他模型时，打开“生成后卸载”。
- Nodes 2.0 可以收起高级参数，经典画布会把参数一起显示。

## 开发

测试方法见 [tests/README.md](tests/README.md)。许可证为 [MIT](LICENSE)。

感谢 Qwen 团队、llama.cpp 和社区 GGUF 发布者。
