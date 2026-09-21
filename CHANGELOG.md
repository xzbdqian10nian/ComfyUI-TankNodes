# TankNodes 更新记录 / Changelog

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
