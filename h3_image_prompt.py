"""H3 Image 官方 PE 协议的本地 Tank/Bonsai 适配；不涉及视频或生成图像缩放。"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from .h3_optimizer import _complete
from .media import pil_to_data_url, tensor_frame_to_pil

T2I_RATIOS = ("1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16")
TEMPLATE_MARKERS = {"[[tank:h3-image:t2i]]": "t2i", "[[tank:h3-image:edit]]": "edit"}
SYSTEM_SEPARATOR = "\n===== SYSTEM =====\n"
TASK_SEPARATOR = "\n===== TASK =====\n"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0}


def _ratio_parts(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", value):
        raise ValueError("比例必须是无空格的正整数 W:H，例如 16:9。")
    try:
        return tuple(map(int, value.split(":")))
    except ValueError as exc:
        raise ValueError("比例整数过长，无法解析。") from exc


def _target_ratio(selection: str, image: Any = None) -> str:
    if selection == "auto":
        return "adaptive"
    if selection == "跟随图像1":
        if image is None:
            raise ValueError("选择跟随图像1时，必须连接原始参考图端口1。")
        return _reduced_ratio(*_image_geometry(image))
    _ratio_parts(selection)
    return selection


def _reduced_ratio(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _image_geometry(image: Any) -> tuple[int, int]:
    shape = tuple(image.shape)
    if len(shape) not in (3, 4) or (len(shape) == 4 and shape[0] < 1):
        raise ValueError("图像必须是非空的 ComfyUI IMAGE（BHWC 或 HWC）。")
    height, width, channels = map(int, shape[-3:])
    if width <= 0 or height <= 0 or channels not in (3, 4):
        raise ValueError("图像宽高必须为正数，且需要 RGB 或 RGBA 通道。")
    return width, height


def _references(ports: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for port in range(1, 17):
        image = ports.get(f"reference_image_{port}")
        if image is None:
            continue
        width, height = _image_geometry(image)
        refs.append({"port": port, "image": image, "width": width, "height": height,
                     "aspect_ratio": _reduced_ratio(width, height)})
    return refs


def _manifest(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"port": ref["port"], "label": f"<Picture {index + 1}>",
             "native_input": f"images.image_{index}", "aspect_ratio": ref["aspect_ratio"],
             "width": ref["width"], "height": ref["height"], "batch_index": 0}
            for index, ref in enumerate(refs)]


def _parse_template(template: str) -> tuple[str, str, str]:
    marker, found, body = template.partition(SYSTEM_SEPARATOR)
    mode = TEMPLATE_MARKERS.get(marker)
    if not found or mode is None:
        raise ValueError("请接入 H3 Image 官方文生图或编辑模板；不能使用视频 H3 模板。")
    system, found, task = body.partition(TASK_SEPARATOR)
    if not found or not task.strip():
        raise ValueError("H3 Image 模板缺少明确的 SYSTEM/TASK 分隔或任务原文。")
    if mode == "t2i" and not system.strip():
        raise ValueError("文生图模板必须包含官方 SYSTEM 原文。")
    if mode == "edit" and system.strip():
        raise ValueError("官方编辑协议只有一条 user 消息，SYSTEM 部分必须留空。")
    return mode, system.strip(), task


def _build_messages(mode: str, system: str, task: str, request: str,
                    refs: list[dict[str, Any]], target_ratio: str) -> list[dict[str, Any]]:
    if mode == "t2i":
        if refs:
            raise ValueError("文生图模板不能接参考图，请改接 H3 Image 官方编辑模板。")
        input_text = request
        if target_ratio != "adaptive":
            input_text += ("\n\n调用者附注（最终画布约束，不是图内文字）：\n"
                           f"用户明确选择的最终画幅为 {target_ratio}（宽:高）。"
                           "请按此画布规划构图，不得把本附注、比例数字或画幅标签作为图像中的文字。")
        extra_sep = "\n\nAdditional rules from the workflow owner:\n"
        base_task, found_extra, extra_tail = task.partition(extra_sep)
        try:
            user_prompt = base_task.format(input_text)
        except (ValueError, IndexError, KeyError) as exc:
            raise ValueError("文生图任务模板格式无效，请保留官方花括号及请求占位符。") from exc
        if found_extra:
            user_prompt = user_prompt + extra_sep + extra_tail
        return [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    if not refs:
        raise ValueError("H3 Image 编辑至少需要一张参考图。")
    count = len(refs)
    content = [{"type": "text", "text": task},
               {"type": "text", "text": (f"Reference Image Count:\n{count} input image(s), "
                                         f"labeled <Picture 1> through <Picture {count}>.")}]
    for index, ref in enumerate(refs, 1):
        image = ref["image"]
        frame = image[0] if len(image.shape) == 4 else image
        preview = tensor_frame_to_pil(frame)
        # 仅供 VLM 看图；原 IMAGE 不修改、不输出，生成引用另走官方 Resize/native 流程。
        preview.thumbnail((1536, 1536))
        content.extend([
            {"type": "text", "text": f"Reference Media:\n- <Picture {index}> [aspect_ratio {ref['aspect_ratio']}]"},
            {"type": "image_url", "image_url": {"url": pil_to_data_url(preview, quality=88)}},
        ])
    content.append({"type": "text", "text": (
        "\nUser Editing Instruction (JSON string):\n"
        f"{json.dumps(request, ensure_ascii=False)}\n\n"
        f"User Aspect Ratio Selection:\n{target_ratio}\n\n"
        "Apply the instructions above to the supplied inputs and reference images. "
        "Output only the required two-field JSON object."
    )})
    return [{"role": "user", "content": content}]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 含有重复字段：{key}。")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"JSON 包含非标准常量：{token}。")


def _parse_result(raw: str, mode: str, target_ratio: str) -> dict[str, Any]:
    # think 由复用的 _complete -> chat._split_message 处理，不搜索/截取任意 JSON 子串。
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    parse_err: ValueError | None = None
    try:
        data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
    except ValueError:
        parse_err = ValueError("模型未返回有效的单一 JSON 对象（可能有未转义引号、重复字段、非标准常量或额外文本）。")
    if parse_err is not None:
        raise parse_err
    ratio_key = "aspect_ratio" if mode == "t2i" else "target_aspect_ratio"
    if not isinstance(data, dict):
        raise ValueError("模型必须返回包含 expanded_caption 的 JSON 对象。")
    if mode == "edit" and set(data) != {"expanded_caption", ratio_key}:
        raise ValueError(f"JSON 必须恰好包含 expanded_caption 和 {ratio_key} 两个字段。")
    caption = data.get("expanded_caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("expanded_caption 必须是非空字符串。")
    if mode == "t2i":
        # 文生图只消费正文；保留旧比例输出端口，但建议比例不能阻断可用正文。
        # 标准 JSON 转义解码后的换行和额外字段不影响正文，不猜测修复畸形 JSON。
        ratio = data.get(ratio_key)
        if not isinstance(ratio, str) or ratio not in T2I_RATIOS:
            ratio = target_ratio if target_ratio != "adaptive" else "1:1"
        return {**data, ratio_key: ratio}
    if "\n" in caption or "\r" in caption:
        raise ValueError("expanded_caption 必须是单段文字，不能含实际换行。")
    ratio = data[ratio_key]
    _ratio_parts(ratio)
    if target_ratio != "adaptive" and ratio != target_ratio:
        raise ValueError(f"编辑结果比例必须逐字保留明确选择 {target_ratio}，不得由 PE 覆盖。")
    return data


class TankH3ImagePromptOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": ("VISION_LLM_BACKEND", {"lazy": True, "tooltip": "连接现有 Bonsai-2 v2 / Tank VLM 后端；关闭 PE 时不加载。"}),
                "template": ("STRING", {"forceInput": True, "tooltip": "连接 H3 Image 官方模板文本，与视频 H3 独立。"}),
                "request": ("STRING", {"multiline": True, "default": "", "tooltip": "原始生图或编辑指令；字面文字和引号按原文传递。"}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "关闭后返回原请求，不调用或卸载后端。"}),
                "thinking_mode": (["high"], {"default": "high", "tooltip": "固定使用 high，不自动降低思考强度。"}),
                "context_length": ("INT", {"default": 65536, "min": 16384, "max": 131072, "step": 1024,
                                            "tooltip": "文生图推荐16384；多图编辑至少32768。旧工作流参数不自动改动。"}),
                "max_tokens": ("INT", {"default": 16384, "min": 4096, "max": 65536, "step": 1024,
                                        "tooltip": "文生图推荐4096；图像编辑至少10240。包含思考与最终答案。"}),
                "correction": ("BOOLEAN", {"default": False, "tooltip": "返回格式失败时额外重试一次，默认关闭；不检查或纠正画面语义。"}),
                "unload_after": ("BOOLEAN", {"default": True, "tooltip": "结束或失败后卸载 VLM，为生图释放显存。"}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "aspect_ratio": ("STRING", {"forceInput": True, "default": "auto", "tooltip": "与尺寸节点共用画幅选择的 STRING：auto、跟随图像1或明确 W:H；auto 在编辑协议中转为 adaptive。"}),
                **{f"reference_image_{i}": ("IMAGE", {"tooltip": f"参考图端口 {i}；接原始图、只取 batch 第一张，按端口连续编号。生成引用另走官方 Resize 流程。"})
                   for i in range(1, 17)},
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("expanded_caption", "recommended_ratio", "report_json")
    OUTPUT_TOOLTIPS = ("仅扩写正文，直接接生成提示词；不是 JSON 包装。", "PE 建议比例，不能覆盖用户明确画布。", "原始回复、解析 JSON、图像编号、用量、耗时及 usedPE。")
    FUNCTION = "run"
    CATEGORY = "TankNodes/H3 Image"
    DESCRIPTION = "以本地 Bonsai/Tank 执行官方 H3 Image PE 协议；与视频 H3 独立，不改变生成参考图。"

    def check_lazy_status(self, backend=None, enabled=True, **kwargs):
        return ["backend"] if enabled and backend is None else []

    def run(self, template, request, backend=None, enabled=True, thinking_mode="high", context_length=65536,
            max_tokens=16384, correction=False, unload_after=True, seed=1234, aspect_ratio="auto", **ports):
        started = time.perf_counter()
        if not enabled:
            refs = _references(ports)
            ratio = _target_ratio(aspect_ratio, ports.get("reference_image_1"))
            if ratio == "adaptive":
                ratio = refs[0]["aspect_ratio"] if refs else ""
            report = {"usedPE": False, "mode": "bypass", "user_selected_ratio": aspect_ratio,
                      "status": "PE 已关闭，原请求直通；未调用、配置或卸载后端。",
                      "raw_json": None, "raw_responses": [], "image_count": len(refs), "images": _manifest(refs),
                      "attempts": 0, "corrected": False, "usage": [], "elapsed_seconds": time.perf_counter() - started}
            report_str = json.dumps(report, ensure_ascii=False)
            return {"ui": {"text": (request,)}, "result": (request, ratio, report_str)}
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise ValueError("请连接现有 Bonsai-2 v2 / Tank VLM 后端。")
        try:
            if thinking_mode != "high":
                raise ValueError("H3 Image PE 使用 high 思考模式，不允许自动降档。")
            mode, system, task = _parse_template(template)
            min_ctx, min_output = (16384, 4096) if mode == "t2i" else (32768, 10240)
            if int(context_length) < min_ctx or int(max_tokens) < min_output:
                raise ValueError(f"H3 Image {mode} PE 上下文至少 {min_ctx}，输出预算至少 {min_output}。")
            if int(max_tokens) >= int(context_length):
                raise ValueError("输出预算必须小于上下文长度，并为模板及参考图留出空间。")
            if not isinstance(request, str) or not request.strip():
                raise ValueError("生图或编辑请求不能为空。")
            target_aspect_ratio = _target_ratio(aspect_ratio, ports.get("reference_image_1"))
            refs = _references(ports)
            messages = _build_messages(mode, system, task, request, refs, target_aspect_ratio)
            configure = getattr(backend, "configure_chat", None)
            if not callable(configure):
                raise ValueError("后端缺少 configure_chat，无法保证所选上下文和 high 思考配置。")
            configure(int(context_length), "high")
            responses, usages, errors = [], [], []
            for attempt in range(2 if correction else 1):
                raw, usage = _complete(backend, messages, max_tokens=int(max_tokens), thinking_mode="high",
                                       seed=(int(seed) + attempt) % (2**31), sampling=dict(SAMPLING))
                responses.append(raw)
                usages.append(usage)
                final_parse_err: ValueError | None = None
                try:
                    result = _parse_result(raw, mode, target_aspect_ratio)
                    break
                except ValueError as exc:
                    errors.append(str(exc))
                    if attempt == (1 if correction else 0):
                        final_parse_err = ValueError(
                            f"H3 Image PE 返回格式不合格（已调用 {attempt + 1} 次），停止生成，不回退原请求：{exc}"
                        )
                    else:
                        required_format = ("必须包含非空字符串 expanded_caption；aspect_ratio 建议字段可选。"
                                           if mode == "t2i" else
                                           "恰好包含 expanded_caption 和 target_aspect_ratio 两个字符串字段，正文非空且保持单段。")
                        fix = (f"上次回答的返回格式不合格：{exc}\n请仅修复格式并返回完整 JSON 对象，"
                               f"{required_format}"
                               "保留原任务和字面文字，正确进行 JSON 转义；不要推理、代码围栏或额外文字。")
                        messages = messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": fix}]
                if final_parse_err is not None:
                    raise final_parse_err
        finally:
            if unload_after and callable(getattr(backend, "unload", None)):
                backend.unload()
        ratio_key = "aspect_ratio" if mode == "t2i" else "target_aspect_ratio"
        report = {"usedPE": True, "mode": mode, "user_selected_ratio": aspect_ratio,
                  "status": ("返回格式校验通过（不检查画面语义）；仅使用 VLM 预览，生成参考图保持原始输入。"
                             if mode == "t2i" else
                             "官方 PE 协议校验通过；仅使用 VLM 预览，生成参考图保持原始输入。"),
                  "raw_json": result, "raw_responses": responses, "image_count": len(refs), "images": _manifest(refs),
                  "attempts": len(responses), "corrected": len(responses) > 1, "validation_errors": errors,
                  "usage": usages, "elapsed_seconds": time.perf_counter() - started,
                  "thinking_mode": "high", "context_length": int(context_length), "max_tokens": int(max_tokens)}
        print(f"[TankNodes] H3 Image PE: mode={mode} attempts={len(responses)} "
              f"elapsed={report['elapsed_seconds']:.2f}s context={int(context_length)} "
              f"output_budget={int(max_tokens)} kept={not unload_after}", flush=True)
        report_str = json.dumps(report, ensure_ascii=False)
        return {"ui": {"text": (result["expanded_caption"] + "\n\n---\n" + report_str,)},
                "result": (result["expanded_caption"], result[ratio_key], report_str)}


class TankH3ImageSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "aspect_ratio": ("STRING", {"forceInput": True, "default": "auto", "tooltip": "与优化器共用同一画幅选择的 STRING。明确比例优先；auto 才采用 PE 建议，其次图像1，最后 1:1。"}),
                    "megapixels": ("FLOAT", {"default": 4.0, "min": 4.0, "max": 20.0, "step": 0.5,
                                             "tooltip": "按 ComfyUI 约定，以 MP × 1024² 计算面积，宽高分别就近 32 对齐。"}),
                },
                "optional": {
                    "recommended_ratio": ("STRING", {"forceInput": True, "tooltip": "连接 PE 建议比例；明确比例或跟随图像1时忽略。"}),
                    "image": ("IMAGE", {"tooltip": "原始图像1；按原始 shape 取比例，不改变图像。"}),
                }}

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "actual_ratio")
    OUTPUT_TOOLTIPS = ("32 对齐的生成宽度。", "32 对齐的生成高度。", "对齐后真实宽高的约分比例。")
    FUNCTION = "run"
    CATEGORY = "TankNodes/H3 Image"
    DESCRIPTION = "H3 Image 尺寸：用户明确比例优先于 PE；仅计算宽高，不缩放参考图。"

    def run(self, aspect_ratio="auto", megapixels=4.0, recommended_ratio="", image=None):
        if not math.isfinite(float(megapixels)) or not 4 <= float(megapixels) <= 20:
            raise ValueError("像素面积必须在 4 到 20 MP 之间。")
        if aspect_ratio == "跟随图像1":
            if image is None:
                raise ValueError("选择跟随图像1时，必须连接原始图像1。")
            rw, rh = _image_geometry(image)
        elif aspect_ratio != "auto":
            rw, rh = _ratio_parts(aspect_ratio)
        elif recommended_ratio:
            rw, rh = _ratio_parts(recommended_ratio)
        elif image is not None:
            rw, rh = _image_geometry(image)
        else:
            rw, rh = 1, 1
        area = float(megapixels) * 1024**2
        try:
            width, height = math.sqrt(area * rw / rh), math.sqrt(area * rh / rw)
            if not math.isfinite(width) or not math.isfinite(height) or min(width, height) < 32:
                raise ValueError
            width = max(32, int(round(width / 32)) * 32)
            height = max(32, int(round(height / 32)) * 32)
        except (ValueError, OverflowError) as exc:
            raise ValueError("所选比例在当前像素面积下无法合理进行 32 对齐，请调整比例、面积或直接指定生成宽高。") from exc
        return width, height, _reduced_ratio(width, height)


class TankH3ImageAspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ratio": (["auto", "跟随图像1", *T2I_RATIOS], {"default": "auto", "tooltip": "同一个 STRING 输出同时接优化器和尺寸节点，避免重复画幅控件。"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("aspect_ratio",)
    FUNCTION = "run"
    CATEGORY = "TankNodes/H3 Image"
    DESCRIPTION = "共享画幅选择；输出 STRING，不把 STRING 错接 COMBO。"

    def run(self, ratio="auto"):
        if ratio not in ("auto", "跟随图像1"):
            _ratio_parts(ratio)
        return (ratio,)


NODE_CLASS_MAPPINGS = {"TankH3ImagePromptOptimizer": TankH3ImagePromptOptimizer,
                       "TankH3ImageSize": TankH3ImageSize, "TankH3ImageAspect": TankH3ImageAspect}
NODE_DISPLAY_NAME_MAPPINGS = {"TankH3ImagePromptOptimizer": "H3 Image Official Prompt Optimizer · Tank",
                              "TankH3ImageSize": "H3 Image Size · Tank", "TankH3ImageAspect": "H3 Image Aspect Ratio · Tank"}
