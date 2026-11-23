"""仅运行本文件：python -B -I tests/test_h3_image_prompt.py。

标准库单元测试：模拟 IMAGE/PIL 转换和后端，不需权重、GPU、Torch 或 Pillow。
真实复用 h3_optimizer._complete 和 chat 的回复解析；不执行插件 __init__。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_tank_h3_image_unit"


def load_nodes():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE] = package
    media = types.ModuleType(PACKAGE + ".media")
    for name in ("encode_video_data_url", "extract_video_frames", "image_batch_frames", "sample_image_batch",
                 "pil_to_data_url", "tensor_frame_to_pil"):
        setattr(media, name, lambda *args, **kwargs: None)
    backends = types.ModuleType(PACKAGE + ".backends")
    backends.BackendError = type("BackendError", (RuntimeError,), {})
    sys.modules[media.__name__] = media
    sys.modules[backends.__name__] = backends
    # 只在导入期提供类型占位；不替换系统已安装的依赖，不执行媒体/后端引擎。
    with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
        module = importlib.import_module(PACKAGE + ".h3_image_prompt")
        video = importlib.import_module(PACKAGE + ".h3_optimizer")
    return module, video


nodes, video_module = load_nodes()
T2I = (ROOT / "prompt_templates/h3_image_t2i_official.txt").read_text(encoding="utf-8")
EDIT = (ROOT / "prompt_templates/h3_image_edit_official.txt").read_text(encoding="utf-8")


def answer(caption='The title reads "你好，世界".', ratio="3:2", edit=False):
    return json.dumps({"expanded_caption": caption, "target_aspect_ratio" if edit else "aspect_ratio": ratio},
                      ensure_ascii=False)


class FakeImage:
    def __init__(self, width, height, batch=2, frame=False):
        self.shape = (height, width, 3) if frame else (batch, height, width, 3)
        self.indexes = []
        self.first = None if frame else FakeImage(width, height, frame=True)

    def __getitem__(self, index):
        if index != 0:
            raise AssertionError("不允许读取 batch 中的其他帧")
        self.indexes.append(index)
        return self.first


class FakePreview:
    def __init__(self, frame):
        self.size = (frame.shape[1], frame.shape[0])
        self.bounds = None

    def thumbnail(self, bounds):
        self.bounds = bounds
        scale = min(1.0, bounds[0] / self.size[0], bounds[1] / self.size[1])
        self.size = tuple(max(1, round(v * scale)) for v in self.size)


class FakeBackend:
    backend_kind = "local_bonsai_server"

    def __init__(self, *responses):
        self.responses = list(responses or [answer()])
        self.calls = []
        self.configurations = []
        self.unloads = 0

    def configure_chat(self, context_length, thinking_mode):
        self.configurations.append((context_length, thinking_mode))

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            return response
        return {"choices": [{"message": {"content": response}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 101, "completion_tokens": 202}}

    def unload(self):
        self.unloads += 1


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.node = nodes.TankH3ImagePromptOptimizer()
        self.frames, self.previews = [], []

        def convert(frame):
            self.frames.append(frame)
            preview = FakePreview(frame)
            self.previews.append(preview)
            return preview

        converter = patch.object(nodes, "tensor_frame_to_pil", side_effect=convert)
        encoder = patch.object(nodes, "pil_to_data_url", side_effect=lambda preview, quality: f"data:test/{preview.size}")
        self.converter = converter.start()
        self.addCleanup(converter.stop)
        self.encoder = encoder.start()
        self.addCleanup(encoder.stop)

    def run_t2i(self, backend, **kwargs):
        out = self.node.run(template=T2I, request='写上“你好”及 ASCII "引号"，保留 {花括号}', backend=backend, **kwargs)
        return out["result"]

    def run_edit(self, backend, **kwargs):
        kwargs.setdefault("reference_image_1", FakeImage(2001, 1000))
        out = self.node.run(template=EDIT, request='把文字换成“你好”与 "你好\\世界"\n第二行', backend=backend, **kwargs)
        return out["result"]

    def test_official_templates_are_verbatim(self):
        mode, system, task = nodes._parse_template(T2I)
        self.assertEqual(mode, "t2i")
        self.assertEqual(hashlib.sha256(system.encode()).hexdigest(), "93e41b4aa53a6e3dc1364b55525b27d58f7c7134bbb4968fbdc7d227e10dc525")
        self.assertEqual(hashlib.sha256(task.encode()).hexdigest(), "890213ad61e143bbd5b7906a802430296e82ca5c61ba16df717dcef5477bbcf4")
        mode, system, task = nodes._parse_template(EDIT)
        self.assertEqual((mode, system), ("edit", ""))
        self.assertEqual(hashlib.sha256(task.encode()).hexdigest(), "c5695dbfc07a9e65555cef67d92467c613371cc2039b96f057022246a30a57b2")

    def test_t2i_format_sampling_budget_and_plain_caption(self):
        backend = FakeBackend()
        caption, ratio, report = self.run_t2i(backend)
        self.assertEqual((caption, ratio), ('The title reads "你好，世界".', "3:2"))
        call = backend.calls[0]
        _, system, task = nodes._parse_template(T2I)
        self.assertEqual(call["messages"], [{"role": "system", "content": system}, {"role": "user", "content": task.format('写上“你好”及 ASCII "引号"，保留 {花括号}')}])
        self.assertIn('{"expanded_caption":"ONE_DETAILED_ENGLISH_CAPTION"', call["messages"][1]["content"])
        self.assertNotIn('{{"expanded_caption"', call["messages"][1]["content"])
        self.assertEqual(backend.configurations, [(65536, "high")])
        self.assertEqual(call["max_tokens"], 16384)
        self.assertEqual(call["thinking_mode"], "high")
        for key, value in nodes.SAMPLING.items():
            self.assertEqual(call[key], value)
        self.assertNotIn("response_format", call)
        self.assertEqual(backend.unloads, 1)
        data = json.loads(report)
        self.assertTrue(data["usedPE"])
        self.assertEqual(data["user_selected_ratio"], "auto")
        self.assertEqual(data["raw_json"]["expanded_caption"], caption)
        self.assertEqual(data["raw_responses"], [answer()])
        self.assertEqual(data["usage"], [{"prompt_tokens": 101, "completion_tokens": 202}])
        self.assertEqual(data["attempts"], 1)
        self.assertGreaterEqual(data["elapsed_seconds"], 0)
        self.assertEqual(self.frames, [])

    def test_edit_has_unchanged_prefix_and_json_escaped_instruction(self):
        backend = FakeBackend(answer(edit=True))
        self.run_edit(backend)
        messages = backend.calls[0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user"])
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": nodes._parse_template(EDIT)[2]})
        self.assertEqual(content[2]["text"], "Reference Media:\n- <Picture 1> [aspect_ratio 2001:1000]")
        final = content[-1]["text"]
        encoded = final.split("User Editing Instruction (JSON string):\n", 1)[1].split("\n\nUser Aspect Ratio", 1)[0]
        self.assertEqual(json.loads(encoded), '把文字换成“你好”与 "你好\\世界"\n第二行')
        self.assertIn("User Aspect Ratio Selection:\nadaptive", final)
        self.assertNotIn("2001:1000", content[3]["image_url"]["url"])

    def test_nine_and_sixteen_images_keep_original_ratios_and_batch_first(self):
        for count in (9, 16):
            with self.subTest(count=count):
                self.frames.clear()
                self.previews.clear()
                ports = {f"reference_image_{i}": FakeImage(2001 + i, 1000 + i) for i in range(count, 0, -1)}
                shapes = {key: image.shape for key, image in ports.items()}
                backend = FakeBackend(answer(edit=True))
                _, _, report = self.run_edit(backend, **ports)
                content = backend.calls[0]["messages"][0]["content"]
                self.assertEqual(len(content), 3 + count * 2)
                self.assertIn(f"{count} input image(s)", content[1]["text"])
                for index in range(1, count + 1):
                    image = ports[f"reference_image_{index}"]
                    ratio = nodes._reduced_ratio(2001 + index, 1000 + index)
                    self.assertEqual(content[2 * index]["text"], f"Reference Media:\n- <Picture {index}> [aspect_ratio {ratio}]")
                    self.assertEqual(content[2 * index + 1]["type"], "image_url")
                    self.assertIs(self.frames[index - 1], image.first)
                    self.assertEqual(image.indexes, [0])
                    self.assertEqual(image.shape, shapes[f"reference_image_{index}"])
                    self.assertEqual(self.previews[index - 1].bounds, (1536, 1536))
                    self.assertLessEqual(max(self.previews[index - 1].size), 1536)
                manifest = json.loads(report)["images"]
                self.assertEqual(json.loads(report)["image_count"], count)
                self.assertEqual([r["native_input"] for r in manifest], [f"images.image_{i}" for i in range(count)])
                self.assertEqual([r["port"] for r in manifest], list(range(1, count + 1)))

    def test_sparse_ports_are_compressed_in_port_order(self):
        backend = FakeBackend(answer(edit=True))
        _, _, report = self.node.run(template=EDIT, request="编辑", backend=backend,
                                    reference_image_16=FakeImage(200, 300), reference_image_4=FakeImage(100, 100))["result"]
        manifest = json.loads(report)["images"]
        self.assertEqual([(r["port"], r["label"], r["aspect_ratio"]) for r in manifest],
                         [(4, "<Picture 1>", "1:1"), (16, "<Picture 2>", "2:3")])

    def test_edit_preserves_explicit_nonreduced_ratio_without_snapping(self):
        backend = FakeBackend(answer(ratio="32:18", edit=True))
        _, ratio, _ = self.run_edit(backend, aspect_ratio="32:18")
        self.assertEqual(ratio, "32:18")
        self.assertIn("User Aspect Ratio Selection:\n32:18", backend.calls[0]["messages"][0]["content"][-1]["text"])

    def test_t2i_explicit_ratio_is_caller_note_inside_input_text(self):
        backend = FakeBackend()
        request = '画海报，标题为“你好”与 "ASCII"，保留 {花括号}'
        out = self.node.run(template=T2I, request=request, backend=backend, aspect_ratio="9:16")
        self.assertFalse(getattr(self.node, "OUTPUT_NODE", False))
        self.assertIn('The title reads "你好，世界".', out["ui"]["text"][0])
        _, _, report = out["result"]
        _, system, task = nodes._parse_template(T2I)
        messages = backend.calls[0]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": system})
        prompt = messages[1]["content"]
        input_text = prompt.split("input_text: ", 1)[1].rsplit("\noutput_text:", 1)[0]
        self.assertTrue(input_text.startswith(request + "\n\n调用者附注"))
        self.assertIn("最终画幅为 9:16（宽:高）", input_text)
        self.assertIn("不是图内文字", input_text)
        self.assertIn("请按此画布规划构图", input_text)
        self.assertEqual(prompt, task.format(input_text))
        self.assertEqual(json.loads(report)["user_selected_ratio"], "9:16")

    def test_explicit_size_wins_even_when_t2i_recommends_other_canvas(self):
        backend = FakeBackend(answer(ratio="16:9"))
        selected = nodes.TankH3ImageAspect().run("9:16")[0]
        _, recommendation, _ = self.run_t2i(backend, aspect_ratio=selected)
        size = nodes.TankH3ImageSize().run(selected, recommended_ratio=recommendation)
        self.assertEqual(recommendation, "16:9")
        self.assertGreater(size[1], size[0])

    def test_adaptive_edit_accepts_nonpreset_ratio(self):
        self.assertEqual(self.run_edit(FakeBackend(answer(ratio="2001:1000", edit=True)))[1], "2001:1000")

    def test_follow_image_one_uses_original_geometry(self):
        backend = FakeBackend(answer(ratio="2001:1000", edit=True))
        self.run_edit(backend, aspect_ratio="跟随图像1")
        self.assertIn("User Aspect Ratio Selection:\n2001:1000", backend.calls[0]["messages"][0]["content"][-1]["text"])

    def test_ratio_override_is_rejected_even_when_numerically_equivalent(self):
        backend = FakeBackend(answer(ratio="16:9", edit=True), answer(ratio="16:9", edit=True))
        with self.assertRaisesRegex(ValueError, "不回退原请求"):
            self.run_edit(backend, aspect_ratio="32:18", correction=True)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.unloads, 1)

    def test_invalid_json_corrects_once_then_fails_and_unloads(self):
        invalid = '{"expanded_caption":"The title is "你好".","aspect_ratio":"1:1"}'
        backend = FakeBackend(invalid, "not JSON")
        with self.assertRaisesRegex(ValueError, "已调用 2 次.*不回退原请求"):
            self.run_t2i(backend, correction=True)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.unloads, 1)
        self.assertEqual(backend.configurations, [(65536, "high")])
        self.assertEqual([call["max_tokens"] for call in backend.calls], [16384, 16384])
        self.assertEqual([call["thinking_mode"] for call in backend.calls], ["high", "high"])
        self.assertEqual([call["seed"] for call in backend.calls], [1234, 1235])
        self.assertIn("上次回答", backend.calls[1]["messages"][-1]["content"])

    def test_successful_correction_reports_both_attempts(self):
        backend = FakeBackend("{}", answer())
        caption, _, report = self.run_t2i(backend, correction=True)
        data = json.loads(report)
        self.assertEqual(caption, 'The title reads "你好，世界".')
        self.assertTrue(data["corrected"])
        self.assertEqual(data["attempts"], 2)
        self.assertEqual(len(data["usage"]), 2)
        self.assertEqual(len(data["validation_errors"]), 1)
        self.assertEqual(data["raw_responses"], ["{}", answer()])
        self.assertEqual(backend.unloads, 1)

    def test_correction_defaults_off_and_can_be_disabled(self):
        for kwargs in ({}, {"correction": False}):
            with self.subTest(kwargs=kwargs):
                backend = FakeBackend("{}")
                with self.assertRaisesRegex(ValueError, "已调用 1 次"):
                    self.run_t2i(backend, **kwargs)
                self.assertEqual(len(backend.calls), 1)
                self.assertEqual(backend.unloads, 1)

    def test_strict_json_keys_types_and_ratios(self):
        invalid_t2i = [
            "[]", "null", "prefix " + answer(), answer() + " trailing", answer() + answer(),
            '{"expanded_caption":"a","expanded_caption":"b","aspect_ratio":"1:1"}',
            '{"expanded_caption":9,"aspect_ratio":"1:1"}', answer(caption="  "),
            '{"aspect_ratio":"1:1"}',
        ]
        for raw in invalid_t2i:
            with self.subTest(mode="t2i", raw=raw):
                backend = FakeBackend(raw)
                with self.assertRaises(ValueError):
                    self.run_t2i(backend)
                self.assertEqual(backend.unloads, 1)

        # 文生图模式下保留有效正文，并容忍额外字段、标准转义换行与非预设建议比例。
        for raw in (
            '{"expanded_caption":"first\\nsecond","aspect_ratio":"17:13","extra":0}',
            '{"expanded_caption":"caption only","aspect_ratio":2}',
        ):
            with self.subTest(mode="t2i_fallback", raw=raw):
                caption, ratio, _ = self.run_t2i(FakeBackend(raw))
                self.assertTrue(caption)
                self.assertEqual(ratio, "1:1")
                caption2, ratio2, _ = self.run_t2i(FakeBackend(raw), aspect_ratio="9:16")
                self.assertTrue(caption2)
                self.assertEqual(ratio2, "9:16")

        invalid_edit = [
            '{"expanded_caption":"a","target_aspect_ratio":"1:1","extra":0}',
            answer(caption="first\nsecond", edit=True),
            answer(ratio="0:1", edit=True),
            answer(ratio="1 : 1", edit=True),
            answer(ratio="1.5:1", edit=True),
            answer(ratio="-1:2", edit=True),
            '{"expanded_caption":"a","target_aspect_ratio":2}',
        ]
        for raw in invalid_edit:
            with self.subTest(mode="edit", raw=raw):
                backend = FakeBackend(raw)
                with self.assertRaises(ValueError):
                    self.run_edit(backend)
                self.assertEqual(backend.unloads, 1)

    def test_existing_completion_parser_handles_think_and_fences(self):
        final = answer()
        for raw in (f"<think>内部思考</think>\n```json\n{final}\n```", f"内部思考</think>\n{final}",
                    {"choices": [{"message": {"content": final, "reasoning_content": "独立思考"}}]}):
            with self.subTest(raw=raw):
                self.assertEqual(self.run_t2i(FakeBackend(raw))[0], 'The title reads "你好，世界".')

    def test_bypass_returns_request_and_never_touches_backend(self):
        backend = FakeBackend()
        request = ' 原样“中文” "ASCII"\n '
        with patch.object(nodes, "_complete", side_effect=AssertionError("不应调用后端")):
            out = self.node.run(template="非模板也允许关闭", request=request, backend=backend,
                                enabled=False, reference_image_1=FakeImage(600, 900))
        self.assertEqual(out["ui"]["text"], (request,))
        caption, ratio, report = out["result"]
        self.assertEqual((caption, ratio), (request, "2:3"))
        data = json.loads(report)
        self.assertFalse(data["usedPE"])
        self.assertEqual(data["mode"], "bypass")
        self.assertEqual(data["attempts"], 0)
        self.assertEqual(data["usage"], [])
        self.assertEqual((backend.calls, backend.configurations, backend.unloads), ([], [], 0))
        self.assertEqual(self.frames, [])

    def test_bypass_without_backend_or_images_and_explicit_ratio(self):
        self.assertEqual(self.node.run(template="", request="原请求", enabled=False)["result"][:2], ("原请求", ""))
        self.assertEqual(self.node.run(template="", request="原请求", enabled=False, aspect_ratio="9:16")["result"][:2], ("原请求", "9:16"))

    def test_lazy_backend_schema_and_disabled_branch(self):
        required = self.node.INPUT_TYPES()["required"]
        self.assertEqual(required["backend"][0], "VISION_LLM_BACKEND")
        self.assertTrue(required["backend"][1]["lazy"])
        self.assertEqual(self.node.check_lazy_status(enabled=False), [])
        self.assertEqual(self.node.check_lazy_status(enabled=True), ["backend"])
        self.assertEqual(self.node.check_lazy_status(enabled=True, backend=FakeBackend()), [])
        self.assertEqual(len([key for key in self.node.INPUT_TYPES()["optional"] if key.startswith("reference_image_")]), 16)

    def test_unload_on_backend_configuration_and_preview_errors(self):
        backend = FakeBackend(RuntimeError("模拟推理失败"))
        with self.assertRaisesRegex(RuntimeError, "模拟推理失败"):
            self.run_t2i(backend)
        self.assertEqual((len(backend.calls), backend.unloads), (1, 1))
        backend = FakeBackend()
        with patch.object(backend, "configure_chat", side_effect=RuntimeError("模拟配置失败")):
            with self.assertRaisesRegex(RuntimeError, "模拟配置失败"):
                self.run_t2i(backend)
        self.assertEqual(backend.unloads, 1)
        backend = FakeBackend()
        with patch.object(nodes, "tensor_frame_to_pil", side_effect=ValueError("模拟预览失败")):
            with self.assertRaisesRegex(ValueError, "模拟预览失败"):
                self.run_edit(backend)
        self.assertEqual(backend.unloads, 1)

    def test_unload_can_be_disabled(self):
        backend = FakeBackend()
        self.run_t2i(backend, unload_after=False)
        self.assertEqual(backend.unloads, 0)

    def test_invalid_template_wrong_media_and_empty_request_stop_before_inference(self):
        cases = [dict(template="[[tank:h3]]"), dict(template=T2I, reference_image_1=FakeImage(100, 100)),
                 dict(template=EDIT), dict(template=T2I, request=" "),
                 dict(template=EDIT, reference_image_1=FakeImage(100, 100, batch=0))]
        for case in cases:
            with self.subTest(case=case):
                backend = FakeBackend()
                args = dict(template=T2I, request="请求", backend=backend)
                args.update(case)
                with self.assertRaises(ValueError):
                    self.node.run(**args)
                self.assertEqual((backend.calls, backend.unloads), ([], 1))

    def test_budget_and_thinking_never_silently_decrease(self):
        for kwargs in (dict(thinking_mode="medium"), dict(context_length=8192), dict(max_tokens=2048),
                       dict(context_length=32768, max_tokens=32768)):
            with self.subTest(kwargs=kwargs):
                backend = FakeBackend()
                with self.assertRaises(ValueError):
                    self.run_t2i(backend, **kwargs)
                self.assertEqual(backend.calls, [])
        backend = FakeBackend()
        self.run_t2i(backend, context_length=98304, max_tokens=24576)
        self.assertEqual(backend.configurations, [(98304, "high")])
        self.assertEqual(backend.calls[0]["max_tokens"], 24576)

    def test_explicit_compact_t2i_budget_is_allowed(self):
        backend = FakeBackend()
        self.run_t2i(backend, context_length=16384, max_tokens=4096)
        self.assertEqual(backend.configurations, [(16384, "high")])
        self.assertEqual(backend.calls[0]["max_tokens"], 4096)

    def test_edit_still_requires_larger_original_budget(self):
        backend = FakeBackend(answer(edit=True))
        with self.assertRaises(ValueError):
            self.node.run(template=EDIT, request="edit", backend=backend,
                          context_length=16384, max_tokens=4096, reference_image_1=FakeImage(100, 100))
        self.assertEqual(backend.calls, [])

    def test_video_node_and_namespace_are_independent(self):
        self.assertEqual(video_module.MODES, ("T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"))
        self.assertEqual(video_module.TankH3PromptOptimizer.INPUT_TYPES()["required"]["thinking_mode"][1]["default"], "medium")
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), {"TankH3ImagePromptOptimizer", "TankH3ImageSize", "TankH3ImageAspect"})
        self.assertNotIn("TankH3PromptOptimizer", nodes.NODE_CLASS_MAPPINGS)


class SizeTests(unittest.TestCase):
    def setUp(self):
        self.node = nodes.TankH3ImageSize()

    def assert_size(self, result, mp):
        width, height, ratio = result
        self.assertEqual((width % 32, height % 32), (0, 0))
        self.assertEqual(ratio, nodes._reduced_ratio(width, height))
        self.assertLess(abs(width * height / (mp * 1024**2) - 1), 0.03)

    def test_landscape_portrait_and_megapixel_convention(self):
        for mp in (4, 20):
            with self.subTest(mp=mp):
                landscape = self.node.run("16:9", mp)
                portrait = self.node.run("9:16", mp)
                self.assertGreater(landscape[0], landscape[1])
                self.assertGreater(portrait[1], portrait[0])
                self.assertEqual(landscape[:2], tuple(reversed(portrait[:2])))
                self.assert_size(landscape, mp)
                self.assert_size(portrait, mp)
        self.assertEqual(self.node.run("1:1", 4), (2048, 2048, "1:1"))

    def test_explicit_ratio_ignores_pe_and_image(self):
        self.assertEqual(self.node.run("9:16", 4, recommended_ratio="无效建议", image=object()),
                         self.node.run("9:16", 4))

    def test_follow_image_one_ignores_pe(self):
        result = self.node.run("跟随图像1", 4, recommended_ratio="16:9", image=FakeImage(300, 900))
        self.assertGreater(result[1], result[0])
        self.assert_size(result, 4)
        with self.assertRaisesRegex(ValueError, "必须连接"):
            self.node.run("跟随图像1", 4, recommended_ratio="1:1")

    def test_auto_precedence_pe_then_image_then_square(self):
        image = FakeImage(300, 900)
        self.assertEqual(self.node.run("auto", 4, recommended_ratio="16:9", image=image), self.node.run("16:9", 4))
        self.assertEqual(self.node.run("auto", 4, image=image), self.node.run("1:3", 4))
        self.assertEqual(self.node.run("auto", 4), (2048, 2048, "1:1"))

    def test_shared_string_selection_feeds_both_nodes_without_combo_mismatch(self):
        optimizer = nodes.TankH3ImagePromptOptimizer()
        self.assertEqual(optimizer.INPUT_TYPES()["optional"]["aspect_ratio"][0], "STRING")
        self.assertEqual(self.node.INPUT_TYPES()["required"]["aspect_ratio"][0], "STRING")
        self.assertEqual(nodes.TankH3ImageAspect.RETURN_TYPES, ("STRING",))
        for choice in ("auto", "跟随图像1", "9:16"):
            with self.subTest(choice=choice):
                ratio = nodes.TankH3ImageAspect().run(choice)[0]
                image = FakeImage(900, 600)
                _, recommendation, _ = optimizer.run(template="", request="原请求", enabled=False,
                                                     aspect_ratio=ratio, reference_image_1=image)["result"]
                result = self.node.run(ratio, 4, recommended_ratio=recommendation, image=image)
                self.assertEqual(result, self.node.run("9:16" if choice == "9:16" else "3:2", 4))

    def test_invalid_area_and_ratio_report_errors(self):
        for area in (0, 3.99, 20.01, float("inf"), float("nan")):
            with self.subTest(area=area), self.assertRaises(ValueError):
                self.node.run("1:1", area)
        for ratio in ("adaptive", "0:1", "2/3", "1: 1", "1:100000000000000000000"):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.node.run("auto", 4, recommended_ratio=ratio)


if __name__ == "__main__":
    unittest.main(verbosity=2)
