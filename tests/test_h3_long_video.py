"""H3 planning tests need neither Torch nor an installed ComfyUI runtime."""

import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

if "tanknodes_test_package" in sys.modules:
    from tanknodes_test_package import h3_long_video as h3
else:
    ROOT_DIR = str(Path(__file__).resolve().parents[1])
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    import h3_long_video as h3


ROOT = Path(h3.__file__).resolve().parent
# Fixed GPU-reference results keep these tests portable to cloud checkouts.
CASES = [
    (736, 736, 56, [0, 170, 340, 510]),
    (1440, 1450, 39, [0, 187, 374, 561, 748, 935, 1122, 1224]),
    (498, 498, 56, [0, 170, 272]),
    (2160, 2164, 56, [0, 170, 340, 510, 680, 850, 1020, 1190, 1360, 1530, 1700, 1870, 1938]),
    (4320, 4323, 39, [0, 187, 374, 561, 748, 935, 1122, 1309, 1496, 1683, 1870, 2057, 2244,
                      2431, 2618, 2805, 2992, 3179, 3366, 3553, 3740, 3927, 4097]),
    (100, 107, 56, [0]),
]


@pytest.mark.parametrize("frames,total,overlap,starts", CASES)
def test_window_plan_matches_reference_results(frames, total, overlap, starts):
    result = h3.TankH3WindowPlan().plan(frames, 226, "56,39,73")
    assert result[:3] == (total, 226, overlap)
    assert json.loads(result[3]) == starts
    assert h3.fit_total(frames) == (total, overlap)
    assert h3.plan_windows(total, 226, overlap) == starts
    assert json.loads(result[4]) == {
        "total_frames": total,
        "context_length": 226,
        "context_overlap": overlap,
        "window_starts": starts,
        "windows": len(starts),
        "frames_needed": frames,
        "pad_frames": total - frames,
    }
    assert total >= frames
    assert (total - 5) % 17 == 0
    assert h3.lat_t(total) == (total - 5) // 17 * 5 + 2
    if total <= 226:
        assert total == h3.pad17(frames)
    else:
        # Independently verify the center-ratio condition selector used upstream.
        latent_starts = [s // 17 * 5 for s in starts]
        count, time, length = len(starts), h3.lat_t(total), h3.lat_t(226)
        regions = [min(max(int(((2 * s + length - 1) / (2 * time)) * count), 0), count - 1)
                   for s in latent_starts]
        assert regions == list(range(count))
        assert starts[-1] + 226 == total


def test_planning_import_does_not_require_torch():
    tree = ast.parse((ROOT / "h3_long_video.py").read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("torch" not in ast.unparse(node) and "comfy" not in ast.unparse(node) for node in imports)


def test_planning_matches_local_reference_when_available():
    path = ROOT.parent / "experiments/h3long/h3lab.py"
    if not path.exists():
        pytest.skip("Optional read-only h3lab reference is absent from this checkout")
    # Do not import the experiment's CLI or mutate its sys.path.
    wanted = {"lat_t", "px_start", "plan_windows", "fit_total", "pad17"}
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    functions = ast.Module(body=[node for node in parsed.body
                                if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
    reference = {}
    exec(compile(functions, str(path), "exec"), reference)
    for frames, total, overlap, starts in CASES:
        assert h3.pad17(frames) == reference["pad17"](frames)
        assert h3.lat_t(total) == reference["lat_t"](total)
        assert h3.fit_total(frames) == reference["fit_total"](frames)
        assert starts == reference["plan_windows"](total, 226, overlap)
        assert [h3.px_start(s // 17 * 5) for s in starts] == starts


@pytest.mark.parametrize("frames", [5, 6, 22, 100, 226, 227, 736, 1440])
def test_pad17_and_lat_t_grid(frames):
    padded = h3.pad17(frames)
    assert padded >= frames
    assert padded - frames < 17
    assert (padded - 5) % 17 == 0
    assert h3.lat_t(padded) == (padded - 5) // 17 * 5 + 2


@pytest.mark.parametrize("frames", [0, 4, 6, 100])
def test_invalid_latent_frame_grid(frames):
    with pytest.raises(AssertionError):
        h3.lat_t(frames)


def test_invalid_window_regions_are_rejected():
    with pytest.raises(AssertionError, match="cannot match|\u4e00\u4e00\u5bf9\u5e94"):
        h3.plan_windows(1450, 226, 56)
    with pytest.raises(ValueError, match="\u627e\u4e0d\u5230"):
        h3.fit_total(736, overlaps=(1,))


@pytest.mark.parametrize("candidates", ["", "bad", "1"])
def test_bad_overlap_candidates_raise_value_error(candidates):
    with pytest.raises(ValueError):
        h3.TankH3WindowPlan().plan(1440, 226, candidates)


def test_locales_and_registration_contract():
    source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "from .h3_long_video import" in source
    assert "NODE_CLASS_MAPPINGS.update(_H3_LONG_VIDEO_NODES)" in source
    assert "NODE_DISPLAY_NAME_MAPPINGS.update(_H3_LONG_VIDEO_NAMES)" in source
    for lang in ("en", "zh"):
        definitions = json.loads((ROOT / f"locales/{lang}/nodeDefs.json").read_text(encoding="utf-8"))
        for name, cls in h3.NODE_CLASS_MAPPINGS.items():
            entry, schema = definitions[name], cls.INPUT_TYPES()
            assert cls.CATEGORY == "TankNodes/H3 Long Video"
            assert entry["display_name"].endswith("\u00b7 Tank")
            assert entry["description"]
            assert set(entry["inputs"]) == set(schema.get("required", {})) | set(schema.get("optional", {}))
            assert set(entry["outputs"]) == {str(i) for i in range(len(cls.RETURN_TYPES))}
            for port in (*entry["inputs"].values(), *entry["outputs"].values()):
                assert port["name"] and port["tooltip"]
            if lang == "en":
                assert entry["display_name"] == h3.NODE_DISPLAY_NAME_MAPPINGS[name]


def _fake_official_nodes(monkeypatch):
    extras = ModuleType("comfy_extras")
    official = ModuleType("comfy_extras.nodes_minimax_h3")
    calls, empty_calls = [], []

    class Reference:
        @classmethod
        def execute(cls, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(result=([[kwargs["prompt"], {"window": len(calls)}]], None))

    class Empty:
        @classmethod
        def execute(cls, **kwargs):
            empty_calls.append(kwargs)
            return SimpleNamespace(result=({"samples": "blank-av"},))

    official.MiniMaxH3ReferenceToVideo = Reference
    official.EmptyMiniMaxH3LatentAV = Empty
    extras.nodes_minimax_h3 = official
    monkeypatch.setitem(sys.modules, "comfy_extras", extras)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", official)
    return calls, empty_calls


def _windowed_kwargs(**extra):
    return dict(clip=object(), vae=object(), audio_vae=object(), width=32, height=32,
                frames_needed=736, context_length=226, prompt="shared", ref_image_size="match", **extra)


def test_windowed_conditions_are_combined_in_order(monkeypatch):
    calls, empty_calls = _fake_official_nodes(monkeypatch)
    image_1, image_4 = object(), object()
    prompts = ["window 0", "window 1", "window 2", "window 3"]
    result = h3.TankH3WindowedReferenceToVideo().execute(**_windowed_kwargs(
        image_1=image_1, image_4=image_4, window_prompts=json.dumps(prompts)))
    assert [part[0] for part in result[0]] == prompts
    assert len(calls) == 4
    for call in calls:
        assert call["length"] == 226
        assert call["ref_images"] == {"ref_image_0": image_1, "ref_image_1": image_4}
        assert call["ref_videos"] == call["ref_video_audios"] == call["ref_audios"] == {}
    assert empty_calls == [{"width": 32, "height": 32, "length": 736}]
    assert result[1:4] == ({"samples": "blank-av"}, 736, 56)


def test_windowed_batched_refs_expand_and_cap_at_nine(monkeypatch):
    torch = pytest.importorskip("torch")
    calls, _ = _fake_official_nodes(monkeypatch)
    batch = torch.arange(3 * 8 * 8 * 3, dtype=torch.float32).reshape(3, 8, 8, 3)
    single = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    h3.TankH3WindowedReferenceToVideo().execute(**_windowed_kwargs(split_conds=False, image_1=single, image_9=batch))
    refs = calls[0]["ref_images"]
    assert list(refs) == ["ref_image_0", "ref_image_1", "ref_image_2", "ref_image_3"]
    assert torch.equal(refs["ref_image_0"], single)
    for idx in range(3):
        assert torch.equal(refs[f"ref_image_{idx + 1}"], batch[idx:idx + 1])

    too_many = torch.zeros((9, 8, 8, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="最多 9 张"):
        h3.TankH3WindowedReferenceToVideo().execute(**_windowed_kwargs(split_conds=False, image_1=single, image_2=too_many))


def test_windowed_shared_condition_for_all_windows(monkeypatch):
    calls, _ = _fake_official_nodes(monkeypatch)
    result = h3.TankH3WindowedReferenceToVideo().execute(**_windowed_kwargs(split_conds=False))
    assert len(calls) == 1 and calls[0]["prompt"] == "shared"
    assert result[2] == 736
    plan = json.loads(result[4])
    assert plan["window_starts"] == [0] and plan["windows"] == 1
    assert plan["split_conds"] is False


def test_windowed_shared_condition_rejects_long_source_video(monkeypatch):
    torch = pytest.importorskip("torch")
    calls, empty_calls = _fake_official_nodes(monkeypatch)
    long_source = torch.zeros((240, 8, 8, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="split_conds"):
        h3.TankH3WindowedReferenceToVideo().execute(
            **{**_windowed_kwargs(split_conds=False, source_video=long_source), "frames_needed": 736}
        )
    assert calls == empty_calls == []

    short_source = torch.zeros((107, 8, 8, 3), dtype=torch.float32)
    h3.TankH3WindowedReferenceToVideo().execute(
        **{**_windowed_kwargs(split_conds=False, source_video=short_source), "frames_needed": 100}
    )
    assert len(calls) == 1 and calls[0]["length"] == 107


@pytest.mark.parametrize("prompts", ['not-json', '{}', '["one"]', '[1, 2, 3, 4]'])
def test_windowed_prompt_validation_happens_before_encoding(monkeypatch, prompts):
    calls, empty_calls = _fake_official_nodes(monkeypatch)
    with pytest.raises(ValueError, match="window_prompts"):
        h3.TankH3WindowedReferenceToVideo().execute(**_windowed_kwargs(window_prompts=prompts))
    assert calls == empty_calls == []


def _install_fake_comfy(monkeypatch, torch):
    comfy = ModuleType("comfy")
    audio = ModuleType("comfy.audio")
    audio.resample = lambda wav, src, dst: torch.nn.functional.interpolate(
        wav[None], size=round(wav.shape[-1] * dst / src), mode="linear", align_corners=False)[0]
    nested = ModuleType("comfy.nested_tensor")

    class NestedTensor:
        def __init__(self, tensors):
            self.tensors = tuple(tensors)

        def unbind(self):
            return self.tensors

    nested.NestedTensor = NestedTensor
    ldm, minimax, model = (ModuleType(name) for name in ("comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.model"))
    model.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    comfy.audio, comfy.nested_tensor, comfy.ldm = audio, nested, ldm
    ldm.minimax, minimax.model = minimax, model
    for module in (comfy, audio, nested, ldm, minimax, model):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return NestedTensor


@pytest.fixture
def tensor_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    nested = _install_fake_comfy(monkeypatch, torch)

    class VideoVAE:
        def encode(self, images):
            nf, height, width = images.shape[:3]
            return torch.zeros((1, 24, h3.lat_t(nf), height // 16, width // 16))

    class AudioVAE:
        def __init__(self):
            self.inputs = []

        def encode(self, audio):
            self.inputs.append(audio.clone())
            return torch.ones((1, 32, 2, audio.shape[1] // 800))

    return SimpleNamespace(torch=torch, nested=nested, vae=VideoVAE(), audio_vae=AudioVAE())


def _masked_inputs(runtime):
    torch = runtime.torch
    images = torch.zeros((22, 64, 64, 3))
    mask = torch.zeros((22, 64, 64))
    mask[:, 20:28, 20:28] = 1
    return dict(vae=runtime.vae, audio_vae=runtime.audio_vae, images=images, mask=mask, dilate=0, invert=False)


def test_masked_latent_shape_bbox_and_protect(tensor_runtime):
    runtime, node = tensor_runtime, h3.TankH3MaskedSourceLatent()
    torch = runtime.torch
    inputs = _masked_inputs(runtime)
    base, preview = node.run(**inputs)
    expanded, expanded_preview = node.run(**inputs, bbox_expand=1.0)
    video, audio = expanded["samples"].unbind()
    vmask, amask = expanded["noise_mask"].unbind()
    assert video.shape == vmask.shape == (1, 24, 7, 4, 4)
    assert audio.shape == amask.shape == (1, 32, 2, round(22 / 24 * 40))
    assert torch.count_nonzero(amask) == 0
    assert expanded_preview.sum() > preview.sum()
    assert vmask.sum() > base["noise_mask"].unbind()[0].sum()
    protect = torch.zeros((1, 64, 64))
    protect[:, 16:32, 16:32] = 1
    protected, protected_preview = node.run(**inputs, bbox_expand=1.0, protect=protect)
    assert protected_preview[:, 16:32, 16:32].sum() == 0
    assert protected_preview.sum() > 0
    assert protected["noise_mask"].unbind()[0][:, :, :, 1, 1].sum() == 0


def test_masked_temporal_pooling_matches_h3_spans(tensor_runtime):
    runtime, node = tensor_runtime, h3.TankH3MaskedSourceLatent()
    inputs = _masked_inputs(runtime)
    mask = runtime.torch.zeros_like(inputs["mask"])
    mask[5, 0:16, 0:16] = 1
    latent, preview = node.run(**{**inputs, "mask": mask}, temporal_dilate=1)
    assert preview[:, 0, 0].tolist() == [0.0] * 4 + [1.0] * 3 + [0.0] * 15
    # H3 pixel spans are [0,1), [1,5), [5,9), [9,13), [13,17), [17,18), [18,22).
    noise = latent["noise_mask"].unbind()[0]
    assert noise[0, 0, :, 0, 0].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert noise.shape[2] == latent["samples"].unbind()[0].shape[2] == 7


def test_masked_invert_and_pixel_dilate(tensor_runtime):
    runtime, node = tensor_runtime, h3.TankH3MaskedSourceLatent()
    inputs = _masked_inputs(runtime)
    _, preview = node.run(**inputs)
    _, inverted = node.run(**{**inputs, "invert": True})
    assert runtime.torch.equal(inverted, 1 - preview)
    _, dilated = node.run(**{**inputs, "dilate": 2})
    assert dilated.sum() > preview.sum()


@pytest.mark.parametrize("sr,channels", [(32000, 1), (16000, 2)])
def test_masked_audio_lock_and_preprocessing(tensor_runtime, sr, channels):
    runtime = tensor_runtime
    inputs = _masked_inputs(runtime)
    audio = {"waveform": runtime.torch.ones((1, channels, sr)), "sample_rate": sr}
    latent, _ = h3.TankH3MaskedSourceLatent().run(**inputs, audio=audio)
    encoded = runtime.audio_vae.inputs[0]
    n = round(23 / 24 * 32000)
    content_samples = round(round(22 / 24 * sr) * 32000 / sr)
    assert encoded.shape == (1, n + (-n % 800), 2)
    assert runtime.torch.all(encoded[:, :content_samples] == 1)
    assert runtime.torch.count_nonzero(encoded[:, content_samples:]) == 0
    assert runtime.torch.count_nonzero(latent["noise_mask"].unbind()[1]) == 0


def test_windowed_source_slices_padding_and_locked_audio(monkeypatch, tensor_runtime):
    runtime = tensor_runtime
    calls, _ = _fake_official_nodes(monkeypatch)
    official = sys.modules["comfy_extras.nodes_minimax_h3"]
    blank_video = runtime.torch.zeros((1, 24, h3.lat_t(736), 2, 2))
    official.EmptyMiniMaxH3LatentAV.execute = lambda **kwargs: SimpleNamespace(
        result=({"samples": runtime.nested((blank_video, runtime.torch.zeros(1)))},))
    source = runtime.torch.arange(735).view(735, 1, 1, 1).expand(735, 32, 32, 3).float()
    audio = {"waveform": runtime.torch.ones((1, 1, 32000)), "sample_rate": 32000}
    result = h3.TankH3WindowedReferenceToVideo().execute(
        **{**_windowed_kwargs(source_video=source, audio=audio), "audio_vae": runtime.audio_vae})
    for call, start in zip(calls, [0, 170, 340, 510]):
        assert call["ref_videos"]["ref_video_0"].shape[0] == 226
        assert call["ref_videos"]["ref_video_0"][0, 0, 0, 0] == start
    assert calls[-1]["ref_videos"]["ref_video_0"][-1, 0, 0, 0] == 734
    vmask, amask = result[1]["noise_mask"].unbind()
    assert runtime.torch.all(vmask == 1) and vmask.shape == blank_video.shape
    assert runtime.torch.count_nonzero(amask) == 0
    assert amask.shape[-1] == round(736 / 24 * 40)
    assert h3.TankH3WindowedReferenceToVideo._pad_source(source, 22).shape[0] == 22
    with pytest.raises(ValueError):
        h3.TankH3WindowedReferenceToVideo._pad_source(source[:0], 22)


def test_api_example_links_and_window_parameters():
    graph = json.loads((ROOT / "docs/h3_long_video_swap_api.json").read_text(encoding="utf-8"))
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert len(value) == 2 and value[0] in graph and isinstance(value[1], int)
    windowed_id, windowed = next((key, node) for key, node in graph.items()
                                if node["class_type"] == "TankH3WindowedReferenceToVideo")
    context = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3ContextWindows")
    assert context["inputs"]["context_overlap"] == [windowed_id, 3]
    assert context["inputs"]["context_length"] == windowed["inputs"]["context_length"]
    assert context["inputs"]["split_conds_to_windows"] is True
