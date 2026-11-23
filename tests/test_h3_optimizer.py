"""Offline checks for the H3 prompt optimizer (no model required)."""
import json
import pytest


def _job(m, request="", duration=5.0, assets=(), mode="auto"):
    return m.Job(request, duration, list(assets), mode)


def test_mode_resolution(module):
    m = module("h3_optimizer")
    pic = lambda role: m.Asset("picture", role, "", ["data:image/jpeg;base64,AA"])
    assert m.resolve_mode(_job(m)) == "T2VA"
    assert m.resolve_mode(_job(m, assets=[pic("first_frame")])) == "I2VA"
    assert m.resolve_mode(_job(m, assets=[pic("last_frame")])) == "L2VA"
    assert m.resolve_mode(_job(m, assets=[pic("first_frame"), pic("last_frame")])) == "FL2VA"
    assert m.resolve_mode(_job(m, assets=[pic("reference")])) == "Ref2VA"
    assert m.resolve_mode(_job(m, assets=[m.Asset("video", "editing", "", ["x"])])) == "Ref2VA"
    assert m.resolve_mode(_job(m, assets=[pic("reference")], mode="T2VA")) == "T2VA"


def test_official_instruction_lines(module):
    m = module("h3_optimizer")
    assert m.instruction_line("T2VA", 5, 1) == ""
    assert m.instruction_line("I2VA", 6, 3) == ("For the target video, at 0.00 seconds into the target video, <Picture 1> "
                                                "(from [Shot 1]) is fully referenced.")
    assert m.instruction_line("FL2VA", 8, 1).endswith("Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.")
    assert m.instruction_line("L2VA", 6, 2) == ("How the reference pictures align with the target video — <Picture 1> "
                                                "(from [Shot 2]) aligns with the 6.00-second mark of the target video.")


def test_repair_rebuilds_instruction_and_timecodes(module):
    m = module("h3_optimizer")
    job = _job(m, duration=6.0, assets=[m.Asset("picture", "last_frame")])
    draft = ("Sure! Here is the prompt:\nintegrated_multimodal_description: [Shot 1] A glass. [Shot 2] At 3.5s, the camera cuts "
             "to the floor.\n\noverall_soundscape: Glass breaks.\n\nnon_diegetic_music: N/A")
    fixed = m.repair(draft, "L2VA", job)
    assert fixed.splitlines()[0] == m.instruction_line("L2VA", 6.0, 2)
    assert "[Shot 2] At 00:03.500," in fixed
    assert not [i for i in m.validate(fixed, "L2VA", job) if i["hard"]]


def test_fl2va_uses_plain_picture_wording(module):
    m = module("h3_optimizer")
    job = _job(m, duration=5.0, assets=[m.Asset("picture", "first_frame"), m.Asset("picture", "last_frame")])
    fixed = m.repair("integrated_multimodal_description: [Shot 1] Starts on <Picture 1> and ends on <Picture 2>.\n\n"
                     "overall_soundscape: Room tone.\n\nnon_diegetic_music: N/A", "FL2VA", job)
    assert "<Picture" not in fixed and "ends on Picture 2" in fixed


def test_validator_catches_contract_breaks(module):
    m = module("h3_optimizer")
    job = _job(m, request="她说：“你自己看吧。”", duration=5.0)
    bad = ("integrated_multimodal_description: [Shot 1] 一个女人站在雨里 holding a letter. [Shot 2] At 00:06.000, cut. "
           "<Picture 1> is shown. A sign reads \"顾家大院\".\n\noverall_soundscape: Rain.\n\nnon_diegetic_music: N/A")
    codes = {i["code"] for i in m.validate(bad, "T2VA", job) if i["hard"]}
    assert {"shot_time", "label", "dialogue", "onscreen_text"} <= codes


def test_ref2va_requires_definitions_and_summary_prefix(module):
    m = module("h3_optimizer")
    job = _job(m, duration=10.0, assets=[m.Asset("picture", "reference"), m.Asset("video", "editing")])
    text = ("subject_definitions:\n<Subject 1> is the man in <Picture 1>.\n\nsummary:\nThe man walks.\n\nretention_analysis:\n"
            "<Subject 1> (appears in [Shot 1]): fully_preserved - kept.\n\ndetailed_description:\nLive-action.\n[Shot 1] <Subject 2> "
            "walks.\n\noverall_soundscape:\nWind.\n\nnon_diegetic_music:\nN/A")
    codes = {i["code"] for i in m.validate(text, "Ref2VA", job) if i["hard"]}
    assert {"summary", "label"} <= codes


def test_messages_carry_guide_labels_and_media(module):
    m = module("h3_optimizer")
    job = _job(m, request="男主走进院子", duration=10.0,
               assets=[m.Asset("picture", "reference", "男主", ["data:image/jpeg;base64,AA"]),
                       m.Asset("video", "editing", "", ["data:image/jpeg;base64,BB", "data:image/jpeg;base64,CC"])])
    messages = m.build_messages(job, "Ref2VA")
    system = messages[0]["content"]
    assert "Full-Reference Mode Rewrite Output Format Guide" in system and "camera" in system.lower()
    assert "[video editing + reference generation]" in system and "<Video 1>" in system
    images = [p for p in messages[1]["content"] if p["type"] == "image_url"]
    assert len(images) == 3


def test_node_registered(plugin):
    assert "TankH3PromptOptimizer" in plugin.NODE_CLASS_MAPPINGS


# ---------------------------------------------------------------- video role / cuts / edit contract (1.x 修正)
def test_infer_video_role_from_request(module):
    m = module("h3_optimizer")
    assert m.infer_video_role("把视频里的主角替换成图1的角色", True)[0] == "editing"
    assert m.infer_video_role("Replace the dancer with the girl in picture 1", True)[0] == "editing"
    assert m.infer_video_role("让图1的女孩跟着视频跳舞", True)[0] == "motion"
    assert m.infer_video_role("参考视频的运镜拍一段海边", True)[0] == "motion"
    assert m.infer_video_role("接着这个视频续拍 5 秒", False)[0] == "continuation"
    assert m.infer_video_role("一个女孩在跳舞", True)[0] == "motion"      # 没关键词：有参考图兜底为动作参考
    assert m.infer_video_role("一个女孩在跳舞", False)[0] == "editing"


def _shots_video(torch, lengths, levels, h=32, w=48):
    parts = [torch.full((n, h, w, 3), v) + torch.rand(n, h, w, 3) * 0.01 for n, v in zip(lengths, levels)]
    return torch.cat(parts).clamp(0, 1)


def test_detect_cuts_and_shot_sampling(module):
    import torch
    m = module("h3_optimizer")
    video = _shots_video(torch, [30, 24, 42], [0.1, 0.7, 0.35])
    assert m.detect_cuts(video, 24.0) == [1.25, 2.25]
    assert m.detect_cuts(_shots_video(torch, [60], [0.5]), 24.0) == []
    asset, cuts, duration = m.video_asset(video, "editing", 24.0, limit=6)
    assert cuts == [1.25, 2.25] and abs(duration - 4.0) < 1e-6
    assert len(asset.data_urls) == 6
    assert {label.split(",")[0] for label in asset.frame_labels} == {"shot 1", "shot 2", "shot 3"}


def test_edit_contract_cuts_audio_and_retention(module):
    import torch
    m = module("h3_optimizer")
    video = _shots_video(torch, [30, 24, 42], [0.1, 0.7, 0.35])
    pic = m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"])
    assets, cuts, duration, lines = m.source_video_assets(video, "auto", "把主角换成图1的角色", True, 24.0, True)
    assert [a.role for a in assets] == ["editing", "source_audio"] and cuts == [1.25, 2.25]
    assert lines[0].startswith("video_role=editing")
    job = m.Job("把主角换成图1的角色", duration, [pic, *assets], "auto", cuts=cuts)
    system = m.build_messages(job, "Ref2VA")[0]["content"]
    assert "[Shot 2] At 00:01.250" in system and "[Shot 3] At 00:02.250" in system
    assert "audio reuse" in system and "fully_copy" in system
    assert "Normally 150-350 English words" in system and "normally has 350-500 English words" not in system
    good = ("subject_definitions:\n<Subject 1> is the girl from <Picture 1>.\n"
            "summary:\n[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>.\n"
            "retention_analysis:\n<Subject 1>: fully_preserved - x\n<Video 1> (shots): partially_preserved - y\n"
            "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.\n"
            "detailed_description:\nStyle. [Shot 1] a. [Shot 2] At 00:01.300, b. [Shot 3] At 00:02.100, c.\n"
            "overall_soundscape:\nThe copied soundtrack continues.\nnon_diegetic_music:\nN/A")
    fixed = m.repair(good, "Ref2VA", job)
    assert "[Shot 2] At 00:01.250" in fixed and "[Shot 3] At 00:02.250" in fixed
    codes = {i["code"] for i in m.validate(fixed, "Ref2VA", job) if i["hard"]}
    assert not codes & {"cuts", "retention"}
    two_shots = good.replace(" [Shot 3] At 00:02.100, c.", "")
    assert "cuts" in {i["code"] for i in m.validate(m.repair(two_shots, "Ref2VA", job), "Ref2VA", job)}
    no_audio = fixed.replace("<Audio 1>: fully_copy", "<Audio 1>: reference")
    assert "retention" in {i["code"] for i in m.validate(no_audio, "Ref2VA", job)}


def test_motion_reference_keeps_old_contract(module):
    import torch
    m = module("h3_optimizer")
    video = _shots_video(torch, [30, 24], [0.1, 0.7])
    assets, cuts, _d, _l = m.source_video_assets(video, "auto", "让图1的女孩跟着视频跳", True, 24.0, True)
    assert [a.role for a in assets] == ["motion"] and cuts is None

# ---------------------------------------------------------------- generic video-edit preset (two-pass reverse analysis)
def _edit_analysis_json(multiple=False, motion=False):
    edits = [{"kind": "character_swap", "target": "the gray wolf in a long beige trench coat",
              "source": {"picture": 1, "text": ""},
              "becomes": "the girl with indigo-to-blue hair in a navy maid dress",
              "keep_from_target": "position, route, scale, pose, movement, expression, gaze and held props"}]
    if multiple:
        edits.append({"kind": "character_swap", "target": "the white sheep in a yellow raincoat",
                      "source": {"picture": 2, "text": ""}, "becomes": "the red-haired boy in a green jacket",
                      "keep_from_target": "position, route, scale, pose, movement, expression, gaze and held props"})
    return {"edit_type": "motion_transfer" if motion else ("mixed" if multiple else "character_swap"),
            "edits": edits, "unchanged": ["the gray round creature", "the white sheep" if not multiple else "the red bus"],
            "style": "flat 2D cartoon", "notes": ""}


def _edit_backend_result(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 12}}


class _SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return _edit_backend_result(self.responses.pop(0))


def _edit_llm_prompt():
    return ("subject_definitions:\n<Subject 1>: the girl with indigo-to-blue hair in a navy maid dress from <Picture 1>.\n"
            "<Video 1>: the source video being edited, 3 shots with hard cuts.\n"
            "<Audio 1>: the original soundtrack of <Video 1>.\n\n"
            "summary:\n[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. "
            "The gray wolf in a long beige trench coat is replaced by <Subject 1> (the girl with indigo-to-blue hair in a navy maid dress), "
            "who takes over that character's position, route, scale, pose, every movement, expression, gaze and the props in hand. "
            "Everything else — the gray round creature, the white sheep, the backgrounds, the camera, every shot and cut, all motion and timing — "
            "stays exactly as in <Video 1>.\n\n"
            "retention_analysis:\n<Video 1>: partially_preserved - only the gray wolf changes; shots, cuts, camera and timing are retained.\n"
            "<Subject 1>: fully_preserved - the identity from <Picture 1> stays identical in every frame and every shot.\n"
            "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.\n\n"
            "detailed_description:\nThe video keeps the flat 2D cartoon of <Video 1>.\n"
            "[Shot 1] Throughout <Video 1>, in every shot and at every moment where the gray wolf in a long beige trench coat appears, "
            "the gray wolf is replaced by <Subject 1> (the girl with indigo-to-blue hair in a navy maid dress), who takes over that character's "
            "position, route, scale, pose, every movement, expression, gaze and the props in hand. The gray round creature and white sheep stay "
            "exactly as in <Video 1>. The camera, framing, cuts and the timing of every movement follow <Video 1> exactly.\n\n"
            "overall_soundscape:\n<Audio 1> continues unchanged as the full soundtrack; nothing is added or removed.\n\n"
            "non_diegetic_music:\nAny music is the original music inside <Audio 1>, unchanged.")


def _edit_job(m, analysis=None, cuts=None):
    assets = [m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]),
              m.Asset("video", "editing", "", ["data:image/jpeg;base64,BB"], ["shot 1 (0.0-4.0 s)"]),
              m.Asset("audio", "source_audio", "")]
    return m.Job("Replace the gray wolf with picture 1", 9.0, assets, "auto", cuts=cuts,
                 generic_edit=True, edit_analysis=analysis)


def test_shot_sheets_split_long_shots_and_merge(module):
    import torch
    m = module("h3_optimizer")
    video = _shots_video(torch, [1, 1, 9], [0.1, 0.7, 0.35], h=4, w=6)
    sheets = m.shot_sheets(video, 1.0, [1.0, 2.0], max_sheets=8, tiles=4, tile_w=32)
    assert len(sheets) == 5
    long_parts = [sheet for sheet in sheets if sheet["shot"] == 3]
    assert len(long_parts) == 3
    assert all(sheet["url"].startswith("data:image/jpeg;base64,") for sheet in sheets)
    assert [(round(s["start"], 1), round(s["end"], 1)) for s in long_parts] == [(2.0, 6.0), (6.0, 10.0), (10.0, 11.0)]
    assert len(m.shot_sheets(video, 1.0, [1.0, 2.0], max_sheets=3, tiles=4, tile_w=32)) <= 3


def test_edit_analysis_json_tolerates_wrappers_and_checks_picture_bounds(module):
    import json
    m = module("h3_optimizer")
    payload = _edit_analysis_json()
    parsed = m.parse_edit_analysis("preface```json\n" + json.dumps(payload) + "\n``` trailing note", 1)
    assert parsed["edits"][0]["target"].startswith("the gray wolf")
    payload["edits"][0]["source"]["picture"] = 2
    with pytest.raises(ValueError, match="exceeds 1"):
        m.parse_edit_analysis(json.dumps(payload), 1)


def test_render_edit_prompt_generic_subjects_motion_and_text_attribute(module):
    m = module("h3_optimizer")
    analysis = _edit_analysis_json(multiple=True)
    prompt = m.render_edit_prompt(analysis, duration=8, has_audio=True, n_shots=4)
    assert prompt.count("[Shot 1]") == 1 and "[Shot 2]" not in prompt
    assert "At 00:" not in prompt and "At 0:" not in prompt
    assert "<Subject 1>" in prompt and "<Picture 1>" in prompt
    assert "<Subject 2>" in prompt and "<Picture 2>" in prompt
    assert "fully_copy" in prompt
    motion = m.render_edit_prompt(_edit_analysis_json(motion=True), duration=5, has_audio=True)
    assert "[reference generation + audio reuse]" in motion
    text_only = {"edit_type": "attribute", "edits": [{"kind": "attribute", "target": "the coat",
                  "source": {"picture": None, "text": "bright red"}, "becomes": "bright red",
                  "keep_from_target": "shape, fabric and motion"}], "unchanged": [], "style": "live action", "notes": ""}
    attr = m.render_edit_prompt(text_only, duration=5, has_audio=False)
    assert "<Subject" not in attr and "<Picture" not in attr


def test_validate_generic_edit_rejects_later_shots_and_timestamps_skips_cuts(module):
    m = module("h3_optimizer")
    analysis = _edit_analysis_json()
    job = _edit_job(m, analysis, cuts=[3.0])
    text = _edit_llm_prompt().replace("[Shot 1] Throughout", "[Shot 1] Throughout")
    text = text.replace("\n\noverall_soundscape:", " [Shot 2] At 00:03.000, a second shot.\n\noverall_soundscape:")
    issues = m.validate(text, "Ref2VA", job, generic_edit=True)
    codes = {issue["code"] for issue in issues if issue["hard"]}
    assert {"generic_shots", "generic_timestamp"} <= codes
    assert "cuts" not in codes


def test_optimize_edit_code_one_call_and_report_mapping(module):
    import json
    m = module("h3_optimizer")
    analysis = _edit_analysis_json()
    backend = _SequenceBackend([json.dumps(analysis)])
    job = _edit_job(m, cuts=[2.0, 5.0])
    result = m.optimize_edit(backend, job, render="code", max_tokens=8192, fallback_job=job)
    assert len(backend.calls) == 1
    assert result["prompt"] == m.render_edit_prompt(analysis, duration=9.0, has_audio=True, n_shots=3)
    assert "图1 → 替换:" in result["edit_report"]
    assert backend.calls[0]["max_tokens"] == 6144


def test_optimize_edit_llm_two_calls_and_invalid_analysis_falls_back(module):
    import json
    m = module("h3_optimizer")
    analysis = _edit_analysis_json()
    backend = _SequenceBackend([json.dumps(analysis), _edit_llm_prompt()])
    result = m.optimize_edit(backend, _edit_job(m), render="llm", handbook="manual")
    assert len(backend.calls) == 2
    assert result["render"] == "llm" and result["analysis"] == analysis

    fallback_backend = _SequenceBackend(["garbage", "still garbage"])
    source_job = _edit_job(m)
    classic_job = m.Job(source_job.request, source_job.duration, source_job.assets, "auto", cuts=None)
    fallback = m.optimize_edit(fallback_backend, source_job, render="code", fallback_job=classic_job)
    assert len(fallback_backend.calls) == 2
    assert "analysis=failed, generic fallback" in fallback["edit_report"]


def test_extra_rules_render_directive_overrides_template_default(module):
    m = module("prompt_nodes")
    template = ("[[tank:h3]]\n[[tank:h3-edit]]\n[[render:llm]]\nHandbook\n\n"
                "Additional rules from the workflow owner:\n[[render:code]]")
    render, handbook = m._h3_edit_template_options(template)
    assert render == "code"
    assert "[[tank:h3" not in handbook and "[[render:" not in handbook


def test_old_h3_template_keeps_single_pass_call_count(module):
    m = module("prompt_nodes")
    backend = _SequenceBackend(["integrated_multimodal_description: [Shot 1] A quiet courtyard.\n\n"
                               "overall_soundscape: Soft wind.\n\nnon_diegetic_music: N/A"])
    node = m.TankPromptOptimizer()
    prompt, _report = node._h3(backend, "A quiet courtyard", 5.0, None, None, [], None, "medium", 6000, 7,
                               "auto", {}, template="[[tank:h3]]\nOld official H3 template")
    assert "integrated_multimodal_description:" in prompt
    assert len(backend.calls) == 1


def test_style_source_directive_and_request_render_picture_style(module):
    import json
    m = module("h3_optimizer")
    base_analysis = _edit_analysis_json()

    # 1) Template directive [[style:picture]] forces picture style (last directive wins)
    backend_dir = _SequenceBackend([json.dumps(base_analysis)])
    res_dir = m.optimize_edit(
        backend_dir,
        _edit_job(m),
        render="code",
        template="[[style:video]]\n[[style:picture]]",
    )
    assert res_dir["analysis"]["style_source"] == "picture"
    assert "keeps the art style and rendering of its own picture" in res_dir["prompt"]
    assert "and its art style" in res_dir["prompt"]
    assert "画风: 保持参考图" in res_dir["edit_report"]

    # 2) Chinese request "保持图片画风" resolves to style_source=picture
    req_job = _edit_job(m)
    req_job.request = "把灰狼换成图1，保持图片画风"
    backend_req = _SequenceBackend([json.dumps(_edit_analysis_json())])
    res_req = m.optimize_edit(backend_req, req_job, render="code")
    assert res_req["analysis"]["style_source"] == "picture"
    assert "keeps the art style and rendering of its own picture" in res_req["prompt"]
    assert "画风: 保持参考图" in res_req["edit_report"]


def test_same_picture_character_and_background_allowed_but_duplicate_character_swap_rejected(module):
    import json
    m = module("h3_optimizer")
    dual_payload = {
        "edit_type": "mixed",
        "edits": [
            {
                "kind": "character_swap",
                "target": "the gray wolf in a long beige trench coat",
                "source": {"picture": 1, "text": ""},
                "becomes": "the girl in a navy maid dress",
                "keep_from_target": "position, route, pose, movement and held props",
            },
            {
                "kind": "background",
                "target": "the street background",
                "source": {"picture": 1, "text": ""},
                "becomes": "the sunlit café interior",
                "keep_from_target": "camera and framing",
            },
        ],
        "unchanged": ["the white sheep"],
        "style": "flat 2D cartoon",
        "notes": "",
    }
    parsed = m.parse_edit_analysis(json.dumps(dual_payload), 1)
    assert len(parsed["edits"]) == 2
    backend = _SequenceBackend([json.dumps(dual_payload)])
    res = m.optimize_edit(backend, _edit_job(m), render="code")
    assert not [i for i in res["issues"] if i["hard"]]
    assert "同时用于" in res["edit_report"]
    assert "人物+背景" in res["edit_report"]

    dup_char = {
        "edit_type": "mixed",
        "edits": [
            dict(dual_payload["edits"][0]),
            {
                "kind": "character_swap",
                "target": "the white sheep",
                "source": {"picture": 1, "text": ""},
                "becomes": "the girl in a navy maid dress",
                "keep_from_target": "position and motion",
            },
        ],
        "unchanged": [],
        "style": "flat 2D cartoon",
        "notes": "",
    }
    with pytest.raises(ValueError, match="multiple character_swap"):
        m.parse_edit_analysis(json.dumps(dup_char), 1)


def test_unused_extra_picture_omitted_from_prompt_and_reported(module):
    import json
    m = module("h3_optimizer")
    payload = _edit_analysis_json(multiple=True)
    payload["pictures"] = [
        {"picture": 1, "use": "used"},
        {"picture": 2, "use": "used"},
        {"picture": 3, "use": "unused"},
    ]
    assets = [
        m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]),
        m.Asset("picture", "reference", "", ["data:image/jpeg;base64,BB"]),
        m.Asset("picture", "reference", "", ["data:image/jpeg;base64,CC"]),
        m.Asset("video", "editing", "", ["data:image/jpeg;base64,DD"], ["shot 1 (0.0-4.0 s)"]),
        m.Asset("audio", "source_audio", ""),
    ]
    job = m.Job("把狼换成图1，羊换成图2", 8.0, assets, "auto", generic_edit=True)
    backend = _SequenceBackend([json.dumps(payload)])
    res = m.optimize_edit(backend, job, render="code")
    assert "<Picture 3>" not in res["prompt"]
    assert "<Subject 3>" not in res["prompt"]
    assert "图3" in res["edit_report"] and "未使用" in res["edit_report"]
    assert not [i for i in res["issues"] if i["hard"]]


def test_picstyle_preset_template_loads_and_applies_picture_style(module):
    import json
    import torch
    pn = module("prompt_nodes")
    label = "H3 视频编辑 · 通用模板（保持参考图画风）"
    labels = [item["label"] for item in pn._builtin_index()]
    assert labels[2] == label
    tpl = pn._template_text(label)
    assert tpl.lstrip().startswith("[[tank:h3]]")
    assert "[[tank:h3-edit]]" in tpl
    assert "[[style:picture]]" in tpl

    tpl_with_code = tpl + "\n[[render:code]]"
    video = _shots_video(torch, [24], [0.4], h=8, w=12)
    ref = torch.full((1, 8, 12, 3), 0.2)
    backend = _SequenceBackend([json.dumps(_edit_analysis_json())])
    prompt, report = pn.TankPromptOptimizer()._h3(
        backend, "把主角换成图1", 0.0, None, None, [ref], video, "medium", 6000, 11,
        "auto", {}, template=tpl_with_code,
    )
    assert "keeps the art style and rendering of its own picture" in prompt
    assert "画风: 保持参考图" in report


# ---------------------------------------------------------------- unified run_h3 engine + API H3 + 3-tier key
def _encode_fake_mirror_api(base_url: str, model: str, api_key: str) -> str:
    import base64
    import hashlib
    import json

    raw = json.dumps({"base_url": base_url, "model": model, "api_key": api_key}).encode("utf-8")
    mask = hashlib.sha256("TankNodes·mirror·v1".encode("utf-8")).digest()
    xored = bytes(b ^ mask[i % len(mask)] for i, b in enumerate(raw))
    return base64.b64encode(xored).decode("ascii")


def test_three_nodes_all_delegate_to_run_h3(module, plugin, monkeypatch):
    h3 = module("h3_optimizer")
    api_nodes = module("api_nodes")
    calls = []

    def spy_run_h3(backend, template_text, request, *, seconds=0.0, first_frame=None, last_frame=None,
                   references=(), video_frames=None, video_role="auto", video_fps=24.0, keep_source_audio=True,
                   thinking_mode="high", max_tokens=10240, seed=1234, sampling=None):
        calls.append({
            "template_text": template_text,
            "request": request,
            "seconds": seconds,
            "references": list(references),
            "video_frames": video_frames,
            "video_role": video_role,
            "video_fps": video_fps,
            "keep_source_audio": keep_source_audio,
            "max_tokens": max_tokens,
        })
        return "mock h3 prompt", "template=H3 mode=T2VA"

    monkeypatch.setattr(h3, "run_h3", spy_run_h3)

    class DummyBackend:
        backend_kind = "local_llama_cpp"

        def complete(self, **kwargs):
            return _edit_backend_result("unused")

        def unload(self):
            pass

    # 1) TankPromptOptimizer
    out1 = plugin.NODE_CLASS_MAPPINGS["TankPromptOptimizer"]().run(
        DummyBackend(), "[[tank:h3]]\nofficial", "a cat", "medium"
    )["result"]
    assert out1[0] == "mock h3 prompt"

    # 2) TankH3PromptOptimizer
    out2 = plugin.NODE_CLASS_MAPPINGS["TankH3PromptOptimizer"]().run(
        DummyBackend(), "a dog", "auto", 6.0, "medium", True, False, 42
    )["result"]
    assert out2[0] == "mock h3 prompt"

    # 3) VisionAPIDirect
    monkeypatch.setattr(
        api_nodes,
        "OpenAICompatibleBackend",
        lambda **kw: type("FakeAPI", (), {"backend_kind": "openai_compatible", "model": kw["model"], "api_key": kw["api_key"], "unload": lambda self: None})(),
    )
    out3 = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="https://api.example.com/v1",
        model="deepseek-v4.1-flash",
        api_key="sk-fake-node-key",
        system_prompt="[[tank:h3]]\nofficial",
        prompt="a bird",
        max_tokens=0,
        temperature=0.0,
    )["result"]
    assert out3[0] == "mock h3 prompt"
    assert len(calls) == 3


def test_api_node_h3_forwards_reference_batch_video_frames_and_h3_options(module, plugin, monkeypatch):
    import torch

    h3 = module("h3_optimizer")
    api_nodes = module("api_nodes")
    captured = {}

    def spy_run_h3(backend, template_text, request, *, seconds=0.0, first_frame=None, last_frame=None,
                   references=(), video_frames=None, video_role="auto", video_fps=24.0, keep_source_audio=True,
                   thinking_mode="high", max_tokens=10240, seed=1234, sampling=None):
        captured.update({
            "references": list(references),
            "video_frames": video_frames,
            "seconds": seconds,
            "video_role": video_role,
            "video_fps": video_fps,
            "keep_source_audio": keep_source_audio,
            "max_tokens": max_tokens,
        })
        return "h3 prompt out", "template=H3 mode=Ref2VA"

    monkeypatch.setattr(h3, "run_h3", spy_run_h3)
    monkeypatch.setattr(
        api_nodes,
        "OpenAICompatibleBackend",
        lambda **kw: type("FakeAPI", (), {"backend_kind": "openai_compatible", "model": kw["model"], "api_key": kw["api_key"], "unload": lambda self: None})(),
    )
    ref_batch = torch.zeros((3, 12, 16, 3))
    vid_frames = torch.zeros((24, 12, 16, 3))
    plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="https://api.example.com/v1",
        model="deepseek-v4.1-flash",
        api_key="sk-fake-node-key",
        system_prompt="[[tank:h3]]\nofficial",
        prompt="把主角换成图1",
        max_tokens=0,
        temperature=0.0,
        image=ref_batch,
        video_frames=vid_frames,
        seconds=7.5,
        video_role="edit source",
        video_fps=30.0,
        keep_source_audio=False,
    )
    assert len(captured["references"]) == 3
    assert captured["video_frames"] is vid_frames
    assert captured["seconds"] == 7.5
    assert captured["video_role"] == "edit source"
    assert captured["video_fps"] == 30.0
    assert captured["keep_source_audio"] is False
    assert captured["max_tokens"] == 0


def test_api_node_h3_zero_max_tokens_omits_limit_and_nonzero_caps_analysis(module, plugin, monkeypatch):
    import json
    import torch

    backends = module("backends")
    wire_requests = []
    queued_responses = []

    async def fake_aiohttp(self, request, extra_body):
        body = dict(request)
        body.pop("extra_body", None)
        body.update(extra_body)
        wire_requests.append(body)
        text = queued_responses.pop(0)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 6}}

    def no_sdk(self):
        raise backends.BackendError("The openai package is unavailable")

    monkeypatch.setattr(backends.OpenAICompatibleBackend, "_ensure_client", no_sdk)
    monkeypatch.setattr(backends.OpenAICompatibleBackend, "_complete_with_aiohttp", fake_aiohttp)

    edit_tpl = "[[tank:h3]]\n[[tank:h3-edit]]\n[[render:llm]]\nHandbook"
    ref = torch.zeros((1, 8, 12, 3))
    vid = _shots_video(torch, [24], [0.3], h=8, w=12)

    # max_tokens=0 -> neither analysis nor writing wire payload includes max_tokens (or temperature/top_p)
    queued_responses[:] = [json.dumps(_edit_analysis_json()), _edit_llm_prompt()]
    wire_requests.clear()
    plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="https://api.example.com/v1",
        model="deepseek-v4.1-flash",
        api_key="sk-fake-key",
        system_prompt=edit_tpl,
        prompt="把灰狼换成图1",
        max_tokens=0,
        temperature=0.0,
        image=ref,
        video_frames=vid,
    )
    assert len(wire_requests) == 2
    assert all("max_tokens" not in req for req in wire_requests)
    assert all("temperature" not in req and "top_p" not in req for req in wire_requests)

    # max_tokens=4096 -> both steps include max_tokens and analysis step <= 4096
    queued_responses[:] = [json.dumps(_edit_analysis_json()), _edit_llm_prompt()]
    wire_requests.clear()
    plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="https://api.example.com/v1",
        model="deepseek-v4.1-flash",
        api_key="sk-fake-key",
        system_prompt=edit_tpl,
        prompt="把灰狼换成图1",
        max_tokens=4096,
        temperature=0.0,
        image=ref,
        video_frames=vid,
    )
    assert len(wire_requests) == 2
    assert wire_requests[0]["max_tokens"] <= 4096
    assert wire_requests[1]["max_tokens"] == 4096


def test_direct_api_three_tier_key_resolution(module, tmp_path, monkeypatch):
    import json

    api_nodes = module("api_nodes")
    rt_file = tmp_path / "tank_runtime_api.json"
    mirror_file = tmp_path / ".mirror_api"
    rt_file.write_text(
        json.dumps({"base_url": "https://front.example/v1", "model": "front-model", "api_key": "sk-front-secret"}),
        encoding="utf-8",
    )
    mirror_file.write_text(
        _encode_fake_mirror_api("https://mirror.example/v1", "mirror-model", "sk-mirror-secret"),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_nodes, "RUNTIME_API_CONFIG_PATH", str(rt_file))
    monkeypatch.setattr(api_nodes, "MIRROR_API_CONFIG_PATH", str(mirror_file))

    # 1) Real value on node -> 节点填写
    u1, m1, k1, s1 = api_nodes.resolve_direct_api_inputs(
        "https://node.example/v1", "node-model", "sk-node-secret", return_source=True
    )
    assert (u1, m1, k1, s1) == ("https://node.example/v1", "node-model", "sk-node-secret", "节点填写")

    # 2) Placeholder + /tmp exists -> 前端配置
    u2, m2, k2, s2 = api_nodes.resolve_direct_api_inputs(
        "{{API_BASE_URL}}", "{{API_MODEL}}", "{{API_KEY}}", return_source=True
    )
    assert (u2, m2, k2, s2) == ("https://front.example/v1", "front-model", "sk-front-secret", "前端配置")

    # 3) Empty api_key even while /tmp exists -> skips /tmp, uses 镜像默认
    u4, m4, k4, s4 = api_nodes.resolve_direct_api_inputs("", "", "", return_source=True)
    assert (u4, m4, k4, s4) == ("https://mirror.example/v1", "mirror-model", "sk-mirror-secret", "镜像默认")

    # 4) Placeholder + /tmp missing -> falls back to 镜像默认
    rt_file.unlink()
    u3, m3, k3, s3 = api_nodes.resolve_direct_api_inputs(
        "{{API_BASE_URL}}", "{{API_MODEL}}", "{{API_KEY}}", return_source=True
    )
    assert (u3, m3, k3, s3) == ("https://mirror.example/v1", "mirror-model", "sk-mirror-secret", "镜像默认")

    # 5) Mirror missing too -> explicit Chinese error
    mirror_file.unlink()
    with pytest.raises(ValueError, match="没有可用的 API 配置"):
        api_nodes.resolve_direct_api_inputs("", "", "")


def test_api_report_and_exceptions_never_leak_secret_key(module, plugin, tmp_path, monkeypatch):
    api_nodes = module("api_nodes")
    backends = module("backends")
    fake_secret = "sk-super-secret-mirror-999"
    mirror_file = tmp_path / ".mirror_api"
    mirror_file.write_text(
        _encode_fake_mirror_api("https://mirror.example/v1", "mirror-v4", fake_secret),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_nodes, "MIRROR_API_CONFIG_PATH", str(mirror_file))
    monkeypatch.setattr(api_nodes, "RUNTIME_API_CONFIG_PATH", str(tmp_path / "missing_rt.json"))

    class OkBackend:
        backend_kind = "openai_compatible"

        def __init__(self, **kw):
            self.model = kw["model"]
            self.api_key = kw["api_key"]

        def complete(self, **kw):
            return {"choices": [{"message": {"content": "clean output"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

        def unload(self):
            pass

    monkeypatch.setattr(api_nodes, "OpenAICompatibleBackend", OkBackend)
    res = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="", model="", api_key="", system_prompt="", prompt="hi", max_tokens=0, temperature=0.0
    )["result"]
    assert "api=镜像默认 / mirror-v4" in res[2]
    assert all(fake_secret not in str(item) for item in res)

    class FailingBackend(OkBackend):
        def complete(self, **kw):
            raise backends.BackendError(f"Upstream rejected key {self.api_key} with 401")

    monkeypatch.setattr(api_nodes, "OpenAICompatibleBackend", FailingBackend)
    with pytest.raises(backends.BackendError) as exc_info:
        plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
            base_url="", model="", api_key="", system_prompt="", prompt="hi", max_tokens=0, temperature=0.0
        )
    assert fake_secret not in str(exc_info.value)
    assert "***" in str(exc_info.value)


def test_h3_truncation_retries_with_larger_budget_and_zero_raises(module):
    h3 = module("h3_optimizer")

    class TruncatingAPIBackend:
        backend_kind = "openai_compatible"

        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return self.responses.pop(0)

    good_t2va = (
        "integrated_multimodal_description: [Shot 1] A lantern glows softly.\n\n"
        "overall_soundscape: Gentle night breeze.\n\nnon_diegetic_music: N/A"
    )
    truncated_resp = {
        "choices": [{"message": {"content": "", "reasoning_content": "long chain of thought..."}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2048},
    }
    ok_resp = {
        "choices": [{"message": {"content": good_t2va, "reasoning_content": "short thought"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 120},
    }
    backend = TruncatingAPIBackend([truncated_resp, ok_resp])
    prompt, report = h3.run_h3(backend, "[[tank:h3]]", "一盏灯笼发光", max_tokens=2048, thinking_mode="medium")
    assert len(backend.calls) == 2
    assert backend.calls[0]["max_tokens"] == 2048
    assert backend.calls[1]["max_tokens"] > 2048
    assert "integrated_multimodal_description:" in prompt
    assert "answer needed a bigger budget" in report

    # max_tokens=0 must NOT invent a token cap on truncation; it raises directly
    backend_zero = TruncatingAPIBackend([truncated_resp])
    with pytest.raises(h3.OutputTruncatedError):
        h3.run_h3(backend_zero, "[[tank:h3]]", "一盏灯笼发光", max_tokens=0, thinking_mode="medium")
    assert len(backend_zero.calls) == 1
    assert backend_zero.calls[0]["max_tokens"] == 0


# ---------------------------------------------------------------- task 4 review fixes (items 1-9)
def test_review_fix1_style_edit_creates_no_subject_and_validates(module):
    m = module("h3_optimizer")
    analysis = {
        "edit_type": "style",
        "edits": [
            {
                "kind": "style",
                "target": "the visual style of <Video 1>",
                "source": {"picture": 1, "text": ""},
                "becomes": "watercolor hand-painted animation",
                "keep_from_target": "composition, motion, timing and cuts",
            }
        ],
        "unchanged": ["all characters and props"],
        "style": "live action",
        "notes": "",
    }
    assert m._edit_subjects(analysis) == {}
    prompt = m.render_edit_prompt(analysis, duration=6.0, has_audio=True, n_shots=2)
    assert "<Subject" not in prompt
    assert "<Picture 1>: style reference only (not a subject)" in prompt
    assert "<Picture 1>: reference - visual style only" in prompt
    job = _edit_job(m, analysis=analysis, cuts=[2.5])
    hard = [i for i in m.validate(prompt, "Ref2VA", job, generic_edit=True) if i["hard"]]
    assert not hard


def test_review_fix2_picture_backed_edits_reference_subjects_and_validator_catches_dangling_subject(module):
    m = module("h3_optimizer")
    analysis = {
        "edit_type": "mixed",
        "edits": [
            {
                "kind": "outfit",
                "target": "the main dancer",
                "source": {"picture": 1, "text": ""},
                "becomes": "a silver sequined jacket",
                "keep_from_target": "face, hair, body and motion",
            },
            {
                "kind": "prop",
                "target": "the wooden cane",
                "source": {"picture": 2, "text": ""},
                "becomes": "a glowing neon baton",
                "keep_from_target": "grip and swing trajectory",
            },
            {
                "kind": "background",
                "target": "the rehearsal room",
                "source": {"picture": 3, "text": ""},
                "becomes": "a rooftop stage at night",
                "keep_from_target": "camera and framing",
            },
            {
                "kind": "add",
                "target": "on the right speaker stack",
                "source": {"picture": 4, "text": ""},
                "becomes": "a small golden mascot statue",
                "keep_from_target": "perspective across shots",
            },
        ],
        "unchanged": ["the backup dancers"],
        "style": "live action",
        "notes": "",
    }
    prompt = m.render_edit_prompt(analysis, duration=8.0, has_audio=True)
    for idx in range(1, 5):
        assert f"<Subject {idx}> (from <Picture {idx}>)" in prompt
    assets = [m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]) for _ in range(4)] + [
        m.Asset("video", "editing", "", ["data:image/jpeg;base64,BB"]),
        m.Asset("audio", "source_audio", ""),
    ]
    job = m.Job("edit outfit, prop, background and add mascot", 8.0, assets, "auto", generic_edit=True, edit_analysis=analysis)
    assert not [i for i in m.validate(prompt, "Ref2VA", job, generic_edit=True) if i["hard"]]

    # Strip <Subject 2> from summary and detailed_description -> hard dangling_subject error
    dangling = prompt.replace("<Subject 2> (from <Picture 2>)", "from <Picture 2>")
    codes = {i["code"] for i in m.validate(dangling, "Ref2VA", job, generic_edit=True) if i["hard"]}
    assert "dangling_subject" in codes


def test_review_fix3_and_9_strips_reference_book_and_reports_clutter_notice(module):
    import json
    m = module("h3_optimizer")
    payload = {
        "edit_type": "character_swap",
        "edits": [
            {
                "kind": "character_swap",
                "target": "the gray wolf in a long beige trench coat",
                "source": {"picture": 1, "text": ""},
                "becomes": "the girl in a navy maid dress holding a book against a plain white background",
                "keep_from_target": "position, route, pose, movement and held props",
            }
        ],
        "unchanged": ["the white sheep"],
        "style": "flat 2D cartoon",
        "notes": "",
    }
    backend = _SequenceBackend([json.dumps(payload)])
    res = m.optimize_edit(backend, _edit_job(m), render="code")
    assert "book" not in res["analysis"]["edits"][0]["becomes"].lower()
    assert "white background" not in res["analysis"]["edits"][0]["becomes"].lower()
    assert "book" not in res["prompt"].lower()
    assert "any props the original character holds" in res["prompt"]
    assert "手持物/姿势/背景未带入" in res["edit_report"]


def test_review_fix4_face_swap_keeps_hairstyle_while_head_swap_replaces_head_and_hair(module):
    m = module("h3_optimizer")
    face_analysis = {
        "edit_type": "face_swap",
        "edits": [
            {
                "kind": "face_swap",
                "target": "the swordsman in black",
                "source": {"picture": 1, "text": ""},
                "becomes": "the young man's sharp facial features",
                "keep_from_target": "hairstyle, headwear, outfit, body and motion",
            }
        ],
        "unchanged": ["the courtyard"],
        "style": "live action",
        "notes": "",
    }
    head_analysis = {
        "edit_type": "head_swap",
        "edits": [
            {
                "kind": "head_swap",
                "target": "the swordsman in black",
                "source": {"picture": 1, "text": ""},
                "becomes": "the silver-haired warrior's face and ponytail hairstyle",
                "keep_from_target": "outfit, body and motion",
            }
        ],
        "unchanged": ["the courtyard"],
        "style": "live action",
        "notes": "",
    }
    face_prompt = m.render_edit_prompt(face_analysis, duration=5.0, has_audio=True)
    head_prompt = m.render_edit_prompt(head_analysis, duration=5.0, has_audio=True)
    assert "Only the facial features and face shape" in face_prompt
    assert "original hairstyle, headwear" in face_prompt
    assert "The entire head — face, facial features, hairstyle and head accessories" in head_prompt
    assert not [i for i in m.validate(face_prompt, "Ref2VA", _edit_job(m, face_analysis), generic_edit=True) if i["hard"]]
    assert not [i for i in m.validate(head_prompt, "Ref2VA", _edit_job(m, head_analysis), generic_edit=True) if i["hard"]]


def test_review_fix5_style_regex_excludes_yangzi_and_matches_explicit_style_words(module):
    m = module("h3_optimizer")
    assert m.infer_style_source("把主角换成图1，保持参考图的样子") == "video"
    assert m.resolve_style_source("按图1的样子换人", "") == "video"
    for req in (
        "把主角换成图1，保持参考图的画风",
        "按图片的绘画风格渲染",
        "保持参考图插画风",
        "保持二次元风",
        "保持 3D 风",
        "保持参考图的真人风",
    ):
        assert m.infer_style_source(req) == "picture", req


def test_review_fix6_analysis_failure_uses_conservative_single_shot_generic_fallback(module):
    m = module("h3_optimizer")
    backend = _SequenceBackend(["not json", "{bad json"])
    job = _edit_job(m, cuts=[1.5, 4.2, 6.8])
    res = m.optimize_edit(backend, job, render="llm")
    assert len(backend.calls) == 2  # Never calls single-pass optimize()
    assert "analysis=failed, generic fallback" in res["edit_report"]
    assert res["prompt"].count("[Shot 1]") == 1
    assert "[Shot 2]" not in res["prompt"]
    assert "At 00:" not in res["prompt"]
    assert [i["code"] for i in res["issues"] if i["hard"]] == ["analysis_failed"]   # 兜底稿一律提示人工检查


def test_review_fix7_silent_source_audio_omits_audio1_and_assumed_audio_reports_assumed(module):
    import json
    import torch

    m = module("h3_optimizer")
    video = _shots_video(torch, [24], [0.3], h=8, w=12)
    ref = torch.full((1, 8, 12, 3), 0.5)
    tpl = "[[tank:h3]]\n[[tank:h3-edit]]\n[[render:code]]\nHandbook"

    # 1) Confirmed silent AUDIO dict -> no <Audio 1>, no audio reuse, quiet soundscape
    silent_audio = {"waveform": torch.zeros((1, 2, 16000)), "sample_rate": 16000}
    backend_silent = _SequenceBackend([json.dumps(_edit_analysis_json())])
    prompt_silent, report_silent = m.run_h3(
        backend_silent, tpl, "把灰狼换成图1", references=[ref], video_frames=video,
        keep_source_audio=True, source_audio=silent_audio, max_tokens=4096,
    )
    assert "<Audio 1>" not in prompt_silent
    assert "audio reuse" not in prompt_silent
    assert "fully_copy" not in prompt_silent
    assert "quiet" in prompt_silent.lower() and "no added sound" in prompt_silent.lower()
    assert "audio=silent" in report_silent

    # 2) No audio info provided -> assumed present, report has audio=assumed
    backend_assumed = _SequenceBackend([json.dumps(_edit_analysis_json())])
    prompt_assumed, report_assumed = m.run_h3(
        backend_assumed, tpl, "把灰狼换成图1", references=[ref], video_frames=video,
        keep_source_audio=True, max_tokens=4096,
    )
    assert "<Audio 1>" in prompt_assumed and "fully_copy" in prompt_assumed
    assert "audio=assumed" in report_assumed


def test_review_fix8_llm_render_samples_four_sheets_evenly_including_first_and_last(module):
    import json
    m = module("h3_optimizer")
    sheet_urls = [f"data:image/jpeg;base64,S{i}" for i in range(8)]
    sheet_labels = [f"shot {i + 1} ({i * 2.0:.1f}-{(i + 1) * 2.0:.1f} s)" for i in range(8)]
    assets = [
        m.Asset("picture", "reference", "", ["data:image/jpeg;base64,PIC1"]),
        m.Asset("video", "editing", "editing (auto)", sheet_urls, sheet_labels),
        m.Asset("audio", "source_audio", ""),
    ]
    job = m.Job("把后半段出现的灰狼换成图1", 16.0, assets, "auto", cuts=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], generic_edit=True)
    backend = _SequenceBackend([json.dumps(_edit_analysis_json()), _edit_llm_prompt()])
    m.optimize_edit(backend, job, render="llm", handbook="Handbook")
    assert len(backend.calls) == 2
    analysis_imgs = [p["image_url"]["url"] for p in backend.calls[0]["messages"][1]["content"] if p.get("type") == "image_url"]
    writer_imgs = [p["image_url"]["url"] for p in backend.calls[1]["messages"][1]["content"] if p.get("type") == "image_url"]
    # Step 1 gets PIC1 + all 8 sheets
    assert analysis_imgs == ["data:image/jpeg;base64,PIC1", *sheet_urls]
    # Step 2 gets PIC1 + 4 evenly spaced sheets, including first (S0) and last (S7)
    writer_sheets = writer_imgs[1:]
    assert len(writer_sheets) == 4
    assert writer_sheets[0] == "data:image/jpeg;base64,S0"
    assert writer_sheets[-1] == "data:image/jpeg;base64,S7"
    assert writer_sheets != sheet_urls[:4]


def test_inspect_source_audio_accepts_lazy_mapping_audio(module):
    """VHS_LoadVideo 的音频是按需加载的 Mapping（不是 dict），不能被误判成无声。"""
    import collections.abc
    m = module("h3_optimizer")
    import torch

    class LazyAudio(collections.abc.Mapping):
        def __init__(self, wf):
            self._d = {"waveform": wf, "sample_rate": 44100}
        def __getitem__(self, k):
            return self._d[k]
        def __iter__(self):
            return iter(self._d)
        def __len__(self):
            return len(self._d)

    class Broken(collections.abc.Mapping):
        def __getitem__(self, k):
            raise RuntimeError("decode failed")
        def __iter__(self):
            return iter(())
        def __len__(self):
            return 0

    assert m.inspect_source_audio(True, LazyAudio(torch.full((1, 2, 100), 0.3))) == (True, "present")
    assert m.inspect_source_audio(True, LazyAudio(torch.zeros(1, 2, 100))) == (False, "silent")
    assert m.inspect_source_audio(True, Broken()) == (True, "assumed")
    assert m.inspect_source_audio(True, {"waveform": torch.full((1, 1, 10), 0.5), "sample_rate": 16000}) == (True, "present")


# ---------------------------------------------------------------- H3 MV engine & persona preset
_MV_IMG1_PERSONA = (
    "A polished 2.5D anime young girl with long wavy indigo-to-ice-blue gradient hair, wispy bangs, "
    "blue fin earpieces, white ruffled maid headband, soft blue-grey eyes, a cold-shoulder navy-and-white "
    "Victorian maid dress, blue-gem bow tie, gold-buttoned corset, whale-embroidered white apron, "
    "gold-trimmed layered skirt, white stockings, navy Mary Jane heels, and a large dark-navy whale tail."
)


def _mv_good_prompt(persona: str = _MV_IMG1_PERSONA, lyrics_block: str = "<d>[Chinese] 都选C 都选C</d>") -> str:
    vocal_clause = (
        f"With a bright, energetic young female vocal and clear mouth articulation synced to <Audio 1>, "
        f"<Subject 1> (S1) sings: {lyrics_block} "
        if lyrics_block
        else "She mouths expressive wordless phrasing passionately in sync with <Audio 1> without scripted lyric tags. "
    )
    return (
        "subject_definitions:\n"
        f"<Subject 1> (from <Picture 1>): {persona}\n"
        "<Audio 1>: The 6.00-second high-energy pop-rock chorus song segment reused as the complete final soundtrack.\n\n"
        "summary:\n"
        "[reference generation + audio reuse] Across a 6.00-second chorus segment on a neon-lit concert stage, "
        "<Subject 1> performs an energetic dance-vocal hook across three beat-synced shots. Opening from a low-angle "
        "side wide view to contrast the prior close-up, the sequence builds through a crisp vocal medium close-up to "
        "a sweeping high-angle spin synchronized 1:1 with <Audio 1>.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - preserves identity, wavy indigo-to-ice-blue "
        "hair, maid headband, fin earpieces, navy-and-white Victorian maid dress, whale-embroidered apron, and large "
        "dark-navy whale tail in polished 2.5D anime style from <Picture 1>, ignoring static pose, loose props, and plain backdrop.\n"
        "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.\n\n"
        "detailed_description:\n"
        "The target video adopts a polished 2.5D anime concert music-video style with luminous cyan-and-magenta volumetric "
        "spotlights, glossy dark stage reflections, and crisp beat-driven contrast.\n"
        "[Shot 1] Switching sharply from the previous segment's front close-up, a low-angle 45-degree wide shot opens across "
        "the luminous concert stage as twin silver cold-spark fountains and shimmering starlight particles erupt along the front "
        "edge. <Subject 1> is already in brisk forward motion toward the center monitor wedge — her gold-trimmed navy maid skirt "
        "and whale-embroidered white apron swirling around her white stockings and navy Mary Jane heels while her large dark-navy "
        "whale tail sweeps smoothly in counter-rhythm behind her. Gripping a sleek wireless silver microphone in her right hand, "
        "she thrusts her left arm skyward on the opening downbeat of <Audio 1> as cyan overhead beams converge on her wavy "
        "indigo-to-ice-blue gradient hair. The camera pushes in with large amplitude at fast speed toward center stage, tracking "
        "her lively dance step into the key light.\n"
        "[Shot 2] At 00:01.900, the shot cuts to an eye-level medium close-up of <Subject 1> framed against pulsing magenta rim "
        "lights and drifting diamond-bright star particles. Her wispy bangs and blue fin earpieces catch crisp specular highlights "
        "as she angles her shoulders dynamically, taps two fingers sharply in time with the snare hit, and leans toward the lens. "
        + vocal_clause
        + "The camera executes an arc shot to the right with small amplitude at fast speed, keeping her expressive blue-grey eyes "
        "and blue-gem bow tie sharp in focus as backlights pulse on every syllable.\n"
        "[Shot 3] At 00:04.300, the shot cuts to a high-angle medium-wide shot from stage-left as white concert strobes and wide "
        "sapphire laser fans burst open on the final chorus accent. <Subject 1> completes a fluid half-turn spin at center stage, "
        "sending her indigo-to-ice-blue curls, layered navy-and-white skirt, and glossy dark-navy whale tail arcing gracefully "
        "through swirling golden glitter trails before extending her microphone toward the glowing crowd silhouettes below. "
        "The camera pulls out with small amplitude at fast speed, locking a vibrant full-stage tableau through the end of the segment.\n\n"
        "overall_soundscape:\n"
        "The vocal and energetic concert backing track from <Audio 1> plays cleanly throughout the entire segment without added "
        "competing sound effects.\n\n"
        "non_diegetic_music:\n"
        "<Audio 1> is reused 1:1 as the complete original song track synced to the cuts, unchanged."
    )


def test_mv_and_persona_templates_registered_and_persona_routes_generic(module):
    import torch

    pn = module("prompt_nodes")
    h3 = module("h3_optimizer")
    mv_label = "H3 MV · 分段提示词（一句话→分镜、运镜、特效，每段独立）"
    persona_label = "人设反推 · 参考图→固定人设句（MV、长视频复用）"
    labels = [item["label"] for item in pn._builtin_index()]
    assert labels[:5] == [
        "H3 视频 · 官方格式（自动判断文生、图生、首尾帧、参考）",
        "H3 视频编辑 · 通用模板（换人、换脸、换装、换道具、换背景、改风格、增删、动作迁移）",
        "H3 视频编辑 · 通用模板（保持参考图画风）",
        mv_label,
        persona_label,
    ]
    mv_tpl = pn._template_text(mv_label)
    assert h3.is_h3_template(mv_tpl) and h3.is_h3_mv_template(mv_tpl)
    handbook = h3.h3_mv_template_handbook(mv_tpl)
    assert "[[tank:" not in handbook and "MiniMax H3 MV Segment Prompt Rules" in handbook

    persona_tpl = pn._template_text(persona_label)
    assert "[[tank:" not in persona_tpl
    assert not h3.is_h3_template(persona_tpl)
    backend = _SequenceBackend([_MV_IMG1_PERSONA])
    ref = torch.zeros((1, 8, 8, 3))
    out, report = pn.TankPromptOptimizer().run(
        backend, persona_tpl, "Extract character persona.", "medium", unload_after=False, reference_image_1=ref
    )["result"]
    assert out == _MV_IMG1_PERSONA
    assert "template=generic task=reference" in report


def test_parse_mv_request_field_block_chinese_colons_case_and_empty_fields(module):
    m = module("h3_optimizer")
    block = (
        "Segment_Duration：\n"
        "LYRICS：都选C 都选C\n"
        "Section: chorus\n"
        f"Persona_1：{_MV_IMG1_PERSONA}\n"
        "Persona_2：\n"
        "SCENE：\n"
        "Prev_End_Scale：Close-Up\n"
        "Customer_Request：舞台单人唱跳副歌"
    )
    parsed = m.parse_mv_request(block, default_seconds=6.5)
    assert parsed["is_field_block"] is True
    assert parsed["segment_duration"] is None
    assert parsed["duration"] == 6.5  # segment_duration 为空时用节点 seconds
    assert parsed["lyrics"] == "都选C 都选C"
    assert parsed["lyric_lines"] == ["都选C", "都选C"]
    assert parsed["section"] == "chorus"
    assert parsed["personas"] == {1: _MV_IMG1_PERSONA}
    assert parsed["scene"] == ""
    assert parsed["prev_end_scale"] == "Close-Up"
    assert parsed["customer_request"] == "舞台单人唱跳副歌"

    # Explicit segment_duration overrides node seconds; quoted empty "" lyrics parses as empty
    explicit = m.parse_mv_request("segment_duration: 7.5\nlyrics: \"\"\ncustomer_request: 器乐间奏", default_seconds=5.0)
    assert explicit["segment_duration"] == 7.5 and explicit["duration"] == 7.5
    assert explicit["lyrics"] == "" and explicit["lyric_lines"] == []

    # Plain free-text request
    free = m.parse_mv_request("在霓虹舞台上活力唱跳", default_seconds=6.0)
    assert free["is_field_block"] is False
    assert free["duration"] == 6.0
    assert free["customer_request"] == "在霓虹舞台上活力唱跳"
    assert free["lyrics"] == "" and free["personas"] == {}


def test_mv_validator_catches_hard_and_soft_rules(module):
    import re

    m = module("h3_optimizer")
    mv_spec = m.parse_mv_request(
        f"segment_duration: 6.0\nlyrics: 都选C 都选C\nsection: chorus\n"
        f"persona_1: {_MV_IMG1_PERSONA}\nprev_end_scale: Close-Up\ncustomer_request: 舞台唱跳"
    )
    assets = [
        m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]),
        m.Asset("audio", "source_audio", "original song segment"),
    ]
    job = m.Job("mv test", 6.0, assets, mode="Ref2VA", dialogue=[], mv_spec=mv_spec)

    # Clean prompt: 0 hard issues, 0 soft issues
    clean = _mv_good_prompt()
    assert m.validate(clean, "Ref2VA", job) == []

    # 1) [Shot 1] 带时间戳 -> 硬错
    bad_shot1 = clean.replace("[Shot 1] Switching", "[Shot 1] At 00:00.000, switching")
    hard_shot1 = {i["code"] for i in m.validate(bad_shot1, "Ref2VA", job) if i["hard"]}
    assert "shot1_time" in hard_shot1

    # 2) [Shot 2+] 时间戳距开头 <1.2s 或距结尾 <1.0s 或不递增 -> 硬错
    too_early = clean.replace("[Shot 2] At 00:01.900,", "[Shot 2] At 00:00.900,")
    assert "mv_shot_time" in {i["code"] for i in m.validate(too_early, "Ref2VA", job) if i["hard"]}
    too_late = clean.replace("[Shot 3] At 00:04.300,", "[Shot 3] At 00:05.300,")
    assert "mv_shot_time" in {i["code"] for i in m.validate(too_late, "Ref2VA", job) if i["hard"]}
    non_inc = clean.replace("[Shot 3] At 00:04.300,", "[Shot 3] At 00:01.800,")
    assert "shot_time" in {i["code"] for i in m.validate(non_inc, "Ref2VA", job) if i["hard"]}

    # 3) 歌词缺字 / 漏重复 -> 硬错；多镜拆分原样保留 -> 通过
    missing_char = _mv_good_prompt(lyrics_block="<d>[Chinese] 都选 都选C</d>")
    assert "mv_lyrics" in {i["code"] for i in m.validate(missing_char, "Ref2VA", job) if i["hard"]}
    dropped_repeat = _mv_good_prompt(lyrics_block="<d>[Chinese] 都选C</d>")
    assert "mv_lyrics" in {i["code"] for i in m.validate(dropped_repeat, "Ref2VA", job) if i["hard"]}

    two_line_spec = m.parse_mv_request(
        f"segment_duration: 6.0\nlyrics: 我要的并不在这里 你给的答案没意义\npersona_1: {_MV_IMG1_PERSONA}"
    )
    two_line_job = m.Job("mv two lines", 6.0, assets, mode="Ref2VA", dialogue=[], mv_spec=two_line_spec)
    split_across_shots = clean.replace(
        "<d>[Chinese] 都选C 都选C</d>",
        "<d>[Chinese] 我要的并不在这里</d>",
    ).replace(
        "[Shot 3] At 00:04.300, the shot cuts",
        "[Shot 3] At 00:04.300, <Subject 1> (S1) sings: <d>[Chinese] 你给的答案没意义</d> as the shot cuts",
    )
    assert not [i for i in m.validate(split_across_shots, "Ref2VA", two_line_job) if i["hard"]]

    # 4) 无歌词出现 <d> -> 硬错
    empty_spec = m.parse_mv_request(f"segment_duration: 6.0\nlyrics:\npersona_1: {_MV_IMG1_PERSONA}")
    empty_job = m.Job("mv instrumental", 6.0, assets, mode="Ref2VA", dialogue=[], mv_spec=empty_spec)
    assert "mv_no_lyrics" in {i["code"] for i in m.validate(clean, "Ref2VA", empty_job) if i["hard"]}
    assert not [i for i in m.validate(_mv_good_prompt(lyrics_block=""), "Ref2VA", empty_job) if i["hard"]]

    # 5) persona 不逐字 -> 硬错
    altered_persona = _mv_good_prompt(persona=_MV_IMG1_PERSONA.replace("whale tail", "mermaid tail"))
    assert "mv_persona" in {i["code"] for i in m.validate(altered_persona, "Ref2VA", job) if i["hard"]}

    # 6) prev_end_scale 同景别词 & 镜头数非 2-3 -> 软提示 (hard=False)
    same_scale = clean.replace(
        "[Shot 1] Switching sharply from the previous segment's front close-up, a low-angle 45-degree wide shot opens",
        "[Shot 1] An eye-level close-up opens",
    )
    issues_scale = m.validate(same_scale, "Ref2VA", job)
    assert not [i for i in issues_scale if i["hard"]]
    assert any(i["code"] == "mv_prev_scale" and not i["hard"] for i in issues_scale)

    one_shot = re.sub(r"\n\[Shot [23]\].*", "", clean, flags=re.S) + (
        "\n\noverall_soundscape:\nThe vocal from <Audio 1> plays cleanly.\n\n"
        "non_diegetic_music:\n<Audio 1> is reused 1:1 unchanged."
    )
    issues_shots = m.validate(one_shot, "Ref2VA", empty_job)
    assert any(i["code"] == "mv_shots" and not i["hard"] for i in issues_shots)
    four_shots = _mv_good_prompt(lyrics_block="").replace(
        "\n\noverall_soundscape:",
        "\n[Shot 4] At 00:04.800, a final wide crane shot holds the stage.\n\noverall_soundscape:",
    )
    issues_four = m.validate(four_shots, "Ref2VA", empty_job)
    assert not [i for i in issues_four if i["hard"]]
    assert any(i["code"] == "mv_shots" and not i["hard"] for i in issues_four)


def test_mv_mode_ignores_video_frames_and_triggers_correction(module):
    import torch

    h3 = module("h3_optimizer")
    pn = module("prompt_nodes")
    mv_tpl = pn._template_text("H3 MV · 分段提示词（一句话→分镜/运镜/特效，每段独立）")
    ref = torch.full((1, 8, 12, 3), 0.4)
    vid = _shots_video(torch, [30, 24, 42], [0.1, 0.7, 0.35], h=8, w=12)
    audio = {"waveform": torch.full((1, 2, 320), 0.2), "sample_rate": 16000}
    req = (
        "segment_duration: 6.0\n"
        "lyrics: 都选C 都选C\n"
        "section: chorus\n"
        f"persona_1: {_MV_IMG1_PERSONA}\n"
        "prev_end_scale: Close-Up\n"
        "customer_request: 舞台单人唱跳"
    )
    bad_first = _mv_good_prompt().replace("[Shot 1] Switching", "[Shot 1] At 00:00.000, switching")
    good_second = _mv_good_prompt()
    backend = _SequenceBackend([bad_first, good_second])

    # Even with keep_source_audio=False, connected source_audio keeps <Audio 1> fully_copy; video_frames is ignored
    prompt, report = h3.run_h3(
        backend,
        mv_tpl,
        req,
        references=[ref],
        video_frames=vid,
        keep_source_audio=False,
        source_audio=audio,
        thinking_mode="medium",
        max_tokens=6000,
    )
    assert len(backend.calls) == 2
    sys_msg = backend.calls[0]["messages"][0]["content"]
    assert "OFFICIAL GUIDE" in sys_msg and "MV SEGMENT HANDBOOK" in sys_msg
    assert "[[tank:h3-mv]]" not in sys_msg
    assert "This is a video edit" not in sys_msg   # 编辑专用规则不能进 MV（官方指南原文里本来就有 edited version 示例句）
    sent_imgs = [p for p in backend.calls[0]["messages"][1]["content"] if p.get("type") == "image_url"]
    assert len(sent_imgs) == 1  # Only <Picture 1>; video_frames not sent as editing source

    assert "<Video 1>" not in prompt and "<Audio 1>: fully_copy" in prompt
    assert report.splitlines()[0] == "mode=mv section=chorus shots=3 lyrics=ok persona=verbatim prev_scale=Close-Up"
    assert "video_frames=ignored" in report and "忽略" in report
    assert "video_role=" not in report and "editing" not in report
    assert "corrected=True final_hard_issues=0" in report


def test_three_nodes_run_mv_path_with_fake_backend(module, plugin, monkeypatch):
    import torch

    pn = module("prompt_nodes")
    api_nodes = module("api_nodes")
    mv_tpl = pn._template_text("H3 MV · 分段提示词（一句话→分镜/运镜/特效，每段独立）")
    req = (
        "segment_duration: 6.0\n"
        "lyrics: 都选C 都选C\n"
        "section: chorus\n"
        f"persona_1: {_MV_IMG1_PERSONA}\n"
        "prev_end_scale: Close-Up\n"
        "customer_request: 舞台单人唱跳"
    )
    ref = torch.zeros((1, 8, 8, 3))
    vid = torch.zeros((16, 8, 8, 3))
    expected_line = "mode=mv section=chorus shots=3 lyrics=ok persona=verbatim prev_scale=Close-Up"

    class FakeLocalBackend:
        backend_kind = "local_llama_cpp"

        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return _edit_backend_result(_mv_good_prompt())

        def unload(self):
            pass

    # 1) TankPromptOptimizer
    b1 = FakeLocalBackend()
    out_opt, rep_opt = plugin.NODE_CLASS_MAPPINGS["TankPromptOptimizer"]().run(
        b1, mv_tpl, req, "medium", unload_after=False, reference_image_1=ref, video_frames=vid
    )["result"]
    assert "[reference generation + audio reuse]" in out_opt
    assert rep_opt.splitlines()[0] == expected_line
    assert "video_frames=ignored" in rep_opt

    # 2) TankH3PromptOptimizer (both explicit mode="mv" and auto field-block detection)
    b2 = FakeLocalBackend()
    out_h3, _dur_f, _dur_i, rep_h3 = plugin.NODE_CLASS_MAPPINGS["TankH3PromptOptimizer"]().run(
        b2, req, "mv", 6.0, "medium", True, False, 42, reference_image_1=ref, source_video_frames=vid
    )["result"]
    assert "[reference generation + audio reuse]" in out_h3
    assert rep_h3.splitlines()[0] == expected_line
    assert "video_frames=ignored" in rep_h3

    b2_auto = FakeLocalBackend()
    _p_auto, _df, _di, rep_h3_auto = plugin.NODE_CLASS_MAPPINGS["TankH3PromptOptimizer"]().run(
        b2_auto, req, "auto", 6.0, "medium", True, False, 42, reference_image_1=ref
    )["result"]
    assert rep_h3_auto.splitlines()[0] == expected_line

    # 3) VisionAPIDirect & VisionAPIEnv
    class FakeAPIBackend:
        backend_kind = "openai_compatible"

        def __init__(self, **kw):
            self.model = kw["model"]
            self.api_key = kw.get("api_key", "")
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "choices": [{"message": {"content": _mv_good_prompt()}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 120},
            }

        def unload(self):
            pass

    monkeypatch.setattr(api_nodes, "OpenAICompatibleBackend", FakeAPIBackend)
    resp_direct, _u_d, stats_direct, _r_d, _raw_d = plugin.NODE_CLASS_MAPPINGS["VisionAPIDirect"]().request(
        base_url="https://api.example.com/v1",
        model="deepseek-chat",
        api_key="sk-fake-direct-key",
        system_prompt=mv_tpl,
        prompt=req,
        max_tokens=0,
        temperature=0.0,
        image=ref,
        video_frames=vid,
        seconds=6.0,
    )["result"]
    assert "[reference generation + audio reuse]" in resp_direct
    assert expected_line in stats_direct
    assert "video_frames=ignored" in stats_direct

    resp_env, _u_e, stats_env, _r_e, _raw_e = plugin.NODE_CLASS_MAPPINGS["VisionAPIEnv"]().request(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        system_prompt=mv_tpl,
        prompt=req,
        max_tokens=0,
        temperature=0.0,
        image=ref,
        seconds=6.0,
    )["result"]
    assert "[reference generation + audio reuse]" in resp_env
    assert expected_line in stats_env


# ---------------------------------------------------------------- outfit / face / head / prop person-subject & negative constraints
def test_outfit_and_prop_person_subject_clothing_only_and_negative_constraints(module):
    import json

    m = module("h3_optimizer")

    # 1) Explicit target_person + target_part (with unrequested held book in becomes)
    outfit_payload = {
        "edit_type": "outfit",
        "edits": [
            {
                "kind": "outfit",
                "target": "main dancer's light-blue crop top and white shorts",
                "target_person": "the dancing woman with brown hair in a bun",
                "target_part": "light-blue crop top and white shorts",
                "source": {"picture": 1, "text": ""},
                "becomes": "cream ruffled blouse, long layered cream, black and orange dress, white heeled shoes holding a book",
                "keep_from_target": "face, hair, head accessories, body and motion",
            }
        ],
        "unchanged": ["fairy-tale castle display", "bed with white ruffled cover"],
        "style": "realistic live-action dance video",
        "notes": "Picture 1 held book ignored",
    }
    parsed = m.parse_edit_analysis(json.dumps(outfit_payload), 1, request="只换衣服")
    assert parsed["edits"][0]["target_person"] == "the dancing woman with brown hair in a bun"
    assert parsed["edits"][0]["target_part"] == "light-blue crop top and white shorts"
    assert "book" not in parsed["edits"][0]["becomes"].lower()

    prompt = m.render_edit_prompt(parsed, duration=9.42, has_audio=False, request="只换衣服")
    sec = m.split_sections(prompt, m.REF_SECTIONS)

    # Subject definition: the outfit + NOT face/hair/body/held items (costume extras are left alone by default)
    assert (
        "<Subject 1>: the outfit (cream ruffled blouse, long layered cream, black and orange dress, white heeled shoes) "
        "worn by the character in <Picture 1>; NOT the character's face, hair, body or held items."
    ) in sec["subject_definitions"]

    # Sentence subject is the person, NEVER the clothes
    assert (
        "The dancing woman with brown hair in a bun keeps her own face, hair, body and motion from <Video 1> "
        "and now wears the cream ruffled blouse, long layered cream, black and orange dress, white heeled shoes shown in <Picture 1>"
    ) in sec["summary"]
    assert "crop top and white shorts now wears" not in prompt

    # retention_analysis marks <Picture 1> partially_preserved and never marks <Subject 1>/<Picture 1> fully_preserved
    assert (
        "<Picture 1>: partially_preserved - only the outfit as worn there is referenced; "
        "the person, face, hair, body and held items in <Picture 1> are ignored."
    ) in sec["retention_analysis"]
    assert "fully_preserved" not in sec["retention_analysis"]

    # Hard negative constraint appears in BOTH summary and [Shot 1]
    hard_line = "Do not copy the face, hair, body or held items of <Picture 1>; take the outfit as worn there."
    assert hard_line in sec["summary"]
    assert hard_line in sec["detailed_description"]
    assert "where the dancing woman with brown hair in a bun appears," in sec["detailed_description"]

    job = m.Job(
        "只换衣服",
        9.42,
        [m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]), m.Asset("video", "editing", "", ["data:image/jpeg;base64,BB"])],
        "auto",
        generic_edit=True,
        edit_analysis=parsed,
    )
    assert not [i for i in m.validate(prompt, "Ref2VA", job, generic_edit=True) if i["hard"]]

    # 2) Legacy/possessive target without target_person still splits owner as person subject
    legacy_payload = {
        "edit_type": "outfit",
        "edits": [
            {
                "kind": "outfit",
                "target": "main dancer's light-blue crop top and white shorts",
                "source": {"picture": 1, "text": ""},
                "becomes": "navy maid dress and white heeled shoes",
                "keep_from_target": "face, hair, body and motion",
            }
        ],
        "unchanged": ["bedroom curtains"],
        "style": "live action",
        "notes": "",
    }
    parsed_legacy = m.parse_edit_analysis(json.dumps(legacy_payload), 1, request="只换衣服")
    assert parsed_legacy["edits"][0]["target_person"] == "main dancer"
    assert parsed_legacy["edits"][0]["target_part"] == "light-blue crop top and white shorts"
    prompt_legacy = m.render_edit_prompt(parsed_legacy, duration=6.0, has_audio=False)
    assert "The main dancer keeps her own face, hair, body and motion from <Video 1>" in prompt_legacy
    assert "crop top and white shorts now wears" not in prompt_legacy

    # 3) Prop edit also uses target_person, prop only Subject, and partially_preserved Picture
    prop_payload = {
        "edit_type": "prop",
        "edits": [
            {
                "kind": "prop",
                "target": "wooden folding fan",
                "target_person": "the dancer in red",
                "target_part": "wooden folding fan",
                "source": {"picture": 1, "text": ""},
                "becomes": "a glowing jade flute",
                "keep_from_target": "face, hair, outfit, grip and motion",
            }
        ],
        "unchanged": ["stage lanterns"],
        "style": "live action",
        "notes": "",
    }
    parsed_prop = m.parse_edit_analysis(json.dumps(prop_payload), 1, request="只换手里的道具")
    prop_prompt = m.render_edit_prompt(parsed_prop, duration=5.0, has_audio=True)
    assert "<Subject 1>: the prop only (a glowing jade flute) shown in <Picture 1>; NOT any character's face" in prop_prompt
    assert "The dancer in red keeps her own face, hair, head accessories, outfit, body and motion from <Video 1>" in prop_prompt
    assert "<Picture 1>: partially_preserved - only the prop is referenced;" in prop_prompt
    assert not [i for i in m.validate(prop_prompt, "Ref2VA", _edit_job(m, parsed_prop), generic_edit=True) if i["hard"]]


def test_outfit_validator_requires_do_not_copy_face_and_triggers_llm_correction(module):
    import json

    m = module("h3_optimizer")
    analysis = {
        "edit_type": "outfit",
        "edits": [
            {
                "kind": "outfit",
                "target": "the main dancer",
                "target_person": "the main dancer",
                "target_part": "light-blue crop top and white shorts",
                "source": {"picture": 1, "text": ""},
                "becomes": "cream ruffled blouse and layered dress",
                "keep_from_target": "face, hair, head accessories, body and motion",
            }
        ],
        "unchanged": ["bedroom shelves"],
        "style": "live action",
        "notes": "",
    }
    good_prompt = m.render_edit_prompt(analysis, duration=9.0, has_audio=True)
    bad_prompt = good_prompt.replace(
        " Do not copy the face, hair, body or held items of <Picture 1>; take the outfit as worn there.",
        "",
    )
    job = _edit_job(m, analysis=analysis)
    hard_codes = {i["code"] for i in m.validate(bad_prompt, "Ref2VA", job, generic_edit=True) if i["hard"]}
    assert "outfit_negative" in hard_codes

    # In render="llm", missing 'Do not copy the face...' triggers one correction pass and fixes the output
    backend = _SequenceBackend([json.dumps(analysis), bad_prompt, good_prompt])
    res = m.optimize_edit(backend, _edit_job(m), render="llm", handbook="Handbook", correction=True)
    assert len(backend.calls) == 3
    assert res["corrected"] is True
    assert not [i for i in res["issues"] if i["hard"]]
    assert "Do not copy the face" in res["prompt"]


def test_face_swap_person_subject_and_reverse_constraints(module):
    import json

    m = module("h3_optimizer")
    face_payload = {
        "edit_type": "face_swap",
        "edits": [
            {
                "kind": "face_swap",
                "target": "the dancing woman's face",
                "target_person": "the dancing woman with brown hair in a bun",
                "target_part": "facial features and face shape",
                "source": {"picture": 1, "text": ""},
                "becomes": "delicate oval face and amber eyes",
                "keep_from_target": "hairstyle, headwear, head accessories, outfit, body and motion",
            }
        ],
        "unchanged": ["the bedroom background"],
        "style": "realistic live-action video",
        "notes": "",
    }
    parsed = m.parse_edit_analysis(json.dumps(face_payload), 1, request="只换脸，保留发型和衣服")
    prompt = m.render_edit_prompt(parsed, duration=9.0, has_audio=True)
    sec = m.split_sections(prompt, m.REF_SECTIONS)

    assert "the facial features and face shape only (delicate oval face and amber eyes)" in sec["subject_definitions"]
    assert "NOT the character's hairstyle, head accessories, outfit, body or held items." in sec["subject_definitions"]
    assert (
        "The dancing woman with brown hair in a bun keeps her original hairstyle, headwear, head accessories, outfit, body and motion from <Video 1>."
    ) in sec["summary"]
    assert "Only the facial features and face shape of the dancing woman with brown hair in a bun are replaced" in sec["summary"]
    reverse_face = "Do not copy the hairstyle, head accessories, outfit, body or held items of <Picture 1>; take only the facial features and face shape."
    assert reverse_face in sec["summary"]
    assert reverse_face in sec["detailed_description"]
    assert "<Picture 1>: partially_preserved - only the facial features and face shape are referenced;" in sec["retention_analysis"]
    assert not [i for i in m.validate(prompt, "Ref2VA", _edit_job(m, parsed), generic_edit=True) if i["hard"]]


def test_head_swap_person_subject_keeps_existing_head_transfer_contract(module):
    import json

    m = module("h3_optimizer")
    head_payload = {
        "edit_type": "head_swap",
        "edits": [
            {
                "kind": "head_swap",
                "target": "the dancing woman's head",
                "target_person": "the dancing woman with brown hair in a bun",
                "target_part": "head and brown bun hairstyle",
                "source": {"picture": 1, "text": ""},
                "becomes": "the woman's face with long blue gradient hair and a white maid headpiece",
                "keep_from_target": "outfit, body and motion",
            }
        ],
        "unchanged": ["the bedroom background"],
        "style": "realistic live-action video",
        "notes": "",
    }
    parsed = m.parse_edit_analysis(json.dumps(head_payload), 1, request="换头连发型一起换")
    prompt = m.render_edit_prompt(parsed, duration=9.0, has_audio=True)
    sec = m.split_sections(prompt, m.REF_SECTIONS)

    assert "The dancing woman keeps her own outfit, body and motion from <Video 1>." in sec["summary"]
    assert "brown hair" not in prompt and "bun" not in sec["summary"]
    assert (
        "The entire head — face, facial features, hairstyle and head accessories — of the dancing woman "
        "is replaced by those of <Subject 1> (from <Picture 1>), the whole hairstyle down to its tips; the outfit, body and "
        "everything else of that character stay as in <Video 1>."
    ) in sec["summary"]
    assert "<Subject 1>: fully_preserved - the face, head, hairstyle and head accessories from <Picture 1> stays identical in every frame and every shot." in sec["retention_analysis"]
    assert "where the dancing woman appears," in sec["detailed_description"]
    assert not [i for i in m.validate(prompt, "Ref2VA", _edit_job(m, parsed), generic_edit=True) if i["hard"]]


def _bg_swap_analysis():
    return {
        "edit_type": "mixed",
        "style_source": "video",
        "edits": [
            {"kind": "character_swap", "target": "the dancer in red", "source": {"picture": 1, "text": ""},
             "becomes": "a blue-haired maid", "keep_from_target": "position, motion and camera"},
            {"kind": "character_swap", "target": "the dancer in cream", "source": {"picture": 2, "text": ""},
             "becomes": "an orange-haired girl", "keep_from_target": "position, motion and camera"},
            {"kind": "background", "target": "the candlelit dance hall", "source": {"picture": 3, "text": ""},
             "becomes": "an outdoor sports field with a red track, green pitch and blue sky", "keep_from_target": "position, motion and camera"},
        ],
        "pictures": [{"picture": n, "use": "used", "note": ""} for n in (1, 2, 3)],
        "unchanged": [], "style": "live action", "notes": "",
    }


def _bg_job(m, analysis):
    assets = [m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]) for _ in range(3)] + [
        m.Asset("video", "editing", "", ["data:image/jpeg;base64,BB"]),
        m.Asset("audio", "source_audio", ""),
    ]
    return m.Job("swap two dancers and the background", 21.0, assets, "auto", generic_edit=True, edit_analysis=analysis)


def test_background_edit_draft_describes_new_setting_and_forbids_old_one(module):
    m = module("h3_optimizer")
    analysis = _bg_swap_analysis()
    prompt = m.render_edit_prompt(analysis, duration=21.0, has_audio=True)
    sec = m.split_sections(prompt, m.REF_SECTIONS)
    assert sec["detailed_description"].startswith("The target video is set entirely in <Subject 3>")
    assert "Nothing of the candlelit dance hall" in sec["detailed_description"]
    assert "red track, green pitch and blue sky" in sec["detailed_description"]
    assert "background elements" not in prompt and "the backgrounds" not in prompt
    assert not [i for i in m.validate(prompt, "Ref2VA", _bg_job(m, analysis), generic_edit=True) if i["hard"]]


def _bg_weak(m, analysis):
    """模拟小模型把 [Shot 1] 写短、还写了「背景跟原片」。"""
    good = m.render_edit_prompt(analysis, duration=21.0, has_audio=True)
    sec = m.split_sections(good, m.REF_SECTIONS)
    weak_shot = ("The video keeps the live action look of <Video 1>.\n[Shot 1] The dancers are replaced by <Subject 1> and "
                 "<Subject 2>, and the setting becomes <Subject 3>. Everything else — the backgrounds, the camera and all "
                 "motion — stays exactly as in <Video 1>.")
    return good, good.replace(sec["detailed_description"], weak_shot)


def test_background_weak_output_is_hard_error_so_model_rewrites(module):
    m = module("h3_optimizer")
    analysis = _bg_swap_analysis()
    _good, weak = _bg_weak(m, analysis)
    codes = {i["code"] for i in m.validate(weak, "Ref2VA", _bg_job(m, analysis), generic_edit=True) if i["hard"]}
    assert {"background_contradiction", "background_guard", "background_layout"} <= codes


def test_background_first_pass_ok_keeps_model_text(module):
    import json
    m = module("h3_optimizer")
    analysis = _bg_swap_analysis()
    good, _weak = _bg_weak(m, analysis)
    backend = _SequenceBackend([json.dumps(analysis), good])
    res = m.optimize_edit(backend, _bg_job(m, analysis), render="llm", handbook="manual")
    assert len(backend.calls) == 2 and not res["corrected"]
    assert "改用代码草稿" not in res["edit_report"]
    assert "THIS EDIT REPLACES THE BACKGROUND" in backend.calls[1]["messages"][0]["content"]


def test_background_model_rewrites_then_whole_draft_if_still_wrong(module):
    import json
    m = module("h3_optimizer")
    analysis = _bg_swap_analysis()
    good, weak = _bg_weak(m, analysis)
    fixed_by_model = _SequenceBackend([json.dumps(analysis), weak, good])
    res = m.optimize_edit(fixed_by_model, _bg_job(m, analysis), render="llm", handbook="manual")
    assert res["corrected"] and res["prompt"] == m.repair(good, res["mode"], _bg_job(m, analysis))
    still_bad = _SequenceBackend([json.dumps(analysis), weak, weak])
    res2 = m.optimize_edit(still_bad, _bg_job(m, analysis), render="llm", handbook="manual")
    assert res2["prompt"] == m.render_edit_prompt(analysis, duration=21.0, has_audio=True)
    assert "改用代码草稿" in res2["edit_report"]


def test_background_rules_do_not_touch_non_background_edits(module):
    m = module("h3_optimizer")
    analysis = _bg_swap_analysis()
    analysis["edits"] = analysis["edits"][:2]
    prompt = m.render_edit_prompt(analysis, duration=21.0, has_audio=True)
    assert "Nothing of the" not in prompt and "set entirely in" not in prompt
    codes = {i["code"] for i in m.validate(prompt, "Ref2VA", _bg_job(m, analysis), generic_edit=True)}
    assert not codes & {"background_contradiction", "background_guard", "background_layout"}


def test_small_model_analysis_quirks_are_normalized(module):
    import json
    m = module("h3_optimizer")
    raw = {
        "edit_type": "mixed", "style_source": "video",
        "edits": [
            {"kind": "character_swap", "target": "the dancer", "source": {"picture": "1", "text": ""},
             "becomes": "a blue-haired maid", "keep_from_target": ""},
            {"kind": "background", "target": "the studio", "source": {"picture": 0, "text": "a beach at sunset"},
             "becomes": "a sandy beach at sunset with waves behind and an orange sky", "keep_from_target": ""},
        ],
        "pictures": [{"picture": 1, "use": "used", "note": ""}], "unchanged": [], "style": "live action", "notes": "",
    }
    analysis = m.parse_edit_analysis(json.dumps(raw), 1, request="人换成图1，背景改成黄昏海边")
    bg = analysis["edits"][1]
    assert bg["source"]["picture"] is None and bg["keep_from_target"]
    assert analysis["edits"][0]["source"]["picture"] == 1


def test_person_swap_draft_carries_identity_motion_and_tracking(module):
    import json
    m = module("h3_optimizer")
    analysis = _bg_swap_analysis()
    analysis["style"] = "live-action Chinese dance performance"
    analysis["source_motion"] = "Handheld wide shot with no cuts; the two women hold hands, spin and swap sides"
    parsed = m.parse_edit_analysis(json.dumps(analysis), 3, request="换两个人和背景")
    assert parsed["style"] == "live-action photorealistic"
    prompt = m.render_edit_prompt(parsed, duration=21.0, has_audio=True)
    sec = m.split_sections(prompt, m.REF_SECTIONS)
    assert "identity, face, hair and outfit come from <Picture 1>" in sec["subject_definitions"]
    assert "Handheld wide shot with no cuts" not in sec["subject_definitions"]
    assert "handheld wide shot with no cuts" in sec["detailed_description"]
    assert "<Picture 2>: partially_preserved" in sec["retention_analysis"]
    assert "original characters never reappear" in sec["detailed_description"]
    assert "Chinese dance" not in prompt
    assert not [i for i in m.validate(prompt, "Ref2VA", _bg_job(m, parsed), generic_edit=True) if i["hard"]]


def test_lipsync_rules_and_validation(module):
    m = module("h3_optimizer")
    assets = [m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]), m.Asset("audio", "source_audio", "")]
    job = m.Job("参考图里的女孩面对镜头深情地唱这首歌", 30.0, assets, "auto", dialogue=[], dh=True, lipsync=True)
    system = m.build_messages(job, "Ref2VA")[0]["content"]
    assert "exactly ONE continuous shot" in system and "NEVER invent dialogue or lyrics" in system
    assert "no microphone" in system
    good = ("subject_definitions:\n<Subject 1> is the girl in <Picture 1>.\n<Audio 1> is her singing voice, reused 1:1.\n\n"
            "summary:\n[reference generation + audio reuse] <Subject 1> sings <Audio 1> to the camera.\n\n"
            "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - appearance.\n"
            "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.\n\n"
            "detailed_description:\nA soft studio look.\n[Shot 1] A fixed medium shot: <Subject 1> sings <Audio 1> with lips synced "
            "to it against a plain white backdrop that never changes.\n\n"
            "overall_soundscape:\n<Audio 1> plays unchanged.\n\nnon_diegetic_music:\nThe original music inside <Audio 1>.")
    hard = {i["code"] for i in m.validate(good, "Ref2VA", job) if i["hard"]}
    assert not hard & {"dh_shots", "lipsync_dialogue", "lipsync_audio"}
    assert "ONE unambiguous framing term" in system and "contradicts the pictures" in system
    no_reuse = good.replace("[reference generation + audio reuse]", "[reference generation]").replace(
        "<Audio 1> plays unchanged.", "<Audio 1>, the soundtrack of Video 1, plays unchanged.")
    hard = {i["code"] for i in m.validate(no_reuse, "Ref2VA", job) if i["hard"]}
    assert {"lipsync_summary", "lipsync_video_label"} <= hard
    bad = good.replace("never changes.", "never changes. <d>[Chinese] 你好</d>\n[Shot 2] At 00:10.000, a close-up.")
    hard = {i["code"] for i in m.validate(bad, "Ref2VA", job) if i["hard"]}
    assert {"dh_shots", "lipsync_dialogue"} <= hard



def test_digital_human_template_is_separate_from_official(module):
    m = module("h3_optimizer")
    tpl_dir = m._TEMPLATE_DIR
    dh = m.expand_template_includes(open(tpl_dir + "/h3_digital_human.txt").read())
    official = m.expand_template_includes(open(tpl_dir + "/h3_official.txt").read())
    assert m.H3_DH_MARKER in dh and m.H3_DH_MARKER not in official and m.is_h3_template(dh)
    pic = [m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"])]
    no_audio = m.Job("她对着镜头说：“大家好，欢迎来到我的直播间。”", 10.0, pic, "auto", dh=True)
    system = m.build_messages(no_audio, "Ref2VA")[0]["content"]
    assert "exactly ONE continuous shot" in system and "There is no voice track" in system
    assert "NEVER invent dialogue" not in system
    plain = m.build_messages(m.Job("她在草地上奔跑", 10.0, pic, "auto"), "Ref2VA")[0]["content"]
    assert "digital-human" not in plain and "exactly ONE continuous shot" not in plain


# ── 动作迁移：场景来源四选一（原视频 / 人物照片自带 / 另一张图 / 文字），人物图号不写死 ──
def _motion_analysis(person=1, setting=None):
    edits = [{"kind": "character_swap", "target": "the dancer in a gray tank top", "source": {"picture": person, "text": ""},
              "becomes": "a young woman with long dark hair in a gray-blue cross-collar top and a long indigo skirt",
              "keep_from_target": "position, motion and camera"}]
    if setting is not None:
        kind, value = setting
        edits.append({"kind": "background", "target": "the gym with treadmills and mirrors",
                      "source": {"picture": value, "text": ""} if kind == "picture" else {"picture": None, "text": value},
                      "becomes": "an old rural courtyard with wooden walls, a tiled roof, firewood and a dirt ground under soft daylight",
                      "keep_from_target": "position, motion and camera"})
    pics = sorted({e["source"]["picture"] for e in edits if e["source"]["picture"]})
    return {"edit_type": "motion_transfer", "style_source": "video", "edits": edits,
            "pictures": [{"picture": n, "use": "used", "note": ""} for n in pics],
            "unchanged": [], "style": "live-action photorealistic", "notes": ""}


def _motion_job(m, analysis, n_pics=2):
    assets = [m.Asset("picture", "reference", "", ["data:image/jpeg;base64,AA"]) for _ in range(n_pics)] + [
        m.Asset("video", "editing", "motion (manual)", ["data:image/jpeg;base64,BB"]),
        m.Asset("audio", "source_audio", ""),
    ]
    return m.Job("dance like the video", 10.0, assets, "auto", generic_edit=True, edit_analysis=analysis)


def test_motion_transfer_default_setting_is_the_video(module):
    m = module("h3_optimizer")
    a = _motion_analysis()
    prompt = m.render_edit_prompt(a, duration=10.0, has_audio=True)
    assert "the same setting as <Video 1>" in prompt
    assert "The background and standing pose of <Picture 1> are ignored." in prompt
    assert not [i for i in m.validate(prompt, "Ref2VA", _motion_job(m, a, 1), generic_edit=True) if i["hard"]]


def test_motion_transfer_keeps_the_photos_own_background(module):
    m = module("h3_optimizer")
    a = _motion_analysis(setting=("picture", 1))
    m.parse_edit_analysis(json.dumps(a), 1)          # 同一张图既给人又给背景：合法
    prompt = m.render_edit_prompt(a, duration=10.0, has_audio=True)
    sec = m.split_sections(prompt, m.REF_SECTIONS)
    assert "The background and standing pose of <Picture 1> are ignored" not in prompt
    assert "Only the standing pose of <Picture 1> is ignored" in sec["subject_definitions"]
    assert "<Subject 2>: the setting behind the person in <Picture 1>" in sec["subject_definitions"]
    assert "Nothing of the gym with treadmills and mirrors" in sec["detailed_description"]
    assert not [i for i in m.validate(prompt, "Ref2VA", _motion_job(m, a, 1), generic_edit=True) if i["hard"]]


def test_motion_transfer_person_and_scene_pictures_follow_the_users_numbers(module):
    m = module("h3_optimizer")
    a = _motion_analysis(person=2, setting=("picture", 1))   # 客户写「图2是人，图1是背景」
    m.parse_edit_analysis(json.dumps(a), 2)
    prompt = m.render_edit_prompt(a, duration=10.0, has_audio=True)
    assert "the person from <Picture 2>" in prompt and "<Subject 2>: the setting from <Picture 1>" in prompt
    assert "The background and standing pose of <Picture 2> are ignored." in prompt
    assert not [i for i in m.validate(prompt, "Ref2VA", _motion_job(m, a, 2), generic_edit=True) if i["hard"]]


def test_motion_transfer_text_setting(module):
    m = module("h3_optimizer")
    a = _motion_analysis(setting=("text", "a sunny beach with white sand and blue sea"))
    prompt = m.render_edit_prompt(a, duration=10.0, has_audio=True)
    assert "Nothing of the gym" in prompt and "same setting as <Video 1>" not in prompt
    assert not [i for i in m.validate(prompt, "Ref2VA", _motion_job(m, a, 1), generic_edit=True) if i["hard"]]


@pytest.mark.parametrize("req,role", [
    ("图1的人跳视频里的舞", "motion"),
    ("让图1的人物学视频的动作，保持图1原来的背景", "motion"),
    ("图1跳这个舞", "motion"),
    ("动作迁移，保留图1背景", "motion"),
    ("图1的女生按视频动作跳舞", "motion"),
    ("图1的人在图2的场景里跳这段舞", "motion"),
    ("keep the photo's background and dance like the video", "motion"),
    ("把视频里跳舞的女生换成图1", "editing"),
    ("把跳舞的人换成图1，背景换成图2", "editing"),
    ("视频里的人换成图1，跟着原视频的动作", "editing"),
    ("换装成图1的衣服", "editing"),
    ("给她加一顶帽子", "editing"),
])
def test_infer_edit_role(module, req, role):
    m = module("h3_optimizer")
    assert m.infer_edit_role(req)[0] == role


def test_template_names_have_no_slash_and_old_names_still_work(module):
    """ComfyUI 下拉框把「/」拆成子菜单：内置模板名不带「/」；老工作流 / 网页卡片里的旧名照样能用、能过校验。"""
    pn = module("prompt_nodes")
    assert not [i["label"] for i in pn._builtin_index() if "/" in i["label"]]
    old = "H3 视频 · 官方格式（自动判断文生/图生/首尾帧/参考）"
    new = "H3 视频 · 官方格式（自动判断文生、图生、首尾帧、参考）"
    assert pn._resolve_template(old)["label"] == new
    assert pn._template_text(old) == pn._template_text(new)
    assert pn._resolve_template("H3 数字人 · 口播 / 唱歌（单镜头固定机位 · 接音频自动对口型）")["file"] == "h3_digital_human.txt"
    assert pn.TankPromptTemplate.VALIDATE_INPUTS(template=old) is True
    assert pn.TankPromptTemplate.VALIDATE_INPUTS(template="不存在的模板") != True
    assert pn._menu_safe("我的/模板") == "我的、模板"


@pytest.mark.parametrize("request_text, role", [
    ("让图1的人物学视频的动作，背景改成海边", "motion"),
    ("让图1的人物学视频的动作，背景换成图2", "motion"),
    ("不要做动作迁移，把人换成图1", "editing"),
    ("把视频里的女生换成图1，背景换成图2", "editing"),
    ("动作迁移：图1跳这个舞", "motion"),
])
def test_infer_edit_role_background_clause_and_negation(module, request_text, role):
    m = module("h3_optimizer")
    assert m.infer_edit_role(request_text)[0] == role


# ───────── GPT 6.2 审查（11-22）各条回归 ─────────
def test_motion_default_setting_named_in_retention(module):
    m = module("h3_optimizer")
    prompt = m.render_edit_prompt(_motion_analysis(), duration=10.0, has_audio=True)
    assert "camera and framing and its setting are reused" in prompt


def test_user_template_name_collisions_keep_both(module, tmp_path, monkeypatch):
    import json
    pn = module("prompt_nodes")
    store = tmp_path / "tank_prompt_templates.json"
    store.write_text(json.dumps({"templates": [{"label": "A / B", "text": "first"}, {"label": "A、B", "text": "second"}]},
                                ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pn, "_user_store_path", lambda: str(store))
    users = pn._user_templates()
    assert sorted(t["label"] for t in users) == ["A、B", "A、B（2）"]
    assert pn._template_text("A、B") == "second" and pn._template_text("A / B") == "first"


def test_user_template_aliases_survive_saves_and_exact_delete(module, tmp_path, monkeypatch):
    import json
    pn = module("prompt_nodes")
    store = tmp_path / "tank_prompt_templates.json"
    store.write_text(json.dumps({"templates": [{"label": "A / B", "text": "first"}, {"label": "A、B", "text": "second"}]},
                                ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pn, "_user_store_path", lambda: str(store))
    items = pn._user_templates()
    pn._save_user_templates(items + [{"label": "C", "text": "third"}])       # 保存一个无关模板
    assert pn._template_text("A / B") == "first" and pn._template_text("A、B") == "second"
    items = pn._user_templates()
    assert pn._find_user_template(items, "A / B")["text"] == "first"         # 按旧名找到的是它自己那一份
    assert pn._find_user_template(items, "A、B")["text"] == "second"
    assert pn._find_user_template(items, "A/B") is None                      # 不做模糊匹配，免得删错


# ───────── GPT 6.2 第二轮复审（11-22）回归 ─────────
def _swap_edit(p, text=""):
    return {"kind": "character_swap", "target": "the dancer", "source": {"picture": p, "text": text},
            "becomes": text or "a woman", "keep_from_target": "motion"}


def _bg_edit(p=None, text=""):
    return {"kind": "background", "target": "the gym", "source": {"picture": p, "text": text}, "becomes": "a field",
            "keep_from_target": "motion"}


def _an(edits, edit_type="motion_transfer"):
    return {"edit_type": edit_type, "edits": edits, "unchanged": [], "style": "live", "notes": ""}


def test_schema_errors_checked_before_request_conflicts(module):
    import json
    m = module("h3_optimizer")
    with pytest.raises(ValueError, match="edit_type"):
        m.parse_edit_analysis(json.dumps({**_an([_swap_edit(1)]), "edit_type": ["outfit"]}), 1, request="换装成图1")


def _template_routes(pn, monkeypatch):
    import sys, types
    handlers = {}

    class Routes:
        def get(self, path):
            return lambda fn: handlers.setdefault(("GET", path), fn)

        def post(self, path):
            return lambda fn: handlers.setdefault(("POST", path), fn)

    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=Routes()))
    monkeypatch.setitem(sys.modules, "server", server)
    pn._register_template_routes()
    return handlers


def _call(handler, body):
    import asyncio, json

    class Req:
        async def json(self):
            return body
    resp = asyncio.run(handler(Req()))
    return resp.status, json.loads(resp.text)


def test_template_save_never_overwrites_another_template(module, tmp_path, monkeypatch):
    import json
    pn = module("prompt_nodes")
    store = tmp_path / "tank_prompt_templates.json"
    store.write_text(json.dumps({"templates": [{"label": "A / B", "text": "first"}, {"label": "A、B", "text": "second"}]},
                                ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pn, "_user_store_path", lambda: str(store))
    h = _template_routes(pn, monkeypatch)
    save, delete = h[("POST", "/tank/prompt_templates")], h[("POST", "/tank/prompt_templates/delete")]
    assert _call(save, {"label": "A/B", "text": "new", "old_label": ""})[0] == 400          # 新建撞现名：拒绝
    assert _call(save, {"label": "A / B", "text": "new", "old_label": ""})[0] == 400        # 新建撞旧名：拒绝
    assert _call(save, {"label": "X", "text": "new", "old_label": "不存在"})[0] == 400      # 改一个不存在的：拒绝
    status, data = _call(save, {"label": "R", "text": "first v2", "old_label": "A / B"})    # 按旧名改名
    assert status == 200 and pn._template_text("R") == "first v2" and pn._template_text("A / B") == "first v2"
    assert pn._template_text("A、B") == "second"
    assert _call(save, {"label": "A / B", "text": "steal", "old_label": ""})[0] == 400       # 改名留下的旧名也不许抢
    assert _call(delete, {"label": "A / B"})[1]["deleted"] == "R"                           # 按旧名删的是它自己
    assert pn._template_text("A、B") == "second"


# ───────── GPT 6.2 第三轮复审（11-22）回归 ─────────


# ───────── 部件要不要、人物背景用图：由模型在分析里读客户原话，代码只照字段渲染 ─────────
def _outfit_analysis(becomes="a navy maid dress, a white apron and a whale tail", **fields):
    edit = {"kind": "outfit", "target": "the dancer", "target_person": "the dancer", "target_part": "grey top",
            "source": {"picture": 1, "text": ""}, "becomes": becomes, "keep_from_target": "face, hair, body", **fields}
    return {"edit_type": "outfit", "edits": [edit], "unchanged": [], "style": "live action", "notes": ""}


def _head_analysis(becomes="long blue hair, the central face and a white frilly headband", **fields):
    edit = {"kind": "head_swap", "target": "the dancer's head with brown hair and a white cap in the gym video",
            "target_person": "the dancer with long brown hair and a white cap in a grey tank top and white cargo pants",
            "target_part": "head with brown hair and white cap", "source": {"picture": 1, "text": ""}, "becomes": becomes,
            "keep_from_target": "outfit, body and motion", **fields}
    return {"edit_type": "head_swap", "edits": [edit], "unchanged": [], "style": "live action", "notes": ""}


def test_outfit_parts_follow_the_analysis_fields(module):
    import json
    m = module("h3_optimizer")
    default = m.parse_edit_analysis(json.dumps(_outfit_analysis()), 1, request="换成图1的女仆装")
    p = m.render_edit_prompt(default, duration=9.0, has_audio=True)
    assert "Do not copy the face, hair, body or held items of <Picture 1>; take the outfit as worn there." in p
    kept = m.parse_edit_analysis(json.dumps(_outfit_analysis(add_parts=["tail"])), 1, request="尾巴也要")
    assert "take the outfit as worn there, including the tail." in m.render_edit_prompt(kept, duration=9.0, has_audio=True)
    declined = m.parse_edit_analysis(json.dumps(_outfit_analysis(drop_parts=["whale tail", "headpiece"])), 1, request="不要尾巴和头饰")
    assert "tail" not in declined["edits"][0]["becomes"]
    p = m.render_edit_prompt(declined, duration=9.0, has_audio=True)
    assert "Do not copy the face, hair, head accessories, tail, body or held items of <Picture 1>" in p
    assert "keeps her own face, hair, body and motion" in p
    for a in (default, kept, declined):
        prompt = m.render_edit_prompt(a, duration=9.0, has_audio=True)
        assert not [i for i in m.validate(prompt, "Ref2VA", _edit_job(m, analysis=a), generic_edit=True) if i["hard"]]


def test_head_swap_fields_and_no_original_head_in_prompt(module):
    import json
    m = module("h3_optimizer")
    a = m.parse_edit_analysis(json.dumps(_head_analysis()), 1, request="把头换成图1的")
    p = m.render_edit_prompt(a, duration=9.0, has_audio=True)
    assert "brown hair" not in p and "white cap" not in p
    assert "grey tank top and white cargo pants" in p and "head accessories" in p
    assert not [i for i in m.validate(p, "Ref2VA", _edit_job(m, analysis=a), generic_edit=True) if i["hard"]]
    declined = m.parse_edit_analysis(json.dumps(_head_analysis(drop_parts=["headband"])), 1, request="图1的头饰不要")
    p = m.render_edit_prompt(declined, duration=9.0, has_audio=True)
    assert "headband" not in p.split("summary:")[0] and "Do not copy the head accessories or headwear of <Picture 1>." in p
    only_cap = m.parse_edit_analysis(json.dumps(_head_analysis(becomes="a white cap", drop_parts=["cap"])), 1, request="帽子不要")
    assert "cap" not in only_cap["edits"][0]["becomes"]
    bad = m.render_edit_prompt(a, duration=9.0, has_audio=True).replace("the dancer in a grey tank top",
                                                                        "the dancer with brown hair in a grey tank top", 1)
    codes = {i["code"]: i["hard"] for i in m.validate(bad, "Ref2VA", _edit_job(m, analysis=a), generic_edit=True)}
    assert codes.get("head_swap_source_head") is False         # 只提示，不打回


def test_text_only_outfit_has_no_picture_negative(module):
    m = module("h3_optimizer")
    a = {"edit_type": "outfit", "edits": [{"kind": "outfit", "target": "the dancer", "target_person": "the dancer",
         "target_part": "grey top", "source": {"picture": None, "text": "a red dress"}, "becomes": "a red dress",
         "keep_from_target": "face, hair, body"}], "unchanged": [], "style": "live action", "notes": ""}
    prompt = m.render_edit_prompt(a, duration=9.0, has_audio=True)
    assert "<Picture" not in prompt
    assert "outfit_negative" not in {i["code"] for i in m.validate(prompt, "Ref2VA", _edit_job(m, analysis=a), generic_edit=True)}


def test_motion_role_is_the_only_hard_request_check(module):
    import json
    m = module("h3_optimizer")
    wrong = dict(_motion_analysis(person=2, setting=("picture", 3)), edit_type="character_swap")
    with pytest.raises(ValueError, match="motion_transfer"):
        m.parse_edit_analysis(json.dumps(wrong), 3, request="动作迁移", video_role="motion")
    m.parse_edit_analysis(json.dumps(_motion_analysis(person=1, setting=("picture", 2))), 3,
                          request="动作迁移，人物用图2，背景用图3", video_role="motion")   # 图号交给模型，代码不二次解读


def test_fallback_is_the_plain_default_and_says_so(module):
    m = module("h3_optimizer")
    assets = [m.Asset("picture", "reference", "", ["x"]) for _ in range(2)] + [m.Asset("video", "editing", "motion (manual)", ["y"])]
    a = m._conservative_fallback_analysis(m.Job("动作迁移，人物用图2", 9.0, assets, "auto", generic_edit=True), 2, "video")
    assert a["edit_type"] == "motion_transfer" and [(e["kind"], e["source"]["picture"]) for e in a["edits"]] == [("character_swap", 1)]
    assert "check the result by hand" in a["notes"]
    edit_assets = [m.Asset("picture", "reference", "", ["x"]), m.Asset("video", "editing", "editing", ["y"])]
    a = m._conservative_fallback_analysis(m.Job("把她的头换成图1的", 9.0, edit_assets, "auto", generic_edit=True), 1, "video")
    assert a["edits"][0]["kind"] == "head_swap"


def test_routing_handles_curly_quotes_and_negation(module):
    m = module("h3_optimizer")
    assert m.infer_edit_role("Don’t do motion transfer; swap her face")[0] == "editing"
    assert m.infer_edit_role("参考原视频的动作，把演员替换成图2穿红衣服的人")[0] == "editing"
    assert m.CJK.search("ｶﾅ") and not m.CJK.search("an Ａ-line silhouette")


# ───────── GPT 6.2 第五轮（新分工）回归 ─────────
def test_motion_keeps_model_position_cue(module):
    m = module("h3_optimizer")
    a = _motion_analysis()
    a["edits"][0]["target"] = "the dancer on the right in a blue jacket"
    a["edits"][0]["performer"] = "the second dancer from the left in the front row"
    p = m.render_edit_prompt(a, duration=10.0, has_audio=True)
    assert "the second dancer from the left in the front row" in p and "blue jacket" not in p
    a["edits"][0]["performer"] = "the dancer in a blue jacket"          # 带了衣着就不用
    assert "the main performer" in m.render_edit_prompt(a, duration=10.0, has_audio=True)


def test_head_keep_original_hat_does_not_also_take_picture_headwear(module):
    import json
    m = module("h3_optimizer")
    a = m.parse_edit_analysis(json.dumps(_head_analysis(becomes="short purple hair under a blue beret",
                                                        keep_original_parts=["head accessories"])), 1, request="保留原来的帽子")
    assert a["edits"][0]["head_options"] == {"picture_accessories": False, "keep_source_headwear": True}
    assert a["edits"][0]["becomes"] == "short purple hair"
    p = m.render_edit_prompt(a, duration=9.0, has_audio=True)
    assert "beret" not in p and "head accessories (" not in p


def test_auto_role_is_decided_by_the_analysis(module):
    m = module("h3_optimizer")
    assets = [m.Asset("picture", "reference", "", ["x"]), m.Asset("video", "editing", "editing (auto)", ["y"])]
    msgs = m._analysis_messages(m.Job("用图1的人，动作跟原视频一样", 9.0, assets, "auto", generic_edit=True))
    text = msgs[-1]["content"][0]["text"]
    assert "not chosen by hand" in text and "keep edit_type in the editing family" not in text


def test_round5_small_fixes(module):
    import json
    m = module("h3_optimizer")
    assert m.infer_style_source("不要保留参考图画风") == "video"
    assert m._norm_parts(["cat ears and tail", "earrings"]) == ["ears", "tail"]
    bad_kind = {**_an([_swap_edit(1)]), "edits": [dict(_swap_edit(1), kind=["outfit", "prop"])]}
    with pytest.raises(ValueError):
        m.parse_edit_analysis(json.dumps(bad_kind), 1, request="x")
    both = {**_an([_swap_edit(1)], "character_swap"), "pictures": [{"picture": 1, "use": "unused"}]}
    with pytest.raises(ValueError, match="marked unused"):
        m.parse_edit_analysis(json.dumps(both), 1, request="x")
    assert not m._BG_CONTRADICTION_RE.search("The original setting does not stay as in <Video 1>.")



# ───────── GPT 6.2 第六轮回归 ─────────
def test_round6_routing_and_manual_edit_role(module):
    import json
    m = module("h3_optimizer")
    assert m.infer_video_role("让图1的人跟着视频跳舞，衣服换成图2的", True)[0] == "motion"
    assert m.infer_video_role("把背景换成图1", True)[0] == "editing"
    with pytest.raises(ValueError, match="do not use motion_transfer"):
        m.parse_edit_analysis(json.dumps(_motion_analysis()), 2, request="x", video_role="editing")


def test_round6_small_fixes(module):
    m = module("h3_optimizer")
    assert m._drop_parts("a fitted red dress with cap sleeves", ["head accessories"]) == "a fitted red dress with cap sleeves"
    assert m.resolve_style_source("保留画风（原视频的），把人物换成图1", "", "video") == "video"
    assert m._request_mentions_prop("手持物也要", "book")
    a = {"edit_type": "background", "edits": [_bg_edit(1)], "unchanged": [], "style": "live", "notes": ""}
    prompt = m.render_edit_prompt(a, duration=9.0, has_audio=False) + "\nNo part of the original setting remains from <Video 1>."
    assert "background_contradiction" not in {i["code"] for i in m.validate(prompt, "Ref2VA", _edit_job(m, analysis=a), generic_edit=True)}


# ───────── GPT 6.2 第七轮回归 ─────────
def test_round7_fixes(module):
    import json
    m = module("h3_optimizer")
    assert m.infer_video_role("只把原视频的衣服换成图1的，动作参考原视频", True)[0] == "editing"
    assert m.infer_video_role("让图1的人跟着视频跳舞，衣服换成图2的", True)[0] == "motion"
    a = _motion_analysis()
    a["edits"][0]["performer"] = "the dancer at the top right of the frame"
    assert m._motion_performer(a["edits"][0]) == "the dancer at the top right of the frame"
    assert "the dancer at the top right of the frame" in m._analysis_english(a)
    p = m.render_edit_prompt(a, duration=10.0, has_audio=True)
    job = _motion_job(m, a, 1)
    assert "performer_lost" not in {i["code"] for i in m.validate(p, "Ref2VA", job, generic_edit=True)}
    lost = p.replace("the dancer at the top right of the frame", "the main performer")
    assert "performer_lost" in {i["code"] for i in m.validate(lost, "Ref2VA", job, generic_edit=True) if i["hard"]}
    assert m._drop_parts("a red cap-sleeve dress", ["head accessories"]) == "a red cap-sleeve dress"
    assert not m._request_mentions_prop("把人物换成图1，用手持运镜", "book")
    hat = m.parse_edit_analysis(json.dumps(_head_analysis(keep_original_parts=["hat"])), 1, request="保留原来的帽子")
    hp = m.render_edit_prompt(hat, duration=9.0, has_audio=True)
    assert "keep_headwear_lost" not in {i["code"] for i in m.validate(hp, "Ref2VA", _edit_job(m, analysis=hat), generic_edit=True)}
    bg = {"edit_type": "background", "edits": [_bg_edit(1)], "unchanged": [], "style": "live", "notes": ""}
    bp = m.render_edit_prompt(bg, duration=9.0, has_audio=False).replace(
        "\n\nretention_analysis:", " Nothing about the performance changes, and the background stays as in <Video 1>.\n\nretention_analysis:", 1)
    assert "background_contradiction" in {i["code"] for i in m.validate(bp, "Ref2VA", _edit_job(m, analysis=bg), generic_edit=True)}
    assets = [m.Asset("picture", "reference", "", ["x"]), m.Asset("video", "editing", "editing", ["y"])]
    fb = m._conservative_fallback_analysis(m.Job("只换背景，用图1", 9.0, assets, "auto", generic_edit=True), 1, "video")
    assert [e["kind"] for e in fb["edits"]] == ["background"]
