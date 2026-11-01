"""Offline checks for the H3 prompt optimizer (no model required)."""


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
