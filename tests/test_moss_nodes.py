import json
import importlib.util, pathlib

spec = importlib.util.spec_from_file_location("moss_nodes", pathlib.Path(__file__).resolve().parents[1] / "moss_nodes.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_target_size_matches_toolkit_rule():
    assert m.target_size(698, 988, 1024) == (688, 976)      # smaller page: never upscaled, floored to 16
    assert m.target_size(2000, 3000, 1024) == (832, 1248)    # area ≈ 1024², native aspect


def test_parse_fenced_and_wrapped():
    ans = 'Here you go:\n```json\n[{"bbox_2d":[700,40,820,300],"ja":"メイド姿の少女が","text":"A girl in a maid outfit"}]\n```'
    assert m.parse_detections(ans)[0]["ja"] == "メイド姿の少女が"
    assert m.parse_detections('{"items":[{"bbox":[1,2,3,4],"text":"x"}]}')[0]["text"] == "x"


def test_layout_relative_coords_and_uppercase():
    dets = [{"bbox_2d": [700, 40, 820, 300], "ja": "訪ねてきた", "text": "A girl came to my door."}]
    layout, kept = m.build_layout(dets, 698, 988, 1024)
    d = json.loads(layout)
    assert d["summary"]["caption"] == "comic, english text"
    item = d["items"][0]
    assert item["text"] == "A GIRL CAME TO MY DOOR."
    w, h = m.target_size(698, 988, 1024)
    x0, y0 = 0.7 * 698 * w / 698, 0.04 * 988 * h / 988
    assert item["bbox"][:2] == [round(y0 - h / 2), round(x0 - w / 2)]
    assert d["bbox_coord_space"] == "abs_centralized" and d["bbox_format"] == "yxyx"


def test_korean_detected_and_kept_case():
    layout, _ = m.build_layout([{"bbox_2d": [10, 10, 200, 200], "text": "뭐 하는 거야?"}], 800, 1200, 1024)
    d = json.loads(layout)
    assert d["summary"]["caption"] == "comic, korean text" and d["items"][0]["text"] == "뭐 하는 거야?"


def test_pixel_mode_and_tiny_boxes_dropped():
    dets = [{"bbox": [1100, 50, 1300, 400], "text": "hi"}, {"bbox": [5, 5, 7, 7], "text": "dot"}]
    layout, kept = m.build_layout(dets, 1400, 2000, 1024)
    assert len(json.loads(layout)["items"]) == 1


def test_prompt_node():
    sys_p, p = m.TankMangaTranslatePrompt().build("korean", "光太郎 = 코타로")
    assert "Korean" in sys_p and "bbox_2d" in sys_p and "코타로" in p


def test_columns_of_one_balloon_merge_right_to_left():
    dets = [{"bbox_2d": [920, 857, 975, 959], "ja": "お願いします", "text": "Please..."},
            {"bbox_2d": [855, 857, 905, 959], "ja": "どんな命令にも従うので", "text": "I'll follow any order."},
            {"bbox_2d": [37, 857, 97, 959], "ja": "メイドとして", "text": "Hire me?"},
            {"bbox_2d": [37, 959, 97, 999], "ja": "", "text": ""}]
    layout, kept = m.build_layout(dets, 698, 988, 1024)
    texts = [i["text"] for i in json.loads(layout)["items"]]
    assert len(texts) == 2 and "PLEASE... I'LL FOLLOW ANY ORDER." in texts


def test_wide_boxes_never_merge():
    dets = [{"bbox_2d": [100, 100, 500, 160], "text": "a"}, {"bbox_2d": [100, 170, 500, 230], "text": "b"}]
    assert len(json.loads(m.build_layout(dets, 1000, 1000, 1024)[0])["items"]) == 2


def test_repeats_and_contained_entries():
    dets = [{"bbox_2d": [816, 84, 935, 293], "ja": "メイド姿の少女が訪ねてきた", "text": "A girl in maid clothes came."},
            {"bbox_2d": [829, 847, 921, 951], "ja": "お願いします…どんな命令にも従うので", "text": "Please... I'll follow any command."},
            {"bbox_2d": [724, 878, 777, 947], "ja": "お願いします…", "text": "I beg you..."},
            {"bbox_2d": [724, 878, 777, 947], "ja": "お願いします…", "text": "I beg you..."},
            {"bbox_2d": [20, 874, 147, 953], "ja": "メイドとして雇ってもらえます…", "text": "Would you hire me as a maid...?"},
            {"bbox_2d": [20, 874, 147, 953], "ja": "メイドとして雇ってもらえます…", "text": "Would you hire me as a maid...?"}]
    items = json.loads(m.build_layout(dets, 698, 988, 1024)[0])["items"]
    texts = sorted(i["text"] for i in items)
    assert texts == ["A GIRL IN MAID CLOTHES CAME.", "PLEASE... I'LL FOLLOW ANY COMMAND.", "WOULD YOU HIRE ME AS A MAID...?"]
    right = next(i for i in items if i["text"].startswith("PLEASE"))
    w, h = m.target_size(698, 988, 1024)
    assert right["bbox"][1] == round(0.724 * w - w / 2)       # the folded column widened the balloon box to the left


def test_chinese_detected_not_uppercased():
    layout, _ = m.build_layout([{"bbox_2d": [10, 10, 300, 300], "text": "穿女仆装的少女来访了。"}], 800, 1200, 1024)
    d = json.loads(layout)
    assert d["summary"]["caption"] == "comic, chinese text" and d["items"][0]["text"].replace(chr(10), "") == "穿女仆装的少女来访了。"
    sys_p, _ = m.TankMangaTranslatePrompt().build("chinese")
    assert "Simplified Chinese" in sys_p


def test_chinese_prewrap_and_kinsoku():
    out = m.wrap_cjk("拜托您了……无论什么命令我都会听从", 100, 150)
    rows = out.split("\n")
    assert "".join(rows) == "拜托您了……无论什么命令我都会听从"
    assert all(r[0] not in m._NO_LINE_START for r in rows)
    layout, _ = m.build_layout([{"bbox_2d": [815, 57, 958, 295], "text": "一位身着女仆装束的少女前来拜访"}], 698, 988, 1024)
    assert json.loads(layout)["items"][0]["text"].replace("\n", "") == "一位身着女仆装束的少女前来拜访"
    assert "\n" in json.loads(layout)["items"][0]["text"]
    en, _ = m.build_layout([{"bbox_2d": [10, 10, 300, 300], "text": "Hello there"}], 800, 1200, 1024)
    assert "\n" not in json.loads(en)["items"][0]["text"]


def test_colorize_layout_per_character():
    sheet = m.parse_sheet("A | wavy side ponytails | light brown hair, green eyes\nB | short black bob | black hair, blue eyes\n# note")
    assert [e["name"] for e in sheet] == ["A", "B"] and sheet[0]["colors"] == "light brown hair, green eyes"
    dets = [{"bbox_2d": [600, 50, 950, 450], "who": "A"}, {"bbox_2d": [100, 500, 400, 950], "who": "b"},
            {"bbox_2d": [600, 50, 950, 450], "who": "A"}, {"bbox_2d": [10, 10, 30, 30], "who": "unknown"}]
    layout, kept = m.build_colorize_layout(dets, sheet, 1000, 1400, 1024, "comic, school uniform")
    d = json.loads(layout)
    assert d["summary"]["caption"] == "comic, school uniform" and len(d["items"]) == 2
    assert d["items"][0]["caption"] == "light brown hair, green eyes" and d["items"][0]["position"] == "top right"
    assert d["items"][1]["caption"] == "black hair, blue eyes" and d["items"][1]["position"] == "bottom left"
    assert d["bbox_format"] == "yxyx"
    sys_p, p, sheet_out = m.TankMangaCharacterPrompt().build("A | side ponytails | brown hair")
    assert "A | side ponytails" in p and "who" in sys_p and sheet_out.startswith("A |")


def test_translate_prompt_meaning_names_stutters():
    sys_p, _ = m.TankMangaTranslatePrompt().build("chinese")
    assert "never the sound" in sys_p and "真冬" in sys_p and "stutter" in sys_p and "{" not in sys_p.split("JSON array")[0]


def test_colorize_layout_position_only_and_order():
    sheet = m.parse_sheet("A | x | red hair\nB | y | blue hair")
    dets = [{"bbox_2d": [100, 600, 400, 950], "who": "B"}, {"bbox_2d": [600, 50, 950, 450], "who": "A"}]
    d = json.loads(m.build_colorize_layout(dets, sheet, 1000, 1400, 1024, add_bbox=False)[0])
    assert "bbox_format" not in d and all("bbox" not in i for i in d["items"])
    assert [i["caption"] for i in d["items"]] == ["red hair", "blue hair"]       # reading order: top row first
