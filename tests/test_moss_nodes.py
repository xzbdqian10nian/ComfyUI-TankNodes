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
