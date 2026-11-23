import json
import types
import torch
import torch.nn as nn


def test_multi_angle_camera_dialects(plugin, monkeypatch, tmp_path):
    cls = plugin.NODE_CLASS_MAPPINGS["TankMultiAngleCamera"]
    node = cls()
    ref = torch.rand((1, 80, 120, 3), dtype=torch.float32)

    import folder_paths
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path), raising=False)

    out_v6 = node.compile(90, 0, "medium", "structured_v6", reference_image=ref)
    assert "90°" in out_v6["result"][0]
    assert json.loads(out_v6["result"][2])["horizontal_angle"] == 90
    assert out_v6["result"][3].shape == (1, 80, 120, 3)
    assert len(out_v6["ui"]["tank_cam_ref"]) == 1

    out_qma = node.compile(180, 30, "wide", "qma_lora")
    assert out_qma["result"][0] == "<qma> back view elevated shot wide shot"
    assert node.compile(90, 0, "medium_wide", "qma_lora")["result"][0] == "<qma> right side view eye-level shot wide shot"
    assert node.compile(90, 0, "medium", "qma_lora")["result"][0] == "<qma> right side view eye-level shot medium shot"
    assert node.compile(90, 0, "medium_close_up", "qma_lora")["result"][0] == "<qma> right side view eye-level shot close-up"
    assert node.compile(90, 0, "close_up", "qma_lora")["result"][0] == "<qma> right side view eye-level shot close-up"


def test_panorama_setup_auto_modes(plugin):
    cls = plugin.NODE_CLASS_MAPPINGS["TankPanoramaSetup"]
    node = cls()
    pos_d, neg_d, w, h, canvas_d, mask_d = node.setup("auto", "structured_equirect", 2880, True, 0.21, 24, "Traditional courtyard")
    assert (w, h) == (2880, 1440)
    assert "UNIQUE LANDMARK" in pos_d
    assert canvas_d.shape == (1, 1440, 2880, 3)
    assert float(mask_d.min()) == 1.0

    ref = torch.full((1, 360, 540, 3), 0.7, dtype=torch.float32)
    pos_e, _, w2, h2, canvas_e, mask_e = node.setup("auto", "concise_qwen_edit", 2048, True, 0.21, 24, "", reference_image=ref)
    assert (w2, h2) == (2048, 1024)
    assert canvas_e.shape == (1, 1024, 2048, 3)
    assert float(mask_e[:, 512, 1024]) == 0.0
    assert float(mask_e[:, 10, 10]) == 1.0


def test_seamless_patch_preserves_conv_padding(plugin):
    cls = plugin.NODE_CLASS_MAPPINGS["TankPanoramaSeamlessPatch"]
    conv = nn.Conv2d(4, 4, kernel_size=3, padding=(1, 1))

    class DummyModel:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=nn.Sequential(conv))
            self.patches = {}

        def clone(self):
            c = DummyModel()
            c.model = self.model
            return c

        def add_object_patch(self, path, fn):
            self.patches[path] = fn

    dm = DummyModel()
    patched, vae_out = cls().patch(dm, enable=True, vae=None)
    assert vae_out is None
    assert "diffusion_model.0.forward" in patched.patches
    fn = patched.patches["diffusion_model.0.forward"]
    y = fn(torch.randn(1, 4, 16, 32))
    assert y.shape == (1, 4, 16, 32)
    assert conv.padding == (1, 1)


def test_panorama_viewer_and_slicer(plugin, monkeypatch, tmp_path):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path), raising=False)

    cls = plugin.NODE_CLASS_MAPPINGS["TankPanoramaViewer"]
    pano = torch.rand((1, 256, 512, 3), dtype=torch.float32)
    res4 = cls().preview_and_slice(pano, 55.0, "4_cardinal_views", 58.0, 512, 512, 0.0, 0.0)
    assert res4["result"][0].shape == (1, 256, 512, 3)
    assert res4["result"][1].shape == (4, 512, 512, 3)
    assert res4["ui"]["tank_pano"][0]["vfov"] == 55.0

    res_none = cls().preview_and_slice(pano, 55.0, "none", 58.0, 512, 512, 0.0, 0.0)
    assert res_none["result"][1].shape == (1, 256, 512, 3)


def test_panorama_slicer_wraps_longitude_seam_smoothly(plugin, monkeypatch, tmp_path):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path), raising=False)

    cls = plugin.NODE_CLASS_MAPPINGS["TankPanoramaViewer"]
    # Only col 0 (u=0 / wrapped u=1) is bright; all remaining columns are 0.
    # Just left of yaw=180° center (x=31), u approaches 1.0 and must interpolate
    # toward the appended wrap column (src_wrap[..., ew]), not stop before col ew-1.
    ew = 128
    pano = torch.zeros((1, 64, ew, 3), dtype=torch.float32)
    pano[:, :, 0, :] = 1.0

    sliced = cls().preview_and_slice(pano, 55.0, "custom", 60.0, 64, 64, 180.0, 0.0)["result"][1]
    left_of_seam = float(sliced[0, 32, 31, 0])
    right_of_seam = float(sliced[0, 32, 32, 0])
    assert left_of_seam > 0.7, f"expected wrap interpolation into col ew, got {left_of_seam}"
    assert right_of_seam > 0.7, f"expected sample near col 0, got {right_of_seam}"
