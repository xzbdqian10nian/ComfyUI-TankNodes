import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


def test_native_equivalence_and_format_adaptations():
    comfy = Path("/root/ComfyUI")
    if comfy.is_dir() and str(comfy) not in sys.path:
        sys.path.insert(0, str(comfy))
    nodes = pytest.importorskip("nodes")
    import torch
    import comfy.sample
    import comfy.latent_formats as formats
    import comfy.model_management as management

    spec = importlib.util.spec_from_file_location("tank_image_size", Path(__file__).resolve().parents[1] / "image_size_nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Model:
        def __init__(self, latent_format):
            self.latent_format = latent_format

        def get_model_object(self, name):
            assert name == "latent_format"
            return self.latent_format

    with patch.object(management, "intermediate_device", return_value=torch.device("cpu")):
        for ratio, mp, multiple, batch in [("3:4", 2, 32, 1), ("16:9", 0.5, 16, 2), ("1:1", 4, 64, 1)]:
            latent, width, height, pixels, actual = module.TankImageSize().generate(ratio, mp, multiple, batch)
            native = nodes.EmptyLatentImage().generate(width, height, batch)[0]
            assert latent.keys() == native.keys()
            assert latent["downscale_ratio_spacial"] == 8
            assert torch.equal(latent["samples"], native["samples"])
            assert latent["samples"].dtype == native["samples"].dtype
            assert pixels == width * height and actual == pixels / 1024**2
            for name, cls, channels, stride, temporal in [
                ("MOSS/FLUX3/Flux2/Ideogram4", formats.Flux2, 128, 16, False),
                ("Qwen2.1", formats.QwenImage21, 64, 16, False),
                ("Z-Image", formats.Flux, 16, 8, False),
                ("Krea2", formats.Wan21, 16, 8, True),
            ]:
                result = comfy.sample.fix_empty_latent_channels(Model(cls()), latent["samples"], latent["downscale_ratio_spacial"])
                expected = (batch, channels, 1, height // stride, width // stride) if temporal else (batch, channels, height // stride, width // stride)
                assert tuple(result.shape) == expected, (name, result.shape, expected)
                assert not torch.count_nonzero(result)
                print("ADAPT_OK", name, width, height, tuple(result.shape))
    print("NATIVE_EQUIVALENCE_AND_12_FORMAT_ADAPTATIONS_PASS; CPU only; no models loaded or GPU generation")


if __name__ == "__main__":
    comfy = Path("/root/ComfyUI")
    if comfy.is_dir() and str(comfy) not in sys.path:
        sys.path.insert(0, str(comfy))
    test_native_equivalence_and_format_adaptations()
