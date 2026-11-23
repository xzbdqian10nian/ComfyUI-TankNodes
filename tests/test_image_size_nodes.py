import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("tank_image_size", Path(__file__).resolve().parents[1] / "image_size_nodes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class _FakeImage:
    def __init__(self, height, width, batch=1, channels=3):
        self.ndim = 4
        self.shape = (batch, height, width, channels)


class ImageSizeTests(unittest.TestCase):
    def test_screenshot_dimensions(self):
        self.assertEqual(module.image_size("3:4", 2, 32), (1248, 1664))

    def test_square_and_portrait(self):
        self.assertEqual(module.image_size("1:1", 4, 32), (2048, 2048))
        self.assertEqual(module.image_size("9:16", 2, 32), tuple(reversed(module.image_size("16:9", 2, 32))))

    def test_all_ratios_and_limits(self):
        self.assertEqual(module.ASPECT_RATIOS[0], module.FOLLOW_IMAGE_1)
        for ratio in module.RATIO_PRESETS:
            for mp in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0):
                for multiple in module.ALIGNMENTS:
                    with self.subTest(ratio=ratio, mp=mp, multiple=multiple):
                        w, h = module.image_size(ratio, mp, multiple)
                        self.assertEqual(w % multiple, 0)
                        self.assertEqual(h % multiple, 0)
                        rw, rh = map(int, ratio.split(":"))
                        self.assertLessEqual(abs(w - math.sqrt(mp * 1024**2 * rw / rh)), multiple / 2 + 1e-9)
                        self.assertLessEqual(abs(h - math.sqrt(mp * 1024**2 * rh / rw)), multiple / 2 + 1e-9)

    def test_follow_image_1_portrait_and_landscape(self):
        # Portrait image 1600x900 (9:16)
        portrait = _FakeImage(1600, 900)
        self.assertEqual(module.image_size(module.FOLLOW_IMAGE_1, 2, 32, image=portrait), module.image_size("9:16", 2, 32))

        # Landscape image 1080x1920 (16:9)
        landscape = _FakeImage(1080, 1920)
        self.assertEqual(module.image_size(module.FOLLOW_IMAGE_1, 2, 32, image=landscape), module.image_size("16:9", 2, 32))

        latent = {"samples": object(), "downscale_ratio_spacial": 8}

        class Empty:
            def generate(self, width, height, batch_size):
                return (latent,)

        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(EmptyLatentImage=Empty)}):
            out = module.TankImageSize().generate(module.FOLLOW_IMAGE_1, 2.0, 32, 1, image=landscape)
        self.assertEqual(out[1:3], module.image_size("16:9", 2.0, 32))

    def test_follow_image_1_errors(self):
        with self.assertRaises(ValueError):
            module.image_size(module.FOLLOW_IMAGE_1, 2, 32, image=None)
        with self.assertRaises(ValueError):
            module.TankImageSize().generate(aspect_ratio=module.FOLLOW_IMAGE_1)
        bad = types.SimpleNamespace(ndim=3, shape=(640, 480, 3))
        with self.assertRaises(ValueError):
            module.image_size(module.FOLLOW_IMAGE_1, 2, 32, image=bad)
        zero = _FakeImage(0, 480)
        with self.assertRaises(ValueError):
            module.image_size(module.FOLLOW_IMAGE_1, 2, 32, image=zero)

    def test_invalid_boundary_inputs(self):
        for mp in (0, -1, 33, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                module.image_size("1:1", mp, 32)
        for ratio in ("0:1", "custom", "1:0"):
            with self.assertRaises(ValueError):
                module.image_size(ratio, 2, 32)
        with self.assertRaises(ValueError):
            module.image_size("1:1", 2, 8)
        for batch in (0, 65, 1.5, True):
            with self.assertRaises(ValueError):
                module.TankImageSize().generate(batch_size=batch)

    def test_delegates_to_native_and_preserves_metadata(self):
        calls = []
        latent = {"samples": object(), "downscale_ratio_spacial": 8}

        class Empty:
            def generate(self, width, height, batch_size):
                calls.append((width, height, batch_size))
                return (latent,)

        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(EmptyLatentImage=Empty)}):
            output = module.TankImageSize().generate("3:4", 2, 32, 3)
        self.assertIs(output[0], latent)
        self.assertEqual(calls, [(1248, 1664, 3)])
        self.assertEqual(output[1:4], (1248, 1664, 2076672))
        self.assertEqual(output[4], 2076672 / 1024**2)

    def test_not_output_node(self):
        self.assertFalse(getattr(module.TankImageSize, "OUTPUT_NODE", False))
        self.assertEqual(module.TankImageSize.RETURN_TYPES, ("LATENT", "INT", "INT", "INT", "FLOAT"))


if __name__ == "__main__":
    unittest.main()
