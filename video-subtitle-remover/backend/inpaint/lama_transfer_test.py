import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_INFERENCE_DEPS = all(importlib.util.find_spec(name) for name in ('torch', 'numpy', 'cv2', 'PIL'))
if HAS_INFERENCE_DEPS:
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.inpaint.lama_inpaint import LamaInpaint


class ProbeModel:
    def __init__(self):
        self.inputs = []

    def eval(self):
        return self

    def __call__(self, image, mask):
        self.inputs.append((image.clone(), mask.clone()))
        # Read across the padded edges and exercise output clipping and truncation.
        return torch.roll(image, shifts=(-2, -3), dims=(2, 3)) * 1.4 - 0.2 + mask * 0.125


@unittest.skipUnless(HAS_INFERENCE_DEPS, 'requires the LaMa inference dependencies')
class LamaTransferTest(unittest.TestCase):
    def make_painter(self, batch_size, **options):
        with patch.object(torch.jit, 'load', return_value=ProbeModel()):
            return LamaInpaint(torch.device('cpu'), batch_size=batch_size, **options)

    def test_compact_input_and_output_match_default(self):
        rng = np.random.default_rng(321)
        for h, w, count, size in ((13, 19, 1, 4), (13, 21, 8, 3), (16, 24, 4, 4), (1, 3, 2, 1)):
            with self.subTest(height=h, width=w, count=count, batch_size=size):
                frames = [
                    rng.integers(256, size=(h, w * 2, 3), dtype=np.uint8)[:, ::2] for _ in range(count)
                ]
                masks = [rng.choice(np.array([0, 1, 7, 255], np.uint8), (h, w)) for _ in frames]
                masks[::2] = [mask[:, :, None] for mask in masks[::2]]
                default = self.make_painter(size)
                compact = self.make_painter(size, compact_transfer=True)
                expected = default._inpaint_batch(frames, masks)
                with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(
                    torch.cuda, 'empty_cache', side_effect=AssertionError('flushed the allocator')
                ):
                    actual = compact._inpaint_batch(frames, masks)

                self.assertEqual(len(actual), count)
                self.assertEqual([img.shape[0] for img, _ in compact.model.inputs],
                                 [min(size, count - i) for i in range(0, count, size)])
                for (img, mask), (ref_img, ref_mask) in zip(compact.model.inputs, default.model.inputs):
                    self.assertEqual(img.dtype, torch.float32)
                    self.assertEqual(mask.dtype, torch.int64)
                    self.assertTrue(img.is_contiguous())
                    torch.testing.assert_close(img, ref_img, rtol=0, atol=0)
                    torch.testing.assert_close(mask, ref_mask, rtol=0, atol=0)
                for got, want in zip(actual, expected):
                    self.assertEqual(got.shape, (h, w, 3))
                    self.assertEqual(got.dtype, np.uint8)
                    np.testing.assert_array_equal(got, want)

    def test_float_arrays_and_pil_keep_their_input_semantics(self):
        frames = [np.linspace(-21.5, 377.25, 315, dtype=np.float32).reshape(7, 15, 3)] * 2
        masks = [np.linspace(-1.5, 0.75, 105, dtype=np.float32).reshape(7, 15)] * 2
        default = self.make_painter(2)
        compact = self.make_painter(2, compact_transfer=True)
        for got, want in zip(compact._inpaint_batch(frames, masks), default._inpaint_batch(frames, masks)):
            np.testing.assert_array_equal(got, want)
        image, mask = Image.fromarray(frames[0].astype(np.uint8)), Image.fromarray(masks[0])
        np.testing.assert_array_equal(compact.inpaint(image, mask), default.inpaint(image, mask))


if __name__ == '__main__':
    unittest.main()
