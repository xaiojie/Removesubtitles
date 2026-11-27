import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'php/inference'), str(ROOT/'CLEAR-web')]


class RevealTests(unittest.TestCase):
    def test_only_copy_from_an_aligned_unobstructed_frame(self):
        from lite_reveal import VisibleCardBackground

        rng = np.random.default_rng(144)
        scene = cv2.GaussianBlur(rng.integers(5, 205, (180, 370, 3), dtype=np.uint8), (3, 3), 0)
        before_cut = np.roll(scene, 53, axis=0)[:, ::-1].copy()
        target = scene.copy()
        roi = (24, 27, 344, 160)
        mask = np.zeros((133, 320), np.uint8)
        mask[32:107, 22:296] = 255
        target[59:134, 46:320] = 247
        boxes = [(30, 39, 220, 53), (30, 64, 220, 78), (30, 89, 220, 103)]

        def read(index):
            if index < 12:
                return before_cut
            if index < 30:
                return target
            return scene

        tool = VisibleCardBackground(read, lambda frame: (), fps=12, total=45)
        repaired = target.copy()
        residual = tool.recover(repaired, 12, roi, mask, boxes)
        self.assertIsNotNone(residual)
        self.assertFalse(residual.any())
        np.testing.assert_allclose(repaired, scene, atol=1)

        wrong_shot = VisibleCardBackground(lambda index: before_cut, lambda frame: (), fps=12, total=45)
        untouched = target.copy()
        self.assertIsNone(wrong_shot.recover(untouched, 12, roi, mask, boxes))
        np.testing.assert_array_equal(untouched, target)

        other_subtitles = VisibleCardBackground(read, lambda frame: [(70, 75, 160, 95)], fps=12, total=45)
        self.assertIsNone(other_subtitles.recover(untouched, 12, roi, mask, boxes))
        np.testing.assert_array_equal(untouched, target)


if __name__ == '__main__':
    unittest.main()
