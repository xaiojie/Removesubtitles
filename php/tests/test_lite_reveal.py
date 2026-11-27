import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'php/inference'), str(ROOT/'CLEAR-web')]


class RevealTests(unittest.TestCase):
    def test_stationary_source_can_restore_a_detail_hidden_by_text(self):
        from lite import LiteOptions, restore_joint
        from lite_reveal import StaticSubtitleBackground

        rng = np.random.default_rng(14)
        scene = cv2.GaussianBlur(rng.integers(24, 120, (96, 400, 3), dtype=np.uint8), (7, 7), 0)
        cv2.circle(scene, (219, 62), 9, (139, 138, 138), 2)
        captioned = scene.copy()
        cv2.putText(captioned, 'CHECK LEVEL', (91, 66), cv2.FONT_HERSHEY_DUPLEX, 1, (15, 15, 15), 5)
        cv2.putText(captioned, 'CHECK LEVEL', (91, 66), cv2.FONT_HERSHEY_DUPLEX, 1, (248, 248, 248), 2)
        roi = (0, 8, 400, 92)
        box = (82, 33, 319, 81)
        from lite_profile import BandProfile
        profile = {roi: BandProfile(400, 84, [], None)}
        class Inpainter:
            device = 'cpu'
            def _inpaint_batch(self, images, masks):
                return [np.full_like(im, 206) for im in images]
        options = LiteOptions(mask_mode='tracked_cards', box_padding=8, layout_profile=True, recover_static_text=True)
        frames = [scene.copy(), scene.copy(), captioned.copy()]
        sources = {roi: StaticSubtitleBackground(75, lambda crop: False)}
        restored = restore_joint(frames, [(), (), (box,)], [roi], Inpainter(), options,
                                 profiles=profile, tracks={}, static_sources=sources)
        np.testing.assert_array_equal(restored[-1], scene)

        l, t, r, b = roi
        area = scene[t:b, l:r]
        mask = np.zeros(area.shape[:2], np.uint8)
        mask[box[1]-t:box[3]-t, box[0]-l:box[2]-l] = 255
        local = [(box[0]-l, box[1]-t, box[2]-l, box[3]-t)]
        for observed, current, detected in (
            (scene, captioned, True),
            (captioned, captioned, False),
            (scene, cv2.add(captioned, np.full_like(scene, 31)), False),
        ):
            tool = StaticSubtitleBackground(75, lambda crop: detected)
            for i in (0, 1):
                tool.observe(observed[t:b, l:r], i)
            frame = current[t:b, l:r].copy()
            self.assertFalse(tool.restore(frame, 2, mask, local))
            np.testing.assert_array_equal(frame, current[t:b, l:r])

    def test_narrow_subtitle_band_keeps_its_vertical_resolution_for_detection(self):
        from lite import LiteOptions, detect
        sizes = []
        class Detector:
            def predict(self, image):
                sizes.append(image.shape[:2])
                return [{'dt_polys': []}]
        frame = np.zeros((560, 1000, 3), dtype=np.uint8)
        opts = LiteOptions(detector_max_side=720, native_thin_ocr=True)
        detect(frame, [(30, 455, 928, 542), (30, 80, 928, 350)], Detector(), opts)
        self.assertEqual(sizes, [(87, 898), (216, 720)])

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

        # Recover the next frame even if its descriptor fit is lost for a single frame.
        motion = np.float32([[1, 0, 1.3], [0, 1, .6]])
        actual = cv2.warpAffine(scene, motion, (370, 180), borderMode=cv2.BORDER_REPLICATE)
        covered = actual.copy()
        covered[59:134, 46:320] = 247
        with patch('lite_reveal.registration', return_value=None):
            residual = tool.recover(covered, 13, roi, mask, boxes)
            self.assertIsNotNone(residual)
            self.assertLess(cv2.absdiff(covered[60:133, 47:319], actual[60:133, 47:319]).mean(), 2)
            cut = before_cut.copy()
            cut[59:134, 46:320] = 247
            self.assertIsNone(tool.recover(cut, 14, roi, mask, boxes))

        wrong_shot = VisibleCardBackground(lambda index: before_cut, lambda frame: (), fps=12, total=45)
        untouched = target.copy()
        self.assertIsNone(wrong_shot.recover(untouched, 12, roi, mask, boxes))
        np.testing.assert_array_equal(untouched, target)

        other_subtitles = VisibleCardBackground(read, lambda frame: [(70, 75, 160, 95)], fps=12, total=45)
        self.assertIsNone(other_subtitles.recover(untouched, 12, roi, mask, boxes))
        np.testing.assert_array_equal(untouched, target)


if __name__ == '__main__':
    unittest.main()
