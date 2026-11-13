import importlib
import unittest

import cv2
import numpy as np


class ProPainterMaskTests(unittest.TestCase):
    def builder(self, regions, dimensions=(128, 96), mode='auto', dilation=8):
        self.assertIsNotNone(importlib.util.find_spec('propainter_masks'),
                             'ProPainter needs generic, frame-aware masks')
        module = importlib.import_module('propainter_masks')
        return module.MaskBuilder(regions, dimensions, mode, dilation)

    def test_jumping_marks_activate_independently_over_moving_background(self):
        regions = [dict(type='watermark', roi=[8, 8, 50, 36]),
                   dict(type='watermark', roi=[78, 60, 124, 92])]
        yy, xx = np.indices((96, 128))
        frames = []
        for index in range(24):
            gray = ((xx * 7 + yy * 11 + index * 13) % 80 + 35).astype(np.uint8)
            frame = np.repeat(gray[..., None], 3, axis=2)
            moving = (index * 9) % 110
            frame[9:12, moving:moving+8] = 240
            if index < 12:
                cv2.rectangle(frame, (14, 14), (26, 26), (245, 245, 245), 2)
            else:
                cv2.rectangle(frame, (86, 68), (100, 82), (245, 245, 245), 2)
            frames.append(frame)
        builder = self.builder(regions)
        for frame in frames:
            builder.observe(frame)
        builder.finalize()
        first, last = builder.mask(frames[3]), builder.mask(frames[20])
        self.assertEqual(int(first[20, 20]), 255, 'cover holes inside the logo bounding box')
        self.assertFalse(first[60:92, 78:124].any())
        self.assertFalse(last[8:36, 8:50].any())
        self.assertEqual(int(last[75, 93]), 255)
        self.assertFalse(first[36:60].any())

    def test_colored_low_confidence_watermark_falls_back_to_rectangle(self):
        builder = self.builder([dict(type='watermark', roi=[10, 12, 38, 32])])
        frame = np.full((96, 128, 3), 45, dtype=np.uint8)
        frame[18:26, 16:30] = (20, 60, 220)
        for _ in range(6):
            builder.observe(frame)
        builder.finalize()
        mask = builder.mask(frame)
        self.assertTrue(np.all(mask[12:32, 10:38] == 255))
        self.assertFalse(mask[:12].any())

    def test_subtitle_and_rectangle_masks_union_and_dilation_stays_inside_selection(self):
        builder = self.builder([dict(type='subtitle', roi=[5, 6, 30, 24]),
                                dict(type='watermark', roi=[20, 14, 48, 36])], mode='rectangle')
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        builder.observe(frame)
        builder.finalize()
        mask = builder.expanded_mask(builder.mask(frame))
        self.assertTrue(np.all(mask[6:24, 5:30] == 255))
        self.assertTrue(np.all(mask[14:36, 20:48] == 255))
        self.assertEqual(int(mask[5, 10]), 0)
        self.assertEqual(int(mask[36, 30]), 0)

    def test_watermark_on_selection_edge_is_covered_without_changing_outside(self):
        builder = self.builder([dict(type='watermark', roi=[8, 8, 48, 36])])
        frame = np.full((96, 128, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (8, 12), (22, 26), (245, 245, 245), 2)
        for _ in range(8):
            builder.observe(frame)
        builder.finalize()
        mask = builder.expanded_mask(builder.mask(frame))
        self.assertEqual(int(mask[18, 8]), 255)
        self.assertEqual(int(mask[18, 7]), 0)
        self.assertFalse(mask[:8].any())

    def test_brief_watermark_is_active_only_during_its_tenth_of_the_video(self):
        region = dict(type='watermark', roi=[8, 8, 50, 36])
        builder = self.builder([region])
        yy, xx = np.indices((96, 128))
        frames = []
        for index in range(100):
            gray = ((xx * 5 + yy * 7 + index * 17) % 80 + 30).astype(np.uint8)
            frame = np.repeat(gray[..., None], 3, axis=2)
            moving = (index * 9) % 110
            frame[9:12, moving:moving+7] = 235
            if 40 <= index < 50:
                cv2.rectangle(frame, (14, 14), (26, 26), (245, 245, 245), 2)
            frames.append(frame)
            builder.observe(frame)
        builder.finalize()
        self.assertIsNotNone(builder.stable[0], 'a brief recurring mark needs an activation core')
        self.assertFalse(builder.mask(frames[20]).any())
        self.assertEqual(int(builder.mask(frames[45])[20, 20]), 255)
        self.assertFalse(builder.mask(frames[70]).any())

    def test_four_jumping_positions_each_get_their_own_quarter_video_activation(self):
        regions = [dict(type='watermark', roi=roi) for roi in
                   ([4, 4, 44, 36], [80, 4, 124, 36], [4, 60, 44, 92], [80, 60, 124, 92])]
        builder = self.builder(regions)
        yy, xx = np.indices((96, 128))
        frames = []
        for index in range(80):
            gray = ((xx * 7 + yy * 11 + index * 13) % 90 + 25).astype(np.uint8)
            frame = np.repeat(gray[..., None], 3, axis=2)
            x1, y1, _, _ = regions[index // 20]['roi']
            cv2.rectangle(frame, (x1+8, y1+8), (x1+20, y1+20), (245, 245, 245), 2)
            frames.append(frame)
            builder.observe(frame)
        builder.finalize()
        self.assertTrue(all(core is not None for core in builder.stable))
        for active, index in enumerate((5, 25, 45, 65)):
            mask = builder.mask(frames[index])
            for position, region in enumerate(regions):
                x1, y1, x2, y2 = region['roi']
                if position == active:
                    self.assertEqual(int(mask[y1+14, x1+14]), 255)
                else:
                    self.assertFalse(mask[y1:y2, x1:x2].any())
