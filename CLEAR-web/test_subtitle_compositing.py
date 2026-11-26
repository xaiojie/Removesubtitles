import unittest
from unittest.mock import patch

import cv2
import numpy as np


class SubtitleCompositingTests(unittest.TestCase):
    def test_dark_scene_pastel_text_and_white_prefix_share_guidance(self):
        from subtitle_compositing import tinted_caption_guidance

        crop = np.full((70, 380, 3), 18, np.uint8)
        cv2.putText(crop, 'END', (20, 46), cv2.FONT_HERSHEY_SIMPLEX, .9, (210, 210, 210), 2)
        cv2.putText(crop, '6.30 USD', (94, 46), cv2.FONT_HERSHEY_SIMPLEX, .9, (166, 160, 216), 2)
        mask = tinted_caption_guidance(crop)
        self.assertIsNotNone(mask)
        self.assertTrue(np.all(mask[crop.max(axis=2) > 128] == 255))
        self.assertFalse(mask[:, 265:].any())
        cv2.rectangle(crop, (0, 0), (379, 69), (18, 18, 18), -1)
        cv2.circle(crop, (155, 34), 12, (166, 160, 216), -1)
        self.assertIsNone(tinted_caption_guidance(crop))

    def test_clipped_outline_cannot_anchor_a_shadow_at_the_frame_edge(self):
        from subtitle_compositing import composite_caption
        # A known clean background with a diagonal gradient and an independently drawn caption.
        y, x = np.indices((80, 140))
        clean = np.repeat((110 + x//5 + y//8).astype(np.uint8)[:, :, None], 3, axis=2)
        marked = clean.copy()
        cv2.putText(marked, 'OUT', (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (5,)*3, 11, cv2.LINE_AA)
        cv2.putText(marked, 'OUT', (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (245,)*3, 3, cv2.LINE_AA)
        mask = cv2.dilate(np.uint8(np.any(marked != clean, axis=2))*255,
                          np.ones((13, 13), np.uint8))
        for turns in range(4):
            with self.subTest(turns=turns):
                original, restored, selected = (np.rot90(a, turns).copy() for a in (marked, clean, mask))
                result = composite_caption(original, restored, selected)
                np.testing.assert_array_equal(result[selected == 0], original[selected == 0])
                self.assertLess(np.abs(result.astype(int)-restored).max(), 5)

    def test_thin_outlines_on_pale_scenery_keep_leading_letters_dots_and_both_lines(self):
        from subtitle_compositing import caption_mask
        for boundary in (95, 215, 310):
            with self.subTest(background_edge=boundary):
                frame = np.full((188, 580, 3), (221, 230, 237), np.uint8)
                frame[:, boundary:] = (48, 68, 113)
                lettering = np.zeros(frame.shape[:2], np.uint8)
                for word, origin in [('A', (40, 72)), ('pale coat...', (99, 72)),
                                     ('end.', (220, 142))]:
                    cv2.putText(frame, word, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (45, 50, 57), 5, cv2.LINE_AA)
                    cv2.putText(frame, word, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (248, 249, 251), 3, cv2.LINE_AA)
                    cv2.putText(lettering, word, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                255, 5, cv2.LINE_AA)
                mask = caption_mask(frame, (15, 12, 565, 176))
                self.assertTrue(np.all(mask[lettering > 16] == 255))
                self.assertFalse(mask[:, :15].any())
                self.assertFalse(mask[100:115, :175].any())
                self.assertFalse(mask[:, 450:].any())

    def test_close_vertical_strokes_and_separate_ellipsis_on_light_scene(self):
        from subtitle_compositing import caption_mask
        frame = np.full((120, 390, 3), 230, np.uint8)
        ink = np.zeros(frame.shape[:2], np.uint8)
        for x in range(87, 143, 7):
            cv2.rectangle(frame, (x-1, 46), (x+5, 67), (55, 55, 55), -1)
            cv2.rectangle(frame, (x, 47), (x+4, 66), (250, 250, 250), -1)
            cv2.rectangle(ink, (x, 47), (x+4, 66), 255, -1)
        for x in (170, 180, 190):
            cv2.circle(frame, (x, 65), 3, (45, 45, 45), -1)
            cv2.circle(frame, (x, 65), 2, (250, 250, 250), -1)
            cv2.circle(ink, (x, 65), 2, 255, -1)
        cv2.circle(frame, (301, 30), 3, (45, 45, 45), -1)
        cv2.circle(frame, (301, 30), 2, (250, 250, 250), -1)
        mask = caption_mask(frame, (0, 0, 390, 120))
        self.assertTrue(np.all(mask[ink > 0] == 255))
        self.assertFalse(mask[:, 220:].any())

    def test_one_roi_covers_orange_upper_line_and_plain_lower_line(self):
        from subtitle_compositing import caption_mask
        for color in ((245, 245, 245), (20, 220, 245)):
            with self.subTest(color=color):
                original = np.full((230, 420, 3), (90, 110, 130), dtype=np.uint8)
                upper = np.zeros(original.shape[:2], np.uint8)
                lower = np.zeros_like(upper)
                cv2.putText(upper, 'TITLE', (105, 72), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            255, 2, cv2.LINE_AA)
                cv2.putText(lower, 'OK', (130, 185), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            255, 2, cv2.LINE_AA)
                outline = cv2.dilate(upper, np.ones((9, 9), np.uint8))
                original[outline > 0] = (15, 85, 235)
                original[upper > 0] = (245, 245, 245)
                original[lower > 0] = color
                mask = caption_mask(original, [65, 20, 355, 215])
                self.assertTrue(np.all(mask[upper > 0] == 255))
                self.assertTrue(np.all(mask[lower > 0] == 255))
                self.assertFalse(mask[105:140].any())
    def test_short_white_outlined_and_plain_yellow_captions_are_detected(self):
        from subtitle_compositing import caption_mask
        for text, color, outlined in (('OK', (245, 245, 245), True),
                                       ('I', (245, 245, 245), True),
                                       ('NO', (20, 220, 245), False)):
            with self.subTest(text=text, outlined=outlined):
                original = np.full((160, 400, 3), (90, 110, 130), dtype=np.uint8)
                strokes = np.zeros(original.shape[:2], np.uint8)
                cv2.putText(strokes, text, (80, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, 255, 2, cv2.LINE_AA)
                if outlined:
                    border = cv2.dilate(strokes, np.ones((7, 7), np.uint8))
                    original[border > 0] = (15, 15, 15)
                original[strokes > 0] = color
                mask = caption_mask(original, [50, 45, 330, 130])
                self.assertTrue(mask.any())
                self.assertTrue(np.all(mask[strokes > 0] == 255))
                self.assertFalse(mask[:45].any())
                self.assertFalse(mask[:, :50].any())

    def test_orange_scenery_and_isolated_bright_objects_are_not_caption_evidence(self):
        from subtitle_compositing import caption_mask
        original = np.full((160, 320, 3), (70, 90, 115), dtype=np.uint8)
        cv2.rectangle(original, (12, 15), (62, 135), (20, 70, 190), -1)
        cv2.rectangle(original, (130, 25), (220, 70), (30, 100, 210), -1)
        cv2.rectangle(original, (260, 105), (312, 145), (245, 245, 245), -1)
        self.assertFalse(caption_mask(original, [0, 0, 320, 160]).any())

    def test_white_text_on_dark_scene_stays_local_without_a_rectangular_band(self):
        from subtitle_compositing import caption_mask
        original = np.full((120, 420, 3), 12, dtype=np.uint8)
        cv2.putText(original, 'A short caption', (80, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (245, 245, 245), 2, cv2.LINE_AA)
        mask = caption_mask(original, [0, 10, 420, 115])
        self.assertTrue(mask[50:78, 80:270].any())
        self.assertFalse(mask[:, :50].any())
        self.assertFalse(mask[:, 340:].any())
        self.assertFalse(mask[10:30].any())

    def test_strong_orange_outline_and_offset_shadow_are_inside_original_derived_mask(self):
        from subtitle_compositing import caption_mask, composite_caption
        original = np.full((150, 280, 3), (90, 110, 130), dtype=np.uint8)
        # White glyph, thick orange outline, then a lower-right dark shadow.
        original[67:104, 85:126] = (15, 15, 15)
        original[58:96, 72:114] = (15, 85, 235)
        original[67:87, 81:105] = (245, 245, 245)
        roi = [35, 40, 230, 125]
        mask = caption_mask(original, roi)
        self.assertTrue(np.all(mask[58:104, 72:126] == 255))
        self.assertFalse(mask[:40].any())
        self.assertFalse(mask[:, :35].any())
        restored = np.full_like(original, (90, 110, 130))
        result = composite_caption(original, restored, mask)
        np.testing.assert_array_equal(result[mask == 0], original[mask == 0])
        self.assertLess(int(np.abs(result[67:87, 81:105].astype(np.int16)-restored[67:87, 81:105]).max()), 4)

    def test_bilingual_yellow_white_captions_include_complete_dark_plates_and_padding(self):
        from subtitle_compositing import caption_mask, composite_caption
        original = np.full((180, 360, 3), (85, 120, 95), dtype=np.uint8)
        original[52:88, 65:300] = (8, 8, 8)
        original[100:132, 88:275] = (12, 12, 12)
        for left in (85, 115, 145, 175, 205, 235, 265):
            original[62:78, left:left+13] = (15, 225, 245)
        for left in (105, 130, 155, 180, 205, 230, 255):
            original[109:123, left:left+10] = (245, 245, 245)
        mask = caption_mask(original, [45, 40, 320, 148])
        self.assertTrue(np.all(mask[52:88, 65:300] == 255))
        self.assertTrue(np.all(mask[100:132, 88:275] == 255))
        self.assertFalse(mask[20].any())
        restored = np.full_like(original, (85, 120, 95))
        result = composite_caption(original, restored, mask)
        self.assertGreater(int(result[55, 70].min()), 75)
        self.assertGreater(int(result[126, 268].min()), 75)
        np.testing.assert_array_equal(result[mask == 0], original[mask == 0])

    def test_plate_touching_dark_picture_content_recovers_line_padding(self):
        from subtitle_compositing import caption_mask
        original = np.full((150, 360, 3), (85, 120, 95), dtype=np.uint8)
        original[50:90, 50:305] = 8
        original[15:105, 298:330] = 8
        for left in (85, 115, 145, 175, 205, 235, 265):
            original[62:78, left:left+13] = (15, 225, 245)
        mask = caption_mask(original, [25, 20, 345, 120])
        self.assertTrue(np.all(mask[50:90, 50:305] == 255))
        self.assertFalse(mask[:20].any())

    def test_no_text_in_flat_bright_or_dark_background_has_empty_mask(self):
        from subtitle_compositing import caption_mask
        for color in ((0, 0, 0), (240, 240, 240), (0, 210, 240), (70, 100, 130)):
            original = np.full((80, 160, 3), color, dtype=np.uint8)
            with self.subTest(color=color):
                self.assertFalse(caption_mask(original, [0, 30, 160, 80]).any())

    def test_mask_mutation_by_opencv_cannot_expand_or_drop_selected_pixels(self):
        from subtitle_compositing import composite_caption
        original = np.full((36, 48, 3), 80, dtype=np.uint8)
        restored = np.full_like(original, 120)
        mask = np.zeros(original.shape[:2], np.uint8)
        mask[8:22, 10:30] = 255
        initial_mask = mask.copy()
        def mutating_clone(source, destination, clone_mask, center, flags):
            clone_mask[:] = 0
            return np.full_like(destination, 105)
        with patch('subtitle_compositing.cv2.seamlessClone', side_effect=mutating_clone):
            result = composite_caption(original, restored, mask)
        np.testing.assert_array_equal(mask, initial_mask)
        self.assertTrue(np.all(result[initial_mask > 0] == 105))
        np.testing.assert_array_equal(result[initial_mask == 0], original[initial_mask == 0])

    def test_tiny_and_frame_edge_masks_are_safe_and_keep_unselected_originals(self):
        from subtitle_compositing import caption_mask, composite_caption
        original = np.full((12, 16, 3), 80, dtype=np.uint8)
        restored = np.full_like(original, 120)
        for roi in ([0, 0, 2, 2], [14, 10, 16, 12], [0, 0, 16, 12]):
            with self.subTest(roi=roi):
                frame = original.copy()
                frame[roi[1]:min(roi[1]+2, roi[3]), roi[0]:min(roi[0]+2, roi[2])] = 255
                detected = caption_mask(frame, roi)
                self.assertEqual(detected.shape, frame.shape[:2])
                mask = np.zeros(frame.shape[:2], np.uint8)
                mask[roi[1]:roi[3], roi[0]:roi[2]] = 255
                result = composite_caption(frame, restored, mask)
                np.testing.assert_array_equal(result[mask == 0], frame[mask == 0])
                if min(roi[2]-roi[0], roi[3]-roi[1]) < 5:
                    np.testing.assert_array_equal(result[mask > 0], restored[mask > 0])
