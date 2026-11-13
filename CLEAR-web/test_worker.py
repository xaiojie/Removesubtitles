import unittest
import numpy as np

from worker import repair_batch


class RegionRepairTests(unittest.TestCase):
    def test_repair_changes_only_selected_pixels_and_keeps_tail(self):
        frames = [np.zeros((24, 40, 3), dtype=np.uint8) for _ in range(3)]
        for frame in frames:
            frame[9, 8] = 255
        calls = []
        def predict(images):
            calls.append(len(images))
            from PIL import Image
            return [Image.new('RGB', image.size, (200, 100, 50)) for image in images]

        repair_batch(frames, [3, 4, 17, 16], predict)
        self.assertEqual(calls, [5])
        for frame in frames:
            self.assertLess(int(frame[9, 8].max()), 5)
            self.assertEqual(frame[6, 2].tolist(), [0, 0, 0])
            self.assertEqual(frame[6, 17].tolist(), [0, 0, 0])

    def test_repair_matches_background_color_at_selection(self):
        frame = np.full((64, 96, 3), 100, dtype=np.uint8)
        frame[24:29, 45:51] = 255

        def predict(images):
            from PIL import Image
            return [Image.new('RGB', image.size, (60, 120, 170)) for image in images]

        repair_batch([frame], [30, 16, 66, 38], predict)
        self.assertLess(abs(int(frame[26, 47, 0]) - 100), 5)
        self.assertEqual(frame[23, 29].tolist(), [100, 100, 100])

    def test_full_frame_context_keeps_video_composition_for_model(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[10:12, 10:12] = 255
        seen = []

        def predict(images):
            seen.extend((image.size, image.getpixel((10, 10))) for image in images)
            return images

        repair_batch([frame], [48, 48, 80, 80], predict, context="frame")
        self.assertEqual(seen, [((128, 128), (255, 255, 255))])
        self.assertEqual(frame[10, 10].tolist(), [255, 255, 255])

    def test_native_context_pads_without_resampling(self):
        frame = np.zeros((49, 65, 3), dtype=np.uint8)
        frame[9, 8] = 255

        def predict(images):
            self.assertEqual(images[0].size, (80, 64))
            self.assertEqual(images[0].getpixel((8, 9)), (255, 255, 255))
            return images

        repair_batch([frame], [20, 20, 40, 40], predict, context="frame", resolution="native")
        self.assertEqual(frame[9, 8].tolist(), [255, 255, 255])

    def test_line_blend_keeps_background_beyond_detected_line(self):
        from worker import blend_subtitle_lines
        source = np.full((5, 64, 96, 3), 90, dtype=np.uint8)
        source[:, 25:35, 30:66] = 255
        generated = np.full_like(source, 75)
        result = np.stack(blend_subtitle_lines(source, generated))
        self.assertEqual(result[2, 2, 2].tolist(), [90, 90, 90])
        self.assertLess(int(result[2, 30, 48].max()), 100)

    def test_line_blend_removes_thin_punctuation_tail(self):
        from worker import blend_subtitle_lines
        source = np.full((5, 64, 160, 3), 80, dtype=np.uint8)
        source[:, 20:30, 30:130] = 255
        source[:, 38:41, 90:92] = 255
        generated = np.full_like(source, 80)
        result = np.stack(blend_subtitle_lines(source, generated))
        self.assertLessEqual(int(result[:, 38:41, 90:92].max()), 82)
        background = np.all(source == 80, axis=-1)
        np.testing.assert_array_equal(result[background], source[background])

    def test_line_blend_falls_back_for_unrecognized_subtitle_style(self):
        from worker import blend_subtitle_lines
        source = np.zeros((1, 48, 48, 3), dtype=np.uint8)
        generated = np.full_like(source, 25)
        np.testing.assert_array_equal(blend_subtitle_lines(source, generated), generated)

class ShotSegmentationTests(unittest.TestCase):
    def test_hard_cut_detected_but_subtitle_only_change_ignored(self):
        from worker import find_shot_cuts
        dark = np.full((80, 100, 3), 30, dtype=np.uint8)
        subtitled = dark.copy()
        subtitled[60:76, 15:85] = 255
        next_shot = np.full_like(dark, 150)
        self.assertEqual(find_shot_cuts([dark, subtitled, next_shot, next_shot], [15, 60, 85, 76]), [2])

    def test_cut_resets_overlap_and_preserves_selected_range_boundaries(self):
        from worker import video_batches
        from inference_options import InferenceOptions
        frames = [np.array([i]) for i in range(13)]
        options = InferenceOptions(chunk_frames=5, overlap_frames=2)
        actual = [(list(map(int, [f[0] for f in carry])), list(map(int, [f[0] for f in frames])), selected)
                  for carry, frames, selected in video_batches(frames, (1, 11), [7, 9], options)]
        self.assertEqual(actual, [
            ([], [0], False), ([], [1,2,3,4,5], True), ([4,5], [6], True),
            ([], [7,8], True), ([], [9,10], True), ([], [11], False), ([], [12], False),
        ])

class MultiRegionRepairTests(unittest.TestCase):
    def repair(self, frames, regions, predict, **options):
        import worker
        self.assertTrue(callable(getattr(worker, "repair_regions", None)),
                        "multiple selected regions need a shared inference entry point")
        worker.repair_regions(frames, regions, predict, **options)

    def test_multiple_subtitle_regions_share_prediction_and_preserve_unselected_pixels(self):
        from PIL import Image
        originals = [np.full((64, 96, 3), 70, dtype=np.uint8) for _ in range(3)]
        for frame in originals:
            frame[12:18, 12:24] = 255
            frame[45:51, 62:78] = 255
        frames = [frame.copy() for frame in originals]
        calls = []
        def predict(images):
            calls.append(len(images))
            self.assertEqual(images[0].getpixel((14, 14)), (255, 255, 255))
            self.assertEqual(images[0].getpixel((66, 47)), (255, 255, 255))
            return [Image.new("RGB", image.size, (70, 70, 70)) for image in images]
        self.repair(frames, [
            {"type": "subtitle", "roi": [8, 8, 28, 24]},
            {"type": "subtitle", "roi": [58, 40, 84, 56]},
        ], predict, context="frame", resolution="native", blend_mode="lines")
        self.assertEqual(calls, [5])
        mask = np.ones((64, 96), dtype=bool)
        mask[8:24, 8:28] = False
        mask[40:56, 58:84] = False
        for original, frame in zip(originals, frames):
            self.assertLess(int(frame[14, 14].max()), 80)
            self.assertLess(int(frame[47, 66].max()), 80)
            np.testing.assert_array_equal(frame[mask], original[mask])

    def test_watermark_removes_colored_transparent_content_and_matches_background(self):
        from PIL import Image
        frame = np.full((48, 64, 3), (15, 80, 35), dtype=np.uint8)
        frame[13:19, 14:24] = (60, 110, 25)
        original = frame.copy()
        def predict(images):
            return [Image.new("RGB", image.size, (100, 60, 20)) for image in images]
        self.repair([frame], [{"type": "watermark", "roi": [8, 8, 32, 24]}], predict,
                    context="frame", resolution="native", blend_mode="lines")
        background = np.full((16, 24, 3), (15, 80, 35), dtype=np.int16)
        self.assertLessEqual(int(np.abs(frame[8:24, 8:32].astype(np.int16) - background).max()), 2)
        selected = np.zeros((48, 64), dtype=bool)
        selected[8:24, 8:32] = True
        np.testing.assert_array_equal(frame[~selected], original[~selected])

    def test_empty_watermark_regions_do_not_introduce_rectangular_color_patches(self):
        from PIL import Image
        frame = np.full((80, 112, 3), (55, 85, 115), dtype=np.uint8)
        original = frame.copy()
        calls = []
        def predict(images):
            calls.append(len(images))
            return [Image.new("RGB", image.size, (30, 150, 90)) for image in images]
        self.repair([frame], [
            {"type": "watermark", "roi": [0, 0, 32, 24]},
            {"type": "watermark", "roi": [76, 52, 112, 80]},
        ], predict, context="frame", resolution="native")
        self.assertEqual(calls, [1])
        self.assertLessEqual(int(np.abs(frame.astype(np.int16) - original).max()), 2)
        np.testing.assert_array_equal(frame[24:52], original[24:52])

    def test_watermark_priority_survives_overlapping_subtitle_and_input_order(self):
        from PIL import Image
        first = np.full((64, 96, 3), 90, dtype=np.uint8)
        first[30:40, 24:36] = 255
        first[16:22, 48:60] = (50, 145, 150)
        second = first.copy()
        watermark = {"type": "watermark", "roi": [40, 12, 80, 28]}
        subtitle = {"type": "subtitle", "roi": [20, 12, 86, 50]}
        def predict(images):
            return [Image.new("RGB", image.size, (50, 50, 50)) for image in images]
        self.repair([first], [watermark, subtitle], predict,
                    context="frame", resolution="native", blend_mode="lines")
        self.repair([second], [subtitle, watermark], predict,
                    context="frame", resolution="native", blend_mode="lines")
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(int(np.abs(first[16:22, 48:60].astype(np.int16) - 90).max()), 2)
        self.assertEqual(first[0, 0].tolist(), [90, 90, 90])

    def test_single_subtitle_matches_legacy_entry_point(self):
        from PIL import Image
        original = np.full((64, 96, 3), 100, dtype=np.uint8)
        original[24:29, 45:51] = 255
        legacy, regions = original.copy(), original.copy()
        def predict(images):
            return [Image.new("RGB", image.size, (60, 120, 170)) for image in images]
        repair_batch([legacy], [30, 16, 66, 38], predict)
        self.repair([regions], [{"type": "subtitle", "roi": [30, 16, 66, 38]}], predict)
        np.testing.assert_array_equal(regions, legacy)

    def test_roi_context_uses_all_regions_and_preserves_input_pixels(self):
        from PIL import Image
        frame = np.zeros((160, 256, 3), dtype=np.uint8)
        frame[42, 42] = (20, 40, 80)
        frame[112, 202] = (50, 70, 90)
        seen = []
        def predict(images):
            seen.append(images[0].size)
            self.assertEqual(images[0].getpixel((26, 42)), (80, 40, 20))
            self.assertEqual(images[0].getpixel((186, 112)), (90, 70, 50))
            return [Image.new("RGB", image.size, (0, 0, 0)) for image in images]
        self.repair([frame], [
            {"type": "watermark", "roi": [40, 40, 50, 50]},
            {"type": "watermark", "roi": [200, 110, 210, 120]},
        ], predict, context="roi", resolution="native")
        self.assertEqual(seen, [(224, 160)])

    def test_shot_detection_excludes_each_region_and_keeps_space_between_them(self):
        from worker import find_shot_cuts
        dark = np.full((80, 100, 3), 30, dtype=np.uint8)
        next_shot = dark.copy()
        next_shot[10:70, 10:90] = 170
        regions = [
            {"type": "watermark", "roi": [10, 10, 20, 20]},
            {"type": "subtitle", "roi": [80, 60, 90, 70]},
        ]
        self.assertEqual(find_shot_cuts([dark, dark, next_shot, next_shot], regions), [2])

    def test_overlapping_watermark_edges_do_not_restore_an_already_repaired_core(self):
        from PIL import Image
        first = np.full((48, 64, 3), 180, dtype=np.uint8)
        first[14:20, 24:30] = (70, 120, 150)
        original = first.copy()
        second = first.copy()
        left = {"type": "watermark", "roi": [8, 8, 36, 32]}
        right = {"type": "watermark", "roi": [24, 8, 52, 32]}
        def predict(images):
            return [Image.new("RGB", image.size, (60, 60, 60)) for image in images]
        self.repair([first], [left, right], predict, context="frame", resolution="native")
        self.repair([second], [right, left], predict, context="frame", resolution="native")
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(int(np.abs(first[14:20, 24:30].astype(np.int16) - 180).max()), 2)
        selected = np.zeros((48, 64), dtype=bool)
        selected[8:32, 8:52] = True
        np.testing.assert_array_equal(first[~selected], original[~selected])
