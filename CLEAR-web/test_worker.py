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
