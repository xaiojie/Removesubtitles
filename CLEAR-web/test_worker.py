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
