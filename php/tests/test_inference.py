import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from fine import ReviewOptions, review, run_pass, suspect
from lite import LiteOptions, detect, restore, sample_block
from media import capture, encode


class InferenceTests(unittest.TestCase):
    def test_sampling_rechecks_changed_intervals_and_keeps_last_frame(self):
        a, b = ((3, 5, 15, 20),), ((35, 4, 70, 21),)
        frames = list(range(7))
        boxes, end = sample_block(frames, a, lambda i: a if i < 2 else b)
        self.assertEqual(boxes, [a, a, b, b, b, b])
        self.assertEqual(end, b)
        self.assertEqual(sample_block([6], b, lambda _: self.fail()), ([], b))

    def test_lite_composites_only_boxes_in_batches_of_four(self):
        class Inpaint:
            def _inpaint_batch(self, frames, masks):
                assert len(frames) <= 4 and all(m.any() for m in masks)
                return [np.full_like(f, 100) for f in frames]

        frames = [np.zeros((48, 64, 3), np.uint8) for _ in range(6)]
        results = restore(frames, [[(10, 10, 17, 22)]] * 5 + [[]], Inpaint())
        for frame in results[:5]:
            self.assertEqual(int(frame.sum()), 100 * 3 * 7 * 12)
        self.assertFalse(results[5].any())

    def test_watermark_review_uses_original_selection(self):
        img = np.zeros((64, 96, 3), np.uint8)
        cv2.putText(img, 'AB', (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        region = {'type': 'watermark', 'roi': [5, 5, 65, 48]}
        self.assertTrue(suspect(img, img, region))
        self.assertFalse(suspect(img, np.zeros_like(img), region))

    def test_lite_uses_selected_detection_size_box_context_and_batch(self):
        options = LiteOptions(detector_max_side=50, box_padding=2, context_padding=3, batch_size=2)
        seen = []

        def predict(image):
            self.assertEqual(image.shape[:2], (20, 50))
            return [{'dt_polys': [np.asarray([[5, 5], [12, 5], [12, 10], [5, 10]])]}]

        def inpaint(batch, masks):
            seen.append((len(batch), batch[0].shape[:2]))
            return batch

        frames = [np.zeros((60, 120, 3), np.uint8) for _ in range(3)]
        boxes = detect(frames[0], [(10, 10, 110, 50)], SimpleNamespace(predict=predict), options)
        self.assertEqual(boxes, ((18, 18, 37, 33),))
        restore(frames, [boxes] * 3, SimpleNamespace(_inpaint_batch=inpaint), options)
        self.assertEqual(seen, [(2, (21, 25)), (1, (21, 25))])

    def test_review_pads_suspect_intervals_per_region_and_clips_to_end(self):
        blank = np.zeros((64, 96, 3), np.uint8)
        text = blank.copy()
        cv2.putText(text, 'AB', (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)

        @contextmanager
        def open_frames(name):
            values = [text if (name == 'input' or i in (0, 8, 9)) else blank for i in range(10)]
            frames = iter((True, v) for v in values)
            yield SimpleNamespace(read=lambda: next(frames, (False, None)))

        regions = [
            {'type': 'watermark', 'roi': [5, 5, 65, 48]},
            {'type': 'subtitle', 'roi': [67, 1, 95, 40]},
        ]
        with patch('fine.capture', open_frames), patch('fine.geometry', return_value=(24, 10)):
            self.assertEqual(review('input', 'first', regions), ([[[0, 3], [6, 10]], []], 10))
            actual = review('input', 'first', regions, ReviewOptions(frame_padding=0))
            self.assertEqual(actual, ([[[0, 1], [8, 10]], []], 10))

    def test_fine_only_reprocesses_selected_frames_and_roi(self):
        # Use the production compositor/encoder, substituting only the diffusion prediction.
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            src = folder / 'in.avi'
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*'MJPG'), 12, (96, 64))
            for _ in range(8):
                frame = np.full((64, 96, 3), 60, np.uint8)
                cv2.putText(frame, 'ABC', (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
                writer.write(frame)
            writer.release()
            options = []

            def predict(**kwargs):
                from PIL import Image

                options.append(kwargs['num_inference_steps'])
                rng = np.random.default_rng(12)
                return [
                    Image.fromarray(rng.integers(20, 80, (im.height, im.width, 3), dtype=np.uint8))
                    for im in kwargs['control_video']
                ]

            region = {'type': 'subtitle', 'roi': [10, 10, 75, 45]}
            composited = []

            @contextmanager
            def record_encode(*args):
                with encode(*args) as pipe:

                    def write(data):
                        composited.append(np.frombuffer(data, np.uint8).reshape(64, 96, 3))
                        return pipe.write(data)

                    yield SimpleNamespace(write=write)

            with patch('worker.load_model', return_value=predict), patch('fine.encode', record_encode):
                run_pass(
                    folder, src, src, folder / 'out.mp4', [96, 64], [(region, [[2, 6]])], 8, 'subtitle', 85,
                    ReviewOptions(steps=7, subtitle_window=2),
                )
            self.assertEqual(options, [7, 7])
            with capture(src) as a, capture(folder / 'out.mp4') as b:
                for i in range(8):
                    left, original = a.read()
                    right, result = b.read()
                    self.assertTrue(left and right)
                    delta = cv2.absdiff(original, composited[i])
                    self.assertFalse(
                        delta[:10].any() or delta[45:].any() or delta[:, :10].any() or delta[:, 75:].any()
                    )
                    if i in range(2, 6):
                        self.assertGreater(delta[15:35, 15:55].mean(), 5)
                        self.assertGreater(cv2.absdiff(original, result)[15:35, 15:55].mean(), 5)
                    else:
                        self.assertFalse(delta.any())
