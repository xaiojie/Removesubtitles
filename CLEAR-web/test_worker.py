import unittest
import numpy as np

from worker import repair_batch


class ProfileGeometryTests(unittest.TestCase):
    def test_fast_large_outlined_caption_gets_background_context_in_one_prediction(self):
        import cv2
        from PIL import Image
        from worker import repair_regions

        roi = [96, 606, 1184, 720]
        clean = np.full((720, 1280, 3), 110, np.uint8)
        blank, lettered = clean.copy(), clean.copy()
        for color, thickness in ((8, 13), (245, 5)):
            cv2.putText(lettered, 'OK', (546, 700), cv2.FONT_HERSHEY_SIMPLEX,
                        2.5, (color,)*3, thickness, cv2.LINE_AA)
        for preset, expected in [('fast', (640, 320)), ('custom', (1280, 176))]:
            with self.subTest(preset=preset):
                calls = []
                def predict(images):
                    calls.append((len(images), images[0].size))
                    return [Image.new('RGB', im.size, (110,)*3) for im in images]
                frames = [blank.copy(), lettered.copy()]
                repair_regions(frames, [dict(type='subtitle', roi=roi)], predict,
                               'roi', 'native', 'precise', preset=preset)
                self.assertEqual(calls, [(5, expected)])
                np.testing.assert_array_equal(frames[0], blank)
                np.testing.assert_array_equal(frames[1][:606], clean[:606])
                self.assertLess(np.abs(frames[1][635:716, 510:700].astype(int)-110).max(), 4)

    def test_fast_keeps_small_text_at_native_size_with_an_isolated_bright_object(self):
        import cv2
        from worker import repair_regions
        frame = np.full((720, 1280, 3), 90, np.uint8)
        cv2.rectangle(frame, (180, 635), (242, 696), (245,)*3, -1)
        cv2.putText(frame, 'Small caption', (480, 690), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (245,)*3, 2, cv2.LINE_AA)
        sizes = []
        def predict(images):
            sizes.append(images[0].size)
            return images
        repair_regions([frame], [dict(type='subtitle', roi=[96, 606, 1184, 720])],
                       predict, 'roi', 'native', 'precise', preset='fast')
        self.assertEqual(sizes, [(1280, 176)])

    def test_fast_hides_opaque_strokes_from_the_model_and_pastes_back_detached_marks(self):
        import cv2
        from unittest.mock import patch
        from worker import repair_regions
        clean = np.full((220, 512, 3), 101, np.uint8)
        frame = clean.copy()
        for color, thickness in ((6, 20), (248, 11)):
            cv2.putText(frame, 'HO', (152, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        2.2, (color,)*3, thickness, cv2.LINE_AA)
        cv2.circle(frame, (263, 181), 7, (6,)*3, -1)
        cv2.circle(frame, (263, 181), 4, (248,)*3, -1)
        calls = []
        def predict(images):
            calls.append(len(images))
            self.assertLess(int(np.asarray(images[0]).max()), 120)
            return images
        # The original component detector can miss a complete bold line or a detached stroke.
        with patch('worker.caption_mask', return_value=np.zeros(frame.shape[:2], np.uint8)):
            repair_regions([frame], [dict(type='subtitle', roi=[90, 105, 412, 213])],
                           predict, 'roi', 'native', 'precise', preset='fast')
        self.assertEqual(calls, [5])
        self.assertLess(np.abs(frame.astype(int)-clean).max(), 7)

    def test_large_caption_context_contains_multiple_edge_selections_and_caps_model_size(self):
        from worker import context_crop, model_geometry
        for width, height in ((721, 1281), (3840, 2160)):
            with self.subTest(width=width):
                rois = [[0, height//2, width//3, height//2+80],
                        [width*2//3, height-115, width, height]]
                regions = [dict(type='subtitle', roi=roi) for roi in rois]
                l, t, r, b = context_crop(regions, width, height, 'roi', large_caption=True)
                self.assertTrue(0 <= l < r <= width and 0 <= t < b <= height)
                for x1, y1, x2, y2 in rois:
                    self.assertTrue(l <= x1 < x2 <= r and t <= y1 < y2 <= b)
                (w, h), (pw, ph) = model_geometry(r-l, b-t, 'native', large_caption=True)
                self.assertLessEqual(abs(w/(r-l)-h/(b-t)), 1/min(r-l, b-t))
                self.assertLessEqual(max(pw, ph), 1024)
                self.assertTrue(w <= pw < w+16 and h <= ph < h+16)
                self.assertEqual((pw % 16, ph % 16), (0, 0))

    def test_expanded_context_reproduces_proportional_landscape_context(self):
        from worker import context_crop
        regions = [{'type': 'subtitle', 'roi': [261, 622, 970, 718]}]
        self.assertEqual(context_crop(regions, 1280, 720, 'expanded'), (128, 336, 1152, 720))

    def test_expanded_context_contains_portrait_edge_and_multiple_selections(self):
        from worker import context_crop
        cases = [
            (720, 1280, [[120, 1040, 600, 1120]]),
            (1920, 1080, [[0, 0, 240, 100]]),
            (1080, 1920, [[870, 1600, 1080, 1920]]),
            (1280, 720, [[0, 30, 180, 120], [1080, 600, 1280, 720]]),
        ]
        for width, height, rois in cases:
            with self.subTest(width=width, height=height, rois=rois):
                left, top, right, bottom = context_crop(
                    [dict(type='subtitle', roi=roi) for roi in rois], width, height, 'expanded')
                self.assertTrue(0 <= left < right <= width)
                self.assertTrue(0 <= top < bottom <= height)
                self.assertGreaterEqual(right-left, min(width, round(width*.8)))
                self.assertGreaterEqual(bottom-top, min(height, round(height*8/15)))
                for x1, y1, x2, y2 in rois:
                    self.assertTrue(left <= x1 < x2 <= right)
                    self.assertTrue(top <= y1 < y2 <= bottom)

    def test_quality_geometry_resizes_proportionally_then_pads_to_model_grid(self):
        from worker import model_geometry
        self.assertEqual(model_geometry(1280, 720, 'auto', preserve_aspect=True),
                         ((848, 477), (848, 480)))
        for width, height in ((720, 1280), (83, 49), (2, 3), (3840, 2160)):
            resized, padded = model_geometry(width, height, 'auto', preserve_aspect=True)
            self.assertLessEqual(abs(resized[0]/width - resized[1]/height), 1/min(width, height))
            self.assertEqual((padded[0] % 16, padded[1] % 16), (0, 0))
            self.assertTrue(resized[0] <= padded[0] and resized[1] <= padded[1])

    def test_precise_subtitle_without_text_skips_prediction_and_retains_pixels(self):
        from worker import repair_regions
        original = np.full((72, 128, 3), (70, 85, 95), dtype=np.uint8)
        frame = original.copy()
        def predict(images):
            self.fail('blank subtitle selections should not require the model')
        methods = repair_regions([frame], [dict(type='subtitle', roi=[0, 48, 128, 72])],
                                 predict, context='expanded', resolution='native', blend_mode='precise')
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(methods, dict(background_regions=1, model_regions=0))

    def test_quality_prediction_uses_full_frame_with_aspect_padding(self):
        from worker import repair_regions
        frame = np.full((720, 1280, 3), 85, dtype=np.uint8)
        import cv2
        cv2.putText(frame, 'Caption', (490, 660), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, 'Caption', (490, 660), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (245, 245, 245), 2, cv2.LINE_AA)
        calls = []
        def predict(images):
            calls.append(len(images))
            self.assertEqual(images[0].size, (848, 480))
            return images
        repair_regions([frame], [dict(type='subtitle', roi=[450, 620, 800, 700])], predict,
                       context='frame', resolution='auto', blend_mode='precise')
        self.assertEqual(calls, [5])

    def test_precise_multiple_rois_process_short_text_and_leave_blank_frames_exact(self):
        from worker import repair_regions
        from PIL import Image
        import cv2
        original = np.full((180, 320, 3), 95, dtype=np.uint8)
        cv2.putText(original, 'OK', (35, 62), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(original, 'NO', (215, 155), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (20, 220, 245), 2, cv2.LINE_AA)
        blank = np.full_like(original, 95)
        frames = [original.copy(), blank.copy()]
        calls = []
        def predict(images):
            calls.append(len(images))
            return [Image.new('RGB', image.size, (95, 95, 95)) for image in images]
        repair_regions(frames, [dict(type='subtitle', roi=[20, 25, 110, 80]),
                                dict(type='subtitle', roi=[195, 115, 300, 175])], predict,
                       context='expanded', resolution='native', blend_mode='precise')
        self.assertEqual(calls, [5])
        self.assertLess(int(frames[0][40:65, 35:78].max()), 100)
        self.assertLess(int(frames[0][135:157, 215:260].max()), 100)
        np.testing.assert_array_equal(frames[1], blank)
        selected = np.zeros(original.shape[:2], bool)
        selected[25:80, 20:110] = True
        selected[115:175, 195:300] = True
        np.testing.assert_array_equal(frames[0][~selected], original[~selected])


class ResidentModelTests(unittest.TestCase):
    def test_opt_in_large_gpu_retains_dit_and_vae_without_losing_other_requested_models(self):
        from worker import configure_model_residency
        class Pipeline:
            def load_models_to_device(self, names=None):
                return names
        for capacity in (20*1024**3, 24*1024**3):
            with self.subTest(capacity=capacity):
                pipeline = Pipeline()
                self.assertTrue(configure_model_residency(pipeline, True, capacity))
                self.assertEqual(pipeline.load_models_to_device(['text_encoder', 'dit']),
                                 ['text_encoder', 'dit', 'vae'])
                self.assertEqual(pipeline.load_models_to_device(), ['dit', 'vae'])
                self.assertEqual(pipeline.load_models_to_device([]), ['dit', 'vae'])

    def test_low_memory_and_legacy_opt_out_retain_original_model_loading(self):
        from worker import configure_model_residency
        class Pipeline:
            def load_models_to_device(self, names=None):
                return names
        for enabled, capacity in ((False, 80*1024**3), (True, 20*1024**3-1), (True, 16*1024**3)):
            with self.subTest(enabled=enabled, capacity=capacity):
                pipeline = Pipeline()
                self.assertFalse(configure_model_residency(pipeline, enabled, capacity))
                self.assertEqual(pipeline.load_models_to_device(['dit']), ['dit'])
                self.assertIsNone(pipeline.load_models_to_device())


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

    def test_one_frame_shot_repeats_its_own_frame_without_extending_the_video(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[10:12, 10:12] = 255
        frames = [frame]
        seen = []

        def predict(images):
            seen.extend((image.size, image.getpixel((10, 10))) for image in images)
            return images

        repair_batch(frames, [48, 48, 80, 80], predict, context="frame")
        self.assertEqual(seen, [((128, 128), (255, 255, 255))] * 5)
        self.assertEqual(len(frames), 1)
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
        self.assertEqual(calls, [5])
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
def test_temporal_stroke_recovery_uses_only_exposed_matching_background():
    from subtitle_compositing import visible_background

    bg = np.full((80, 100, 3), (62, 130, 170), np.uint8)
    mask = np.zeros(bg.shape[:2], np.uint8)
    mask[32:44, 40:52] = 255
    text = bg.copy()
    text[mask > 0] = 245
    blank = np.zeros_like(mask)
    values = visible_background([text, bg], [mask, blank], [mask, None])
    colors, valid = values[0]
    np.testing.assert_array_equal(valid, mask > 0)
    np.testing.assert_array_equal(colors[valid], bg[valid])
    assert values[1] is None
    assert visible_background([text, bg], [mask, mask], [mask, None])[0] is None
    assert visible_background([text, 255 - bg], [mask, blank], [mask, None])[0] is None
