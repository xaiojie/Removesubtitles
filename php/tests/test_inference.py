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
from lite import LiteOptions, detect, restore, restore_joint, sample_block
from media import capture, encode


class InferenceTests(unittest.TestCase):
    def test_enlarged_word_highlight_confirms_caption_rows_without_erasing_unrelated_print(self):
        from lite_profile import BandProfile

        width, height = 820, 360
        roi = (0, 0, width, height)
        font = cv2.FONT_HERSHEY_DUPLEX
        fill = (65, 47, 32)
        painter = SimpleNamespace(device='cpu', _inpaint_batch=lambda images, masks: [
            np.full_like(image, 19) for image in images
        ])
        options = LiteOptions(mask_mode='tracked_cards', layout_profile=True, follow_scaled_rows=True)
        for color in [(46, 168, 61), (178, 63, 144)]:
            hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0].tolist()
            profile = BandProfile(width, height, [(185, 38), (228, 38)], None, (*hsv, 270, 35))
            for backing in ['word', 'unrelated', 'whole-scene']:
                with self.subTest(color=color, backing=backing):
                    image = np.full((height, width, 3), color if backing == 'whole-scene' else fill, np.uint8)
                    boxes = []
                    for words, baseline in [(('FAST', 'MOVING'), 166), (('TRAIN', 'TODAY'), 239)]:
                        sizes = [cv2.getTextSize(word, font, 2.5, 3)[0] for word in words]
                        x = (width-sum(w for w, _ in sizes)-25)//2
                        for word, (w, h) in zip(words, sizes):
                            if word == 'FAST' and backing == 'word':
                                cv2.rectangle(image, (x-12, baseline-h-11), (x+w+12, baseline+10), color, -1)
                            cv2.putText(image, word, (x, baseline), font, 2.5, (245, 245, 245), 3, cv2.LINE_AA)
                            boxes.append((x-5, baseline-h-6, x+w+5, baseline+8))
                            x += w+25
                    if backing == 'unrelated':
                        cv2.rectangle(image, (12, 100), (145, 175), color, -1)
                    original = image.copy()
                    self.assertFalse(profile.dialogue_rows(boxes) or profile.select(boxes))
                    restore_joint([image], [boxes], [roi], painter, options, profiles={roi: profile})
                    if backing == 'word':
                        letters = np.all(original > 225, axis=2)
                        panel = np.all(original == color, axis=2)
                        self.assertTrue(np.all(image[letters] == 19))
                        self.assertTrue(np.all(image[panel] == 19))
                        np.testing.assert_array_equal(image[:78], original[:78])
                    else:
                        np.testing.assert_array_equal(image, original)

    def test_new_caption_size_and_extra_row_keep_only_matching_print(self):
        from lite_profile import BandProfile

        width, height = 820, 360
        font = cv2.FONT_HERSHEY_DUPLEX
        profile = BandProfile(width, height, [(185, 38), (228, 38)], None)
        image = np.full((height, width, 3), (56, 41, 35), np.uint8)

        def put_words(words, baseline, scale, colour):
            widths = [cv2.getTextSize(word, font, scale, 3)[0][0] for word in words]
            gap = round(10*scale)
            x = (width-sum(widths)-(len(words)-1)*gap)//2
            result = []
            for word, span in zip(words, widths):
                cap_height = cv2.getTextSize(word, font, scale, 3)[0][1]
                cv2.putText(image, word, (x, baseline), font, scale, colour, 3, cv2.LINE_AA)
                result.append((x-5, baseline-cap_height-6, x+span+5, baseline+8))
                x += span+gap
            return result

        white = (242, 242, 242)
        large = put_words(['FAST', 'MOVING'], 166, 2.5, white)
        large += put_words(['TRAIN', 'TODAY'], 239, 2.5, white)
        self.assertEqual(profile.dialogue_rows(large) or profile.select(large), [])
        self.assertCountEqual(profile.scaled_dialogue(image, large, []), large)

        image[:] = (56, 41, 35)
        first = put_words(['EVENING', 'SERVICE'], 194, 1.3, white)
        second = put_words(['TOWN', 'CENTRE'], 237, 1.3, white)
        third = put_words(['LAST', 'STOP'], 280, 1.3, white)
        scene_text = put_words(['STATION'], 323, 1.3, (45, 55, 190))
        selected = first + second
        self.assertCountEqual(profile.scaled_dialogue(image, selected+third+scene_text, selected),
                              selected+third)

    def test_centered_dialogue_with_a_row_split_into_word_boxes(self):
        from lite_profile import BandProfile

        width, height = 860, 160
        image = np.full((height, width, 3), 55, np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX

        def draw_row(text, baseline):
            (span, cap_height), _ = cv2.getTextSize(text, font, 1.15, 2)
            left = (width - span) // 2
            cv2.putText(image, text, (left, baseline), font, 1.15, (242, 242, 242), 2, cv2.LINE_AA)
            return (left-6, baseline-cap_height-6, left+span+6, baseline+7)

        upper = draw_row('EVENING TRAIN', 63)
        lower = draw_row('FREE RIDE', 104)
        x, y, r, b = lower
        split = x + 6 + cv2.getTextSize('FREE ', font, 1.15, 2)[0][0]
        words = [(x, y, split+3, b), (split-3, y, r, b)]
        native_label = (12, upper[1], 65, upper[3])
        boxes = [native_label, upper, *words]
        profile = BandProfile(width, height, [((y+b)/2, b-y)], None)
        self.assertCountEqual(profile.select(boxes), words)
        self.assertEqual(profile.wrapped_dialogue(image, boxes, words), [])
        self.assertCountEqual(profile.wrapped_dialogue(image, boxes, words, group_words=True), [upper, *words])
        image[:71] = 55
        self.assertEqual(profile.wrapped_dialogue(image, boxes, words, group_words=True), [])

    def test_scaled_dialogue_preserves_screen_print_outside_a_confirmed_caption_card(self):
        from lite_profile import BandProfile
        from subtitle_compositing import colored_caption_cards

        image = np.full((300, 800, 3), 64, np.uint8)
        profile = BandProfile(800, 300, [(141, 51), (190, 51)], None)
        # Both rows have the same print. Only the lower one sits on the wide subtitle backing.
        rows = [(226, 61, 574, 111), (226, 117, 574, 167)]
        for bounds, expected in [((75, 114, 724, 223), []), ((75, 51, 724, 220), rows),
                                 ((218, 114, 565, 223), rows)]:
            with self.subTest(card=bounds):
                image[:] = 64
                left, top, right, bottom = bounds
                cv2.rectangle(image, (left, top), (right, bottom), (185, 117, 49), cv2.FILLED)
                for x, y, right, low in rows:
                    cv2.putText(image, 'MONDAY NEWS', (x+4, low-6), cv2.FONT_HERSHEY_DUPLEX,
                                1.45, (244, 244, 244), 3, cv2.LINE_AA)
                self.assertIsNotNone(colored_caption_cards(image))
                self.assertEqual(profile.scaled_dialogue(image, rows, [rows[1]]), expected)

    def test_a_short_final_line_does_not_hide_the_matching_line_above_it(self):
        from lite_profile import BandProfile

        frame = np.full((160, 700, 3), 77, np.uint8)
        boxes = []
        for text, baseline, scale in [('LONG CAPTION ABOVE', 74, 1.6), ('ends.', 126, 1.6)]:
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
            x = (700-w)//2
            cv2.putText(frame, text, (x, baseline), cv2.FONT_HERSHEY_DUPLEX, scale, (13, 13, 13), 7, cv2.LINE_AA)
            cv2.putText(frame, text, (x, baseline), cv2.FONT_HERSHEY_DUPLEX, scale, (246, 246, 246), 3, cv2.LINE_AA)
            boxes.append((x-7, baseline-h-7, x+w+7, baseline+9))
        row = boxes[1]
        profile = BandProfile(700, 160, [((row[1]+row[3])/2, row[3]-row[1])], None)
        self.assertEqual(profile.select(boxes), [row])
        self.assertCountEqual(profile.wrapped_dialogue(frame, boxes, [row]), boxes)
        # A native heading with a different fill or an off-centre label must remain visible.
        frame[:83] = (50, 83, 202)
        self.assertFalse(profile.wrapped_dialogue(frame, boxes, [row]))
        extra = (1, boxes[0][1], 125, boxes[0][3])
        self.assertFalse(profile.wrapped_dialogue(frame, [extra, row], [row]))

    def test_follow_a_verified_translucent_line_through_one_weak_edge(self):
        from lite_line_panel import LinePanel

        roi = (0, 20, 420, 134)
        panel = LinePanel(roi, (420, 114), (90, 28, 330, 86), .55,
                          np.full(3, 25., np.float32), (0, 48, 420, 106))
        row = [(107, 37, 313, 77)]
        def caption(frame):
            image = frame.copy()
            image[48:106, 90:330] = np.rint(.55 * image[48:106, 90:330] + 25).astype(np.uint8)
            cv2.putText(image, 'CAPTION', (115, 88), cv2.FONT_HERSHEY_DUPLEX,
                        1.35, (245, 245, 245), 3, cv2.LINE_AA)
            return image

        clear = np.full((160, 420, 3), 160, np.uint8)
        weak = clear.copy()
        weak[98:118, 117:306] = 57
        self.assertIsNotNone(panel.restore(caption(clear), roi, row))
        self.assertIsNone(panel.restore(caption(weak), roi, row))
        self.assertIsNotNone(panel.restore(caption(clear), roi, row, frame_index=4))
        recovered = panel.restore(caption(weak), roi, row, frame_index=5)
        self.assertIsNotNone(recovered)
        self.assertLess(np.abs(recovered[1][8:-8, 5:18].astype(int) - 160).mean(), 2)
        self.assertIsNone(panel.restore(clear, roi, row, frame_index=6))
        self.assertIsNone(panel.restore(caption(weak), roi, row, frame_index=60))

    def test_learned_horizontal_backing_protects_slanted_scene_signs_after_the_caption_ends(self):
        from lite_line_panel import LinePanel
        from lite_profile import BandProfile

        roi = (10, 10, 290, 125)
        panel = LinePanel((0, 0, 300, 140), (300, 140), (90, 47, 210, 94),
                          .55, np.array([28, 28, 28], np.float32), (0, 47, 300, 94))
        profile = BandProfile(280, 115, [(60, 30)], panel)
        flat = np.array([[36, 48], [222, 49], [222, 68], [36, 67]])
        slanted = np.array([[56, 42], [184, 73], [180, 89], [52, 58]])
        frame = np.full((140, 300, 3), 170, np.uint8)
        opts = LiteOptions(box_padding=0)
        detector = SimpleNamespace(predict=lambda crop: [{'dt_polys': [flat, slanted]}])
        expected = detect(frame, [roi], SimpleNamespace(predict=lambda crop: [{'dt_polys': [flat]}]), opts)
        self.assertEqual(detect(frame, [roi], detector, opts, profiles={roi: profile}), expected)
        frame[47:94, 24:275] = 122
        self.assertEqual(detect(frame, [roi], detector, opts, profiles={roi: profile}),
                         detect(frame, [roi], detector, opts))

    def test_centered_caption_band_can_contain_a_whole_multiline_white_card(self):
        from subtitle_compositing import light_caption_cards

        scene = np.full((250, 630, 3), 75, np.uint8)
        scene[30:222, 28:602] = 249
        boxes = []
        for y, text in zip((83, 111, 139, 167), (
                'A clear subtitle in this scene', 'a second sentence follows here',
                'the third line in this dialogue', 'another part of the sentence')):
            cv2.putText(scene, text, (47, y), cv2.FONT_HERSHEY_DUPLEX, .72,
                        (11, 11, 11), 2, cv2.LINE_AA)
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, .72, 2)
            boxes.append((43, y-height-4, 47+width+4, y+8))
        self.assertIsNone(light_caption_cards(scene))
        mask = light_caption_cards(scene, ocr_boxes=boxes)
        self.assertIsNotNone(mask)
        self.assertTrue(mask[30:222, 28:602].all())
        self.assertEqual(mask[12, 100], 0)
        self.assertIsNone(light_caption_cards(scene, ocr_boxes=[boxes[0]]))
        self.assertIsNone(light_caption_cards(scene, ocr_boxes=[boxes[0]] * 4))
        empty_screen = scene.copy()
        empty_screen[30:222, 28:602] = 249
        self.assertIsNone(light_caption_cards(empty_screen, ocr_boxes=boxes))

    def test_short_clip_graph_threshold_uses_the_decoded_duration(self):
        from lite import LiteOptions

        options = LiteOptions(skip_graph_up_to_seconds=30)
        self.assertFalse(options.for_clip(720, 24).optimize_lama_graph)
        self.assertFalse(options.for_clip(899, 30000/1001).optimize_lama_graph)
        self.assertTrue(options.for_clip(900, 30000/1001).optimize_lama_graph)
        self.assertTrue(options.for_clip(1, 30).optimize_lama_graph)
        self.assertTrue(LiteOptions().for_clip(720, 24).optimize_lama_graph)

    def test_translucent_line_backing_changes_width_and_disappears(self):
        from lite_line_panel import learn_line_panel

        height, width = 100, 380
        top, bottom = 32, 67
        x = np.arange(width)[None, :]
        y = np.arange(height)[:, None]
        frames, clean, spans = [], [], []
        for index in range(13):
            tone = np.clip(130 + 63*np.sin(x/70 + index*.6) + 23*np.sin(y/65), 0, 255)
            background = np.stack([tone, tone*.73, tone*.87], axis=2).astype(np.uint8)
            image = background.copy()
            left = 35 + (index % 4)*16
            right = width-left
            if index:
                image[top:bottom, left:right] = .54 * background[top:bottom, left:right] + 26
                cv2.putText(image, 'SUBTITLE', (left+14, top+24), cv2.FONT_HERSHEY_SIMPLEX, .68,
                            (248, 248, 248), 2, cv2.LINE_AA)
            frames.append(image)
            clean.append(background)
            spans.append((left, right))
        roi = (0, 0, width, height)
        model = learn_line_panel(frames, roi)
        self.assertIsNotNone(model)
        self.assertTrue(model.calibrate([model.measure_edges(im) for im in frames[::2]]))
        for i in (1, 3, 6, 8):
            left, right = spans[i]
            box = (left+6, top+4, right-6, bottom-4)
            bounds, recovered, mask = model.restore(frames[i], roi, [box])
            self.assertEqual(bounds, (left, top, right, bottom))
            self.assertGreater(cv2.countNonZero(mask), 0)
            unprinted = mask[top:bottom, left:right] == 0
            error = cv2.absdiff(recovered, clean[i][top:bottom, left:right])
            self.assertLess(error[unprinted].mean(), 4)
            self.assertIsNone(model.restore(frames[0], roi, [box]))

    def test_flat_outlined_glyph_is_kept_whole_between_letters(self):
        from lite import outlined_text_mask

        def patch_with_bar(outlined):
            patch = np.full((76, 210, 3), 150, np.uint8)
            for x in (9, 34, 119, 146, 173, 192):
                cv2.rectangle(patch, (x-2, 13), (x+7, 60), (10, 10, 10), -1)
                cv2.rectangle(patch, (x, 17), (x+5, 56), (245, 245, 245), -1)
            if outlined:
                cv2.rectangle(patch, (73, 32), (115, 42), (8, 8, 8), -1)
            cv2.rectangle(patch, (76, 36), (112, 38), (245, 245, 245), -1)
            return patch

        patch = patch_with_bar(True)
        prior = outlined_text_mask(patch, 9)
        self.assertIsNotNone(prior)
        self.assertFalse(prior[37, 76])
        self.assertTrue(outlined_text_mask(patch, 9, complete_flat_strokes=True)[32:43, 73:116].all())
        unbordered = patch_with_bar(False)
        self.assertFalse(outlined_text_mask(unbordered, 9, complete_flat_strokes=True)[37, 76])

    def test_weak_highlight_requires_current_edges_and_a_recent_match_in_this_shot(self):
        from lite_highlight import MovingHighlight

        model = MovingHighlight((0, 0, 128, 60), (128, 60), 14, 44, 0, 2, .6,
                                np.array([75, 48, 12], np.float32))
        frame = np.full((60, 128, 3), 80, np.uint8)

        def detect(scores, span=(25, 55), shot=frame):
            with patch('lite_highlight.segments', return_value=[span] if span else []), \
                 patch('lite_highlight.blend_quality', side_effect=scores):
                return model.mask(shot, (0, 0, 128, 60), track=True)

        self.assertIsNone(detect([.8, .20]))
        self.assertIsNotNone(detect([.8, .7]))
        self.assertIsNone(detect([.9, .22], span=(74, 105)))
        self.assertIsNotNone(detect([.8, .20]))
        self.assertIsNone(detect([], span=None))
        self.assertIsNotNone(detect([.8, .20]))
        self.assertIsNone(detect([.8, .20]))
        self.assertIsNotNone(detect([.8, .7]))
        self.assertIsNone(detect([.8, .20], shot=np.full_like(frame, 160)))

    def test_tint_fringe_is_removed_only_on_a_supported_smooth_edge(self):
        from lite_gradient import reduce_edge_fringe

        strip = np.full((17, 90, 3), 110, np.uint8)
        strip[8, :60] = (91, 143, 83)
        strip[8:, 60:] = (202, 67, 38)
        strip[7, 15] = (5, 92, 215)
        object_edge = strip[:, 60:].copy()
        reduce_edge_fringe(strip)
        np.testing.assert_array_equal(strip[8, :60], np.full((60, 3), 110, np.uint8))
        np.testing.assert_array_equal(strip[:, 60:], object_edge)
        np.testing.assert_array_equal(strip[7, 15], [5, 92, 215])

    def test_similar_frames_require_matching_native_masks_and_stay_close_to_anchor(self):
        from lite_reuse import inference_frames
        images = [np.full((32, 56, 3), n, np.uint8) for n in (80, 82, 84, 86, 86, 86)]
        masks = [np.full((32, 56), 255, np.uint8) for image in images]
        masks[4][15, 12] = 0
        images[5][2, 43] = 211
        self.assertEqual(inference_frames(images, masks, 4), ([0, 3, 4, 5], [0, 0, 0, 1, 2, 3]))
        self.assertEqual(inference_frames(images, masks, 0), (list(range(6)), list(range(6))))

    def test_known_highlight_can_catch_a_short_word_between_ocr_detections(self):
        from lite_masks import chip_examples,locate_chip,repeated_chips
        samples=[]
        for x in (22,80,155,104):
            crop=np.full((80,300,3),45,np.uint8)
            crop[23:53,x:x+51]=(50,178,52)
            cv2.putText(crop,'DA',(x+4,46),cv2.FONT_HERSHEY_SIMPLEX,.67,(244,244,244),2)
            samples.append(chip_examples(crop,[(10,14,275,61)]))
        template=repeated_chips(samples)
        self.assertIsNotNone(template)
        found=locate_chip(crop,template)
        self.assertTrue(np.all(found[23:53,104:155]))
        # A switch to a sign of the same color away from the subtitle row stays intact.
        blank=np.full_like(crop,45)
        blank[:12,104:155]=(50,178,52)
        self.assertIsNone(locate_chip(blank,template))

    def test_translucent_panel_uses_visible_texture_and_masks_only_ink(self):
        from lite_layer import learn_panel
        rng=np.random.default_rng(21)
        backgrounds=[]; inputs=[]
        for i in range(9):
            coarse=rng.integers(40,235,(8,12,3),dtype=np.uint8)
            background=cv2.resize(coarse,(240,94),interpolation=cv2.INTER_LINEAR)
            ink=background.copy()
            ink[18:76,20:218]=(ink[18:76,20:218].astype(float)*.45+4).astype(np.uint8)
            cv2.putText(ink,'SUBTITLE',(51,60),cv2.FONT_HERSHEY_SIMPLEX,.85,(241,241,241),2)
            backgrounds.append(background)
            inputs.append(ink)
        panel=learn_panel(inputs,(0,0,240,94))
        self.assertIsNotNone(panel)
        self.assertAlmostEqual(panel.transmission,.45,delta=.065)
        rect,repair,mask=panel.restore(inputs[3],(0,0,240,94),[(43,28,184,72)])
        l,t,r,b=rect
        compare=np.zeros((94,240),bool)
        compare[23:31,32:208]=True
        selected=compare[t:b,l:r]
        reality=backgrounds[3][t:b,l:r].astype(float)
        self.assertLess(np.abs(repair.astype(float)-reality)[selected].mean(),9)
        self.assertGreater(np.count_nonzero(mask),100)
        self.assertEqual(np.count_nonzero(mask[:25]),0)
        self.assertIsNone(panel.restore(backgrounds[3],(0,0,240,94),[(43,28,184,72)]))

    def test_scaled_panel_recovers_half_open_edges_at_source_resolution(self):
        from lite_layer import learn_panel
        frames, thumbs = [], []
        rng = np.random.default_rng(27)
        for _ in range(7):
            texture = rng.integers(60, 235, (3, 5, 3), dtype=np.uint8)
            frame = cv2.resize(texture, (610, 206), interpolation=cv2.INTER_LINEAR)
            frame[37:165, 39:571] = (frame[37:165, 39:571].astype(float) * .45).astype(np.uint8)
            cv2.putText(frame, 'SAMPLE', (130, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 2)
            frames.append(frame)
            thumbs.append(cv2.resize(frame, (240, 81), interpolation=cv2.INTER_AREA))
        panel = learn_panel(thumbs, (0, 0, 610, 206))
        self.assertIsNotNone(panel)
        self.assertTrue(panel.calibrate([panel.measure_edges(frame) for frame in frames[:5]]))
        self.assertEqual(panel.native_box, (39, 37, 571, 165))
        # The user's top boundary cuts through the layer. Never brighten the exterior or cross the selection.
        rect, _, _ = panel.restore(frames[0], (12, 40, 601, 197), [(108, 41, 296, 93)])
        self.assertEqual(rect, (39, 40, 571, 165))

    def test_gradient_panel_recovers_scene_print_and_tracks_fixed_decoration(self):
        from lite_gradient import learn_gradient_panel
        rng = np.random.default_rng(52)
        frames, scenes = [], []
        left, top, right, bottom = 12, 14, 232, 78
        offset = np.stack((np.linspace(104,120,220), np.linspace(43,111,220), np.full(220,127)), axis=1)
        for index in range(11):
            coarse = rng.integers(50, 225, (4, 8, 3), dtype=np.uint8)
            clean = cv2.resize(coarse, (244, 96))
            cv2.putText(clean, 'STAMP', (90, 50), cv2.FONT_HERSHEY_SIMPLEX, .46, (10, 10, 10), 2)
            source = clean.copy()
            source[top:bottom,left:right] = (clean[top:bottom,left:right]*.47 + offset).astype(np.uint8)
            cv2.circle(source,(47,69),6,(12,37,180),-1)
            cv2.putText(source,str(index),(139,41),cv2.FONT_HERSHEY_SIMPLEX,.48,(250,250,250),2)
            frames.append(source)
            scenes.append(clean)
        panel = learn_gradient_panel(np.array(frames), (0,0,244,96), [left,right], [top,bottom])
        self.assertIsNotNone(panel)
        self.assertTrue(panel.calibrate([panel.measure_edges(frame) for frame in frames[:5]]))
        rect, bg, mask = panel.restore(frames[3],(0,0,244,96),[(134,24,172,47)])
        l,t,r,b = rect
        self.assertEqual(rect,(left,top,right,bottom))
        self.assertEqual(int(mask[69,47]),255)
        self.assertLess(np.abs(bg[43-t:52-t,94-l:124-l].astype(float) -
                               scenes[3][43:52,94:124]).mean(),9)
        self.assertIsNone(panel.restore(scenes[3],(0,0,244,96),[]))

    def test_printed_border_keeps_its_low_contrast_ink_inside_the_repair(self):
        from lite_gradient import GradientPanel

        size = (100, 50)
        panel = GradientPanel((0, 0, *size), size, (0, 0, *size), .5,
                              np.array([70., 70., 70.]), np.array([0., 0., 0.]))
        samples = np.full((8, 50, 100, 3), 133, np.uint8)
        for x in range(0, 100, 10):
            samples[:, 40:, x:x+5] = (10, 70, 180)
        panel.learn_decoration(samples)
        self.assertTrue(np.all(panel.decoration[41:] == 255))
        self.assertFalse(panel.decoration[:35].any())
        # A single station logo in the corner does not turn into a full-width border.
        samples[:, 40:, 15:] = 133
        panel.learn_decoration(samples)
        self.assertFalse(panel.decoration[42:, 30:].any())

    def test_profile_preserves_scene_print_outside_the_dialogue_row(self):
        from lite_profile import BandProfile,find_rows
        samples=[[(70,52,310,84),(10,4,46,24)]]*12
        rows=find_rows(samples,400,100)
        profile=BandProfile(400,100,rows,None)
        self.assertEqual(profile.select(samples[0]),[samples[0][0]])
        self.assertEqual(profile.select([(10,4,46,24)]),[])

    def test_two_dialogue_rows_keep_short_words_and_exclude_printed_titles(self):
        from lite_profile import BandProfile
        profile = BandProfile(500, 150, [(60,48), (99,48)], None)
        top, short = (65,34,434,86), (195,75,308,124)
        title = (122,0,378,40)
        self.assertCountEqual(profile.dialogue_rows([title,top,short]), [top,short])
        fragments = [(76,12,169,59), (152,16,299,62), (294,14,427,61), (176,48,327,102)]
        print_on_shirt = (25,95,245,144)
        self.assertCountEqual(profile.dialogue_rows([print_on_shirt,*fragments]), fragments)
        self.assertEqual(profile.dialogue_rows([title,print_on_shirt]), [])

    def test_translucent_word_highlight_can_move_outside_current_ocr_bounds(self):
        from lite_highlight import learn_highlight
        rng = np.random.default_rng(33)
        frames, scenes = [], []
        positions = [18,64,121,158,85,20,126,47,155,90,21]
        for left in positions:
            clean = cv2.resize(rng.integers(32,230,(4,6,3),dtype=np.uint8),(244,84))
            source = clean.copy()
            cv2.putText(source,'WORD',(left+2,50),cv2.FONT_HERSHEY_SIMPLEX,.45,(248,248,248),1)
            chip = source[25:59,left:left+68].astype(np.float32)
            source[25:59,left:left+68] = np.clip(.53*chip+[93,46,14],0,255).astype(np.uint8)
            frames.append(source)
            scenes.append(clean)
        model = learn_highlight(frames, (0,0,244,84))
        self.assertIsNotNone(model)
        mask = model.mask(frames[2], (0,0,244,84))
        self.assertTrue(np.all(mask[25:59,122:187]))
        self.assertFalse(mask[:15].any())
        self.assertIsNone(model.mask(scenes[2], (0,0,244,84)))

    def test_transient_pruning_preserves_confirmed_neighbors_and_clip_ends(self):
        from lite_tracking import reject_transient_skin
        frames=[np.full((45,100,3),(130,163,211),np.uint8) for _ in range(3)]
        box=(20,9,77,37)
        self.assertEqual(reject_transient_skin(frames,[(),(box,),()],None,None),[(),(),()])
        self.assertEqual(reject_transient_skin(frames,[(box,),(box,),()],None,None)[1],(box,))
        self.assertEqual(reject_transient_skin(frames,[(),(),(box,)],None,None)[-1],(box,))

    def test_tracked_card_handles_one_word_and_protects_print_during_the_gap(self):
        from lite_masks import DarkCardTrack
        track = DarkCardTrack()
        for left,right,text in [(44,360,'VOICE OVER'),(168,230,'OK')]:
            frame = np.full((135,400,3),110,np.uint8)
            frame[42:92,left:right] = 0
            cv2.putText(frame,text,(left+8,75),cv2.FONT_HERSHEY_SIMPLEX,.75,(222,222,222),2)
            words = [(left+4,48,right-4,84)]
            kept,mask = track.update(frame,words)
            self.assertEqual(kept,words)
            self.assertIsNotNone(mask)
            self.assertTrue(np.all(mask[42:92,left:right]))
        kept,mask = track.update(np.full_like(frame,110),[(33,0,68,25)])
        self.assertEqual(kept,[])
        self.assertIsNone(mask)

    def test_brush_card_extends_past_the_last_character(self):
        from lite_masks import anchored_brush_card
        frame = np.full((110,520,3),54,np.uint8)
        frame[45:95,10:505] = (185,250,250)
        cv2.putText(frame,'TESTING THE BRUSH',(43,67),cv2.FONT_HERSHEY_SIMPLEX,1.25,(0,0,0),8)
        cv2.putText(frame,'TESTING THE BRUSH',(43,67),cv2.FONT_HERSHEY_SIMPLEX,1.25,(248,248,248),2)
        mask = anchored_brush_card(frame,[(38,22,460,79)])
        self.assertIsNotNone(mask)
        self.assertTrue(np.all(mask[80:94,12:502]))
        self.assertFalse(mask[5].any())
        self.assertIsNone(anchored_brush_card(frame,[]))

    def test_caption_free_foreground_cannot_trigger_outline_repair(self):
        frame = np.full((90,300,3),88,np.uint8)
        original = frame.copy()
        with patch('lite.large_caption_guidance',return_value=(True,np.full(frame.shape[:2],255,np.uint8))):
            restore_joint([frame],[()],[(0,0,300,90)],None,LiteOptions(mask_mode='tracked_cards'))
        self.assertTrue(np.array_equal(frame,original))

    def test_lama_resolution_limit_preserves_output_size_and_pixels_outside_mask(self):
        frame = np.full((160, 480, 3), (70, 110, 150), np.uint8)
        original = frame.copy()
        sizes=[]
        def inpaint(images, masks):
            sizes.append(images[0].shape[:2])
            self.assertEqual(images[0].shape[:2],masks[0].shape)
            return [np.full_like(im, 21) for im in images]
        painter = SimpleNamespace(device='cpu',_inpaint_batch=inpaint)
        restore_joint([frame], [[(50, 40, 430, 118)]], [(0, 0, 480, 160)], painter,
                      LiteOptions(mask_mode='joint_cards',inpaint_max_side=260,context_padding=20))
        self.assertLessEqual(max(sizes[0]),260)
        self.assertTrue(np.all(frame[45:115,70:400]==21))
        self.assertTrue(np.array_equal(frame[:30],original[:30]))

    def test_dark_card_boundaries_survive_touching_black_scene_object(self):
        from lite_masks import black_caption_cards
        frame = np.full((130, 400, 3), 90, np.uint8)
        frame[:, 163:190] = 0
        frame[38:94, 55:337] = 0
        cv2.putText(frame, 'VOICE OVER', (85, 76), cv2.FONT_HERSHEY_SIMPLEX,
                    1., (218, 218, 218), 2)
        boxes = [(79, 47, 278, 83)]
        mask = black_caption_cards(frame, boxes)
        self.assertIsNotNone(mask)
        self.assertTrue(np.all(mask[38:94, 55:337]))
        self.assertFalse(mask[10, 170])
        self.assertFalse(mask[120, 170])
        frame[:] = 0
        cv2.putText(frame, 'VOICE OVER', (85, 76), cv2.FONT_HERSHEY_SIMPLEX, 1., (218, 218, 218), 2)
        self.assertIsNone(black_caption_cards(frame, boxes))

    def test_tracking_detects_one_frame_line_between_empty_anchors(self):
        from lite_tracking import track_block
        frames = [np.full((64, 230, 3), 60, np.uint8) for _ in range(9)]
        cv2.putText(frames[2], 'ONE FRAME', (30,42), cv2.FONT_HERSHEY_SIMPLEX, .8, (245,245,245),2)
        target=((25,16,190,51),)
        def find(frame):
            return target if frame.max() > 200 else ()
        selected,end=track_block(frames,(),find,[(0,0,230,64)])
        self.assertEqual(selected,[(),(),target,(),(),(),(),()])
        self.assertEqual(end,())

    def test_tracking_localizes_a_subtitle_change_without_ocr_on_every_frame(self):
        from lite_tracking import track_block
        a,b=((10,10,90,40),),((90,10,180,40),)
        frames=[np.full((56,200,3),i,np.uint8) for i in range(9)]
        seen=[]
        def find(frame):
            i=int(frame[0,0,0])
            seen.append(i)
            return a if i<4 else b
        selected,end=track_block(frames,a,find,[(0,0,200,56)])
        self.assertEqual(selected,[a]*4+[b]*4)
        self.assertEqual(end,b)
        self.assertLess(len(seen),8)
        self.assertEqual(track_block(frames[:1],a,find,[(0,0,200,56)]),([],a))

    def test_focused_mask_keeps_smaller_peripheral_game_prompt(self):
        frame = np.full((80, 600, 3), 85, np.uint8)
        cv2.putText(frame, 'COMMENTARY', (180, 46), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(frame, 'UI', (553, 48), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        src = frame.copy()
        painter = SimpleNamespace(device='cpu', _inpaint_batch=lambda imgs, masks:
                                  [np.full_like(im, 21) for im in imgs])
        options = LiteOptions(mask_mode='focused_cards', box_padding=3)
        restore_joint([frame], [[(172, 17, 394, 54), (545, 36, 580, 52)]],
                      [(0, 0, 600, 80)], painter, options)
        self.assertTrue(np.array_equal(frame[:, 520:], src[:, 520:]))
        self.assertFalse(np.array_equal(frame[:, 170:400], src[:, 170:400]))

    def test_focused_mask_keeps_a_standalone_short_subtitle_and_both_lines(self):
        painter = SimpleNamespace(device='cpu', _inpaint_batch=lambda imgs, masks:
                                  [np.full_like(im, 21) for im in imgs])
        opts = LiteOptions(mask_mode='focused_cards', box_padding=3)
        for boxes in [[(540, 42, 580, 60)], [(130, 15, 455, 40), (180, 45, 430, 68)]]:
            frame = np.full((80, 600, 3), 85, np.uint8)
            restore_joint([frame], [boxes], [(0, 0, 600, 80)], painter, opts)
            for x, y, r, b in boxes:
                self.assertEqual(int(frame[(y+b)//2, (x+r)//2, 0]), 21)

    def test_focused_mask_preserves_the_space_around_outlined_letters(self):
        frame = np.full((72, 300, 3), (110, 180, 110), np.uint8)
        for index, letter in enumerate('SAMPLE'):
            origin = (34 + index*36, 47)
            cv2.putText(frame, letter, origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
            cv2.putText(frame, letter, origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        original = frame.copy()
        painter = SimpleNamespace(device='cpu', _inpaint_batch=lambda imgs, masks:
                                  [np.full_like(im, 21) for im in imgs])
        restore_joint([frame], [[(26, 17, 247, 58)]], [(0, 0, 300, 72)], painter,
                      LiteOptions(mask_mode='focused_cards', box_padding=6))
        ink = (original.min(axis=2) > 245) | (original.max(axis=2) < 10)
        self.assertTrue(np.all(frame[ink] == 21))
        self.assertTrue(np.array_equal(frame[56, 100], original[56, 100]))

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
