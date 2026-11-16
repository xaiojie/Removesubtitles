from unittest import TestCase
from dataclasses import asdict
import json
from inference_options import InferenceOptions


class InferenceOptionsTests(TestCase):
    def test_old_job_without_preset_keeps_legacy_defaults_and_explicit_options(self):
        options = InferenceOptions.parse('{}')
        self.assertEqual(options.preset, 'custom')
        self.assertTrue(options.adaptive_processing)
        self.assertFalse(options.resident_models)
        self.assertEqual((options.steps, options.context, options.resolution, options.blend_mode),
                         (5, 'frame', 'native', 'lines'))
        explicit = dict(steps=9, context='roi', resolution='auto', adaptive_processing=True,
                        blend_mode='rectangle', vae_tiled=True, lora_scale=1.3)
        parsed = asdict(InferenceOptions.parse(json.dumps(explicit)))
        self.assertEqual({name: parsed[name] for name in explicit}, explicit)

    def test_presets_merge_defaults_before_explicit_overrides(self):
        fast = InferenceOptions.parse('{"preset":"fast"}')
        self.assertEqual((fast.context, fast.steps, fast.lora_scale, fast.resolution),
                         ('roi', 5, 1.2, 'native'))
        balanced = InferenceOptions.parse('{"preset":"balanced"}')
        self.assertEqual((balanced.context, balanced.steps, balanced.lora_scale),
                         ('expanded', 5, 1.0))
        quality = InferenceOptions.parse('{"preset":"quality"}')
        self.assertEqual((quality.context, quality.steps, quality.resolution, quality.blend_mode),
                         ('frame', 12, 'auto', 'precise'))
        for options in (fast, balanced, quality):
            self.assertFalse(options.adaptive_processing)
            self.assertTrue(options.resident_models)
            self.assertEqual((options.chunk_frames, options.overlap_frames, options.seed,
                              options.cfg_scale, options.export_crf, options.segment_mode,
                              options.vae_tiled), (33, 4, 20261113, 1.0, 10, 'shots', False))
        custom = InferenceOptions.parse(
            '{"preset":"quality","steps":7,"context":"roi","resolution":"native",'
            '"blend_mode":"rectangle","resident_models":false,"adaptive_processing":true}')
        self.assertEqual((custom.steps, custom.context, custom.resolution, custom.blend_mode),
                         (7, 'roi', 'native', 'rectangle'))
        self.assertTrue(custom.adaptive_processing)
        self.assertFalse(custom.resident_models)

    def test_rejects_unknown_preset_and_validates_merged_overrides(self):
        for raw in ('{"preset":"unknown"}', '{"preset":null}', '{"preset":true}',
                    '{"preset":"quality","steps":0}', '{"resident_models":1}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                InferenceOptions.parse(raw)

    def test_partial_options_keep_unspecified_defaults(self):
        options = InferenceOptions.parse('{"steps":9,"lora_scale":1.3,"overlap_frames":0}')
        self.assertEqual((options.steps, options.lora_scale, options.overlap_frames), (9, 1.3, 0))
        self.assertEqual(options.chunk_frames, 33)
        self.assertEqual(options.context, "frame")

    def test_rejects_invalid_and_ambiguous_parameters(self):
        for raw in ('{"steps":true}', '{"chunk_frames":20}', '{"overlap_frames":5,"chunk_frames":5}',
                    '{"cfg_scale":NaN}', '{"lora_scale":null}', '{"unknown":1}', '[]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                InferenceOptions.parse(raw)

    def test_time_window_uses_frame_timestamps_and_clamps_to_end(self):
        opts = InferenceOptions.parse('{"start_seconds":0.11,"duration_seconds":0.3}')
        self.assertEqual(opts.frame_range(30, 10), (4, 10))
        self.assertEqual(InferenceOptions.parse('{}').frame_range(30, 10), (0, None))
        with self.assertRaises(ValueError):
            opts.frame_range(30, 4)
        for raw in ('{"start_seconds":-1}', '{"duration_seconds":0}', '{"duration_seconds":NaN}'):
            with self.assertRaises(ValueError):
                InferenceOptions.parse(raw)
