from unittest import TestCase
from inference_options import InferenceOptions


class InferenceOptionsTests(TestCase):
    def test_partial_options_keep_unspecified_defaults(self):
        options = InferenceOptions.parse('{"steps":9,"lora_scale":1.3,"overlap_frames":0}')
        self.assertEqual((options.steps, options.lora_scale, options.overlap_frames), (9, 1.3, 0))
        self.assertEqual(options.chunk_frames, 21)
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
