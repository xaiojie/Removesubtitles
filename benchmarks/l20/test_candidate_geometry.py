import unittest

import run_candidate


class CandidateGeometryChecks(unittest.TestCase):
    def setUp(self):
        self.regions = [
            {"type": "subtitle", "roi": [36, 455, 927, 542]},
            {"type": "watermark", "roi": [900, 0, 960, 40]},
        ]
        self.clip = {"width": 960, "height": 544, "regions": self.regions}
        self.probe = {"width": 960, "height": 544, "sample_aspect_ratio": "136:135"}

    def test_anamorphic_selection_uses_browser_edges_and_excludes_encoder_pad(self):
        g = run_candidate.benchmark_geometry(self.clip, self.probe)
        self.assertEqual(g["display_dimensions"], [967, 544])
        self.assertEqual(g["encoded_dimensions"], [968, 544])
        self.assertEqual(g["coded_regions"], self.regions)
        self.assertEqual(g["display_regions"], [
            {"type": "subtitle", "roi": [36, 455, 934, 542]},
            {"type": "watermark", "roi": [906, 0, 967, 40]},
        ])
        self.assertEqual(self.clip["regions"][0]["roi"], [36, 455, 927, 542])

    def test_orthogonal_rotation_follows_ffprobes_counterclockwise_convention(self):
        expected = {
            90: ([544, 967], [455, 33, 542, 931]),
            180: ([967, 544], [33, 2, 931, 89]),
            -90: ([544, 967], [2, 36, 89, 934]),
        }
        for degrees, (size, roi) in expected.items():
            with self.subTest(degrees=degrees):
                probe = dict(self.probe, side_data_list=[{"rotation": degrees}])
                g = run_candidate.benchmark_geometry(self.clip, probe)
                self.assertEqual(g["display_dimensions"], size)
                self.assertEqual(g["display_regions"][0]["roi"], roi)

    def test_square_pixels_and_legacy_single_roi_keep_the_request(self):
        g = run_candidate.benchmark_geometry(
            {"width": 960, "height": 544, "roi": self.regions[0]["roi"]},
            dict(self.probe, sample_aspect_ratio="0:1"),
        )
        self.assertEqual(g["display_dimensions"], [960, 544])
        self.assertEqual(g["encoded_dimensions"], [960, 544])
        self.assertEqual(g["display_regions"], self.regions[:1])

    def test_half_pixel_dimensions_round_up_and_invalid_coded_dimensions_fail(self):
        clip = {"width": 10, "height": 6, "roi": [0, 0, 10, 6]}
        probe = {"width": 10, "height": 6, "sample_aspect_ratio": "21:20"}
        self.assertEqual(run_candidate.benchmark_geometry(clip, probe)["display_dimensions"], [11, 6])
        with self.assertRaises(ValueError):
            run_candidate.benchmark_geometry(dict(clip, width=11), probe)

    def test_only_square_padded_complete_output_validates(self):
        g = run_candidate.benchmark_geometry(self.clip, self.probe)
        source = dict(self.probe, nb_read_frames="750", avg_frame_rate="25/1")
        final = dict(source, width=968, sample_aspect_ratio="1:1")
        run_candidate.validate_output(source, final, g)
        for change in ({"width": 960}, {"width": 967}, {"nb_read_frames": "749"},
                       {"avg_frame_rate": "24/1"}, {"sample_aspect_ratio": "136:135"},
                       {"side_data_list": [{"rotation": 180}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_candidate.validate_output(source, dict(final, **change), g)


if __name__ == "__main__":
    unittest.main()
