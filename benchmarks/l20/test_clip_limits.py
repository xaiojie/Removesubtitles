import unittest

from clip_limits import ClipLimitError, checked_duration


class ClipDurationChecks(unittest.TestCase):
    def test_exactly_two_minutes_is_allowed(self):
        self.assertEqual(checked_duration({"streams": [{"codec_type": "video",
            "duration_ts": "3600", "time_base": "1/30"}], "format": {"duration": "120.0"}}), 120)

    def test_rejects_extra_frame_even_when_container_rounds_down(self):
        with self.assertRaises(ClipLimitError):
            checked_duration({"streams": [{"codec_type": "video", "duration_ts": "3601",
                "time_base": "1/30"}], "format": {"duration": "120.0"}})

    def test_rejects_long_container_missing_duration_or_audio_only(self):
        for metadata in (
            {"streams": [{"codec_type": "video", "duration": "119"}], "format": {"duration": "121"}},
            {"streams": [{"codec_type": "video"}], "format": {"duration": "N/A"}},
            {"streams": [{"codec_type": "audio", "duration": "5"}], "format": {"duration": "5"}},
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ClipLimitError):
                checked_duration(metadata)
