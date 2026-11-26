import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from summarize_acceptance import batch_data


class ConcurrentQualityTests(unittest.TestCase):
    def test_distinct_outputs_need_their_own_review_and_invalid_frames_never_pass(self):
        jobs = []
        for name in ('same', 'reviewed', 'unknown', 'wrong_frames'):
            jobs.append(dict(id=name, clip='c01', mode='lite', state='done', source_seconds=4,
                             attempts=[{}], peaks={}, output_sha256=name,
                             same_frames_and_dimensions=name != 'wrong_frames', same_fps=True))
        data = dict(jobs=jobs, capacity=3, completed=3, failed=1, batch_seconds=20,
                    completed_source_minutes_per_hour=36, overlaps=[])
        single = {('c01', 'lite'): {'output_sha256': 'same', 'quality_pass': True}}
        notes = {'batch_outputs': {
            key: dict(clip='c01', tier='lite', quality_pass=True, basis='reviewed_batch_file')
            for key in ('reviewed', 'wrong_frames')
        }}
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'batch.json').write_text(json.dumps(data))
            result = batch_data(folder, single, notes)
        self.assertEqual([job['quality_pass'] for job in result['jobs']], [True, True, None, False])
        self.assertEqual(result['quality_pending'], 1)
        self.assertIsNone(result['qualified_source_minutes_per_hour'])


if __name__ == '__main__':
    unittest.main()
