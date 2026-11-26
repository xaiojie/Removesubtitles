import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from summarize_acceptance import batch_data, file_digest, one_clip


class ConcurrentQualityTests(unittest.TestCase):
    def test_an_isolated_review_matches_only_the_file_that_was_seen(self):
        clip = dict(id='c01', style='normal', split='tune', duration_seconds=1, frames=1,
                    width=16, height=16, fps='1/1', roi=[0, 0, 8, 8], sha256='input')
        record = dict(status='ok', directory='trial')
        assessment = dict(verdict='通过', note='checked', text_clean=True, quality_pass=True)
        notes = {'c01': {'lite': assessment}}
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = folder / 'result.mp4'
            result.write_bytes(b'reviewed result')
            trial = dict(validated=True, clip='c01', input_sha256='input', elapsed_seconds=1,
                         source_revision='test', output=result.name)
            (folder / 'trial.json').write_text(json.dumps(trial))
            self.assertNotIn('quality_pass', one_clip(clip, 'lite', record, folder, notes, None))
            assessment['output_sha256'] = file_digest(result)
            self.assertTrue(one_clip(clip, 'lite', record, folder, notes, None)['quality_pass'])
            result.write_bytes(b'regenerated result')
            self.assertNotIn('quality_pass', one_clip(clip, 'lite', record, folder, notes, None))

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
