import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from check_gpu_contention import audit


class BatchGpuAuditTests(unittest.TestCase):
    def test_peer_jobs_are_counted_together_and_missing_process_samples_are_unknown(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            samples = {
                'a': {'ts': 14, 'pids': [11], 'gpu_processes': {'11': 100, '22': 200, '33': 75}},
                'b': {'ts': 14.2, 'pids': [22], 'gpu_processes': None},
            }
            jobs = []
            for name, sample in samples.items():
                attempt = root / name / 'runs' / 'attempt'
                attempt.mkdir(parents=True)
                (attempt / 'metrics.jsonl').write_text(json.dumps(sample) + '\n')
                jobs.append({'id': name, 'attempts': [{'run_id': 'attempt'}]})
            batch = root / 'batch.json'
            stage = {'stage': 'lite_lama', 'ok': True, 'started_at': 10, 'finished_at': 16}
            batch.write_text(json.dumps({'jobs': jobs, 'timelines': {'a': [stage], 'b': [stage]}}))
            checked = audit(batch, root)
            self.assertEqual(checked['samples'], 2)
            self.assertEqual(checked['process_samples'], 1)
            self.assertEqual(checked['external_samples'], 1)
            self.assertEqual(checked['external_peak_mib'], 75)
            self.assertEqual(checked['at_least_two_in_model_calls_samples'], 1)
            self.assertEqual(checked['max_resident_jobs_in_model_calls'], 2)

            batch.write_text(json.dumps({'jobs': jobs[1:]}))
            self.assertIsNone(audit(batch, root)['external_peak_mib'])
