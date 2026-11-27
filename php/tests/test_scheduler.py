"""Exercise the public worker against a disposable queue with controlled GPU capacity."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

PHP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHP / 'bin'))

FAKE_ENGINE = '''#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path
folder = Path(sys.argv[3])
request = json.loads((folder / 'request.json').read_text())
with (folder / 'starts').open('a') as f:
    f.write(str(os.getpid()) + '\\n')
print('JOB_STATUS {"percent":15,"stage":"fixture running"}', flush=True)
if request.get('oom_once') and len((folder / 'starts').read_text().splitlines()) == 1:
    time.sleep(.3)
    print('JOB_STATUS {"error":"CUDA out of memory"}', flush=True)
    sys.exit(1)
until = time.monotonic() + 10
while not (folder / 'release').exists():
    if time.monotonic() > until:
        sys.exit(23)
    time.sleep(.025)
if request.get('fail'):
    print('JOB_STATUS {"error":"isolated test failure"}', flush=True)
    sys.exit(1)
(folder / 'result.mp4').write_bytes(folder.name.encode())
'''
FAKE_SMI = '''#!/usr/bin/env python3
import os
import sys
from pathlib import Path
if any('query-compute-apps' in arg for arg in sys.argv):
    sys.exit(0)
print('GPU-fixture, 16000, ' + Path(os.environ['FIXTURE_FREE']).read_text() + ', 12')
'''


class ReservationTests(unittest.TestCase):
    def test_large_gpu_ocr_jobs_reserve_the_measured_memory_and_respect_host_overrides(self):
        from unittest.mock import patch
        from resources import Settings

        with tempfile.TemporaryDirectory() as path, patch.dict(os.environ, {'LITE_DEVICE': 'cuda'}):
            directory = Path(path)
            request = {'dimensions': [1620, 2880], 'regions': [{'type': 'subtitle'}],
                       'resolved_plan': {'parameters': {'lite': {'ocr_backend': 'onnx_cuda'}}}}
            (directory/'request.json').write_text(json.dumps(request))
            estimate = Settings().estimate(directory, 'lite')
            self.assertEqual(estimate['gpu_mib'], 8192)
            self.assertGreater(estimate['ram_mib'], 6415)
            settings = Settings({'profiles': {'lite': {'gpu_mib': 6200}}})
            self.assertEqual(settings.estimate(directory, 'lite')['gpu_mib'], 6200)


class QueueTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.queue = self.root / 'queue'
        config = self.root / 'scheduler.json'
        profiles = {key: dict(gpu_mib=gpu, ram_mib=1, cpu_cores=0.01, disk_mib=1)
                    for key, gpu in [('lite', 2000), ('clear', 6000), ('propainter', 3000)]}
        config.write_text(json.dumps(dict(profiles=profiles, headroom=dict(gpu_mib=400, ram_mib=0,
                                   cpu_cores=0, disk_mib=0), poll_seconds=.1, max_bypasses=1)))
        engine = self.root / 'model-python'
        engine.write_text(FAKE_ENGINE)
        engine.chmod(0o755)
        smi = self.root / 'nvidia-smi'
        smi.write_text(FAKE_SMI)
        smi.chmod(0o755)
        self.free = self.root / 'free'
        self.free.write_text('10000')
        self.env = dict(os.environ, SUBTITLE_DATA_DIR=str(self.queue), VSR_PYTHON=str(engine),
                        CLEAR_PYTHON=str(engine), SUBTITLE_SCHEDULER_CONFIG=str(config),
                        SUBTITLE_MAX_JOBS='3', CUDA_VISIBLE_DEVICES='0', LITE_DEVICE='cuda',
                        PATH=str(self.root) + os.pathsep + os.environ['PATH'], FIXTURE_FREE=str(self.free))
        subprocess.run(['php', '-r', "require $argv[1]; db();", str(PHP / 'common.php')],
                       env=self.env, check=True)
        self.db = sqlite3.connect(self.queue / 'queue.sqlite', timeout=5)
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)
        self.sequence = 0
        self.workers = []
        self.addCleanup(self.cleanup_workers)

    def cleanup_workers(self):
        for file in self.queue.glob('*/request.json'):
            (file.parent / 'release').touch()
        for worker in self.workers:
            if worker.poll() is None:
                worker.terminate()
            worker.wait(timeout=5)

    def job(self, mode, **options):
        self.sequence += 1
        job = format(self.sequence, '032x')
        folder = self.queue / job
        folder.mkdir()
        (folder / 'source.video').write_bytes(b'fixture')
        (folder / 'request.json').write_text(json.dumps(dict(mode=mode, dimensions=[32, 32],
                           regions=[dict(type='subtitle', roi=[0, 0, 32, 32])], **options)))
        self.db.execute("INSERT INTO jobs(id,name,mode,state,stage,created) VALUES(?,?,?,'queued','queued',?)",
                        (job, job, mode, time.time()))
        self.db.commit()
        return folder

    def worker(self, *args, **env):
        process = subprocess.Popen(['php', str(PHP / 'bin/worker.php'), *args], env=dict(self.env, **env))
        self.workers.append(process)
        return process

    def wait(self, predicate):
        until = time.monotonic() + 7
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(.04)
        self.fail('timeout; jobs=' + repr([tuple(r) for r in self.db.execute('select id,state,stage,error from jobs')]))

    def state(self, folder):
        return self.db.execute('SELECT state FROM jobs WHERE id=?', (folder.name,)).fetchone()[0]

    def runs(self, folder):
        return self.db.execute('SELECT * FROM job_runs WHERE job_id=? ORDER BY started_at', (folder.name,)).fetchall()

    def test_admission_uses_real_free_capacity_and_bounded_backfill(self):
        first, heavy, light, younger = [self.job(m) for m in ['pro', 'pro', 'lite', 'lite']]
        self.free.write_text('1500')
        service = self.worker('--drain')
        self.wait(lambda: '显存' in self.db.execute('SELECT stage FROM jobs WHERE id=?', (first.name,)).fetchone()[0])
        self.assertTrue(all(self.state(f) == 'queued' for f in (first, heavy, light, younger)))
        self.free.write_text('10000')
        self.wait(lambda: (first / 'starts').exists() and (light / 'starts').exists())
        self.assertEqual(self.state(heavy), 'queued')
        (light / 'release').touch()
        self.wait(lambda: self.state(light) == 'done')
        self.assertEqual(self.state(younger), 'queued')
        (first / 'release').touch()
        self.wait(lambda: (heavy / 'starts').exists() and (younger / 'starts').exists())
        for folder in (heavy, younger):
            (folder / 'release').touch()
        self.assertEqual(service.wait(timeout=6), 0)
        a, b = self.runs(first)[0], self.runs(light)[0]
        self.assertLess(b['started_at'], a['finished_at'])
        self.assertLess(a['started_at'], b['finished_at'])
        self.assertTrue(all(self.state(f) == 'done' for f in (first, heavy, light, younger)))
        for folder in (first, light):
            run = self.runs(folder)[0]
            samples = [json.loads(s) for s in (folder / 'runs' / run['run_id'] / 'metrics.jsonl').read_text().splitlines()]
            self.assertTrue(any(s['gpu_free_mib'] == 10000 and s['job_rss_mib'] is not None for s in samples))
            self.assertEqual((folder / 'result.mp4').read_bytes(), folder.name.encode())

    def test_serial_cap_and_isolated_failure(self):
        bad, good = self.job('lite', fail=True), self.job('lite')
        service = self.worker('--drain', SUBTITLE_MAX_JOBS='1')
        self.wait(lambda: (bad / 'starts').exists())
        self.assertEqual(self.state(good), 'queued')
        (bad / 'release').touch()
        self.wait(lambda: (good / 'starts').exists())
        self.assertEqual(self.state(bad), 'failed')
        self.assertGreaterEqual(self.runs(good)[0]['started_at'], self.runs(bad)[0]['finished_at'])
        (good / 'release').touch()
        self.assertEqual(service.wait(timeout=6), 0)
        self.assertEqual(self.state(good), 'done')

    def test_overlap_oom_retries_once_in_exclusive_slot(self):
        sibling, victim = self.job('pro'), self.job('lite', oom_once=True)
        service = self.worker('--drain')
        self.wait(lambda: (victim / 'starts').exists() and self.state(victim) == 'queued')
        self.assertEqual(self.runs(victim)[0]['outcome'], 'oom')
        self.assertEqual(self.state(sibling), 'running')
        (sibling / 'release').touch()
        (victim / 'release').touch()
        self.assertEqual(service.wait(timeout=6), 0)
        attempts = self.runs(victim)
        self.assertEqual([r['outcome'] for r in attempts], ['oom', 'done'])
        self.assertEqual(attempts[1]['exclusive'], 1)
        self.assertGreaterEqual(attempts[1]['started_at'], self.runs(sibling)[0]['finished_at'])

    def test_restart_adopts_live_job_and_releases_orphaned_attempt(self):
        live = self.job('lite')
        old = self.worker()
        self.wait(lambda: (live / 'starts').exists())
        old.kill()
        old.wait(timeout=5)
        new = self.worker()
        lost = self.job('lite')
        self.wait(lambda: (lost / 'starts').exists())
        self.assertEqual(len(self.runs(live)), 1)
        os.kill(self.runs(lost)[0]['runner_pid'], signal.SIGKILL)
        self.wait(lambda: self.state(lost) == 'failed')
        self.assertEqual(self.runs(lost)[0]['outcome'], 'interrupted')
        (live / 'release').touch()
        self.wait(lambda: self.state(live) == 'done')
        self.assertEqual(len((live / 'starts').read_text().splitlines()), 1)
        new.terminate()
        new.wait(timeout=5)


class ResourceTests(unittest.TestCase):
    def test_subtracts_unconsumed_reservations_from_observed_free_resources(self):
        from resources import remaining

        free = dict(gpu_mib=10000, ram_mib=12000, cpu_cores=5, disk_mib=6000)
        headroom = dict(gpu_mib=500, ram_mib=1000, cpu_cores=.5, disk_mib=100)
        reserved = dict(gpu_mib=6000, ram_mib=4000, cpu_cores=3, disk_mib=1500)
        usage = dict(job_gpu_mib=4000, job_rss_mib=1500, job_cpu_pct=200)
        self.assertEqual(remaining(free, headroom, [(reserved, usage)]),
                         dict(gpu_mib=7500, ram_mib=8500, cpu_cores=3.5, disk_mib=4400))
        usage['job_gpu_mib'] = None
        self.assertEqual(remaining(free, headroom, [(reserved, usage)])['gpu_mib'], 3500)

    def test_per_job_gpu_attribution_excludes_other_visible_processes_and_marks_unknown(self):
        from resources import job_sample

        gpu = dict(uuid='GPU-test', total=16000, free=5000, util=97, processes={11: 3400, 21: 800})
        snap = dict(ts=123, gpu=gpu, processes={10: (10, 10, 3), 11: (10, 200, 125), 21: (20, 250, 80)},
                    free=dict(ram_mib=4030, cpu_cores=2, disk_mib=15500))
        own = job_sample(snap, 10)
        self.assertEqual((own['job_gpu_mib'], own['job_rss_mib'], own['job_cpu_pct']), (3400, 210, 128))
        gpu['processes'][100021] = 200
        self.assertIsNone(job_sample(snap, 10)['job_gpu_mib'])


if __name__ == '__main__':
    unittest.main()
