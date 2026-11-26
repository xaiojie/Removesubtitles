"""Reserve single-card/host resources, supervise independent attempts and preserve the PHP queue contract."""

import argparse
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

from resources import KEYS, LABELS, Probe, Settings, job_sample, remaining, working_mib

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = '''
CREATE TABLE IF NOT EXISTS job_runs(
    run_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, queued_at REAL NOT NULL, started_at REAL NOT NULL,
    finished_at REAL, outcome TEXT, error TEXT, runner_pid INTEGER, reservation_json TEXT NOT NULL,
    exclusive INTEGER NOT NULL DEFAULT 0, overlapped INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS job_runs_history ON job_runs(job_id,started_at);
CREATE UNIQUE INDEX IF NOT EXISTS job_runs_active ON job_runs(job_id) WHERE finished_at IS NULL;
CREATE TABLE IF NOT EXISTS scheduler_wait(
    job_id TEXT PRIMARY KEY, bypasses INTEGER NOT NULL DEFAULT 0
);
'''


def connect(directory):
    db = sqlite3.connect(str(directory / 'queue.sqlite'), timeout=15)
    db.row_factory = sqlite3.Row
    return db


def append_json(file, record):
    with file.open('ab', buffering=0) as stream:
        stream.write((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode())


def attempt_dir(directory, row):
    return directory / row['job_id'] / 'runs' / row['run_id']


def finish(db, run_id, outcome, reason=None):
    with db:
        # Acquire the write transaction before reading overlap. A second launch may mark this run concurrently.
        updated = db.execute('UPDATE job_runs SET finished_at=?,outcome=?,error=? WHERE run_id=? AND finished_at IS NULL',
                             (time.time(), outcome, reason, run_id))
        if updated.rowcount != 1:
            return
        run = db.execute('SELECT * FROM job_runs WHERE run_id=?', (run_id,)).fetchone()
        job = run['job_id']
        if outcome == 'done':
            db.execute("UPDATE jobs SET state='done',percent=100,stage='处理完成',error=NULL WHERE id=?", (job,))
        elif outcome == 'oom' and run['overlapped'] and not run['exclusive']:
            # Exactly one automatic retry; retain mode, parameters and completed model-stage artifacts.
            fields = {col[1] for col in db.execute('PRAGMA table_info(jobs)')}
            stamp = ', queued_at=?' if 'queued_at' in fields else ''
            args = (time.time(), job) if stamp else (job,)
            db.execute("UPDATE jobs SET state='queued',percent=0,stage='显存不足，等待独占重试',error=NULL" +
                       stamp + " WHERE id=?", args)
        else:
            label = '任务中断' if outcome == 'interrupted' else '处理失败'
            db.execute("UPDATE jobs SET state='failed',stage=?,error=? WHERE id=?", (label, reason, job))


def execute(directory, run_id, lease):
    # This independent process group survives a scheduler restart and owns only a single job.
    db = connect(directory)
    row = db.execute('SELECT * FROM job_runs WHERE run_id=? AND finished_at IS NULL', (run_id,)).fetchone()
    if row is None:
        return 1
    folder = directory / row['job_id']
    run = attempt_dir(directory, row)
    mode = db.execute('SELECT mode FROM jobs WHERE id=?', (row['job_id'],)).fetchone()[0]
    env = os.environ.copy()
    reserve = json.loads(row['reservation_json'])
    env.update(SUBTITLE_RUN_ID=run_id, SUBTITLE_TRACE_FILE=str(run / 'stages.jsonl'), TMPDIR=str(run / 'tmp'))
    if reserve.get('gpu_uuid') and reserve['gpu_mib']:
        env['CUDA_VISIBLE_DEVICES'] = reserve['gpu_uuid']
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env.setdefault(name, str(max(1, math.ceil(reserve['cpu_cores']))))
    python = (os.getenv('VSR_PYTHON') or ROOT / 'video-subtitle-remover/.venv/bin/python') if mode == 'lite' else (
        os.getenv('CLEAR_PYTHON') or ROOT / '.venv/bin/python')
    outcome = 'failed'
    reason = None
    try:
        with db:
            db.execute('UPDATE job_runs SET runner_pid=? WHERE run_id=?', (os.getpid(), run_id))
        if reserve.get('cpu_ids') and hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, set(reserve['cpu_ids']))
        (run / 'tmp').mkdir(exist_ok=True)
        (folder / 'result.mp4').unlink(missing_ok=True)
        if not os.access(python, os.X_OK):
            raise RuntimeError('推理 Python 不可用，请检查对应模型环境')
        command = [str(python), '-u', str(ROOT / 'php/inference/run.py'), str(folder), mode]
        oom = False
        with (folder / 'worker.log').open('a', encoding='utf8', buffering=1) as log:
            print(f'\n[{run_id}] {mode} pid={os.getpid()}', file=log)
            with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', env=env, pass_fds=(lease,)) as proc:
                for line in proc.stdout:
                    log.write(line)
                    message = line.lower()
                    oom |= any(marker in message for marker in ('out of memory', '显存不足', 'cuda_error_out_of_memory',
                                                               'memoryerror', 'unable to allocate'))
                    if not line.startswith('JOB_STATUS '):
                        continue
                    try:
                        event = json.loads(line[len('JOB_STATUS '):])
                        if event.get('error'):
                            reason = str(event['error'])[:540]
                        if 'percent' in event and 'stage' in event:
                            with db:
                                db.execute('UPDATE jobs SET percent=max(percent,?),stage=? WHERE id=? AND state=\'running\'',
                                           (min(99, max(0, int(event['percent']))), str(event['stage']), row['job_id']))
                    except (AttributeError, TypeError, ValueError):
                        continue
                code = proc.wait()
        if code:
            if oom:
                outcome = 'oom'
                raise RuntimeError(reason or '推理内存不足，请检查任务日志')
            raise RuntimeError(reason or f'推理进程退出（{code}），请检查任务日志')
        result = folder / 'result.mp4'
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError('推理未生成成片')
        outcome = 'done'
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        reason = str(exc)[:540]
    finally:
        shutil.rmtree(run / 'tmp', ignore_errors=True)
        finish(db, run_id, outcome, reason)
    return 0


def held(path):
    try:
        with path.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
    except BlockingIOError:
        return True
    except FileNotFoundError:
        return False


def stamped_members(run_id, pids):
    marker = b'SUBTITLE_RUN_ID=' + run_id.encode()
    for pid in pids:
        try:
            if marker in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
                return True
        except OSError:
            pass
    return False


class Scheduler:
    def __init__(self, directory, settings):
        self.dir = directory
        self.db = connect(directory)
        self.settings = settings
        self.db.executescript(SCHEMA)
        # This runs only after taking the legacy worker.lock. Legacy workers passed it to their inference child.
        with self.db:
            self.db.execute("""UPDATE jobs SET state='failed',stage='任务中断',error='旧队列进程曾退出，可提交重试'
                               WHERE state='running' AND id NOT IN
                               (SELECT job_id FROM job_runs WHERE finished_at IS NULL)""")
        self.probe = Probe(directory)
        self.children = {}
        self.orphans = {}
        self.disk_samples = {}

    def active(self):
        return self.db.execute('SELECT * FROM job_runs WHERE finished_at IS NULL ORDER BY started_at').fetchall()

    def reap(self, rows, snapshot):
        self.children = {pid: p for pid, p in self.children.items() if p.poll() is None}
        live = []
        ids = {row['run_id'] for row in rows}
        self.disk_samples = {key: sample for key, sample in self.disk_samples.items() if key in ids}
        for row in rows:
            group = row['runner_pid']
            sample = job_sample(snapshot, group)
            run = attempt_dir(self.dir, row)
            reserve = json.loads(row['reservation_json'])
            disk_ts, disk = self.disk_samples.get(row['run_id'], (0, None))
            if sample['ts'] - disk_ts >= 5:
                disk = working_mib(self.dir / row['job_id'])
                self.disk_samples[row['run_id']] = (sample['ts'], disk)
            sample['job_work_mib'] = disk
            baseline = reserve.get('start_work_mib')
            sample['job_disk_growth_mib'] = max(0, disk - baseline) if disk is not None and baseline is not None else None
            busy = held(run / 'lease.lock')
            if busy and (not group or group in sample['pids']):
                live.append((row, reserve, sample))
                try:
                    append_json(run / 'metrics.jsonl', sample)
                except OSError as exc:
                    print(f"Cannot write metrics for {row['job_id']}: {exc}", file=sys.stderr, flush=True)
                continue
            # If the per-job supervisor died, terminate only its identified process group before releasing capacity.
            descendants = sample['pids']
            if group and descendants and (busy or stamped_members(row['run_id'], descendants)):
                since = self.orphans.setdefault(row['run_id'], time.monotonic())
                try:
                    os.killpg(group, signal.SIGKILL if time.monotonic() - since >= 3 else signal.SIGTERM)
                except ProcessLookupError:
                    pass
                live.append((row, reserve, sample))
                continue
            if busy:
                # Probe can race a newly created process. Its inherited lease still owns the attempt.
                live.append((row, reserve, sample))
                continue
            finish(self.db, row['run_id'], 'interrupted', '任务进程已退出，可提交重试')
            shutil.rmtree(run / 'tmp', ignore_errors=True)
            self.orphans.pop(row['run_id'], None)
        return live

    def start(self, job, reserve, active, snapshot, skipped):
        run_id = uuid.uuid4().hex
        job_id = job['id']
        if Path(job_id).name != job_id or job_id in ('.', '..'):
            with self.db:
                self.db.execute("UPDATE jobs SET state='failed',stage='处理失败',error='无效任务目录' WHERE id=?", (job_id,))
            return None
        folder = self.dir / job_id
        run = folder / 'runs' / run_id
        if not folder.is_dir():
            with self.db:
                self.db.execute("UPDATE jobs SET state='failed',stage='处理失败',error='找不到任务目录' WHERE id=?", (job_id,))
            return None
        run.mkdir(parents=True)
        claimed = {cpu for _, r, _ in active for cpu in r.get('cpu_ids', [])}
        available = [cpu for cpu in self.probe.allowed if cpu not in claimed]
        reserve = dict(reserve, gpu_uuid=snapshot['gpu']['uuid'], start_work_mib=working_mib(folder),
                       cpu_ids=(available or self.probe.allowed[:1])[:max(1, math.ceil(reserve['cpu_cores']))])
        stamp = time.time()
        queued_at = job['queued_at'] if 'queued_at' in job.keys() and job['queued_at'] else job['created']
        with (run / 'lease.lock').open('a') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            with self.db:
                claim = self.db.execute("UPDATE jobs SET state='running',stage='准备推理',error=NULL WHERE id=? AND state='queued'",
                                        (job_id,))
                if not claim.rowcount:
                    return None
                overlap = self.db.execute('UPDATE job_runs SET overlapped=1 WHERE finished_at IS NULL').rowcount
                self.db.execute('''INSERT INTO job_runs(run_id,job_id,queued_at,started_at,reservation_json,exclusive,overlapped)
                                   VALUES(?,?,?,?,?,?,?)''',
                                (run_id, job_id, queued_at, stamp, json.dumps(reserve), job['exclusive'], bool(overlap)))
                self.db.execute('DELETE FROM scheduler_wait WHERE job_id=?', (job_id,))
                self.db.executemany('''INSERT INTO scheduler_wait(job_id,bypasses) VALUES(?,1)
                                       ON CONFLICT(job_id) DO UPDATE SET bypasses=bypasses+1''', [(j,) for j in skipped])
            try:
                append_json(run / 'stages.jsonl', dict(stage='queue_wait', started_at=queued_at, finished_at=stamp,
                                                       duration_seconds=max(0, stamp-queued_at), pid=os.getpid(), ok=True))
                append_json(run / 'metrics.jsonl', job_sample(snapshot, None))
                with (folder / 'worker.log').open('ab') as log:
                    proc = subprocess.Popen([sys.executable, __file__, '--work', run_id, '--lease-fd', str(lease.fileno())],
                                            start_new_session=True, stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),))
                self.children[proc.pid] = proc
                with self.db:
                    self.db.execute('UPDATE job_runs SET runner_pid=? WHERE run_id=?', (proc.pid, run_id))
            except OSError as exc:
                finish(self.db, run_id, 'failed', str(exc))
        return run_id

    def pending(self):
        return self.db.execute("""SELECT jobs.*, COALESCE(w.bypasses,0) AS bypasses,
                   COALESCE((SELECT outcome='oom' FROM job_runs r WHERE r.job_id=jobs.id
                             ORDER BY started_at DESC LIMIT 1),0) AS exclusive
                   FROM jobs LEFT JOIN scheduler_wait w ON w.job_id=jobs.id WHERE state='queued'
                   ORDER BY exclusive DESC, coalesce(queued_at,created), jobs.rowid""").fetchall()

    def admit(self, pending, active, snapshot):
        if len(active) >= self.settings.max_jobs or any(row['exclusive'] for row, _, _ in active):
            return None
        budget = remaining(snapshot['free'], self.settings.margin, [(r, usage) for _, r, usage in active])
        skipped = []
        for job in pending:
            if job['exclusive'] and active:
                return None
            reserve = self.settings.estimate(self.dir / job['id'], job['mode'])
            blocked = [key for key in KEYS if reserve[key] > budget[key]]
            if not blocked:
                try:
                    return self.start(job, reserve, active, snapshot, skipped)
                except OSError as exc:
                    opened = self.db.execute('SELECT run_id FROM job_runs WHERE job_id=? AND finished_at IS NULL',
                                             (job['id'],)).fetchone()
                    if opened:
                        finish(self.db, opened['run_id'], 'failed', str(exc))
                    else:
                        with self.db:
                            self.db.execute("UPDATE jobs SET state='failed',stage='处理失败',error=? WHERE id=? AND state='queued'",
                                            (str(exc), job['id']))
                    return None
            reason = '等待资源 · ' + '、'.join(LABELS[k] for k in blocked)
            if job['stage'] != reason:
                with self.db:
                    self.db.execute("UPDATE jobs SET stage=? WHERE id=? AND state='queued'", (reason, job['id']))
            # Stop backfilling only for an older job the installed hardware could actually fit.
            possible = all(reserve[k] <= snapshot['total'][k] - self.settings.margin[k] for k in KEYS if reserve[k])
            if possible and job['bypasses'] >= self.settings.bypasses:
                break
            if possible:
                skipped.append(job['id'])
        return None

    def loop(self, once=False, drain=False):
        launched = None
        while True:
            rows = self.active()
            pending = self.pending()
            if not rows and not pending:
                if once or drain:
                    return 0
            else:
                snap = self.probe.sample()
                live = self.reap(rows, snap)
                if launched and once and all(row['run_id'] != launched for row, _, _ in live):
                    return 0
                if pending and (not once or not launched):
                    run_id = self.admit(pending, live, snap)
                    if run_id:
                        launched = run_id
                        # Refresh external usage before granting a second reservation.
                        continue
                    if once and self.pending():
                        print('任务等待可用资源，保留在队列中', file=sys.stderr)
                        return 75
            time.sleep(self.settings.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--once', action='store_true', help='Process at most one job; exit 75 if capacity is unavailable')
    group.add_argument('--drain', action='store_true', help='Process queued jobs and exit once idle')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--work', help=argparse.SUPPRESS)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = Path(os.getenv('SUBTITLE_DATA_DIR') or ROOT / 'php/var').resolve()
    if args.work:
        return execute(directory, args.work, args.lease_fd)
    if args.max_jobs is not None:
        os.environ['SUBTITLE_MAX_JOBS'] = str(args.max_jobs)
    settings = Settings.load()
    with (directory / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            print('已有调度器使用此队列', file=sys.stderr)
            return 1
        return Scheduler(directory, settings).loop(args.once, args.drain)


if __name__ == '__main__':
    sys.exit(main())
