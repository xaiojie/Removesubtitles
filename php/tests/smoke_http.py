"""Local HTTP/queue smoke: Pro and Max on blank video, optionally exercise real Lite weights."""

import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

PHP = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as workspace, socket.socket() as sock:
        temp = Path(workspace)
        env = {**os.environ, 'SUBTITLE_DATA_DIR': str(temp / 'queue'), 'SUBTITLE_API_TOKEN': 'test-only'}
        if not shutil.which('nvidia-smi') and not env.get('SUBTITLE_SCHEDULER_CONFIG'):
            # This tiny blank-video smoke takes CLEAR's CPU-only no-text path. The production GPU probe stays live.
            fixture = temp / 'smoke-capacity.json'
            fixture.write_text(json.dumps({
                'profiles': {stage: {'gpu_mib': 0, 'ram_mib': 256, 'cpu_cores': 1, 'disk_mib': 64}
                             for stage in ('lite', 'clear', 'propainter')},
                'headroom': {'gpu_mib': 0, 'ram_mib': 128, 'cpu_cores': 0, 'disk_mib': 128},
                'poll_seconds': .2,
            }))
            env['SUBTITLE_SCHEDULER_CONFIG'] = str(fixture)
        sock.bind(('127.0.0.1', 0))
        host = f'127.0.0.1:{sock.getsockname()[1]}'
        sock.close()
        base = f'http://{host}'

        def call(path, method='GET'):
            req = urllib.request.Request(
                base + path, method=method, headers={'Authorization': 'Bearer test-only'}
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, r.read(), r.headers

        def upload(path, mode=None, kind='subtitle', roi=None):
            region = {'type': kind, 'roi': roi or [4, 12, 80, 40]}
            cmd = [
                'curl',
                '-sS',
                '-w',
                '\n%{http_code}',
                '-H',
                'Authorization: Bearer test-only',
                '-F',
                f'video=@{path};filename=样片.mp4',
                '-F',
                'regions=' + json.dumps([region]),
            ]
            if mode:
                cmd += ['-F', 'mode=' + mode]
            data, code = subprocess.check_output(cmd + [base + '/api/jobs'], text=True).rsplit('\n', 1)
            return int(code), json.loads(data)

        def work():
            subprocess.run(['php', str(PHP / 'bin/worker.php'), '--once'], env=env, check=True, timeout=300)

        srv = subprocess.Popen(
            ['php', '-S', host, '-t', str(PHP / 'public'), str(PHP / 'public/index.php')],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(100):
                try:
                    modes = json.loads(call('/api/modes')[1])
                    break
                except OSError:
                    time.sleep(0.05)
            assert modes['default'] == 'pro'
            assert {k: v['label'] for k, v in modes['modes'].items()} == {
                'lite': 'Lite · 快速', 'pro': 'Pro · 标准', 'max': 'Max · 增加'
            }
            try:
                urllib.request.urlopen(base + '/api/modes', timeout=5)
                raise AssertionError('unauthenticated request succeeded')
            except urllib.error.HTTPError as error:
                assert error.code == 401
            src = temp / 'blank.mp4'
            subprocess.run(
                [
                    'ffmpeg',
                    '-nostdin',
                    '-v',
                    'error',
                    '-f',
                    'lavfi',
                    '-i',
                    'color=c=blue:size=96x64:rate=12',
                    '-f',
                    'lavfi',
                    '-i',
                    'sine=frequency=500',
                    '-t',
                    '1',
                    '-c:v',
                    'libx264',
                    '-c:a',
                    'aac',
                    str(src),
                ],
                check=True,
            )
            assert upload(src, 'lite', 'watermark')[0] == 422
            for invalid in ('standard', 'fine', 'unknown'):
                code, error = upload(src, invalid)
                assert code == 422 and error['error'] == 'mode 可选 lite、pro、max'
            assert upload(src, roi=[0, 0, 300, 400])[0] == 422
            todo = [(src, None), (src, 'max')]
            if os.getenv('LITE_TEST_VIDEO'):
                sys.path.insert(0, str(PHP.parent / 'benchmarks/l20'))
                from clip_limits import check_clip

                clip = Path(os.environ['LITE_TEST_VIDEO'])
                check_clip(clip)
                todo.append((clip, 'lite'))
            for video, mode in todo:
                roi = json.loads(os.environ['LITE_TEST_ROI']) if mode == 'lite' else None
                code, job = upload(video, mode, roi=roi)
                assert code == 202 and job['mode'] == (mode or 'pro'), (code, job)
                path = job['status_url']
                try:
                    call(path + '/output')
                    raise AssertionError('unprocessed download succeeded')
                except urllib.error.HTTPError as error:
                    assert error.code == 409
                work()
                status = json.loads(call(path)[1])
                if status['state'] != 'done':
                    raise AssertionError(
                        (status, (temp / 'queue' / job['id'] / 'worker.log').read_text()[-1800:])
                    )
                assert status['percent'] == 100 and status['mode'] == job['mode']
                assert (temp / 'queue' / job['id'] / 'review.json').is_file() == ('selective_review' in job['resolved_plan']['stages'])
                if mode == 'lite':
                    log = (temp / 'queue' / job['id'] / 'worker.log').read_text()
                    detected = re.search(r'Lite: ([0-9]+)/([0-9]+)', log)
                    assert detected and int(detected[1]) > 0, 'Test clip must contain a detectable subtitle'
                    print(detected[0], flush=True)
                _, body, headers = call(status['output_url'])
                assert "filename*=UTF-8''" + quote('样片_处理后.mp4') in headers['Content-Disposition']
                out = temp / f'{mode}.mp4'
                out.write_bytes(body)

                def streams(file):
                    return json.loads(
                        subprocess.check_output(
                            [
                                'ffprobe',
                                '-v',
                                'error',
                                '-show_entries',
                                'stream=codec_type,nb_frames,width,height',
                                '-of',
                                'json',
                                str(file),
                            ]
                        )
                    )['streams']

                orig, result = streams(video), streams(out)
                assert orig[0] == result[0] and any(s['codec_type'] == 'audio' for s in result)
                print(job['mode'], len(body), 'bytes', result[0], flush=True)
            # A missing runtime must yield a retryable failure, not an empty or stale result.
            _, retry = upload(src, 'pro')
            subprocess.run(
                ['php', str(PHP / 'bin/worker.php'), '--once'],
                env={**env, 'CLEAR_PYTHON': '/missing/python'},
                check=True,
            )
            path = retry['status_url']
            assert json.loads(call(path)[1])['state'] == 'failed'
            assert call(path + '/retry', 'POST')[0] == 202
            work()
            assert json.loads(call(path)[1])['state'] == 'done'

            # Reopen a pre-rename queue: its saved request stays usable, including on retry.
            code, archived = upload(src, 'max')
            assert code == 202
            job_dir = temp / 'queue' / archived['id']
            config = job_dir / 'request.json'
            old_request = json.loads(config.read_text())
            old_request['mode'] = 'fine'
            old_request.pop('resolved_plan', None)
            config.write_text(json.dumps(old_request))
            with sqlite3.connect(temp / 'queue' / 'queue.sqlite') as db:
                db.execute("UPDATE jobs SET mode='standard' WHERE id=?", (retry['id'],))
                db.execute("UPDATE jobs SET mode='fine', state='failed' WHERE id=?", (archived['id'],))
                db.execute('PRAGMA user_version=0')
            assert json.loads(call(path)[1])['mode'] == 'pro'
            assert call(archived['status_url'] + '/retry', 'POST')[0] == 202
            work()
            status = json.loads(call(archived['status_url'])[1])
            assert (status['mode'], status['state']) == ('max', 'done') and (job_dir / 'review.json').is_file()
        finally:
            srv.terminate()
            srv.wait(timeout=8)


if __name__ == '__main__':
    main()
