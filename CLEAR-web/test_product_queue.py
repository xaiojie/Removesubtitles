"""Exercise both real HTTP submission paths using a tiny CPU-generated video and a private SQLite queue."""

import io
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

import app as web

ROOT = Path(__file__).resolve().parents[1]
PHP = ROOT / 'php'
TOKEN = 'private-integration-fixture'
SUB = {'type': 'subtitle', 'roi': [4, 20, 48, 42]}
WM = {'type': 'watermark', 'roi': [3, 3, 13, 9]}


@pytest.fixture
def queue(tmp_path, monkeypatch):
    if not all(shutil.which(tool) for tool in ('php', 'curl', 'ffmpeg', 'ffprobe')):
        pytest.skip('PHP CLI, curl and FFmpeg are required for the HTTP queue integration test')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    data = tmp_path / 'queue'
    data.mkdir()
    managed = tmp_path / '.deploy/api'
    managed.mkdir(parents=True)
    (managed / 'settings.json').write_text(json.dumps({'port': port, 'bind': '0.0.0.0', 'data': str(data)}))
    (managed / 'api.token').write_text(TOKEN)
    for key in ('SUBTITLE_API_URL', 'SUBTITLE_API_TOKEN', 'SUBTITLE_API_TOKEN_FILE', 'SUBTITLE_DATA_DIR'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(web, 'WORKSPACE', tmp_path)
    monkeypatch.setattr(web, 'DATA', tmp_path / 'old-web-jobs')
    monkeypatch.setattr(web, 'jobs', {})
    monkeypatch.setattr(web, 'comparisons', {})
    monkeypatch.delenv('MUSE_USER', raising=False)
    monkeypatch.delenv('MUSE_PASSWORD', raising=False)
    source = tmp_path / '样片.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=64x48:r=5:d=0.4',
                    '-c:v', 'libx264', str(source)], check=True)
    server = subprocess.Popen(['php', '-S', f'127.0.0.1:{port}', str(PHP / 'public/index.php')],
                              env={**os.environ, 'SUBTITLE_API_TOKEN': TOKEN, 'SUBTITLE_DATA_DIR': str(data)},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for attempt in range(50):
            try:
                with urlopen(Request(base + '/api/modes', headers={'Authorization': 'Bearer ' + TOKEN}), timeout=1):
                    break
            except URLError:
                if attempt == 49 or server.poll() is not None:
                    raise
                time.sleep(0.04)
        with web.app.test_client() as client:
            yield client, base, data, source
    finally:
        server.terminate()
        server.wait(timeout=5)


def upload_php(base, source, regions, mode=None):
    command = ['curl', '--noproxy', '*', '-sS', '-w', '\n%{http_code}', '-H', 'Authorization: Bearer ' + TOKEN,
               '-F', 'video=@' + str(source), '-F', 'regions=' + json.dumps(regions)]
    if mode is not None:
        command += ['-F', 'mode=' + mode]
    body, code = subprocess.check_output([*command, base + '/api/jobs'], text=True).rsplit('\n', 1)
    return json.loads(body), int(code)


def upload_web(client, source, regions, mode=None, *, streaming=False, **extra):
    fields = dict(video=(io.BytesIO(source.read_bytes()), source.name), regions=json.dumps(regions),
                  dimensions='[64,48]', **extra)
    if mode is not None:
        fields['mode'] = mode
    headers = {'X-Subtitle-Workflow': 'product'} if streaming else {}
    return client.post('/api/jobs', data=fields, headers=headers)


def test_web_and_api_resolve_identical_jobs_for_all_tiers(queue):
    client, base, data, source = queue
    with urlopen(Request(base + '/api/modes', headers={'Authorization': 'Bearer ' + TOKEN})) as result:
        catalog = json.load(result)
    assert client.get('/api/modes').json == catalog
    assert catalog['default'] == 'pro' and catalog['order'] == ['lite', 'pro', 'max']
    assert [c['label'] for c in catalog['modes'].values()] == ['Lite · 快速', 'Pro · 标准', 'Max · 增加']
    for mode, selected, stages in [
        ('lite', [SUB], ['ocr_lama']), (None, [SUB, WM], ['clear', 'propainter']),
        ('max', [WM], ['propainter', 'selective_review']), ('max', [SUB], ['clear', 'selective_review']),
    ]:
        direct, status = upload_php(base, source, selected, mode)
        assert status == 202
        browser = upload_web(client, source, selected, mode, streaming=mode is not None, queue='1')
        assert browser.status_code == 202, browser.json
        standalone = json.loads(subprocess.check_output([
            'php', str(PHP / 'product_modes.php'), mode or 'pro', *[r['type'] for r in selected],
        ]))
        assert browser.json['resolved_plan'] == direct['resolved_plan'] == standalone
        assert standalone['stages'] == stages and standalone['mode'] == (mode or 'pro')
        for queued, entry in [(direct, 'api'), (browser.json, 'web')]:
            path = queued['status_url']
            status = client.get(path)
            assert status.json['state'] == 'queued' and status.json['pipeline'] == stages
            assert status.json['submitted_via'] == entry
            assert status.json['resolved_plan'] == standalone
            saved = json.loads((data / queued['id'] / 'request.json').read_text())
            assert saved['resolved_plan'] == standalone and saved['regions'] == selected
            assert (data / queued['id'] / 'source.video').read_bytes() == source.read_bytes()
            assert TOKEN not in status.get_data(as_text=True)
            assert client.get(path + '/output').status_code == 409
    with sqlite3.connect(data / 'queue.sqlite') as db:
        assert db.execute("SELECT COUNT(*) FROM jobs WHERE state='queued'").fetchone() == (8,)
    assert not web.jobs and not web.DATA.exists()


def test_same_capability_checks_and_stale_catalog(queue, monkeypatch):
    client, base, data, source = queue
    direct, status = upload_php(base, source, [WM], 'lite')
    browser = upload_web(client, source, [WM], 'lite')
    assert status == browser.status_code == 422 and direct['error'] == browser.json['error']
    assert upload_web(client, source, [SUB], 'pro', catalog_sha256='obsolete').status_code == 409
    assert upload_web(client, source, [SUB], 'pro', inference='{"steps":1}').status_code == 422
    monkeypatch.setenv('SUBTITLE_API_TOKEN', 'invalid')
    assert client.get('/api/modes').status_code == 503
    assert not list(data.glob('*/request.json'))


def test_status_retry_playback_download_and_comparison_keep_canonical_id(queue):
    client, _, data, source = queue
    job = upload_web(client, source, [SUB], 'max').json
    path, folder = job['status_url'], data / job['id']
    with sqlite3.connect(data / 'queue.sqlite') as db:
        db.execute("UPDATE jobs SET state='failed', error='test' WHERE id=?", (job['id'],))
    assert client.get(path).json['can_retry']
    assert client.post(path + '/retry?queue=1').json['id'] == job['id']
    assert client.get(path).json['resolved_plan'] == job['resolved_plan']
    # Supply a completed artifact to test serving and comparison without invoking a model.
    shutil.copyfile(source, folder / 'result.mp4')
    with sqlite3.connect(data / 'queue.sqlite') as db:
        db.execute("UPDATE jobs SET state='done', percent=100 WHERE id=?", (job['id'],))
    with client.get(path + '/output') as res:
        assert res.data == source.read_bytes() and res.headers['Content-Disposition'].startswith('attachment')
    with client.get(path + '/output?inline=1', headers={'Range': 'bytes=0-4'}) as res:
        assert res.status_code == 206 and res.data == source.read_bytes()[:5]
    assert client.get(path + '/comparison').json['state'] == 'pending'
    assert client.post(path + '/comparison').status_code == 202
    for _ in range(80):
        state = client.get(path + '/comparison').json
        if state['state'] != 'running':
            break
        time.sleep(0.1)
    assert state['state'] == 'done', state
    with client.get(state['url']) as res:
        assert res.status_code == 200 and res.data == (folder / 'comparison.mp4').read_bytes()


def test_inference_uses_frozen_pro_options(queue, monkeypatch):
    client, _, data, source = queue
    monkeypatch.syspath_prepend(str(PHP / 'inference'))
    from run import record_execution, standard_task

    job = upload_web(client, source, [SUB, WM], 'pro').json
    folder = data / job['id']
    request = json.loads((folder / 'request.json').read_text())
    request['resolved_plan']['parameters']['clear']['steps'] = 7
    request['resolved_plan']['parameters']['propainter']['raft_iter'] = 8
    monkeypatch.setenv('CLEAR_PRESET', 'quality')
    task = standard_task(request)
    assert task['pipeline'] == ['clear', 'propainter']
    for stage in task['pipeline']:
        assert task['inference'][stage] == request['resolved_plan']['parameters'][stage]
    record_execution(folder, request, 'pro', task)
    execution = json.loads((folder / 'execution.json').read_text())
    assert execution['parameters'] == request['resolved_plan']['parameters']
    assert set(execution['entrypoint_sha256']) >= {'CLEAR-web/worker.py', 'CLEAR-web/propainter_worker.py'}


def test_old_sqlite_queue_migrates_and_retries_without_inventing_a_frozen_plan(queue):
    client, _, data, source = queue
    job_id = 'b' * 32
    folder = data / job_id
    folder.mkdir()
    shutil.copyfile(source, folder / 'source.video')
    original = {'mode': 'fine', 'regions': [SUB], 'dimensions': [64, 48], 'original_filename': source.name}
    (folder / 'request.json').write_text(json.dumps(original))
    with sqlite3.connect(data / 'queue.sqlite') as db:
        db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, name TEXT NOT NULL, mode TEXT NOT NULL, state TEXT NOT NULL, "
                   "percent INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL, error TEXT, created INTEGER NOT NULL)")
        db.execute("INSERT INTO jobs (id,name,mode,state,stage,created) VALUES (?,?, 'fine','failed','old',10)",
                   (job_id, source.name))
    path = '/api/jobs/' + job_id
    job = client.get(path).json
    assert (job['mode'], job['resolved_plan'], job['queued_at']) == ('max', None, 10)
    assert client.post(path + '/retry').status_code == 202
    assert client.get(path).json['queued_at'] > 10
    assert json.loads((folder / 'request.json').read_text()) == original
