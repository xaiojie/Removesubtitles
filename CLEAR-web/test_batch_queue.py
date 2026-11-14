import io
import json
from unittest.mock import MagicMock

import pytest
import app as web


@pytest.fixture
def batch_client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'DATA', tmp_path)
    monkeypatch.setattr(web, 'jobs', {})
    monkeypatch.setattr(web, 'engine_status', lambda engine: (True, 'ready'))
    thread = MagicMock()
    monkeypatch.setattr(web, 'Thread', thread)
    with web.app.test_client() as client:
        yield client, tmp_path, thread


def upload(queue=False):
    return dict(video=(io.BytesIO(b'video'), 'sample.mp4'), dimensions='[64,48]',
                regions=json.dumps([dict(type='subtitle', roi=[2, 20, 60, 44])]),
                engine='auto', inference='{}', queue='1' if queue else '0')


def test_batch_upload_waits_behind_running_job(batch_client):
    client, root, thread = batch_client
    web.jobs['current'] = dict(state='running')
    response = client.post('/api/jobs', data=upload(True))
    assert response.status_code == 202
    job_id = response.json['id']
    assert web.jobs[job_id]['state'] == 'queued'
    assert json.loads((root / job_id / 'status.json').read_text())['state'] == 'queued'
    thread.assert_not_called()
    assert client.post('/api/jobs', data=upload()).status_code == 409


def test_queue_dispatches_one_job_at_a_time_in_upload_order(batch_client):
    client, root, thread = batch_client
    first = client.post('/api/jobs', data=upload(True)).json['id']
    second = client.post('/api/jobs', data=upload(True)).json['id']
    assert web.jobs[first]['state'] == 'running'
    assert web.jobs[second]['state'] == 'queued'
    assert thread.call_count == 1
    web.set_job(first, state='done')
    web.dispatch_next()
    assert web.jobs[second]['state'] == 'running'
    assert thread.call_count == 2
    assert thread.call_args.kwargs['args'][0] == second


def test_failed_job_automatically_advances_queue(batch_client, monkeypatch):
    client, root, thread = batch_client
    first = client.post('/api/jobs', data=upload(True)).json['id']
    second = client.post('/api/jobs', data=upload(True)).json['id']
    process = MagicMock()
    process.stdout = []
    process.wait.return_value = 1
    popen = MagicMock()
    popen.return_value.__enter__.return_value = process
    monkeypatch.setattr(web.subprocess, 'Popen', popen)
    web.run_job(first, root / first)
    assert web.jobs[first]['state'] == 'failed'
    assert web.jobs[second]['state'] == 'running'
