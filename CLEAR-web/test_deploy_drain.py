import io
import json

import pytest
import app as web


@pytest.fixture
def deployment_client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'WORKSPACE', tmp_path)
    monkeypatch.setattr(web, 'DATA', tmp_path / 'jobs')
    web.DATA.mkdir()
    monkeypatch.setattr(web, 'jobs', {})
    monkeypatch.setattr(web, 'comparisons', {})
    monkeypatch.setattr(web, 'draining', False, raising=False)
    monkeypatch.setattr(web, 'active_requests', {}, raising=False)
    monkeypatch.setattr(web, 'model_status', lambda: (True, 'ready'))
    monkeypatch.setattr(web, 'propainter_status', lambda: (True, 'ready'))
    monkeypatch.setattr(web, 'Thread', lambda **kwargs: type('IdleThread', (), {'start': lambda self: None})())
    state = tmp_path / '.deploy'
    state.mkdir()
    token = 'deployment-test-token-0123456789abcdef'
    (state / 'maintenance.token').write_text(token)
    return web.app.test_client(), {'X-Deploy-Token': token}


@pytest.mark.parametrize('method', ['get', 'post', 'delete'])
def test_deployment_gate_requires_private_token(deployment_client, method):
    client, _ = deployment_client
    assert getattr(client, method)('/api/deploy/drain').status_code == 403
    assert getattr(client, method)('/api/deploy/drain', headers={'X-Deploy-Token': 'wrong'}).status_code == 403


def test_deployment_gate_rejects_nonlocal_peer(deployment_client):
    client, headers = deployment_client
    result = client.post('/api/deploy/drain', headers=headers, environ_overrides={'REMOTE_ADDR': '203.0.113.2'})
    assert result.status_code == 403


@pytest.mark.parametrize('address,peer', [('10.0.0.4', '10.0.0.4'), ('fd00:0:0::4', 'fd00::4')])
def test_deployment_gate_accepts_authenticated_bound_interface(deployment_client, address, peer):
    client, headers = deployment_client
    result = client.post('/api/deploy/drain', headers=headers,
                         environ_overrides={'REMOTE_ADDR': peer, 'SERVER_NAME': address})
    assert result.status_code == 200
    assert result.json['draining'] is True


def test_drain_reports_jobs_and_comparisons_then_releases_gate(deployment_client):
    client, headers = deployment_client
    web.jobs.update(first={'state': 'running'}, second={'state': 'queued'}, third={'state': 'uploading'}, done={'state': 'done'})
    web.comparisons['compare'] = {'state': 'running'}
    result = client.post('/api/deploy/drain', headers=headers)
    assert result.status_code == 200
    assert result.json['draining'] is True
    assert {item['id'] for item in result.json['busy']} == {'first', 'second', 'third', 'compare'}
    assert client.delete('/api/deploy/drain', headers=headers).json['draining'] is False


@pytest.mark.parametrize('path', ['/api/jobs', '/api/jobs/' + 'a' * 32 + '/retry', '/api/jobs/' + 'a' * 32 + '/comparison'])
def test_drain_blocks_new_work_before_parsing_uploads(deployment_client, path):
    client, headers = deployment_client
    client.post('/api/deploy/drain', headers=headers)
    result = client.post(path)
    assert result.status_code == 503
    assert '升级' in result.json['error']
    assert not list(web.DATA.iterdir())
    assert client.get('/api/system').status_code == 200


def test_inflight_request_stays_visible_while_it_finishes(deployment_client):
    client, headers = deployment_client
    with web.app.test_request_context('/api/jobs', method='POST'):
        assert web.app.preprocess_request() is None
        result = client.post('/api/deploy/drain', headers=headers)
        assert any(item['kind'] == 'request' for item in result.json['busy'])
    assert client.get('/api/deploy/drain', headers=headers).json['busy'] == []


def test_already_queued_jobs_continue_during_drain(deployment_client):
    client, headers = deployment_client
    web.jobs['queued'] = {'state': 'queued'}
    client.post('/api/deploy/drain', headers=headers)
    web.dispatch_next()
    assert web.jobs['queued']['state'] == 'running'
    assert client.get('/api/deploy/drain', headers=headers).json['busy'][0]['id'] == 'queued'


def test_releasing_gate_accepts_new_video(deployment_client):
    client, headers = deployment_client
    client.post('/api/deploy/drain', headers=headers)
    client.delete('/api/deploy/drain', headers=headers)
    result = client.post('/api/jobs', data={
        'video': (io.BytesIO(b'video'), 'test.mp4'), 'dimensions': '[300,200]',
        'regions': json.dumps([{'type': 'subtitle', 'roi': [20, 120, 200, 170]}]), 'engine': 'auto',
    }, content_type='multipart/form-data')
    assert result.status_code == 202
