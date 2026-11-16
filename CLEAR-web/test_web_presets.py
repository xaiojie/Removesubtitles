import io
import json
from unittest.mock import MagicMock

import pytest
import app as web
from pipeline_worker import completed_stages, signature
from task_options import prepare_task


SUBTITLES = [{'type': 'subtitle', 'roi': [20, 120, 200, 170]}]
WATERMARKS = [{'type': 'watermark', 'roi': [220, 10, 280, 50]}]


@pytest.fixture
def profile_client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'DATA', tmp_path)
    monkeypatch.setattr(web, 'jobs', {})
    monkeypatch.setattr(web, 'model_status', lambda: (True, 'ready'))
    monkeypatch.setattr(web, 'propainter_status', lambda: (True, 'ready'))
    monkeypatch.setattr(web, 'gpu_status', lambda: {'name': 'test', 'used': 0, 'total': 80000, 'util': 0})
    monkeypatch.setattr(web, 'Thread', MagicMock())
    return web.app.test_client(), tmp_path


def upload(client, inference=None):
    return client.post('/api/jobs', data={
        'video': (io.BytesIO(b'video'), 'test.mp4'), 'dimensions': '[300,200]',
        'regions': json.dumps(SUBTITLES), 'engine': 'auto',
        'inference': json.dumps(inference or {}),
    }, content_type='multipart/form-data')


def test_task_default_profile_can_be_injected_without_changing_legacy_calls():
    modern = prepare_task(SUBTITLES, '{}', 'auto', default_preset='quality')
    assert modern['inference']['clear']['preset'] == 'quality'
    assert modern['inference']['clear']['steps'] == 12
    legacy = prepare_task(SUBTITLES, '{}', 'auto')
    assert legacy['inference']['clear']['context'] == 'frame'
    assert legacy['inference']['clear']['resolution'] == 'native'


def test_system_advertises_catalog_and_selected_default(profile_client, monkeypatch):
    client, _ = profile_client
    monkeypatch.setenv('CLEAR_PRESET', 'quality')
    response = client.get('/api/system').json
    assert response['clear_default_preset'] == 'quality'
    assert {item['id'] for item in response['clear_presets']} == {'fast', 'balanced', 'quality'}
    assert response['inference_defaults']['preset'] == 'quality'
    assert response['inference_defaults']['steps'] == 12


def test_new_video_uses_configured_default_and_explicit_options_win(profile_client, monkeypatch):
    client, root = profile_client
    monkeypatch.setenv('CLEAR_PRESET', 'fast')
    response = upload(client, {'clear': {'steps': 7}})
    assert response.status_code == 202
    configuration = json.loads((root / response.json['id'] / 'config.json').read_text())
    assert configuration['inference']['clear']['preset'] == 'fast'
    assert configuration['inference']['clear']['lora_scale'] == 1.2
    assert configuration['inference']['clear']['steps'] == 7


def test_new_post_without_engine_or_inference_uses_deployment_default(profile_client, monkeypatch):
    client, root = profile_client
    monkeypatch.setenv('CLEAR_PRESET', 'balanced')
    response = client.post('/api/jobs', data={
        'video': (io.BytesIO(b'video'), 'test.mp4'), 'dimensions': '[300,200]',
        'regions': json.dumps(SUBTITLES),
    }, content_type='multipart/form-data')
    assert response.status_code == 202
    options = json.loads((root / response.json['id'] / 'config.json').read_text())['inference']['clear']
    assert options['preset'] == 'balanced'
    assert options['context'] == 'expanded'


def test_explicit_custom_options_are_preserved(profile_client, monkeypatch):
    client, root = profile_client
    monkeypatch.setenv('CLEAR_PRESET', 'quality')
    response = upload(client, {'clear': {'preset': 'custom', 'steps': 9, 'context': 'roi', 'resolution': 'native'}})
    assert response.status_code == 202
    options = json.loads((root / response.json['id'] / 'config.json').read_text())['inference']['clear']
    assert (options['preset'], options['steps'], options['context'], options['resolution']) == ('custom', 9, 'roi', 'native')


def test_retry_preserves_old_stage_signature_and_saved_parameters(profile_client, monkeypatch):
    client, root = profile_client
    monkeypatch.setenv('CLEAR_PRESET', 'quality')
    job_id = 'b' * 32
    folder = root / job_id
    folder.mkdir()
    (folder / 'source.video').write_bytes(b'original')
    configuration = {
        'engine': 'hybrid', 'pipeline': ['clear', 'propainter'], 'dimensions': [300, 200],
        'regions': SUBTITLES + WATERMARKS,
        'inference': {'start_seconds': 0, 'duration_seconds': None, 'export_crf': 10,
                      'clear': {'steps': 5, 'context': 'frame', 'resolution': 'native'},
                      'propainter': {'mask_dilation': 8}},
    }
    (folder / 'config.json').write_text(json.dumps(configuration))
    stage = folder / 'stages' / 'clear'
    stage.mkdir(parents=True)
    (stage / 'output.mkv').write_bytes(b'completed first stage')
    (stage / 'success.json').write_text(json.dumps({
        'stage': 'clear', 'artifact': 'output.mkv', 'config_sha256': signature(configuration),
    }))
    monkeypatch.setattr(web, 'model_status', lambda: (False, 'missing'))
    response = client.post(f'/api/jobs/{job_id}/retry')
    assert response.status_code == 202
    assert json.loads((folder / 'config.json').read_text()) == configuration
    assert completed_stages(folder) == ['clear']


@pytest.mark.parametrize('runtime_marker', ['runtime.complete', 'test.complete'])
def test_ltx_accepts_current_and_legacy_verified_installations(tmp_path, monkeypatch, runtime_marker):
    monkeypatch.setattr(web, 'LTX_ROOT', tmp_path)
    monkeypatch.setattr(web, 'LTX_PYTHON', tmp_path / 'python')
    (tmp_path / 'run_pipeline.py').touch()
    (tmp_path / 'python').touch()
    state = tmp_path / '.install'
    state.mkdir()
    for name in ('models.complete', 'environment.complete', runtime_marker):
        (state / name).touch()
    assert web.ltx_status()[0] is True


def test_ltx_installed_modes_are_advertised_and_missing_lora_rejects_before_upload(profile_client, monkeypatch, tmp_path):
    client, _ = profile_client
    monkeypatch.setattr(web, 'LTX_ROOT', tmp_path / 'ltx')
    monkeypatch.setattr(web, 'ltx_status', lambda: (True, 'ready'))
    state = web.LTX_ROOT / '.install'
    state.mkdir(parents=True)
    (state / 'selection.json').write_text(json.dumps({'tasks': ['subtitle']}))
    assert client.get('/api/system').json['ltx_modes'] == ['subtitle_rm']
    result = upload(client, {'subtitle_model': 'ltx', 'ltx': {'mode': 'watermark_rm'}})
    assert result.status_code == 503
    assert 'LoRA' in result.json['error']
    assert not web.jobs
