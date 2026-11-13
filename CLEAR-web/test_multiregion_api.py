import io
import json
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
import app as web
from inference_options import InferenceOptions


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'DATA', tmp_path)
    monkeypatch.setattr(web, 'jobs', {})
    monkeypatch.setattr(web, 'Thread', MagicMock())
    monkeypatch.setattr(web, 'model_status', lambda: (True, 'ready'))
    monkeypatch.setattr(web, 'propainter_status', lambda: (True, 'ready'))
    with web.app.test_client() as connection:
        yield connection, tmp_path


def test_accepts_and_persists_multiple_typed_regions(client):
    connection, root = client
    regions = [
        {'type': 'subtitle', 'roi': [100, 620, 1100, 705]},
        {'type': 'watermark', 'roi': [10, 10, 250, 150]},
        {'type': 'watermark', 'roi': [1000, 570, 1278, 718]},
    ]
    response = connection.post('/api/jobs', data={
        'regions': json.dumps(regions), 'dimensions': '[1280,720]',
        'video': (io.BytesIO(b'video'), '多处字幕水印.mov'),
    })
    assert response.status_code == 202, response.get_data(as_text=True)
    config = json.loads((root / response.json['id'] / 'config.json').read_text())
    assert config['regions'] == regions
    assert config['roi'] == [10, 10, 1278, 718]
    assert config['inference']['clear']['vae_tiled'] is False


@pytest.mark.parametrize('regions', [[], [{'type':'logo','roi':[0,0,20,20]}],
    [{'type':'watermark','roi':[0,0,1281,20]}],
    [{'type':'subtitle','roi':[False,0,20,20]}],
    [{'type':'watermark','roi':[0,0,20]}],
    {'subtitles': [[0,0,20,20]]}])
def test_rejects_invalid_region_lists_before_upload(client, regions):
    connection, root = client
    response=connection.post('/api/jobs',data={'regions':json.dumps(regions),'dimensions':'[1280,720]'})
    assert response.status_code == 400
    assert not list(root.iterdir())


def test_selected_defaults_are_published_in_system(client):
    connection, _ = client
    defaults=connection.get('/api/system').json['inference_defaults']
    assert defaults['seed']==20261113
    assert defaults['chunk_frames']==33
    assert defaults['resolution']=='native'
    assert defaults['export_crf']==10
    assert defaults.get('vae_tiled') is False


def test_vae_tiled_requires_json_boolean():
    assert InferenceOptions.parse('{"vae_tiled":true}').vae_tiled is True
    assert InferenceOptions.parse('{"vae_tiled":false}').vae_tiled is False
    for value in (0,1,'false',None):
        with pytest.raises(ValueError):
            InferenceOptions.parse(json.dumps({'vae_tiled':value}))
