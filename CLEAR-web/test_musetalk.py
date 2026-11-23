from io import BytesIO
import json
import wave

import numpy as np

import app as web
import musetalk_worker as muse


def test_lip_sync_defaults_to_complete_audio_without_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'DATA', tmp_path)
    monkeypatch.setattr(web, 'jobs', {})
    monkeypatch.setattr(web, 'engine_status', lambda engine: (True, 'ready'))
    monkeypatch.setattr(web, 'dispatch_next', lambda: None)
    response = web.app.test_client().post('/api/jobs', data={
        'mode': 'lip_sync',
        'video': (BytesIO(b'video'), 'sample.mp4'),
        'audio': (BytesIO(b'audio'), 'voice.wav'),
    })
    assert response.status_code == 202
    folder = tmp_path / response.json['id']
    configuration = json.loads((folder / 'config.json').read_text())
    assert configuration['pipeline'] == ['musetalk']
    assert configuration['duration_mode'] == 'full'
    assert (folder / 'source.audio').read_bytes() == b'audio'


def test_only_long_near_silent_spans_are_eligible(tmp_path):
    amplitudes = np.repeat([3000, 0, 260, 0, 2000], [8, 10, 18, 16, 8])
    samples = np.repeat(amplitudes, muse.SAMPLES_PER_FRAME).astype('<i2')
    path = tmp_path / 'voice.wav'
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, muse.SAMPLE_RATE, 0, 'NONE', ''))
        stream.writeframes(samples.tobytes())
    assert muse.quiet_spans(path) == [(36, 52)]
