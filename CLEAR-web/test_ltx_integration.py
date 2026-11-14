import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import imageio_ffmpeg

from task_options import prepare_task

SUB = [{'type': 'subtitle', 'roi': [2, 2, 8, 6]}, {'type': 'subtitle', 'roi': [20, 12, 30, 20]}]
WATER = [{'type': 'watermark', 'roi': [10, 2, 18, 9]}]


def test_ltx_selection_routes_subtitles_and_preserves_separate_parameters():
    task = prepare_task(SUB + WATER, json.dumps({'subtitle_model': 'ltx', 'ltx': {'prompt': 'Clean scene', 'seed': 42}, 'propainter': {'mask_dilation': 4}}), 'auto')
    assert task['pipeline'] == ['ltx', 'propainter']
    assert task['inference']['subtitle_model'] == 'ltx'
    assert task['inference']['ltx']['prompt'] == 'Clean scene'
    assert task['inference']['propainter']['mask_dilation'] == 4
    assert prepare_task(SUB, json.dumps(task['inference']), 'auto')['pipeline'] == ['ltx']


def test_ltx_options_validate_temporal_shape_and_prompt():
    from ltx_options import LTXOptions
    options = LTXOptions.parse('{}')
    assert (options.mode, options.chunk_frames, options.seed, options.lora_scale) == ('subtitle_rm', 97, 42, 1.0)
    assert options.model_shape(1920, 1080) == (1248, 704)
    assert options.model_shape(1080, 1920) == (704, 1248)
    assert LTXOptions.parse('{"chunk_frames":305}').chunk_frames == 305
    for values in ({'chunk_frames': 96}, {'overlap_frames': 97}, {'prompt': ''}, {'model_width': 500, 'model_height': 288}, {'single_stage': False}, {'seed': True}):
        with pytest.raises(ValueError):
            LTXOptions.parse(json.dumps(values))


def test_ltx_multichunk_composition_preserves_total_frames_fps_audio_and_outside_roi(tmp_path):
    from ltx_worker import encode
    from ltx_options import LTXOptions
    import cv2
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source = tmp_path / 'source.video'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=32x24:r=30', '-f', 'lavfi', '-i', 'sine=frequency=500', '-t', '1', '-c:v', 'libx264', '-c:a', 'aac', '-f', 'mp4', str(source)], check=True)
    cap = cv2.VideoCapture(str(source))
    ok, first = cap.read()
    calls = []
    def predict(frames, fps):
        calls.append((len(frames), fps))
        return [np.full_like(frame, 190) for frame in frames]
    options = LTXOptions(chunk_frames=17, overlap_frames=8)
    count = encode(cap, first, 30, 30, SUB, tmp_path, options, (0, 30), predict, lossless_output=True)
    cap.release()
    assert count == 30
    assert calls == [(17, 30), (17, 30), (17, 30)]
    result = cv2.VideoCapture(str(tmp_path / 'output.mkv'))
    assert result.get(cv2.CAP_PROP_FPS) == 30
    decoded = []
    while True:
        ok, frame = result.read()
        if not ok:
            break
        decoded.append(frame)
    result.release()
    assert len(decoded) == 30
    for frame in decoded:
        assert np.all(frame[2:6, 2:8] == 190)
        assert np.all(frame[12:20, 20:30] == 190)
        assert np.all(frame[0, :] == 0)
        assert np.all(frame[8:11, 1:30] == 0)


def test_ltx_api_accepts_selection_and_reports_readiness(tmp_path, monkeypatch):
    import app as web
    monkeypatch.setattr(web, 'DATA', tmp_path)
    monkeypatch.setattr(web, 'ltx_status', lambda: (True, 'ready'))
    monkeypatch.setattr(web, 'gpu_status', lambda: {'name': 'test', 'used': 0, 'total': 80000, 'util': 0})
    class NoThread:
        def __init__(self, **kwargs): pass
        def start(self): pass
    monkeypatch.setattr(web, 'Thread', NoThread)
    web.jobs.clear()
    client = web.app.test_client()
    system = client.get('/api/system').json
    assert system['engines']['ltx']['ready']
    assert system['ltx_defaults']['chunk_frames'] == 97
    response = client.post('/api/jobs', data={'video': (io.BytesIO(b'video'), '原视频.mp4'), 'dimensions': '[32,24]', 'regions': json.dumps(SUB), 'engine': 'auto', 'inference': json.dumps({'subtitle_model': 'ltx'})}, content_type='multipart/form-data')
    assert response.status_code == 202
    config = json.loads((tmp_path / response.json['id'] / 'config.json').read_text())
    assert config['pipeline'] == ['ltx']
    assert config['original_filename'] == '原视频.mp4'
    web.jobs.clear()


def test_ltx_stage_uses_subtitle_regions_and_lossless_intermediate():
    import pipeline_worker
    from dataclasses import asdict
    from ltx_options import LTXOptions
    from propainter_options import ProPainterOptions
    config = {'pipeline': ['ltx', 'propainter'], 'dimensions': [32, 24], 'regions': SUB + WATER,
              'inference': {'ltx': asdict(LTXOptions()), 'propainter': asdict(ProPainterOptions())}}
    ltx = pipeline_worker.stage_config(config, 'ltx', Path('/job/source.video'))
    pp = pipeline_worker.stage_config(config, 'propainter', Path('/job/source.video'))
    assert ltx['regions'] == SUB
    assert ltx['lossless_output'] is True
    assert pp['regions'] == WATER
    assert pp['mask_reference'] == '/job/source.video'
    assert pipeline_worker.artifact_path(Path('/job'), 'ltx', config['pipeline']) == Path('/job/stages/ltx/output.mkv')


def test_ltx_short_time_range_preserves_unselected_frames_and_audio(tmp_path):
    import cv2
    from ltx_worker import encode
    from ltx_options import LTXOptions
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source = tmp_path / 'source.video'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=32x24:r=30', '-f', 'lavfi', '-i', 'sine=frequency=600', '-t', '0.1', '-c:v', 'libx264', '-c:a', 'aac', '-f', 'mp4', str(source)], check=True)
    capture = cv2.VideoCapture(str(source))
    _, first = capture.read()
    calls = []
    def predict(frames, fps):
        calls.append(len(frames))
        return [np.full_like(frame, 190) for frame in frames]
    count = encode(capture, first, 30, 3, SUB, tmp_path, LTXOptions(), (1, 2), predict)
    capture.release()
    assert calls == [9]
    assert count == 3
    output = tmp_path / 'output.mp4'
    capture = cv2.VideoCapture(str(output))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        decoded.append(frame)
    capture.release()
    assert len(decoded) == 3
    assert decoded[0].max() < 10 and decoded[2].max() < 10
    assert decoded[1][3, 3].min() > 170
    audio = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(output), '-map', '0:a:0', '-f', 'null', '-'], capture_output=True)
    assert audio.returncode == 0, audio.stderr
