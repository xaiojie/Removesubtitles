import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from werkzeug.http import parse_options_header
import app as web

JOB_ID = 'c' * 32


@pytest.fixture
def comparison_job(tmp_path, monkeypatch):
    folder = tmp_path / JOB_ID
    folder.mkdir()
    (folder / 'output.mp4').write_bytes(b'processed')
    (folder / 'source.video').write_bytes(b'original')
    (folder / 'config.json').write_text(json.dumps({'original_filename': '旅行.第一集.MOV'}), encoding='utf-8')
    monkeypatch.setattr(web, 'DATA', tmp_path)
    monkeypatch.setattr(web, 'comparisons', {}, raising=False)
    thread = MagicMock()
    monkeypatch.setattr(web, 'Thread', thread)
    with web.app.test_client() as client:
        yield client, folder, thread


def test_comparison_starts_once_for_repeated_requests(comparison_job):
    client, folder, thread = comparison_job
    first = client.post(f'/api/jobs/{JOB_ID}/comparison')
    second = client.post(f'/api/jobs/{JOB_ID}/comparison')
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json['state'] == 'running'
    assert thread.call_count == 1
    assert (folder / 'source.video').is_file()


def test_cached_comparison_download_uses_original_name(comparison_job):
    client, folder, thread = comparison_job
    (folder / 'comparison.mp4').write_bytes(b'joined')
    started = client.post(f'/api/jobs/{JOB_ID}/comparison')
    assert started.status_code == 200
    assert started.json['state'] == 'done'
    assert client.get(f'/api/jobs/{JOB_ID}/comparison').json['state'] == 'done'
    with client.get(started.json['url']) as result:
        assert result.data == b'joined'
        _, options = parse_options_header(result.headers['Content-Disposition'])
        assert options['filename'] == '旅行.第一集_处理前后对比.mp4'
    thread.assert_not_called()


def test_legacy_job_requests_original_without_starting(comparison_job):
    client, folder, thread = comparison_job
    (folder / 'source.video').unlink()
    result = client.post(f'/api/jobs/{JOB_ID}/comparison')
    assert result.status_code == 409
    assert result.json['needs_original'] is True
    thread.assert_not_called()


def test_comparison_rejects_missing_or_invalid_job(comparison_job):
    client, _, thread = comparison_job
    assert client.post('/api/jobs/invalid/comparison').status_code == 404
    assert client.get('/api/jobs/invalid/comparison/output').status_code == 404
    thread.assert_not_called()


def test_successful_processing_keeps_original_for_comparison(comparison_job, monkeypatch):
    _, folder, _ = comparison_job
    monkeypatch.setattr(web, 'jobs', {JOB_ID: {'state': 'running'}})
    process = MagicMock()
    process.stdout = []
    process.wait.return_value = 0
    popen = MagicMock()
    popen.return_value.__enter__.return_value = process
    monkeypatch.setattr(web.subprocess, 'Popen', popen)
    web.run_job(JOB_ID, folder)
    assert web.jobs[JOB_ID]['state'] == 'done'
    assert (folder / 'source.video').is_file()


@pytest.mark.parametrize("processed_frames", [5, 2])
def test_comparison_uses_decoded_frames_when_mov_header_count_is_inaccurate(tmp_path, monkeypatch, processed_frames):
    import cv2
    import numpy as np
    import comparison_video

    source = tmp_path / "source.mp4"
    processed = tmp_path / "processed.mp4"
    for path, count in ((source, 5), (processed, processed_frames)):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
        assert writer.isOpened()
        for index in range(count):
            writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
        writer.release()

    open_capture = cv2.VideoCapture

    def capture_with_inaccurate_header(path, *args):
        capture = open_capture(path, *args)
        proxy = MagicMock(wraps=capture)

        def metadata(property_id):
            value = capture.get(property_id)
            if str(path) == str(source) and property_id == cv2.CAP_PROP_FRAME_COUNT:
                return value + 3
            return value

        proxy.get.side_effect = metadata
        return proxy

    monkeypatch.setattr(comparison_video.cv2, "VideoCapture", capture_with_inaccurate_header)
    if processed_frames == 5:
        assert comparison_video.validate_video_pair(source, processed)[:2] == (32, 32)
    else:
        with pytest.raises(ValueError, match="帧数"):
            comparison_video.validate_video_pair(source, processed)
