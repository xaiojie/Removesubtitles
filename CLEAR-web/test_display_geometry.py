import json
import subprocess

import cv2
import imageio_ffmpeg

import comparison_video
import worker


def make_anamorphic_video(path):
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'color=c=red:s=64x64:r=5', '-t', '0.4',
        '-vf', 'setsar=64/63', '-c:v', 'libx264', '-f', 'mp4', str(path),
    ], check=True)


def test_clear_accepts_browser_display_dimensions_for_anamorphic_pixels(tmp_path, monkeypatch):
    make_anamorphic_video(tmp_path / 'source.video')
    regions = [dict(type='subtitle', roi=[6, 10, 50, 30])]
    (tmp_path / 'config.json').write_text(json.dumps(dict(
        dimensions=[65, 64], regions=regions, inference=dict(segment_mode='fixed'))))
    monkeypatch.setattr(worker, 'load_model', lambda *args: object())
    observed = []

    def encode(capture, first, fps, total, model, actual_regions, *args, **kwargs):
        observed.extend([first.shape, *(frame.shape for frame in worker.decoded_frames(capture))])
        assert actual_regions == regions

    monkeypatch.setattr(worker, 'encode', encode)
    worker.run(tmp_path, tmp_path / 'models')
    assert observed == [(64, 65, 3), (64, 65, 3)]


def test_comparison_scales_anamorphic_source_to_processed_display_size(tmp_path):
    source, output = tmp_path / 'source.mp4', tmp_path / 'output.mp4'
    comparison = tmp_path / 'comparison.mp4'
    make_anamorphic_video(source)
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error',
        '-i', str(source), '-vf', 'scale=66:64,setsar=1', '-c:v', 'libx264', str(output),
    ], check=True)
    comparison_video.create_comparison(source, output, comparison)
    capture = cv2.VideoCapture(str(comparison), cv2.CAP_FFMPEG)
    try:
        ok, frame = capture.read()
        assert ok and frame.shape == (64, 132, 3)
        assert frame[:, 64:66, 2].mean() > 180
    finally:
        capture.release()
