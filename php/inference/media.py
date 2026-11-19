"""Display-coordinate video I/O shared by the PHP job workers."""

import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'CLEAR-web'))
from video_capture import open_video_capture


def report(percent, stage):
    print(
        'JOB_STATUS ' + json.dumps({'percent': int(percent), 'stage': stage}, ensure_ascii=False), flush=True
    )


@contextmanager
def capture(path):
    video = open_video_capture(path)
    try:
        if not video.isOpened():
            raise ValueError('无法打开视频')
        yield video
    finally:
        video.release()


def geometry(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('无效的视频帧率')
    return fps, max(1, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))


@contextmanager
def encode(output, audio_source, dimensions, fps, crf):
    w, h = dimensions
    args = [
        os.getenv('FFMPEG', 'ffmpeg'),
        '-nostdin',
        '-v',
        'error',
        '-y',
        '-f',
        'rawvideo',
        '-pixel_format',
        'bgr24',
        '-video_size',
        f'{w}x{h}',
        '-framerate',
        str(fps),
        '-i',
        'pipe:0',
        '-i',
        str(audio_source),
        '-map',
        '0:v:0',
        '-map',
        '1:a?',
        '-c:v',
        'libx264',
        '-preset',
        'fast',
        '-crf',
        str(crf),
        '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt',
        'yuv420p',
    ]
    audio = subprocess.check_output(
        [
            os.getenv('FFPROBE', 'ffprobe'),
            '-v',
            'error',
            '-select_streams',
            'a',
            '-show_entries',
            'stream=codec_name',
            '-of',
            'csv=p=0',
            str(audio_source),
        ],
        text=True,
    )
    args += ['-c:a', 'copy'] if all(v == 'aac' for v in audio.split()) else ['-c:a', 'aac', '-b:a', '192k']
    temporary = output.with_suffix('.partial.mp4')
    try:
        with subprocess.Popen(
            args + ['-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE
        ) as process:
            try:
                yield process.stdin
                process.stdin.close()
                if process.wait():
                    raise RuntimeError('视频编码失败')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
