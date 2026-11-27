"""Display-coordinate video I/O shared by the PHP job workers."""

import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'CLEAR-web'))
from job_trace import span
from video_capture import open_video_capture


SDR_MATRICES = {'bt709': 'bt709', 'bt470bg': 'bt601', 'smpte170m': 'bt601',
                'fcc': 'fcc', 'smpte240m': 'smpte240m'}
SDR_TRANSFERS = {None, 'unknown', 'bt709', 'bt470m', 'bt470bg', 'smpte170m', 'smpte240m', 'iec61966-2-1'}
SCALE_FLAGS = 'bicubic+accurate_rnd+full_chroma_int'


def _streams(path):
    data = subprocess.check_output([
        os.getenv('FFPROBE', 'ffprobe'), '-v', 'error', '-show_entries',
        'stream=codec_type,codec_name,color_space,color_range,color_transfer,color_primaries',
        '-of', 'json', str(path),
    ])
    return json.loads(data)['streams']


def _sdr_video(streams):
    video = next((s for s in streams if s.get('codec_type') == 'video'), {})
    if (video.get('color_space') in SDR_MATRICES and video.get('color_range') in ('tv', 'pc')
            and video.get('color_transfer') in SDR_TRANSFERS):
        return video
    return None


@lru_cache(maxsize=4)
def _passthrough(ffmpeg):
    # FFmpeg 5.1 introduced fps_mode; FFmpeg 9 removed the older vsync option.
    current = subprocess.run([ffmpeg, '-v', 'error', '-fps_mode', 'passthrough', '-version'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ('-fps_mode', 'passthrough') if current.returncode == 0 else ('-vsync', '0')


class _ColorCapture:
    def __init__(self, process, width, height, fps, count):
        self.process = process
        self.shape = (height, width, 3)
        self.properties = {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: count,
                           cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height,
                           cv2.CAP_PROP_SAR_NUM: 1, cv2.CAP_PROP_SAR_DEN: 1}

    def get(self, property_id):
        return self.properties.get(property_id, 0)

    def read(self):
        frame = np.empty(self.shape, dtype=np.uint8)
        size = self.process.stdout.readinto(frame)
        if size == frame.nbytes:
            return True, frame
        if self.process.wait() or size:
            raise RuntimeError('视频解码提前结束')
        return False, None


def report(percent, stage):
    print(
        'JOB_STATUS ' + json.dumps({'percent': int(percent), 'stage': stage}, ensure_ascii=False), flush=True
    )


def sdr_decode_command(path, color, width, height, *, timestamp=None):
    # Sequential frames and frames borrowed from the source must use the same colour conversion.
    matrix = SDR_MATRICES[color['color_space']]
    filters = (f'scale=in_color_matrix={matrix}:in_range={color["color_range"]}:out_range=pc:flags={SCALE_FLAGS},'
               f'format=bgr24,scale={width}:{height}:flags=bilinear')
    ffmpeg = os.getenv('FFMPEG', 'ffmpeg')
    seek = [] if timestamp is None else ['-ss', f'{max(0., timestamp - .000001):.6f}']
    limit = [] if timestamp is None else ['-frames:v', '1']
    return [ffmpeg, '-nostdin', '-v', 'error', '-filter_threads', '1', '-threads:v', '2',
            *seek, '-i', str(path), '-map', '0:v:0', '-vf', filters, *_passthrough(ffmpeg),
            *limit, '-threads:v', '1', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']


class NearbyFrames:
    """Open the independent decoder only when a large caption needs a reference frame."""
    def __init__(self, path):
        self.path = path
        self.video = None
        self.color = None

    def read(self, index, *, reference=False):
        if self.video is None:
            self.video = open_video_capture(self.path)
            self.color = _sdr_video(_streams(self.path))
        if not self.video.set(cv2.CAP_PROP_POS_FRAMES, index):
            return None
        ok, frame = self.video.read()
        if not ok or abs(self.video.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > .5:
            return None
        if not reference or not self.color:
            return frame
        timestamp = self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not math.isfinite(timestamp) or timestamp < 0 or (index > 0 and timestamp == 0):
            return None
        height, width = frame.shape[:2]
        command = sdr_decode_command(self.path, self.color, width, height, timestamp=timestamp)
        result = subprocess.run(command, stdout=subprocess.PIPE)
        if result.returncode or len(result.stdout) != frame.nbytes:
            return None
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(frame.shape)

    def close(self):
        if self.video is not None:
            self.video.release()


@contextmanager
def capture(path):
    video = open_video_capture(path)
    try:
        if not video.isOpened():
            raise ValueError('无法打开视频')
        color = _sdr_video(_streams(path))
        if not color:
            yield video
            return
        # Use the same frame rate and rotated, square-pixel geometry as every job worker.
        fps, count = geometry(video)
        ok, first = video.read()
        if not ok:
            raise ValueError('无法解码视频首帧')
        height, width = first.shape[:2]
        del first
    finally:
        video.release()

    # Older OpenCV FFmpeg backends convert tagged BT.709 as BT.601 before returning BGR.
    args = sdr_decode_command(path, color, width, height)
    with subprocess.Popen(args, stdout=subprocess.PIPE) as process:
        try:
            yield _ColorCapture(process, width, height, fps, count)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def geometry(video):
    fps = video.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('无效的视频帧率')
    return fps, max(1, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))


@contextmanager
def encode(output, audio_source, dimensions, fps, crf):
    w, h = dimensions
    streams = _streams(audio_source)
    color = _sdr_video(streams)
    filters = 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
    tags = []
    primaries_and_transfer = []
    if color:
        matrix = SDR_MATRICES[color['color_space']]
        filters += (f',scale=in_range=pc:out_color_matrix={matrix}:out_range={color["color_range"]}'
                    f':flags={SCALE_FLAGS},format=yuv420p,setsar=1')
        tags = ['-colorspace', color['color_space'], '-color_range', color['color_range']]
        for field, option in (('color_primaries', '-color_primaries'), ('color_transfer', '-color_trc')):
            if color.get(field) not in (None, 'unknown', 'reserved'):
                primaries_and_transfer.extend((option, color[field]))
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
        # The raw BGR pipe carries no metadata. FFmpeg also needs these values on input.
        *primaries_and_transfer,
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
        filters,
        '-pix_fmt',
        'yuv420p',
        *tags,
        *primaries_and_transfer,
    ]
    audio = [s.get('codec_name') for s in streams if s.get('codec_type') == 'audio']
    args += ['-c:a', 'copy'] if all(v == 'aac' for v in audio) else ['-c:a', 'aac', '-b:a', '192k']
    temporary = output.with_suffix('.partial.mp4')
    try:
        with subprocess.Popen(
            args + ['-movflags', '+faststart', str(temporary)], stdin=subprocess.PIPE
        ) as process, span('ffmpeg_encode', interval='process_lifetime', process_pid=process.pid):
            try:
                yield process.stdin
                with span('ffmpeg_finalize', process_pid=process.pid):
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
