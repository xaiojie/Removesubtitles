"""Stream long videos through LTX and composite only user-selected rectangles."""
from collections import deque
from contextlib import contextmanager
from itertools import chain
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg

from video_capture import open_video_capture

from ltx_options import LTXOptions
from regions import regions_from_config
from video_segments import decoded_frames, video_batches

ROOT = Path(__file__).resolve().parent


def report(**fields):
    print('CLEAR_STATUS ' + json.dumps(fields, ensure_ascii=False), flush=True)


def read_event(process, expected):
    recent = deque(maxlen=12)
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        recent.append(line.rstrip())
        if line.startswith('LTX_ERROR '):
            raise RuntimeError(json.loads(line[len('LTX_ERROR '):])['error'])
        if line.startswith(expected + ' '):
            return json.loads(line[len(expected) + 1:])
    raise RuntimeError('LTX 进程提前退出。' + '\n'.join(recent)[-1200:])


def write_reference(frames, path, shape, fps):
    width, height = shape
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', 'pipe:0', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '0',
               '-pix_fmt', 'yuv444p', str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
        try:
            for frame in frames:
                resized = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
                writer.stdin.write(resized.tobytes())
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError('LTX 参考片段编码失败。')
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.wait()


@contextmanager
def predictor(folder, source, shape):
    python = os.environ.get('LTX_PYTHON', str(source / '.venv/bin/python'))
    environment = {**os.environ, 'PYTHONPATH': str(source), 'PYTHONUNBUFFERED': '1'}
    command = [python, '-u', str(ROOT / 'ltx_adapter.py'), str(source), str(folder / 'config.json')]
    input_path, output_path = folder / 'ltx-input.mp4', folder / 'ltx-generated.mp4'
    with subprocess.Popen(command, cwd=source, env=environment, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
        try:
            read_event(process, 'LTX_READY')

            def predict(frames, fps):
                write_reference(frames, input_path, shape, fps)
                output_path.unlink(missing_ok=True)
                process.stdin.write(json.dumps(dict(input=str(input_path), output=str(output_path),
                                                    width=shape[0], height=shape[1], frames=len(frames), fps=fps)) + '\n')
                process.stdin.flush()
                read_event(process, 'LTX_RESULT')
                capture = open_video_capture(str(output_path), cv2.CAP_FFMPEG)
                try:
                    outputs = list(decoded_frames(capture))
                finally:
                    capture.release()
                if len(outputs) != len(frames):
                    raise RuntimeError(f'LTX 返回 {len(outputs)} 帧，期望 {len(frames)} 帧。')
                return outputs

            yield predict
            process.stdin.close()
            if process.wait(timeout=60):
                raise RuntimeError('LTX 进程退出失败。')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


def compose(frames, generated, regions):
    if len(frames) != len(generated):
        raise RuntimeError('LTX 输出帧数与输入不一致。')
    results = []
    height, width = frames[0].shape[:2]
    for original, repaired in zip(frames, generated):
        repaired = cv2.resize(repaired, (width, height), interpolation=cv2.INTER_CUBIC)
        result = original.copy()
        for region in regions:
            left, top, right, bottom = region['roi']
            result[top:bottom, left:right] = repaired[top:bottom, left:right]
        results.append(result)
    return results


def encode(cap, first, fps, total, regions, folder, options, frame_range, predict, lossless_output=False):
    height, width = first.shape[:2]
    suffix = 'mkv' if lossless_output else 'mp4'
    temporary = folder / f'partial.{suffix}'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0']
    if lossless_output:
        command += ['-map', '0:v:0', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-an']
    else:
        pixel_format = 'yuv444p' if width % 2 or height % 2 else 'yuv420p'
        command += ['-i', str(folder / 'source.video'), '-map', '0:v:0', '-map', '1:a?',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', str(options.export_crf),
                    '-pix_fmt', pixel_format, '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart']
    command.append(str(temporary))
    selected_total = (frame_range[1] if frame_range[1] is not None else total) - frame_range[0]
    done, written, segment = 0, 0, 0
    deferred = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
        def write(frames):
            nonlocal written
            for frame in frames:
                writer.stdin.write(frame.tobytes())
                written += 1
        try:
            frames = chain((first,), decoded_frames(cap))
            for carry, pending, selected in video_batches(frames, frame_range, [], options):
                if not selected:
                    write(deferred)
                    deferred = []
                    write(pending)
                    continue
                segment += 1
                report(percent=min(94, int(8 + 86 * done / max(selected_total, 1))),
                       stage=f'LTX 第 {segment} 段 · 正在生成', segment=segment, frames_done=done)
                batch = carry + pending
                # LTX requires 8n+1 frames. Pad the tail and discard padding.
                padded_count = max(9, len(batch) + (-(len(batch) - 1) % 8))
                padded = batch + [batch[-1]] * (padded_count - len(batch))
                repaired = compose(batch, predict(padded, fps)[:len(batch)], regions)
                overlap = len(carry)
                if overlap:
                    if len(deferred) != overlap:
                        raise RuntimeError('LTX 重叠片段帧数不一致。')
                    for index in range(overlap):
                        alpha = (index + 1) / (overlap + 1)
                        repaired[index] = cv2.addWeighted(deferred[index], 1 - alpha, repaired[index], alpha, 0)
                keep = min(options.overlap_frames, len(repaired))
                write(repaired[:-keep] if keep else repaired)
                deferred = repaired[-keep:] if keep else []
                done += len(pending)
                report(percent=min(94, int(8 + 86 * done / max(selected_total, done))),
                       stage=f'LTX 已修复 {done} 帧', segment=segment, frames_done=done)
            write(deferred)
            if not done:
                raise ValueError('所选时间段没有可处理的视频帧。')
            report(percent=97, stage='LTX 正在封装视频与原音轨', frames_done=done, output_frames=written)
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError('LTX 最终视频编码失败。')
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.wait()
    temporary.replace(folder / f'output.{suffix}')
    return written


def run(folder, source):
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    options = LTXOptions.parse(json.dumps(config.get('inference', {})))
    cap = open_video_capture(str(folder / 'source.video'), cv2.CAP_FFMPEG)
    try:
        ok, first = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError('无法读取视频或识别帧率。')
        height, width = first.shape[:2]
        if [width, height] != config['dimensions']:
            raise ValueError('浏览器与解码器读取的画面尺寸不同。请先导出为标准 MP4。')
        regions = regions_from_config(config)
        frame_range = options.frame_range(fps, total)
        shape = options.model_shape(width, height)
        report(percent=1, stage=f'LTX 正在加载模型 · {shape[0]}×{shape[1]} · 原帧率 {fps:.3f}')
        with predictor(folder, source, shape) as predict:
            encode(cap, first, fps, total, regions, folder, options, frame_range, predict,
                   lossless_output=config.get('lossless_output') is True)
    finally:
        cap.release()


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    except Exception as error:
        report(error='LTX CUDA 显存不足，可降低推理尺寸或分段帧数。'
               if 'out of memory' in str(error).lower() else str(error)[-1500:])
        raise
