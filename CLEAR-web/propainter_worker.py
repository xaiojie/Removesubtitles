"""Bounded, lossless-frame adapter for the installed upstream ProPainter CLI."""
from collections import deque
from itertools import chain, islice
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import cv2
import imageio_ffmpeg

from video_capture import open_video_capture
import numpy as np

from propainter_masks import MaskBuilder
from regions import regions_from_config
from video_segments import decoded_frames, find_shot_cuts, video_batches


def report(**fields):
    print('CLEAR_STATUS ' + json.dumps(fields, ensure_ascii=False), flush=True)


def processing_size(width, height, resolution):
    if resolution == 'native':
        return (width + (-width % 8), height + (-height % 8))
    scale = min(1, math.sqrt((832 * 480) / (width * height)), 1024 / max(width, height))
    return max(8, int(width * scale) // 8 * 8), max(8, int(height * scale) // 8 * 8)


def window_settings(width, height):
    # Keep 8 original context frames, with at least 2 new frames per window.
    capacity = max(10, min(80, (80 * 1280 * 720) // (width * height)))
    return capacity, 8


def validate_assets(source):
    required = [source / 'inference_propainter.py']
    required.extend(source / 'weights' / name for name in
                    ('raft-things.pth', 'recurrent_flow_completion.pth', 'ProPainter.pth'))
    if any(not path.is_file() for path in required):
        raise RuntimeError('ProPainter 程序或本地权重尚未安装完整。')


def run_upstream(source, frames_dir, masks_dir, result_dir, options, size, progress):
    command = [sys.executable, '-u', str(source / 'inference_propainter.py'),
               '--video', str(frames_dir), '--mask', str(masks_dir), '--output', str(result_dir),
               '--width', str(size[0]), '--height', str(size[1]), '--save_frames',
               '--mask_dilation', str(options.mask_dilation),
               '--neighbor_length', str(options.neighbor_length), '--ref_stride', str(options.ref_stride),
               '--subvideo_length', str(options.subvideo_length), '--raft_iter', str(options.raft_iter)]
    if options.fp16:
        command.append('--fp16')
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    recent = deque(maxlen=30)
    with subprocess.Popen(command, cwd=source, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                recent.append(line.strip())
                match = re.search(r'(?<!\d)(\d{1,3})%\|', line)
                if match:
                    progress(min(100, int(match.group(1))))
            if process.wait():
                raise RuntimeError('ProPainter 修复失败。' + '\n'.join(recent)[-1200:])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return result_dir / frames_dir.name / 'frames'


def process_window(frames, masks, source, folder, options, progress):
    if len(frames) != len(masks) or not frames:
        raise ValueError('待修复帧与遮罩数量不一致。')
    if not any(mask.any() for mask in masks):
        return [frame.copy() for frame in frames]
    height, width = frames[0].shape[:2]
    size = processing_size(width, height, options.resolution)
    inputs, input_masks = list(frames), list(masks)
    if len(inputs) == 1:
        inputs.append(inputs[0])
        input_masks.append(input_masks[0])
    with tempfile.TemporaryDirectory(prefix='.propainter-', dir=folder) as temporary:
        work = Path(temporary)
        frames_dir, masks_dir, result_dir = work / 'input', work / 'masks', work / 'results'
        frames_dir.mkdir()
        masks_dir.mkdir()
        for index, (frame, mask) in enumerate(zip(inputs, input_masks)):
            if options.resolution == 'native':
                pixels = cv2.copyMakeBorder(frame, 0, size[1]-height, 0, size[0]-width,
                                            cv2.BORDER_REPLICATE)
                mask = cv2.copyMakeBorder(mask, 0, size[1]-height, 0, size[0]-width,
                                          cv2.BORDER_CONSTANT)
            else:
                pixels = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            if not (cv2.imwrite(str(frames_dir / f'{index:06d}.png'), pixels)
                    and cv2.imwrite(str(masks_dir / f'{index:06d}.png'), mask)):
                raise RuntimeError('无法写入 ProPainter 临时帧。')
        output = run_upstream(source, frames_dir, masks_dir, result_dir, options, size, progress)
        files = sorted(output.glob('*.png'))
        if len(files) != len(inputs) or [path.name for path in files] != [f'{i:04d}.png' for i in range(len(inputs))]:
            raise RuntimeError('ProPainter 返回的帧数与输入不一致。')
        results = []
        for path in files[:len(frames)]:
            repaired = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if repaired is None or repaired.shape[:2] != (size[1], size[0]):
                raise RuntimeError('ProPainter 返回的帧尺寸无效。')
            if options.resolution == 'native':
                repaired = repaired[:height, :width].copy()
            else:
                repaired = cv2.resize(repaired, (width, height), interpolation=cv2.INTER_CUBIC)
            results.append(repaired)
        return results


def analyze_video(input_file, regions, options):
    capture = open_video_capture(str(input_file), cv2.CAP_FFMPEG)
    try:
        ok, first = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        estimated = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError('无法读取视频或识别帧率。')
        height, width = first.shape[:2]
        builder = MaskBuilder(regions, (width, height), options.watermark_mask_mode, options.mask_dilation)
        sample_stride = max(1, math.ceil(max(estimated, 1) / 120))
        total = 0
        def scan():
            nonlocal total
            for index, frame in enumerate(chain((first,), decoded_frames(capture))):
                if frame.shape != first.shape:
                    raise ValueError('视频帧尺寸发生变化。请先导出为标准 MP4。')
                if index % sample_stride == 0:
                    builder.observe(frame)
                total += 1
                yield frame
        if options.segment_mode == 'shots':
            cuts = find_shot_cuts(scan(), regions)
        else:
            for _ in scan():
                pass
            cuts = []
        builder.finalize()
        return width, height, fps, total, cuts, builder
    finally:
        capture.release()


def composite_frame(original, repaired, mask):
    selected = mask > 0
    original[selected] = repaired[selected]


def encode(folder, source, options, information, frame_range, mask_reference=None):
    width, height, fps, total, cuts, builder = information
    selected_total = frame_range[1] - frame_range[0] if frame_range[1] is not None else total - frame_range[0]
    capacity, overlap = window_settings(*processing_size(width, height, options.resolution))
    batching = SimpleNamespace(chunk_frames=capacity, overlap_frames=overlap)
    temporary = folder / 'partial.mp4'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', f'{width}x{height}',
               '-framerate', f'{fps:.12g}', '-i', 'pipe:0', '-i', str(folder / 'source.video'),
               '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'fast',
               '-crf', str(options.export_crf), '-threads', '8',
               '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(temporary)]
    capture = open_video_capture(str(folder / 'source.video'), cv2.CAP_FFMPEG)
    reference_capture = open_video_capture(str(mask_reference), cv2.CAP_FFMPEG) if mask_reference else None
    reference_frames = iter(decoded_frames(reference_capture)) if reference_capture is not None else None
    reference_carry = []
    done = 0
    written = 0
    segment = 0
    try:
        actual_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        if actual_size != (width, height) or not math.isfinite(actual_fps) or abs(actual_fps - fps) > 0.001:
            raise ValueError('中间视频与原视频的尺寸或帧率不一致。')
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as writer:
            try:
                for carry, pending, selected in video_batches(decoded_frames(capture), frame_range, cuts, batching):
                    reference_pending = list(islice(reference_frames, len(pending))) if reference_frames is not None else pending
                    if len(reference_pending) != len(pending):
                        raise RuntimeError('原视频参考与中间视频帧数不一致。')
                    if selected:
                        segment += 1
                        batch = carry + pending
                        reference_batch = batch
                        if reference_frames is not None:
                            reference_batch = (reference_carry[-len(carry):] if carry else []) + reference_pending
                            reference_carry = reference_batch[-overlap:] if overlap else []
                        masks = [builder.mask(frame) for frame in reference_batch]
                        def progress(percent):
                            fraction = (done + len(pending) * percent / 100) / max(1, selected_total)
                            report(percent=min(94, int(8 + 86 * fraction)),
                                   stage=f'第 {segment} 段 · 视频修复 {percent}%',
                                   completed_frames=done, total_frames=selected_total)
                        progress(0)
                        repaired = process_window(batch, masks, source, folder, options, progress)
                        for frame, result, mask in zip(batch, repaired, masks):
                            composite_frame(frame, result, builder.expanded_mask(mask))
                        done += len(pending)
                    for frame in pending:
                        writer.stdin.write(frame.tobytes())
                        written += 1
                    if selected:
                        report(percent=min(94, int(8 + 86 * done / max(1, selected_total))),
                               stage=f'已修复 {done} 帧', completed_frames=done, total_frames=selected_total)
                if not done or written != total or done != selected_total:
                    raise RuntimeError('实际解码或修复帧数与视频时轴不一致。')
                report(percent=97, stage='正在封装视频与音轨', completed_frames=done, total_frames=selected_total)
                writer.stdin.close()
                error = writer.stderr.read().decode(errors='replace')
                if writer.wait():
                    raise RuntimeError('FFmpeg 视频编码失败。' + error[-800:])
            finally:
                if writer.poll() is None:
                    writer.kill()
                    writer.wait()
        temporary.replace(folder / 'output.mp4')
    finally:
        capture.release()
        if reference_capture is not None:
            reference_capture.release()
        temporary.unlink(missing_ok=True)
    report(percent=100, stage='视频修复完成', completed_frames=done, total_frames=selected_total)


def run(folder, source):
    from propainter_options import ProPainterOptions
    folder, source = Path(folder).resolve(), Path(source).resolve()
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    regions = regions_from_config(config)
    options = ProPainterOptions.parse(json.dumps(config.get('inference', {})))
    validate_assets(source)
    report(percent=5, stage='正在采样水印与检测切镜', completed_frames=0, total_frames=0)
    mask_reference = Path(config['mask_reference']).resolve() if config.get('mask_reference') else None
    information = analyze_video(mask_reference or folder / 'source.video', regions, options)
    width, height, fps, total, _, _ = information
    if [width, height] != config['dimensions']:
        raise ValueError('浏览器与解码器读取的画面尺寸不同。请先导出为标准 MP4。')
    frame_range = options.frame_range(fps, total)
    report(percent=8, stage='正在准备视频修复', completed_frames=0,
           total_frames=(frame_range[1] or total) - frame_range[0])
    encode(folder, source, options, information, frame_range, mask_reference=mask_reference)


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]), Path(sys.argv[2]))
    except Exception as error:
        message = 'CUDA 显存不足，请降低处理分辨率。' if 'out of memory' in str(error).lower() else str(error)[-1500:]
        report(error=message)
        raise
