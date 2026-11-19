"""Review every original/standard frame pair, refine only suspect intervals and ROIs."""

import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from media import ROOT, capture, encode, geometry, report

# isort: split
# Import CLEAR helpers after media has added the repository's CLEAR-web directory.
from adaptive_processing import foreground_core
from propainter_masks import MaskBuilder
from subtitle_compositing import caption_mask


def suspect(original, first, region):
    x, y, r, b = region['roi']
    area = np.s_[y:b, x:r]
    src, out = original[area], first[area]
    ink = foreground_core(src) > 0
    if region['type'] == 'subtitle':
        # Use original geometry to catch a plate/outline which no longer resembles a word.
        known = caption_mask(original, region['roi'])[area] > 0
        remnant = caption_mask(first, region['roi'])[area] > 0
        if known.any():
            ink &= known
            if np.count_nonzero(known & remnant) > max(10, 0.25 * np.count_nonzero(known)):
                return True
        elif not remnant.any():
            return False
    else:
        ink &= MaskBuilder.overlay_core(src)
    if ink.sum() < 5:
        return False
    unchanged = np.max(cv2.absdiff(src, out), axis=2) <= 18
    remaining = ink & (foreground_core(out) > 0) & unchanged
    return np.count_nonzero(remaining) >= max(5, 0.33 * np.count_nonzero(ink))


def review(source, standard, regions):
    spans = [[] for _ in regions]
    with capture(source) as a, capture(standard) as b:
        _, estimate = geometry(a)
        n = 0
        while True:
            ok, orig = a.read()
            other, prior = b.read()
            if ok != other:
                raise ValueError('首遍成片与原片的解码帧数不同')
            if not ok:
                break
            h, w = orig.shape[:2]
            prior = prior[:h, :w]
            for intervals, region in zip(spans, regions):
                if suspect(orig, prior, region):
                    left, end = max(0, n - 2), n + 3
                    if intervals and left <= intervals[-1][1]:
                        intervals[-1][1] = end
                    else:
                        intervals.append([left, end])
            n += 1
            if n % 24 == 0:
                report(min(65, 55 + 10 * n / estimate), '逐帧检查字幕切换、底板和水印')
    if not n:
        raise ValueError('视频无可解码帧')
    for intervals in spans:
        if intervals:
            intervals[-1][1] = min(n, intervals[-1][1])
    return spans, n


def refine(folder, request, first):
    source = folder / 'source.video'
    regions = request['regions']
    spans, count = review(source, first, regions)
    (folder / 'review.json').write_text(
        json.dumps({'regions': regions, 'frame_intervals': spans, 'frames': count})
    )
    if not any(spans):
        first.replace(folder / 'result.mp4')
        return

    # Release the diffusion model before any ProPainter pass. Each pass touches only one region type.
    previous = first
    for kind, limit in [('subtitle', 85), ('watermark', 98)]:
        selected = [(r, intervals) for r, intervals in zip(regions, spans) if r['type'] == kind and intervals]
        if not selected:
            continue
        destination = folder / f'fine-{kind}.mp4'
        run_pass(folder, source, previous, destination, request['dimensions'], selected, count, kind, limit)
        previous = destination
    previous.replace(folder / 'result.mp4')


def run_pass(folder, source, previous, destination, dimensions, selected, count, kind, limit):
    from propainter_options import ProPainterOptions
    from propainter_worker import process_window, validate_assets
    from video_segments import decoded_frames, find_shot_cuts
    from worker import PROMPT, load_model, repair_regions

    model = None
    regions = [r for r, _ in selected]
    with capture(source) as scan:
        cuts = set(find_shot_cuts(decoded_frames(scan), regions))
    indices = [0] * len(selected)
    last_active = ()
    pending = []
    reference = []
    propainter = Path(os.getenv('PROPAINTER_ROOT', ROOT / 'ProPainter'))
    label = '12 步局部字幕精修' if kind == 'subtitle' else '局部水印精修'

    def repair(frames, active):
        nonlocal model
        if not frames or not active:
            return
        if kind == 'watermark':
            validate_assets(propainter)
            masks = []
            for original, frame in zip(reference, frames):
                mask = np.zeros(frame.shape[:2], np.uint8)
                for i in active:
                    x, y, r, b = regions[i]['roi']
                    area = np.s_[y:b, x:r]
                    known = MaskBuilder.overlay_core(original[area]).astype(np.uint8)
                    remaining = MaskBuilder.overlay_core(frame[area]).astype(np.uint8)
                    remaining &= cv2.dilate(known, np.ones((5, 5), np.uint8))
                    mask[area] |= cv2.dilate(remaining, np.ones((9, 9), np.uint8)) * 255
                masks.append(mask)
            options = ProPainterOptions(mask_dilation=0, adaptive_processing=True)
            fixed = process_window(frames, masks, propainter, folder, options, lambda _: None)
            for f, new, mask in zip(frames, fixed, masks):
                f[mask > 0] = new[mask > 0]
            return
        for i in active:
            x, y, r, b = regions[i]['roi']

            def predict(images):
                nonlocal model
                if model is None:
                    model = load_model(Path(os.getenv('CLEAR_WORKSPACE', ROOT)) / 'CLEAR', 1.0, True)
                return model(
                    prompt=PROMPT,
                    control_video=images,
                    width=images[0].width,
                    height=images[0].height,
                    num_frames=len(images),
                    num_inference_steps=12,
                    cfg_scale=1.0,
                    seed=20261113,
                    tiled=False,
                    tile_size=(30, 52),
                    tile_stride=(15, 26),
                )

            # Enlarge only the detected remnant mask, clipped to the authorized selection.
            masks = []
            for original, f in zip(reference, frames):
                core = caption_mask(f, regions[i]['roi'])[y:b, x:r]
                if not core.any():
                    core = caption_mask(original, regions[i]['roi'])[y:b, x:r]
                if not core.any():
                    core = foreground_core(f[y:b, x:r])
                masks.append(cv2.dilate(core, np.ones((9, 9), np.uint8)) > 0)
            generated = [f.copy() for f in frames]
            repair_regions(
                generated,
                [regions[i]],
                predict,
                context='expanded',
                resolution='auto',
                blend_mode='rectangle',
            )
            for f, new, mask in zip(frames, generated, masks):
                f[y:b, x:r][mask] = new[y:b, x:r][mask]

    with capture(previous) as video, capture(source) as originals:
        fps, _ = geometry(video)
        with encode(destination, source, dimensions, fps, 10) as writer:

            def flush():
                repair(pending, last_active)
                for f in pending:
                    writer.write(f.tobytes())
                pending.clear()
                reference.clear()

            for frame_no in range(count):
                ok, frame = video.read()
                valid, original = originals.read()
                if not ok or not valid:
                    raise ValueError('精修读帧提前结束')
                active = []
                for i, (_, intervals) in enumerate(selected):
                    while indices[i] < len(intervals) and intervals[indices[i]][1] <= frame_no:
                        indices[i] += 1
                    if indices[i] < len(intervals) and intervals[indices[i]][0] <= frame_no:
                        active.append(i)
                active = tuple(active)
                if (
                    active != last_active
                    or frame_no in cuts
                    or len(pending) >= (17 if kind == 'subtitle' else 10)
                ):
                    flush()
                last_active = active
                frame = frame[: dimensions[1], : dimensions[0]].copy()
                if active:
                    pending.append(frame)
                    reference.append(original)
                else:
                    writer.write(frame.tobytes())
                if frame_no % 24 == 0:
                    start = 65 if kind == 'subtitle' else 85
                    report(start + (limit - start) * frame_no / count, label)
            flush()
    del model
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
