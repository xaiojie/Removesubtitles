"""PP-OCRv5 Mobile on user bands, followed by masked, sentence-local LaMa."""

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from media import ROOT, capture, encode, geometry, report
from job_trace import span
from lite_masks import black_caption_cards
from lite_tracking import track_block
from subtitle_compositing import colored_caption_cards, large_caption_guidance, light_caption_cards


@dataclass(frozen=True)
class LiteOptions:
    # Jobs created before product plans were persisted used these settings.
    sample_stride: int = 6
    detector_max_side: int = 720
    box_padding: int = 5
    context_padding: int = 28
    batch_size: int = 4
    export_crf: int = 18
    mask_mode: str = 'boxes'
    adaptive_ocr: bool = False
    inpaint_max_side: int = 0

    def __post_init__(self):
        if min(self.sample_stride, self.detector_max_side, self.batch_size) < 1:
            raise ValueError('Lite 检测间隔、检测尺寸和批量大小须大于零')
        if min(self.box_padding, self.context_padding) < 0 or not 0 <= self.export_crf <= 51:
            raise ValueError('Lite 扩框或编码参数无效')
        if self.mask_mode not in ('boxes', 'joint', 'joint_cards', 'focused_cards'):
            raise ValueError('Lite 遮罩模式无效')
        if type(self.inpaint_max_side) is not int or self.inpaint_max_side < 0:
            raise ValueError('Lite 局部推理尺寸应为非负整数')


def detect(frame, rois, detector, options=LiteOptions()):
    boxes = []
    for x, y, r, b in rois:
        crop = frame[y:b, x:r]
        scale = min(1.0, options.detector_max_side / max(crop.shape[:2]))
        if scale < 1:
            crop = cv2.resize(crop, (max(1, round((r - x) * scale)), max(1, round((b - y) * scale))))
        sx, sy = (r - x) / crop.shape[1], (b - y) / crop.shape[0]
        with span('lite_ocr', device='cpu', width=crop.shape[1], height=crop.shape[0]):
            predictions = list(detector.predict(crop))
        for result in predictions:
            for polygon in result['dt_polys']:
                pts = np.asarray(polygon)
                left, top = pts.min(axis=0)
                right, bottom = pts.max(axis=0)
                rect = (
                    max(x, x + math.floor(left * sx) - options.box_padding),
                    max(y, y + math.floor(top * sy) - options.box_padding),
                    min(r, x + math.ceil(right * sx) + options.box_padding + 1),
                    min(b, y + math.ceil(bottom * sy) + options.box_padding + 1),
                )
                if rect[0] < rect[2] and rect[1] < rect[3]:
                    boxes.append(rect)
    return tuple(sorted(set(boxes)))


def similar(a, b):
    return len(a) == len(b) and all(max(abs(x - y) for x, y in zip(i, j)) <= 3 for i, j in zip(a, b))


def sample_block(frames, start_boxes, find):
    # Include the next anchor for comparison but emit it in the next block.
    end_boxes = find(frames[-1]) if len(frames) > 1 else start_boxes
    if similar(start_boxes, end_boxes):
        boxes = tuple(
            (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
            for a, b in zip(start_boxes, end_boxes)
        )
        return [boxes] * (len(frames) - 1), end_boxes
    return [start_boxes, *(find(f) for f in frames[1:-1])], end_boxes


def restore(frames, boxes_by_frame, lama, options=LiteOptions()):
    h, w = frames[0].shape[:2]
    by_rect = defaultdict(list)
    for i, boxes in enumerate(boxes_by_frame):
        for box in boxes:
            by_rect[box].append(i)
    for (x, y, r, b), indices in by_rect.items():
        pad = options.context_padding
        l, t, rr, bb = max(0, x - pad), max(0, y - pad), min(w, r + pad), min(h, b + pad)
        area = np.s_[y - t : b - t, x - l : r - l]
        mask = np.zeros((bb - t, rr - l), np.uint8)
        mask[area] = 255
        for start in range(0, len(indices), options.batch_size):
            batch = indices[start : start + options.batch_size]
            crops = [cv2.cvtColor(frames[i][t:bb, l:rr], cv2.COLOR_BGR2RGB) for i in batch]
            with span('lite_lama', device=str(getattr(lama, 'device', 'unknown')), frames=len(crops)):
                repairs = lama._inpaint_batch(crops, [mask] * len(crops))
            for i, repaired in zip(batch, repairs):
                frames[i][y:b, x:r] = cv2.cvtColor(repaired[area], cv2.COLOR_RGB2BGR)
    return frames


def caption_plates(crop, boxes):
    """Find a near-uniform outlined card surrounding an OCR detection in the user band."""
    h, w = crop.shape[:2]
    result = np.zeros((h, w), np.uint8)
    for x, y, r, b in boxes:
        height = b-y
        if height < 5:
            continue
        area = crop[y:b:2, x:r:2]
        quantized = (area.astype(np.uint16) // 32)
        bins = (quantized[..., 0] * 64 + quantized[..., 1] * 8 + quantized[..., 2]).ravel()
        counts = np.bincount(bins, minlength=512)
        for index in np.argsort(counts)[-2:]:
            if counts[index] < .12 * bins.size:
                continue
            colors = area.reshape(-1, 3)[bins == index]
            color = np.median(colors, axis=0).astype(int)
            uniform = cv2.inRange(crop, tuple(map(int, np.maximum(0, color-18))),
                                  tuple(map(int, np.minimum(255, color+18))))
            closed = cv2.morphologyEx(uniform, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                left, top, width, depth = cv2.boundingRect(contour)
                right, end = left+width, top+depth
                if width < max(2*depth, .65*(r-x)) or not .5*height <= depth <= 3*height:
                    continue
                overlap = max(0, min(r,right)-max(x,left)) * max(0, min(b,end)-max(y,top))
                if overlap < .50*(b-y)*(r-x) or cv2.contourArea(contour) < .64*width*depth:
                    continue
                # An unbounded flat scene or letterbox is not a subtitle card.
                if depth >= .91*h and width >= .91*w:
                    continue
                core = uniform[top:end, left:right]
                edges = np.concatenate((core[0], core[-1], core[:, 0], core[:, -1]))
                if np.count_nonzero(core) < .41*core.size or np.count_nonzero(edges) < .55*edges.size:
                    continue
                cv2.drawContours(result, [contour], -1, 255, cv2.FILLED)
    return cv2.dilate(result, np.ones((5, 5), np.uint8)) if result.any() else result


def caption_band(boxes, width, height, padding):
    """Protect a distant, smaller HUD label when the band has a dominant dialogue line."""
    if len(boxes) < 2 or width < 4 * height:
        return 0, width
    central = [box for box in boxes if abs((box[0]+box[2])/2-width/2) < width*.25
               and box[2]-box[0] >= max(width*.16, 3*(box[3]-box[1]-2*padding))]
    if not central:
        return 0, width
    main = max(central, key=lambda box: (box[3]-box[1], box[2]-box[0]))
    main_height = max(1, main[3]-main[1]-2*padding)
    start, end = 0, width
    for x, y, r, b in boxes:
        if b-y-2*padding > .72*main_height or r-x > .45*(main[2]-main[0]):
            continue
        gap = max(2*main_height, .12*width)
        if (x+r)/2 > .84*width and x-main[2] > gap:
            end = min(end, max(main[2], x-padding))
        elif (x+r)/2 < .16*width and main[0]-r > gap:
            start = max(start, min(main[0], r+padding))
    return start, end


def outlined_text_mask(patch, padding):
    """Cover confirmed light text and its dark outline while preserving gaps between glyphs."""
    height, width = patch.shape[:2]
    glyph_height = height-2*padding
    if glyph_height < 8 or width < glyph_height:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    radius = max(2, min(12, round(.18*glyph_height)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    near_dark = cv2.erode(gray, kernel)
    white = cv2.inRange(patch, (162, 162, 162), (255, 255, 255))
    yellow = cv2.inRange(patch, (0, 145, 176), (135, 255, 255))
    strokes = cv2.bitwise_and(white | yellow, cv2.inRange(cv2.subtract(gray, near_dark), 54, 255))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    keep = np.zeros(count, np.uint8)
    for index, (_, _, w, h, area) in enumerate(stats[1:], 1):
        if h >= max(3, .18*glyph_height) and w <= 4*glyph_height and area >= max(4, .12*glyph_height):
            keep[index] = 255
    seed = keep[labels]
    ys, xs = np.nonzero(seed)
    if len(xs) < .025*glyph_height*max(1, width-2*padding):
        return None
    if xs.max()-xs.min() < .70*(width-2*padding) or ys.max()-ys.min() < .50*glyph_height:
        return None
    occupied = np.flatnonzero(seed.any(axis=0))
    if np.max(np.diff(occupied), initial=0) > 1.5*glyph_height:
        return None
    # Include detached dots, accents and punctuation near the supported text.
    extent = max(2, round(.40*glyph_height))
    close_to_line = cv2.dilate(seed, np.ones((2*extent+1, 2*extent+1), np.uint8))
    radius = max(3, min(18, round(.17*glyph_height)+2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(strokes & close_to_line, kernel)


def restore_joint(frames, boxes_by_frame, rois, lama, options):
    height, width = frames[0].shape[:2]
    for left, top, right, bottom in rois:
        active = []
        for index, (frame, boxes) in enumerate(zip(frames, boxes_by_frame)):
            crop = frame[top:bottom, left:right]
            local = []
            for x, y, r, b in boxes:
                x, y = max(x, left), max(y, top)
                r, b = min(r, right), min(b, bottom)
                if x < r and y < b:
                    local.append((x - left, y - top, r - left, b - top))
            focused = options.mask_mode == 'focused_cards'
            left_edge, right_edge = (caption_band(local, right-left, bottom-top, options.box_padding)
                                     if focused else (0, right-left))
            if focused:
                local = [box for box in local if left_edge <= box[0] and box[2] <= right_edge]
            mask = caption_plates(crop, local)
            if focused:
                dark_card = black_caption_cards(crop, local)
                if dark_card is not None:
                    mask |= dark_card
            if options.mask_mode in ('joint_cards', 'focused_cards'):
                colored = colored_caption_cards(crop)
                if colored is not None:
                    mask |= cv2.dilate(colored, np.ones((11, 11), np.uint8))
            cards = light_caption_cards(crop)
            if cards is not None:
                mask |= cards
            _, outline = large_caption_guidance(crop)
            if outline is not None:
                mask |= outline
            for x, y, r, b in local:
                area = mask[y:b, x:r]
                strokes = outlined_text_mask(crop[y:b, x:r], options.box_padding) if focused else None
                if strokes is None:
                    area[:] = 255
                else:
                    area |= strokes
            if focused:
                mask[:, :left_edge] = 0
                mask[:, right_edge:] = 0
            if mask.any():
                active.append((index, mask))

        for start in range(0, len(active), options.batch_size):
            batch = active[start:start + options.batch_size]
            # One crop per batch keeps nearby words in context without running LaMa on an empty band.
            union = np.maximum.reduce([mask for _, mask in batch])
            x, y, w, h = cv2.boundingRect(union)
            pad = options.context_padding
            l, t = max(0, left + x - pad), max(0, top + y - pad)
            r, b = min(width, left + x + w + pad), min(height, top + y + h + pad)
            masks = []
            crops = []
            for index, mask in batch:
                full = np.zeros((b - t, r - l), np.uint8)
                full[top + y - t:top + y - t + h, left + x - l:left + x - l + w] = mask[y:y+h, x:x+w]
                masks.append(full)
                crops.append(cv2.cvtColor(frames[index][t:b, l:r], cv2.COLOR_BGR2RGB))
            input_crops, input_masks = crops, masks
            scale = 1.
            if options.inpaint_max_side:
                # Keep fine print at native detail; cap only sufficiently tall, wide repair areas.
                scale = min(1., max(options.inpaint_max_side/max(r-l,b-t), 32/h))
            if scale < 1:
                size = (max(1, round((r-l)*scale)), max(1, round((b-t)*scale)))
                input_crops = [cv2.resize(im,size,interpolation=cv2.INTER_AREA) for im in crops]
                input_masks = [(cv2.resize(m,size,interpolation=cv2.INTER_AREA)>0).astype(np.uint8)*255 for m in masks]
            with span('lite_lama', device=str(lama.device), frames=len(batch)):
                outputs = lama._inpaint_batch(input_crops,input_masks)
            for (index, _), mask, repaired in zip(batch, masks, outputs):
                if scale < 1:
                    repaired = cv2.resize(repaired, (r-l,b-t), interpolation=cv2.INTER_LINEAR)
                selected = mask != 0
                frames[index][t:b, l:r][selected] = cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR)[selected]
    return frames


def load_models(options=LiteOptions()):
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    import torch
    from paddleocr import TextDetection

    vsr = ROOT / 'video-subtitle-remover'
    sys.path.insert(0, str(vsr))
    from backend.inpaint.lama_inpaint import LamaInpaint

    models = Path(os.getenv('VSR_MODEL_DIR', vsr / 'backend/models'))
    ocr_path, weights = models / 'V5/ch_det_fast', models / 'big-lama/big-lama.pt'
    if not weights.is_file() or not (ocr_path / 'inference.pdiparams').is_file():
        raise ValueError('缺少 Lite 模型，请下载 LaMa 和 Mobile 检测器')
    report(2, '加载 Mobile OCR 与 LaMa')
    with span('lite_ocr_load', device='cpu'):
        detector = TextDetection(
            model_name='PP-OCRv5_mobile_det',
            model_dir=str(ocr_path),
            device='cpu',
            cpu_threads=3,
            enable_mkldnn=False,
        )
    device = os.getenv('LITE_DEVICE', 'cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Lite 未检测到 CUDA；CPU 测试可设置 LITE_DEVICE=cpu')
    with span('lite_lama_load', device=device):
        lama = LamaInpaint(torch.device(device), str(weights), batch_size=options.batch_size)
    return detector, lama


def run(folder, request):
    plan = request.get('resolved_plan')
    options = LiteOptions(**plan['parameters']['lite']) if plan else LiteOptions()
    detector, lama = load_models(options)
    source = folder / 'source.video'
    rois = [r['roi'] for r in request['regions']]
    find = lambda frame: detect(frame, rois, detector, options)
    with capture(source) as video:
        fps, total = geometry(video)
        ok, first = video.read()
        if not ok or list(first.shape[1::-1]) != request['dimensions']:
            raise ValueError('视频无法解码或显示尺寸不一致')
        anchor = find(first)
        count = 0
        repaired_count = 0
        with encode(folder / 'result.mp4', source, request['dimensions'], fps, options.export_crf) as writer:
            while True:
                frames = [first]
                for _ in range(options.sample_stride):
                    ok, frame = video.read()
                    if not ok:
                        break
                    frames.append(frame)
                boxes, anchor = (track_block(frames,anchor,find,rois) if options.adaptive_ocr
                                 else sample_block(frames,anchor,find))
                if ok:
                    first = frames.pop()
                else:
                    boxes.append(anchor)
                repaired = (restore_joint(frames, boxes, rois, lama, options) if options.mask_mode != 'boxes'
                            else restore(frames, boxes, lama, options))
                for frame in repaired:
                    writer.write(frame.tobytes())
                repaired_count += sum(bool(selection) for selection in boxes)
                count += len(frames)
                report(min(98, 4 + 94 * count / total), '裁带检测并逐句去字幕')
                if not ok:
                    break
    print(f'Lite: {repaired_count}/{count} 帧命中字幕框并调用 LaMa', flush=True)


if __name__ == '__main__' and sys.argv[1:] == ['--verify']:
    detector, lama = load_models()
    frame = np.zeros((64, 160, 3), dtype=np.uint8)
    cv2.putText(frame, 'TEST', (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    detect(frame, [(0, 0, 160, 64)], detector)
    restore([frame], [[(10, 15, 95, 47)]], lama)
    print('Lite 检测器与 LaMa 推理校验通过。')
