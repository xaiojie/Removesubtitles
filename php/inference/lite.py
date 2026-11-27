"""PP-OCRv5 Mobile on user bands, followed by masked, sentence-local LaMa."""

import math
import os
import sys
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from media import ROOT, capture, encode, geometry, report
from job_trace import span
from lite_masks import DarkCardTrack, anchored_brush_card, black_caption_cards, word_highlights, locate_chip
from lite_reuse import inference_frames
from lite_gradient import GradientPanel
from lite_line_panel import LinePanel
from lite_tracking import reject_transient_skin, track_block
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
    compact_transfer: bool = False
    ocr_backend: str = 'paddle'
    layout_profile: bool = False
    frame_tolerance: int = 0
    repair_gradient_edges: bool = False
    track_highlight_edges: bool = False
    complete_flat_strokes: bool = False
    optimize_lama_graph: bool = True
    variable_line_panels: bool = False
    skip_graph_up_to_seconds: int = 0

    def __post_init__(self):
        if min(self.sample_stride, self.detector_max_side, self.batch_size) < 1:
            raise ValueError('Lite 检测间隔、检测尺寸和批量大小须大于零')
        if min(self.box_padding, self.context_padding) < 0 or not 0 <= self.export_crf <= 51:
            raise ValueError('Lite 扩框或编码参数无效')
        if self.mask_mode not in ('boxes', 'joint', 'joint_cards', 'focused_cards', 'tracked_cards'):
            raise ValueError('Lite 遮罩模式无效')
        if type(self.inpaint_max_side) is not int or self.inpaint_max_side < 0:
            raise ValueError('Lite 局部推理尺寸应为非负整数')
        if self.ocr_backend not in ('paddle','onnx_cuda'):
            raise ValueError('Lite 字幕定位后端无效')
        if type(self.frame_tolerance) is not int or not 0<=self.frame_tolerance<=8:
            raise ValueError('相邻帧复用的像素容差须为 0 至 8')
        if type(self.skip_graph_up_to_seconds) is not int or self.skip_graph_up_to_seconds < 0:
            raise ValueError('短片图优化阈值须为非负整数秒数')

    def for_clip(self, frame_count, fps):
        # Some decoders use 0 or 1 when the length is unknown.
        if self.optimize_lama_graph and 1 < frame_count <= self.skip_graph_up_to_seconds * fps:
            return replace(self, optimize_lama_graph=False)
        return self


def detect(frame, rois, detector, options=LiteOptions()):
    boxes = []
    for x, y, r, b in rois:
        crop = frame[y:b, x:r]
        scale = min(1.0, options.detector_max_side / max(crop.shape[:2]))
        if scale < 1:
            crop = cv2.resize(crop, (max(1, round((r - x) * scale)), max(1, round((b - y) * scale))))
        sx, sy = (r - x) / crop.shape[1], (b - y) / crop.shape[0]
        with span('lite_ocr', device=getattr(detector,'inference_device','cpu'),
                  width=crop.shape[1], height=crop.shape[0]):
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


def outlined_text_mask(patch, padding, *, complete_flat_strokes=False):
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
    _, labels, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    widths, heights, areas = stats[:, 2:5].T
    keep = ((heights >= max(3, .18*glyph_height)) & (widths <= 4*glyph_height) &
            (areas >= max(4, .12*glyph_height))).astype(np.uint8) * 255
    keep[0] = 0
    seed = keep[labels]
    if cv2.countNonZero(seed) < .025*glyph_height*max(1, width-2*padding):
        return None
    _, _, span_width, span_height = cv2.boundingRect(seed)
    if span_width-1 < .70*(width-2*padding) or span_height-1 < .50*glyph_height:
        return None
    occupied = np.flatnonzero(seed.any(axis=0))
    if np.max(np.diff(occupied), initial=0) > 1.5*glyph_height:
        return None
    # Include detached dots, accents and punctuation near the supported text.
    extent = max(2, round(.40*glyph_height))
    close_to_line = cv2.dilate(seed, np.ones((2*extent+1, 2*extent+1), np.uint8))
    supported = strokes & close_to_line
    if complete_flat_strokes:
        # A thin glyph such as 一 can be wider than the reach of either adjacent letter.
        candidates = ((widths >= .30*glyph_height) & (widths <= 1.25*glyph_height) &
                      (widths >= 3*heights) & (heights <= .18*glyph_height) &
                      (areas >= .60*widths*heights))
        candidates[0] = False
        rim = max(2, round(.09*glyph_height))
        for component in np.flatnonzero(candidates):
            x, y, w, h, _ = stats[component]
            if y < rim or y+h+rim > height or not .25*height <= y+h/2 <= .75*height:
                continue
            region = np.s_[y:y+h, x:x+w]
            core = labels[region] == component
            overlap = close_to_line[region][core]
            if not overlap.any() or overlap.all():
                continue
            # Require a dark printed rim on both sides along the same columns.
            above = gray[y-rim:y, x:x+w].min(axis=0)
            below = gray[y+h:y+h+rim, x:x+w].min(axis=0)
            if np.count_nonzero((above < 96) & (below < 96)) >= .70*w:
                supported[region][core] = 255
    radius = max(3, min(18, round(.17*glyph_height)+2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(supported, kernel)


def include_punctuation(mask,patch,padding):
    height,width=mask.shape
    glyph=max(8,height-2*padding)
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
    outlines,_=cv2.findContours((gray<112).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    extra=np.zeros_like(mask)
    for curve in outlines:
        x,y,w,h=cv2.boundingRect(curve)
        if not .05*glyph <= h <= .47*glyph or not .05*glyph <= w <= .63*glyph:
            continue
        if min(x,width-x-w)>1.35*glyph or y+h/2<.48*height:
            continue
        cv2.drawContours(extra,[curve],-1,255,cv2.FILLED)
    return mask | cv2.dilate(extra,np.ones((5,5),np.uint8))


def restore_joint(frames, boxes_by_frame, rois, lama, options, tracks=None,profiles=None):
    height, width = frames[0].shape[:2]
    tracked = options.mask_mode == 'tracked_cards'
    if tracks is None:
        tracks = {}
    # An overlapping region must be detected on original pixels even if a previous region has been repaired.
    originals = {}
    if tracked:
        for index, (x,y,r,b) in enumerate(rois):
            if any(x < rr and xx < r and y < bb and yy < b for xx,yy,rr,bb in rois[:index]):
                originals[(x,y,r,b)] = [frame[y:b,x:r].copy() for frame in frames]
    for left, top, right, bottom in rois:
        active = []
        key = (left,top,right,bottom)
        if tracked and key not in tracks:
            tracks[key] = DarkCardTrack()
        for index, (frame, boxes) in enumerate(zip(frames, boxes_by_frame)):
            crop = originals[key][index] if key in originals else frame[top:bottom, left:right]
            local = []
            for x, y, r, b in boxes:
                x, y = max(x, left), max(y, top)
                r, b = min(r, right), min(b, bottom)
                if x < r and y < b:
                    local.append((x - left, y - top, r - left, b - top))
            profile = profiles.get(key) if profiles else None
            if profile is not None and profile.panel is not None and key not in originals:
                layer = profile.panel.restore(frame,key,local)
                if layer is not None:
                    (l,t,r,b),background,ink = layer
                    frame[t:b,l:r] = background
                    if options.repair_gradient_edges and isinstance(profile.panel, GradientPanel):
                        profile.panel.repair_edges(frame, key)
                    active.append((index,ink))
                    continue
                if isinstance(profile.panel, LinePanel):
                    local = profile.panel.fallback_boxes(key, local)
                    if not local:
                        continue
            dialogue = profile.dialogue_rows(local) if profile is not None else []
            if profile is not None:
                local = dialogue or profile.select(local)
            chip = locate_chip(crop,profile.chip) if profile is not None and profile.chip is not None else None
            highlight = (profile.highlight.mask(frame, key, track=options.track_highlight_edges)
                         if profile is not None and profile.highlight is not None and key not in originals else None)
            focused = options.mask_mode in ('focused_cards', 'tracked_cards')
            left_edge, right_edge = (caption_band(local, right-left, bottom-top, options.box_padding)
                                     if focused else (0, right-left))
            if focused:
                local = [box for box in local if left_edge <= box[0] and box[2] <= right_edge]
            dark_card = None
            if tracked:
                local, dark_card = tracks[key].update(crop, local)
                if dialogue:
                    # A stored single black card must not hide words on a verified second row.
                    local = dialogue
            elif focused:
                dark_card = black_caption_cards(crop, local)
            if profile is not None and not local and dark_card is None and chip is None and highlight is None:
                continue
            mask = caption_plates(crop, local)
            if dark_card is not None:
                mask |= dark_card
            if options.mask_mode in ('joint_cards', 'focused_cards', 'tracked_cards'):
                colored = colored_caption_cards(crop)
                if colored is not None:
                    mask |= cv2.dilate(colored, np.ones((11, 11), np.uint8))
            if tracked:
                brush = anchored_brush_card(crop,local)
                if brush is not None:
                    mask |= brush
                highlights = word_highlights(crop,local)
                if highlights is not None:
                    mask |= highlights
            cards = light_caption_cards(crop)
            if cards is not None:
                mask |= cards
            if local or not tracked:
                _, outline = large_caption_guidance(crop)
                if outline is not None:
                    if tracked:
                        support = np.zeros_like(mask)
                        for x,y,r,b in local:
                            pad = options.box_padding
                            support[max(0,y-pad):min(bottom-top,b+pad),max(0,x-pad):min(right-left,r+pad)] = 255
                        outline &= support
                    mask |= outline
            for x, y, r, b in local:
                area = mask[y:b, x:r]
                preserve_gaps = focused and (profile is None or profile.chip is None)
                strokes = outlined_text_mask(crop[y:b, x:r], options.box_padding,
                                             complete_flat_strokes=options.complete_flat_strokes) if preserve_gaps else None
                if strokes is None:
                    area[:] = 255
                else:
                    if options.layout_profile:
                        strokes=include_punctuation(strokes,crop[y:b,x:r],options.box_padding)
                    area |= strokes
            if profile is not None and local and profile.rows:
                # Standalone graphics elsewhere in a broad user selection have no dialogue anchor.
                count,indices=cv2.connectedComponents(mask)
                keep=np.zeros(count,np.uint8)
                for x, y, r, b in local:
                    keep[indices[y:b, x:r]] = 255
                keep[0]=0
                mask=keep[indices]
            if chip is not None:
                mask |= chip
            if focused:
                mask[:, :left_edge] = 0
                mask[:, right_edge:] = 0
            if highlight is not None:
                mask |= highlight
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
            anchors, order = inference_frames(crops, masks, options.frame_tolerance)
            with span('lite_lama', device=str(lama.device), frames=len(anchors), output_frames=len(batch),
                      optimize_graph=options.optimize_lama_graph):
                repairs = lama._inpaint_batch([input_crops[i] for i in anchors], [input_masks[i] for i in anchors])
            if scale < 1:
                repairs = [cv2.resize(image, (r-l, b-t), interpolation=cv2.INTER_LINEAR) for image in repairs]
            for i, ((index, _), mask) in enumerate(zip(batch, masks)):
                slot = order[i]
                repaired = repairs[slot]
                if i != anchors[slot]:
                    # Retain the source frame's own small color and lighting differences.
                    delta = cv2.subtract(crops[i], crops[anchors[slot]], dtype=cv2.CV_16S)
                    repaired = cv2.add(repaired, delta, dtype=cv2.CV_8U)
                cv2.copyTo(cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR), mask, frames[index][t:b, l:r])
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
    if options.ocr_backend == 'onnx_cuda' and device != 'cpu':
        from lite_ocr import cuda_detector
        detector = cuda_detector(detector,ocr_path)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Lite 未检测到 CUDA；CPU 测试可设置 LITE_DEVICE=cpu')
    checkpoint_info = {}
    if os.getenv('LITE_BENCH_LAMA'):
        import hashlib
        weights = Path(os.environ['LITE_BENCH_LAMA']).resolve(strict=True)
        with weights.open('rb') as reader:
            sha = hashlib.sha256()
            for block in iter(lambda: reader.read(1 << 20), b''):
                sha.update(block)
        if sha.hexdigest() != os.environ.get('LITE_BENCH_LAMA_SHA256'):
            raise ValueError('试验 LaMa 权重摘要不匹配')
        checkpoint_info['checkpoint_sha256'] = sha.hexdigest()
    with span('lite_lama_load', device=device, **checkpoint_info):
        lama = LamaInpaint(torch.device(device), str(weights), batch_size=options.batch_size,
                           compact_transfer=options.compact_transfer)
    return detector, lama


def run(folder, request):
    plan = request.get('resolved_plan')
    options = LiteOptions(**plan['parameters']['lite']) if plan else LiteOptions()
    detector, lama = load_models(options)
    source = folder / 'source.video'
    rois = [r['roi'] for r in request['regions']]
    find = lambda frame: detect(frame, rois, detector, options)
    profiles = {}
    if options.layout_profile:
        from lite_profile import profile_source
        with span('lite_layout_probe',device='cpu'):
            profiles = profile_source(source, rois, find, variable_line_panels=options.variable_line_panels)
    with ExitStack() as resources:
        video = resources.enter_context(capture(source))
        fps, total = geometry(video)
        options = options.for_clip(video.get(cv2.CAP_PROP_FRAME_COUNT), fps)
        if not options.optimize_lama_graph:
            import torch
            resources.enter_context(torch.jit.optimized_execution(False))
        ok, first = video.read()
        if not ok or list(first.shape[1::-1]) != request['dimensions']:
            raise ValueError('视频无法解码或显示尺寸不一致')
        anchor = find(first)
        tracks = {}
        previous_boxes = None
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
                if options.layout_profile:
                    original_last = boxes[-1]
                    boxes = reject_transient_skin(frames,boxes,previous_boxes,anchor if ok else None)
                    previous_boxes = original_last
                repaired = (restore_joint(frames, boxes, rois, lama, options, tracks, profiles) if options.mask_mode != 'boxes'
                            else restore(frames, boxes, lama, options))
                for frame in repaired:
                    writer.write(memoryview(np.ascontiguousarray(frame)))
                repaired_count += sum(bool(selection) for selection in boxes)
                count += len(frames)
                report(min(98, 4 + 94 * count / total), '裁带检测并逐句去字幕')
                if not ok:
                    break
    print(f'Lite: {repaired_count}/{count} 帧命中 OCR', flush=True)


def verify_models(product=False):
    options = LiteOptions()
    if product:
        import json
        import subprocess

        raw = subprocess.check_output(['php', str(ROOT/'php/product_modes.php'), 'lite', 'subtitle'])
        options = LiteOptions(**json.loads(raw)['parameters']['lite'])
    detector, lama = load_models(options)
    frame = np.zeros((64, 160, 3), dtype=np.uint8)
    cv2.putText(frame, 'TEST', (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    detect(frame, [(0, 0, 160, 64)], detector, options)
    restore([frame], [[(10, 15, 95, 47)]], lama, options)
    print(f'Lite 字幕定位设备 {getattr(detector, "inference_device", "cpu")} 与 LaMa 设备 {lama.device} 校验通过。')


if __name__ == '__main__' and sys.argv[1:] in (['--verify'], ['--verify-product']):
    verify_models(product=sys.argv[1] == '--verify-product')
