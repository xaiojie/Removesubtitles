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


@dataclass(frozen=True)
class LiteOptions:
    # Jobs created before product plans were persisted used these settings.
    sample_stride: int = 6
    detector_max_side: int = 720
    box_padding: int = 5
    context_padding: int = 28
    batch_size: int = 4
    export_crf: int = 18

    def __post_init__(self):
        if min(self.sample_stride, self.detector_max_side, self.batch_size) < 1:
            raise ValueError('Lite 检测间隔、检测尺寸和批量大小须大于零')
        if min(self.box_padding, self.context_padding) < 0 or not 0 <= self.export_crf <= 51:
            raise ValueError('Lite 扩框或编码参数无效')


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


def load_models():
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
        lama = LamaInpaint(torch.device(device), str(weights))
    return detector, lama


def run(folder, request):
    plan = request.get('resolved_plan')
    options = LiteOptions(**plan['parameters']['lite']) if plan else LiteOptions()
    detector, lama = load_models()
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
                boxes, anchor = sample_block(frames, anchor, find)
                if ok:
                    first = frames.pop()
                else:
                    boxes.append(anchor)
                for frame in restore(frames, boxes, lama, options):
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
