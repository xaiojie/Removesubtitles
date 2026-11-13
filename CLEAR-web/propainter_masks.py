"""Frame-aware, region-limited masks for the upstream ProPainter pipeline."""
import cv2
import numpy as np


def bright_core(crop):
    minimum = crop.min(axis=2)
    chroma = crop.max(axis=2).astype(np.int16) - minimum
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    contrast = gray - cv2.GaussianBlur(gray, (0, 0), 3)
    return (minimum >= 150) & (chroma <= 90) & (contrast >= 7)


def line_boxes(core):
    """Fill line/logo components, including holes and thin internal strokes."""
    mask = np.zeros(core.shape, dtype=np.uint8)
    linked = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(linked, 8)
    height, width = core.shape
    for x, y, w, h, area in statistics[1:count]:
        if area < 4:
            continue
        mask[max(0, y-2):min(height, y+h+2), max(0, x-2):min(width, x+w+2)] = 255
    return mask


class MaskBuilder:
    def __init__(self, regions, dimensions, mode='auto', dilation=8):
        self.regions = regions
        self.width, self.height = dimensions
        self.mode = mode
        self.dilation = dilation
        self.sample_count = 0
        self.support = []
        self.stable = []
        self.templates = []
        self.allowed = np.zeros((self.height, self.width), dtype=np.uint8)
        for region in regions:
            x1, y1, x2, y2 = region['roi']
            self.support.append(np.zeros((y2-y1, x2-x1), dtype=np.uint32))
            self.allowed[y1:y2, x1:x2] = 255

    def observe(self, frame):
        self.sample_count += 1
        if self.mode == 'rectangle':
            return
        for region, support in zip(self.regions, self.support):
            if region['type'] == 'watermark':
                x1, y1, x2, y2 = region['roi']
                support += bright_core(frame[y1:y2, x1:x2])

    def finalize(self):
        self.stable = []
        self.templates = []
        for region, support in zip(self.regions, self.support):
            positive = support[support > 0]
            peak = float(np.percentile(positive, 95)) if positive.size else 0
            # Support adapts to brief/jumping marks while rejecting one-off pixels.
            threshold = max(2, int(np.ceil(min(self.sample_count * 0.30, peak * 0.60))))
            stable = support >= threshold
            template = line_boxes(stable)
            # An unsupported/colored watermark uses the complete user selection.
            if (region['type'] != 'watermark' or self.mode == 'rectangle'
                    or stable.sum() < 6 or not template.any()):
                stable = None
                template = np.full(support.shape, 255, dtype=np.uint8)
            self.stable.append(stable)
            self.templates.append(template)

    def mask(self, frame):
        if len(self.templates) != len(self.regions):
            raise RuntimeError('水印采样尚未完成。')
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for region, stable, template in zip(self.regions, self.stable, self.templates):
            x1, y1, x2, y2 = region['roi']
            if stable is not None:
                current = bright_core(frame[y1:y2, x1:x2])
                agreement = np.count_nonzero(current & stable) / np.count_nonzero(stable)
                if agreement < 0.30:
                    continue
            np.maximum(mask[y1:y2, x1:x2], template, out=mask[y1:y2, x1:x2])
        return mask

    def expanded_mask(self, mask):
        if self.dilation:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                              iterations=self.dilation)
        return cv2.bitwise_and(mask, self.allowed)
