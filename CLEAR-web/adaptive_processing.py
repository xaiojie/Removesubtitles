"""Region analysis shared by the CLEAR and ProPainter production workers."""
from collections import Counter
import json

import cv2
import numpy as np


def foreground_core(crop):
    """Find light, dark and colored strokes using local multichannel contrast."""
    height, width = crop.shape[:2]
    size = max(3, min(21, min(height, width) // 3) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    light = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, kernel).max(axis=2)
    dark = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel).max(axis=2)
    contrast = np.maximum(light, dark)
    noise = float(np.median(contrast))
    threshold = max(12, min(36, noise * 2.5))
    return (contrast >= threshold).astype(np.uint8) * 255


def complete_components(core):
    """Cover component interiors, thin punctuation and the gaps within a glyph."""
    linked = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(core)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height >= 3:
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return mask


def subtitle_mask(crop):
    mask = complete_components(foreground_core(crop))
    radius = max(2, min(10, round(min(crop.shape[:2]) * 0.04)))
    mask = cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    coverage = np.count_nonzero(mask) / mask.size
    # A weak or highly textured detection uses the entire user-authorized region.
    if coverage < 0.005 or coverage > 0.65:
        mask[:] = 255
    return mask


def border_background(frame, roi):
    """Identify flat rows connected to a frame edge using both sides of the ROI."""
    left, top, right, bottom = roi
    outside = np.concatenate((frame[:, :left], frame[:, right:]), axis=1)
    count = bottom - top
    if outside.shape[1] < max(16, frame.shape[1] // 12):
        return np.zeros(count, dtype=bool), np.zeros((count, 3), dtype=np.uint8)
    background = np.median(outside, axis=1).astype(np.uint8)
    difference = np.abs(outside.astype(np.int16) - background[:, None, :]).max(axis=2)
    uniform = (difference <= 8).mean(axis=1) >= 0.97
    # Uniform image rows separated from the frame boundary are not letterboxing.
    top_band = np.logical_and.accumulate(uniform)
    bottom_band = np.logical_and.accumulate(uniform[::-1])[::-1]
    edge_color = np.minimum(
        np.abs(background.astype(np.int16) - background[0]).max(axis=1),
        np.abs(background.astype(np.int16) - background[-1]).max(axis=1),
    )
    rows = (top_band | bottom_band) & (edge_color <= 8)
    crop = frame[top:bottom, left:right]
    local_difference = np.abs(crop.astype(np.int16) - background[top:bottom, None, :]).max(axis=2)
    # Dense picture content inside a dark scene keeps the model route.
    rows = rows[top:bottom] & ((local_difference <= 18).mean(axis=1) >= 0.55)
    return rows, background[top:bottom]


def prepare_subtitles(frames, regions):
    analyses = []
    for region in regions:
        if region['type'] != 'subtitle':
            continue
        left, top, right, bottom = region['roi']
        originals = [frame[top:bottom, left:right].copy() for frame in frames]
        observations = [border_background(frame, region['roi']) for frame in frames]
        # A one-frame black flash cannot establish a new border inside a shot.
        support = np.mean([rows for rows, _ in observations], axis=0)
        established = support >= 0.75
        masks, background_rows = [], []
        for frame, original, (rows, colors) in zip(frames, originals, observations):
            rows = rows & established
            frame[top:bottom, left:right][rows] = colors[rows, None, :]
            mask = subtitle_mask(original)
            mask[rows] = 0
            masks.append(mask)
            background_rows.append(rows)
        analyses.append(dict(region=region, masks=masks,
                             background_rows=background_rows,
                             needs_model=any(mask.any() for mask in masks)))
    return analyses


def composite_subtitle(original, generated, mask):
    if not mask.any():
        return original.copy()
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.25)
    alpha[mask > 0] = 1.0
    result = original.astype(np.float32) * (1 - alpha[..., None])
    result += generated.astype(np.float32) * alpha[..., None]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def context_boxes(mask, width, height):
    """Merge overlapping context crops while keeping distant selections separate."""
    count, _, statistics, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for x, y, selected_width, selected_height, area in statistics[1:count]:
        if area < 1:
            continue
        margin_x = max(64, selected_width)
        margin_y = max(64, selected_height)
        box = [max(0, x - margin_x), max(0, y - margin_y),
               min(width, x + selected_width + margin_x),
               min(height, y + selected_height + margin_y)]
        index = 0
        while index < len(boxes):
            previous = boxes[index]
            if (box[0] <= previous[2] and previous[0] <= box[2]
                    and box[1] <= previous[3] and previous[1] <= box[3]):
                box = [min(box[0], previous[0]), min(box[1], previous[1]),
                       max(box[2], previous[2]), max(box[3], previous[3])]
                boxes.pop(index)
                index = 0
            else:
                index += 1
        boxes.append(box)
    return boxes


class QualityReport:
    """Bounded, heuristic observations of target features surviving composition."""
    def __init__(self, engine, fps):
        self.engine = engine
        self.fps = fps
        self.frames = 0
        self.methods = Counter()
        self.suspected = []
        self.suspected_count = 0

    def observe(self, index, original, result, regions):
        self.frames += 1
        suspicious = []
        for number, region in enumerate(regions, 1):
            left, top, right, bottom = region['roi']
            source = original[top:bottom, left:right]
            output = result[top:bottom, left:right]
            core = foreground_core(source) > 0
            coverage = np.count_nonzero(core) / core.size
            if not 0.005 <= coverage <= 0.4:
                continue
            surviving = foreground_core(output) > 0
            agreement = np.count_nonzero(core & surviving) / np.count_nonzero(core)
            change = np.abs(source.astype(np.int16) - output).mean(axis=2)
            if agreement >= 0.75 and float(np.median(change[core])) < 8:
                suspicious.append(number)
        if suspicious:
            self.suspected_count += 1
            if len(self.suspected) < 200:
                self.suspected.append(dict(seconds=round(index / self.fps, 3), regions=suspicious))

    def summary(self):
        return dict(engine=self.engine, detector='local-contrast-heuristic',
                    checked_frames=self.frames, methods=dict(self.methods),
                    suspected_frames=self.suspected_count, observations=self.suspected,
                    observations_truncated=self.suspected_count > len(self.suspected))

    def save(self, folder):
        destination = folder / 'quality.json'
        temporary = folder / 'quality.pending.json'
        temporary.write_text(json.dumps(self.summary(), ensure_ascii=False), encoding='utf-8')
        temporary.replace(destination)
