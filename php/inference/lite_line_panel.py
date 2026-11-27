"""Recover a translucent line backing whose width changes with the caption."""
from itertools import combinations

import numpy as np

from lite_layer import Panel, boundary_peaks


def edge_gain(outside, inside, transmission, offset):
    contrast = np.abs(inside - outside).mean(axis=-1)
    error = np.abs(inside - (transmission * outside + offset)).mean(axis=-1)
    return contrast - error


def fit_edges(images, top, bottom):
    width = images.shape[2]
    columns = slice(round(.32 * width), round(.68 * width), 3)
    outside = images[:, [top - 2, bottom + 1], columns].astype(np.float32)
    inside = images[:, [top + 1, bottom - 2], columns].astype(np.float32)
    xs, ys = outside.reshape(-1, 3), inside.reshape(-1, 3)
    contrast = np.abs(ys - xs).mean(axis=1)
    seen = contrast > 7
    if seen.mean() < .33:
        return None
    x, y = xs[seen], ys[seen]
    # Uncaptioned frames can contribute a second, unchanged layer. Seed the
    # fit on high-contrast observations before refining its inliers.
    fits = []
    for transmission in np.arange(.15, .88, .05):
        offset = np.median(y - transmission * x, axis=0)
        fits.append((np.median(np.abs(y - (transmission * x + offset)).mean(axis=1)),
                     transmission, offset))
    _, transmission, offset = min(fits, key=lambda item: item[0])
    for _ in range(3):
        error = np.abs(y - (transmission * x + offset)).mean(axis=1)
        keep = error <= max(5., np.percentile(error, 55))
        xx, yy = x[keep], y[keep]
        dx, dy = xx - xx.mean(0), yy - yy.mean(0)
        spread = (dx * dx).sum()
        if spread < 120 * len(xx):
            return None
        transmission = float((dx * dy).sum() / spread)
        offset = np.median(yy - transmission * xx, axis=0)
    if not .18 < transmission < .84 or offset.min() < -7 or offset.max() > 255 * (1 - transmission) + 7:
        return None
    residual = np.abs(inside - (transmission * outside + offset)).mean(axis=3)
    raw = np.abs(inside - outside).mean(axis=3)
    supports = ((residual + 3 < .68 * raw) & (residual < 16)).mean(axis=2).min(axis=1)
    if (supports > .38).sum() < max(4, .38 * len(images)):
        return None
    return float(np.median(supports)), transmission, offset.astype(np.float32)


class LinePanel(Panel):
    variable_edge = None
    edge_positions = ()

    def scene_print(self, frame, quad):
        """Keep slanted print if the learned horizontal caption backing is absent."""
        if self.native_box is None or quad.shape != (4, 2):
            return False
        upper = quad[1] - quad[0]
        lower = quad[2] - quad[3]
        run, rise = np.abs((upper + lower) / 2)
        depth = (np.linalg.norm(quad[3]-quad[0]) + np.linalg.norm(quad[2]-quad[1])) / 2
        if run < 1.5*depth or rise < max(.45*depth, .075*run):
            return False
        left, top, right, bottom = self.native_box
        x = max(left, int(np.floor(quad[:, 0].min())))
        end = min(right, int(np.ceil(quad[:, 0].max())))
        if end <= x or top < 2 or bottom+1 >= frame.shape[0]:
            return False
        cols = np.arange(x, end, max(1, (end-x)//80))
        outside = frame[[[top-2], [bottom+1]], cols].astype(np.float32)
        inside = frame[[[top+1], [bottom-2]], cols].astype(np.float32)
        supported = edge_gain(outside, inside, self.transmission, self.offset) > 3
        return supported.mean(axis=1).min() < .27

    def fallback_boxes(self, roi, boxes):
        # Words entering the selection from outside the measured backing are
        # often scene signs. Keep full words available for a change of style.
        _, y, _, end = roi
        _, top, _, bottom = self.native_box
        height = bottom - top
        keep = []
        for box in boxes:
            _, a, _, b = box
            clipped = a <= 1 or b >= end-y-1
            overlap = max(0, min(y+b, bottom) - max(y+a, top))
            if not clipped or overlap >= .68*(b-a) and b-a <= 1.12*height:
                keep.append(box)
        return keep

    def measure_edges(self, frame):
        left, top, right, bottom = self.estimated_bounds()
        step = max(1, (right - left) // 130)
        columns = np.arange(left, right, step)
        radius = int(np.ceil((self.window[3] - self.window[1]) / self.size[1])) + 2
        estimates = []
        for edge, direction in ((top, 1), (bottom, -1)):
            # Edge indexes denote the first pixel on the lower side of the jump.
            ys = np.arange(max(3, edge - radius), min(frame.shape[0] - 2, edge + radius + 1))
            if not len(ys):
                return (None, None)
            a = frame[(ys - 2 * direction)[:, None], columns].astype(np.float32)
            b = frame[(ys + direction)[:, None], columns].astype(np.float32)
            broad = np.median(edge_gain(a, b, self.transmission, self.offset), axis=1)
            near = frame[(ys - 1)[:, None], columns].astype(np.float32)
            far = frame[ys[:, None], columns].astype(np.float32)
            slope = np.median(np.abs(far - near).mean(axis=2), axis=1)
            winner = int(np.argmax(broad + slope))
            estimates.append(int(ys[winner]) if broad[winner] >= 3 else None)
        return estimates

    def calibrate(self, observations):
        if not observations:
            return False
        native = []
        for column in zip(*observations):
            observed = [value for value in column if value is not None]
            if len(observed) < 2:
                return False
            median = round(float(np.median(observed)))
            if sum(abs(value - median) <= 1 for value in observed) < .7 * len(observed):
                return False
            native.append(median)
        if native[1] - native[0] < 8:
            return False
        self.native_box = self.window[0], native[0], self.window[2], native[1]
        return True

    def calibrate_heights(self, observations):
        """Accept two repeatedly measured heights with one shared horizontal edge."""
        pairs = np.asarray([pair for pair in observations if None not in pair], dtype=int)
        if len(pairs) < 4:
            return False
        top, bottom = np.rint(np.median(pairs, axis=0)).astype(int)
        spread = np.ptp(pairs, axis=0)
        varying = int(np.argmax(spread))
        if spread[1-varying] > 2 or not 3 <= spread[varying] <= .22*(bottom-top):
            return False
        positions = np.sort(pairs[:, varying])
        split = int(np.argmax(np.diff(positions))) + 1
        groups = (positions[:split], positions[split:])
        if (positions[split] - positions[split-1] < 3
                or any(len(group) < 2 or np.ptp(group) > 2 for group in groups)):
            return False
        self.native_box = self.window[0], top, self.window[2], bottom
        self.variable_edge = varying
        self.edge_positions = np.unique([int(p)+d for p in positions for d in (-1, 0, 1)])
        return True

    def measured_height(self, frame, left, right, top, bottom):
        """Locate the learned boundary on the current frame before reversing the tint."""
        positions = self.edge_positions
        positions = positions[(positions >= 3) & (positions+2 < frame.shape[0])]
        columns = np.arange(left+(right-left)//8, right-(right-left)//8, max(1, (right-left)//88))
        if not len(positions) or not len(columns):
            return top, bottom
        direction = 1 if self.variable_edge == 0 else -1
        outer = positions - int(direction == 1)
        inner = outer + direction
        def pixels(rows):
            return frame[rows[:, None], columns].astype(np.float32)
        broad = edge_gain(pixels(outer-direction), pixels(inner+direction), self.transmission, self.offset)
        edge = edge_gain(pixels(outer), pixels(inner), self.transmission, self.offset)
        winner = int(positions[np.argmax(np.median(broad+edge, axis=1))])
        return (winner, bottom) if self.variable_edge == 0 else (top, winner)

    def restore(self, frame, roi, boxes, *, frame_index=None):
        xmin, top, xmax, bottom = self.native_box
        x, y, _, _ = roi
        height = bottom - top
        boxes = [box for box in boxes if top + .18 * height < y + (box[1] + box[3]) / 2 < bottom - .18 * height]
        if not boxes:
            if frame_index is not None:
                self._verified_line = None
            return None
        left = x + min(box[0] for box in boxes)
        right = x + max(box[2] for box in boxes)
        if self.variable_edge is not None:
            top, bottom = self.measured_height(frame, left, right, top, bottom)
            height = bottom - top
        # Search for each side beside the detected ink. This avoids confusing
        # letters and nearby scene objects with the boundary of the overlay.
        ys = np.arange(top + 3, bottom - 3, max(1, height // 34))
        sides = []
        for edge, direction in ((left, 1), (right, -1)):
            limits = sorted((edge - direction * round(.8 * height), edge + direction * round(.18 * height)))
            positions = np.arange(max(xmin + 3, limits[0]), min(xmax - 3, limits[1]) + 1)
            if not len(positions):
                return None
            pixels = frame[ys[:, None], positions].astype(np.float32)
            before = frame[ys[:, None], positions - 1].astype(np.float32)
            a = frame[ys[:, None], positions - 2 * direction].astype(np.float32)
            b = frame[ys[:, None], positions + direction].astype(np.float32)
            gain = np.median(edge_gain(a, b, self.transmission, self.offset), axis=0)
            slope = np.median(np.abs(pixels - before).mean(axis=2), axis=0)
            i = int(np.argmax(gain + np.minimum(slope, np.maximum(0, gain))))
            sides.append((int(positions[i]), float(gain[i])))
        (l, a), (r, b) = sides
        if r - l < 1.4 * height or (frame_index is None and min(a, b) < 2):
            return None
        cols = np.arange(l + 3, r - 3, max(1, (r - l) // 100))
        outside = frame[[[top - 2], [bottom + 1]], cols].astype(np.float32)
        inside = frame[[[top + 1], [bottom - 2]], cols].astype(np.float32)
        gains = edge_gain(outside, inside, self.transmission, self.offset)
        upper, lower = (gains > 3).mean(axis=1)
        strong = (a >= 2, b >= 2, upper >= .27, lower >= .27)
        verified = all(strong)
        if not verified:
            previous = getattr(self, '_verified_line', None)
            if (frame_index is None or previous is None or not 0 < frame_index-previous[0] <= 12
                    or (top, bottom) != previous[-2:]):
                return None
            _, old_left, old_right, old_text_left, old_text_right, _, _ = previous
            # A background colour close to the panel tint can hide one edge.
            # The other three edges, word positions and current ink must still agree.
            if (sum(strong) < 3 or min(a, b) < 0 or min(upper, lower) < .12 or max(upper, lower) < .38
                    or max(abs(l-old_left), abs(r-old_right)) > max(3, .09*height)
                    or max(abs(left-old_text_left), abs(right-old_text_right)) > .15*height):
                return None
            if not strong[0]:
                l = old_left
            if not strong[1]:
                r = old_right
        restored = self.restore_bounds(frame, roi, boxes, (l, top, r, bottom), cover_words=True)
        if frame_index is not None and verified and restored is not None:
            self._verified_line = frame_index, l, r, left, right, top, bottom
        return restored


def learn_line_panel(thumbnails, window):
    images = np.asarray(thumbnails)
    _, height, width = images.shape[:3]
    if height < 14 or width < 2 * height:
        return None
    rows = boundary_peaks(images, False)
    if len(rows) < 2:
        return None
    region = images[:, :, width // 3:2 * width // 3:2].astype(np.int16)
    delta = np.abs(region[:, 1:] - region[:, :-1]).mean(axis=3)
    quality = np.percentile(np.median(delta, axis=2), 75, axis=0)
    extended = []
    for i in np.argsort(quality)[::-1]:
        if quality[i] < 4 or len(extended) >= 8:
            break
        if all(abs(int(i) + 1 - v) > 3 for v in extended):
            extended.append(int(i) + 1)
    best, result = 0., None
    for top, bottom in combinations(sorted(set(rows + extended)), 2):
        if top < 2 or bottom + 1 >= height or not .18 * height < bottom - top < .83 * height:
            continue
        fit = fit_edges(images, top, bottom)
        if fit is not None and fit[0] > best:
            best, transmission, offset = fit
            probe = (round(.34 * width), top, round(.66 * width), bottom)
            result = LinePanel(window, (width, height), probe, transmission, offset)
    return result
