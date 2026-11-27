"""Recover a repeated, translucent horizontal color gradient and its printed decoration."""
from itertools import combinations

import cv2
import numpy as np

from lite_layer import Panel


def edge_samples(images, box):
    left, top, right, bottom = box
    xs = np.arange(left+3, right-2, max(1, (right-left)//160))
    ys = np.arange(top+3, (top+bottom)//2, 2)
    return (
        (images[:, top-2, xs], images[:, top+2, xs], xs),
        (images[:, ys, left-2], images[:, ys, left+2], np.full(len(ys), left)),
        (images[:, ys, right+2], images[:, ys, right-2], np.full(len(ys), right)),
    )


def fit_gradient(outside, inside, positions):
    """Fit inside = transmission * outside + offset + horizontal position * slope."""
    n = len(outside)
    x, y = outside.reshape(-1, 3).astype(float), inside.reshape(-1, 3).astype(float)
    u = np.tile(positions, n)[:, None]
    weight = np.ones((len(x), 1))
    for _ in range(5):
        total = weight.sum()
        center = (u*weight).sum()/total
        axis = u-center
        spread = (weight*axis**2).sum()
        if spread < .001:
            return None
        xmean, ymean = (weight*x).sum(0)/total, (weight*y).sum(0)/total
        xslope, yslope = (weight*axis*x).sum(0)/spread, (weight*axis*y).sum(0)/spread
        dx, dy = x-xmean-axis*xslope, y-ymean-axis*yslope
        variation = (weight*dx*dx).sum()
        if variation/total < 120:
            return None
        transmission = float((weight*dx*dy).sum()/variation)
        slope = yslope-transmission*xslope
        offset = ymean-transmission*xmean-center*slope
        error = np.abs(y-(transmission*x+offset+u*slope)).mean(axis=1)
        weight = np.minimum(1, max(2., np.median(error)*1.4)/np.maximum(.1, error[:, None]))**2
    if not .18 <= transmission <= .82 or np.max(np.abs(slope)) < 22 or np.median(error) > 7:
        return None
    endpoints = np.array([offset-.5*slope, offset+.5*slope])
    if endpoints.min() < -8 or endpoints.max() > (1-transmission)*255+10:
        return None
    return transmission, offset.astype(np.float32), slope.astype(np.float32)


class GradientPanel(Panel):
    calibration_tolerance = 2

    def __init__(self, window, size, box, transmission, offset, slope):
        super().__init__(window, size, box, transmission, offset)
        self.slope = slope
        self.decoration = None

    def offsets(self, xs, left, right):
        u = ((xs-left)/(right-left)-.5).astype(np.float32)
        return self.offset + u[:, None]*self.slope

    def active(self, images):
        left, _, right, _ = self.box
        votes = []
        for outside, inside, xs in edge_samples(images, self.box):
            outside, inside = outside.astype(np.float32), inside.astype(np.float32)
            distance = np.abs(inside-outside).mean(axis=2)
            residual = np.abs(inside-(self.transmission*outside+self.offsets(xs, left, right))).mean(axis=2)
            raw, error = np.median(distance, axis=1), np.median(residual, axis=1)
            votes.append((error+4 < .66*raw) & (error < 15))
        # Decorations can interrupt a side border. The opposite side and top must agree.
        return votes[0] & (votes[1] | votes[2])

    def measure_edges(self, frame):
        wx, wy, wr, wb = self.window
        small = cv2.resize(frame[wy:wb, wx:wr], self.size, interpolation=cv2.INTER_AREA)
        if not self.active(small[None])[0]:
            return [None]*4
        bounds = self.estimated_bounds()
        edges = []
        for side, initial in enumerate(bounds):
            vertical = side in (0, 2)
            image = frame.transpose(1, 0, 2) if vertical else frame
            lo, hi = bounds[1::2] if vertical else bounds[::2]
            margin = max(3, (hi-lo)//11)
            radius = 2+int(np.ceil((wr-wx)/self.size[0] if vertical else (wb-wy)/self.size[1]))
            a, b = max(1, initial-radius), min(len(image)-1, initial+radius)
            strip = image[a-1:b+1, lo+margin:hi-margin:max(1, (hi-lo)//160)].astype(np.int16)
            if len(strip) < 2 or not strip.shape[1]:
                edges.append(None)
                continue
            strength = np.median(np.abs(np.diff(strip, axis=0)).mean(2), axis=1)
            edges.append(a+int(np.argmax(strength)) if strength.max() >= 6 else None)
        return edges

    def learn_decoration(self, images):
        l, t, r, b = self.box
        visible = images[:, t:b, l:r].astype(np.float32)
        offset = self.offsets(np.arange(l,r), l, r)
        outside_range = ((offset-visible).max(3)>11) | ((visible-(offset+self.transmission*255)).max(3)>11)
        # Fixed printed motifs recur at the same screen pixels while dialogue and scenery change.
        self.decoration = (outside_range.mean(0) >= .57).astype(np.uint8)*255
        self.decoration = cv2.morphologyEx(self.decoration, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        # Ink inside the admissible tint range still belongs to a printed border.
        # Complete only narrow edge strips with recurring pigment across their width.
        for edge in (self.decoration, self.decoration[::-1]):
            supported = (edge != 0).mean(axis=1) >= .24
            rows = np.flatnonzero(supported)
            if not len(rows) or rows[0] > .08*len(edge):
                continue
            gaps = np.flatnonzero(np.diff(rows) > 2)
            depth = int(rows[gaps[0]] if len(gaps) else rows[-1]) + 1
            if not 4 <= depth <= .38*len(edge):
                continue
            strip = edge[:depth]
            if all((tile != 0).mean() > .13 for tile in np.array_split(strip, 8, axis=1)):
                edge[:min(len(edge), depth + 2)] = 255

    def restore(self, frame, roi, boxes):
        wx, wy, wr, wb = self.window
        small = cv2.resize(frame[wy:wb, wx:wr], self.size, interpolation=cv2.INTER_AREA)
        if not self.active(small[None])[0]:
            return None
        l, t, r, b = self.native_box
        x, y, rr, bb = roi
        left, top, right, bottom = max(x,l), max(y,t), min(rr,r), min(bb,b)
        if left >= right or top >= bottom:
            return None
        visible = frame[top:bottom, left:right].astype(np.float32)
        offset = self.offsets(np.arange(left, right), l, r)
        ink = np.zeros((bb-y, rr-x), np.uint8)
        decoration = cv2.resize(self.decoration, (r-l,b-t), interpolation=cv2.INTER_NEAREST)
        ink[top-y:bottom-y,left-x:right-x] = decoration[top-t:bottom-t,left-l:right-l]
        foreign = ((offset-visible).max(2)>9) | ((visible-(offset+self.transmission*255)).max(2)>9)
        for bx, by, br, bt in boxes:
            bx, by, br, bt = max(bx+x,left), max(by+y,top), min(br+x,right), min(bt+y,bottom)
            if bx < br and by < bt:
                ink[by-y:bt-y,bx-x:br-x] |= foreign[by-top:bt-top,bx-left:br-left].astype(np.uint8)*255
        ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        # Inversion preserves the portion of the original scene transmitted through the colored plate.
        restored = np.clip((visible-offset)/self.transmission, 0, 255).astype(np.uint8)
        return (left,top,right,bottom), restored, ink


def learn_gradient_panel(images, window, xs, ys):
    n, height, width = images.shape[:3]
    best = None
    best_count = 0
    for left, right in combinations(xs, 2):
        if right-left < .64*width or min(left, width-1-right) < 2:
            continue
        for top, bottom in combinations(ys, 2):
            if bottom-top < .22*height or min(top,height-1-bottom) < 2:
                continue
            box = (left, top, right, bottom)
            outside, inside, pos = edge_samples(images, box)[0]
            fit = fit_gradient(outside, inside, (pos-left)/(right-left)-.5)
            if fit is None:
                continue
            panel = GradientPanel(window, (width,height), box, *fit)
            present = panel.active(images)
            count = int(present.sum())
            if count >= max(4, .40*n) and count > best_count:
                panel.learn_decoration(images[present])
                best, best_count = panel, count
    return best
