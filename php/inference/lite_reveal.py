"""Recover large opaque captions from a nearby visible part of the source shot."""
from dataclasses import dataclass

import cv2
import numpy as np

from job_trace import span


def features(image, rect, sift):
    h, w = image.shape[:2]
    scale = min(1., 800 / max(w, h))
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None,
                      fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    left, top, right, bottom = np.asarray(rect) * scale
    allowed = np.full(gray.shape, 255, np.uint8)
    allowed[max(0, int(top)-4):int(np.ceil(bottom))+4,
            max(0, int(left)-4):int(np.ceil(right))+4] = 0
    points, descriptors = sift.detectAndCompute(gray, allowed)
    if descriptors is None:
        return np.empty((0, 2), np.float32), None
    return np.float32([p.pt for p in points]) / scale, descriptors


def registration(source, donor):
    p, d = source
    q, e = donor
    if min(len(p), len(q)) < 12:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d, e, k=2)
    matches = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance:
            matches.append((pair[0].queryIdx, pair[0].trainIdx))
    if len(matches) < 12:
        return None
    i, j = np.asarray(matches).T
    affine, inliers = cv2.estimateAffinePartial2D(q[j], p[i], method=cv2.RANSAC, ransacReprojThreshold=2.)
    if affine is None or inliers.sum() < 12 or inliers.mean() < .36:
        return None
    scale = np.hypot(*affine[0, :2])
    if not .80 < scale < 1.22 or abs(affine[1, 0]) > .2*scale:
        return None
    return affine


def agrees_around_card(source, warped, valid, rect):
    l, t, r, b = rect
    h, w = valid.shape
    margin = max(8, min(24, (b-t)//5))
    # Exclude the original antialiased edge and out-of-view parts of the donor.
    sides = [(l, max(0,t-margin), r, t-3), (l, b+3, r, min(h,b+margin)),
             (max(0,l-margin), t, l-3, b), (r+3, t, min(w,r+margin), b)]
    agreeing = 0
    for x, y, end, bottom in sides:
        if end <= x or bottom <= y:
            continue
        region = np.s_[y:bottom:2, x:end:2]
        present = valid[region] != 0
        if present.mean() < .60 or present.sum() < 20:
            continue
        error = cv2.absdiff(source[region], warped[region]).mean(2)[present]
        if np.median(error) > 7 or np.quantile(error, .80) > 16:
            return False
        agreeing += 1
    return agreeing >= 3


@dataclass
class Donor:
    frame_index: int
    image: np.ndarray
    descriptors: tuple
    valid: np.ndarray


class VisibleCardBackground:
    def __init__(self, read, find, *, fps, total, read_reference=None):
        self.read = read
        self.read_reference = read_reference
        self.find = find
        self.radius = max(1, round(3.5*fps))
        self.total = total
        self.sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=.006)
        self.cache = {}

    def candidates(self, index, bounds, window, rect, shape):
        l, t, r, b = bounds
        x, y, end, bottom = window
        def still_covered(frame):
            return (frame[t:b:5, l:r:5].min(2) > 229).mean() > .55

        found = []
        for direction in (-1, 1):
            available = min(self.radius, index if direction < 0 else self.total-1-index)
            previous = 0
            for step in sorted({min(available, max(1, round(self.radius*fraction)))
                                for fraction in (.04, .14, .30, .58, 1.)}):
                if step == 0:
                    break
                frame = self.read(index+direction*step)
                if frame is None or frame.shape != shape:
                    break
                if still_covered(frame):
                    previous = step
                    continue
                # The clear interval can last just one frame before the next caption.
                lo, hi = previous, step
                while hi-lo > 1:
                    mid = (lo+hi)//2
                    probe = self.read(index+direction*mid)
                    if probe is None or probe.shape != shape:
                        break
                    if still_covered(probe):
                        lo = mid
                    else:
                        hi, frame = mid, probe
                source_index = index + direction*hi
                if self.read_reference is not None:
                    frame = self.read_reference(source_index)
                    if frame is None or frame.shape != shape or still_covered(frame):
                        break
                detected = self.find(frame)
                if not any(max(a,l) < min(c,r) and max(v,t) < min(z,b)
                           for a, v, c, z in detected):
                    image = frame[y:bottom, x:end].copy()
                    valid = np.full(image.shape[:2], 255, np.uint8)
                    for a, v, c, z in detected:
                        if a < end and x < c and v < bottom and y < z:
                            valid[max(0,v-y):z-y, max(0,a-x):c-x] = 0
                    found.append(Donor(source_index, image, features(image, rect, self.sift), valid))
                break
        return found

    def recover(self, frame, index, roi, mask, boxes):
        x, y, end, bottom = roi
        bx, by, w, h = cv2.boundingRect(mask)
        if w*h < .15*mask.size or cv2.countNonZero(mask) < .92*w*h:
            return None
        if any(not (bx <= (l+r)/2 < bx+w and by <= (t+b)/2 < by+h) for l,t,r,b in boxes):
            return None
        l, t, r, b = x+bx, y+by, x+bx+w, y+by+h
        margin = max(75, round(.55*h))
        wx, wy = max(0, l-margin), max(0, t-margin)
        wr, wb = min(frame.shape[1], r+margin), min(frame.shape[0], b+margin)
        rect = (l-wx, t-wy, r-wx, b-wy)
        bounds = (l, t, r, b)
        item = self.cache.get(roi)
        if item is None or item[0] != bounds or abs(item[1]-index) > self.radius:
            donors = self.candidates(index, bounds, (wx, wy, wr, wb), rect, frame.shape)
            self.cache[roi] = bounds, index, donors
        donors = [v for v in self.cache[roi][2] if abs(v.frame_index-index) <= self.radius]
        if not donors:
            return None
        source = frame[wy:wb, wx:wr]
        keypoints = features(source, rect, self.sift)
        for donor in sorted(donors, key=lambda v: abs(v.frame_index-index)):
            affine = registration(keypoints, donor.descriptors)
            if affine is None:
                continue
            size = (wr-wx, wb-wy)
            warped = cv2.warpAffine(donor.image, affine, size)
            valid = cv2.warpAffine(donor.valid, affine, size)
            valid = cv2.erode((valid == 255).astype(np.uint8)*255, np.ones((3, 3), np.uint8))
            area = np.s_[t-wy:b-wy, l-wx:r-wx]
            use = valid[area] & mask[by:by+h, bx:bx+w]
            if cv2.countNonZero(use) < .70*cv2.countNonZero(mask):
                continue
            if not agrees_around_card(source, warped, valid, rect):
                continue
            with span('lite_visible_card', device='cpu', frame=index, source_frame=donor.frame_index,
                      recovered_pixels=cv2.countNonZero(use), mask_pixels=cv2.countNonZero(mask)):
                cv2.copyTo(warped[area], use, frame[t:b, l:r])
                residual = mask.copy()
                residual[by:by+h, bx:bx+w] &= cv2.bitwise_not(use)
                return residual
        return None


def restore_from_source(tool, frame, index, roi, boxes, mask, card):
    if tool is None or card is None or len(boxes) < 3:
        return mask
    # Never expand the write area beyond the chosen mask or overwrite a nearby native graphic.
    selected = mask & card
    residual = tool.recover(frame, index, roi, selected, boxes)
    if residual is None:
        return mask
    return mask & (cv2.bitwise_not(selected) | residual)
