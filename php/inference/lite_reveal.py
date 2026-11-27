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
    # Repeated print or photographic grain can create a large pool of ambiguous descriptors.
    for ratio in (.75, .70):
        matches = []
        for pair in pairs:
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                matches.append((pair[0].queryIdx, pair[0].trainIdx))
        if len(matches) < 12:
            break
        i, j = np.asarray(matches).T
        affine, inliers = cv2.estimateAffinePartial2D(q[j], p[i], method=cv2.RANSAC, ransacReprojThreshold=2.)
        if affine is None or inliers.sum() < 12 or inliers.mean() < .36:
            continue
        scale = np.hypot(*affine[0, :2])
        if .80 < scale < 1.22 and abs(affine[1, 0]) <= .2*scale:
            return affine
    return None


def flow_view(image):
    scale = min(1., 512 / max(image.shape[:2]))
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)
    return gray, np.float32([gray.shape[1] / image.shape[1], gray.shape[0] / image.shape[0]])


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
    last_index: int = -10
    last_matrix: np.ndarray = None
    flow_reference: tuple = None

    def follow(self, source, rect):
        """Match original donor corners directly to this frame, using the last fit as a starting point."""
        if self.flow_reference is None:
            gray, scale = flow_view(self.image)
            allowed = cv2.resize(self.valid, gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
            x, y, r, b = np.rint(np.asarray(rect) * np.tile(scale, 2)).astype(int)
            allowed[max(0, y-4):b+4, max(0, x-4):r+4] = 0
            points = cv2.goodFeaturesToTrack(gray, 220, .0005, 5, mask=allowed)
            self.flow_reference = gray, scale, points
        gray, scale, original = self.flow_reference
        if original is None or len(original) < 16:
            return None
        current, _ = flow_view(source)
        projected = cv2.transform(original / scale, self.last_matrix)
        x, y, r, b = rect
        px, py = projected.reshape(-1, 2).T
        outside = (px < x-6) | (px > r+6) | (py < y-6) | (py > b+6)
        origins = original[outside]
        if len(origins) < 16:
            return None
        settings = dict(winSize=(15, 15), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        dest, ok, error = cv2.calcOpticalFlowPyrLK(gray, current, origins, projected[outside] * scale, **settings)
        if dest is None:
            return None
        back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(current, gray, dest, origins.copy(), **settings)
        if back is None:
            return None
        consistent = (ok.ravel() > 0) & (reverse_ok.ravel() > 0) & (error.ravel() < 16)
        consistent &= np.linalg.norm((back-origins).reshape(-1, 2), axis=1) < .75
        if consistent.sum() < 16:
            return None
        reference = (origins[consistent] / scale).reshape(-1, 2)
        matrix, support = cv2.estimateAffinePartial2D(reference, dest[consistent] / scale,
                                                      method=cv2.RANSAC, ransacReprojThreshold=1.6)
        if matrix is None or support.sum() < 12 or support.mean() < .73:
            return None
        spread = np.ptp(reference[support.ravel() != 0], axis=0)
        if np.any(spread < .30*np.asarray(self.image.shape[1::-1])):
            return None
        # Only small camera motion can continue a previously verified registration.
        delta = cv2.transform(origins / scale, matrix) - projected[outside]
        if np.linalg.norm(delta.reshape(-1, 2), axis=1).max() > .035*max(source.shape[:2]):
            return None
        return matrix


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
        keypoints = None

        def alignments(donor):
            nonlocal keypoints
            if 0 < index-donor.last_index <= 2 and donor.last_matrix is not None:
                affine = donor.follow(source, rect)
                if affine is not None:
                    yield affine, 'flow'
            if keypoints is None:
                keypoints = features(source, rect, self.sift)
            affine = registration(keypoints, donor.descriptors)
            if affine is not None:
                yield affine, 'features'

        def order(donor):
            return (not 0 < index-donor.last_index <= 2, abs(donor.frame_index-index))

        for donor, affine, method in ((d, a, m) for d in sorted(donors, key=order) for a, m in alignments(d)):
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
            donor.last_index, donor.last_matrix = index, affine
            with span('lite_visible_card', device='cpu', frame=index, source_frame=donor.frame_index, method=method,
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


class StaticSubtitleBackground:
    """Restore a stationary shot from consecutive subtitle-free frames observed in the same band."""

    def __init__(self, max_age, contains_text):
        self.max_age = max_age
        self.contains_text = contains_text
        self.index = -2
        self.image = None
        self.confirmed = False
        self.checked = False

    def observe(self, crop, index):
        self.confirmed = False
        if index == self.index + 1 and self.image is not None and self.image.shape == crop.shape:
            delta = cv2.absdiff(crop[::4, ::4], self.image[::4, ::4])
            steady = cv2.inRange(delta, (0, 0, 0), (8, 8, 8))
            self.confirmed = cv2.countNonZero(steady) >= .97 * steady.size
        self.image = crop.copy()
        self.index = index
        self.checked = False

    def restore(self, crop, index, mask, boxes):
        if not self.confirmed or not 0 < index - self.index <= self.max_age or not boxes:
            return False
        # Keep scene cuts and movement out of the reference. The pixels between the
        # printed letters must also agree, even when the inpaint mask is a rectangle.
        delta = cv2.absdiff(crop, self.image)
        matched = cv2.inRange(delta, (0, 0, 0), (14, 14, 14))
        outside = cv2.bitwise_not(cv2.dilate(mask, np.ones((5, 5), np.uint8)))
        visible = cv2.countNonZero(outside)
        if visible < .15 * mask.size or cv2.countNonZero(matched & outside) < .97 * visible:
            self.confirmed = False
            return False
        # A cached frame on which OCR missed the same line must not reinsert it.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        donor_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        dark = cv2.erode(gray, np.ones((7, 7), np.uint8))
        bright = cv2.inRange(crop, (157, 157, 157), (255, 255, 255))
        bright |= cv2.inRange(crop, (0, 145, 175), (140, 255, 255))
        ink = bright & cv2.inRange(cv2.subtract(gray, dark), 42, 255) & mask
        removed = cv2.inRange(cv2.subtract(gray, donor_gray), 35, 255) & ink
        for x, y, r, b in boxes:
            region = np.s_[y:b, x:r]
            count = cv2.countNonZero(ink[region])
            if (count < max(12, .008 * (b-y) * (r-x))
                    or cv2.countNonZero(removed[region]) < .85 * count
                    or cv2.countNonZero(matched[region]) < .44 * (b-y) * (r-x)):
                return False
        # Changes remote from the subtitle strokes may be a hand or a prop crossing the band.
        reach = max(3, min(11, round(.16 * max(b-y for _, y, _, b in boxes))))
        near_ink = cv2.dilate(ink, np.ones((2*reach+1, 2*reach+1), np.uint8))
        unexplained = cv2.bitwise_not(matched | near_ink) & mask
        if cv2.countNonZero(unexplained) > max(12, .009 * cv2.countNonZero(mask)):
            return False
        if not self.checked:
            if self.contains_text(self.image):
                self.confirmed = False
                return False
            self.checked = True
        with span('lite_static_background', device='cpu', frame=index, source_frame=self.index,
                  recovered_pixels=cv2.countNonZero(mask)):
            cv2.copyTo(self.image, mask, crop)
        return True
