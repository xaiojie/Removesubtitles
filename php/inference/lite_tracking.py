"""Spend OCR calls on caption changes inside a bounded lookahead interval."""
import cv2
import numpy as np


class SideLabelTrack:
    """Protect a previously excluded interface label while the same graphic remains visible."""

    def __init__(self):
        self.labels = []
        self.last_frame = -2

    def select(self, crop, boxes, kept, frame_index):
        if len(kept) < len(boxes):
            width = crop.shape[1]
            self.labels = []
            for box in boxes:
                x, y, r, b = box
                if box not in kept and r-x < .22*width and abs((x+r)/2-width/2) > .34*width:
                    self.labels.append((box, crop[y:b, x:r].copy()))
            self.last_frame = frame_index
            return kept
        previous = self.labels if frame_index == self.last_frame+1 else ()
        self.labels = []
        self.last_frame = frame_index
        protected = set()
        for bounds, reference in previous:
            x, y, r, b = bounds
            matched = []
            for box in kept:
                left, top, right, bottom = box
                overlap = max(0, min(r,right)-max(x,left)) * max(0, min(b,bottom)-max(y,top))
                if overlap >= .87*(right-left)*(bottom-top):
                    matched.append(box)
            if matched:
                # Compare against the last confirmed graphic, so small frame-by-frame drift cannot
                # turn a different word in the same place into a permanently protected label.
                delta = cv2.absdiff(crop[y:b, x:r], reference).max(axis=2)
                if delta.mean() <= 3 and np.count_nonzero(delta > 16) <= max(3, .008*delta.size):
                    self.labels.append((bounds, reference))
                    protected.update(matched)
        return [box for box in kept if box not in protected]


def reject_transient_skin(frames,boxes,previous,following):
    """Ignore isolated OCR responses on moving skin when there is no dark or bright printed ink."""
    def overlaps(box,neighbor):
        x,y,r,b=box
        return any(max(0,min(r,rr)-max(x,l))*max(0,min(b,bb)-max(y,t)) > .26*(r-x)*(b-y)
                   for l,t,rr,bb in neighbor)

    result=[]
    for i,regions in enumerate(boxes):
        before=boxes[i-1] if i else previous
        after=boxes[i+1] if i+1<len(boxes) else following
        if before is None or after is None:
            result.append(regions)
            continue
        keep=[]
        for box in regions:
            if not overlaps(box,before) and not overlaps(box,after):
                x,y,r,b=box
                patch=frames[i][y:b,x:r]
                chroma=cv2.cvtColor(patch,cv2.COLOR_BGR2YCrCb)
                skin=cv2.inRange(chroma,(50,133,72),(255,183,135))
                low=patch.max(axis=2)<75
                high=patch.min(axis=2)>225
                if (cv2.countNonZero(skin)>.52*skin.size and (low|high).mean()<.013):
                    continue
            keep.append(box)
        result.append(tuple(keep))
    return result


def covered_union(a, b):
    if len(a) != len(b):
        return None
    merged = []
    for left, right in zip(a, b):
        x,y,r,t = left
        x2,y2,r2,t2 = right
        overlap = max(0,min(r,r2)-max(x,x2))*max(0,min(t,t2)-max(y,y2))
        total = (r-x)*(t-y)+(r2-x2)*(t2-y2)-overlap
        if overlap < .74*total or max(abs(x-x2),abs(r-r2)) > max(5, .30*min(t-y,t2-y2)):
            return None
        merged.append((min(x,x2),min(y,y2),max(r,r2),max(t,t2)))
    return tuple(merged)


def track_block(frames, first, find, rois):
    last = len(frames)-1
    if last == 0:
        return [], first
    anchors = {0:first, last:find(frames[last])}
    output = [None]*last
    sketches = {}

    def edges(index, roi):
        key = (index,tuple(roi))
        if key not in sketches:
            x,y,r,b=roi
            scale = min(1.,288/(r-x))
            size = (max(1,round((r-x)*scale)),max(1,round((b-y)*scale)))
            thumb = cv2.resize(frames[index][y:b,x:r], size, interpolation=cv2.INTER_AREA)
            sketches[key] = cv2.Canny(cv2.cvtColor(thumb,cv2.COLOR_BGR2GRAY),45,115)
        return sketches[key]

    def novelty(lo, hi, boxes):
        scores = np.zeros(hi-lo-1,np.int32)
        for roi in rois:
            x,y,r,b = roi
            known = cv2.dilate(edges(lo,roi) | edges(hi,roi), np.ones((3,3),np.uint8))
            sy,sx = known.shape[0]/(b-y),known.shape[1]/(r-x)
            for bx,by,br,bb in boxes:
                if bx>=r or br<=x or by>=b or bb<=y:
                    continue
                known[max(0,int((by-y)*sy)):int(np.ceil((bb-y)*sy)),
                      max(0,int((bx-x)*sx)):int(np.ceil((br-x)*sx))]=255
            unseen = cv2.bitwise_not(known)
            for j,i in enumerate(range(lo+1,hi)):
                count=cv2.countNonZero(edges(i,roi) & unseen)
                scores[j] += max(0,count-max(12,round(.0018*known.size)))
        return lo+1+int(np.argmax(scores)) if scores.max()>0 else None

    def fill(lo,hi):
        if hi == lo+1:
            output[lo] = anchors[lo]
            return
        merged = covered_union(anchors[lo],anchors[hi])
        if merged is not None:
            probe = novelty(lo,hi,merged)
            if probe is not None:
                anchors[probe] = find(frames[probe])
            confirmed = merged if probe is None else covered_union(merged,anchors[probe])
            if confirmed is not None:
                output[lo:hi] = [confirmed]*(hi-lo)
                return
        else:
            probe = (lo+hi)//2
            anchors[probe] = find(frames[probe])
        fill(lo,probe)
        fill(probe,hi)

    fill(0,last)
    return output,anchors[last]
