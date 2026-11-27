"""Spend OCR calls on caption changes inside a bounded lookahead interval."""
import cv2
import numpy as np


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
