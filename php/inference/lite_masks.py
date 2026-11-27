"""Recover the full bounds of a dark subtitle card, including edges touching dark scenery."""
import cv2
import numpy as np


def intervals(values):
    return np.flatnonzero(np.diff(np.r_[False, values.astype(bool), False])).reshape(-1, 2)


def edge_rows(edges):
    counts = np.count_nonzero(edges, axis=1)
    threshold = max(18, round(edges.shape[1]*.065))
    return [int(a + np.argmax(counts[a:b])) for a, b in intervals(counts >= threshold)]


def black_caption_cards(crop, boxes):
    height, width = crop.shape[:2]
    if min(width, height) < 16:
        return None
    scale = min(1., 640/width)
    small = cv2.resize(crop, (round(width*scale), max(1, round(height*scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else crop
    h, w = small.shape[:2]
    black = cv2.inRange(small, (0, 0, 0), (12, 12, 12))
    if cv2.countNonZero(black) < max(40, .015*h*w):
        return None
    # Long horizontal transitions remain visible when, for example, a black mic joins the card.
    delta = max(2, min(4, round(h*.014)))
    enter, leave = np.zeros_like(black), np.zeros_like(black)
    enter[delta:] = cv2.subtract(black[delta:], black[:-delta])
    leave[:-delta] = cv2.subtract(black[:-delta], black[delta:])
    kernel = np.ones((1, max(11, min(31, round(w*.04)))), np.uint8)
    enter = cv2.morphologyEx(enter, cv2.MORPH_OPEN, kernel)
    leave = cv2.morphologyEx(leave, cv2.MORPH_OPEN, kernel)
    tops, bottoms = edge_rows(enter), edge_rows(leave)
    result = None
    for top in tops:
        for bottom in bottoms:
            depth = bottom-top+1
            if not max(9, .04*h) <= depth <= .91*h:
                continue
            rim = max(2, round(.08*depth))
            side = ((black[top:top+rim].mean(axis=0)>150) &
                    (black[bottom-rim+1:bottom+1].mean(axis=0)>150)).astype(np.uint8)
            side = cv2.morphologyEx(side[None], cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))[0]
            for left, right in intervals(side):
                span = right-left
                if span < max(25, 1.8*depth):
                    continue
                if min(np.count_nonzero(enter[top,left:right]), np.count_nonzero(leave[bottom,left:right])) < .43*span:
                    continue
                interior = black[top:bottom+1,left:right]
                if not .50 <= cv2.countNonZero(interior)/interior.size <= .985:
                    continue
                # Require either an OCR word centred inside the card or a row of letter-sized contrast islands.
                anchored = any(left <= (x+r)*.5*scale < right and top < (y+b)*.5*scale < bottom
                               for x,y,r,b in boxes)
                if not anchored:
                    bright = (small[top:bottom+1,left:right].min(axis=2)>60).astype(np.uint8)
                    _, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
                    bases = [int(y+gh) for _,y,gw,gh,area in stats[1:] if .16*depth<=gh<=.83*depth
                             and .06*gh<=gw<=3.4*gh and area>gh]
                    if not any(sum(abs(v-base)<max(3,.12*depth) for v in bases)>=3 for base in bases):
                        continue
                if result is None:
                    result = np.zeros((height,width),np.uint8)
                pad = max(3, int(np.ceil(.10*depth/scale)))
                x0,y0 = max(0,int(left/scale)-pad),max(0,int(top/scale)-pad)
                x1,y1 = min(width,int(np.ceil(right/scale))+pad),min(height,int(np.ceil((bottom+1)/scale))+pad)
                result[y0:y1,x0:x1]=255
    return result
