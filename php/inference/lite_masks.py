"""Recover the full bounds of a dark subtitle card, including edges touching dark scenery."""
import cv2
import numpy as np


def intervals(values):
    return np.flatnonzero(np.diff(np.r_[False, values.astype(bool), False])).reshape(-1, 2)


def edge_rows(edges):
    counts = np.count_nonzero(edges, axis=1)
    threshold = max(18, round(edges.shape[1]*.065))
    return [int(a + np.argmax(counts[a:b])) for a, b in intervals(counts >= threshold)]


def has_multiline_card(crop, boxes, bounds, color):
    """Verify two text rows, a flat fill and four visible transitions at the proposed edges."""
    x, y, w, h = bounds
    margin = max(3, min(8, round(.04*h)))
    if x < margin or y < margin or x+w+margin > crop.shape[1] or y+h+margin > crop.shape[0]:
        return False
    centers = [(top+bottom)/2 for left,top,right,bottom in boxes
               if y < (top+bottom)/2 < y+h and min(x+w, right)-max(x, left) >= .75*(right-left)]
    if len(centers) < 2 or max(centers)-min(centers) < .25*h:
        return False
    patch = crop[y-margin:y+h+margin, x-margin:x+w+margin]
    fill = cv2.inRange(patch, tuple(np.maximum(color-16, 0).tolist()),
                       tuple(np.minimum(color+16, 255).tolist())) != 0
    body = fill[margin:-margin, margin:-margin]
    if body[::3, ::3].mean() < .57:
        return False
    # A small decoration can cross one edge. A wider panel or a uniform wall lacks a transition.
    for scan in (fill, fill.T):
        core = scan[margin:-margin, margin:-margin:3]
        if (core[2:margin+2].mean() < .43 or core[-margin-2:-2].mean() < .43
                or scan[:margin, margin:-margin:3].mean() > .25
                or scan[-margin:, margin:-margin:3].mean() > .25):
            return False
    return True


class OpaqueCardTrack:
    """Keep a confirmed multiline card separate from same-color objects touching its edges."""

    def __init__(self):
        self.last = -2
        self.contour = None

    def update(self, crop, boxes, detected, index):
        contours = (cv2.findContours(detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if detected is not None else ())
        if (self.contour is not None and index == self.last+1
                and has_multiline_card(crop, boxes, self.bounds, self.color)):
            self.last = index
            x, y, w, h = self.bounds
            mask = np.zeros(crop.shape[:2], np.uint8) if detected is None else detected
            for contour in contours:
                if cv2.pointPolygonTest(contour, (x+w/2, y+h/2), False) >= 0:
                    cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)
            cv2.drawContours(mask, [self.contour], -1, 255, cv2.FILLED)
            return mask

        self.contour = None
        for contour in contours:
            x, y, w, h = bounds = cv2.boundingRect(contour)
            if w < max(.5*crop.shape[1], 3*h) or h < 16 or cv2.contourArea(contour) < .90*w*h:
                continue
            color = np.median(crop[y:y+h:4, x:x+w:4].reshape(-1, 3), axis=0).astype(int)
            if color.max()-color.min() >= 24 and has_multiline_card(crop, boxes, bounds, color):
                self.bounds, self.color, self.contour, self.last = bounds, color, contour, index
                break
        return detected


def black_caption_cards(crop, boxes, previous_band=None):
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
                # A short single-word card has the same top and bottom edges as the preceding line.
                same_band = previous_band is not None and all(
                    abs(a-b) <= max(3, .17*depth)
                    for a,b in zip((top,bottom), (previous_band[0]*scale,previous_band[1]*scale)))
                if span < max(15 if same_band else 25, (.65 if same_band else 1.8)*depth):
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


class DarkCardTrack:
    """Remember confirmed horizontal card edges across short gaps between caption lines."""

    def __init__(self):
        self.band = None
        self.missing_frames = 0

    def update(self, crop, boxes):
        card = black_caption_cards(crop, boxes, self.band)
        if card is not None:
            x,y,w,h = cv2.boundingRect(card)
            # The geometric detector pads each edge by a tenth of the unpadded height.
            inset = max(3, round(h/12))
            if w >= h*.65 and abs(x+w/2-crop.shape[1]/2) < crop.shape[1]*.24:
                self.band = (y+inset, y+h-inset-1)
                self.missing_frames = 0
        else:
            self.missing_frames += 1
            if self.missing_frames > 48:
                self.band = None
        if self.band is None:
            return boxes, card
        top, bottom = self.band
        mid = (top+bottom)/2
        # Small off-line printing on objects stays visible between statements.
        filtered = [box for box in boxes if box[2]-box[0] >= .24*crop.shape[1]
                    or abs((box[1]+box[3])/2-mid) <= .30*(bottom-top)]
        return filtered, card


def anchored_brush_card(crop, boxes):
    """Find broad, opaque colored bands behind OCR lines, including soft brush edges."""
    height, width = crop.shape[:2]
    if width < height*2 or not boxes:
        return None
    lines = [(x,y,r,b) for x,y,r,b in boxes if r-x >= max(.34*width, 3*(b-y))]
    if not lines:
        return None
    scale = min(1., 384/width)
    small = cv2.resize(crop, (round(width*scale), max(1,round(height*scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h,w = hsv.shape[:2]
    result = np.zeros((height,width), np.uint8)
    for x,y,r,b in lines:
        x,y,r,b = max(0,int(x*scale)), max(0,int(y*scale)), min(w,int(np.ceil(r*scale))), min(h,int(np.ceil(b*scale)))
        depth = b-y
        if depth < 5:
            continue
        near = hsv[max(0,y-depth//3):min(h,b+depth//2),x:r]
        color = (near[...,1]>=42) & (near[...,2]>=100)
        hues = near[...,0][color]
        if len(hues) < .20*near.shape[0]*near.shape[1]:
            continue
        counts = np.bincount(hues//6, minlength=30)
        peak = int(np.argmax(counts))
        if counts[[(peak-1)%30,peak,(peak+1)%30]].sum() < .24*near.shape[0]*near.shape[1]:
            continue
        samples = near[color & (near[...,0]//6==peak)]
        hue,sat,val = np.median(samples,axis=0)
        delta = abs(hsv[...,0].astype(np.int16)-int(hue))
        close = ((np.minimum(delta,180-delta)<=6) &
                 (abs(hsv[...,1].astype(np.int16)-int(sat))<=32) &
                 (abs(hsv[...,2].astype(np.int16)-int(val))<=35)).astype(np.uint8)
        merged = cv2.morphologyEx(close,cv2.MORPH_CLOSE,np.ones((3,5),np.uint8))
        contours,_ = cv2.findContours(merged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            l,t,cw,ch = cv2.boundingRect(contour)
            if cw < max(.74*(r-x),3*ch) or not .31*depth <= ch <= 1.8*depth:
                continue
            if min(r,l+cw)-max(x,l) < .68*(r-x) or min(b,t+ch)-max(y,t) < .17*depth:
                continue
            if (t <= 0 and t+ch >= h) or cv2.contourArea(contour) < .39*cw*ch:
                continue
            patch = close[t:t+ch,l:l+cw]
            # A plate has a long, almost uniform horizontal edge outside the lettering.
            if patch.mean()<.40 or np.count_nonzero(patch.mean(axis=1)>.82)<max(2,.09*ch):
                continue
            outer = [close[a:z,l:l+cw] for a,z in ((max(0,t-3),t),(t+ch,min(h,t+ch+3))) if z>a]
            if not outer or min(strip.mean() for strip in outer)>.32:
                continue
            pad = max(3,round(ch*.08/scale))
            xl,yt = max(0,int(l/scale)-pad),max(0,int(t/scale)-pad)
            xr,yb = min(width,int(np.ceil((l+cw)/scale))+pad),min(height,int(np.ceil((t+ch)/scale))+pad)
            result[yt:yb,xl:xr] = 255
    return result if result.any() else None


def word_highlights(crop, boxes):
    """Mask the entire colored highlight behind a word, including the padding outside its letters."""
    if not boxes:
        return None
    height,width = crop.shape[:2]
    scale = min(1.,512/width)
    small = cv2.resize(crop,(round(width*scale),max(1,round(height*scale))),interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    saturated = (hsv[...,1]>=80) & (hsv[...,2]>=90)
    result = None
    for x,y,r,b in boxes:
        l,t = int(x*scale),int(y*scale)
        rr,bb = min(hsv.shape[1],int(np.ceil(r*scale))),min(hsv.shape[0],int(np.ceil(b*scale)))
        patch = saturated[t:bb,l:rr]
        if not patch.size or cv2.countNonZero(patch.astype(np.uint8)) < .05*patch.size:
            continue
        hues = hsv[t:bb,l:rr,0][patch]
        hist = np.bincount(hues//8,minlength=23)
        for bin_index in np.argsort(hist)[-2:]:
            if hist[bin_index] < max(12,.04*patch.size):
                continue
            hue = int(np.median(hues[hues//8==bin_index]))
            diff = abs(hsv[...,0].astype(np.int16)-hue)
            core = (saturated & (np.minimum(diff,180-diff)<=8)).astype(np.uint8)
            contours,_ = cv2.findContours(core,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cx,cy,w,h = cv2.boundingRect(contour)
                if (h < max(7,.30*(bb-t)) or h > min(.80*height*scale,1.40*(bb-t))
                        or w < .85*h or w > max(rr-l,2*h)*1.15 or cy<1 or cy+h>=core.shape[0]):
                    continue
                overlap = max(0,min(rr,cx+w)-max(l,cx))*max(0,min(bb,cy+h)-max(t,cy))
                if overlap < .58*w*h or cv2.contourArea(contour)<.70*w*h:
                    continue
                area = core[cy:cy+h,cx:cx+w]
                if np.max(area.mean(axis=1)) < .78:
                    continue
                patch = small[cy:cy+h, cx:cx+w]
                light = cv2.inRange(patch, (150, 150, 150), (255, 255, 255))
                dark = cv2.inRange(patch, (0, 0, 0), (85, 85, 85))
                ink = ((light != 0) & (hsv[cy:cy+h, cx:cx+w, 1] < 75)) | (dark != 0)
                if np.mean(ink) < .045:
                    continue
                if result is None:
                    result = np.zeros((height,width),np.uint8)
                pad = max(2,round(.10*h/scale))
                left,top = max(0,int(cx/scale)-pad),max(0,int(cy/scale)-pad)
                right,bottom = min(width,int(np.ceil((cx+w)/scale))+pad),min(height,int(np.ceil((cy+h)/scale))+pad)
                result[top:bottom,left:right] = 255
    return result


def chip_examples(crop,boxes):
    mask=word_highlights(crop,boxes)
    if mask is None:
        return []
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    examples=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        hsv=cv2.cvtColor(crop[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
        colors=hsv[(hsv[...,1]>75)&(hsv[...,2]>90)]
        if len(colors)<.24*w*h:
            continue
        hist=np.bincount(colors[:,0]//6,minlength=30)
        peak=int(np.argmax(hist))
        hue,sat,val=np.median(colors[colors[:,0]//6==peak],axis=0)
        examples.append((float(hue),float(sat),float(val),y+h/2,h))
    return examples


def repeated_chips(frames):
    candidates=[(i,*chip) for i,examples in enumerate(frames) for chip in examples]
    winners=[]
    for _,hue,sat,val,center,height in candidates:
        matched=[v for v in candidates if min(abs(v[1]-hue),180-abs(v[1]-hue))<=8 and
                 abs(v[4]-center)<.45*height and .60*height<v[5]<1.55*height]
        support=len({v[0] for v in matched})
        if support>=max(3,.28*len(frames)):
            typical=np.median([v[1:] for v in matched],axis=0)
            typical[0]=(hue+np.median([(v[1]-hue+90)%180-90 for v in matched]))%180
            winners.append((support,tuple(typical)))
    return max(winners)[1] if winners else None


def word_shades(value, supported):
    """Separate two visibly different shades of the same hue inside a word's search region."""
    samples = value[supported != 0]
    if samples.size < 2:
        return ()
    threshold, _ = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = samples <= threshold
    if not dark.any() or dark.all() or samples[~dark].mean() - samples[dark].mean() < 20:
        return ()
    return (supported & (value <= threshold), supported & (value > threshold))


def locate_chip(crop,template, *, split_shades=False):
    hue,sat,val,center,height=template
    h,w=crop.shape[:2]
    t,b=max(0,int(center-height)),min(h,int(np.ceil(center+height)))
    hsv=cv2.cvtColor(crop[t:b],cv2.COLOR_BGR2HSV)
    delta=abs(hsv[...,0].astype(np.int16)-hue)
    supported=((np.minimum(delta,180-delta)<=8)&(hsv[...,1]>max(50,sat-72))&
               (hsv[...,2]>max(66,val-75))).astype(np.uint8)
    layers = word_shades(hsv[..., 2], supported) if split_shades else (supported,)
    result=None
    for layer in layers:
        contours,_=cv2.findContours(layer,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,cw,ch=cv2.boundingRect(contour)
            if (not .46*height<=ch<=1.52*height or cw<.65*ch or
                    abs(y+t+ch/2-center)>.36*height or cv2.contourArea(contour)<.65*cw*ch):
                continue
            # A surrounding shade may enclose the word's tile as a hole.
            if split_shades and (x == 0 or x+cw == w):
                continue
            panel=crop[t+y:t+y+ch,x:x+cw]
            ink=(panel.min(axis=2)>165)|(panel.max(axis=2)<85)
            if ink.mean()<.037 or (layer[y:y+ch,x:x+cw].mean(axis=1)>.76).sum()<2:
                continue
            if result is None:
                result=np.zeros((h,w),np.uint8)
            pad=max(3,round(.12*ch))
            result[max(0,t+y-pad):min(h,t+y+ch+pad),max(0,x-pad):min(w,x+cw+pad)]=255
    return result


def scaled_word_chip(crop, boxes, template):
    """Confirm a learned highlight behind a word at its current size and position."""
    for split_shades in (False, True):
        mask = word_chip_at_boxes(crop, boxes, template, split_shades=split_shades)
        if mask is not None:
            return mask
    return None


def word_chip_at_boxes(crop, boxes, template, *, split_shades):
    height, width = crop.shape[:2]
    result = None
    for x, y, right, bottom in boxes:
        depth = bottom-y
        margin = max(4, depth//2)
        left, top = max(0, x-margin), max(0, y-margin)
        patch = crop[top:min(height, bottom+margin), left:min(width, right+margin)]
        current = (*template[:3], (y+bottom)/2-top, depth)
        mask = locate_chip(patch, current, split_shades=split_shades)
        if mask is None:
            continue
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            cx, cy, w, h = cv2.boundingRect(contour)
            # A colored surface continuing on either side is insufficient evidence of a word tile.
            if cx == 0 or cx+w == patch.shape[1]:
                continue
            l, t = left+cx, top+cy
            overlap = max(0, min(l+w, right)-max(l, x)) * max(0, min(t+h, bottom)-max(t, y))
            if overlap < .5*max(w*h, (right-x)*depth):
                continue
            if result is None:
                result = np.zeros((height, width), np.uint8)
            result[t:t+h, l:l+w] = 255
    return result
