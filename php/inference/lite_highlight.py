"""Learn moving translucent word highlights from matching top and bottom edges in the subtitle band."""
from collections import deque
from dataclasses import dataclass, field
from itertools import permutations, product

import cv2
import numpy as np

from lite_masks import intervals


def color_steps(images, first, second):
    chroma = images[...,first].astype(np.int16)-images[...,second].astype(np.int16)
    steps = np.zeros_like(chroma)
    steps[:,2:-1] = chroma[:,3:]-chroma[:,:-3]
    return steps


def common_edges(steps):
    # A highlight can move horizontally from word to word while its top and bottom stay at the same height.
    kernel = np.ones((1,9), np.uint8)
    opened = [cv2.morphologyEx((frame>9).astype(np.uint8),cv2.MORPH_OPEN,kernel) for frame in steps]
    scores = np.percentile(np.sum(opened,axis=2),65,axis=0)
    selected = []
    for y in np.argsort(scores)[::-1]:
        if scores[y] < 10:
            break
        if all(abs(y-row)>3 for row in selected):
            selected.append(int(y))
            if len(selected) == 6:
                break
    return selected


def segments(steps, top, bottom):
    entering = steps[top-1:top+2].max(0)
    leaving = -steps[bottom-1:bottom+2].min(0)
    both = ((entering>9)&(leaving>9))[None].astype(np.uint8)
    joined = cv2.morphologyEx(both,cv2.MORPH_CLOSE,np.ones((1,5),np.uint8))
    height = bottom-top
    return [(left,right) for left,right in intervals(joined[0]) if
            max(9,.34*height) <= right-left <= min(steps.shape[1]*.77,8*height)]


def border_observations(image, top, bottom, left, right):
    margin = max(2,(right-left)//12)
    xs = np.arange(left+margin,right-margin,2)
    outside = np.concatenate([image[top-2,xs],image[bottom+2,xs]]).astype(np.float32)
    inside = np.concatenate([image[top+2,xs],image[bottom-2,xs]]).astype(np.float32)
    return outside, inside


def blend_quality(outside, inside, transmission, offset):
    residual = np.abs(inside-(transmission*outside+offset)).mean(1)
    contrast = np.abs(inside-outside).mean(1)
    return ((residual < 13) & (residual+3 < .67*contrast)).mean()


def learn_blend(images, top, bottom, first, second, spans):
    pairs = [border_observations(image, top, bottom, *span) for image,span in zip(images,spans) if span]
    x, y = (np.concatenate(values) for values in zip(*pairs))
    weight = np.ones((len(x),1), np.float32)
    for _ in range(5):
        total = weight.sum()
        xmean, ymean = (x*weight).sum(0)/total, (y*weight).sum(0)/total
        dx, dy = x-xmean, y-ymean
        variance = (weight*dx*dx).sum()
        if variance < 140*total:
            return None
        transmission = float((weight*dx*dy).sum()/variance)
        offset = ymean-transmission*xmean
        error = np.abs(y-(transmission*x+offset)).mean(1)
        weight = np.minimum(1.,max(2.,1.4*np.median(error))/np.maximum(.1,error[:,None]))**2
    if (not .15 < transmission < .88 or np.min(offset) < -9 or np.max(offset) > 255*(1-transmission)+10
            or offset[first]-offset[second] < 10 or np.median(error)>10):
        return None
    count = sum(blend_quality(o,i,transmission,offset)>.65 for o,i in pairs)
    return (count, transmission, offset) if count >= max(5,.53*len(images)) else None


@dataclass
class MovingHighlight:
    window: tuple
    size: tuple
    top: int
    bottom: int
    first: int
    second: int
    transmission: float
    offset: np.ndarray
    recent_strong: deque = field(default_factory=lambda: deque(maxlen=4), init=False, repr=False)
    previous_scene: np.ndarray = field(default=None, init=False, repr=False)

    def mask(self, frame, roi, *, track=False):
        wx, wy, wr, wb = self.window
        x, y, right, bottom = roi
        image = cv2.resize(frame[wy:wb,wx:wr],self.size,interpolation=cv2.INTER_AREA)
        steps = color_steps(image[None],self.first,self.second)[0]
        sx, sy = (wr-wx)/self.size[0], (wb-wy)/self.size[1]
        height = self.bottom-self.top
        if track:
            scene = cv2.resize(image, (40, 14), interpolation=cv2.INTER_AREA)
            if self.previous_scene is not None and cv2.absdiff(scene, self.previous_scene).mean() > 27:
                self.recent_strong.clear()
            self.previous_scene = scene
        confirmed = []
        result = None
        for left,end in segments(steps,self.top,self.bottom):
            outside,inside = border_observations(image,self.top,self.bottom,left,end)
            half = len(outside)//2
            scores = [blend_quality(outside[edge], inside[edge], self.transmission, self.offset)
                      for edge in (slice(None, half), slice(half, None))]
            if min(scores) >= .38:
                confirmed.append((left, end))
            else:
                # Keep a briefly obscured border only if this frame still contains both
                # colour steps at the same word, with strong support on at least one side.
                if not track or min(scores) < .12 or max(scores) < .55:
                    continue
                if not any(max(abs(left-a), abs(end-b)) <= .14*height and
                           min(end,b)-max(left,a) >= .8*max(end-left,b-a)
                           for spans in self.recent_strong for a,b in spans):
                    continue
            # Account for rounded corners and the soft border beyond the flat parts of the two edges.
            l = max(x, wx+int(np.floor((left-.15*height-1)*sx)))
            r = min(right, wx+int(np.ceil((end+.15*height+1)*sx)))
            t = max(y, wy+int(np.floor((self.top-2)*sy)))
            b = min(bottom, wy+int(np.ceil((self.bottom+2)*sy)))
            if l<r and t<b:
                if result is None:
                    result = np.zeros((bottom-y,right-x),np.uint8)
                result[t-y:b-y,l-x:r-x] = 255
        if track:
            self.recent_strong.append(confirmed)
        return result


def learn_highlight(images, window):
    images = np.asarray(images)
    n, height, width = images.shape[:3]
    winner = None
    best = (0,0.)
    for first,second in permutations(range(3),2):
        steps = color_steps(images,first,second)
        for top,bottom in product(common_edges(steps),common_edges(-steps)):
            if min(top,height-1-bottom)<2 or not .12*height<=bottom-top<=.87*height:
                continue
            spans = [max(segments(frame,top,bottom),key=lambda x:x[1]-x[0],default=None) for frame in steps]
            valid = [span for span in spans if span]
            if len(valid)<max(5,.53*n) or np.ptp([(l+r)/2 for l,r in valid])<.17*width:
                continue
            blend = learn_blend(images,top,bottom,first,second,spans)
            if blend is not None:
                count, transmission, offset = blend
                score = (count,len(valid)+min(1,.4*np.median([r-l for l,r in valid])/(bottom-top)))
                if score>best:
                    winner = MovingHighlight(window,(width,height),top,bottom,first,second,transmission,offset)
                    best = score
    return winner
