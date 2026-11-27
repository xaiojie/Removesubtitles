"""Estimate uniform translucent subtitle panels from the input and recover the visible background."""
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np


def boundary_peaks(images, vertical):
    data = images.astype(np.int16)
    if vertical:
        data = data.transpose(0,2,1,3)
    width = data.shape[2]
    strip = data[:,:,width//6:5*width//6]
    if strip.shape[1] < 12 or not strip.shape[2]:
        return []
    delta = strip[:,1:]-strip[:,:-1]
    strength = np.percentile(abs(np.median(delta,axis=2)).max(axis=-1),65,axis=0)
    selected = []
    for index in np.argsort(strength)[::-1]:
        if strength[index] < 6:
            break
        edge = int(index+1)
        if all(abs(edge-other)>5 for other in selected):
            selected.append(edge)
            if len(selected) == 6:
                break
    return sorted(selected)


def border_pairs(images, box, step=2):
    left,top,right,bottom = box
    height,width = images.shape[1:3]
    if min(left,top,width-1-right,height-1-bottom) < step:
        return None
    xs = np.arange(left+3,right-2,3)
    ys = np.arange(top+3,bottom-2,2)
    if not xs.size or not ys.size:
        return None
    outside,inside,sides = [],[],[]
    for side,(y,x,dy,dx) in enumerate((
        (np.full_like(xs,top),xs,1,0), (np.full_like(xs,bottom),xs,-1,0),
        (ys,np.full_like(ys,left),0,1), (ys,np.full_like(ys,right),0,-1),
    )):
        src = images[:,y-step*dy,x-step*dx].reshape(-1,3)
        dst = images[:,y+step*dy,x+step*dx].reshape(-1,3)
        outside.append(src)
        inside.append(dst)
        sides.append(np.full(len(src),side))
    return np.vstack(outside).astype(np.float32),np.vstack(inside).astype(np.float32),np.concatenate(sides)


def fit_panel(images,box):
    pairs = border_pairs(images,box)
    if pairs is None:
        return None
    src,dst,sides = pairs
    weight = np.ones((len(src),1),np.float32)
    for _ in range(4):
        src_mean = (src*weight).sum(axis=0)/weight.sum()
        dst_mean = (dst*weight).sum(axis=0)/weight.sum()
        x,y = src-src_mean,dst-dst_mean
        # Both sides contain spatial sampling and compression error. Ordinary regression
        # attenuates the transmission and over-amplifies texture when the transform is inverted.
        xx,yy,xy=(x*x*weight).sum(),(y*y*weight).sum(),(x*y*weight).sum()
        if xy <= 0 or xy**2 < .35*xx*yy:
            return None
        transmission = float(np.sqrt(yy/max(1,xx)))
        offset = dst_mean-transmission*src_mean
        error = abs(dst-(src*transmission+offset)).mean(axis=1)
        weight = np.minimum(1,max(3.,1.5*np.median(error))/np.maximum(.01,error[:,None]))**2
    if not .16 <= transmission <= .84 or np.any(offset < -12) or np.any(offset>255*(1-transmission)+12):
        return None
    control = abs(dst-src).mean(axis=1)
    lit = src.max(axis=1)>15
    if lit.mean()<.4 or np.median(error[lit])>10:
        return None
    improved = [np.median(error[lit&(sides==i)])+2 < .62*np.median(control[lit&(sides==i)])
                for i in range(4) if (lit&(sides==i)).any()]
    gain = float(np.median(control[lit])-np.median(error[lit]))
    if sum(improved)<3 or gain<7:
        return None
    return gain*sum(improved),transmission,offset


@dataclass
class Panel:
    calibration_tolerance = 1
    window: tuple
    size: tuple
    box: tuple
    transmission: float
    offset: np.ndarray
    native_box: tuple = None

    def estimated_bounds(self):
        wx, wy, wr, wb = self.window
        sx, sy = (wr-wx)/self.size[0], (wb-wy)/self.size[1]
        left, top, right, bottom = self.box
        return (wx+round(sx*left), wy+round(sy*top), wx+round(sx*right), wy+round(sy*bottom))

    def measure_edges(self, frame):
        """Find the transition from outside to inside using the fitted layer color at native resolution."""
        bounds = self.estimated_bounds()
        wx, wy, wr, wb = self.window
        radii = [int(np.ceil((wr-wx)/self.size[0]))+2, int(np.ceil((wb-wy)/self.size[1]))+2]
        observations = []
        for side, center in enumerate(bounds):
            vertical = side in (0, 2)
            image = frame.transpose(1, 0, 2) if vertical else frame
            lo, hi = bounds[1::2] if vertical else bounds[::2]
            margin = max(2, (hi-lo)//12)
            radius = radii[0 if vertical else 1]
            first, last = max(1, center-radius), min(image.shape[0]-1, center+radius)
            if last < first or hi-lo <= 2*margin:
                observations.append(None)
                continue
            step = max(1, (hi-lo)//160)
            strip = image[first-1:last+1, lo+margin:hi-margin:step].astype(np.float32)
            outside, inside = (strip[:-1], strip[1:]) if side < 2 else (strip[1:], strip[:-1])
            contrast = np.abs(inside-outside).mean(axis=2)
            error = np.abs(inside-(self.transmission*outside+self.offset)).mean(axis=2)
            gain = np.median(contrast-error, axis=1)
            index = int(np.argmax(gain))
            observations.append(first+index if gain[index] >= 5 else None)
        return observations

    def calibrate(self, observations):
        selected = []
        for edge in zip(*observations):
            values = [value for value in edge if value is not None]
            if len(values) < 2:
                return False
            middle = int(np.median(values))
            if sum(abs(value-middle) <= self.calibration_tolerance for value in values) < max(2, .7*len(values)):
                return False
            selected.append(middle)
        if len(selected) != 4 or selected[2] <= selected[0] or selected[3] <= selected[1]:
            return False
        self.native_box = tuple(selected)
        return True

    def restore(self,frame,roi,boxes):
        """Return the panel's ink mask and corrected pixels only when the layer is visible."""
        wx,wy,wr,wb = self.window
        small = cv2.resize(frame[wy:wb,wx:wr],self.size,interpolation=cv2.INTER_AREA)
        outside,inside,_ = border_pairs(small[None],self.box)
        control = abs(inside-outside).mean(axis=1)
        error = abs(inside-(self.transmission*outside+self.offset)).mean(axis=1)
        if np.median(error)+3 >= .68*np.median(control) or np.median(error)>15:
            return None
        l,t,r,b = self.native_box or self.estimated_bounds()
        x,y,rr,bb = roi
        left,top = max(x,l),max(y,t)
        right,bottom = min(rr,r),min(bb,b)
        if right<=left or bottom<=top:
            return None
        ink = np.zeros((bb-y,rr-x),np.uint8)
        # A pixel brighter than even white behind this layer cannot be the original background.
        visible = frame[top:bottom,left:right].astype(np.float32)
        limit = self.transmission*255+self.offset
        foreign = ((visible-limit).max(axis=2)>10) | ((self.offset-visible).max(axis=2)>10)
        foreground = np.zeros_like(ink)
        foreground[top-y:bottom-y,left-x:right-x] = foreign.astype(np.uint8)*255
        accepted = 0
        for bx,by,br,bt in boxes:
            if not (left-x <= (bx+br)/2 < right-x and top-y <= (by+bt)/2 < bottom-y):
                continue
            region = foreground[by:bt,bx:br]
            if region.size and cv2.countNonZero(region) >= max(6,.008*region.size):
                ink[by:bt,bx:br] |= region
                accepted += 1
        if not accepted:
            return None
        # Include anti-aliased edges. Use the untouched input to determine the mask.
        ink = cv2.dilate(ink,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        restored = np.clip((visible-self.offset)/self.transmission,0,255).astype(np.uint8)
        return (left,top,right,bottom),restored,ink


def learn_panel(thumbnails,window):
    images = np.asarray(thumbnails)
    if len(images)<5:
        return None
    height,width = images.shape[1:3]
    best = 0
    winner = None
    xs,ys = boundary_peaks(images,True),boundary_peaks(images,False)
    for left,right in combinations(xs,2):
        if right-left < .48*width:
            continue
        for top,bottom in combinations(ys,2):
            if bottom-top < .14*height or right-left < 1.7*(bottom-top):
                continue
            box = (left,top,right,bottom)
            fit = fit_panel(images,box)
            if fit is not None and fit[0]>best:
                best,transmission,offset = fit
                winner = Panel(window,(width,height),box,transmission,offset)
    if winner is None:
        from lite_gradient import learn_gradient_panel
        winner = learn_gradient_panel(images, window, xs, ys)
    return winner
