"""Learn repeated subtitle rows and stable panel boundaries from a bounded set of input frames."""
from dataclasses import dataclass
import cv2
import numpy as np
from video_capture import open_video_capture
from lite_layer import learn_panel
from lite_masks import chip_examples, repeated_chips


def local_boxes(boxes,roi):
    x,y,r,b=roi
    return [(max(l,x)-x,max(t,y)-y,min(rr,r)-x,min(bb,b)-y) for l,t,rr,bb in boxes
            if max(l,x)<min(rr,r) and max(t,y)<min(bb,b)]


def find_rows(samples,width,height):
    if width<1.8*height or len(samples)<5:
        return []
    observations = [(i,(y+b)/2,b-y) for i,boxes in enumerate(samples) for x,y,r,b in boxes
                    if r-x>=.18*width and r-x>=1.8*(b-y) and abs((r+x)/2-width/2)<.31*width]
    bands = []
    for _,center,size in observations:
        related=[(i,c,h) for i,c,h in observations if abs(c-center)<=max(.36*size,.033*height)
                 and .66*size<=h<=1.52*size]
        hits=len({i for i,_,_ in related})
        if hits>=max(4,.38*len(samples)):
            bands.append((hits,float(np.median([c for _,c,_ in related])),
                          float(np.median([h for _,_,h in related]))))
    chosen=[]
    for _,center,size in sorted(bands,reverse=True):
        if not any(abs(center-other)<max(.48*size,.045*height) for other,_ in chosen):
            chosen.append((center,size))
    return sorted(chosen)


@dataclass
class BandProfile:
    width: int
    height: int
    rows: list
    panel: object
    chip: object = None

    def dialogue_rows(self, boxes):
        """Recognize multiple centered dialogue rows moving together, including one-word lines."""
        if len(self.rows) < 2 or len(boxes) < 2:
            return []
        height = float(np.median([h for _, h in self.rows]))
        groups = []
        for box in sorted(boxes, key=lambda rect: rect[1]+rect[3]):
            x, y, right, bottom = box
            if not .59*height <= bottom-y <= 1.52*height:
                continue
            center = (y+bottom)/2
            if groups and abs(groups[-1][0]-center) < .28*height:
                groups[-1][1].append(box)
            else:
                groups.append((center, [box]))
        centered = []
        for center, line in groups:
            left = min(box[0] for box in line)
            right = max(box[2] for box in line)
            if abs((left+right)/2-self.width/2) < .16*self.width:
                centered.append((center, right-left, line))
        best, best_score = [], (0, float('-inf'))
        for center, _, _ in centered:
            for expected, _ in self.rows:
                shift = center-expected
                if abs(shift) > .75*height:
                    continue
                matches = []
                used = set()
                for row, _ in self.rows:
                    choices = [(abs(c-row-shift), i) for i,(c,_,_) in enumerate(centered) if i not in used]
                    if choices:
                        distance, i = min(choices)
                        if distance <= .23*height:
                            used.add(i)
                            matches.append(centered[i])
                score = (len(matches), -abs(shift))
                if len(matches) >= 2 and max(width for _,width,_ in matches) >= .35*self.width and score > best_score:
                    best = [box for _,_,line in matches for box in line]
                    best_score = score
        return best

    def select(self,boxes):
        if not self.rows or not boxes:
            return boxes
        def fits(box):
            _,y,_,b=box
            return any(abs((y+b)/2-center)<=.48*size and .48*size<=b-y<=1.7*size
                       for center,size in self.rows)
        confirmed=[box for box in boxes if fits(box)]
        # A new central multiline card can legitimately move outside the usual dialogue band.
        wide=[box for box in boxes if box[2]-box[0]>=.40*self.width and
              abs((box[0]+box[2])/2-self.width/2)<.22*self.width and
              box[1]>1 and box[3]<self.height-1]
        if len(wide)>=2:
            confirmed.extend(box for box in wide if box not in confirmed)
        elif not confirmed and wide:
            box=wide[0]
            if any(.62*h<box[3]-box[1]<1.45*h for _,h in self.rows):
                confirmed.append(box)
        return confirmed


def profile_source(path,rois,find):
    sampler=open_video_capture(path)
    grids={tuple(roi):[] for roi in rois}
    samples={key:[] for key in grids}
    chips={key:[] for key in grids}
    windows={}
    try:
        total=int(sampler.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.unique(np.linspace(0,max(0,total-1),min(13,max(1,total)),dtype=int)):
            if index and not sampler.set(cv2.CAP_PROP_POS_FRAMES,int(index)):
                break
            ok,frame=sampler.read()
            if not ok:
                continue
            boxes=find(frame)
            height,width=frame.shape[:2]
            for key,thumbs in grids.items():
                x,y,r,b=key
                samples[key].append(local_boxes(boxes,key))
                if r-x < 1.7*(b-y):
                    continue
                chips[key].append(chip_examples(frame[y:b,x:r],samples[key][-1]))
                l,t,rr,bb=max(0,x-20),max(0,y-20),min(width,r+20),min(height,b+20)
                windows[key]=(l,t,rr,bb)
                scale=min(1.,448/(rr-l))
                thumbs.append(cv2.resize(frame[t:bb,l:rr],(round((rr-l)*scale),max(1,round((bb-t)*scale))),
                                         interpolation=cv2.INTER_AREA))
    finally:
        sampler.release()
    result={}
    for key,thumbs in grids.items():
        x,y,r,b=key
        panel=learn_panel(thumbs,windows[key]) if len(thumbs)>=5 else None
        result[key]=BandProfile(r-x,b-y,find_rows(samples[key],r-x,b-y),panel,repeated_chips(chips[key]))
    panels = {key: profile.panel for key, profile in result.items() if profile.panel is not None}
    if panels:
        # One additional sparse decode for all candidate panels. Keep just four coordinates per sample.
        observations = {key: [] for key in panels}
        sampler = open_video_capture(path)
        try:
            for index in np.unique(np.linspace(0, max(0, total-1), min(5, max(1, total)), dtype=int)):
                if index and not sampler.set(cv2.CAP_PROP_POS_FRAMES, int(index)):
                    continue
                ok, frame = sampler.read()
                if ok:
                    for key, panel in panels.items():
                        observations[key].append(panel.measure_edges(frame))
        finally:
            sampler.release()
        for key, panel in panels.items():
            if not panel.calibrate(observations[key]):
                result[key].panel = None
    return result
