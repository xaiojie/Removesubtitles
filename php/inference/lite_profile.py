"""Learn repeated subtitle rows and stable panel boundaries from a bounded set of input frames."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from video_capture import open_video_capture
from lite_layer import learn_panel
from lite_highlight import learn_highlight
from lite_masks import OpaqueCardTrack, chip_examples, repeated_chips
from lite_tracking import SideLabelTrack
from subtitle_compositing import colored_caption_cards


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


def printed_row_style(frame, box):
    """Measure the height and fill of several high-contrast glyphs inside one OCR row."""
    x, y, right, bottom = box
    patch = frame[y:bottom, x:right]
    height, width = patch.shape[:2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    radius = max(2, min(9, round(height * .12)))
    dark = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
    white = cv2.inRange(patch, (169, 169, 169), (255, 255, 255))
    yellow = cv2.inRange(patch, (0, 148, 174), (146, 255, 255))
    ink = (white | yellow) & cv2.inRange(cv2.subtract(gray, dark), 53, 255)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    xx, yy, ww, hh, area = stats.T
    letters = ((xx > 0) & (xx+ww < width) & (yy > 0) & (yy+hh < height)
               & (hh >= .22*height) & (hh < .90*height) & (ww < 2.3*hh) & (area >= hh))
    if np.count_nonzero(letters) < 3:
        return None
    # An ascender still carries the font size in a mostly lowercase word.
    glyph_height = float(np.percentile(hh[letters], 75))
    return glyph_height, np.median(patch[letters[labels]], axis=0)


def word_lines(boxes):
    """Group nearby OCR words on a common baseline, retaining their individual boxes for masking."""
    lines = []
    for box in sorted(boxes):
        x, y, right, bottom = box
        for i, (bounds, words) in enumerate(lines):
            left, top, end, low = bounds
            depth = min(low-top, bottom-y)
            if (max(low-top, bottom-y) > 1.52*depth or
                    abs(top+low-y-bottom) > .56*depth or x-end > depth):
                continue
            words.append(box)
            lines[i] = ((left, min(top, y), max(end, right), max(low, bottom)), words)
            break
        else:
            lines.append((box, [box]))
    return lines


def next_subtitle_row(above, below, width):
    x, y, r, b = above
    left, top, right, bottom = below
    depth, other_depth = b-y, bottom-top
    spacing = (top+bottom-y-b) / 2
    return (.73*depth <= other_depth <= 1.37*depth
            and .66*min(depth, other_depth) < spacing < 1.45*max(depth, other_depth)
            and abs(left+right-x-r) <= .22*width)


def shared_caption_card(frame, selected):
    """Find an opaque multi-line backing that contains every established word box."""
    mask = colored_caption_cards(frame)
    if mask is None:
        return None
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, width, height = cv2.boundingRect(contour)
        if width < max(.55*frame.shape[1], 3*height) or cv2.contourArea(contour) < .8*width*height:
            continue
        right, bottom = x+width, y+height
        for left, top, end, low in selected:
            overlap = max(0, min(end, right)-max(left, x)) * max(0, min(low, bottom)-max(top, y))
            if height < 1.25*(low-top) or overlap < .70*(end-left)*(low-top):
                break
        else:
            return x, y, right, bottom
    return None


@dataclass
class BandProfile:
    width: int
    height: int
    rows: list
    panel: object
    chip: object = None
    highlight: object = None
    opaque_card: OpaqueCardTrack = field(default_factory=OpaqueCardTrack, repr=False, compare=False)
    side_labels: SideLabelTrack = field(default_factory=SideLabelTrack, repr=False, compare=False)

    def wrapped_dialogue(self, frame, boxes, selected, *, group_words=False):
        """Keep an adjacent row matching the font of an established, centred subtitle."""
        if len(self.rows) != 1 or not selected or len(selected) == len(boxes) or self.panel or self.chip or self.highlight:
            return []
        others = [box for box in boxes if box not in selected]
        if group_words:
            anchors = word_lines(selected)
            if len(anchors) != 1:
                return []
            anchor = anchors[0][0]
            candidates = word_lines(others)
        else:
            if len(selected) != 1:
                return []
            anchor = selected[0]
            candidates = [(box, [box]) for box in others]
        x, y, right, bottom = anchor
        height = bottom-y
        center = (x+right)/2
        if height < 10 or abs(center-self.width/2) > .12*self.width:
            return []
        expected, size = self.rows[0]
        if abs((y+bottom)/2-expected) > .45*size:
            return []
        anchor_style = None
        for other, words in candidates:
            left, top, end, low = other
            depth = low-top
            distance = abs(top+low-y-bottom)/2
            if not .73*height <= depth <= 1.37*height or not .66*height < distance < 1.43*height:
                continue
            # A wide user selection can contain two fairly short, centred rows.
            minimum_span = min(.42*self.width, 4*height) if group_words else .42*self.width
            if min(right-x,end-left) < 1.65*min(height,depth) or max(right-x,end-left) < minimum_span:
                continue
            if abs((left+end)/2-center) > .11*self.width or top <= 1 or low >= self.height-1:
                continue
            if anchor_style is None:
                anchor_style = printed_row_style(frame, anchor)
                if anchor_style is None:
                    break
            style = printed_row_style(frame, other)
            if (style is not None and .73*anchor_style[0] <= style[0] <= 1.38*anchor_style[0]
                    and np.abs(style[1]-anchor_style[1]).max() < 31):
                return [*selected, *words]
        return []

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

    def scaled_dialogue(self, frame, boxes, selected):
        """Follow matching adjacent print when an established multiline subtitle changes size or wraps again."""
        if len(self.rows) < 2 or len(boxes) < 2 or len(selected) == len(boxes) or self.panel or self.highlight:
            return []
        if self.chip is not None and selected:
            return []
        lines = [(bounds, words) for bounds, words in word_lines(boxes)
                 if bounds[2]-bounds[0] >= 1.65*(bounds[3]-bounds[1])
                 and abs(bounds[0]+bounds[2]-self.width) < .32*self.width]
        lines.sort(key=lambda line: line[0][1]+line[0][3])
        # Check actual glyphs only for geometric neighbours that could extend the mask.
        styles = {}
        groups = []
        for index, (bounds, _) in enumerate(lines):
            if index and next_subtitle_row(lines[index-1][0], bounds, self.width):
                for i in (index-1, index):
                    if i not in styles:
                        styles[i] = printed_row_style(frame, lines[i][0])
                previous, current = styles[index-1], styles[index]
                if (previous is not None and current is not None
                        and .73*previous[0] <= current[0] <= 1.38*previous[0]
                        and np.abs(previous[1]-current[1]).max() < 31):
                    groups[-1].append(index)
                    continue
            groups.append([index])
        kept = set(selected)
        for group in groups:
            if len(group) < 2:
                continue
            words = {box for i in group for box in lines[i][1]}
            if words <= kept:
                continue
            bounds = [lines[i][0] for i in group]
            # A newly scaled block still crosses the normal subtitle band.
            near_band = any(abs((y+b)/2-center) <= .75*max(b-y, size)
                            for _, y, _, b in bounds for center, size in self.rows)
            if (words.intersection(selected) or (not selected and near_band)) and any(
                    r-x >= .35*self.width for x, _, r, _ in bounds):
                kept.update(words)
        if selected and len(kept) > len(selected):
            card = shared_caption_card(frame, selected)
            if card is not None:
                x, y, right, bottom = card
                # Exclude an unrelated sign above a complete caption card even when its print matches.
                kept = {box for box in kept if box[0] < right and x < box[2]
                        and box[1] < bottom and y < box[3]}
        return [box for box in boxes if box in kept] if len(kept) > len(selected) else []

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


def profile_source(path,rois,find,*,variable_line_panels=False,alternate_line_heights=False):
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
        if panel is None and len(thumbs) >= 5 and sum(bool(values) for values in samples[key]) >= 4:
            result[key].highlight = learn_highlight(thumbs, windows[key])
            if variable_line_panels and result[key].highlight is None:
                from lite_line_panel import learn_line_panel
                result[key].panel = learn_line_panel(thumbs, windows[key])
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
            if panel.calibrate(observations[key]):
                continue
            if alternate_line_heights:
                from lite_line_panel import LinePanel
                if isinstance(panel, LinePanel) and panel.calibrate_heights(observations[key]):
                    continue
            result[key].panel = None
    return result
