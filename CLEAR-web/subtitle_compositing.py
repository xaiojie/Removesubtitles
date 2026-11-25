"""Conservative original-derived caption masks and selection-bounded Poisson fusion."""
import cv2
import numpy as np


def large_caption_guidance(crop):
    """Find tall aligned glyphs and a fill mask for opaque strokes that CLEAR tends to copy."""
    if min(crop.shape[:2]) < 40:
        return False, None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    nearby = cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    bright = cv2.inRange(crop, (180, 180, 180), (255, 255, 255))
    bright |= cv2.inRange(crop, (0, 145, 180), (130, 255, 255))
    strokes = ((bright > 0) & (cv2.subtract(gray, nearby) >= 90)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    glyphs = []
    pairs = []
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if h < 40 or not .2*h <= w <= 1.7*h or area < 1.6*h:
            continue
        for other in glyphs:
            ox, oy, ow, oh = stats[other, :4]
            if (.75*h <= oh <= 1.33*h and abs(y+h/2-oy-oh/2) <= .25*min(h, oh)
                    and .8*h <= abs(x+w/2-ox-ow/2) <= 2.8*h):
                pairs.append((other, label))
        glyphs.append(label)
    if not pairs:
        return False, None
    depth = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    thick_pixels = np.bincount(labels[depth >= 3.5], minlength=count)
    bold = thick_pixels >= .18*stats[:, cv2.CC_STAT_AREA]
    bold_pairs = [(a, b) for a, b in pairs if bold[a] and bold[b]]
    if not bold_pairs:
        return True, None
    # Include short detached strokes and punctuation beside the supporting pair.
    # The mask remains confined to bright, locally contrasted strokes and their outlines.
    mask = np.zeros_like(strokes)
    for a, b in bold_pairs:
        x, y, w, h = stats[a, :4]
        ox, oy, ow, oh = stats[b, :4]
        reach = max(h, oh)
        left, right = max(0, min(x, ox)-reach), min(mask.shape[1], max(x+w, ox+ow)+reach)
        top, bottom = max(0, min(y, oy)-reach//6), min(mask.shape[0], max(y+h, oy+oh)+reach//6)
        mask[top:bottom, left:right] = strokes[top:bottom, left:right]*255
    anchors = list({label for pair in bold_pairs for label in pair})
    # Leave clean pixels outside the outline for seamlessClone's inner boundary.
    radius = max(6, min(18, round(float(np.median(stats[anchors, cv2.CC_STAT_HEIGHT]))*.16)))+3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return True, cv2.dilate(mask, kernel)


def _glyph_components(candidates, minimum_height, vertical=3):
    connected = cv2.morphologyEx(candidates.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((vertical, 3), np.uint8))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(connected, 8)
    height, width = candidates.shape
    components = []
    for label, (x, y, glyph_width, glyph_height, area) in enumerate(statistics[1:count], 1):
        if area < max(3, minimum_height//2) or glyph_height < minimum_height:
            continue
        if glyph_width > width*.35 or glyph_height > height*.95:
            continue
        components.append((label, int(x), int(y), int(glyph_width), int(glyph_height)))
    return labels, components


def _coherent_glyphs(components, allow_single=False):
    """Keep dominant aligned lines and reject isolated scene highlights."""
    if not components:
        return []
    groups = []
    for anchor in components:
        _, _, y, _, height = anchor
        group = [other for other in components
                 if .65*height <= other[4] <= 1.5*height
                 and abs(y+height-other[2]-other[4]) <= max(3, min(height, other[4])*.3)]
        score = len(group)*float(np.median([item[4] for item in group]))
        groups.append((score, group))
    best = max(score for score, _ in groups)
    selected = {}
    for score, group in groups:
        if score < best*.55:
            continue
        if len(group) < 3 and not (allow_single or any(item[3] >= item[4]*3 for item in group)):
            continue
        selected.update((item[0], item) for item in group)
    return list(selected.values())


def _line_plates(candidates, dark):
    """Recover black padding from each text line even when it touches dark scenery."""
    height, width = candidates.shape
    minimum_height = max(2, min(8, round(height*.06)))
    labels, components = _glyph_components(candidates, minimum_height)
    plates = np.zeros_like(dark)
    if not components:
        return plates
    typical_height = float(np.median([item[4] for item in components]))
    maximum_height = height*(.8 if height < typical_height*3 else .45)
    glyphs = np.zeros_like(dark)
    for label, _, _, _, glyph_height in components:
        if glyph_height <= maximum_height:
            glyphs[labels == label] = 255
    linked = cv2.morphologyEx(glyphs, cv2.MORPH_CLOSE,
                              np.ones((3, max(3, round(typical_height*1.25)) | 1), np.uint8))
    count, _, statistics, _ = cv2.connectedComponentsWithStats(linked, 8)
    for x, y, line_width, line_height, area in statistics[1:count]:
        if line_width < line_height*2 or area < minimum_height*3:
            continue
        sample = dark[max(0, y-3):y+line_height+3, max(0, x-3):x+line_width+3]
        if float(sample.mean()) < .35:
            continue
        left, top, right, bottom = int(x), int(y), int(x+line_width), int(y+line_height)
        vertical_limit = max(3, int(line_height*2))
        horizontal_limit = max(3, int(line_height*3))
        while top > max(0, y-vertical_limit) and dark[top-1, x:x+line_width].mean() >= .65:
            top -= 1
        while bottom < min(height, y+line_height+vertical_limit) and dark[bottom, x:x+line_width].mean() >= .65:
            bottom += 1
        while left > max(0, x-horizontal_limit) and dark[top:bottom, left-1].mean() >= .65:
            left -= 1
        while right < min(width, x+line_width+horizontal_limit) and dark[top:bottom, right].mean() >= .65:
            right += 1
        # Plate edges must be visible locally. A uniformly dark scene has no plate boundary.
        if (top == max(0, y-vertical_limit) or bottom == min(height, y+line_height+vertical_limit)):
            continue
        plate = dark[top:bottom, left:right]
        boundary = np.concatenate((plate[0], plate[-1], plate[:, 0], plate[:, -1]))
        if float(plate.mean()) < .55 or float(boundary.mean()) < .8:
            continue
        padding = max(2, round(line_height*.25))
        plates[max(0, top-padding):bottom+padding, max(0, left-padding):right+padding] = 255
    return plates


def _paired_stroke_mask(crop, candidates):
    """Find bright pen strokes bounded on both sides, even on a similarly bright background."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    reach = max(5, min(12, round(min(height, width)*.055)))
    contrast = np.zeros_like(gray)
    for dx, dy in ((reach, 0), (0, reach)):
        kernel = np.ones((dy+1, dx+1), np.uint8)
        before = cv2.erode(gray, kernel, anchor=(dx, dy))
        after = cv2.erode(gray, kernel, anchor=(0, 0))
        contrast = np.maximum(contrast, cv2.subtract(gray, np.maximum(before, after)))
    strokes = (candidates & (contrast >= 40)).astype(np.uint8)
    # Keep adjacent glyphs separate until their shared line has been located.
    _, _, stats, _ = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    anchors = []
    centers = np.zeros_like(gray)
    for x, y, w, h, area in stats[1:]:
        if h < 4 or h > height*.9 or w > 3*h or area < max(8, h):
            continue
        x, y, w, h = map(int, (x, y, w, h))
        centers[y+h//2, x+w//2] = 1
        anchors.append((x, y, w, h))

    # Require another letter on the same line. Its neighborhood can then
    # admit small detached dots, accents and outlines without filling the line.
    counts = cv2.integral(centers)
    supported = np.zeros_like(gray)
    for x, y, w, h in anchors:
        middle = y+h//2
        left, right = max(0, x-3*h), min(width, x+w+3*h)
        top, bottom = max(0, middle-h//3), min(height, middle+h//3+1)
        neighbors = counts[bottom, right]-counts[top, right]-counts[bottom, left]+counts[top, left]
        if neighbors >= 2:
            supported[max(0, y-h//3):min(height, y+h+h//3), left:right] = 1
    # Once a line is supported, include faint punctuation and antialiased strokes.
    strokes = candidates & (contrast >= 24) & (supported != 0)
    # Include the outline and leave clean pixels for the Poisson boundary.
    padding = reach+2
    return cv2.dilate(strokes.astype(np.uint8)*255, np.ones((2*padding+1, 2*padding+1), np.uint8))


def caption_mask(frame, roi):
    """Cover light/colored glyphs, their outlines and rectangular dark caption plates.

    Detection is confined to the selected ROI. An empty detection stays empty.
    Coordinates and stroke expansion are derived from the current image and glyphs.
    """
    left, top, right, bottom = roi
    crop = frame[top:bottom, left:right]
    full_mask = np.zeros(frame.shape[:2], np.uint8)
    if not crop.size:
        return full_mask
    pixels = crop.astype(np.int16)
    blue, green, red = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    white = (pixels.min(axis=2) >= 180) & (pixels.max(axis=2)-pixels.min(axis=2) <= 70)
    yellow = (red >= 165) & (green >= 145) & (blue <= 130)
    orange = (red >= 140) & (green >= 30) & (blue < 110) & (red > green*1.3)
    glyphs = np.zeros(crop.shape[:2], np.uint8)
    height, width = glyphs.shape
    minimum_height = max(3, min(16, round(height*.18)))
    vertical = max(3, min(9, round(height*.1)) | 1)
    outline_labels, outline_components = _glyph_components(white, minimum_height, vertical)
    outlined = []
    for component in outline_components:
        _, x, y, glyph_width, glyph_height = component
        margin = max(3, round(glyph_height*.25))
        vicinity = orange[max(0, y-margin):y+glyph_height+margin,
                          max(0, x-margin):x+glyph_width+margin]
        if np.count_nonzero(vicinity) >= max(3, glyph_width*glyph_height*.02):
            outlined.append(component)
    outlined = _coherent_glyphs(outlined, allow_single=True)
    heights = []
    for label, _, _, _, glyph_height in outlined:
        glyphs[outline_labels == label] = 255
        heights.append(glyph_height)

    candidates = white | yellow
    labels, raw_components = _glyph_components(candidates, max(2, min(8, round(height*.06))))
    components = _coherent_glyphs(raw_components)
    line_glyphs = np.zeros_like(glyphs)
    for label, _, _, _, _ in components:
        line_glyphs[labels == label] = 255

    dark = (pixels.max(axis=2) <= 50).astype(np.uint8)
    plates = _line_plates(candidates, dark)
    count, _, statistics, _ = cv2.connectedComponentsWithStats(dark, 8)
    for x, y, plate_width, plate_height, area in statistics[1:count]:
        if plate_width < plate_height*1.4 or area < 20:
            continue
        plate = dark[y:y+plate_height, x:x+plate_width]
        text = line_glyphs[y:y+plate_height, x:x+plate_width]
        if not text.any() or float(plate.mean()) < .35:
            continue
        boundary = np.concatenate((plate[0], plate[-1], plate[:, 0], plate[:, -1]))
        if float(boundary.mean()) < .7:
            continue
        # A dark scene surrounding the text must not become a whole-ROI plate.
        if plate_width*plate_height > width*height*.85 or plate_height > height*.75:
            outer = frame[max(0, top+y-1):min(frame.shape[0], top+y+plate_height+1),
                          max(0, left+x-1):min(frame.shape[1], left+x+plate_width+1)]
            ring = np.concatenate((outer[0], outer[-1], outer[:, 0], outer[:, -1]))
            if float((ring.max(axis=1) <= 50).mean()) > .65:
                continue
        plates[y:y+plate_height, x:x+plate_width] = 255
    for label, x, y, glyph_width, glyph_height in raw_components:
        if plates[y:y+glyph_height, x:x+glyph_width].any():
            glyphs[labels == label] = 255
            heights.append(glyph_height)
    # Local stroke shape and contrast also support short or single-glyph captions.
    for label, x, y, glyph_width, glyph_height in raw_components:
        margin = max(2, round(glyph_height*.2))
        sx, sy = max(0, x-margin), max(0, y-margin)
        sample = pixels[sy:y+glyph_height+margin, sx:x+glyph_width+margin]
        core = (labels[sy:y+glyph_height+margin, sx:x+glyph_width+margin] == label).astype(np.uint8)
        ring = (cv2.dilate(core, np.ones((2*margin+1, 2*margin+1), np.uint8)) > 0) & (core == 0)
        if not ring.any():
            continue
        contrast = np.abs(np.median(sample[core > 0], axis=0)-np.median(sample[ring], axis=0)).max()
        dark_outline = float((sample[ring].max(axis=1) <= 80).mean()) >= .12
        fill = float((labels[y:y+glyph_height, x:x+glyph_width] == label).mean())
        stroke_shape = fill <= .85 or glyph_width <= glyph_height*.3
        touches_edge = x == 0 or y == 0 or x+glyph_width == width or y+glyph_height == height
        if touches_edge and glyph_height > height*.5:
            continue
        if contrast >= 24 and (dark_outline or stroke_shape):
            glyphs[labels == label] = 255
            heights.append(glyph_height)
    stroke_mask = _paired_stroke_mask(crop, candidates)
    if not heights:
        full_mask[top:bottom, left:right] = stroke_mask
        return full_mask
    # Leave clean boundary pixels beyond the outline/shadow for the Poisson solve.
    radius = max(4, min(32, int(np.ceil(float(np.median(heights))*.6))+4))
    kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
    vicinity = cv2.dilate(glyphs, kernel)
    if outlined:
        glyphs |= orange.astype(np.uint8)*255 & vicinity
    mask = cv2.dilate(glyphs, kernel)
    padding = max(2, round(radius*.5))
    plates = cv2.dilate(plates, np.ones((2*padding+1, 2*padding+1), np.uint8))
    mask |= plates | stroke_mask
    full_mask[top:bottom, left:right] = mask
    return full_mask


def composite_caption(original, restored, mask):
    """Fuse generated pixels while keeping every pixel outside the mask exact."""
    selected = mask > 0
    result = original.copy()
    if not selected.any():
        return result
    x, y, width, height = cv2.boundingRect(mask)
    fused = restored
    if min(width, height) >= 5:
        touches_edge = x == 0 or y == 0 or x+width == mask.shape[1] or y+height == mask.shape[0]
        pad = 16 if touches_edge else 1
        # Continue selected pixels beyond the frame using the reconstruction.
        # Replicating the original subtitle here would pull its shadow back into the solve.
        continuation = np.where(selected[:, :, None], restored, original) if touches_edge else original
        destination = cv2.copyMakeBorder(continuation, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        source = cv2.copyMakeBorder(restored, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        padded_mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        if touches_edge:
            destination[pad:-pad, pad:-pad] = original
            padded_mask[[0, -1], :] = 0
            padded_mask[:, [0, -1]] = 0
        x, y, width, height = cv2.boundingRect(padded_mask)
        center = (x+width//2, y+height//2)
        # seamlessClone mutates its mask. Keep the selection and caller's mask intact.
        fused = cv2.seamlessClone(source, destination, padded_mask, center,
                                  cv2.NORMAL_CLONE)[pad:-pad, pad:-pad]
    result[selected] = fused[selected]
    return result
