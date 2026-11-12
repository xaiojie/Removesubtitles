from itertools import chain, islice

import cv2
import numpy as np


def decoded_frames(capture):
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


def find_shot_cuts(frames, roi):
    """Return frame indices that start new shots, using one frame of lookahead."""
    previous = None
    weights = None
    preceding_delta = 0.0
    pending = None
    cuts = []

    def consider(candidate, following_delta):
        index, delta, changed = candidate
        strong = delta >= 35 and changed >= 0.30
        isolated = (delta >= 18 and changed >= 0.23
                    and delta > 1.15 * preceding_delta and delta > 2 * following_delta)
        if strong or isolated:
            cuts.append(index)

    for index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        thumbnail = cv2.resize(frame, (48, 64), interpolation=cv2.INTER_AREA).astype(np.int16)
        if weights is None:
            left, top, right, bottom = roi
            weights = np.ones(thumbnail.shape[:2], dtype=np.float32)
            weights[top*64//height:(bottom*64+height-1)//height,
                    left*48//width:(right*48+width-1)//width] = 0.03
            weights /= weights.sum()
        if previous is not None:
            difference = np.abs(thumbnail - previous).mean(axis=2)
            delta = float(np.sum(difference * weights))
            changed = float(np.sum((difference > 25) * weights))
            if pending is not None:
                consider(pending, delta)
                preceding_delta = pending[1]
            pending = (index, delta, changed)
        previous = thumbnail
    if pending is not None:
        consider(pending, pending[1])
    return cuts


def video_batches(frames, frame_range, cuts, options):
    """Yield context, new frames, and whether the batch should run inference."""
    start, end = frame_range
    frames = iter(frames)
    for frame in islice(frames, start):
        yield [], [frame], False

    position = start
    shot_ends = (cut for cut in cuts if cut > start and (end is None or cut < end))
    for shot_end in chain(shot_ends, (end,)):
        carry = []
        while shot_end is None or position < shot_end:
            capacity = options.chunk_frames - len(carry)
            if shot_end is not None:
                capacity = min(capacity, shot_end - position)
            pending = list(islice(frames, capacity))
            if not pending:
                return
            batch = carry + pending
            overlap = options.overlap_frames
            next_carry = [frame.copy() for frame in batch[-overlap:]] if overlap else []
            yield carry, pending, True
            position += len(pending)
            carry = next_carry

    for frame in frames:
        yield [], [frame], False
