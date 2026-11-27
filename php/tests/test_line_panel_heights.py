import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from lite_line_panel import LinePanel


def test_follow_two_verified_backing_heights_without_touching_uncaptioned_frames():
    roi = (0, 0, 360, 130)
    panel = LinePanel(roi, (360, 130), (100, 43, 260, 91), .55, np.full(3, 23., np.float32))
    observations = [(43, 91), (43, 97), (42, 90), (43, 96), (43, 91)]
    assert not panel.calibrate(observations)
    assert panel.calibrate_heights(observations)
    background = np.full((130, 360, 3), (118, 186, 144), np.uint8)
    boxes = [(100, 48, 262, 84)]
    for bottom in (91, 97, 91):
        frame = background.copy()
        patch = frame[43:bottom, 86:279]
        patch[:] = np.rint(.55 * patch + 23).astype(np.uint8)
        cv2.putText(frame, 'SUBTITLE', (107, 77), cv2.FONT_HERSHEY_SIMPLEX,
                    .84, (252, 252, 252), 2, cv2.LINE_AA)
        result = panel.restore(frame, roi, boxes)
        assert result is not None
        bounds, corrected, _ = result
        assert bounds == (86, 43, 279, bottom)
        assert cv2.absdiff(corrected[-2:], background[bottom-2:bottom, 86:279]).max() < 3
    assert panel.restore(background, roi, boxes) is None
    assert panel.restore(frame, roi, []) is None


def test_unrelated_scene_edges_do_not_establish_an_alternate_height():
    roi = (0, 0, 360, 130)
    panel = LinePanel(roi, (360, 130), (100, 43, 260, 91), .55, np.full(3, 23., np.float32))
    assert not panel.calibrate_heights([(42, 90), (48, 75), (37, 83), (46, 94), (49, 64)])
    assert not panel.calibrate_heights([(43, 90), (43, 91), (42, 93), (43, 100), (43, 80)])
