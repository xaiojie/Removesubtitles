"""Share LaMa predictions for repeated caption crops within a batch."""
import cv2
import numpy as np


def inference_frames(images, masks, tolerance):
    if tolerance == 0:
        indices = list(range(len(images)))
        return indices, indices

    anchors = []
    order = []
    for i, image in enumerate(images):
        # Compare against the actual model input frame to prevent drift on slow pans and fades.
        if not anchors or not np.array_equal(masks[i], masks[anchors[-1]]) or (
            cv2.norm(image, images[anchors[-1]], cv2.NORM_INF) > tolerance
        ):
            anchors.append(i)
        order.append(len(anchors) - 1)
    return anchors, order
