"""Decode videos in the square-pixel display coordinates used by browsers."""
import math

import cv2


class DisplayCapture:
    def __init__(self, capture, pixel_aspect):
        self.capture = capture
        self.pixel_aspect = pixel_aspect
        rotation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
        self.rotated = bool(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)) and rotation % 180 == 90
        self.dimensions = None

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            return ok, frame
        height, width = frame.shape[:2]
        if self.rotated:
            height = max(1, math.floor(height * self.pixel_aspect + 0.5))
        else:
            width = max(1, math.floor(width * self.pixel_aspect + 0.5))
        self.dimensions = width, height
        return True, cv2.resize(frame, self.dimensions, interpolation=cv2.INTER_LINEAR)

    def get(self, property_id):
        if self.dimensions is not None:
            if property_id == cv2.CAP_PROP_FRAME_WIDTH:
                return self.dimensions[0]
            if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.dimensions[1]
        if property_id in (cv2.CAP_PROP_SAR_NUM, cv2.CAP_PROP_SAR_DEN):
            return 1
        return self.capture.get(property_id)

    def __getattr__(self, name):
        return getattr(self.capture, name)


def open_video_capture(path, backend=cv2.CAP_FFMPEG):
    capture = cv2.VideoCapture(str(path), backend)
    numerator = capture.get(cv2.CAP_PROP_SAR_NUM)
    denominator = capture.get(cv2.CAP_PROP_SAR_DEN)
    if (not math.isfinite(numerator) or not math.isfinite(denominator)
            or numerator <= 0 or denominator <= 0 or numerator == denominator):
        return capture
    return DisplayCapture(capture, numerator / denominator)
