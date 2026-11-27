import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'CLEAR-web'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from lite_ocr import cuda_detector


@pytest.mark.parametrize(('separate', 'slots'), [(True, [0, 0, 1, 0, 1, 1]), (False, [0]*6)])
def test_different_band_shapes_reuse_at_most_two_sessions(tmp_path, separate, slots):
    (tmp_path/'inference.onnx').write_bytes(b'test')
    calls = []
    sessions = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.slot = len(sessions)
            sessions.append(self)

        def get_providers(self):
            return ['CUDAExecutionProvider']

        def get_inputs(self):
            return [SimpleNamespace(name='image')]

        def run(self, outputs, feed):
            calls.append(self.slot)
            return [feed['image'].copy()]

    ort = SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session, preload_dlls=lambda **kwargs: None)
    detector = SimpleNamespace(paddlex_predictor=SimpleNamespace())
    with patch.dict(sys.modules, onnxruntime=ort), patch.dict('os.environ', {'LITE_OCR_ONNX': ''}):
        result = cuda_detector(detector, tmp_path, split_shapes=separate)
        for i, dims in enumerate([(32, 64), (32, 64), (64, 32), (32, 64), (64, 64), (64, 32)]):
            pixels = np.full((1, 3, *dims), i, np.float32)
            response = result.paddlex_predictor.infer([pixels])
            np.testing.assert_array_equal(response[0], pixels)
            if i == 1:
                assert len(sessions) == 1
    assert calls == slots
    assert len(sessions) == max(slots) + 1
