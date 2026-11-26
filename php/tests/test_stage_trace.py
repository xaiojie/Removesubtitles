import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from lite import detect, restore


class StageTraceTests(unittest.TestCase):
    def test_lite_records_only_invoked_ocr_and_inpaint_calls(self):
        class Detector:
            def predict(self, _crop):
                return [{'dt_polys': []}]

        class Inpaint:
            device = 'cpu'

            def _inpaint_batch(self, crops, masks):
                return crops

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stages.jsonl'
            frame = np.zeros((32, 40, 3), np.uint8)
            with patch.dict(os.environ, SUBTITLE_TRACE_FILE=str(path)):
                detect(frame, [(0, 0, 40, 32)], Detector())
                restore([frame], [[]], Inpaint())
                restore([frame], [[(5, 8, 14, 20)]], Inpaint())
            self.assertTrue(path.is_file())
            spans = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([s['stage'] for s in spans], ['lite_ocr', 'lite_lama'])
            self.assertEqual([s['device'] for s in spans], ['cpu', 'cpu'])
            self.assertTrue(all(s['finished_at'] >= s['started_at'] and s['ok'] for s in spans))

    def test_started_calls_survive_interrupt_and_allocator_peaks_identify_the_process(self):
        from job_trace import span

        gpu = SimpleNamespace(is_initialized=lambda: True, memory_reserved=lambda: 3 * 1024 ** 2,
                              max_memory_reserved=lambda: 7 * 1024 ** 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'stages.jsonl'
            with patch.dict(os.environ, SUBTITLE_TRACE_FILE=str(path)), patch.dict(sys.modules, torch=SimpleNamespace(cuda=gpu)):
                with span('clear_inference', device='cuda'):
                    start = json.loads((path.parent / 'stage_starts.jsonl').read_text())
                    self.assertEqual(start['stage'], 'clear_inference')
            end = json.loads(path.read_text())
            self.assertEqual(start['span_id'], end['span_id'])
            self.assertEqual(end['torch_peak_reserved_mib'], 7)
