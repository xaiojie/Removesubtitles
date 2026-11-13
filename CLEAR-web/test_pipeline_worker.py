import importlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np


class PipelineWorkerTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('pipeline_worker'))
        return importlib.import_module('pipeline_worker')

    def config(self, pipeline=('clear', 'propainter')):
        return dict(engine='hybrid' if len(pipeline) == 2 else pipeline[0], pipeline=list(pipeline),
                    dimensions=[32, 24], original_filename='input.mp4',
                    regions=[dict(type='subtitle', roi=[4, 12, 20, 22]),
                             dict(type='watermark', roi=[22, 2, 30, 10])],
                    inference=dict(start_seconds=1.25, duration_seconds=2.5, export_crf=10,
                                   clear=dict(steps=5), propainter=dict(mask_dilation=8)))

    def test_stage_configs_filter_region_types_and_keep_shared_time_coordinates(self):
        worker = self.api()
        config = self.config()
        clear = worker.stage_config(config, 'clear', Path('/job/source.video'))
        water = worker.stage_config(config, 'propainter', Path('/job/source.video'))
        self.assertEqual(clear['regions'], [dict(type='subtitle', roi=[4, 12, 20, 22])])
        self.assertEqual(water['regions'], [dict(type='watermark', roi=[22, 2, 30, 10])])
        self.assertTrue(clear['lossless_output'])
        self.assertEqual(water['mask_reference'], '/job/source.video')
        for stage in (clear, water):
            self.assertEqual(stage['dimensions'], [32, 24])
            self.assertEqual(stage['inference']['start_seconds'], 1.25)
            self.assertEqual(stage['inference']['duration_seconds'], 2.5)
            self.assertEqual(stage['inference']['export_crf'], 10)
        single = worker.stage_config(self.config(('clear',)), 'clear', Path('/job/source.video'))
        self.assertFalse(single.get('lossless_output', False))
        single_water = worker.stage_config(self.config(('propainter',)), 'propainter', Path('/job/source.video'))
        self.assertNotIn('mask_reference', single_water)

    def test_failed_second_stage_resumes_without_rerunning_clear_and_progress_is_monotone(self):
        worker = self.api()
        calls, statuses = [], []
        fail_water = [True]
        def phase(name, stage, source, python, forward):
            calls.append(name)
            for percent in (5, 80, 50, 100):
                forward(dict(percent=percent, stage='working'))
            if name == 'propainter' and fail_water[0]:
                fail_water[0] = False
                raise RuntimeError('temporary failure')
            config = json.loads((stage / 'config.json').read_text())
            (stage / ('output.mkv' if config.get('lossless_output') else 'output.mp4')).write_bytes(b'finished video')
        def finalize(artifact, original, destination, remux_audio):
            shutil.copyfile(artifact, destination)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'source.video').write_bytes(b'original video')
            (folder / 'config.json').write_text(json.dumps(self.config()))
            with patch.object(worker, 'run_phase', side_effect=phase), patch.object(worker, 'report', side_effect=lambda **v: statuses.append(v)), patch.object(worker, 'finalize_output', side_effect=finalize):
                with self.assertRaisesRegex(RuntimeError, 'temporary failure'):
                    worker.run(folder, Path('/clear'), Path('/pp'), Path('/pp/python'))
                self.assertFalse((folder / 'output.mp4').exists())
                self.assertEqual(worker.completed_stages(folder), ['clear'])
                statuses.clear()
                worker.run(folder, Path('/clear'), Path('/pp'), Path('/pp/python'))
            self.assertEqual(calls, ['clear', 'propainter', 'propainter'])
            self.assertEqual(worker.completed_stages(folder), ['clear', 'propainter'])
            self.assertEqual((folder / 'output.mp4').read_bytes(), b'finished video')
            self.assertEqual((folder / 'stages/propainter/source.video').resolve(),
                             (folder / 'stages/clear/output.mkv').resolve())
            self.assertEqual(statuses[-1]['percent'], 100)
            self.assertEqual(statuses[-1]['phase_label'], '封装导出')
            values = [v['percent'] for v in statuses if 'percent' in v]
            self.assertEqual(values, sorted(values))
            self.assertGreaterEqual(values[0], 58)
            stage_reports = [v for v in statuses if v.get('phase') in ('clear', 'propainter')]
            self.assertTrue(all(v['percent'] < 100 for v in stage_reports))
            (folder / 'stages/clear/output.mkv').unlink()
            self.assertEqual(worker.completed_stages(folder), [])

    def test_success_marker_without_an_actual_product_is_never_completed(self):
        worker = self.api()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = self.config(('clear',))
            (folder / 'config.json').write_text(json.dumps(config))
            stage = folder / 'stages/clear'
            stage.mkdir(parents=True)
            (stage / 'success.json').write_text(json.dumps(dict(stage='clear', artifact='output.mp4')))
            self.assertEqual(worker.completed_stages(folder), [])

    def test_clear_lossless_intermediate_roundtrips_odd_sized_rgb_exactly(self):
        import worker
        from inference_options import InferenceOptions
        frame = np.arange(5*7*3, dtype=np.uint8).reshape(5, 7, 3)
        class Capture:
            def read(self):
                return False, None
        def pipe(**values):
            return values['control_video']
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            options = InferenceOptions(context='frame', resolution='native', blend_mode='rectangle', segment_mode='fixed')
            worker.encode(Capture(), frame.copy(), 30, 1, pipe,
                          [dict(type='subtitle', roi=[1, 1, 3, 3])], folder, options, (0, 1), [],
                          lossless_output=True)
            decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                                      '-i', str(folder / 'output.mkv'), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                                     capture_output=True, check=True)
            self.assertEqual(decoded.stdout, frame.tobytes())
            self.assertFalse((folder / 'output.mp4').exists())
            inspect = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', str(folder / 'output.mkv')], capture_output=True)
            self.assertIn(b'ffv1', inspect.stderr)
            self.assertIn(b'bgr0', inspect.stderr)

    def test_final_mux_preserves_original_audio_and_video_pixels(self):
        worker = self.api()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            original, repaired = folder/'original.mp4', folder/'repaired.mp4'
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=32x24:r=5',
                            '-f', 'lavfi', '-i', 'sine=frequency=500', '-t', '0.4', '-c:v', 'libx264', '-c:a', 'aac', str(original)], check=True)
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(original), '-map', '0:v:0', '-c:v', 'copy', '-an', str(repaired)], check=True)
            result = folder/'output.mp4'
            worker.finalize_output(repaired, original, result, remux_audio=True)
            audio = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(result), '-map', '0:a:0', '-f', 'null', '-'], capture_output=True)
            self.assertEqual(audio.returncode, 0, audio.stderr)
            def pixels(path):
                return subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True, check=True).stdout
            self.assertEqual(pixels(result), pixels(repaired))

    def test_propainter_samples_and_activates_from_original_reference(self):
        import propainter_worker as worker
        from propainter_options import ProPainterOptions
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        seen = []
        class Builder:
            def mask(self, frame):
                seen.append(int(frame[6, 6, 0]))
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[4:12, 4:12] = 255
                return mask
            def expanded_mask(self, mask):
                return mask
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reference = folder/'reference.mkv'
            for path, value in ((reference, 200), (folder/'source.video', 40)):
                pixels = np.full((3, 24, 32, 3), value, dtype=np.uint8).tobytes()
                subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '32x24', '-r', '5', '-i', 'pipe:0', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-f', 'matroska', str(path)], input=pixels, check=True)
            information = (32, 24, 5, 3, [], Builder())
            def process(frames, *args):
                return [np.full_like(frame, 230) for frame in frames]
            with patch.object(worker, 'process_window', side_effect=process):
                worker.encode(folder, Path('/pp'), ProPainterOptions(segment_mode='fixed'), information, (0, 3), mask_reference=reference)
            self.assertEqual(seen, [200, 200, 200])
            seen.clear()
            with patch.object(worker, 'process_window', side_effect=process):
                with patch.object(worker, 'window_settings', return_value=(2, 1)):
                    worker.encode(folder, Path('/pp'), ProPainterOptions(segment_mode='fixed'), information, (0, 3))
            self.assertEqual(seen, [40, 40, 40, 40])
