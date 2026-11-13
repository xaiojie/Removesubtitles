import importlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import imageio_ffmpeg
import numpy as np


class ProPainterWorkerTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('propainter_worker'),
                             'ProPainter needs a bounded video worker')
        return importlib.import_module('propainter_worker')

    def options(self, **values):
        defaults = dict(resolution='native', mask_dilation=8, neighbor_length=10,
                        ref_stride=5, subvideo_length=40, raft_iter=20, fp16=True)
        defaults.update(values)
        return SimpleNamespace(**defaults)

    def fake_upstream(self, source, frames_dir, masks_dir, result_dir, options, size, progress):
        output = result_dir / 'input' / 'frames'
        output.mkdir(parents=True)
        for index, file in enumerate(sorted(frames_dir.glob('*.png'))):
            image = cv2.imread(str(file))
            image[:] = 230
            cv2.imwrite(str(output / f'{index:04d}.png'), image)
        progress(50)
        progress(100)
        return output

    def test_window_budget_scales_down_for_large_native_frames(self):
        worker = self.api()
        self.assertEqual(worker.processing_size(1280, 720, 'native'), (1280, 720))
        self.assertEqual(worker.window_settings(1280, 720), (80, 8))
        large, overlap = worker.window_settings(3840, 2160)
        self.assertLess(large, 80)
        self.assertGreater(large, overlap)
        self.assertEqual(overlap, 8)
        width, height = worker.processing_size(1920, 1080, 'auto')
        self.assertLessEqual(width * height, 832 * 480)
        self.assertEqual(width % 8, 0)
        self.assertEqual(height % 8, 0)

    def test_single_frame_is_duplicated_for_cli_and_temporary_material_is_cleaned(self):
        worker = self.api()
        frame = np.full((49, 65, 3), 70, dtype=np.uint8)
        mask = np.full((49, 65), 255, dtype=np.uint8)
        seen = []
        def run(*arguments):
            seen.append(len(list(arguments[1].glob('*.png'))))
            return self.fake_upstream(*arguments)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, 'run_upstream', side_effect=run):
                outputs = worker.process_window([frame], [mask], Path('/opt/propainter'),
                                                Path(directory), self.options(), lambda percent: None)
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0].shape, frame.shape)
            self.assertTrue(np.all(outputs[0] == 230))
            self.assertEqual(seen, [2])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_wrong_output_count_is_rejected_and_cleanup_runs(self):
        worker = self.api()
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        mask = np.full((32, 48), 255, dtype=np.uint8)
        def incomplete(source, frames_dir, masks_dir, result_dir, options, size, progress):
            output = result_dir / 'input' / 'frames'
            output.mkdir(parents=True)
            cv2.imwrite(str(output / '0000.png'), frame)
            return output
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, 'run_upstream', side_effect=incomplete):
                with self.assertRaisesRegex(RuntimeError, '帧数'):
                    worker.process_window([frame], [mask], Path('/opt/propainter'), Path(directory),
                                          self.options(), lambda percent: None)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unmasked_window_skips_inference(self):
        worker = self.api()
        frame = np.full((32, 48, 3), 70, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, 'run_upstream', side_effect=AssertionError('empty mask must skip model')):
                outputs = worker.process_window([frame], [np.zeros((32, 48), dtype=np.uint8)],
                                                Path('/opt/propainter'), Path(directory), self.options(),
                                                lambda percent: None)
            np.testing.assert_array_equal(outputs[0], frame)

    def test_full_entry_preserves_timeline_audio_and_unselected_pixels(self):
        import json
        worker = self.api()
        frames = [np.full((48, 64, 3), 40 + index * 5, dtype=np.uint8) for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                       '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '64x48', '-r', '5', '-i', 'pipe:0',
                       '-f', 'lavfi', '-i', 'sine=frequency=400:sample_rate=44100', '-t', '1.6',
                       '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                       '-f', 'mp4', str(folder / 'source.video')]
            subprocess.run(command, input=b''.join(frame.tobytes() for frame in frames), check=True)
            source_capture = cv2.VideoCapture(str(folder / 'source.video'))
            source_frames = []
            while True:
                ok, source_frame = source_capture.read()
                if not ok:
                    break
                source_frames.append(source_frame)
            source_capture.release()
            config = dict(dimensions=[64, 48], regions=[dict(type='subtitle', roi=[20, 16, 42, 34])],
                          inference=dict(start_seconds=0.4, duration_seconds=0.6, segment_mode='fixed'))
            (folder / 'config.json').write_text(json.dumps(config))
            seen_context = []
            def upstream(*arguments):
                first = sorted(arguments[1].glob('*.png'))[0]
                seen_context.append(int(cv2.imread(str(first))[24, 30, 0]))
                return self.fake_upstream(*arguments)
            with patch.object(worker, 'run_upstream', side_effect=upstream):
                with patch.object(worker, 'window_settings', return_value=(2, 1)):
                    worker.run(folder, Path('/opt/propainter'))
            self.assertEqual(seen_context, [int(source_frames[2][24, 30, 0]),
                                            int(source_frames[3][24, 30, 0])])
            capture = cv2.VideoCapture(str(folder / 'output.mp4'))
            decoded = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded.append(frame)
            capture.release()
            self.assertEqual(len(decoded), 8)
            for index, frame in enumerate(decoded):
                if 2 <= index < 5:
                    self.assertGreater(int(frame[24, 30].min()), 210)
                else:
                    self.assertLessEqual(abs(int(frame[24, 30, 0]) - int(source_frames[index][24, 30, 0])), 6)
                self.assertLessEqual(abs(int(frame[4, 4, 0]) - int(source_frames[index][4, 4, 0])), 6)
            audio = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                                    '-i', str(folder / 'output.mp4'), '-map', '0:a:0', '-f', 'null', '-'],
                                   capture_output=True)
            self.assertEqual(audio.returncode, 0, audio.stderr)
            self.assertEqual(sorted(path.name for path in folder.iterdir()),
                             ['config.json', 'output.mp4', 'source.video'])

    def test_compositor_preserves_every_pixel_outside_the_mask(self):
        worker = self.api()
        self.assertTrue(callable(getattr(worker, 'composite_frame', None)))
        frame = np.arange(18*24*3, dtype=np.uint16).reshape(18, 24, 3).astype(np.uint8)
        original = frame.copy()
        result = np.full_like(frame, 240)
        mask = np.zeros((18, 24), dtype=np.uint8)
        mask[0:5, 0:7] = 255
        mask[12:18, 16:24] = 255
        worker.composite_frame(frame, result, mask)
        np.testing.assert_array_equal(frame[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(frame[mask != 0], result[mask != 0])

    def test_upstream_cli_reads_progress_and_forwards_parameters(self):
        import io
        from unittest.mock import MagicMock
        worker = self.api()
        process = MagicMock()
        process.stdout = io.StringIO('loading weights\n 20%|abc\n100%|done\n')
        process.wait.return_value = 0
        process.poll.return_value = 0
        process.__enter__.return_value = process
        values = []
        with patch.object(worker.subprocess, 'Popen', return_value=process) as launch:
            output = worker.run_upstream(Path('/opt/propainter'), Path('/tmp/job/input'),
                                         Path('/tmp/job/masks'), Path('/tmp/job/results'),
                                         self.options(), (1280, 720), values.append)
        self.assertEqual(values, [20, 100])
        self.assertEqual(output, Path('/tmp/job/results/input/frames'))
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index('--mask_dilation') + 1], '8')
        self.assertEqual(command[command.index('--subvideo_length') + 1], '40')
        self.assertIn('--save_frames', command)
        self.assertIn('--fp16', command)
        self.assertEqual(launch.call_args.kwargs['cwd'], Path('/opt/propainter'))
