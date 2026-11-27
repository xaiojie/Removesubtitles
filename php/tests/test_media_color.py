import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from media import capture, encode, geometry, open_video_capture


def make_clip(path, matrix=None, color_range='tv', sar='1', transfer=None):
    bars = np.zeros((32, 64, 3), dtype=np.uint8)
    for i, bgr in enumerate(((53, 173, 94), (30, 35, 218), (207, 65, 14), (76, 76, 76))):
        bars[:, 16 * i:16 * (i + 1)] = bgr
    vf = f'setsar={sar}'
    tags = []
    if matrix:
        conversion = 'bt709' if matrix == 'bt709' else 'bt601'
        vf = f'scale=out_color_matrix={conversion}:out_range={color_range},format=yuv420p,{vf}'
        tags = ['-colorspace', matrix, '-color_range', color_range]
        if transfer:
            vf += f',setparams=color_primaries={matrix}:color_trc={transfer}'
            tags += ['-color_primaries', matrix, '-color_trc', transfer]
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', '64x32',
        '-framerate', '8', '-i', 'pipe:0', '-f', 'lavfi', '-i', 'sine=duration=0.5:sample_rate=48000',
        '-map', '0:v', '-map', '1:a', '-vf', vf, '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-crf', '0',
        *tags, '-c:a', 'aac', str(path),
    ], input=bars.tobytes() * 4, check=True)


def streams(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path),
    ]))['streams']


def first_frame(path, width=64, height=32):
    data = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1', '-an',
        '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1',
    ])
    return np.frombuffer(data, np.uint8).reshape(height, width, 3)


def copy_pixels(source, output):
    with capture(source) as cap:
        fps, count = geometry(cap)
        ok, frame = cap.read()
        assert ok
        with encode(output, source, frame.shape[1::-1], fps, 18) as writer:
            while ok:
                writer.write(frame.tobytes())
                ok, frame = cap.read()
        return count


class MediaColorTests(unittest.TestCase):
    def test_display_colors_metadata_and_aac_packets_survive_round_trip(self):
        cases = [('bt709', 'tv', 'bt709'), ('bt709', 'pc', 'bt709'),
                 ('bt470bg', 'tv', None), ('bt470bg', 'tv', 'bt470bg')]
        for matrix, color_range, transfer in cases:
            with self.subTest(matrix=matrix, color_range=color_range, transfer=transfer), tempfile.TemporaryDirectory() as tmp:
                src, out = Path(tmp) / 'source.mp4', Path(tmp) / 'result.mp4'
                make_clip(src, matrix, color_range, transfer=transfer)
                self.assertEqual(copy_pixels(src, out), 4)
                a, b = first_frame(src), first_frame(out)
                np.testing.assert_allclose(a[16, 8::16].astype(int), b[16, 8::16], atol=3)
                source_video, result_video = streams(src)[0], streams(out)[0]
                if transfer:
                    self.assertEqual(source_video['color_primaries'], matrix)
                    self.assertEqual(source_video['color_transfer'], transfer)
                for field in ('color_space', 'color_range', 'color_transfer', 'color_primaries', 'nb_frames'):
                    self.assertEqual(result_video.get(field), source_video.get(field), field)
                audio_digest = ['ffmpeg', '-v', 'error', '-i']
                audio_map = ['-map', '0:a:0', '-c:a', 'copy', '-f', 'hash', 'pipe:1']
                self.assertEqual(subprocess.check_output(audio_digest + [str(src)] + audio_map),
                                 subprocess.check_output(audio_digest + [str(out)] + audio_map))

    def test_rotated_anamorphic_source_keeps_display_orientation_and_square_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, turn, out = (Path(tmp) / f'{name}.mp4' for name in ('source', 'rotated', 'result'))
            make_clip(src, 'bt709', sar='3/2')
            subprocess.run(['ffmpeg', '-v', 'error', '-display_rotation', '90', '-i', str(src),
                            '-map', '0', '-c', 'copy', str(turn)], check=True)
            with capture(turn) as cap:
                ok, decoded = cap.read()
                self.assertTrue(ok)
                self.assertEqual(decoded.shape, (96, 32, 3))
                # A .read() result must also allow the model to composite in place.
                decoded[0, 0] = 0
            copy_pixels(turn, out)
            info = streams(out)[0]
            self.assertEqual((info['width'], info['height']), (32, 96))
            self.assertIn(info.get('sample_aspect_ratio'), (None, '1:1'))
            self.assertFalse(any('rotation' in side for side in info.get('side_data_list', ())))
            reference = np.rot90(first_frame(src))
            result = first_frame(out, 32, 96)
            np.testing.assert_allclose(result[12::24, 16].astype(int), reference[8::16, 16], atol=3)

    def test_unspecified_matrix_keeps_existing_decode_and_output_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / 'source.mp4', Path(tmp) / 'result.mp4'
            make_clip(src)
            old = open_video_capture(src)
            try:
                with capture(src) as cap:
                    np.testing.assert_array_equal(cap.read()[1], old.read()[1])
            finally:
                old.release()
            copy_pixels(src, out)
            self.assertIsNone(streams(out)[0].get('color_space'))
