import unittest
import io
import json
from types import SimpleNamespace
from werkzeug.http import parse_options_header
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from app import app, region, model_status


class WebTests(unittest.TestCase):

    def test_model_status_counts_written_bytes_of_sparse_partial_download(self):
        with TemporaryDirectory() as root:
            model = Path(root) / 'model.bin'
            partial = model.with_suffix('.bin.partial')
            partial.write_bytes(b'partial')
            original_stat = Path.stat
            def file_stat(path, *args, **kwargs):
                if path == partial:
                    return SimpleNamespace(st_size=1_000_000_000, st_blocks=500_000)
                return original_stat(path, *args, **kwargs)
            with patch('app.REQUIRED', {model: 1_000_000_000}), patch.object(Path, 'stat', file_stat):
                ready, message = model_status()
            self.assertFalse(ready)
            self.assertIn('0.3 / 1.0 GB', message)

    def test_rejects_rectangle_outside_video(self):
        with self.assertRaises(ValueError):
            region('[0,0,641,360]', '[640,360]')

    def test_system_reports_status_with_no_video(self):
        with app.test_client() as client:
            response = client.get('/api/system')
        self.assertEqual(response.status_code, 200)
        self.assertIn('gpu', response.json)
        self.assertIsInstance(response.json['ready'], bool)


    def test_job_rejects_invalid_inference_before_saving_video(self):
        with patch('app.model_status', return_value=(True, 'ready')), app.test_client() as client:
            response = client.post('/api/jobs', data={
                'roi': '[11,1046,720,1222]',
                'dimensions': '[720,1280]',
                'inference': '{"steps":0}',
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn('steps', response.json['error'])


    def test_completed_output_can_be_retrieved_after_restart(self):
        with TemporaryDirectory() as root, patch('app.DATA', Path(root)), app.test_client() as client:
            job_id = 'a' * 32
            folder = Path(root) / job_id
            folder.mkdir()
            (folder / 'output.mp4').write_bytes(b'result')
            (folder / 'config.json').write_text('{"inference":{"steps":5}}', encoding='utf-8')
            self.assertEqual(client.get(f'/api/jobs/{job_id}').json['state'], 'done')
            with client.get(f'/api/jobs/{job_id}/output') as response:
                self.assertEqual(response.data, b'result')
            self.assertEqual(client.get('/api/jobs/invalid').status_code, 404)



class DownloadFilenameTests(unittest.TestCase):
    def test_download_preserves_uploaded_name_after_restart(self):
        cases = [
            ('旅行记录.mp4', '旅行记录_处理后.mp4'),
            ('Episode 01.final.MOV', 'Episode 01.final_处理后.mp4'),
            (r'C:\fakepath\测试.第一集.mp4', '测试.第一集_处理后.mp4'),
        ]
        for filename, expected in cases:
            with self.subTest(filename=filename), TemporaryDirectory() as root, \
                    patch('app.DATA', Path(root)), patch('app.jobs', {}) as saved_jobs, \
                    patch('app.Thread'), patch('app.model_status', return_value=(True, 'ready')), \
                    app.test_client() as client:
                response = client.post('/api/jobs', data={
                    'roi': '[0,0,20,20]',
                    'dimensions': '[40,40]',
                    'video': (io.BytesIO(b'video'), filename.replace(chr(92), chr(92) * 2)),
                })
                self.assertEqual(response.status_code, 202)
                job_id = response.json['id']
                folder = Path(root) / job_id
                config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
                self.assertEqual(config.get('original_filename'), filename)
                (folder / 'output.mp4').write_bytes(b'result')
                saved_jobs.clear()
                with client.get(f'/api/jobs/{job_id}/output') as download:
                    disposition, options = parse_options_header(download.headers['Content-Disposition'])
                    self.assertEqual(disposition, 'attachment')
                    self.assertEqual(options['filename'], expected)

    def test_legacy_download_without_filename_metadata_still_works(self):
        with TemporaryDirectory() as root, patch('app.DATA', Path(root)), app.test_client() as client:
            job_id = 'b' * 32
            folder = Path(root) / job_id
            folder.mkdir()
            (folder / 'output.mp4').write_bytes(b'result')
            with client.get(f'/api/jobs/{job_id}/output') as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, b'result')
                _, options = parse_options_header(response.headers['Content-Disposition'])
                self.assertEqual(options['filename'], 'CLEAR-processed.mp4')

if __name__ == '__main__':
    unittest.main()
