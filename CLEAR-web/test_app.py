import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from app import app, region


class WebTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
