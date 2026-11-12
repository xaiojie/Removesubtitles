import unittest
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


if __name__ == '__main__':
    unittest.main()
