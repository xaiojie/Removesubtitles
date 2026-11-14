import os
from pathlib import Path
import subprocess
import sys
import unittest


WEB = Path(__file__).resolve().parents[1] / 'CLEAR-web'


class PublicHostTests(unittest.TestCase):
    def test_monitoring_uses_selected_physical_gpu_or_uuid(self):
        code = '''
import os
from unittest.mock import patch
import app
with patch.object(app.subprocess, 'check_output', return_value='A100, 90, 12000, 40960\\n') as query:
    status = app.gpu_status()
    assert status['name'] == 'A100'
    print(query.call_args.args[0][2])
'''
        for selected, expected in (('3', '3'), ('GPU-example-uuid', 'GPU-example-uuid'), ('3,2', '3'), ('', '0')):
            with self.subTest(selected=selected):
                environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': selected}
                result = subprocess.check_output([sys.executable, '-c', code], cwd=WEB, env=environment, text=True)
                self.assertEqual(result.strip(), expected)

    def test_configured_public_host_is_accepted(self):
        environment = {**os.environ, 'CLEAR_TRUSTED_HOSTS': 'localhost,127.0.0.1,video.example.test'}
        result = subprocess.check_output(
            [sys.executable, '-c', 'import app; print(app.app.test_client().get("/", headers={"Host":"video.example.test"}).status_code)'],
            cwd=WEB, env=environment, text=True,
        )
        self.assertEqual(result.strip(), '200')

    def test_default_rejects_unconfigured_host(self):
        environment = dict(os.environ)
        environment.pop('CLEAR_TRUSTED_HOSTS', None)
        result = subprocess.check_output(
            [sys.executable, '-c', 'import app; print(app.app.test_client().get("/", headers={"Host":"video.example.test"}).status_code)'],
            cwd=WEB, env=environment, text=True,
        )
        self.assertEqual(result.strip(), '400')


if __name__ == '__main__':
    unittest.main()
