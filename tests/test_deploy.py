import contextlib
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import io
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/deploy.sh'
HELPER = ROOT / 'scripts/deploy_helper.py'
PAYLOAD = b'verified-model-' * 4096


def load_helper():
    assert HELPER.is_file(), 'Deployment helper has not been implemented'
    spec = importlib.util.spec_from_file_location('deployment_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def download_server():
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get('Range')))
            if self.path == '/unavailable':
                self.send_error(503)
                return
            if self.path in ('/short', '/short-no-length'):
                # Bound the fixture so the old downloader fails rather than hanging the suite.
                if sum(path == self.path for path, _ in requests) > 3:
                    self.send_error(503)
                    return
                self.send_response(200)
                if self.path == '/short':
                    self.send_header('Content-Length', '100')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(PAYLOAD[:100])
                self.close_connection = True
                return
            data = b'x' * len(PAYLOAD) if self.path == '/corrupt' else PAYLOAD
            offset = 0
            if self.path != '/ignore' and self.headers.get('Range'):
                offset = int(self.headers['Range'].split('=')[1].split('-')[0])
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {offset}-{len(data)-1}/{len(data)}')
            else:
                self.send_response(200)
            self.send_header('Content-Length', str(len(data) - offset))
            self.end_headers()
            if self.path == '/interrupted' and not self.headers.get('Range'):
                self.wfile.write(data[:2345])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(data[offset:])

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class DeploymentTests(unittest.TestCase):
    def fake_linux_environment(self, root):
        tools = root / 'tools'
        tools.mkdir()
        commands = {
            'uname': '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; *) echo Linux;; esac\n',
            'getconf': '#!/bin/sh\necho "glibc 2.35"\n',
            'flock': '#!/bin/sh\nexit 0\n',
            'nvidia-smi': '#!/bin/sh\ncase "$*" in *compute_cap*) echo 8.0;; *) echo "CUDA Version: 12.8";; esac\n',
        }
        for name, content in commands.items():
            path = tools / name
            path.write_text(content)
            path.chmod(0o755)
        (root / 'scripts').mkdir()
        (root / 'scripts/requirements-linux.txt').touch()
        (root / 'scripts/requirements-propainter.txt').touch()
        python = root / '.venv/bin/python'
        python.parent.mkdir(parents=True)
        python.write_text('#!/bin/sh\nprintf "%s\\037" "$@" >> "$TEST_COMMAND_LOG"\nprintf "\\n" >> "$TEST_COMMAND_LOG"\ncase "$*" in *"print(torch.version.cuda or"*) echo 12.8;; esac\nexit 0\n')
        python.chmod(0o755)
        return python, {**os.environ, 'PATH': str(tools) + os.pathsep + os.environ['PATH'], 'TEST_COMMAND_LOG': str(root / 'commands.log')}

    def test_cuda_override_selects_requested_index_with_space_in_path(self):
        with tempfile.TemporaryDirectory(prefix='deployment with spaces ') as directory:
            root = Path(directory)
            python, environment = self.fake_linux_environment(root)
            result = subprocess.run(
                ['bash', str(SCRIPT), 'env', '--dir', str(root), '--python', str(python), '--network', 'global', '--cuda', 'cu126', '--gpu', '3'],
                capture_output=True, text=True, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / 'commands.log').read_text()
            self.assertIn('https://download.pytorch.org/whl/cu126', calls)
            self.assertIn('uninstall\x1f-y\x1ftorch\x1ftorchvision', calls)
            config = (root / '.deploy/config.sh').read_text()
            self.assertIn('GPU=3', config)

    def test_offline_environment_uses_only_wheelhouse(self):
        with tempfile.TemporaryDirectory(prefix='offline installation ') as directory:
            root = Path(directory)
            python, environment = self.fake_linux_environment(root)
            wheels = root / 'local wheels'
            wheels.mkdir()
            result = subprocess.run(
                ['bash', str(SCRIPT), 'env', '--dir', str(root), '--python', str(python), '--network', 'global', '--cuda', 'cu128', '--offline', '--wheelhouse', str(wheels)],
                capture_output=True, text=True, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [line.split('\x1f') for line in (root / 'commands.log').read_text().splitlines()]
            pip_installs = [call for call in calls if call[:3] == ['-m', 'pip', 'install']]
            self.assertTrue(pip_installs)
            for call in pip_installs:
                self.assertIn('--no-index', call)
                self.assertIn(str(wheels), call)
                self.assertNotIn('--index-url', call)

    def test_help_does_not_create_installation_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'new installation'
            result = subprocess.run(['bash', str(SCRIPT), '--dir', str(target), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--offline', result.stdout)
            self.assertFalse(target.exists())

    def test_invalid_port_fails_before_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'new installation'
            result = subprocess.run(['bash', str(SCRIPT), '--dir', str(target), '--port', '0', 'check'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(target.exists())

    def test_unknown_option_fails(self):
        result = subprocess.run(['bash', str(SCRIPT), '--unknown-option'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def download(self, endpoint, partial=False):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory, download_server() as (base, requests):
            root = Path(directory)
            destination = root / 'models/test.bin'
            if partial:
                destination.parent.mkdir()
                destination.with_name('test.bin.part').write_bytes(PAYLOAD[:1234])
            asset = {'path': 'models/test.bin', 'size': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [base + endpoint]}
            helper.download_asset(asset, root, retries=2, timeout=5)
            self.assertEqual(destination.read_bytes(), PAYLOAD)
            self.assertFalse(destination.with_name('test.bin.part').exists())
            return requests

    def test_download_resumes_existing_prefix(self):
        self.assertIn(('/range', 'bytes=1234-'), self.download('/range', partial=True))

    def test_ignored_range_restarts_safely(self):
        self.download('/ignore', partial=True)

    def test_interrupted_response_resumes_remaining_bytes(self):
        requests = self.download('/interrupted')
        self.assertIn(('/interrupted', 'bytes=2345-'), requests)

    def test_unavailable_source_falls_back(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory, download_server() as (base, _):
            asset = {'path': 'test.bin', 'size': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [base + '/unavailable', base + '/range']}
            with patch.object(helper.time, 'sleep'):
                helper.download_asset(asset, Path(directory), retries=1, timeout=5)
            self.assertEqual((Path(directory) / 'test.bin').read_bytes(), PAYLOAD)

    def test_short_full_responses_fall_back_with_bounded_requests(self):
        helper = load_helper()
        for endpoint, maximum in (('/short', 1), ('/short-no-length', 2)):
            with self.subTest(endpoint=endpoint):
                with tempfile.TemporaryDirectory() as directory, download_server() as (base, requests):
                    asset = {'path': 'test.bin', 'size': len(PAYLOAD),
                             'sha256': hashlib.sha256(PAYLOAD).hexdigest(),
                             'urls': [base + endpoint, base + '/range']}
                    helper.download_asset(asset, Path(directory), retries=1, timeout=5)
                    self.assertEqual((Path(directory) / 'test.bin').read_bytes(), PAYLOAD)
                    self.assertLessEqual(sum(path == endpoint for path, _ in requests), maximum)

    def test_bad_checksum_never_publishes_model(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory, download_server() as (base, _):
            asset = {'path': 'test.bin', 'size': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [base + '/corrupt']}
            with patch.object(helper.time, 'sleep'), self.assertRaises(RuntimeError):
                helper.download_asset(asset, Path(directory), retries=1, timeout=5)
            self.assertFalse((Path(directory) / 'test.bin').exists())

    def test_offline_verification_avoids_network(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory, download_server() as (base, requests):
            root = Path(directory)
            (root / 'test.bin').write_bytes(PAYLOAD)
            asset = {'path': 'test.bin', 'size': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [base + '/range']}
            helper.download_asset(asset, root, offline=True)
            self.assertEqual(requests, [])

    def test_offline_missing_file_fails_without_network(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory, download_server() as (base, requests):
            asset = {'path': 'test.bin', 'size': len(PAYLOAD), 'sha256': hashlib.sha256(PAYLOAD).hexdigest(), 'urls': [base + '/range']}
            with self.assertRaises(RuntimeError):
                helper.download_asset(asset, Path(directory), offline=True)
            self.assertEqual(requests, [])

    def test_asset_cannot_escape_installation_directory(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                helper.download_asset({'path': '../escape.bin'}, Path(directory), offline=True)

    def test_failed_verification_removes_old_readiness_marker(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'CLEAR-web/models.ready'
            marker.parent.mkdir()
            marker.write_text('old verified marker')
            options = SimpleNamespace(root=root, components='clear', workers=1, retries=1, timeout=1)
            asset = {'path': 'missing.bin', 'size': 1, 'sha256': 'bad', 'urls': []}
            with patch.object(helper, 'load_assets', return_value=[(asset, root)]), self.assertRaises(RuntimeError):
                helper.models(options, offline=True)
            self.assertFalse(marker.exists())

    def test_source_archive_cannot_escape_target(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'source.tar.gz'
            with tarfile.open(archive, 'w:gz') as output:
                member = tarfile.TarInfo('upstream/../../escape.py')
                member.size = 4
                output.addfile(member, io.BytesIO(b'pass'))
            with self.assertRaises(ValueError):
                helper.extract_propainter(archive, root / 'target')
            self.assertFalse((root / 'escape.py').exists())

    def test_stale_process_record_preserves_other_process(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen(['sleep', '30'], start_new_session=True)
            state = Path(directory)
            try:
                (state / 'server.json').write_text(json.dumps({'pid': process.pid, 'start_ticks': 'wrong'}))
                self.assertFalse(helper.stop_service(state))
                self.assertIsNone(process.poll())
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
