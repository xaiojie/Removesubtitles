import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import deploy_api
import deploy_helper
import test_deploy
import test_deploy_update


def process_identity(pid):
    # The deployment uses Linux /proc. For the macOS integration test, use ps start time and state.
    result = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart='], capture_output=True, text=True, check=False
    )
    parts = result.stdout.strip().split(None, 1)
    return parts[1] if len(parts) == 2 and 'Z' not in parts[0] else None


class ApiDeploymentTests(unittest.TestCase):
    def test_one_key_entry_runs_model_verification_before_web_and_queue_start(self):
        with tempfile.TemporaryDirectory(prefix='fresh api install ') as folder:
            root = Path(folder)
            python, env = test_deploy.DeploymentTests().fake_linux_environment(root)
            lite = root / 'video-subtitle-remover/.venv/bin/python'
            lite.parent.mkdir(parents=True)
            shutil.copy2(python, lite)
            for name in (
                'CLEAR-web/app.py',
                'php/public/index.php',
                'scripts/deploy_api.py',
                'scripts/deploy_helper.py',
                'video-subtitle-remover/deploy/models.json',
                'ProPainter/inference_propainter.py',
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for name in ('CLEAR-web', 'ProPainter'):
                (root / name / 'models.ready').touch()
            # The fake CUDA verifier publishes the same readiness markers as the real helper.
            text = python.read_text().replace(
                'exit 0',
                'case "$*" in *" runtime "*) touch "$TEST_ROOT/CLEAR-web/runtime.ready" "$TEST_ROOT/ProPainter/runtime.ready";; esac\nexit 0',
            )
            python.write_text(text)
            for name, output in [('php', ''), ('ffmpeg', 'libx264'), ('ffprobe', '')]:
                tool = root / 'tools' / name
                tool.write_text(f'#!/bin/sh\necho "{output}"\n')
                tool.chmod(0o755)
            result = subprocess.run(
                [
                    'bash',
                    str(test_deploy.SCRIPT),
                    'all',
                    '--skip-system',
                    '--network',
                    'global',
                    '--yes',
                    '--dir',
                    folder,
                    '--python',
                    str(python),
                ],
                env={**env, 'TEST_ROOT': folder},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / 'commands.log').read_text().splitlines()

            def index(fragment):
                return next(i for i, call in enumerate(calls) if fragment in call)

            self.assertLess(index('torch==2.7.0'), index('download_models.py'))
            self.assertLess(index('download_models.py'), index('lite.py\x1f--verify'))
            self.assertLess(index('lite.py\x1f--verify'), index('deploy_helper.py\x1fstart'))
            self.assertLess(index('deploy_helper.py\x1fstart'), index('deploy_api.py\x1fstart'))

    def test_options_persist_and_offline_lite_uses_its_own_python(self):
        with tempfile.TemporaryDirectory(prefix='api setup ') as folder:
            root = Path(folder)
            python, env = test_deploy.DeploymentTests().fake_linux_environment(root)
            lite = root / 'video-subtitle-remover/.venv/bin/python'
            lite.parent.mkdir(parents=True)
            shutil.copy2(python, lite)
            wheelhouse = root / 'wheels'
            wheelhouse.mkdir()
            result = subprocess.run(
                [
                    'bash',
                    str(test_deploy.SCRIPT),
                    'env',
                    '--with-api',
                    '--api-port',
                    '9200',
                    '--offline',
                    '--wheelhouse',
                    str(wheelhouse),
                    '--dir',
                    folder,
                    '--python',
                    str(python),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = (root / '.deploy/config.sh').read_text()
            self.assertIn('WITH_API=1', config)
            self.assertIn('API_PORT=9200', config)
            calls = (root / 'commands.log').read_text().splitlines()
            lite_install = next(line for line in calls if 'torch==2.7.0' in line)
            self.assertIn('--no-index', lite_install)
            self.assertIn(str(wheelhouse), lite_install)
            result = subprocess.run(
                ['bash', str(test_deploy.SCRIPT), 'models', '--offline', '--dir', folder],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / 'commands.log').read_text().splitlines()
            lite_models = next(line for line in calls if 'download_models.py' in line)
            for value in ['--verify', 'lama', 'mobile']:
                self.assertIn(value, lite_models.split('\x1f'))
            for extra in (['--port', '9200'], ['--components', 'clear']):
                result = subprocess.run(
                    ['bash', str(test_deploy.SCRIPT), 'check', '--dir', folder, *extra],
                    env=env,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)

    @unittest.skipUnless(
        shutil.which('php')
        and subprocess.run(
            ['php', '-r', 'exit(extension_loaded("pdo_sqlite") ? 0 : 1);'], check=False
        ).returncode
        == 0,
        'needs PHP/PDO SQLite',
    )
    def test_real_services_drain_timeout_and_restart(self):
        with tempfile.TemporaryDirectory(prefix='php deploy ') as folder:
            root = Path(folder)
            state = root / '.deploy/api'
            # Only link program files; all jobs and credentials belong to this temporary installation.
            (root / 'php').mkdir()
            for part in ('common.php', 'public', 'bin'):
                (root / 'php' / part).symlink_to(ROOT / 'php' / part)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            args = ['--root', folder, '--port', str(port), '--wait', '0']
            if not sys.platform.startswith('linux'):
                for module in (deploy_api, deploy_helper):
                    p = patch.object(module, 'process_ticks', process_identity)
                    p.start()
                    self.addCleanup(p.stop)
            with patch.dict(os.environ, {'SUBTITLE_DATA_DIR': str(root / 'data')}):
                try:
                    deploy_api.main(['start', *args])
                    pids = [deploy_helper.service_record(state / n)['pid'] for n in deploy_api.SERVICES]
                    deploy_api.main(['start', *args])
                    self.assertEqual(
                        pids, [deploy_helper.service_record(state / n)['pid'] for n in deploy_api.SERVICES]
                    )
                    self.assertEqual((state / 'api.token').stat().st_mode & 0o777, 0o600)
                    url = f'http://127.0.0.1:{port}/api/modes'
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(url, timeout=2)
                    self.assertEqual(error.exception.code, 401)
                    token = (state / 'api.token').read_text()
                    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
                    with urllib.request.urlopen(req, timeout=3) as response:
                        self.assertEqual(json.load(response)['default'], 'pro')
                    with deploy_api.drain(root / 'data', 1, deploy_api.SERVICES):
                        post = urllib.request.Request(
                            url.replace('/modes', '/jobs'), data=b'', headers=req.headers
                        )
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            urllib.request.urlopen(post, timeout=3)
                        self.assertEqual(error.exception.code, 503)
                    db_path = root / 'data/queue.sqlite'
                    with sqlite3.connect(db_path) as db:
                        db.execute(
                            "INSERT INTO jobs(id,name,mode,state,stage,created) VALUES('test','test','pro','running','test',0)"
                        )
                    with self.assertRaisesRegex(RuntimeError, '超时'):
                        deploy_api.main(['stop', *args])
                    self.assertEqual(deploy_api.active(state), list(deploy_api.SERVICES))
                    with sqlite3.connect(db_path) as db:
                        db.execute("UPDATE jobs SET state='done' WHERE id='test'")
                    deploy_api.main(['stop', *args])
                    self.assertFalse(deploy_api.active(state))
                    deploy_api.main(['start', *args])
                    self.assertEqual(deploy_api.active(state), list(deploy_api.SERVICES))
                    self.assertEqual((state / 'api.token').read_text(), token)
                finally:
                    for name in reversed(deploy_api.active(state)):
                        deploy_helper.stop_service(state / name)
                self.assertFalse(deploy_api.active(state))

    def test_update_rolls_back_source_if_new_api_does_not_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = root / '.deploy/api/settings.json'
            settings.parent.mkdir(parents=True)
            settings.write_text(json.dumps({'data': str(root / 'php/var')}))
            opts = test_deploy_update.options(root)
            incoming = test_deploy_update.source_files()
            for name in ('php/public/index.php', 'php/bin/worker.php', 'php/common.php'):
                incoming[name] = '<?php'
            incoming['scripts/deploy_api.py'] = ''
            for name, content in incoming.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            incoming['php/public/index.php'] = '<?php /* changed */'
            test_deploy_update.archive_at(opts.archive, incoming)
            with (
                patch.object(deploy_helper, 'validate_updated_source'),
                patch.object(deploy_helper, 'ensure_no_unmanaged_service'),
                patch.object(deploy_api, 'active', return_value=['worker', 'http']),
                patch.object(deploy_api, 'drain', return_value=nullcontext()),
                patch.object(deploy_api, 'start', side_effect=[RuntimeError('failed'), None]) as start,
                self.assertRaisesRegex(RuntimeError, '恢复'),
            ):
                deploy_helper.update_project(opts)
            self.assertEqual(start.call_count, 2)
            self.assertEqual((root / 'php/public/index.php').read_text(), '<?php')
