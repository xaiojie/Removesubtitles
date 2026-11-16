"""Deployment upgrades use isolated files, local Git fixtures and fake service boundaries."""

import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('deploy_tests', ROOT / 'tests/test_deploy.py')
DEPLOY_TESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY_TESTS)


def options(root, **changes):
    values = dict(root=root, python=Path(sys.executable), propainter_root=root / 'ProPainter',
                  preset='balanced', with_ltx=False, ltx_root=root / 'LTX2-ICEdit-Insight',
                  bind='127.0.0.1', port=8765, public_host='', gpu='', offline=True,
                  archive=root / '.deploy/update.tar.gz', update_ref='main', wait_idle=0)
    values.update(changes)
    return SimpleNamespace(**values)


def archive_at(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as archive:
        for name, content in files.items():
            data = content.encode()
            member = tarfile.TarInfo('project/' + name)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))


def source_files(app='VALUE = "new"\n'):
    return {'CLEAR-web/app.py': app, 'scripts/deploy.sh': '#!/bin/bash\n',
            'scripts/deploy_helper.py': 'VALUE = "new helper"\n'}


class DeploymentConfigurationTests(unittest.TestCase):
    fake_linux_environment = DEPLOY_TESTS.DeploymentTests.fake_linux_environment
    # Reuse the fake Linux boundary without repeating its existing tests here.
    def test_preset_and_ltx_options_persist_across_repeat_runs(self):
        with tempfile.TemporaryDirectory(prefix='saved deployment ') as directory:
            root = Path(directory)
            python, environment = self.fake_linux_environment(root)
            (root / 'scripts/deploy_ltx.py').touch()
            result = subprocess.run(['bash', str(DEPLOY_TESTS.SCRIPT), 'env', '--dir', str(root),
                                     '--python', str(python), '--network', 'global', '--cuda', 'cu128', '--preset', 'quality',
                                     '--with-ltx', '--ltx-root', str(root / 'custom LTX'), '--ltx-tasks', 'both'],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(['bash', str(DEPLOY_TESTS.SCRIPT), 'env', '--dir', str(root)],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = subprocess.check_output(['bash', '-c', 'source "$1"; printf "%s|%s|%s|%s" "$PRESET" "$WITH_LTX" "$LTX_DIR" "$LTX_TASKS"',
                                              'config', str(root / '.deploy/config.sh')], text=True)
            self.assertEqual(config, f'quality|1|{root / "custom LTX"}|both')
            calls = (root / 'commands.log').read_text()
            self.assertIn('deploy_ltx.py\x1fenv\x1f', calls)

    def test_legacy_config_gains_balanced_without_losing_saved_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python, environment = self.fake_linux_environment(root)
            state = root / '.deploy'
            state.mkdir()
            (state / 'config.sh').write_text(f'# Local deployment settings\nLOCAL_KEEP=custom\nCOMPONENTS=clear\nPORT=9001\nNETWORK=global\nPYTHON_BIN={python}\n')
            result = subprocess.run(['bash', str(DEPLOY_TESTS.SCRIPT), 'env', '--dir', str(root)],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = (state / 'config.sh').read_text()
            self.assertIn('PRESET=balanced\n', config)
            self.assertIn('PORT=9001\n', config)
            self.assertIn('COMPONENTS=clear\n', config)
            self.assertIn('# Local deployment settings\nLOCAL_KEEP=custom\n', config)
            self.assertNotIn('deploy_ltx.py', (root / 'commands.log').read_text())

    def test_invalid_new_options_fail_before_installation(self):
        for arguments in (['--preset', 'unknown'], ['--ltx-tasks', 'unknown'], ['--wait-idle', '-1']):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'uncreated'
                result = subprocess.run(['bash', str(DEPLOY_TESTS.SCRIPT), 'check', '--dir', str(root), *arguments],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(root.exists())

    def test_current_installer_upgrades_installation_with_legacy_helper(self):
        with tempfile.TemporaryDirectory(prefix='legacy archive installation ') as directory:
            root = Path(directory)
            python, environment = self.fake_linux_environment(root)
            python.unlink()
            python.symlink_to(sys.executable)
            (root / 'CLEAR-web').mkdir()
            (root / 'CLEAR-web/app.py').write_text('VALUE = "old"\n')
            (root / 'scripts/deploy_helper.py').write_text(
                'import sys\n'
                'if "--help" in sys.argv:\n'
                '    print("legacy helper help")\n'
                '    raise SystemExit(0)\n'
                'raise RuntimeError("legacy helper cannot update")\n')
            archive = root / 'new-source.tar.gz'
            incoming = source_files()
            incoming['scripts/deploy_helper.py'] = DEPLOY_TESTS.HELPER.read_text()
            archive_at(archive, incoming)
            installer = root / 'scripts/deploy.sh'
            installer.write_text(DEPLOY_TESTS.SCRIPT.read_text())
            # Reserve an unused local port without starting a listener or contacting a real service.
            with socket.socket() as reserved:
                reserved.bind(('127.0.0.1', 0))
                result = subprocess.run(['bash', str(installer), 'update', '--dir', str(root),
                                         '--update-archive', str(archive), '--offline',
                                         '--port', str(reserved.getsockname()[1])],
                                        capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'CLEAR-web/app.py').read_text(), 'VALUE = "new"\n')

    def test_ltx_enabled_all_routes_every_stage_and_requires_readiness(self):
        with tempfile.TemporaryDirectory(prefix='all optional stages ') as directory:
            root = Path(directory)
            python, environment = self.fake_linux_environment(root)
            web = root / 'CLEAR-web'
            web.mkdir()
            for name in ('app.py', 'models.ready', 'runtime.ready'):
                (web / name).touch()
            (root / 'scripts/deploy_ltx.py').touch()
            ltx = root / 'LTX2-ICEdit-Insight'
            (ltx / '.venv/bin').mkdir(parents=True)
            (ltx / '.venv/bin/python').symlink_to(python)
            (ltx / '.install').mkdir()
            (ltx / '.install/runtime.complete').touch()
            wheels = root / 'wheels'
            wheels.mkdir()
            result = subprocess.run(['bash', str(DEPLOY_TESTS.SCRIPT), 'all', '--dir', str(root),
                                     '--components', 'clear', '--with-ltx', '--python', str(python),
                                     '--network', 'global', '--cuda', 'cu128', '--offline',
                                     '--wheelhouse', str(wheels), '--yes'], capture_output=True, text=True, env=environment)
            # The fake runtime validator has no side effects. Its pre-existing readiness is cleared by env.
            self.assertIn('CLEAR 尚未就绪', result.stderr)
            calls = (root / 'commands.log').read_text()
            for stage in ('source', 'env', 'models', 'verify'):
                self.assertIn(f'deploy_ltx.py\x1f{stage}\x1f', calls)
            (web / 'runtime.ready').touch()
            (ltx / '.install/runtime.complete').unlink()
            result = subprocess.run(['bash', str(DEPLOY_TESTS.SCRIPT), 'start', '--dir', str(root)],
                                    capture_output=True, text=True, env=environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('LTX 尚未就绪', result.stderr)
            (ltx / '.install/runtime.complete').touch()
            result = subprocess.run(['bash', str(DEPLOY_TESTS.SCRIPT), 'start', '--dir', str(root)],
                                    capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)


class DeploymentUpgradeTests(unittest.TestCase):
    def helper(self):
        helper = DEPLOY_TESTS.load_helper()
        self.assertTrue(callable(getattr(helper, 'update_project', None)), 'explicit update command is missing')
        guard = patch.object(helper, 'ensure_no_unmanaged_service')
        guard.start()
        self.addCleanup(guard.stop)
        return helper

    def test_archive_upgrade_preserves_data_models_config_environments_and_unknown_files(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained = {'CLEAR-web/jobs/a/source.video': 'video', 'CLEAR-models/model.bin': 'weights',
                        '.venv/bin/python': 'environment', '.deploy/config.sh': 'PORT=9001',
                        'CLEAR-web/config.local.json': 'local', 'CLEAR-web/custom.py': 'local source'}
            for name, value in retained.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            (root / 'CLEAR-web/app.py').write_text('VALUE = "old"\n')
            incoming = source_files()
            incoming.update({name: 'must stay local' for name in retained if name != 'CLEAR-web/custom.py'})
            opts = options(root)
            archive_at(opts.archive, incoming)
            with patch.object(helper, 'service_record', return_value=None):
                helper.update_project(opts)
            self.assertEqual((root / 'CLEAR-web/app.py').read_text(), 'VALUE = "new"\n')
            for name, value in retained.items():
                self.assertEqual((root / name).read_text(), value, name)

    def test_failed_validation_restores_previous_source(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CLEAR-web').mkdir()
            (root / 'CLEAR-web/app.py').write_text('VALUE = "old"\n')
            opts = options(root)
            archive_at(opts.archive, source_files('raise RuntimeError("missing dependency")\n'))
            with patch.object(helper, 'service_record', return_value=None), self.assertRaisesRegex(RuntimeError, '保留|还原|恢复'):
                helper.update_project(opts)
            self.assertEqual((root / 'CLEAR-web/app.py').read_text(), 'VALUE = "old"\n')
            self.assertFalse((root / 'scripts/deploy_helper.py').exists())

    def test_unsafe_archive_fails_before_service_is_stopped(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CLEAR-web').mkdir()
            (root / 'CLEAR-web/app.py').write_text('VALUE = 1\n')
            opts = options(root)
            archive_at(opts.archive, {**source_files(), '../escape.py': 'escape'})
            with patch.object(helper, 'stop_service') as stop, self.assertRaises(ValueError):
                helper.update_project(opts)
            stop.assert_not_called()
            self.assertFalse((root / 'escape.py').exists())

    def test_busy_or_uninspectable_service_keeps_source_and_service_running(self):
        helper = self.helper()
        for failure in ('任务仍在处理', '旧版服务无法查询任务'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'CLEAR-web').mkdir()
                (root / 'CLEAR-web/app.py').write_text('VALUE = "old"\n')
                opts = options(root)
                archive_at(opts.archive, source_files())
                with patch.object(helper, 'service_record', return_value={'pid': 123, 'port': 8765}), \
                     patch.object(helper, 'wait_for_idle', side_effect=RuntimeError(failure)), \
                     patch.object(helper, 'stop_service') as stop, self.assertRaises(RuntimeError):
                    helper.update_project(opts)
                stop.assert_not_called()
                self.assertEqual((root / 'CLEAR-web/app.py').read_text(), 'VALUE = "old"\n')

    def test_failed_restart_restores_code_and_restarts_previous_service(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CLEAR-web').mkdir()
            (root / 'CLEAR-web/app.py').write_text('VALUE = "old"\n')
            opts = options(root)
            archive_at(opts.archive, source_files())
            started_code = []

            def start(_):
                started_code.append((root / 'CLEAR-web/app.py').read_text())
                if len(started_code) == 1:
                    raise RuntimeError('new server failed')

            with patch.object(helper, 'service_record', return_value={'pid': 123, 'port': 8765}), \
                 patch.object(helper, 'wait_for_idle'), patch.object(helper, 'stop_service', return_value=True), \
                 patch.object(helper, 'start_service', side_effect=start), self.assertRaisesRegex(RuntimeError, '恢复|还原'):
                helper.update_project(opts)
            self.assertEqual(started_code, ['VALUE = "new"\n', 'VALUE = "old"\n'])

    def test_failed_manifest_publish_rolls_back_source(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CLEAR-web').mkdir()
            (root / 'CLEAR-web/app.py').write_text('VALUE = "old"\n')
            opts = options(root)
            archive_at(opts.archive, source_files())
            replace = helper.os.replace

            def publish(source, target):
                if Path(target).name == 'source-files.json':
                    raise OSError('manifest disk write failed')
                return replace(source, target)

            with patch.object(helper, 'service_record', return_value=None), \
                 patch.object(helper, 'ensure_no_unmanaged_service'), \
                 patch.object(helper.os, 'replace', side_effect=publish), self.assertRaisesRegex(RuntimeError, '恢复'):
                helper.update_project(opts)
            self.assertEqual((root / 'CLEAR-web/app.py').read_text(), 'VALUE = "old"\n')

    def test_upgrade_does_not_write_through_existing_source_symlink(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / 'local-original.py'
            outside.write_text('VALUE = "local"\n')
            (root / 'CLEAR-web').mkdir()
            (root / 'CLEAR-web/app.py').symlink_to(outside)
            opts = options(root)
            archive_at(opts.archive, source_files())
            with self.assertRaisesRegex(RuntimeError, '符号链接'):
                helper.update_project(opts)
            self.assertEqual(outside.read_text(), 'VALUE = "local"\n')

    def test_git_fast_forward_updates_head_and_removes_obsolete_managed_code(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.STDOUT).strip()
            git('init', '-b', 'main')
            git('config', 'user.email', 'test@example.test')
            git('config', 'user.name', 'Deployment test')
            for name, value in {**source_files('VALUE = "old"\n'), 'CLEAR-web/obsolete.py': 'old = True\n'}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            git('add', '.')
            git('commit', '-m', 'old fixture')
            old = git('rev-parse', 'HEAD')
            (root / 'CLEAR-web/app.py').write_text('VALUE = "new"\n')
            (root / 'CLEAR-web/obsolete.py').unlink()
            git('add', '-A')
            git('commit', '-m', 'new fixture')
            new = git('rev-parse', 'HEAD')
            git('reset', '--hard', old)
            with patch.object(helper, 'service_record', return_value=None):
                helper.update_project(options(root, archive=None, update_ref=new))
            self.assertEqual(git('rev-parse', 'HEAD'), new)
            self.assertEqual(git('status', '--porcelain', '--untracked-files=no'), '')
            self.assertFalse((root / 'CLEAR-web/obsolete.py').exists())

            # A second candidate introduces a new file. A restart failure must undo HEAD, index and files.
            git('reset', '--hard', old)
            (root / 'CLEAR-web/added.py').write_text('new = True\n')
            git('add', '.')
            git('commit', '-m', 'another fixture')
            candidate = git('rev-parse', 'HEAD')
            git('reset', '--hard', old)
            with patch.object(helper, 'service_record', return_value={'pid': 123, 'port': 8765}), \
                 patch.object(helper, 'wait_for_idle'), patch.object(helper, 'stop_service', return_value=True), \
                 patch.object(helper, 'start_service', side_effect=[RuntimeError('restart failed'), None]), \
                 self.assertRaisesRegex(RuntimeError, '恢复'):
                helper.update_project(options(root, archive=None, update_ref=candidate))
            self.assertEqual(git('rev-parse', 'HEAD'), old)
            self.assertEqual(git('status', '--porcelain', '--untracked-files=no'), '')
            self.assertFalse((root / 'CLEAR-web/added.py').exists())

            (root / 'CLEAR-web/app.py').write_text('VALUE = "local modifications"\n')
            with self.assertRaisesRegex(RuntimeError, '本地修改'):
                helper.update_project(options(root, archive=None, update_ref=candidate))
            self.assertEqual((root / 'CLEAR-web/app.py').read_text(), 'VALUE = "local modifications"\n')

    def test_drain_timeout_releases_gate_and_does_not_stop_service(self):
        helper = self.helper()
        opts = options(Path('/unused'))
        with patch.object(helper, 'drain_request', return_value={'draining': True, 'busy': [{'id': 'a', 'state': 'queued', 'kind': 'job'}]}) as request, \
             patch.object(helper, 'stop_service') as stop, self.assertRaisesRegex(RuntimeError, 'a|queued'):
            helper.wait_for_idle(opts, {'bind': '127.0.0.1', 'port': 8765})
        self.assertEqual([call.args[2] for call in request.call_args_list], ['POST', 'DELETE'])
        stop.assert_not_called()

    def test_start_passes_processing_preset_and_independent_ltx_runtime(self):
        helper = DEPLOY_TESTS.load_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CLEAR-web').mkdir()
            opts = options(root, preset='quality', with_ltx=True, ltx_root=root / 'separate LTX')
            process = SimpleNamespace(pid=123, poll=lambda: None)
            response = SimpleNamespace(status=200)
            with patch.object(helper, 'service_record', return_value=None), \
                 patch.object(helper.socket, 'socket'), patch.object(helper, 'process_ticks', return_value='ticks'), \
                 patch.object(helper.subprocess, 'Popen', return_value=process) as launch, \
                 patch.object(helper.urllib.request, 'build_opener') as opener:
                opener.return_value.open.return_value.__enter__.return_value = response
                helper.start_service(opts)
            environment = launch.call_args.kwargs['env']
            self.assertEqual(environment['CLEAR_PRESET'], 'quality')
            self.assertEqual(environment['LTX_ROOT'], str(root / 'separate LTX'))
            self.assertEqual(environment['LTX_PYTHON'], str(root / 'separate LTX/.venv/bin/python'))
            saved = json.loads((root / '.deploy/server.json').read_text())
            self.assertEqual(saved['settings']['preset'], 'quality')

    def test_drain_uses_private_token_and_bypasses_proxy(self):
        helper = self.helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(helper.urllib.request, 'build_opener') as opener:
                stream = io.StringIO('{"draining":true,"busy":[]}')
                opener.return_value.open.return_value.__enter__.return_value = stream
                status = helper.drain_request(options(root), {'bind': '0.0.0.0', 'port': 9999}, 'POST')
            self.assertEqual(status, {'draining': True, 'busy': []})
            token_path = root / '.deploy/maintenance.token'
            self.assertGreaterEqual(len(token_path.read_text()), 32)
            self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.full_url, 'http://127.0.0.1:9999/api/deploy/drain')
            self.assertEqual(request.get_header('X-deploy-token'), token_path.read_text())


if __name__ == '__main__':
    unittest.main()
