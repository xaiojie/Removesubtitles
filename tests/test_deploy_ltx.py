import ast
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/deploy_ltx.py'


def load_installer():
    spec = importlib.util.spec_from_file_location('ltx_installer', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(root, **overrides):
    values = dict(root=Path(root), python='/usr/bin/python3', network='cn',
                  hf_endpoint='https://hf-mirror.com', github_prefix='', workers=2,
                  tasks='subtitle', offline=False, wheelhouse=None, pip_index=None,
                  torch_index='https://download.pytorch.org/whl/cu128', gpu='0')
    return SimpleNamespace(**{**values, **overrides})


def asset(path, payload, **extra):
    return dict(path=path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                urls=['https://huggingface.co/test/model/resolve/revision/' + Path(path).name],
                task='shared', **extra)


def write_tensor(path, name='model.embed_tokens.weight'):
    header = json.dumps({name: dict(dtype='F32', shape=[1], data_offsets=[0, 4])}).encode()
    header += b' ' * (-len(header) % 8)
    payload = struct.pack('<Q', len(header)) + header + struct.pack('<f', 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def archive_bytes(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        for name, payload, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == 'file':
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                member.type = kind
                member.linkname = '../outside'
                archive.addfile(member)
    return data.getvalue()


@contextlib.contextmanager
def download_server(payload):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get('Range')))
            if self.path == '/missing':
                self.send_error(404)
                return
            offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
            self.send_response(206 if offset else 200)
            if offset:
                self.send_header('Content-Range', f'bytes {offset}-{len(payload) - 1}/{len(payload)}')
            self.send_header('Content-Length', str(len(payload) - offset))
            self.end_headers()
            self.wfile.write(payload[offset:])

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class LTXInstallerTests(unittest.TestCase):
    def setUp(self):
        self.installer = load_installer()

    def test_cli_defaults_to_subtitle_and_separate_environment(self):
        args = self.installer.parse_args(['env', '--root', '/tmp/ltx root', '--python', '/tmp/clear/bin/python'])
        self.assertEqual(args.tasks, 'subtitle')
        self.assertEqual(args.python, '/tmp/clear/bin/python')
        self.assertEqual(self.installer.venv_python(args.root), Path('/tmp/ltx root').resolve() / '.venv/bin/python')

    def test_tasks_select_only_shared_and_requested_loras(self):
        manifest = self.installer.read_manifest()
        with tempfile.TemporaryDirectory() as directory:
            selected = self.installer.selected_assets(manifest, options(directory))
            self.assertTrue(any(item['role'] == 'checkpoint' for item in selected))
            self.assertEqual({item['task'] for item in selected}, {'shared', 'subtitle'})
            self.assertTrue(any(item['role'] == 'gemma_weights' for item in selected))
            self.assertFalse(any('upscal' in item['path'] or 'restoration' in item['path'] for item in selected))
            both = self.installer.selected_assets(manifest, options(directory, tasks='both'))
            self.assertEqual(len(both), len(selected) + 1)
            for item in both:
                self.assertRegex(item['sha256'], r'^[0-9a-f]{64}$')
                self.assertGreater(item['size'], 0)
                self.assertTrue(all('/resolve/main/' not in url for url in item['urls']))

    def test_mirrors_fall_back_to_pinned_original_without_mutating_manifest(self):
        item = asset('models/checkpoint.bin', b'model')
        original = list(item['urls'])
        adjusted = self.installer.with_download_urls(item, options('/tmp/ltx'))
        self.assertTrue(adjusted['urls'][0].startswith('https://hf-mirror.com/'))
        self.assertEqual(adjusted['urls'][-1], original[0])
        self.assertEqual(item['urls'], original)
        global_urls = self.installer.with_download_urls(item, options('/tmp/ltx', network='global'))['urls']
        self.assertEqual(global_urls[0], original[0])

    def test_source_extraction_rejects_unsafe_members_before_any_write(self):
        for name, kind in [('repo/../outside', 'file'), ('repo/link', tarfile.SYMTYPE),
                           ('repo/hardlink', tarfile.LNKTYPE), ('repo/device', tarfile.CHRTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'installation'
                archive = Path(directory) / 'source.tar.gz'
                archive.write_bytes(archive_bytes([('repo/run_pipeline.py', b'ok', 'file'), (name, b'bad', kind)]))
                with self.assertRaises(ValueError):
                    self.installer.extract_source(archive, root, [asset('run_pipeline.py', b'ok')])
                self.assertFalse((root / 'run_pipeline.py').exists())

    def test_source_rejects_symlink_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root, outside = Path(directory) / 'root', Path(directory) / 'outside'
            root.mkdir()
            outside.mkdir()
            (root / 'ltx_vendor').symlink_to(outside, target_is_directory=True)
            archive = Path(directory) / 'source.tar.gz'
            archive.write_bytes(archive_bytes([('repo/ltx_vendor/test.py', b'code', 'file')]))
            with self.assertRaises(ValueError):
                self.installer.extract_source(archive, root, [asset('ltx_vendor/test.py', b'code')])
            self.assertFalse((outside / 'test.py').exists())

    def test_offline_source_reuses_verified_files_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run_pipeline.py').write_bytes(b'code')
            manifest = dict(source=dict(revision='fixed', files=[asset('run_pipeline.py', b'code')]))
            with patch.object(self.installer.downloader, 'download_asset') as download:
                self.installer.source(options(root, offline=True), manifest)
            download.assert_not_called()
            self.assertTrue((root / '.install/source.complete').is_file())

    def test_source_archive_download_resume_fallback_and_offline_reuse(self):
        content = archive_bytes([('repo/run_pipeline.py', b'code', 'file')])
        with tempfile.TemporaryDirectory() as directory, download_server(content) as (base, requests):
            root = Path(directory)
            archive = asset('.install/source.tar.gz', content)
            archive['urls'] = [base + '/missing', base + '/source']
            partial = root / '.install/source.tar.gz.part'
            partial.parent.mkdir()
            partial.write_bytes(content[:20])
            manifest = dict(source=dict(revision='fixed', archive=archive, files=[asset('run_pipeline.py', b'code')]))
            with patch.object(self.installer.downloader.time, 'sleep'):
                self.installer.source(options(root), manifest)
            self.assertIn(('/source', 'bytes=20-'), requests)
            self.assertEqual((root / 'run_pipeline.py').read_bytes(), b'code')
            (root / 'run_pipeline.py').unlink()
            self.installer.source(options(root, offline=True), manifest)
            self.assertEqual((root / 'run_pipeline.py').read_bytes(), b'code')

    def test_models_remove_stale_readiness_and_publish_only_verified_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / '.install'
            marker.mkdir()
            for name in ['models.complete', 'runtime.complete', 'test.complete']:
                (marker / name).touch()
            manifest = dict(assets=[asset('models/shared.bin', b'shared')])
            with self.assertRaises(RuntimeError):
                self.installer.models(options(root, offline=True), manifest)
            self.assertFalse(any((marker / name).exists() for name in ['models.complete', 'runtime.complete', 'test.complete']))
            (root / 'models').mkdir()
            (root / 'models/shared.bin').write_bytes(b'shared')
            self.installer.models(options(root, offline=True), manifest)
            state = json.loads((marker / 'models.complete').read_text())
            self.assertEqual(state['tasks'], ['subtitle'])
            self.assertEqual(json.loads((marker / 'selection.json').read_text())['tasks'], ['subtitle'])

    def test_models_downloads_fixture_and_checks_disk_before_network(self):
        payload = b'fixture weights' * 64
        with tempfile.TemporaryDirectory() as directory, download_server(payload) as (base, requests):
            item = asset('models/model.bin', payload)
            item['urls'] = [base + '/model']
            manifest = dict(assets=[item])
            with patch.object(self.installer.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError, '空间'):
                    self.installer.models(options(directory), manifest)
            self.assertEqual(requests, [])
            self.installer.models(options(directory), manifest)
            self.assertEqual((Path(directory) / item['path']).read_bytes(), payload)

    def test_environment_bootstraps_own_venv_and_keeps_offline_pip_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'ltx root'
            wheelhouse = Path(directory) / 'wheelhouse'
            wheelhouse.mkdir()
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                if command[1:3] == ['-m', 'venv']:
                    python = root / '.venv/bin/python'
                    python.parent.mkdir(parents=True)
                    python.touch()
                output = json.dumps([str(root / '.venv'), '/base']) if 'sys.base_prefix' in command[-1] else '[3, 12]'
                return subprocess.CompletedProcess(command, 0, stdout=output + '\n')

            with patch.object(self.installer.subprocess, 'run', side_effect=run):
                self.installer.environment(options(root, offline=True, wheelhouse=wheelhouse), {})
            self.assertIn(['/usr/bin/python3', '-m', 'venv', str(root / '.venv')], calls)
            installs = [call for call in calls if call[1:4] == ['-m', 'pip', 'install']]
            self.assertTrue(installs)
            for call in installs:
                self.assertEqual(call[0], str(root / '.venv/bin/python'))
                self.assertIn('--no-index', call)
                self.assertIn(str(wheelhouse), call)
                self.assertNotIn('--index-url', call)
            self.assertTrue((root / '.install/environment.complete').exists())

    def test_environment_replaces_cpu_or_wrong_cuda_and_reuses_matching_stack(self):
        for cuda, replace in [(None, True), ('12.6', True), ('12.8', False)]:
            with self.subTest(cuda=cuda), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                python = root / '.venv/bin/python'
                python.parent.mkdir(parents=True)
                python.touch()

                def run(command, **kwargs):
                    output = '[3, 12]'
                    if 'sys.base_prefix' in command[-1]:
                        output = json.dumps([str(root / '.venv'), '/base'])
                    if 'LTX_TORCH_STATUS' in command[-1]:
                        output = 'LTX_TORCH_STATUS ' + json.dumps(dict(cuda=cuda, torch='2.8.0',
                                                                 torchvision='0.23.0', torchaudio='2.8.0'))
                    return subprocess.CompletedProcess(command, 0, stdout=output + '\n')

                with patch.object(self.installer.subprocess, 'run', side_effect=run) as execute:
                    self.installer.environment(options(root), {})
                torch_call = next(call.args[0] for call in execute.call_args_list
                                  if 'torch==2.8.0' in call.args[0])
                self.assertEqual('--force-reinstall' in torch_call, replace)

    def test_environment_rejects_interpreter_outside_own_venv_before_pip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / '.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()

            def run(command, **kwargs):
                output = '["/clear/.venv", "/base"]' if 'sys.base_prefix' in command[-1] else '[3, 12]'
                return subprocess.CompletedProcess(command, 0, stdout=output)

            with patch.object(self.installer.subprocess, 'run', side_effect=run) as execute:
                with self.assertRaisesRegex(RuntimeError, '独立虚拟环境'):
                    self.installer.environment(options(root), {})
            self.assertFalse(any('-m' in call.args[0] for call in execute.call_args_list))

    def test_install_lock_prevents_concurrent_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.installer.install_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, '正在运行'):
                    with self.installer.install_lock(Path(directory)):
                        self.fail('Concurrent stage acquired the same lock')

    def test_safetensors_structure_rejects_truncated_and_invalid_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            payload = write_tensor(path)
            self.assertIn('model.embed_tokens.weight', self.installer.read_tensor_header(path))
            path.write_bytes(payload[:-1])
            with self.assertRaises(ValueError):
                self.installer.read_tensor_header(path)
            path.write_bytes(struct.pack('<Q', 2**40))
            with self.assertRaises(ValueError):
                self.installer.read_tensor_header(path)

    def test_verify_removes_stale_markers_before_process_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / '.install'
            state.mkdir()
            for name in ['runtime.complete', 'test.complete', 'environment.complete', 'models.complete']:
                (state / name).touch()
            with patch.object(self.installer, 'validate_source'), patch.object(self.installer, 'validate_assets'), \
                    patch.object(self.installer, 'validate_model_layout'), \
                    patch.object(self.installer, 'run_runtime_probe', side_effect=RuntimeError('CUDA failure')):
                with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
                    self.installer.verify(options(root), {})
            self.assertFalse((state / 'runtime.complete').exists())
            self.assertFalse((state / 'test.complete').exists())
            self.assertFalse((state / 'environment.complete').exists())
            self.assertFalse((state / 'models.complete').exists())

    def test_verify_publishes_only_after_actual_probe_and_uses_selected_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(self.installer, 'validate_source'), patch.object(self.installer, 'validate_assets'), \
                    patch.object(self.installer, 'validate_model_layout'), \
                    patch.object(self.installer, 'run_runtime_probe', return_value={'cuda': '12.8', 'gpu': 'fixture'}) as probe:
                self.installer.verify(options(root, gpu='3'), {})
            probe.assert_called_once()
            self.assertEqual(probe.call_args[0][0].gpu, '3')
            report = json.loads((root / '.install/runtime.complete').read_text())
            self.assertEqual(report['probe']['gpu'], 'fixture')
            self.assertEqual(report['tasks'], ['subtitle'])

    def test_runtime_probe_runs_own_python_offline_and_preserves_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / '.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.dict('os.environ', {'HTTPS_PROXY': 'http://proxy.example:1234'}), \
                    patch.object(self.installer.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='LTX_RUNTIME {"cuda":"12.8"}\n')) as run:
                report = self.installer.run_runtime_probe(options(root, gpu='2'))
            self.assertEqual(report['cuda'], '12.8')
            command = run.call_args.args[0]
            self.assertEqual(command[0], str(python))
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(run.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertEqual(run.call_args.kwargs['env']['HTTPS_PROXY'], 'http://proxy.example:1234')
            self.assertIn('torch.cuda.synchronize()', command[-1])
            self.assertIn('import run_pipeline', command[-1])

    def test_cuda_probe_requires_cuda_and_bfloat16_and_executes_operations(self):
        compile(self.installer.RUNTIME_PROBE, '<LTX runtime>', 'exec')
        function = next(node for node in ast.parse(self.installer.RUNTIME_PROBE).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'validate_cuda')
        torch = MagicMock()
        torch.version.cuda = '12.8'
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<CUDA probe>', 'exec'), namespace)
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'cannot use CUDA'):
            namespace['validate_cuda']()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'bfloat16'):
            namespace['validate_cuda']()
        torch.cuda.is_bf16_supported.return_value = True
        torch.ones.return_value.matmul.return_value.__getitem__.return_value.item.return_value = 16
        namespace['validate_cuda']()
        torch.cuda.synchronize.assert_called_once()
        torch.nn.functional.scaled_dot_product_attention.assert_called_once()


if __name__ == '__main__':
    unittest.main()
