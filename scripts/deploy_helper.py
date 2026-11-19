#!/usr/bin/env python3
"""Download verification and Linux service management for deploy.sh."""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


def valid_file(path, asset):
    if not path.is_file() or path.stat().st_size != asset['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == asset['sha256']


def asset_path(root, asset):
    root = Path(root).resolve()
    relative = Path(asset['path'])
    destination = (root / relative).resolve()
    if relative.is_absolute() or root not in destination.parents:
        raise ValueError('模型路径必须位于安装目录内。')
    return destination


def download_asset(asset, root, offline=False, retries=3, timeout=60):
    destination = asset_path(root, asset)
    if valid_file(destination, asset):
        print(f"校验通过 {asset['path']}", flush=True)
        return
    if offline:
        raise RuntimeError(f'离线文件缺失或校验失败 {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    if destination.exists():
        print(f'重新下载校验失败的文件 {destination.name}', flush=True)
        destination.unlink()
    expected_size = asset['size']
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.unlink()
    started = time.monotonic()
    initial_size = partial.stat().st_size if partial.exists() else 0
    last_report = 0
    last_error = '没有可用的下载地址'
    for url in asset['urls']:
        for attempt in range(retries):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                while offset < expected_size:
                    requested_offset = offset
                    headers = {'User-Agent': 'RemoveSubtitles-Installer/1', 'Accept-Encoding': 'identity'}
                    if offset:
                        headers['Range'] = f'bytes={offset}-'
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        if response.status == 206:
                            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                            if not match or int(match[1]) != offset or int(match[3]) != expected_size:
                                raise RuntimeError('下载服务器返回了不匹配的 Range。')
                        elif response.status == 200:
                            length = response.headers.get('Content-Length')
                            if length is not None and int(length) != expected_size:
                                raise RuntimeError('下载服务器返回了不匹配的文件大小。')
                            offset = 0
                        else:
                            raise RuntimeError(f'下载服务器返回 HTTP {response.status}')
                        previous_size = offset
                        with partial.open('ab' if offset else 'wb') as output:
                            while True:
                                chunk = response.read(1024 * 1024)
                                if not chunk:
                                    break
                                if offset + len(chunk) > expected_size:
                                    raise RuntimeError('下载内容超过已知文件大小。')
                                output.write(chunk)
                                offset += len(chunk)
                                now = time.monotonic()
                                if now - last_report >= 2:
                                    speed = max(0, offset - initial_size) / max(.001, now - started)
                                    eta = (expected_size - offset) / max(speed, 1)
                                    print(f'{destination.name} {offset / 1024**2:.1f}/{expected_size / 1024**2:.1f} MiB  {speed / 1024**2:.2f} MiB/s  剩余约 {eta / 60:.1f} 分钟', flush=True)
                                    last_report = now
                            output.flush()
                            os.fsync(output.fileno())
                        if offset <= previous_size or (response.status == 200 and offset <= requested_offset):
                            raise RuntimeError('下载没有取得进展。')
                if not valid_file(partial, asset):
                    partial.unlink(missing_ok=True)
                    initial_size = 0
                    raise RuntimeError('SHA-256 校验失败。')
                os.replace(partial, destination)
                print(f'SHA-256 通过 {destination.name}', flush=True)
                return
            except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
                last_error = str(error)
                print(f'{destination.name} 下载重试 {attempt + 1}/{retries}，源 {urllib.parse.urlsplit(url).hostname}', flush=True)
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'{destination.name} 下载失败，请更换网络源或代理。{last_error}')


def load_assets(options):
    manifest = json.loads(Path(__file__).with_name('model_manifest.json').read_text())
    components = {'clear', 'propainter'} if options.components == 'both' else {options.components}
    result = []
    for item in manifest['assets']:
        if item['component'] not in components:
            continue
        asset = dict(item)
        originals = list(asset['urls'])
        if options.network == 'global':
            originals.sort(key=lambda url: 'huggingface.co' not in url)
        urls = []
        for url in originals:
            if url.startswith('https://huggingface.co/'):
                urls.append(options.hf_endpoint.rstrip('/') + url[len('https://huggingface.co'):])
            elif url.startswith('https://github.com/') and options.github_prefix:
                urls.append(options.github_prefix.rstrip('/') + '/' + url)
            else:
                urls.append(url)
        asset['urls'] = list(dict.fromkeys(urls + originals))
        root = options.propainter_root if item['component'] == 'propainter' else options.root
        result.append((asset, root))
    return result


def models(options, offline):
    state = options.root / '.deploy'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'models.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有模型下载或校验任务正在运行。') from None
        if options.components in ('clear', 'both'):
            (options.root / 'CLEAR-web/models.ready').unlink(missing_ok=True)
        if options.components in ('propainter', 'both'):
            (options.propainter_root / 'models.ready').unlink(missing_ok=True)
            (options.propainter_root / 'runtime.ready').unlink(missing_ok=True)
        assets = load_assets(options)
        pending = []
        required = {}
        for asset, root in assets:
            root.mkdir(parents=True, exist_ok=True)
            destination = asset_path(root, asset)
            if valid_file(destination, asset):
                print(f"校验通过 {asset['path']}", flush=True)
                continue
            partial = destination.with_name(destination.name + '.part')
            prefix = min(asset['size'], partial.stat().st_size) if partial.exists() else 0
            # Invalid complete files will be removed, releasing their allocated blocks.
            released = destination.stat().st_blocks * 512 if destination.is_file() else 0
            device = root.stat().st_dev
            required.setdefault(device, [root, 0])[1] += max(0, asset['size'] - prefix - released)
            pending.append((asset, root))
        if not offline:
            for root, needed in required.values():
                if shutil.disk_usage(root).free < needed + 512 * 1024**2:
                    raise RuntimeError(f'模型目录空间不足 {root}，还需约 {needed / 1024**3:.1f} GiB，并预留 512 MiB。')
        with ThreadPoolExecutor(max_workers=options.workers) as pool:
            futures = [pool.submit(download_asset, asset, root, offline, options.retries, options.timeout) for asset, root in pending]
            for future in as_completed(futures):
                future.result()
        if options.components in ('clear', 'both'):
            marker = options.root / 'CLEAR-web/models.ready'
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text('SHA-256 verified\n')
        if options.components in ('propainter', 'both'):
            (options.propainter_root / 'models.ready').write_text('SHA-256 verified\n')


def extract_propainter(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    allowed = {'inference_propainter.py', 'model', 'core', 'RAFT', 'utils', 'requirements.txt', 'LICENSE', 'README.md'}
    with tarfile.open(archive, 'r:gz') as source:
        for member in source.getmembers():
            parts = Path(member.name).parts
            if member.name.startswith('/') or '..' in parts or member.issym() or member.islnk():
                raise ValueError('源码压缩包包含不安全路径。')
            if len(parts) < 2 or parts[1] not in allowed:
                continue
            target = destination.joinpath(*parts[1:]).resolve()
            if destination not in target.parents:
                raise ValueError('源码路径超出目标目录。')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as incoming, target.open('wb') as output:
                    shutil.copyfileobj(incoming, output)
    if not (destination / 'inference_propainter.py').is_file():
        raise RuntimeError('ProPainter 源码压缩包不完整。')


def validate_cuda(components):
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch 无法使用 CUDA，请检查驱动和 CUDA wheel。')
    torch.ones(1, device='cuda').add_(1)
    image = torch.ones((1, 1, 8, 8), device='cuda')
    offset = torch.zeros((1, 18, 6, 6), device='cuda')
    torchvision.ops.deform_conv2d(image, offset, torch.ones((1, 1, 3, 3), device='cuda'))
    if components in ('clear', 'both'):
        tensor = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
        tensor.matmul(tensor)
    print(f'CUDA 验证通过，PyTorch {torch.__version__}，CUDA {torch.version.cuda}，GPU {torch.cuda.get_device_name(0)}')


def runtime(options):
    if options.components in ('clear', 'both'):
        (options.root / 'CLEAR-web/runtime.ready').unlink(missing_ok=True)
    if options.components in ('propainter', 'both'):
        (options.propainter_root / 'runtime.ready').unlink(missing_ok=True)
    validate_cuda(options.components)
    if options.components in ('clear', 'both'):
        from transformers import AutoTokenizer
        import diffsynth
        import peft
        AutoTokenizer.from_pretrained(str(options.root / 'CLEAR-models/Wan2.1-T2V-1.3B/google/umt5-xxl'), local_files_only=True)
    if options.components in ('propainter', 'both'):
        sys.path.insert(0, str(options.propainter_root))
        from model.propainter import InpaintGenerator
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    # Publish readiness only after all selected runtime checks have passed.
    if options.components in ('clear', 'both'):
        (options.root / 'CLEAR-web/runtime.ready').touch()
    if options.components in ('propainter', 'both'):
        (options.propainter_root / 'runtime.ready').touch()


def process_ticks(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]
    except (OSError, IndexError):
        return None


def service_record(state):
    try:
        record = json.loads((state / 'server.json').read_text())
        actual = process_ticks(record['pid'])
        if actual is not None and actual == record['start_ticks']:
            return record
    except (OSError, ValueError, KeyError):
        pass
    return None


def stop_service(state):
    record = service_record(state)
    if record is None:
        (state / 'server.json').unlink(missing_ok=True)
        (state / 'server.pid').unlink(missing_ok=True)
        print('服务未运行，或进程记录已失效。')
        return False
    pid = record['pid']
    if os.getpgid(pid) != pid:
        raise RuntimeError('服务进程组与记录不一致，已保留进程。')
    os.killpg(pid, signal.SIGTERM)
    for _ in range(50):
        if process_ticks(pid) is None:
            break
        time.sleep(.2)
    if process_ticks(pid) == record['start_ticks']:
        os.killpg(pid, signal.SIGKILL)
    (state / 'server.json').unlink(missing_ok=True)
    (state / 'server.pid').unlink(missing_ok=True)
    print('服务已停止。')
    return True


def drain_request(options, record, method):
    state = options.root / '.deploy'
    state.mkdir(parents=True, exist_ok=True)
    token_path = state / 'maintenance.token'
    try:
        descriptor = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, 'w') as output:
            output.write(secrets.token_urlsafe(32))
    token_path.chmod(0o600)
    token = token_path.read_text().strip()
    if not token:
        raise RuntimeError(f'维护令牌为空，请检查 {token_path}。')
    host = record.get('bind', options.bind)
    if host in ('0.0.0.0', '::'):
        host = '::1' if host == '::' else '127.0.0.1'
    if ':' in host:
        host = f'[{host}]'
    port = record.get('port', options.port)
    request = urllib.request.Request(
        f'http://{host}:{port}/api/deploy/drain', method=method,
        headers={'Host': 'localhost', 'X-Deploy-Token': token},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:
        status = json.load(response)
    if not isinstance(status, dict) or not isinstance(status.get('busy'), list) or not isinstance(status.get('draining'), bool):
        raise RuntimeError('维护接口返回了无法识别的任务状态，已保留服务。')
    return status


def legacy_work_details(options, record):
    busy = []
    for status_path in sorted((options.root / 'CLEAR-web/jobs').glob('*/status.json')):
        try:
            status = json.loads(status_path.read_text())
            if status.get('state') in ('running', 'queued', 'uploading'):
                busy.append(f"{status_path.parent.name} ({status['state']})")
        except (OSError, ValueError):
            busy.append(f'{status_path.parent.name} (状态无法读取)')
    descendants = {record.get('pid')}
    processes = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            processes.append((int(process.name), int(fields[1]), command))
        except (OSError, ValueError, IndexError):
            continue
    while True:
        children = {pid for pid, parent, _ in processes if parent in descendants}
        if children.issubset(descendants):
            break
        descendants.update(children)
    for pid, _, command in processes:
        if pid in descendants and any(name in command for name in ('worker.py', 'ffmpeg')):
            busy.append(f'工作进程 {pid}')
    return '，'.join(busy) or '未发现已保存的忙碌任务或推理子进程'


def wait_for_idle(options, record):
    deadline = time.monotonic() + options.wait_idle
    try:
        status = drain_request(options, record, 'POST')
        while status['busy']:
            details = ', '.join(f"{job.get('id', '?')} ({job.get('state', '?')})" for job in status['busy'])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f'任务队列仍在处理 {details}。本次升级已停止，服务继续运行。完成后重试或增加 --wait-idle。')
            print(f'等待任务完成 {details}，剩余等待约 {remaining:.0f} 秒', flush=True)
            time.sleep(min(2, remaining))
            status = drain_request(options, record, 'GET')
        if not status['draining']:
            raise RuntimeError('服务未进入维护状态，已保留服务，请重新执行 update。')
    except Exception as error:
        try:
            drain_request(options, record, 'DELETE')
        except (OSError, ValueError, RuntimeError):
            pass
        if isinstance(error, urllib.error.HTTPError) and error.code in (404, 405):
            details = legacy_work_details(options, record)
            raise RuntimeError(
                f'旧版服务缺少安全维护接口，已检查任务文件与工作进程 {details}。'
                '请在网页确认所有上传、排队、处理与对比任务完成，并关闭上传页面，'
                f'再执行 bash "{options.root / "scripts/deploy.sh"}" stop --dir "{options.root}"，随后重试 update。'
            ) from error
        raise


def managed_source_path(relative):
    path = Path(relative)
    if not path.parts or path.is_absolute() or '..' in path.parts:
        return False
    roots = {'CLEAR', 'CLEAR-web', 'scripts', 'tests', 'docs', 'video-subtitle-remover', 'php'}
    files = {'README.md', 'LICENSE', 'LICENSE.md', '.gitignore', '.gitattributes', 'pyproject.toml'}
    if path.parts[0] not in roots and str(path) not in files:
        return False
    if path.parts[:2] == ('php', 'var'):
        return False
    if path.parts[0] == 'video-subtitle-remover' and any(
        part in {'.venv-web', 'vsr-data', 'config', 'ffmpeg'} for part in path.parts[1:]
    ):
        return False
    protected = {'jobs', 'videos', 'models', 'weights', 'checkpoints', '.venv', 'venv', '.deploy', '.install', '__pycache__'}
    if any(part in protected for part in path.parts):
        return False
    name = path.name.lower()
    if name.endswith(('.pt', '.pth', '.safetensors', '.bin', '.whl', '.ready', '.complete', '.mp4', '.video')):
        return False
    return not (name.startswith(('.env', 'config.local', 'settings.local')) or name in ('config.json', 'config.sh'))


def extract_update_source(archive, destination):
    files = []
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        names = {member.name for member in members}
        prefix = 0 if 'CLEAR-web/app.py' in names else 1
        for member in members:
            parts = Path(member.name).parts
            if member.name.startswith('/') or '..' in parts or member.issym() or member.islnk():
                raise ValueError('升级压缩包包含不安全路径或符号链接。')
            if not member.isfile() or len(parts) <= prefix:
                continue
            relative = Path(*parts[prefix:])
            if not managed_source_path(relative):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, target.open('wb') as output:
                shutil.copyfileobj(incoming, output)
            target.chmod(member.mode & 0o777)
            files.append(str(relative))
    required = {'CLEAR-web/app.py', 'scripts/deploy.sh', 'scripts/deploy_helper.py'}
    if not required.issubset(files):
        raise RuntimeError('升级源码不完整，需要 CLEAR-web/app.py 和部署脚本。')
    return set(files)


def update_target(root, relative):
    target = root / relative
    for path in (target, *target.parents):
        if path == root:
            break
        if path.is_symlink():
            raise RuntimeError(f'升级目标包含符号链接，请先检查 {path}。')
    if target.exists() and not target.is_file():
        raise RuntimeError(f'升级目标不是普通文件 {target}。')
    return target


def publish_source(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.deploy-pending')
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def git_output(root, *arguments):
    result = subprocess.run(['git', '-C', str(root), *arguments], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Git 操作失败 {result.stderr.strip() or result.stdout.strip()}')
    return result.stdout.strip()


def git_update_archive(options, archive):
    root = options.root
    previous = git_output(root, 'rev-parse', 'HEAD')
    if git_output(root, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Git 源码含有本地修改，请先保存或提交修改后重试 update。模型、任务与环境会保留。')
    reference = options.update_ref
    if not options.offline:
        git_output(root, 'fetch', '--no-tags', 'origin', reference)
        reference = 'FETCH_HEAD'
    revision = git_output(root, 'rev-parse', '--verify', f'{reference}^{{commit}}')
    try:
        git_output(root, 'merge-base', '--is-ancestor', previous, revision)
    except RuntimeError as error:
        raise RuntimeError('升级目标与当前 Git 历史无法快进合并，请检查 --update-ref 并保留本地提交后重试。') from error
    git_output(root, 'archive', '--format=tar.gz', f'--output={archive}', revision)
    previous_files = set(git_output(root, 'ls-files').splitlines())
    return previous, revision, previous_files


def validate_updated_source(options, source):
    code = (
        'from pathlib import Path\n'
        'for path in Path("..").rglob("*.py"):\n'
        '    if not any(part in {"jobs", "__pycache__", ".venv"} for part in path.parts):\n'
        '        compile(path.read_bytes(), str(path), "exec")\n'
        'import app\n'
    )
    environment = os.environ.copy()
    environment.update(CLEAR_WORKSPACE=str(options.root), PROPAINTER_ROOT=str(options.propainter_root),
                       PYTHONDONTWRITEBYTECODE='1', CLEAR_PRESET=options.preset)
    result = subprocess.run([str(options.python), '-c', code], cwd=source / 'CLEAR-web',
                            env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'升级环境检查失败，原源码和服务已保留。请先检查依赖并执行 env 后重试。{result.stderr.strip()}')


def ensure_no_unmanaged_service(options):
    host = options.bind
    if host in ('0.0.0.0', '::'):
        host = '::1' if host == '::' else '127.0.0.1'
    try:
        connection = socket.create_connection((host, options.port), timeout=.5)
    except OSError:
        return
    connection.close()
    raise RuntimeError(f'端口 {options.port} 存在未受部署脚本管理的服务。请确认任务全部完成，再通过原启动方式停止服务后重试 update。')


def update_project(options):
    root = options.root
    state = root / '.deploy'
    state.mkdir(parents=True, exist_ok=True)
    if not (root / 'CLEAR-web/app.py').is_file():
        raise RuntimeError('update 需要已有安装，请先执行 source 或 all。')
    manifest = state / 'source-files.json'
    manifest_original = manifest.read_bytes() if manifest.is_file() else None
    previous_files = set(json.loads(manifest_original)) if manifest_original is not None else set()
    previous = revision = None
    with tempfile.TemporaryDirectory(prefix='update-', dir=state) as directory:
        work = Path(directory)
        archive = options.archive
        if archive is None:
            if not (root / '.git').exists():
                raise RuntimeError('归档安装升级需要源码压缩包，请使用 deploy.sh update。')
            archive = work / 'source.tar.gz'
            previous, revision, previous_files = git_update_archive(options, archive)
        elif (root / '.git').exists():
            raise RuntimeError('Git 安装请使用 --update-ref 指定版本。--update-archive 用于归档安装。')
        stage = work / 'source'
        stage.mkdir()
        incoming = extract_update_source(archive, stage)
        changed = incoming | {name for name in previous_files if managed_source_path(name)}
        for name in changed:
            update_target(root, name)
        validate_updated_source(options, stage)
        backup = work / 'backup'
        existed = set()
        required = sum((root / name).stat().st_size for name in changed if (root / name).is_file())
        if shutil.disk_usage(state).free < required + 1024 * 1024:
            raise RuntimeError('升级备份空间不足，请释放磁盘空间后重试。')
        for name in changed:
            target = root / name
            if target.is_file():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
                existed.add(name)
        record = service_record(state)
        if record:
            wait_for_idle(options, record)
        else:
            ensure_no_unmanaged_service(options)
        stopped = False
        metadata_changed = False
        try:
            if record:
                stopped = stop_service(state)
                if not stopped:
                    raise RuntimeError('原服务进程已变化，升级已停止，请检查 status 后重试。')
            for name in sorted(changed):
                target = root / name
                if name in incoming:
                    publish_source(stage / name, target)
                else:
                    target.unlink(missing_ok=True)
            if previous:
                git_output(root, 'read-tree', revision)
                metadata_changed = True
                git_output(root, 'update-ref', 'HEAD', revision, previous)
            temporary = manifest.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(sorted(incoming), ensure_ascii=False, indent=2))
            os.replace(temporary, manifest)
            if record:
                start_service(options)
        except Exception as error:
            restore_errors = []
            if stopped:
                try:
                    stop_service(state)
                except (OSError, RuntimeError) as stop_error:
                    restore_errors.append(str(stop_error))
            for name in sorted(changed):
                try:
                    if name in existed:
                        publish_source(backup / name, root / name)
                    else:
                        (root / name).unlink(missing_ok=True)
                except OSError as restore_error:
                    restore_errors.append(str(restore_error))
            if metadata_changed:
                try:
                    git_output(root, 'read-tree', previous)
                    current = git_output(root, 'rev-parse', 'HEAD')
                    if current == revision:
                        git_output(root, 'update-ref', 'HEAD', previous, revision)
                except RuntimeError as restore_error:
                    restore_errors.append(str(restore_error))
            try:
                if manifest_original is None:
                    manifest.unlink(missing_ok=True)
                else:
                    temporary = manifest.with_suffix('.json.tmp')
                    temporary.write_bytes(manifest_original)
                    os.replace(temporary, manifest)
            except OSError as restore_error:
                restore_errors.append(str(restore_error))
            if stopped and not restore_errors:
                original = copy.copy(options)
                original.port = record.get('port', options.port)
                original.bind = record.get('bind', options.bind)
                for setting, value in record.get('settings', {}).items():
                    setattr(original, setting, Path(value) if setting.endswith('_root') or setting == 'python' else value)
                try:
                    start_service(original)
                except (OSError, RuntimeError) as restart_error:
                    restore_errors.append(str(restart_error))
            if restore_errors:
                retained = state / f'update-recovery-{time.time_ns()}'
                shutil.copytree(backup, retained, dirs_exist_ok=True)
                raise RuntimeError(f'升级失败 {error}。恢复遇到问题 {"，".join(restore_errors)}。源码备份保留于 {retained}。') from error
            raise RuntimeError(f'升级失败，已恢复原源码和服务状态。{error}') from error
        finally:
            manifest.with_suffix('.json.tmp').unlink(missing_ok=True)
            if record and not stopped:
                try:
                    drain_request(options, record, 'DELETE')
                except (OSError, ValueError, RuntimeError):
                    pass
    print('项目源码升级完成，模型、任务、本地配置与 Python 环境已保留。')


def start_service(options):
    state = options.root / '.deploy'
    state.mkdir(parents=True, exist_ok=True)
    existing = service_record(state)
    if existing:
        print(f"服务已运行，PID {existing['pid']}")
        return
    family = socket.AF_INET6 if ':' in options.bind else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((options.bind, options.port))
        except OSError:
            raise RuntimeError(f'端口 {options.port} 已被占用，请指定其他 --port。') from None
    environment = os.environ.copy()
    environment.update(CLEAR_WORKSPACE=str(options.root), PROPAINTER_ROOT=str(options.propainter_root), PROPAINTER_PYTHON=str(options.python), HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
    environment['CLEAR_PRESET'] = options.preset
    if options.with_ltx:
        environment.update(LTX_ROOT=str(options.ltx_root), LTX_PYTHON=str(options.ltx_root / '.venv/bin/python'))
    hosts = ['localhost', '127.0.0.1', '[::1]']
    hosts.extend(host.strip() for host in options.public_host.split(',') if host.strip())
    environment['CLEAR_TRUSTED_HOSTS'] = ','.join(hosts)
    if options.gpu:
        environment['CUDA_VISIBLE_DEVICES'] = options.gpu
    address = f'[{options.bind}]:{options.port}' if ':' in options.bind else f'{options.bind}:{options.port}'
    command = [str(options.python), '-m', 'gunicorn', '--workers', '1', '--threads', '8', '--timeout', '0', '--bind', address, '--access-logfile', str(state / 'access.log'), '--error-logfile', str(state / 'server.log'), 'app:app']
    with (state / 'launcher.log').open('ab') as log:
        process = subprocess.Popen(command, cwd=options.root / 'CLEAR-web', env=environment, stdout=log, stderr=log, start_new_session=True)
    ticks = process_ticks(process.pid)
    record = {'pid': process.pid, 'start_ticks': ticks, 'port': options.port, 'bind': options.bind}
    record['settings'] = {name: str(getattr(options, name)) if isinstance(getattr(options, name), Path) else getattr(options, name)
                          for name in ('python', 'propainter_root', 'preset', 'with_ltx', 'ltx_root', 'port', 'bind', 'public_host', 'gpu')}
    temporary = state / 'server.json.tmp'
    temporary.write_text(json.dumps(record))
    os.replace(temporary, state / 'server.json')
    (state / 'server.pid').write_text(str(process.pid) + '\n')
    local_host = options.bind
    if local_host in ('0.0.0.0', '::'):
        local_host = '::1' if family == socket.AF_INET6 else '127.0.0.1'
    if ':' in local_host:
        local_host = f'[{local_host}]'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(60):
        if process.poll() is not None:
            (state / 'server.json').unlink(missing_ok=True)
            (state / 'server.pid').unlink(missing_ok=True)
            raise RuntimeError(f'服务启动失败，请查看 {state / "server.log"} 和 launcher.log。')
        try:
            request = urllib.request.Request(f'http://{local_host}:{options.port}/api/system', headers={'Host': 'localhost'})
            with opener.open(request, timeout=1) as response:
                if response.status == 200:
                    host = options.public_host.split(',')[0] or local_host
                    print(f'网页已启动 http://{host}:{options.port}/')
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(.5)
    stop_service(state)
    raise RuntimeError('网页启动检查超时，请查看 .deploy/server.log。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['models', 'verify', 'extract-propainter', 'cuda', 'runtime', 'capabilities', 'update', 'start', 'stop', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--propainter-root', type=Path)
    parser.add_argument('--components', choices=['clear', 'propainter', 'both'], default='both')
    parser.add_argument('--network', choices=['cn', 'global'], default='global')
    parser.add_argument('--hf-endpoint', default='https://huggingface.co')
    parser.add_argument('--github-prefix', default='')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--public-host', default='')
    parser.add_argument('--gpu', default='')
    parser.add_argument('--preset', choices=['fast', 'balanced', 'quality'], default='balanced')
    parser.add_argument('--with-ltx', action='store_true')
    parser.add_argument('--ltx-root', type=Path)
    parser.add_argument('--update-ref', default='main')
    parser.add_argument('--wait-idle', type=int, default=300)
    options = parser.parse_args()
    options.root = options.root.resolve()
    options.propainter_root = (options.propainter_root or options.root / 'ProPainter').resolve()
    options.python = options.python or options.root / '.venv/bin/python'
    options.ltx_root = (options.ltx_root or options.root / 'LTX2-ICEdit-Insight').resolve()
    if not 1 <= options.workers <= 8 or not 1 <= options.port <= 65535 or options.retries < 1 or options.timeout < 1 or options.wait_idle < 0:
        parser.error('workers 必须为 1～8，port 必须为 1～65535，重试次数和超时必须为正数。')
    if options.command in ('models', 'verify'):
        models(options, options.offline or options.command == 'verify')
    elif options.command == 'runtime':
        runtime(options)
    elif options.command == 'cuda':
        validate_cuda(options.components)
    elif options.command == 'extract-propainter':
        if options.archive is None:
            parser.error('源码解压需要 --archive。')
        extract_propainter(options.archive, options.propainter_root)
    elif options.command == 'start':
        start_service(options)
    elif options.command == 'update':
        update_project(options)
    elif options.command == 'capabilities':
        print('update-v1')
    elif options.command == 'stop':
        stop_service(options.root / '.deploy')
    else:
        record = service_record(options.root / '.deploy')
        print(json.dumps({'running': bool(record), 'service': record}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f'错误 {error}', file=sys.stderr)
        sys.exit(1)
