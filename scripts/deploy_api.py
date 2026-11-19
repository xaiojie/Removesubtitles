"""Manage the PHP test endpoint and serial GPU worker on Linux, including hosts without systemd."""

import argparse
import fcntl
import json
import os
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from deploy_helper import process_ticks, service_record, stop_service

SERVICES = ('worker', 'http')


def address(host, port):
    return f'[{host}]:{port}' if ':' in host else f'{host}:{port}'


def token_for(state):
    path = state / 'api.token'
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'w') as output:
            output.write(os.getenv('SUBTITLE_API_TOKEN') or secrets.token_urlsafe(32))
    token = path.read_text().strip()
    if not token or not token.isascii() or any(c.isspace() or not c.isprintable() for c in token):
        raise ValueError(f'接口令牌需要可打印、不含空白的 ASCII 字符，请检查 {path}')
    return token


def active(state):
    return [name for name in SERVICES if service_record(state / name)]


def worker_busy(data):
    with (data / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def launch(directory, command, env, cwd):
    directory.mkdir(exist_ok=True)
    with (directory / 'service.log').open('ab') as log:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=cwd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        ticks = process_ticks(process.pid)
        if ticks is None or process.poll() is not None:
            raise RuntimeError(f'启动失败，请查看 {directory / "service.log"}')
        tmp = directory / 'server.json.tmp'
        tmp.write_text(json.dumps({'pid': process.pid, 'start_ticks': ticks}))
        tmp.replace(directory / 'server.json')
        return process
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise


def catalog(settings, token):
    host = {'0.0.0.0': '127.0.0.1', '::': '::1'}.get(settings['bind'], settings['bind'])
    url = 'http://' + address(host, settings['port']) + '/api/modes'
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=1) as response:
        modes = json.load(response)
    return modes.get('default') == 'pro' and set(modes.get('modes', {})) == {'lite', 'pro', 'max'}


def start(root, state, settings, services=SERVICES):
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing = active(state)
    saved = state / 'settings.json'
    if existing and (not saved.is_file() or json.loads(saved.read_text()) != settings):
        raise RuntimeError('运行中的 API 使用另一组配置，请执行 api-stop 后再以新配置启动。')
    data = Path(settings['data'])
    data.mkdir(parents=True, exist_ok=True)
    token = token_for(state)
    php = shutil.which('php')
    if not php:
        raise ValueError('未安装 PHP CLI')
    webroot = root / 'php/public'
    env = dict(
        os.environ,
        SUBTITLE_DATA_DIR=str(data),
        CLEAR_WORKSPACE=str(root),
        CLEAR_PYTHON=str(root / '.venv/bin/python'),
        PROPAINTER_ROOT=settings['propainter'],
        PROPAINTER_PYTHON=str(root / '.venv/bin/python'),
        LITE_DEVICE='cuda',
        VSR_MODEL_DIR=str(root / 'video-subtitle-remover/backend/models'),
        VSR_PYTHON=str(root / 'video-subtitle-remover/.venv/bin/python'),
    )
    env.pop('SUBTITLE_API_TOKEN', None)
    if settings['gpu']:
        env['CUDA_VISIBLE_DEVICES'] = settings['gpu']
    bind = address(settings['bind'], settings['port'])
    if 'http' in services and 'http' not in existing:
        family = socket.AF_INET6 if ':' in settings['bind'] else socket.AF_INET
        with socket.socket(family) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((settings['bind'], settings['port']))
    if 'worker' in services and 'worker' not in existing and worker_busy(data):
        raise RuntimeError('已有独立 PHP worker 使用相同任务目录，请通过原方式管理该进程。')
    commands = {
        'worker': [php, str(root / 'php/bin/worker.php')],
        'http': [
            php,
            '-d',
            'upload_max_filesize=2G',
            '-d',
            'post_max_size=2100M',
            '-S',
            bind,
            '-t',
            str(webroot),
            str(webroot / 'index.php'),
        ],
    }
    processes = {}
    tmp = saved.with_suffix('.tmp')
    tmp.write_text(json.dumps(settings))
    tmp.replace(saved)
    try:
        for name in services:
            if name not in existing:
                environment = (
                    dict(env, SUBTITLE_API_TOKEN=token, PHP_CLI_SERVER_WORKERS='4') if name == 'http' else env
                )
                processes[name] = launch(state / name, commands[name], environment, root)
        for _ in range(60):
            if any(p.poll() is not None for p in processes.values()):
                break
            ready = 'worker' not in services or ((data / 'queue.sqlite').is_file() and worker_busy(data))
            try:
                if ready and ('http' not in services or catalog(settings, token)):
                    print(f'PHP 服务已就绪 {bind}，接口 /api/modes，令牌文件 {state / "api.token"}')
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise RuntimeError(f'PHP 服务启动检查失败，请查看 {state}/{{http,worker}}/service.log')
    except BaseException:
        for name in reversed(processes):
            stop_service(state / name)
            processes[name].wait()
        raise


def pending(data):
    database = data / 'queue.sqlite'
    if not database.is_file():
        return 0
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone():
            return 0
        return db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]


@contextmanager
def drain(data, seconds, services):
    deadline = time.monotonic() + seconds
    with (data / 'submissions.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not pending(data) or ('worker' not in services and not worker_busy(data)):
                    break
            except BlockingIOError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('等待 PHP 队列空闲超时，服务继续运行。')
            time.sleep(0.2)
        yield


def manage(options):
    root = options.root.resolve()
    state = root / '.deploy/api'
    services = active(state)
    if options.action == 'status':
        for name in SERVICES:
            print('PHP', name, '运行中' if name in services else '已停止')
        if 'http' in services:
            settings = json.loads((state / 'settings.json').read_text())
            print('监听', address(settings['bind'], settings['port']), '接口 /api/modes')
        return
    if options.action == 'start':
        data = Path(os.getenv('SUBTITLE_DATA_DIR') or root / 'php/var').resolve()
        settings = {
            'bind': options.bind,
            'port': options.port,
            'gpu': options.gpu,
            'data': str(data),
            'propainter': str((options.propainter_root or root / 'ProPainter').resolve()),
        }
        if (state / 'settings.json').exists() and not os.getenv('SUBTITLE_DATA_DIR'):
            settings['data'] = json.loads((state / 'settings.json').read_text())['data']
        start(root, state, settings)
        return
    if services:
        settings = json.loads((state / 'settings.json').read_text())
        with drain(Path(settings['data']), options.wait, services):
            for name in reversed(services):
                stop_service(state / name)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait', type=int, default=300)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8810)
    parser.add_argument('--gpu', default='')
    parser.add_argument('--propainter-root', type=Path)
    manage(parser.parse_args(argv))


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.CalledProcessError) as err:
        sys.exit(str(err))
