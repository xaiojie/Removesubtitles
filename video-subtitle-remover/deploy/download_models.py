"""Install VSR inference weights from the pinned upstream Git objects."""
import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
MODES = {
    'sttn-auto': ('sttn-auto',),
    'sttn-det': ('sttn-det',),
    'lama': ('big-lama',),
    'opencv': (),
    'propainter': ('propainter', 'big-lama'),  # LaMa handles isolated frames.
}
DETECTORS = {'mobile': 'V5/ch_det_fast', 'server': 'V5/ch_det'}


def selected_files(manifest, mode, detector):
    files = manifest['files']
    if mode == 'all':
        return files
    directories = set(MODES[mode])
    if mode != 'sttn-auto':
        directories.add(DETECTORS[detector])
    return {name: parts for name, parts in files.items()
            if name.rsplit('/', 1)[0] in directories}


def verify_stream(stream, parts):
    """Check every upstream blob boundary, including in an assembled weight."""
    for part in parts:
        remaining = part['size']
        digest = hashlib.sha1(f'blob {remaining}\0'.encode())
        while remaining:
            data = stream.read(min(remaining, 1024 * 1024))
            if not data:
                return False
            digest.update(data)
            remaining -= len(data)
        if digest.hexdigest() != part['git_blob']:
            return False
    return not stream.read(1)


def valid(path, parts):
    if not path.is_file() or path.stat().st_size != sum(p['size'] for p in parts):
        return False
    with path.open('rb') as stream:
        return verify_stream(stream, parts)


def download_part(part, bases, output):
    offset = output.tell()
    errors = []
    for base in bases:
        try:
            url = f"{base.rstrip('/')}/{part['path']}"
            with urllib.request.urlopen(url, timeout=60) as response:
                remaining = part['size']
                while remaining:
                    chunk = response.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError('下载提前结束')
                    output.write(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise ValueError('文件大小不符')
            output.seek(offset)
            if not verify_stream(output, [part]):
                raise ValueError('文件内容与上游 Git blob 不一致')
            return
        except (OSError, ValueError, EOFError, HTTPException) as exc:
            errors.append(str(exc))
            output.seek(offset)
            output.truncate()
    raise ValueError(f"无法下载 {part['path']}: " + ' / '.join(errors))


def install(root, files, bases, verify_only=False):
    for relative, parts in files.items():
        target = root / relative
        if valid(target, parts):
            print('已校验', relative, flush=True)
            continue
        if verify_only:
            raise ValueError(f'模型缺失或内容不符: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        print('下载', relative, flush=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as output:
                tmp = Path(output.name)
                for part in parts:
                    download_part(part, bases, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(tmp, target)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='下载或校验 VSR 模型。默认下载全部原生模式和两种检测器。')
    parser.add_argument('--mode', choices=['all', *MODES], default='all')
    parser.add_argument('--detector', choices=DETECTORS, default='server', help='与网页选择一致；图片使用 LaMa 和检测器')
    parser.add_argument('--model-dir', type=Path, default=os.getenv('VSR_MODEL_DIR') or HERE.parent/'backend/models')
    parser.add_argument('--base-url', help='可选镜像地址，目录结构与上游 backend/models 相同')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--list', action='store_true', help='列出模型和大小，不联网')
    group.add_argument('--verify', action='store_true', help='只校验本地模型，不联网')
    args = parser.parse_args()
    manifest = json.loads((HERE/'models.json').read_text(encoding='utf-8'))
    files = selected_files(manifest, args.mode, args.detector)
    if args.list:
        total = 0
        for name, parts in files.items():
            size = sum(p['size'] for p in parts)
            print(f'{size / 1024**2:8.2f} MiB  {name}')
            total += size
        print(f'合计 {total / 1024**2:.2f} MiB')
        return
    repo, rev = manifest['repository'], manifest['revision']
    bases = [f'https://raw.githubusercontent.com/{repo}/{rev}/backend/models',
             f'https://github.com/{repo}/raw/{rev}/backend/models']
    if args.base_url:
        bases.insert(0, args.base_url)
    install(args.model_dir.resolve(), files, bases, args.verify)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
