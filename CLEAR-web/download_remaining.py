from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
from pathlib import Path
import shutil
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent.parent / 'CLEAR-models'
WAN = ROOT / 'Wan2.1-T2V-1.3B'
MS = 'https://www.modelscope.cn/models/'
HF = 'https://hf-mirror.com/charlesw09/CLEAR-mask-free-video-subtitle-removal/resolve/main/'
ASSETS = {
    't5': (WAN / 'models_t5_umt5-xxl-enc-bf16.pth', 11361920418,
           '7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d',
           MS + 'Wan-AI/Wan2.1-T2V-1.3B/resolve/master/models_t5_umt5-xxl-enc-bf16.pth'),
    'clear': (ROOT / 'CLEAR-mask-free-subtitle-removal.pt', 175181778,
              '2c0bc5c1317098fbf5f98b83ea8c495d0e7a82cd2557b757139399d174fa08b2',
              HF + 'CLEAR-mask-free-subtitle-removal.pt?download=true'),
}


def range_file(url, part, first, last, total):
    for retry in range(12):
        present = part.stat().st_size if part.exists() else 0
        start = first + present
        if start == last + 1:
            return part
        try:
            request = Request(url, headers={'Range': f'bytes={start}-{last}', 'User-Agent': 'curl/8.13.0'})
            with build_opener(ProxyHandler({})).open(request, timeout=25) as response:
                if response.headers.get('Content-Range') != f'bytes {start}-{last}/{total}':
                    raise OSError('Server did not honor the download range')
                with part.open('ab') as writer:
                    shutil.copyfileobj(response, writer, length=1024**2)
        except OSError as error:
            print(f'{part.name}: {error}', flush=True)
            time.sleep(min(15, retry * 2))
    if part.stat().st_size != last - first + 1:
        raise OSError(f'Incomplete download: {part}')
    return part


def download(path, total, sha, url):
    partial = path.with_suffix(path.suffix + '.prefix')
    if not partial.exists() and path.exists() and path.stat().st_size < total:
        path.replace(partial)
    if not path.exists():
        start = partial.stat().st_size if partial.exists() else 0
        block = 128 * 1024**2 if total > 2**30 else 16 * 1024**2
        spans = [(n, min(n + block, total) - 1) for n in range(start, total, block)]
        print(f'{path.name}: {start}/{total} bytes, {len(spans)} ranges pending', flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(range_file, url, path.with_suffix(f'.range-{a}'), a, b, total)
                       for a, b in spans]
            parts = [future.result() for future in futures]
        assembled = path.with_suffix(path.suffix + '.assembling')
        with assembled.open('wb') as destination:
            for component in ([partial] if partial.exists() else []) + parts:
                with component.open('rb') as source:
                    shutil.copyfileobj(source, destination, length=8 * 1024**2)
        check(assembled, total, sha)
        assembled.replace(path)
        for component in ([partial] if partial.exists() else []) + parts:
            component.unlink()
    else:
        check(path, total, sha)
    print(f'{path.name}: SHA-256 OK', flush=True)


def check(path, total, sha):
    with path.open('rb') as handle:
        if path.stat().st_size != total or file_digest(handle, 'sha256').hexdigest() != sha:
            raise OSError(f'SHA-256 or size mismatch: {path}')


if __name__ == '__main__':
    download(*ASSETS[sys.argv[1]])
