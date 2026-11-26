"""Time uploading each frozen clip from the client to an isolated PHP queue with its worker stopped."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from check_live_entries import get, submit
from clip_limits import check_clip


def measure(args):
    base = args.api.rstrip('/')
    headers = {'Authorization': 'Bearer ' + args.token_file.read_text().strip()}
    catalog = get(base, '/api/modes', headers)['catalog_sha256']
    clips = json.loads(args.manifest.read_text())['clips']
    with args.output.open('x', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['clip', 'tier', 'source_seconds', 'bytes', 'upload_submit_seconds', 'accepted_job'])
        for i, clip in enumerate(clips):
            path = args.manifest.parent / clip['input']
            seconds = check_clip(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != clip['sha256']:
                raise ValueError(f"输入摘要不符 {clip['id']}")
            mode = ('lite', 'pro', 'max')[i % 3]
            start = time.perf_counter()
            ident = submit(base, headers, clip, path, mode, catalog)
            elapsed = time.perf_counter() - start
            state = get(base, '/api/jobs/' + ident, headers)
            if state['state'] != 'queued' or state['mode'] != mode:
                raise RuntimeError('上传测速需要独立且未启动推理的队列')
            writer.writerow([clip['id'], mode, float(seconds), path.stat().st_size, round(elapsed, 4), ident])
            file.flush()
    print(f'{len(clips)} 段视频分别完成一次上传，已记录从客户端发送至收到任务编号的耗时')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', required=True)
    for key in ('manifest', 'output', 'token-file'):
        parser.add_argument('--' + key, type=Path, required=True)
    measure(parser.parse_args())
