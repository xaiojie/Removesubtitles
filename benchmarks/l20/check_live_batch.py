"""Measure a Web batch and an independent API request on the deployed shared queue."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from check_live_entries import get, submit
from clip_limits import check_clip
from run_candidate import json_file
from run_queue_batch import report


def check(args):
    catalog_token = args.token_file.read_text().strip()
    entries = {
        'web': (args.web.rstrip('/'), {'X-Subtitle-Workflow': 'product'}),
        'api': (args.api.rstrip('/'), {'Authorization': 'Bearer ' + catalog_token}),
    }
    catalogs = [get(base, '/api/modes', headers) for base, headers in entries.values()]
    if catalogs[0] != catalogs[1]:
        raise ValueError('Web 与 API 的档位目录不一致')
    catalog = catalogs[0]['catalog_sha256']
    clips = {c['id']: c for name in ('corpus.json', 'extras.json')
             for c in json.loads((args.corpus / name).read_text())['clips']}
    cases = [('web', 'x02', 'lite'), ('web', 'c06', 'pro'), ('api', 'c01', 'lite')]
    jobs = []
    for source, cid, tier in cases:
        c = clips[cid]
        file = (args.corpus / c['input']).resolve(strict=True)
        seconds = float(check_clip(file))
        if c['split'] == 'holdout' or hashlib.sha256(file.read_bytes()).hexdigest() != c['sha256']:
            raise ValueError(f'联调输入与调参素材不符 {cid}')
        # x02 is the long, subtitle-only case. Use its subtitle box from the recorded mixed source.
        regions = [r for r in c.get('regions', []) if r['type'] == 'subtitle']
        regions = regions or [{'type': 'subtitle', 'roi': c['roi']}]
        jobs.append(dict(clip=cid, mode=tier, entry=source, file=str(file), source_seconds=seconds,
                         sha256=c['sha256'], regions=regions))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    for job in jobs:
        base, headers = entries[job['entry']]
        upload_start = time.perf_counter()
        job['id'] = submit(base, headers, job, Path(job['file']), job['mode'], catalog)
        job['upload_submit_seconds'] = round(time.perf_counter() - upload_start, 4)
        json_file(args.output / 'inputs.json', jobs)
    for job in jobs:
        base, headers = entries[job['entry']]
        path = '/api/jobs/' + job['id']
        while True:
            state = get(base, path, headers)
            if state['state'] in ('done', 'failed'):
                break
            time.sleep(2)
        if (state['state'] != 'done' or state['submitted_via'] != job['entry'] or
                state['mode'] != job['mode'] or state['regions'] != job['regions'] or
                state['resolved_plan']['catalog_sha256'] != catalog):
            raise RuntimeError(f"任务的结果、提交入口或选区不符 {job['id']}")
        digest = hashlib.sha256()
        with urlopen(Request(base + path + '/output', headers=headers), timeout=120) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        job['download_sha256'] = digest.hexdigest()
    measured = report(args.data, jobs, started, args.limit)
    checks = dict(three_downloads=all(j['output_sha256'] == j['download_sha256'] for j in measured['jobs']),
                  complete=measured['completed'] == len(jobs),
                  model_overlap=any(r['kind'] == 'gpu_gpu' for r in measured['overlaps']))
    json_file(args.output / 'live-batch.json', dict(checks=checks, catalog_sha256=catalog, **measured))
    if not all(checks.values()):
        raise RuntimeError(f'混合入口并发未通过 {checks}')
    print(f"{len(jobs)} 个独立视频已通过两个入口完成，含提交共 {measured['batch_seconds']:.1f} 秒")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'output', 'data', 'token-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--web', default='http://127.0.0.1:8800')
    parser.add_argument('--api', default='http://127.0.0.1:8801')
    parser.add_argument('--limit', type=int, required=True, help='当前生产调度器配置的并发上限，用于记录')
    check(parser.parse_args())
