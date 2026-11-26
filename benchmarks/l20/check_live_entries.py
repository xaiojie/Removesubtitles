"""Submit matching jobs over real HTTP, then compare the frozen plan, execution record and output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

from clip_limits import check_clip
from run_candidate import json_file


def get(base, path, headers):
    with urlopen(Request(base + path, headers=headers), timeout=20) as response:
        return json.load(response)


def submit(base, headers, clip, source, mode, catalog_hash):
    regions = clip.get('regions') or [dict(type='subtitle', roi=clip['roi'])]
    cmd = ['curl', '-fsS', '--max-time', '90', '-F', 'video=@' + str(source),
           '--form-string', 'mode=' + mode, '--form-string', 'regions=' + json.dumps(regions),
           '--form-string', 'catalog_sha256=' + catalog_hash]
    for name, value in headers.items():
        cmd.extend(['-H', name + ': ' + value])
    # Keep credentials out of logs and exceptions.
    proc = subprocess.run(cmd + [base + '/api/jobs'], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'提交失败 HTTP {base}: {proc.stderr.strip()}')
    return json.loads(proc.stdout)['id']


def check(args):
    manifest = args.manifest.resolve()
    clip = next(c for c in json.loads(manifest.read_text())['clips'] if c['id'] == args.clip)
    if clip['split'] == 'holdout':
        raise ValueError('入口核对使用调参样片')
    source = (manifest.parent / clip['input']).resolve(strict=True)
    check_clip(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != clip['sha256']:
        raise ValueError('入口测试输入摘要不符')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    token = os.environ.get('SUBTITLE_API_TOKEN') or args.token_file.read_text().strip()
    endpoints = [(args.web.rstrip('/'), {'X-Subtitle-Workflow': 'product'}, 'web'),
                 (args.api.rstrip('/'), {'Authorization': 'Bearer ' + token}, 'api')]
    catalogs = [get(base, '/api/modes', headers) for base, headers, _ in endpoints]
    if catalogs[0] != catalogs[1]:
        raise ValueError('两个入口的档位目录不一致')
    jobs = {}
    for base, headers, entry in endpoints:
        jobs[entry] = submit(base, headers, clip, source, args.mode, catalogs[0]['catalog_sha256'])
        json_file(output/'jobs.json', jobs)
    records = {}
    for base, headers, entry in endpoints:
        url = '/api/jobs/' + jobs[entry]
        while True:
            state = get(base, url, headers)
            if state['state'] in ('done', 'failed'):
                break
            time.sleep(2)
        records[entry] = state
        if state['state'] != 'done':
            continue
        digest = hashlib.sha256()
        with urlopen(Request(base + url + '/output', headers=headers), timeout=120) as result:
            for block in iter(lambda: result.read(1024 * 1024), b''):
                digest.update(block)
        records[entry]['download_sha256'] = digest.hexdigest()
    left, right = records['web'], records['api']
    def execution(row):
        value = dict(row.get('execution') or {})
        value.pop('run_id', None)
        return value
    checks = dict(catalog=True, submitted_via=[left.get('submitted_via'), right.get('submitted_via')] == ['web','api'],
                  both_done=all(s['state']=='done' for s in records.values()),
                  plans=left.get('resolved_plan') == right.get('resolved_plan'),
                  execution=bool(execution(left)) and execution(left) == execution(right),
                  identical_download=bool(left.get('download_sha256')) and left['download_sha256']==right.get('download_sha256'))
    json_file(output/'equivalence.json', dict(mode=args.mode,clip=clip['id'],jobs=jobs,checks=checks,records=records))
    print(json.dumps(checks), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'output', 'token-file'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--clip', required=True)
    cli.add_argument('--mode', choices=('lite','pro','max'), required=True)
    cli.add_argument('--web', default='http://127.0.0.1:8800')
    cli.add_argument('--api', default='http://127.0.0.1:8801')
    check(cli.parse_args())
