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
from run_candidate import benchmark_geometry, fingerprint, json_file, validate_output, video_info


def get(base, path, headers):
    with urlopen(Request(base + path, headers=headers), timeout=20) as response:
        return json.load(response)


def submit(base, headers, clip, source, mode, catalog_hash):
    regions = benchmark_geometry(clip, video_info(source,count_frames=False))['display_regions']
    cmd = ['curl', '-fsS', '--max-time', '90', '-F', 'video=@' + str(source),
           '--form-string', 'mode=' + mode, '--form-string', 'regions=' + json.dumps(regions),
           '--form-string', 'catalog_sha256=' + catalog_hash]
    if headers:
        cmd.extend(['-H', '@-'])
    proc = subprocess.run(cmd + [base + '/api/jobs'], capture_output=True, text=True,
                          input=''.join(f'{name}: {value}\n' for name, value in headers.items()))
    if proc.returncode:
        raise RuntimeError(f'提交失败 HTTP {base}: {proc.stderr.strip()}')
    return json.loads(proc.stdout)['id']


def check(args):
    manifest = args.manifest.resolve()
    corpus = json.loads(manifest.read_text())
    clip = next(c for c in corpus['clips'] if c['id'] == args.clip)
    if clip['split'] == 'holdout':
        raise ValueError('入口核对使用调参样片')
    source = (manifest.parent / clip['input']).resolve(strict=True)
    seconds = check_clip(source)
    if corpus.get('max_seconds') and seconds > corpus['max_seconds']:
        raise ValueError('入口测试输入超过本轮时长上限')
    if fingerprint(source) != clip['sha256']:
        raise ValueError('入口测试输入摘要不符')
    raw = video_info(source)
    geometry = benchmark_geometry(clip, raw)
    if clip.get('frames') and int(raw['nb_read_frames']) != clip['frames']:
        raise ValueError('入口测试输入帧数不符')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    token = os.environ.get('SUBTITLE_API_TOKEN') or args.token_file.read_text().strip()
    endpoints = [(args.web.rstrip('/'), {'X-Subtitle-Workflow': 'product'}, 'web'),
                 (args.api.rstrip('/'), {'Authorization': 'Bearer ' + token}, 'api')]
    catalogs = [get(base, '/api/modes', headers) for base, headers, _ in endpoints]
    if catalogs[0] != catalogs[1]:
        raise ValueError('两个入口的档位目录不一致')
    catalog_hash = catalogs[0]['catalog_sha256']
    expected = None
    if args.expect_local_plan:
        resolver = Path(__file__).resolve().parents[2] / 'php/product_modes.php'
        types = sorted({region['type'] for region in geometry['display_regions']})
        expected = json.loads(subprocess.check_output(['php', str(resolver), args.mode, *types]))
        if expected['catalog_sha256'] != catalog_hash:
            raise ValueError('服务返回的档位目录与要验收的版本不同')
    jobs, upload_seconds = {}, {}
    for base, headers, entry in endpoints:
        start = time.monotonic()
        jobs[entry] = submit(base, headers, clip, source, args.mode, catalog_hash)
        upload_seconds[entry] = round(time.monotonic() - start, 4)
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
        partial = output / (entry + '.partial.mp4')
        with urlopen(Request(base + url + '/output', headers=headers), timeout=120) as result, partial.open('wb') as media:
            for block in iter(lambda: result.read(1024 * 1024), b''):
                digest.update(block)
                media.write(block)
        probe = video_info(partial)
        validate_output(raw, probe, geometry)
        partial.replace(output / (entry + '.mp4'))
        records[entry]['download_video'] = probe
        records[entry]['download_sha256'] = digest.hexdigest()
    left, right = records['web'], records['api']
    def execution(row):
        value = dict(row.get('execution') or {})
        value.pop('run_id', None)
        return value
    checks = dict(catalog=True, submitted_via=[left.get('submitted_via'), right.get('submitted_via')] == ['web','api'],
                  requested_mode=all(s.get('mode') == args.mode for s in records.values()),
                  requested_regions=all(s.get('dimensions') == geometry['display_dimensions'] and
                                        s.get('regions') == geometry['display_regions'] for s in records.values()),
                  both_done=all(s['state']=='done' for s in records.values()),
                  plans=left.get('resolved_plan') == right.get('resolved_plan'),
                  execution=bool(execution(left)) and execution(left) == execution(right),
                  identical_download=bool(left.get('download_sha256')) and left['download_sha256']==right.get('download_sha256'))
    for entry, record in records.items():
        run = execution(record)
        plan = record.get('resolved_plan') or {}
        checks[entry + '_used_plan'] = (bool(run.get('entrypoint_sha256')) and run.get('mode') == args.mode and
                                        run.get('catalog_sha256') == catalog_hash and
                                        run.get('stages') == plan.get('stages') and
                                        run.get('parameters') == plan.get('parameters') and
                                        (expected is None or plan == expected))
    json_file(output/'equivalence.json', dict(mode=args.mode,clip=clip['id'],source_sha256=clip['sha256'],
              source_seconds=float(seconds),geometry=geometry,expected_plan=expected,upload_seconds=upload_seconds,
              jobs=jobs,checks=checks,records=records))
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
    cli.add_argument('--expect-local-plan', action='store_true',
                     help='使用当前检出的 PHP 档位定义核对实际处理版本、步骤和参数')
    check(cli.parse_args())
