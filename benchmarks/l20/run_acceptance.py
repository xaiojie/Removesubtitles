"""Measure the frozen product catalog, pinned baseline, public entrypoints and single-card scheduler."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from run_candidate import BASELINE, json_file

CANDIDATES = {'lite': 'ocr-lama', 'pro': 'product-pro', 'max': 'product-max', 'baseline': 'baseline'}
CASES = {
    'lite-three': ['lite:c01', 'lite:c06', 'lite:c16'],
    'mixed-three': ['pro:c06', 'lite:c01', 'lite:c16'],
    'heavy-three': ['pro:c06', 'max:c07', 'pro:c03'],
    'long-short': ['lite:x02', 'pro:x01'],
}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def model_tree(repo, revision, role):
    # Catalog metadata can change when a tested setting becomes the new default.
    # Only reuse pilot timings when every inference source blob is identical.
    paths = ['CLEAR-web', 'php/inference/media.py', 'php/inference/run.py']
    if role == 'lite':
        paths.extend(['php/inference/lite.py', 'video-subtitle-remover/backend'])
    else:
        paths.extend(['CLEAR', 'php/inference/fine.py'])
    files = git(repo, 'ls-tree', '-r', revision, '--', *paths).splitlines()
    return [line for line in files if line.endswith('.py') and '/test_' not in line]


def measured(path, clip, role, expected, root, revision):
    file = path / 'trial.json'
    if not file.is_file():
        return False
    record = json.loads(file.read_text())
    if (not record.get('validated') or record['input_sha256'] != clip['sha256'] or
            record['candidate'] != CANDIDATES[role] or record.get('checkpoint_sha256') or
            not (path / record['output']).is_file()):
        return False
    if role == 'baseline':
        return record['source_revision'] == BASELINE
    plan = record['parameters']['resolved_plan']
    if any(plan.get(k) != expected[k] for k in ('mode', 'stages', 'parameters')):
        return False
    return record['source_revision'] == revision or model_tree(root, record['source_revision'], role) == model_tree(root, revision, role)


def main(args):
    root = Path(__file__).resolve().parents[2]
    install = args.install.resolve()
    bench = args.state.resolve()
    corpus = bench / 'corpus'
    trials = bench / 'trials'
    batches = bench / 'queues'
    output = bench / 'acceptance'
    output.mkdir(parents=True, exist_ok=True)
    trials.mkdir(exist_ok=True)
    batches.mkdir(exist_ok=True)
    cli = Path(__file__).resolve().parent
    interpreter = str(install / '.venv/bin/python')
    revision = git(root, 'rev-parse', 'HEAD')
    catalog = hashlib.sha256((root/'php/product_modes.json').read_bytes()).hexdigest()
    if git(root, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('冻结的推理目录有未提交改动')
    freeze = dict(revision=revision, catalog_sha256=catalog, baseline=BASELINE,
                  corpus_sha256=hashlib.sha256((corpus/'corpus.json').read_bytes()).hexdigest())
    frozen = output/'freeze.json'
    if frozen.exists() and json.loads(frozen.read_text()) != freeze:
        raise ValueError('验收目录与冻结版本不一致')
    json_file(frozen, freeze)
    index_file = output/'records.json'
    records = json.loads(index_file.read_text()) if index_file.exists() else {}
    def record(key, **details):
        records[key] = details
        json_file(index_file, records)
        print(time.strftime('%F %T'), key, details.get('status'), details.get('directory'), flush=True)
    def run(key, directory, cmd, filename, verified):
        if verified:
            record(key, status='reused', directory=directory.name)
            return
        if directory.exists():
            directory.rename(directory.with_name(directory.name + '.attempt-' + str(time.time_ns())))
        print(time.strftime('%F %T'), 'start', key, flush=True)
        rc = subprocess.call(cmd)
        record(key, status='ok' if rc==0 and (directory/filename).is_file() else 'failed',
               returncode=rc, directory=directory.name)
    clips = json.loads((corpus/'corpus.json').read_text())['clips']
    extras = json.loads((corpus/'extras.json').read_text())['clips']
    def isolated(c, role, manifest):
        key = role + '-' + c['id']
        label = key if role=='baseline' else 'final-' + key
        dest = trials / label
        kinds = sorted({r['type'] for r in c.get('regions',[{'type':'subtitle'}])})
        expected = None if role == 'baseline' else json.loads(subprocess.check_output(
            ['php', str(root/'php/product_modes.php'), role, *kinds]))
        if not (dest/'trial.json').exists() and role in ('pro','lite'):
            pilot = trials / (('card5-' if role=='pro' else 'currentlite-') + c['id'])
            if measured(pilot, c, role, expected, root, revision):
                record(key, status='verified pilot', directory=pilot.name)
                return
        cmd=[interpreter, str(cli/'run_candidate.py'),'--candidate',CANDIDATES[role],'--clip',c['id'],
             '--manifest',str(corpus/manifest),'--source',str(args.baseline if role=='baseline' else root),
             '--install',str(install),'--output',str(dest),'--acceptance']
        run(key, dest, cmd, 'trial.json', measured(dest,c,role,expected,root,revision))
    def batch(case, limit, profile=None):
        specs = CASES[case]
        name = 'frozen-' + case + '-at-' + str(limit) + ('-measured' if profile else '')
        dest = batches/name
        marker = dest/'batch.json'
        same = False
        if marker.exists():
            j=json.loads(marker.read_text())
            same = (j['failed'] == 0 and j['capacity'] == limit and j.get('catalog_sha256') == catalog
                    and j.get('source_revision') == revision and [x['mode']+':'+x['clip'] for x in j['jobs']] == specs
                    and j.get('admission_config', {}) == (json.loads(profile.read_text()) if profile else {}))
            for job in j['jobs']:
                plan = json.loads((dest/'jobs'/job['id']/'request.json').read_text())['resolved_plan']
                types=sorted({r['type'] for r in job['regions']})
                expected=json.loads(subprocess.check_output(['php',str(root/'php/product_modes.php'),job['mode'],*types]))
                same &= all(plan[k]==expected[k] for k in ('mode','stages','parameters'))
        manifest = 'extras.json' if case=='long-short' else 'corpus.json'
        cmd=[interpreter,str(cli/'run_queue_batch.py'),'--manifest',str(corpus/manifest),'--output',str(dest),
             '--install',str(install),'--limit',str(limit),*specs]
        if profile:
            cmd.extend(['--scheduler-config',str(profile)])
        run(name,dest,cmd,'batch.json',same)
    def entry(mode):
        label='live-'+mode
        directory=batches/label
        marker=directory/'equivalence.json'
        same=False
        if marker.exists():
            j=json.loads(marker.read_text())
            same=all(j['checks'].values()) and j['records']['web']['resolved_plan']['catalog_sha256']==catalog
        run(label,directory,[interpreter,str(cli/'check_live_entries.py'),'--manifest',str(corpus/'corpus.json'),
            '--output',str(directory),'--clip','c06','--mode',mode,'--token-file',str(install/'.deploy/api/api.token'),
            '--web',args.web,'--api',args.api], 'equivalence.json',same)
    for mode in ('lite','pro','max'):
        entry(mode)
    for c in extras:
        if c['id']=='x01':
            for role in ('pro','max'):
                isolated(c,role,'extras.json')
    for case, limit in [('lite-three',1),('lite-three',2),('lite-three',3),
                        ('heavy-three',2),('mixed-three',2),('long-short',2)]:
        batch(case,limit)
    # Independently held-out sources enter only after the catalog and input set are frozen.
    for c in clips:
        for role in ('baseline','lite','pro','max'):
            isolated(c,role,'corpus.json')
    for case,limit in [('mixed-three',1),('mixed-three',3),('heavy-three',1),('heavy-three',3),('long-short',1)]:
        batch(case,limit)
    if args.heavy_profile:
        batch('heavy-three',3,args.heavy_profile.resolve())
    failed = sorted(key for key, item in records.items() if item['status'] == 'failed')
    if failed:
        raise SystemExit('未完成的测试组：' + '、'.join(failed))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('state','install','baseline'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--heavy-profile',type=Path)
    parser.add_argument('--web',default='http://127.0.0.1:8800')
    parser.add_argument('--api',default='http://127.0.0.1:8801')
    main(parser.parse_args())
