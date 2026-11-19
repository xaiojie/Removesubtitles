"""Inference entry point. PHP owns API, queue, state and output authorization."""

import json
import os
import subprocess
import sys
from pathlib import Path

from media import ROOT, report


def standard(folder, request, fine):
    from pipeline_worker import atomic_json
    from task_options import prepare_task

    # Existing CLEAR/ProPainter workers use these exact filenames and option parsers.
    task = prepare_task(request['regions'], '{}', default_preset='balanced')
    work = folder / 'standard'
    work.mkdir(exist_ok=True)
    link = work / 'source.video'
    if not link.exists():
        link.symlink_to(folder / 'source.video')
    atomic_json(work / 'config.json', dict(**request, **task))
    command = [
        sys.executable,
        '-u',
        str(ROOT / 'CLEAR-web/pipeline_worker.py'),
        str(work),
        str(Path(os.getenv('CLEAR_WORKSPACE', ROOT)) / 'CLEAR'),
        os.getenv('PROPAINTER_ROOT', str(ROOT / 'ProPainter')),
        os.getenv('PROPAINTER_PYTHON', sys.executable),
    ]
    error = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            if line.startswith('CLEAR_STATUS '):
                update = json.loads(line[13:])
                if 'error' in update:
                    error = update['error'].split('Traceback')[0].strip()
                if 'percent' in update:
                    report(update['percent'] * (0.55 if fine else 0.99), update.get('stage', '标准处理'))
            else:
                print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError((error or '标准处理失败')[:500])
    return work / 'output.mp4'


def run(folder):
    from regions import validate_regions

    request = json.loads((folder / 'request.json').read_text())
    validate_regions(request['regions'], request['dimensions'])
    if request['mode'] == 'lite':
        from lite import run as run_lite

        run_lite(folder, request)
    elif request['mode'] in ('standard', 'fine'):
        fine = request['mode'] == 'fine'
        first = standard(folder, request, fine)
        if fine:
            from fine import refine

            refine(folder, request, first)
        else:
            first.replace(folder / 'result.mp4')
    else:
        raise ValueError('未知质量档位')


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]).resolve())
    except Exception as exc:
        print('JOB_STATUS ' + json.dumps({'error': str(exc)[:540]}, ensure_ascii=False), flush=True)
        raise
