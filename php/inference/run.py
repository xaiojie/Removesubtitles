"""Inference entry point. PHP owns API, queue, state and output authorization."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from media import ROOT, report


def standard_task(request):
    from task_options import COMMON_DEFAULTS, prepare_task

    plan = request.get('resolved_plan')
    if plan is None:
        return prepare_task(request['regions'], '{}', default_preset='balanced')
    groups = {k: dict(v) for k, v in plan['parameters'].items() if k in ('clear', 'propainter')}
    if 'clear' in groups:
        # Earlier plan snapshots used the preset's conditioning rules.
        groups['clear'].setdefault('subtitle_guidance', 'preset')
    if not groups:
        raise ValueError('处理方案缺少首遍参数')
    first = next(iter(groups.values()))
    values = {**groups, **{k: first[k] for k in COMMON_DEFAULTS}}
    task = prepare_task(request['regions'], json.dumps(values))
    if (task['pipeline'] != [s for s in plan['stages'] if s in groups]
            or any(task['inference'][s] != group for s, group in groups.items())):
        raise ValueError('档位快照与实际推理参数不一致')
    return task


def record_execution(folder, request, mode, task):
    if mode == 'lite':
        stages = ['ocr_lama']
    else:
        stages = task['pipeline'] + (['selective_review'] if mode == 'max' else [])
    plan = request.get('resolved_plan')
    if plan is not None and (mode != plan['mode'] or stages != plan['stages']):
        raise ValueError('档位快照与任务模式或选区不一致')
    paths = ['php/inference/run.py', 'php/inference/media.py']
    entries = {
        'ocr_lama': ['php/inference/lite.py'],
        'clear': ['CLEAR-web/worker.py'],
        'propainter': ['CLEAR-web/propainter_worker.py'],
        'selective_review': ['php/inference/fine.py'],
    }
    if task:
        paths.append('CLEAR-web/pipeline_worker.py')
    for stage in stages:
        paths.extend(entries[stage])
    execution = {
        'mode': mode, 'stages': stages, 'catalog_sha256': plan and plan['catalog_sha256'],
        'run_id': os.environ.get('SUBTITLE_RUN_ID'),
        'parameters': plan['parameters'] if plan else (task or {}).get('inference'),
        'entrypoint_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
    }
    temporary = folder / 'execution.json.tmp'
    temporary.write_text(json.dumps(execution, ensure_ascii=False), encoding='utf-8')
    temporary.replace(folder / 'execution.json')


def standard(folder, request, fine, task):
    from pipeline_worker import atomic_json

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


def run(folder, mode=None):
    from regions import validate_regions

    request = json.loads((folder / 'request.json').read_text())
    validate_regions(request['regions'], request['dimensions'])
    if mode is None:
        mode = request['mode']
    if mode not in ('lite', 'pro', 'max'):
        raise ValueError('未知质量档位')
    task = None if mode == 'lite' else standard_task(request)
    record_execution(folder, request, mode, task)
    if mode == 'lite':
        from lite import run as run_lite

        run_lite(folder, request)
    else:
        fine = mode == 'max'
        first = standard(folder, request, fine, task)
        if fine:
            from fine import refine

            refine(folder, request, first)
        else:
            first.replace(folder / 'result.mp4')


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]).resolve(), *sys.argv[2:])
    except Exception as exc:
        print('JOB_STATUS ' + json.dumps({'error': str(exc)[:540]}, ensure_ascii=False), flush=True)
        raise
