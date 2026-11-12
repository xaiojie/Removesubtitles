import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock, Thread
from uuid import uuid4

from flask import Flask, abort, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
MODELS = WORKSPACE / 'CLEAR-models'
WAN = MODELS / 'Wan2.1-T2V-1.3B'
REQUIRED = {
    WAN / 'fun-control.safetensors': 3129305312,
    WAN / 'Wan2.1_VAE.pth': 507609880,
    WAN / 'models_t5_umt5-xxl-enc-bf16.pth': 11361920418,
    WAN / 'google/umt5-xxl/tokenizer.json': 16837417,
    MODELS / 'CLEAR-mask-free-subtitle-removal.pt': 175181778,
}
SOURCE = WORKSPACE / 'CLEAR'
DATA = ROOT / 'jobs'
DATA.mkdir(exist_ok=True)
tempfile.tempdir = str(DATA)
HIDE = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=8 * 1024**3, TRUSTED_HOSTS=['localhost', '127.0.0.1'])
jobs = {}
lock = Lock()


def region(raw_roi, raw_dimensions):
    try:
        roi = json.loads(raw_roi)
        dimensions = json.loads(raw_dimensions)
    except (ValueError, TypeError):
        raise ValueError('字幕区域无效。') from None
    if not isinstance(roi, list) or not isinstance(dimensions, list):
        raise ValueError('字幕区域无效。')
    if len(roi) != 4 or len(dimensions) != 2 or any(type(n) is not int for n in roi + dimensions):
        raise ValueError('字幕区域无效。')
    x1, y1, x2, y2 = roi
    width, height = dimensions
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2 - x1 >= 2 and y2 - y1 >= 2):
        raise ValueError('字幕区域超出视频画面或过小。')
    return roi, dimensions


def gpu_status():
    try:
        report = subprocess.check_output(
            ['nvidia-smi', '-i', '0', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            text=True, timeout=3, stderr=subprocess.DEVNULL, creationflags=HIDE,
        )
        name, util, used, total = [field.strip() for field in report.splitlines()[0].split(',')]
        return dict(name=name, util=int(util), used=int(used), total=int(total))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def model_status():
    completed = 0
    received = 0
    for path, size in REQUIRED.items():
        length = path.stat().st_size if path.exists() else 0
        if length == size:
            completed += 1
        elif not length:
            parts = list(path.parent.glob(path.stem + '.range-*'))
            parts.append(path.with_suffix(path.suffix + '.prefix'))
            for part in parts:
                try:
                    length += part.stat().st_size
                except FileNotFoundError:
                    continue
        received += min(length, size)
    if completed < len(REQUIRED):
        total = sum(REQUIRED.values())
        return False, f'下载中 · {received / 1e9:.1f} / {total / 1e9:.1f} GB'
    if not (SOURCE / 'inference.py').is_file():
        return False, '等待 CLEAR 源码'
    if not (ROOT / 'models.ready').is_file():
        return False, '正在校验权重'
    if not (ROOT / 'runtime.ready').is_file():
        return False, '正在验证 CUDA 环境'
    return True, '已就绪'


@app.after_request
def prevent_cache(response):
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/')
def index():
    return app.send_static_file('index.html')


@app.get('/api/system')
def system():
    ready, model = model_status()
    gpu = gpu_status()
    return jsonify(ready=ready and gpu is not None, model=model, gpu=gpu)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='视频超过 8 GiB。'), 413


def set_job(job_id, **changes):
    with lock:
        jobs[job_id].update(changes)


def run_job(job_id, folder):
    set_job(job_id, stage='正在加载 CLEAR 模型', percent=5)
    try:
        with (folder / 'worker.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(
                [sys.executable, '-u', str(ROOT / 'worker.py'), str(folder), str(SOURCE)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'HF_HUB_OFFLINE': '1',
                     'PYTHONPATH': str(WORKSPACE / 'CLEAR-runtime')},
                encoding='utf-8', errors='replace', creationflags=HIDE,
            ) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith('CLEAR_STATUS '):
                        set_job(job_id, **json.loads(line[len('CLEAR_STATUS '):]))
                if process.wait():
                    raise RuntimeError('处理失败，详情见任务日志。')
        if not (folder / 'output.mp4').is_file():
            raise RuntimeError('视频导出失败。')
        set_job(job_id, state='done', percent=100, stage='处理完成')
    except Exception as exc:
        with lock:
            reason = jobs[job_id].get('error')
        set_job(job_id, state='failed', stage='处理失败', error=reason or str(exc))
    finally:
        (folder / 'source.video').unlink(missing_ok=True)
        (folder / 'partial.mp4').unlink(missing_ok=True)


@app.post('/api/jobs')
def submit():
    if not model_status()[0]:
        return jsonify(error='CLEAR 模型或 CUDA 环境仍在准备。'), 503
    if request.headers.get('Origin', request.host_url.rstrip('/')) != request.host_url.rstrip('/'):
        abort(403)
    try:
        roi, dimensions = region(request.form.get('roi'), request.form.get('dimensions'))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    video = request.files.get('video')
    if video is None or not video.filename:
        return jsonify(error='请选择视频。'), 400

    job_id = uuid4().hex
    with lock:
        if any(job['state'] == 'running' for job in jobs.values()):
            return jsonify(error='GPU 正在处理另一个视频。'), 409
        jobs[job_id] = dict(state='running', stage='正在保存视频', percent=4)
    folder = DATA / job_id
    try:
        folder.mkdir()
        video.save(folder / 'source.video')
        (folder / 'config.json').write_text(json.dumps(dict(roi=roi, dimensions=dimensions)), encoding='utf-8')
        Thread(target=run_job, args=(job_id, folder), daemon=True).start()
    except OSError:
        (folder / 'source.video').unlink(missing_ok=True)
        set_job(job_id, state='failed', stage='保存失败')
        return jsonify(error='无法保存视频，请检查工作区是否可写及磁盘空间。'), 500
    return jsonify(id=job_id), 202


@app.get('/api/jobs/<job_id>')
def status(job_id):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error='任务已结束或本地服务已重启。'), 404
        return jsonify(job.copy())


@app.get('/api/jobs/<job_id>/output')
def output(job_id):
    with lock:
        if jobs.get(job_id, {}).get('state') != 'done':
            abort(404)
    return send_file(DATA / job_id / 'output.mp4', as_attachment=True,
                     download_name='CLEAR-processed.mp4', mimetype='video/mp4')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8765, threaded=True)
