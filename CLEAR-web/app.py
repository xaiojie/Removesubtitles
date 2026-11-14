from dataclasses import asdict
import json
import os
import re
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
from threading import Lock, Thread
from uuid import uuid4

from flask import Flask, abort, jsonify, request, send_file

from inference_options import InferenceOptions
from propainter_options import ProPainterOptions
from regions import region, parse_regions, bounds, regions_from_config
from task_options import prepare_task
from comparison_video import create_comparison, validate_video_pair

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("CLEAR_WORKSPACE", ROOT.parent)).resolve()
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
PROPAINTER_ROOT = Path(os.environ.get("PROPAINTER_ROOT", "/opt/propainter"))
PROPAINTER_PYTHON = Path(os.environ.get("PROPAINTER_PYTHON", "/opt/propainter-eval/venv/bin/python"))
PROPAINTER_WEIGHTS = {
    "ProPainter.pth": 157780510,
    "raft-things.pth": 21108000,
    "recurrent_flow_completion.pth": 20348681,
}
DATA = ROOT / 'jobs'
DATA.mkdir(exist_ok=True)
tempfile.tempdir = str(DATA)
HIDE = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
app = Flask(__name__)
trusted_hosts = os.environ.get('CLEAR_TRUSTED_HOSTS') or 'localhost,127.0.0.1'
app.config.update(
    MAX_CONTENT_LENGTH=8 * 1024**3,
    TRUSTED_HOSTS=[host.strip() for host in trusted_hosts.split(',') if host.strip()],
)
jobs = {}
comparisons = {}
lock = Lock()


def gpu_status():
    try:
        selected_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0].strip() or '0'
        report = subprocess.check_output(
            ['nvidia-smi', '-i', selected_gpu, '--query-gpu=name,utilization.gpu,memory.used,memory.total',
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
            parts.append(path.with_suffix(path.suffix + '.partial'))
            for part in parts:
                try:
                    details = part.stat()
                    if part.suffix == '.partial':
                        length += getattr(details, 'st_blocks', (details.st_size + 511) // 512) * 512
                    else:
                        length += details.st_size
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


def propainter_status():
    for name, expected_size in PROPAINTER_WEIGHTS.items():
        path = PROPAINTER_ROOT / "weights" / name
        if not path.is_file() or path.stat().st_size != expected_size:
            return False, "正在准备 ProPainter 模型"
    if not (PROPAINTER_ROOT / "inference_propainter.py").is_file():
        return False, "正在准备 ProPainter 程序"
    if not PROPAINTER_PYTHON.is_file() or not (PROPAINTER_ROOT / "runtime.ready").is_file():
        return False, "正在准备 ProPainter 运行环境"
    return True, "已就绪"


def engine_status(engine):
    if engine == "propainter":
        return propainter_status()
    return model_status()


def worker_command(folder):
    return [sys.executable, "-u", str(ROOT / "pipeline_worker.py"), str(folder),
            str(SOURCE), str(PROPAINTER_ROOT), str(PROPAINTER_PYTHON)]


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
    propainter_ready, propainter_model = propainter_status()
    engines = {
        "clear": dict(ready=ready and gpu is not None, model=model),
        "propainter": dict(ready=propainter_ready and gpu is not None, model=propainter_model),
    }
    return jsonify(ready=ready and gpu is not None, model=model, gpu=gpu,
                   inference_defaults=asdict(InferenceOptions()), region_types=["subtitle", "watermark"],
                   default_engine="auto", auto_routing=True,
                   routing={"subtitle": "clear", "watermark": "propainter"}, engines=engines,
                   propainter_defaults=asdict(ProPainterOptions()))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error='视频超过 8 GiB。'), 413


def save_configuration(folder, configuration):
    temporary = folder / "config.pending.json"
    try:
        temporary.write_text(json.dumps(configuration, ensure_ascii=False), encoding="utf-8")
        temporary.replace(folder / "config.json")
    finally:
        temporary.unlink(missing_ok=True)


def set_job(job_id, **changes):
    with lock:
        jobs[job_id].update(changes)
        if changes.get("state") in ("done", "failed"):
            folder = DATA / job_id
            if folder.is_dir():
                temporary = folder / "status.partial.json"
                try:
                    temporary.write_text(json.dumps(jobs[job_id], ensure_ascii=False), encoding="utf-8")
                    temporary.replace(folder / "status.json")
                except OSError:
                    pass
                finally:
                    temporary.unlink(missing_ok=True)


def run_job(job_id, folder):
    set_job(job_id, stage="正在准备自动处理流程", percent=5, can_retry=False)
    try:
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(
                worker_command(folder),
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
                     "PYTHONPATH": os.pathsep.join([str(ROOT), str(WORKSPACE / "CLEAR-runtime")])},
                encoding="utf-8", errors="replace", creationflags=HIDE,
            ) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("CLEAR_STATUS "):
                        set_job(job_id, **json.loads(line[len("CLEAR_STATUS "):]))
                if process.wait():
                    raise RuntimeError("处理失败，详情见任务日志。")
        if not (folder / "output.mp4").is_file():
            raise RuntimeError("视频导出失败。")
        set_job(job_id, state="done", percent=100, stage="处理完成", can_retry=False)
    except Exception as exc:
        with lock:
            reason = jobs[job_id].get("error")
        set_job(job_id, state="failed", stage="处理失败", error=reason or str(exc),
                can_retry=(folder / "source.video").is_file())
    finally:
        (folder / "partial.mp4").unlink(missing_ok=True)


@app.post("/api/jobs")
def submit():
    if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
        abort(403)
    try:
        regions, dimensions = parse_regions(request.form.get("regions"), request.form.get("roi"), request.form.get("dimensions"))
        task = prepare_task(regions, request.form.get("inference", "{}"), request.form.get("engine", "clear"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    for engine in task["pipeline"]:
        if not engine_status(engine)[0]:
            return jsonify(error=f"{engine} 模型或运行环境仍在准备。"), 503
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify(error="请选择视频。"), 400
    job_id = uuid4().hex
    with lock:
        if any(job["state"] == "running" for job in jobs.values()):
            return jsonify(error="GPU 正在处理另一个视频。"), 409
        jobs[job_id] = dict(state="running", stage="正在保存视频", percent=4,
                            regions=regions, can_retry=False, **task)
    folder = DATA / job_id
    try:
        folder.mkdir()
        video.save(folder / "source.video")
        configuration = dict(roi=bounds(regions), regions=regions, dimensions=dimensions,
                             original_filename=video.filename, **task)
        save_configuration(folder, configuration)
        Thread(target=run_job, args=(job_id, folder), daemon=True).start()
    except (OSError, RuntimeError):
        (folder / "source.video").unlink(missing_ok=True)
        set_job(job_id, state="failed", stage="保存失败", can_retry=False)
        return jsonify(error="无法保存视频，请检查工作区空间。"), 500
    return jsonify(id=job_id), 202


def completed_output(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        return None
    path = DATA / job_id / 'output.mp4'
    return path if path.is_file() else None


@app.get("/api/jobs/<job_id>")
def status(job_id):
    with lock:
        job = jobs.get(job_id)
        if job is not None:
            return jsonify(job.copy())
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return jsonify(error="Job not found."), 404
    folder = DATA / job_id
    config_path = folder / "config.json"
    if not config_path.is_file():
        return jsonify(error="Job not found."), 404
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = dict(inference=config.get("inference", {}), regions=config.get("regions", []),
                    engine=config.get("engine", "clear"), pipeline=config.get("pipeline", [config.get("engine", "clear")]))
    if (folder / "output.mp4").is_file():
        return jsonify(state="done", percent=100, stage="处理完成", can_retry=False, **metadata)
    saved = folder / "status.json"
    state = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
    state.update(metadata)
    state.update(state="failed", stage=state.get("stage", "任务中断，可继续处理"),
                 percent=state.get("percent", 4), can_retry=(folder / "source.video").is_file())
    return jsonify(state)



@app.post("/api/jobs/<job_id>/retry")
def retry_job(job_id):
    if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
        abort(403)
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return jsonify(error="任务不存在。"), 404
    folder = DATA / job_id
    config_path = folder / "config.json"
    if not config_path.is_file():
        return jsonify(error="任务不存在。"), 404
    if (folder / "output.mp4").is_file():
        return jsonify(error="任务已完成。"), 409
    if not (folder / "source.video").is_file():
        return jsonify(error="原视频已不存在，请重新上传。"), 409
    config = json.loads(config_path.read_text(encoding="utf-8"))
    regions = regions_from_config(config)
    legacy = "auto" if "pipeline" in config else config.get("engine", "clear")
    task = prepare_task(regions, json.dumps(config.get("inference", {})), legacy)
    from pipeline_worker import completed_stages
    completed = completed_stages(folder)
    for engine in task["pipeline"]:
        if engine not in completed and not engine_status(engine)[0]:
            return jsonify(error=f"{engine} 模型或运行环境仍在准备。"), 503
    with lock:
        if any(job["state"] == "running" for job in jobs.values()):
            return jsonify(error="GPU 正在处理另一个视频。"), 409
        jobs[job_id] = dict(state="running", stage="正在继续处理", percent=4,
                            regions=regions, can_retry=False, **task)
    config.update(task)
    try:
        save_configuration(folder, config)
        Thread(target=run_job, args=(job_id, folder), daemon=True).start()
    except (OSError, RuntimeError):
        set_job(job_id, state="failed", stage="任务启动失败", can_retry=True)
        return jsonify(error="无法启动任务，已保留处理数据，请重试。"), 500
    return jsonify(id=job_id, percent=4, pipeline=task["pipeline"]), 202

@app.get('/api/jobs/<job_id>/output')
def output(job_id):
    path = completed_output(job_id)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=download_filename(path, '_处理后', 'CLEAR-processed.mp4'),
                     mimetype='video/mp4')


def download_filename(path, suffix, fallback):
    try:
        configuration = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        configuration = {}
    original_filename = configuration.get('original_filename')
    if not original_filename:
        return fallback
    stem = re.sub(r'[\x00-\x1f\x7f]', '', PureWindowsPath(original_filename).stem) or 'video'
    return stem + suffix + '.mp4'


def run_comparison(job_id, folder):
    try:
        create_comparison(folder / 'source.video', folder / 'output.mp4', folder / 'comparison.mp4')
        with lock:
            comparisons[job_id] = dict(state='done', stage='对比视频已就绪')
    except Exception as error:
        with lock:
            comparisons[job_id] = dict(state='failed', stage='对比视频生成失败', error=str(error)[:600])


@app.post('/api/jobs/<job_id>/comparison')
def start_comparison(job_id):
    path = completed_output(job_id)
    if path is None:
        return jsonify(error='处理结果不存在。'), 404
    if request.headers.get('Origin', request.host_url.rstrip('/')) != request.host_url.rstrip('/'):
        abort(403)
    folder = path.parent
    original = folder / 'source.video'
    upload = request.files.get('video')
    with lock:
        if (folder / 'comparison.mp4').is_file():
            return jsonify(state='done', url=f'/api/jobs/{job_id}/comparison/output')
        current = comparisons.get(job_id, {})
        if current.get('state') == 'running':
            return jsonify(current.copy()), 202
        if not original.is_file() and upload is None:
            return jsonify(error='请重新选择原视频以生成对比视频。', needs_original=True), 409
        comparisons[job_id] = dict(state='running', stage='正在生成左右对比视频')
    temporary = folder / 'comparison-source.partial.video'
    try:
        if not original.is_file():
            upload.save(temporary)
            validate_video_pair(temporary, path)
            temporary.replace(original)
        Thread(target=run_comparison, args=(job_id, folder), daemon=True).start()
    except (OSError, ValueError, RuntimeError) as error:
        temporary.unlink(missing_ok=True)
        with lock:
            comparisons[job_id] = dict(state='failed', stage='对比视频准备失败', error=str(error)[:600])
        return jsonify(error=str(error)[:600]), 400 if isinstance(error, ValueError) else 500
    return jsonify(state='running', stage='正在生成左右对比视频'), 202


@app.get('/api/jobs/<job_id>/comparison')
def comparison_status(job_id):
    path = completed_output(job_id)
    if path is None:
        return jsonify(error='处理结果不存在。'), 404
    if (path.parent / 'comparison.mp4').is_file():
        return jsonify(state='done', stage='对比视频已就绪', url=f'/api/jobs/{job_id}/comparison/output')
    with lock:
        current = comparisons.get(job_id, dict(state='pending', stage='等待生成对比视频')).copy()
    return jsonify(current)


@app.get('/api/jobs/<job_id>/comparison/output')
def comparison_output(job_id):
    path = completed_output(job_id)
    if path is None:
        abort(404)
    comparison = path.parent / 'comparison.mp4'
    if not comparison.is_file():
        return jsonify(error='对比视频尚未生成。'), 409
    return send_file(comparison, as_attachment=True,
                     download_name=download_filename(path, '_处理前后对比', 'CLEAR_处理前后对比.mp4'),
                     mimetype='video/mp4')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('CLEAR_PORT', '8765')), threaded=True)
