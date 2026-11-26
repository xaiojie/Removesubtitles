from dataclasses import asdict
import ipaddress
import json
import os
import re
import secrets
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import time
from uuid import uuid4

from flask import Flask, abort, g, jsonify, request, send_file

from inference_options import InferenceOptions
from propainter_options import ProPainterOptions
from ltx_options import LTXOptions
from regions import region, parse_regions, bounds, regions_from_config
from task_options import prepare_task
from subtitle_queue import QueueError, SubtitleQueue
from processing_profiles import default_profile, profile_catalog, profile_defaults
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
LTX_ROOT = Path(os.environ.get('LTX_ROOT', '/opt/LTX2-ICEdit-Insight'))
LTX_PYTHON = Path(os.environ.get('LTX_PYTHON', str(LTX_ROOT / '.venv/bin/python')))
MUSE_ROOT = Path(os.environ.get('MUSE_ROOT', '/opt/MuseTalk')).resolve()
MUSE_PYTHON = Path(os.environ.get('MUSE_PYTHON', str(MUSE_ROOT / '.venv/bin/python')))
PROPAINTER_WEIGHTS = {
    "ProPainter.pth": 157780510,
    "raft-things.pth": 21108000,
    "recurrent_flow_completion.pth": 20348681,
}
DATA = Path(os.environ.get('CLEAR_DATA_DIR', ROOT / 'jobs')).resolve()
DATA.mkdir(parents=True, exist_ok=True)
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
draining = False
active_requests = {}


@app.before_request
def check_web_auth():
    username, password = os.environ.get('MUSE_USER'), os.environ.get('MUSE_PASSWORD')
    if not (username or password):
        return
    credentials = request.authorization
    if (username and password and credentials and credentials.type == 'basic'
            and secrets.compare_digest((credentials.username or '').encode(), username.encode())
            and secrets.compare_digest((credentials.password or '').encode(), password.encode())):
        return
    return '请输入网页登录凭据。', 401, {'WWW-Authenticate': 'Basic realm="Video Workbench", charset="UTF-8"'}


@app.before_request
def admit_processing_request():
    if request.endpoint not in ('submit', 'retry_job', 'start_comparison'):
        return
    with lock:
        if draining:
            return jsonify(error='服务正在准备升级，请稍后再提交任务。'), 503
        admission_id = uuid4().hex
        active_requests[admission_id] = dict(id=admission_id, state='uploading', kind='request')
        g.deployment_admission = admission_id


@app.teardown_request
def finish_processing_request(_error):
    admission_id = g.pop('deployment_admission', None)
    if admission_id is not None:
        with lock:
            active_requests.pop(admission_id, None)


@app.route('/api/deploy/drain', methods=['GET', 'POST', 'DELETE'])
def deployment_drain():
    global draining
    try:
        peer = ipaddress.ip_address(request.remote_addr)
    except ValueError:
        abort(403)
    if not peer.is_loopback:
        server_socket = request.environ.get('gunicorn.socket') or request.environ.get('werkzeug.socket')
        try:
            address = server_socket.getsockname()[0] if server_socket else request.environ.get('SERVER_NAME')
            if peer != ipaddress.ip_address(address):
                abort(403)
        except (OSError, ValueError, TypeError):
            abort(403)
    try:
        token = (WORKSPACE / '.deploy/maintenance.token').read_text().strip()
    except OSError:
        abort(403)
    if not token or not secrets.compare_digest(token, request.headers.get('X-Deploy-Token', '')):
        abort(403)
    with lock:
        if request.method != 'GET':
            draining = request.method == 'POST'
        busy = [dict(id=job_id, state=job['state'], kind='job')
                for job_id, job in jobs.items() if job['state'] in ('running', 'queued', 'uploading')]
        busy.extend(dict(id=job_id, state='running', kind='comparison')
                    for job_id, comparison in comparisons.items() if comparison.get('state') == 'running')
        busy.extend(active_requests.values())
        return jsonify(draining=draining, busy=busy)


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


def ltx_modes():
    selection = LTX_ROOT / '.install/selection.json'
    if not selection.is_file():
        return ['subtitle_rm', 'watermark_rm']
    try:
        tasks = json.loads(selection.read_text(encoding='utf-8'))['tasks']
        if not isinstance(tasks, list) or not tasks or any(task not in ('subtitle', 'watermark') for task in tasks):
            return []
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [mode for task, mode in (('subtitle', 'subtitle_rm'), ('watermark', 'watermark_rm')) if task in tasks]


def ltx_status():
    if not (LTX_ROOT / 'run_pipeline.py').is_file() or not LTX_PYTHON.is_file():
        return False, '正在准备 LTX 程序与环境'
    for marker in ('models.complete', 'environment.complete'):
        if not (LTX_ROOT / '.install' / marker).is_file():
            return False, '正在校验 LTX 模型与环境'
    if not any((LTX_ROOT / '.install' / marker).is_file() for marker in ('runtime.complete', 'test.complete')):
        return False, '正在校验 LTX 模型与环境'
    if not ltx_modes():
        return False, 'LTX 模型用途配置无效，请重新校验安装'
    return True, 'LTX 已就绪（测试）'


def musetalk_status():
    if not MUSE_PYTHON.is_file() or not (MUSE_ROOT / 'musetalk/utils/preprocessing.py').is_file():
        return False, 'MuseTalk 运行环境未安装'
    weights = ['musetalkV15/unet.pth', 'musetalkV15/musetalk.json',
               'whisper/config.json', 'sd-vae/config.json', 'dwpose/dw-ll_ucoco_384.pth']
    if not all((MUSE_ROOT / 'models' / name).is_file() for name in weights):
        return False, 'MuseTalk 1.5 权重未就绪'
    return True, 'MuseTalk 1.5 已就绪'


def lip_sync_task(duration_mode):
    if duration_mode not in ('preview', 'full'):
        raise ValueError('生成范围请选择试听或完整。')
    return dict(engine='musetalk', pipeline=['musetalk'], inference={}, duration_mode=duration_mode)


def engine_status(engine):
    if engine == 'musetalk':
        return musetalk_status()
    if engine == 'ltx':
        return ltx_status()
    if engine == "propainter":
        return propainter_status()
    return model_status()


def worker_command(folder, engine=None):
    if engine == 'musetalk':
        return [str(MUSE_PYTHON), '-u', str(ROOT / 'musetalk_worker.py'), str(folder)]
    return [sys.executable, "-u", str(ROOT / "pipeline_worker.py"), str(folder),
            str(SOURCE), str(PROPAINTER_ROOT), str(PROPAINTER_PYTHON)]


@app.after_request
def prevent_cache(response):
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/')
def index():
    if os.environ.get('CLEAR_HOME_MODE') == 'lip_sync':
        return lip_sync_page()
    return remove_page()


@app.get('/remove')
def remove_page():
    return app.send_static_file('index.html')


@app.get('/lip-sync')
def lip_sync_page():
    return app.send_static_file('lip_sync.html')


@app.get('/api/system')
def system():
    ready, model = model_status()
    gpu = gpu_status()
    propainter_ready, propainter_model = propainter_status()
    ltx_ready, ltx_model = ltx_status()
    muse_ready, muse_model = musetalk_status()
    engines = {
        "clear": dict(ready=ready and gpu is not None, model=model),
        "propainter": dict(ready=propainter_ready and gpu is not None, model=propainter_model),
        'ltx': dict(ready=ltx_ready and gpu is not None, model=ltx_model),
        'musetalk': dict(ready=muse_ready and gpu is not None, model=muse_model),
    }
    return jsonify(ready=ready and gpu is not None, model=model, gpu=gpu,
                   inference_defaults=asdict(InferenceOptions.parse(json.dumps(profile_defaults(default_profile())))),
                   clear_presets=profile_catalog(), clear_default_preset=default_profile(),
                   draining=draining, region_types=["subtitle", "watermark"],
                   product_modes_url="/api/modes", default_engine="auto", auto_routing=True,
                   routing={"subtitle": "clear", "watermark": "propainter"}, engines=engines,
                   propainter_defaults=asdict(ProPainterOptions()), ltx_defaults=asdict(LTXOptions()),
                   ltx_modes=ltx_modes(),
                   subtitle_models=['clear', 'ltx'], batch_queue=True,
                   clear_processing_profile='clear-presets-v2')


@app.get('/api/modes')
def product_modes():
    catalog, status = SubtitleQueue(WORKSPACE).json_request('GET', '/api/modes')
    return jsonify(catalog), status


@app.errorhandler(QueueError)
def queue_error(error):
    return jsonify(error=str(error)), error.status


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
        if changes.get("state") in ("queued", "done", "failed"):
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


def dispatch_next():
    with lock:
        if any(job['state'] == 'running' for job in jobs.values()):
            return
        job_id = next((key for key, job in jobs.items() if job['state'] == 'queued'), None)
        if job_id is None:
            return
        jobs[job_id].update(state='running', stage='正在准备处理', can_retry=False)
    try:
        Thread(target=run_job, args=(job_id, DATA / job_id), daemon=True).start()
    except RuntimeError:
        set_job(job_id, state='failed', stage='任务启动失败', can_retry=True)
        dispatch_next()


def run_job(job_id, folder):
    engine = jobs[job_id].get('engine')
    is_lip_sync = engine == 'musetalk'
    started = time.monotonic()
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'HF_HUB_OFFLINE': '1',
           'PYTHONPATH': os.pathsep.join([str(ROOT), str(WORKSPACE / 'CLEAR-runtime')])}
    if is_lip_sync:
        env.update(TRANSFORMERS_OFFLINE='1',
                   PYTHONPATH=os.pathsep.join([str(MUSE_ROOT), str(MUSE_ROOT / 'musetalk/utils')]))
        set_job(job_id, stage='正在准备口型同步', percent=5, can_retry=False, started_at=time.time())
    else:
        set_job(job_id, stage="正在准备自动处理流程", percent=5, can_retry=False)
    try:
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            with subprocess.Popen(
                worker_command(folder, engine) if is_lip_sync else worker_command(folder),
                cwd=MUSE_ROOT if is_lip_sync else ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env,
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
        timing = {'elapsed_seconds': round(time.monotonic() - started, 2)} if is_lip_sync else {}
        set_job(job_id, state="done", percent=100, stage="处理完成", can_retry=False, **timing)
    except Exception as exc:
        with lock:
            reason = jobs[job_id].get("error")
        timing = {'elapsed_seconds': round(time.monotonic() - started, 2)} if is_lip_sync else {}
        set_job(job_id, state="failed", stage="处理失败", error=reason or str(exc), **timing,
                can_retry=(folder / "source.video").is_file()
                and (not is_lip_sync or (folder / 'source.audio').is_file()))
    finally:
        (folder / "partial.mp4").unlink(missing_ok=True)
        dispatch_next()


@app.post("/api/jobs")
def submit():
    if request.headers.get("Origin", request.host_url.rstrip("/")) != request.host_url.rstrip("/"):
        abort(403)
    if (request.headers.get('X-Subtitle-Workflow') == 'product'
            and request.mimetype == 'multipart/form-data' and request.content_length is not None):
        result, code = SubtitleQueue(WORKSPACE).json_request('POST', '/api/jobs', body=request.stream, headers={
            'Content-Type': request.content_type, 'Content-Length': str(request.content_length),
        })
        return jsonify(result), code
    mode = request.form.get('mode')
    # Existing custom-engine clients and the original single-roi form keep their explicit contract.
    legacy_fields = {'roi', 'engine', 'inference'}
    if mode != 'lip_sync' and (mode is not None or not legacy_fields.intersection(request.form)):
        video = request.files.get('video')
        if video is None or not video.filename:
            return jsonify(error='请选择视频。'), 400
        result, code = SubtitleQueue(WORKSPACE).submit(request.form, video)
        return jsonify(result), code
    audio = None
    try:
        if mode == 'lip_sync':
            task = lip_sync_task(request.form.get('duration_mode', 'full'))
            regions, dimensions = [], None
            audio = request.files.get('audio')
            if audio is None or not audio.filename:
                return jsonify(error='请选择配音文件。'), 400
        else:
            regions, dimensions = parse_regions(request.form.get("regions"), request.form.get("roi"), request.form.get("dimensions"))
            task = prepare_task(regions, request.form.get("inference", "{}"), request.form.get("engine", "clear"),
                                default_preset=default_profile())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    for engine in task["pipeline"]:
        if not engine_status(engine)[0]:
            return jsonify(error=f"{engine} 模型或运行环境仍在准备。"), 503
        if engine == 'ltx' and task['inference']['ltx']['mode'] not in ltx_modes():
            return jsonify(error='所选 LTX 用途的 LoRA 尚未安装，请补充对应模型。'), 503
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify(error="请选择视频。"), 400
    job_id = uuid4().hex
    enqueue = audio is not None or request.form.get('queue') == '1'
    with lock:
        if not enqueue and any(job["state"] == "running" for job in jobs.values()):
            return jsonify(error="GPU 正在处理另一个视频。"), 409
        jobs[job_id] = dict(state="uploading" if enqueue else "running", stage="正在保存视频", percent=4,
                            regions=regions, can_retry=False, **task)
    folder = DATA / job_id
    try:
        folder.mkdir()
        video.save(folder / "source.video")
        if audio is not None:
            audio.save(folder / 'source.audio')
        configuration = dict(roi=bounds(regions) if regions else None, regions=regions, dimensions=dimensions,
                             original_filename=video.filename, **task)
        save_configuration(folder, configuration)
        if enqueue:
            set_job(job_id, state='queued', stage='等待前面的视频完成', percent=0)
            dispatch_next()
        else:
            Thread(target=run_job, args=(job_id, folder), daemon=True).start()
    except (OSError, RuntimeError):
        (folder / "source.video").unlink(missing_ok=True)
        (folder / 'source.audio').unlink(missing_ok=True)
        set_job(job_id, state="failed", stage="保存失败", can_retry=False)
        dispatch_next()
        return jsonify(error="无法保存视频，请检查工作区空间。"), 500
    return jsonify(id=job_id, state=jobs[job_id]['state']), 202


def completed_output(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        return None
    path = DATA / job_id / 'output.mp4'
    if path.is_file():
        return path
    queue = SubtitleQueue(WORKSPACE)
    folder = queue.data / job_id
    if (folder / 'request.json').is_file():
        job, _ = queue.json_request('GET', f'/api/jobs/{job_id}')
        if job['state'] != 'done':
            raise QueueError('成片尚未就绪', 409)
        result = folder / 'result.mp4'
        if not result.is_file():
            raise QueueError('成片文件缺失', 500)
        return result
    return None


@app.get("/api/jobs/<job_id>")
def status(job_id):
    with lock:
        job = jobs.get(job_id)
        if job is not None:
            current = job.copy()
            if current['state'] == 'running' and 'started_at' in current:
                current['elapsed_seconds'] = round(max(0, time.time() - current['started_at']), 2)
            return jsonify(current)
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return jsonify(error="Job not found."), 404
    folder = DATA / job_id
    config_path = folder / "config.json"
    if not config_path.is_file():
        result, code = SubtitleQueue(WORKSPACE).json_request('GET', f'/api/jobs/{job_id}')
        return jsonify(result), code
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = dict(inference=config.get("inference", {}), regions=config.get("regions", []),
                    engine=config.get("engine", "clear"), pipeline=config.get("pipeline", [config.get("engine", "clear")]))
    saved = folder / "status.json"
    state = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
    if config.get('engine') == 'musetalk':
        metadata['duration_mode'] = config['duration_mode']
        for field in ('elapsed_seconds', 'output_seconds'):
            if field in state:
                metadata[field] = state[field]
    if (folder / "output.mp4").is_file():
        quality_path = folder / 'quality.json'
        if quality_path.is_file():
            metadata['quality'] = json.loads(quality_path.read_text(encoding='utf-8'))
        return jsonify(state="done", percent=100, stage="处理完成", can_retry=False, **metadata)
    state.update(metadata)
    state.update(state="failed", stage=state.get("stage", "任务中断，可继续处理"),
                 percent=state.get("percent", 4), can_retry=(folder / "source.video").is_file()
                 and (config.get('engine') != 'musetalk' or (folder / 'source.audio').is_file()))
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
        result, code = SubtitleQueue(WORKSPACE).json_request('POST', f'/api/jobs/{job_id}/retry')
        return jsonify(result), code
    if (folder / "output.mp4").is_file():
        return jsonify(error="任务已完成。"), 409
    if not (folder / "source.video").is_file():
        return jsonify(error="原视频已不存在，请重新上传。"), 409
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get('engine') == 'musetalk':
        if not (folder / 'source.audio').is_file():
            return jsonify(error='配音文件已不存在，请重新上传。'), 409
        regions, completed = [], []
        task = lip_sync_task(config['duration_mode'])
    else:
        regions = regions_from_config(config)
        legacy = "auto" if "pipeline" in config else config.get("engine", "clear")
        task = prepare_task(regions, json.dumps(config.get("inference", {})), legacy)
        if config.get('pipeline') == task['pipeline'] and config.get('engine') == task['engine']:
            task['inference'] = config['inference']
        from pipeline_worker import completed_stages
        completed = completed_stages(folder)
    for engine in task["pipeline"]:
        if engine not in completed and not engine_status(engine)[0]:
            return jsonify(error=f"{engine} 模型或运行环境仍在准备。"), 503
        if engine == 'ltx' and engine not in completed and task['inference']['ltx']['mode'] not in ltx_modes():
            return jsonify(error='所选 LTX 用途的 LoRA 尚未安装，请补充对应模型。'), 503
    enqueue = task['engine'] == 'musetalk' or request.args.get('queue') == '1'
    with lock:
        if jobs.get(job_id, {}).get('state') in ('running', 'queued', 'uploading'):
            return jsonify(error='任务已经在处理队列中。'), 409
        if not enqueue and any(job["state"] == "running" for job in jobs.values()):
            return jsonify(error="GPU 正在处理另一个视频。"), 409
        jobs[job_id] = dict(state="uploading" if enqueue else "running", stage="正在继续处理", percent=4,
                            regions=regions, can_retry=False, **task)
    config.update(task)
    try:
        save_configuration(folder, config)
        if enqueue:
            set_job(job_id, state='queued', stage='等待继续处理', percent=0)
            dispatch_next()
        else:
            Thread(target=run_job, args=(job_id, folder), daemon=True).start()
    except (OSError, RuntimeError):
        set_job(job_id, state="failed", stage="任务启动失败", can_retry=True)
        dispatch_next()
        return jsonify(error="无法启动任务，已保留处理数据，请重试。"), 500
    return jsonify(id=job_id, percent=4, pipeline=task["pipeline"]), 202

@app.get('/api/jobs/<job_id>/output')
def output(job_id):
    path = completed_output(job_id)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=request.args.get('inline') != '1',
                     download_name=download_filename(path, '_处理后', 'CLEAR-processed.mp4'),
                     mimetype='video/mp4')


def download_filename(path, suffix, fallback):
    try:
        config = 'request.json' if path.name == 'result.mp4' else 'config.json'
        configuration = json.loads((path.parent / config).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        configuration = {}
    original_filename = configuration.get('original_filename')
    if not original_filename:
        return fallback
    stem = re.sub(r'[\x00-\x1f\x7f]', '', PureWindowsPath(original_filename).stem) or 'video'
    return stem + suffix + '.mp4'


def run_comparison(job_id, processed):
    folder = processed.parent
    try:
        create_comparison(folder / 'source.video', processed, folder / 'comparison.mp4')
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
        Thread(target=run_comparison, args=(job_id, path), daemon=True).start()
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
