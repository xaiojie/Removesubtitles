"""Video Subtitle Remover - Web 服务

在原有 CLI/算法基础上包一层 Web 服务：
  * 上传视频/图片
  * 自定义框选字幕区域（前端框选，归一化坐标回传，按原分辨率换算成像素）
  * 选择重绘算法 / 检测精度 / 高级参数
  * 任务队列、实时进度、日志、在线预览、下载、历史记录
  * GPU 状态（利用率、显存、温度、功耗）实时展示

后端不做任何算法改动，通过调用 `python backend/main.py` 子进程执行，
每个任务独立进程 + 独立进程组，可随时取消，互不干扰。
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent

IS_WIN = os.name == 'nt'

DATA_DIR = Path(os.environ.get('VSR_DATA_DIR') or (REPO_DIR / 'vsr-data')).resolve()
UPLOAD_DIR = DATA_DIR / 'uploads'
OUTPUT_DIR = DATA_DIR / 'outputs'
THUMB_DIR = DATA_DIR / 'thumbs'
STATE_FILE = DATA_DIR / 'tasks_state.json'

for _d in (UPLOAD_DIR, OUTPUT_DIR, THUMB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PYTHON = os.environ.get('VSR_PYTHON', sys.executable)
MAIN_PY = REPO_DIR / 'backend' / 'main.py'
# Match the worker's system FFmpeg selection; the Web UI can start before FFmpeg is installed.
FFMPEG = shutil.which(os.environ.get('VSR_FFMPEG') or 'ffmpeg') or os.environ.get('VSR_FFMPEG') or 'ffmpeg'
WORKERS = max(1, int(os.environ.get('VSR_WORKERS', '1')))
MAX_UPLOAD_MB = int(os.environ.get('VSR_MAX_UPLOAD_MB', '8192'))
CHUNK = 4 * 1024 * 1024

VIDEO_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v',
             '.ts', '.mpg', '.mpeg', '.rmvb', '.3gp', '.m2ts'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
INPAINT_MODES = ('sttn-auto', 'sttn-det', 'lama', 'propainter', 'opencv')
DETECT_MODES = ('PP_OCRv5_SERVER', 'PP_OCRv5_MOBILE')

ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')
PCT_RE = re.compile(r'(\d{1,3})%\|')
FRAME_RE = re.compile(r'(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([^\]]+)\]')

# ---------------------------------------------------------------------------
# 任务状态存储
# ---------------------------------------------------------------------------
_lock = threading.RLock()
_cond = threading.Condition(_lock)
_tasks = {}
_pending = []
_procs = {}
_last_save = 0.0
_workers_started = False


def _save_state(force=False):
    global _last_save
    now = time.time()
    if not force and now - _last_save < 2:
        return
    _last_save = now
    with _lock:
        payload = {tid: {k: v for k, v in t.items() if k != '_proc'} for tid, t in _tasks.items()}
    tmp = str(STATE_FILE) + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except Exception:
        traceback.print_exc()


def _load_state():
    if not STATE_FILE.exists():
        return
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return
    for tid, t in data.items():
        if t.get('status') in ('running', 'queued'):
            t['status'] = 'failed'
            t['error'] = '服务重启，任务已中断'
            t['finished'] = t.get('finished') or time.time()
            t.setdefault('log', []).append('[system] 服务重启，任务已中断')
        _tasks[tid] = t


def _public(task):
    out = {k: v for k, v in task.items() if not k.startswith('_')}
    out['queue_position'] = _pending.index(task['id']) + 1 if task['id'] in _pending else 0
    # 原素材的静态 URL，供前端"原片 vs 处理后"对比播放
    try:
        out['src_url'] = '/media/' + str(Path(task['src']).relative_to(DATA_DIR))
    except Exception:
        out['src_url'] = ''
    return out


def _get_task(tid):
    with _lock:
        t = _tasks.get(tid)
    if not t:
        raise HTTPException(status_code=404, detail='任务不存在')
    return t


def _log(task, line, keep=600):
    line = ANSI_RE.sub('', line).replace('\x00', '')
    task.setdefault('log', []).append(line)
    if len(task['log']) > keep:
        del task['log'][:len(task['log']) - keep]


# ---------------------------------------------------------------------------
# 媒体探测
# ---------------------------------------------------------------------------
def _probe(path: Path, kind: str):
    info = {'kind': kind, 'width': 0, 'height': 0, 'duration': 0.0, 'fps': 0.0}
    if kind == 'image':
        try:
            import cv2
            img = cv2.imread(str(path))
            if img is not None:
                info['height'], info['width'] = img.shape[:2]
        except Exception:
            pass
        return info
    try:
        out = subprocess.run([str(FFMPEG), '-hide_banner', '-i', str(path)],
                             capture_output=True, text=True, timeout=60).stderr
    except Exception:
        return info
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.?\d*)', out)
    if m:
        info['duration'] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r'Stream #\d+:\d+.*?Video:.*?(\d{2,5})x(\d{2,5})', out)
    if m:
        info['width'], info['height'] = int(m.group(1)), int(m.group(2))
    m = re.search(r'([\d.]+)\s*fps', out)
    if m:
        info['fps'] = float(m.group(1))
    return info


def _make_thumb(src: Path, dst: Path, kind: str, seek=0.0):
    try:
        if kind == 'image':
            cmd = [str(FFMPEG), '-y', '-loglevel', 'error', '-i', str(src),
                   '-vf', "scale='min(1600,iw)':-2", '-frames:v', '1', str(dst)]
        else:
            cmd = [str(FFMPEG), '-y', '-loglevel', 'error', '-ss', f'{seek:.2f}', '-i', str(src),
                   '-vf', "scale='min(1600,iw)':-2", '-frames:v', '1', str(dst)]
        subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        traceback.print_exc()


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
_gpu_cache = {'ts': 0.0, 'data': None}


def _gpu_info():
    now = time.time()
    if _gpu_cache['data'] is not None and now - _gpu_cache['ts'] < 2:
        return _gpu_cache['data']
    data = {'available': False, 'gpus': []}
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,'
             'temperature.gpu,power.draw,power.limit,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5).stdout.strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 9:
                continue
            data['gpus'].append({
                'index': parts[0], 'name': parts[1],
                'mem_total': float(parts[2]), 'mem_used': float(parts[3]),
                'mem_free': float(parts[4]), 'util': float(parts[5]),
                'temp': float(parts[6]),
                'power': float(parts[7]) if parts[7] not in ('[N/A]', 'N/A', '') else 0.0,
                'power_limit': float(parts[8]) if parts[8] not in ('[N/A]', 'N/A', '') else 0.0,
                'driver': parts[9] if len(parts) > 9 else '',
            })
        data['available'] = bool(data['gpus'])
    except Exception:
        pass
    _gpu_cache['ts'] = now
    _gpu_cache['data'] = data
    return data


# ---------------------------------------------------------------------------
# 执行任务
# ---------------------------------------------------------------------------
def _build_command(task):
    params = task['params']
    src = Path(task['src'])
    stem = Path(task['name']).stem
    ext = Path(task['name']).suffix or '.mp4'
    out_dir = OUTPUT_DIR / task['id']
    out_dir.mkdir(parents=True, exist_ok=True)
    if task['kind'] == 'image':
        out_path = out_dir / f'{stem}_no_sub{ext}'
    else:
        out_path = out_dir / f'{stem}_no_sub.mp4'

    cmd = [PYTHON, str(MAIN_PY), '-i', str(src), '-o', str(out_path),
           '--inpaint-mode', params['mode']]
    # 自定义字幕区域（像素坐标，顺序 ymin ymax xmin xmax）
    for area in params.get('areas') or []:
        ymin, ymax, xmin, xmax = area
        cmd += ['-c', str(int(ymin)), str(int(ymax)), str(int(xmin)), str(int(xmax))]

    env = dict(os.environ)
    env.update({
        'PYTHONUNBUFFERED': '1',
        'PYTHONIOENCODING': 'utf-8',
        'VSR_INPAINT_MODE': params['mode'],
        'VSR_DETECT_MODE': params.get('detect_mode', 'PP_OCRv5_SERVER'),
        'VSR_TL_BACKWARD': str(params.get('tl_backward', 3)),
        'VSR_TL_FORWARD': str(params.get('tl_forward', 3)),
        'VSR_STTN_NEIGHBOR_STRIDE': str(params.get('sttn_stride', 5)),
        'VSR_STTN_REFERENCE_LENGTH': str(params.get('sttn_ref', 10)),
        'VSR_STTN_MAX_LOAD_NUM': str(params.get('sttn_max', 50)),
        'VSR_PROPAINTER_MAX_LOAD_NUM': str(params.get('propainter_max', 70)),
        'VSR_HW_ACCEL': '1' if params.get('hw', True) else '0',
        'KMP_DUPLICATE_LIB_OK': 'TRUE',
    })
    return cmd, env, out_path


def _handle_output(task, raw):
    line = raw.strip()
    if not line:
        return
    _log(task, line)
    task['last_line'] = line[:300]

    if line.startswith('Subtitle removal model:'):
        task['stage'] = '模型已加载：' + line.split(':', 1)[1].strip()
        task['progress'] = max(task['progress'], 3)
    elif 'Detecting subtitles' in line:
        task['stage'] = '正在检测字幕位置…'
        task['progress'] = max(task['progress'], 5)
    elif line.startswith('Subtitle detected:'):
        task['stage'] = '字幕检测完成，准备重绘'
        task['progress'] = max(task['progress'], 10)
    elif 'Removing subtitles' in line:
        task['stage'] = '正在去除字幕（AI 重绘）'
        task['progress'] = max(task['progress'], 10)
    elif 'Processing time:' in line:
        task['stage'] = '处理完成'
    elif 'FinishedProcessing' in line or 'Subtitles removed' in line:
        task['stage'] = '正在合并音轨/写出文件…'
        task['progress'] = 97

    m = PCT_RE.search(line)
    if m:
        pct = min(100, int(m.group(1)))
        task['progress'] = max(task['progress'], 10 + pct * 0.85)
        f = FRAME_RE.search(line)
        if f:
            task['detail'] = f'{f.group(1)}/{f.group(2)} 帧 · {f.group(5).strip()} · 剩余 {f.group(4)}'
        else:
            task['detail'] = f'{pct}%'


def _stream_output(task, proc):
    fd = proc.stdout.fileno()
    buf = ''
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk.decode('utf-8', 'replace')
        while True:
            m = re.search(r'[\r\n]', buf)
            if not m:
                break
            line, buf = buf[:m.start()], buf[m.end():]
            _handle_output(task, line)
        if len(buf) > 65536:
            _handle_output(task, buf)
            buf = ''
    if buf.strip():
        _handle_output(task, buf)


def _execute(task_id):
    task = _get_task(task_id)
    cmd, env, out_path = _build_command(task)

    with _lock:
        task['status'] = 'running'
        task['started'] = time.time()
        task['progress'] = 1.0
        task['stage'] = '启动中…'
        task['detail'] = ''
        _log(task, f'[system] {" ".join(cmd)}')
    _save_state(force=True)

    proc = None
    try:
        spawn_kwargs = {}
        if IS_WIN:
            spawn_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            spawn_kwargs['start_new_session'] = True
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, **spawn_kwargs)
        with _lock:
            _procs[task_id] = proc
        _stream_output(task, proc)
        code = proc.wait()

        with _lock:
            _procs.pop(task_id, None)
            if task['status'] == 'canceled':
                task['finished'] = time.time()
                return
        if code != 0:
            raise RuntimeError(f'处理失败（退出码 {code}）')
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError('处理结束但未生成输出文件，请查看日志')

        size = out_path.stat().st_size
        with _lock:
            task['status'] = 'done'
            task['progress'] = 100.0
            task['stage'] = '完成'
            task['detail'] = ''
            task['finished'] = time.time()
            task['output'] = {
                'file': str(out_path),
                'url': f"/media/outputs/{task_id}/{out_path.name}",
                'name': out_path.name,
                'size': size,
            }
        _log(task, f'[system] 输出文件：{out_path.name}（{size / 1048576:.1f} MB）')
    except Exception as e:
        with _lock:
            if task['status'] != 'canceled':
                task['status'] = 'failed'
                task['error'] = str(e)
                task['finished'] = time.time()
                task['stage'] = '失败'
        _log(task, f'[error] {e}')
        traceback.print_exc()
    finally:
        with _lock:
            _procs.pop(task_id, None)
        _save_state(force=True)


def _worker_loop():
    while True:
        with _cond:
            while not _pending:
                _cond.wait()
            task_id = _pending.pop(0)
            task = _tasks.get(task_id)
            # 若任务在排队期间被取消/删除，直接跳过
            if not task or task['status'] != 'queued':
                continue
        try:
            _execute(task_id)
        except Exception:
            traceback.print_exc()


def _enqueue(task_id):
    with _cond:
        _pending.append(task_id)
        _cond.notify()


def _start_workers():
    global _workers_started
    with _lock:
        if _workers_started:
            return
        _workers_started = True
    for i in range(WORKERS):
        threading.Thread(target=_worker_loop, name=f'vsr-worker-{i}', daemon=True).start()


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title='Video Subtitle Remover Web', docs_url='/api/docs', redoc_url=None)
app.mount('/media', StaticFiles(directory=str(DATA_DIR)), name='media')

_load_state()
_start_workers()


@app.get('/', response_class=HTMLResponse)
def index():
    return (APP_DIR / 'static' / 'index.html').read_text(encoding='utf-8')


@app.get('/api/system')
def system():
    with _lock:
        running = sum(1 for t in _tasks.values() if t['status'] == 'running')
        queued = sum(1 for t in _tasks.values() if t['status'] == 'queued')
        done = sum(1 for t in _tasks.values() if t['status'] == 'done')
        total = len(_tasks)
    return {
        'gpu': _gpu_info(),
        'queue': {'running': running, 'queued': queued, 'done': done, 'total': total,
                  'workers': WORKERS},
        'modes': list(INPAINT_MODES),
        'detect_modes': list(DETECT_MODES),
        'max_upload_mb': MAX_UPLOAD_MB,
    }


@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or 'upload')
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXT and ext not in IMAGE_EXT:
        raise HTTPException(400, f'不支持的文件类型：{ext or "未知"}（支持视频与图片）')
    kind = 'image' if ext in IMAGE_EXT else 'video'

    tid = uuid.uuid4().hex[:12]
    dst = UPLOAD_DIR / f'{tid}{ext}'
    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    try:
        with open(dst, 'wb') as f:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f'文件超过 {MAX_UPLOAD_MB} MB 限制')
                f.write(chunk)
    except HTTPException:
        dst.unlink(missing_ok=True)
        raise
    except Exception as e:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, f'上传失败：{e}')

    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, '空文件')

    info = _probe(dst, kind)
    thumb = THUMB_DIR / f'{tid}.jpg'
    _make_thumb(dst, thumb, kind)

    task = {
        'id': tid, 'name': name, 'kind': kind, 'size': size,
        'width': info['width'], 'height': info['height'],
        'duration': info['duration'], 'fps': info['fps'],
        'status': 'ready', 'progress': 0.0, 'stage': '待处理', 'detail': '',
        'created': time.time(), 'src': str(dst),
        'thumb': f'/media/thumbs/{tid}.jpg' if thumb.exists() else None,
        'log': [], 'params': None, 'output': None, 'error': '',
    }
    with _lock:
        _tasks[tid] = task
    _save_state(force=True)
    return _public(task)


@app.post('/api/tasks/{tid}/start')
def start(tid: str, payload: dict):
    task = _get_task(tid)
    with _lock:
        if task['status'] in ('queued', 'running'):
            raise HTTPException(400, '任务已在队列中')
        mode = payload.get('mode', 'sttn-auto')
        if mode not in INPAINT_MODES:
            raise HTTPException(400, f'未知算法：{mode}')
        detect = payload.get('detect_mode', 'PP_OCRv5_SERVER')
        if detect not in DETECT_MODES:
            detect = 'PP_OCRv5_SERVER'

        w, h = task['width'], task['height']
        areas = []
        for a in payload.get('areas') or []:
            try:
                ymin, ymax, xmin, xmax = [float(x) for x in a]
            except Exception:
                continue
            # 前端传归一化 0~1，服务端按原分辨率换算成像素
            if max(ymin, ymax, xmin, xmax) <= 1.0001:
                areas.append([int(ymin * h), int(ymax * h), int(xmin * w), int(xmax * w)])
            else:
                areas.append([int(ymin), int(ymax), int(xmin), int(xmax)])
        # 修正区间顺序与边界
        clean = []
        for ymin, ymax, xmin, xmax in areas:
            ymin, ymax = sorted((max(0, ymin), min(h, ymax)))
            xmin, xmax = sorted((max(0, xmin), min(w, xmax)))
            if ymax - ymin >= 4 and xmax - xmin >= 4:
                clean.append([ymin, ymax, xmin, xmax])

        params = {
            'mode': mode, 'detect_mode': detect, 'areas': clean,
            'tl_backward': int(payload.get('tl_backward', 3)),
            'tl_forward': int(payload.get('tl_forward', 3)),
            'sttn_stride': int(payload.get('sttn_stride', 5)),
            'sttn_ref': int(payload.get('sttn_ref', 10)),
            'sttn_max': int(payload.get('sttn_max', 50)),
            'propainter_max': int(payload.get('propainter_max', 70)),
            'hw': bool(payload.get('hw', True)),
        }
        task['params'] = params
        task['status'] = 'queued'
        task['progress'] = 0.0
        task['stage'] = '排队中'
        task['error'] = ''
        task['output'] = None
        task['log'] = []
        task.update({'started': None, 'finished': None})
    _save_state(force=True)
    _enqueue(tid)
    return _public(task)


@app.get('/api/tasks')
def list_tasks():
    with _lock:
        tasks = sorted(_tasks.values(), key=lambda t: t.get('created', 0), reverse=True)
        return [_public(t) for t in tasks]


@app.get('/api/tasks/{tid}')
def get_task(tid: str):
    return _public(_get_task(tid))


@app.post('/api/tasks/{tid}/cancel')
def cancel(tid: str):
    task = _get_task(tid)
    with _lock:
        if task['status'] == 'queued':
            if tid in _pending:
                _pending.remove(tid)
            task['status'] = 'canceled'
            task['stage'] = '已取消'
            task['finished'] = time.time()
            _save_state(force=True)
            return _public(task)
        proc = _procs.get(tid)
        if not proc:
            raise HTTPException(400, '任务未在运行')
        task['status'] = 'canceled'
        task['stage'] = '正在取消…'

    def _kill():
        def _signal_tree(force):
            if IS_WIN:
                cmd = ['taskkill', '/T', '/PID', str(proc.pid)]
                if force:
                    cmd.insert(1, '/F')
                subprocess.run(cmd, capture_output=True)
                return
            try:
                os.killpg(os.getpgid(proc.pid),
                          signal.SIGKILL if force else signal.SIGTERM)
            except Exception:
                pass

        _signal_tree(force=False)
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        _signal_tree(force=True)

    threading.Thread(target=_kill, daemon=True).start()
    return _public(task)


@app.delete('/api/tasks/{tid}')
def delete_task(tid: str):
    task = _get_task(tid)
    if task['status'] == 'running':
        raise HTTPException(400, '请先取消正在运行的任务')
    with _lock:
        _tasks.pop(tid, None)
    for p in (UPLOAD_DIR / f"{tid}{Path(task['name']).suffix.lower()}",
              THUMB_DIR / f'{tid}.jpg'):
        Path(p).unlink(missing_ok=True)
    shutil.rmtree(OUTPUT_DIR / tid, ignore_errors=True)
    _save_state(force=True)
    return {'ok': True}


@app.get('/api/tasks/{tid}/download')
def download(tid: str):
    task = _get_task(tid)
    out = task.get('output')
    if not out:
        raise HTTPException(404, '还没有生成结果')
    path = Path(out['file'])
    if not path.exists():
        raise HTTPException(404, '结果文件已丢失')
    return FileResponse(str(path), filename=out['name'], media_type='application/octet-stream')


@app.get('/api/tasks/{tid}/frame')
def frame(tid: str, t: float = 0.0):
    task = _get_task(tid)
    dst = THUMB_DIR / f'{tid}_t{t:.1f}.jpg'
    _make_thumb(Path(task['src']), dst, task['kind'], seek=t)
    if not dst.exists():
        raise HTTPException(500, '抽帧失败')
    return FileResponse(str(dst), media_type='image/jpeg')


@app.get('/api/health')
def health():
    return {'ok': True, 'ts': time.time()}
