import { selection } from './selection.mjs';

const $ = (id) => document.getElementById(id);
const video = $('video');
const resultVideo = $('result-video');
const players = $('players');
const audio = $('audio');
const overlay = $('overlay');
const box = $('box');
const timeline = $('timeline');
const fileInput = $('file');
const settings = $('processing-options');
const startTime = $('start-seconds');
const duration = $('duration-seconds');
const tuningFields = [...settings.querySelectorAll('[data-option]')];
let comparisonStart = 0;
let defaultsLoaded = false;
let supportsSettings = false;
let supportsSegmentation = false;
let url, file, roi, drag, job;
let active = false;
let ready = false;
let comparing = false;
const clock = (seconds) => [Math.floor(seconds / 60), Math.floor(seconds % 60)]
  .map(n => String(n).padStart(2, '0')).join(':');

function refreshButtons() {
  $('submit').disabled = !roi || active || !ready;
  $('reset').disabled = !roi || active;
  fileInput.disabled = active;
  settings.disabled = active || players.hidden || !supportsSettings;
}

function showSelection(rectangle) {
  box.hidden = !rectangle;
  if (!rectangle) return;
  const [left, top, right, bottom] = rectangle;
  Object.assign(box.style, {
    left: (100 * left / video.videoWidth) + '%',
    top: (100 * top / video.videoHeight) + '%',
    width: (100 * (right - left) / video.videoWidth) + '%',
    height: (100 * (bottom - top) / video.videoHeight) + '%',
  });
}

function updateSelection(value) {
  roi = value;
  $('roi').textContent = roi ? ((roi[2] - roi[0]) + ' × ' + (roi[3] - roi[1]) + ' 像素') : '等待框选';
  showSelection(roi);
  refreshButtons();
}

function point(event) {
  const bounds = overlay.getBoundingClientRect();
  return [(event.clientX - bounds.left) / bounds.width, (event.clientY - bounds.top) / bounds.height];
}
function currentSelection(event) {
  return selection(drag.start, point(event), video.videoWidth, video.videoHeight);
}

overlay.onpointerdown = (event) => {
  if (active || drag || event.button !== 0 || !video.videoWidth) return;
  video.pause();
  overlay.setPointerCapture(event.pointerId);
  drag = { id: event.pointerId, start: point(event) };
};
overlay.onpointermove = (event) => {
  if (drag?.id === event.pointerId) showSelection(currentSelection(event) || roi);
};
overlay.onpointerup = (event) => {
  if (drag?.id !== event.pointerId) return;
  const next = currentSelection(event);
  drag = null;
  updateSelection(next || roi);
};
overlay.onlostpointercapture = () => {
  drag = null;
  showSelection(roi);
};

function showError(message = '') {
  $('error').textContent = message;
  $('error').hidden = !message;
}
function showProgress(percent, stage) {
  $('progress').value = percent;
  $('percent').textContent = percent + '%';
  $('status').textContent = stage;
}

function syncResult(force = false) {
  if (!comparing || resultVideo.readyState === 0) return;
  const position = Math.min(video.currentTime, resultVideo.duration);
  if (force || Math.abs(resultVideo.currentTime - position) > 0.12) {
    resultVideo.currentTime = position;
  }
}

function playResult() {
  if (!comparing || video.paused || video.seeking || video.readyState < 3) return;
  syncResult();
  if (resultVideo.paused && video.currentTime < resultVideo.duration) {
    resultVideo.play().catch(() => {});
  }
}

function clearComparison() {
  comparing = false;
  resultVideo.onloadeddata = null;
  resultVideo.onerror = null;
  resultVideo.pause();
  resultVideo.removeAttribute('src');
  resultVideo.load();
  players.classList.remove('compare');
  $('result-panel').hidden = $('original-title').hidden = audio.hidden = true;
  $('reset').textContent = '清除框选';
  video.muted = false;
  showSelection(roi);
}

function openComparison(url) {
  video.pause();
  comparing = true;
  players.style.setProperty('--video-ratio', video.videoWidth + ' / ' + video.videoHeight);
  players.classList.add('compare');
  $('result-panel').hidden = $('original-title').hidden = audio.hidden = false;
  $('reset').textContent = '重新框选';
  box.hidden = true;
  video.muted = true;
  audio.textContent = '开启声音';
  video.currentTime = comparisonStart;
  resultVideo.onloadeddata = () => {
    resultVideo.onloadeddata = null;
    syncResult(true);
    video.play().catch(() => {});
  };
  resultVideo.onerror = () => {
    clearComparison();
    showError('浏览器无法预览处理结果。');
  };
  resultVideo.src = url;
}

audio.onclick = () => {
  video.muted = !video.muted;
  audio.textContent = video.muted ? '开启声音' : '静音';
};
fileInput.onchange = () => {
  const chosen = fileInput.files[0];
  if (!chosen) return;
  clearComparison();
  if (url) URL.revokeObjectURL(url);
  file = chosen;
  url = URL.createObjectURL(file);
  updateSelection(null);
  showError();
  showProgress(0, '等待开始');
  $('download').hidden = true;
  players.hidden = true;
  startTime.value = 0;
  duration.value = '';
  refreshButtons();
  $('empty').hidden = false;
  $('play').disabled = timeline.disabled = true;
  $('file-name').textContent = file.name;
  video.src = url;
};
video.onloadedmetadata = () => {
  if (video.videoWidth === 960 && video.videoHeight === 544) {
    updateSelection([264, 472, 696, 526]);
  }
};
video.onloadeddata = () => {
  players.hidden = false;
  $('empty').hidden = true;
  $('play').disabled = timeline.disabled = false;
  timeline.max = Number.isFinite(video.duration) ? video.duration : 0;
  video.ontimeupdate();
  refreshButtons();
};
video.onerror = () => showError('浏览器无法预览此视频。请使用 H.264 MP4 或 WebM。');
video.ontimeupdate = () => {
  syncResult();
  timeline.value = video.currentTime;
  $('time').textContent = clock(video.currentTime) + ' / ' + clock(Number(timeline.max));
};
video.onplay = () => { $('play').textContent = '暂停'; };
video.onplaying = playResult;
video.onwaiting = () => resultVideo.pause();
video.onseeking = () => {
  resultVideo.pause();
  syncResult(true);
};
video.onseeked = playResult;
video.onpause = () => {
  resultVideo.pause();
  $('play').textContent = '播放';
};
timeline.oninput = () => { video.currentTime = Number(timeline.value); };
$('play').onclick = () => video.paused ? video.play().catch(() => showError('视频无法播放。')) : video.pause();
$('reset').onclick = () => {
  if (comparing) {
    video.pause();
    clearComparison();
  }
  updateSelection(null);
};

async function json(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '请求失败');
  return payload;
}

$('use-current-time').onclick = () => {
  startTime.value = Math.min(video.currentTime, Math.max(0, video.duration - 0.01)).toFixed(3);
};

function readSettings() {
  for (const field of [startTime, duration, ...tuningFields]) {
    if (!field.validity.valid) {
      const panel = field.closest('details');
      if (panel) panel.open = true;
      field.reportValidity();
      return null;
    }
  }
  const start = startTime.valueAsNumber;
  const length = duration.value ? duration.valueAsNumber : null;
  if (start >= video.duration || (length !== null && start + length > video.duration + 0.001)) {
    showError('开始时间和处理时长必须在视频范围内。');
    return null;
  }
  const inference = {start_seconds: start, duration_seconds: length};
  for (const field of tuningFields) {
    if (field.dataset.option === 'segment_mode' && !supportsSegmentation) continue;
    inference[field.dataset.option] = ['context', 'resolution', 'blend_mode', 'segment_mode'].includes(field.dataset.option)
      ? field.value : Number(field.value);
  }
  if (inference.overlap_frames >= inference.chunk_frames) {
    showError('重叠帧数必须小于分段帧数。');
    return null;
  }
  return inference;
}
$('submit').onclick = () => {
  const inference = supportsSettings ? readSettings() : {};
  if (!inference) return;
  comparisonStart = inference.start_seconds ?? 0;
  clearComparison();
  active = true;
  refreshButtons();
  showError();
  $('download').hidden = true;
  showProgress(0, '正在读取本地视频');
  const request = new XMLHttpRequest();
  request.open('POST', '/api/jobs');
  request.responseType = 'json';
  request.upload.onprogress = (event) => {
    if (event.lengthComputable) showProgress(Math.min(4, Math.floor(4 * event.loaded / event.total)), '正在读取本地视频');
  };
  function fail(message) {
    active = false;
    refreshButtons();
    showProgress(0, '启动失败');
    showError(message);
  }
  request.onload = () => {
    if (request.status !== 202) return fail(request.response?.error || '无法启动处理');
    job = request.response.id;
  };
  request.onerror = () => fail('本地服务连接中断，请重试。');
  const body = new FormData();
  body.set('video', file);
  body.set('roi', JSON.stringify(roi));
  body.set('inference', JSON.stringify(inference));
  body.set('dimensions', JSON.stringify([video.videoWidth, video.videoHeight]));
  request.send(body);
};

async function poll() {
  try {
    const system = await json(await fetch('/api/system'));
    ready = system.ready;
    supportsSettings = Object.hasOwn(system.inference_defaults || {}, 'start_seconds');
    supportsSegmentation = Object.hasOwn(system.inference_defaults || {}, 'segment_mode');
    $('segment-option').hidden = !supportsSegmentation;
    if (!defaultsLoaded && supportsSettings) {
      for (const field of tuningFields) {
        const value = system.inference_defaults[field.dataset.option];
        if (value !== undefined) field.value = value;
      }
      defaultsLoaded = true;
    }
    $('model').textContent = system.model;
    refreshButtons();
    const gpu = system.gpu;
    $('gpu-name').textContent = gpu?.name || 'GPU 数据不可用';
    $('util-value').textContent = gpu ? gpu.util + '%' : '—';
    $('memory-value').textContent = gpu ? (gpu.used + ' / ' + gpu.total + ' MiB') : '—';
    $('util').value = gpu?.util || 0;
    $('memory').value = gpu ? 100 * gpu.used / gpu.total : 0;
    if (job) {
      const response = await fetch('/api/jobs/' + job);
      if (response.status === 404) {
        const previous = await response.json();
        job = null;
        active = false;
        refreshButtons();
        showProgress(0, '任务已结束');
        showError(previous.error);
        return;
      }
      const result = await json(response);
      showProgress(result.percent, result.stage);
      if (result.state === 'failed' || result.state === 'done') {
        if (result.error) showError(result.error);
        if (result.state === 'done') {
          $('download').href = '/api/jobs/' + job + '/output';
          $('download').hidden = false;
          openComparison($('download').href);
        }
        job = null;
        active = false;
        refreshButtons();
      }
    }
  } catch (error) {
    $('gpu-name').textContent = '正在重新连接本地服务…';
    if (!job) {
      ready = false;
      refreshButtons();
    }
  } finally {
    setTimeout(poll, 1500);
  }
}
poll();
