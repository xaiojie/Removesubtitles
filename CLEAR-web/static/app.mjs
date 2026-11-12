import { selection } from './selection.mjs';

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const box = $('box');
const timeline = $('timeline');
const fileInput = $('file');
let url, file, roi, drag, job;
let active = false;
let ready = false;
const clock = (seconds) => [Math.floor(seconds / 60), Math.floor(seconds % 60)]
  .map(n => String(n).padStart(2, '0')).join(':');

function refreshButtons() {
  $('submit').disabled = !roi || active || !ready;
  $('reset').disabled = !roi || active;
  fileInput.disabled = active;
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

fileInput.onchange = () => {
  const chosen = fileInput.files[0];
  if (!chosen) return;
  if (url) URL.revokeObjectURL(url);
  file = chosen;
  url = URL.createObjectURL(file);
  updateSelection(null);
  showError();
  showProgress(0, '等待开始');
  $('download').hidden = true;
  $('surface').hidden = true;
  $('empty').hidden = false;
  $('play').disabled = timeline.disabled = true;
  $('file-name').textContent = file.name;
  video.src = url;
};
video.onloadeddata = () => {
  $('surface').hidden = false;
  $('empty').hidden = true;
  $('play').disabled = timeline.disabled = false;
  timeline.max = Number.isFinite(video.duration) ? video.duration : 0;
  video.ontimeupdate();
};
video.onerror = () => showError('浏览器无法预览此视频。请使用 H.264 MP4 或 WebM。');
video.ontimeupdate = () => {
  timeline.value = video.currentTime;
  $('time').textContent = clock(video.currentTime) + ' / ' + clock(Number(timeline.max));
};
video.onplay = () => { $('play').textContent = '暂停'; };
video.onpause = () => { $('play').textContent = '播放'; };
timeline.oninput = () => { video.currentTime = Number(timeline.value); };
$('play').onclick = () => video.paused ? video.play().catch(() => showError('视频无法播放。')) : video.pause();
$('reset').onclick = () => updateSelection(null);

async function json(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '请求失败');
  return payload;
}

$('submit').onclick = () => {
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
  body.set('dimensions', JSON.stringify([video.videoWidth, video.videoHeight]));
  request.send(body);
};

async function poll() {
  try {
    const system = await json(await fetch('/api/system'));
    ready = system.ready;
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
