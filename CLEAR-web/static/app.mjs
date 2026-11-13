import { selection } from './selection.mjs';
import { estimateRemainingSeconds } from './timing.mjs';

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
const exportQuality = $('export-crf');
const startTime = $('start-seconds');
const duration = $('duration-seconds');
const tuningFields = [...settings.querySelectorAll('[data-option]')];
let comparisonStart = 0;
const defaultsLoaded = new Set();
let commonDefaultsLoaded = false;
let latestSystem = null;
let supportsSettings = false;
let supportsSegmentation = false;
let url, file, roi, drag, job;
let regions = [];
let selectedRegion = -1;
let selectionMode = 'add';
let drawingType = 'subtitle';
let active = false;
let ready = false;
let comparing = false;
let timingTimer = null;
let startedAt = null;
let firstRepairSample = null;
let latestRepairSample = null;
let latestPercent = 0;
let timingDisconnected = false;
let completedJobId = null;
let comparisonSession = null;
let comparisonPollTimer = null;
let retryJobId = null;
let retryController = null;

function renderTiming() {
  if (startedAt === null) return;
  const now = performance.now();
  $('elapsed').textContent = '已用时 ' + clock(Math.floor((now - startedAt) / 1000));
  if (timingDisconnected) {
    $('eta').textContent = '连接中断，恢复后重新估算';
  } else if (latestPercent >= 97) {
    $('eta').textContent = '正在导出视频…';
  } else if (latestPercent >= 94) {
    $('eta').textContent = '修复完成，正在准备导出…';
  } else {
    const remaining = estimateRemainingSeconds(firstRepairSample, latestRepairSample, now);
    $('eta').textContent = remaining === null ? (latestRepairSample ? '正在重新估算剩余时间' : '正在估算剩余时间') : '预计总剩余 ' + clock(remaining);
  }
}

function resetTiming() {
  clearInterval(timingTimer);
  timingTimer = null;
  startedAt = null;
  firstRepairSample = latestRepairSample = null;
  latestPercent = 0;
  timingDisconnected = false;
  $('task-timing').hidden = true;
  $('elapsed').textContent = '';
  $('eta').textContent = '正在估算剩余时间';
}

function startTiming() {
  startedAt = performance.now();
  $('task-timing').hidden = false;
  renderTiming();
  timingTimer = setInterval(renderTiming, 1000);
}

function finishTiming(message) {
  renderTiming();
  clearInterval(timingTimer);
  timingTimer = null;
  startedAt = null;
  $('eta').textContent = message;
}

const clock = (seconds) => [Math.floor(seconds / 60), Math.floor(seconds % 60)]
  .map(n => String(n).padStart(2, '0')).join(':');

function refreshButtons() {
  $('submit').disabled = !regions.length || active || !ready || Boolean(drag);
  $('reset').disabled = !regions.length || active;
  $('redraw').disabled = selectedRegion < 0 || active;
  $('delete-region').disabled = selectedRegion < 0 || active;
  $('copy-settings').disabled = !regions.length;
  fileInput.disabled = active;
  $('retry').disabled = active;
  settings.disabled = active || players.hidden || !supportsSettings;
  for (const id of ['select-single', 'select-double', 'add-subtitle', 'add-watermark']) {
    $(id).disabled = active || players.hidden;
  }
  for (const button of document.querySelectorAll('.region-item, .region-label')) button.disabled = active;
  $('add-subtitle').setAttribute('aria-pressed', String(selectionMode === 'add' && drawingType === 'subtitle'));
  $('add-watermark').setAttribute('aria-pressed', String(selectionMode === 'add' && drawingType === 'watermark'));
  const typeName = drawingType === 'watermark' ? '水印' : '字幕';
  $('selection-hint').textContent = players.hidden
    ? '打开视频后，可添加多个字幕框和水印框。'
    : active ? '正在处理，选区已锁定。'
    : comparing ? '同步查看处理前后画面。选择列表中的区域可继续调整。'
    : selectionMode === 'add' ? '连续添加' + typeName + '框，在画面上拖动即可。修改已有框请先选中它。'
    : '当前选区已选中，可在画面上拖动重画。添加其他位置请点击 ＋ 字幕框或 ＋ 水印框。';
}

function fieldsForEngine(engine) {
  return tuningFields.filter(field => field.closest('[data-engine]').dataset.engine === engine);
}

function requiredEngines() {
  const engines = [];
  if (regions.some(region => region.type === 'subtitle')) engines.push('clear');
  if (regions.some(region => region.type === 'watermark')) engines.push('propainter');
  return engines;
}

function refreshRoutingState() {
  const engines = requiredEngines();
  for (const panel of settings.querySelectorAll('[data-engine]')) {
    panel.hidden = !engines.includes(panel.dataset.engine);
    panel.disabled = panel.hidden;
  }
  const states = engines.map(engine => latestSystem?.engines?.[engine] || (engine === 'clear' && latestSystem
    ? { ready: latestSystem.ready, model: latestSystem.model } : null));
  ready = engines.length > 0 && states.every(state => state?.ready) && !timingDisconnected;
  supportsSettings = true;
  supportsSegmentation = Object.hasOwn(latestSystem?.inference_defaults || {}, 'segment_mode');
  $('segment-option').hidden = !supportsSegmentation;
  $('model').textContent = states.length ? states.map((state, index) => state?.model || (engines[index] === 'clear' ? 'CLEAR 正在检查' : 'ProPainter 正在检查')).join(' · ') : '等待选区';
  $('routing-flow').textContent = engines.length === 2 ? 'CLEAR 擦字幕 → ProPainter 去水印'
    : engines[0] === 'clear' ? '字幕 · CLEAR'
    : engines[0] === 'propainter' ? '水印 · ProPainter' : '请先框选字幕或水印';
  refreshButtons();
}

function positionBox(element, rectangle) {
  const [left, top, right, bottom] = rectangle;
  Object.assign(element.style, {
    left: (100 * left / video.videoWidth) + '%',
    top: (100 * top / video.videoHeight) + '%',
    width: (100 * (right - left) / video.videoWidth) + '%',
    height: (100 * (bottom - top) / video.videoHeight) + '%',
  });
}

function showSelection(rectangle) {
  box.hidden = !rectangle || comparing;
  if (!rectangle) return;
  box.dataset.type = drawingType;
  positionBox(box, rectangle);
}

function selectRegion(index) {
  if (active) return;
  if (comparing) clearComparison();
  selectedRegion = index;
  drawingType = regions[index].type;
  selectionMode = 'redraw';
  renderRegions();
  refreshButtons();
}

function renderRegions() {
  $('region-boxes').replaceChildren();
  $('region-list').replaceChildren();
  $('region-boxes').hidden = comparing;
  const counts = { subtitle: 0, watermark: 0 };
  regions.forEach((region, index) => {
    const label = (region.type === 'subtitle' ? '字幕 ' : '水印 ') + (++counts[region.type]);
    const selected = index === selectedRegion;
    const rectangle = document.createElement('div');
    rectangle.className = 'region-box ' + region.type + (selected ? ' selected' : '');
    positionBox(rectangle, region.roi);
    const tag = document.createElement('button');
    tag.type = 'button';
    tag.className = 'region-label';
    tag.textContent = label;
    tag.setAttribute('aria-label', '选中' + label);
    tag.onclick = () => selectRegion(index);
    rectangle.append(tag);
    $('region-boxes').append(rectangle);
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'region-item ' + region.type + (selected ? ' selected' : '');
    item.setAttribute('aria-pressed', String(selected));
    const name = document.createElement('strong');
    name.textContent = label;
    const coordinates = document.createElement('code');
    coordinates.textContent = '[' + region.roi.join(', ') + ']';
    item.append(name, coordinates);
    item.onclick = () => selectRegion(index);
    $('region-list').append(item);
  });
}

function updateRegions() {
  roi = regions.length ? regions.reduce((bounds, region) => [
    Math.min(bounds[0], region.roi[0]), Math.min(bounds[1], region.roi[1]),
    Math.max(bounds[2], region.roi[2]), Math.max(bounds[3], region.roi[3]),
  ], regions[0].roi.slice()) : null;
  const subtitles = regions.filter(region => region.type === 'subtitle').length;
  const watermarks = regions.length - subtitles;
  $('roi').textContent = regions.length ? subtitles + ' 处字幕 · ' + watermarks + ' 处水印' : '等待框选';
  $('copy-status').textContent = '';
  showSelection(null);
  renderRegions();
  refreshRoutingState();
}

function beginRegion(type) {
  if (active || players.hidden) return;
  if (comparing) clearComparison();
  selectedRegion = -1;
  drawingType = type;
  selectionMode = 'add';
  showSelection(null);
  renderRegions();
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
  if (active || drag || event.button !== 0 || !video.videoWidth || event.target.closest('.region-label')) return;
  video.pause();
  overlay.setPointerCapture(event.pointerId);
  drag = { id: event.pointerId, start: point(event) };
  refreshButtons();
};
overlay.onpointermove = (event) => {
  if (drag?.id === event.pointerId) showSelection(currentSelection(event));
};
overlay.onpointerup = (event) => {
  if (drag?.id !== event.pointerId) return;
  const next = currentSelection(event);
  drag = null;
  if (next && !active) {
    if (selectionMode === 'redraw' && selectedRegion >= 0) regions[selectedRegion].roi = next;
    else {
      regions.push({ type: drawingType, roi: next });
      selectedRegion = regions.length - 1;
    }
  }
  updateRegions();
};
overlay.onlostpointercapture = () => {
  drag = null;
  showSelection(null);
  refreshButtons();
};

function showError(message = '') {
  $('error').textContent = message;
  $('error').hidden = !message;
}
function showProgress(percent, stage, frameProgress = {}) {
  $('progress').value = percent;
  $('percent').textContent = percent + '%';
  $('status').textContent = stage;
  latestPercent = percent;
  const pipeline = frameProgress.pipeline;
  const names = { clear: 'CLEAR 擦字幕', propainter: 'ProPainter 去水印' };
  const pipelineText = Array.isArray(pipeline) ? pipeline.map(step => {
    const name = typeof step === 'string' ? step : step.label || step.engine;
    return names[name] || name || '';
  }).filter(Boolean).join(' → ') : typeof pipeline === 'string' ? pipeline : '';
  $('pipeline-status').hidden = !pipelineText;
  $('pipeline-status').textContent = pipelineText;
  $('phase-status').hidden = !frameProgress.phase_label;
  $('phase-label').textContent = frameProgress.phase_label || '';
  $('phase-percent').textContent = Number.isFinite(frameProgress.phase_percent)
    ? Math.round(frameProgress.phase_percent) + '%' : '';
  const repairing = /去噪|修复|处理帧/.test(stage) || frameProgress.completed_frames > 0;
  const repairPercent = 8 + 86 * Math.min(1, Math.max(0, percent / 100));
  if (startedAt !== null && repairing && repairPercent < 94) {
    const sample = { percent: repairPercent, at: performance.now() };
    if (!firstRepairSample) firstRepairSample = sample;
    if (!latestRepairSample || repairPercent > latestRepairSample.percent) latestRepairSample = sample;
  }
  renderTiming();
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
  video.muted = false;
  showSelection(null);
  renderRegions();
}

function openComparison(url) {
  video.pause();
  comparing = true;
  players.style.setProperty('--video-ratio', video.videoWidth + ' / ' + video.videoHeight);
  players.classList.add('compare');
  $('result-panel').hidden = $('original-title').hidden = audio.hidden = false;
  box.hidden = true;
  $('region-boxes').hidden = true;
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
  resetTiming();
  resetComparisonDownload();
  resetRetry();
  if (url) URL.revokeObjectURL(url);
  file = chosen;
  url = URL.createObjectURL(file);
  regions = [];
  selectedRegion = -1;
  selectionMode = 'add';
  drawingType = 'subtitle';
  drag = null;
  updateRegions();
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
  if (active) return;
  if (comparing) clearComparison();
  regions = [];
  selectedRegion = -1;
  selectionMode = 'add';
  updateRegions();
};
$('delete-region').onclick = () => {
  if (active || selectedRegion < 0) return;
  if (comparing) clearComparison();
  regions.splice(selectedRegion, 1);
  selectedRegion = Math.min(selectedRegion, regions.length - 1);
  selectionMode = selectedRegion < 0 ? 'add' : 'redraw';
  if (selectedRegion >= 0) drawingType = regions[selectedRegion].type;
  updateRegions();
};
$('add-subtitle').onclick = () => beginRegion('subtitle');
$('add-watermark').onclick = () => beginRegion('watermark');

function selectSubtitlePreset(lines) {
  if (active || !video.videoWidth) return;
  video.pause();
  if (comparing) clearComparison();
  const top = lines === 1 ? 0.85 : 0.76;
  regions.push({ type: 'subtitle', roi: selection([0.14, top], [0.86, 0.96], video.videoWidth, video.videoHeight) });
  selectedRegion = regions.length - 1;
  drawingType = 'subtitle';
  selectionMode = 'redraw';
  updateRegions();
}
$('select-single').onclick = () => selectSubtitlePreset(1);
$('select-double').onclick = () => selectSubtitlePreset(2);
$('redraw').onclick = () => {
  if (selectedRegion >= 0) selectRegion(selectedRegion);
};

async function json(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '请求失败');
  return payload;
}

function resetComparisonDownload() {
  comparisonSession?.controller.abort();
  clearTimeout(comparisonPollTimer);
  comparisonPollTimer = null;
  comparisonSession = null;
  completedJobId = null;
  $('download-actions').hidden = true;
  $('download-comparison').disabled = true;
  $('download-comparison').textContent = '下载对比视频';
  $('comparison-status').hidden = true;
  $('comparison-status').textContent = '';
  $('comparison-status').classList.remove('error');
}

function failComparisonDownload(session, error) {
  if (comparisonSession !== session) return;
  clearTimeout(comparisonPollTimer);
  comparisonPollTimer = null;
  session.running = false;
  $('download-comparison').disabled = false;
  $('download-comparison').textContent = '下载对比视频';
  $('comparison-status').textContent = error.message || '对比视频生成失败，请重试。';
  $('comparison-status').classList.add('error');
}

function updateComparisonDownload(session, result) {
  if (comparisonSession !== session) return;
  if (result.state === 'done') {
    if (!result.url) throw new Error('对比视频下载地址缺失，请重试。');
    session.running = false;
    $('download-comparison').disabled = false;
    $('download-comparison').textContent = '下载对比视频';
    $('comparison-status').textContent = '对比视频已生成，正在下载。';
    const link = document.createElement('a');
    link.href = result.url;
    link.download = '';
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    return;
  }
  if (result.state !== 'running') {
    throw new Error(result.error || '对比视频生成已中断，请点击重试。');
  }
  $('comparison-status').textContent = result.stage || '正在生成左右对比视频…';
  comparisonPollTimer = setTimeout(async () => {
    if (comparisonSession !== session) return;
    try {
      const response = await fetch(session.endpoint, { signal: session.controller.signal });
      updateComparisonDownload(session, await json(response));
    } catch (error) {
      failComparisonDownload(session, error);
    }
  }, 2000);
}

$('download-comparison').onclick = async () => {
  if (!completedJobId || comparisonSession?.running) return;
  const session = {
    endpoint: '/api/jobs/' + completedJobId + '/comparison',
    controller: new AbortController(),
    originalFile: file,
    running: true,
  };
  comparisonSession = session;
  $('download-comparison').disabled = true;
  $('download-comparison').textContent = '正在生成…';
  $('comparison-status').hidden = false;
  $('comparison-status').classList.remove('error');
  $('comparison-status').textContent = '正在生成左右对比视频…';
  try {
    let response = await fetch(session.endpoint, { method: 'POST', signal: session.controller.signal });
    let result = await response.json();
    if (comparisonSession !== session) return;
    if (response.status === 409 && result.needs_original && session.originalFile) {
      $('comparison-status').textContent = '正在上传原视频以生成对比视频…';
      const body = new FormData();
      body.set('video', session.originalFile);
      response = await fetch(session.endpoint, { method: 'POST', body, signal: session.controller.signal });
      result = await response.json();
    }
    if (!response.ok) throw new Error(result.error || '无法生成对比视频，请重试。');
    updateComparisonDownload(session, result);
  } catch (error) {
    failComparisonDownload(session, error);
  }
};

$('use-current-time').onclick = () => {
  startTime.value = Math.min(video.currentTime, Math.max(0, video.duration - 0.01)).toFixed(3);
};

function readSettings() {
  const engines = requiredEngines();
  const engineFields = engines.flatMap(engine => fieldsForEngine(engine));
  for (const field of [startTime, duration, exportQuality, ...engineFields]) {
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
  const inference = { start_seconds: start, duration_seconds: length, export_crf: exportQuality.valueAsNumber };
  for (const engine of engines) {
    const options = {};
    for (const field of fieldsForEngine(engine)) {
      if (engine === 'clear' && field.dataset.option === 'segment_mode' && !supportsSegmentation) continue;
      if (['vae_tiled', 'fp16'].includes(field.dataset.option)) {
        options[field.dataset.option] = field.type === 'checkbox' ? field.checked : field.value === 'true';
      } else {
        options[field.dataset.option] = ['context', 'resolution', 'blend_mode', 'segment_mode', 'watermark_mask_mode'].includes(field.dataset.option)
          ? field.value : Number(field.value);
      }
    }
    inference[engine] = options;
  }
  if (inference.clear && inference.clear.overlap_frames >= inference.clear.chunk_frames) {
    showError('重叠帧数必须小于分段帧数。');
    return null;
  }
  return inference;
}

function resetRetry() {
  retryController?.abort();
  retryController = null;
  retryJobId = null;
  $('retry').hidden = true;
}

$('retry').onclick = async () => {
  if (active || !retryJobId) return;
  const id = retryJobId;
  const controller = new AbortController();
  retryController = controller;
  active = true;
  resetTiming();
  $('task-timing').hidden = false;
  $('eta').textContent = '正在准备继续任务';
  showProgress(0, '正在继续失败任务');
  showError();
  refreshButtons();
  try {
    const response = await fetch('/api/jobs/' + id + '/retry', { method: 'POST', signal: controller.signal });
    const result = await json(response);
    if (response.status !== 202) throw new Error(result.error || '无法继续任务，请重试。');
    if (retryController !== controller) return;
    job = result.id || id;
    $('retry').hidden = true;
    startTiming();
    showProgress(result.percent || 0, result.stage || '正在继续处理', result);
  } catch (error) {
    if (retryController !== controller) return;
    active = false;
    finishTiming('继续任务失败');
    showError(error.message);
    refreshButtons();
  }
};

$('copy-settings').onclick = async () => {
  const configuration = {
    subtitles: regions.filter(region => region.type === 'subtitle').map(region => region.roi.slice()),
    watermarks: regions.filter(region => region.type === 'watermark').map(region => region.roi.slice()),
  };
  try {
    await navigator.clipboard.writeText(JSON.stringify(configuration, null, 2));
    $('copy-status').textContent = '选区参数已复制。';
  } catch {
    $('copy-status').textContent = '复制未完成，请允许浏览器访问剪贴板后重试。';
  }
};

$('submit').onclick = () => {
  if (active || !regions.length) return;
  const inference = supportsSettings ? readSettings() : {};
  if (!inference) return;
  comparisonStart = inference.start_seconds ?? 0;
  clearComparison();
  active = true;
  resetComparisonDownload();
  resetRetry();
  resetTiming();
  $('task-timing').hidden = false;
  $('eta').textContent = '正在上传视频，完成后估算';
  refreshButtons();
  showError();
  $('download').hidden = true;
  showProgress(0, '正在上传视频');
  const request = new XMLHttpRequest();
  request.open('POST', '/api/jobs');
  request.responseType = 'json';
  request.upload.onprogress = (event) => {
    if (event.lengthComputable) showProgress(Math.min(4, Math.floor(4 * event.loaded / event.total)), '正在上传视频');
  };
  function fail(message) {
    active = false;
    refreshButtons();
    showProgress(0, '启动失败');
    finishTiming('任务启动失败');
    showError(message);
  }
  request.onload = () => {
    if (request.status !== 202) return fail(request.response?.error || '无法启动处理');
    job = request.response.id;
    startTiming();
    showProgress(4, '视频已上传，正在启动处理');
  };
  request.onerror = () => fail('服务连接中断，请重试。');
  const body = new FormData();
  body.set('video', file);
  body.set('engine', 'auto');
  body.set('roi', JSON.stringify(roi));
  body.set('regions', JSON.stringify(regions.map(region => ({ type: region.type, roi: region.roi }))));
  body.set('inference', JSON.stringify(inference));
  body.set('dimensions', JSON.stringify([video.videoWidth, video.videoHeight]));
  request.send(body);
};

async function poll() {
  try {
    const system = await json(await fetch('/api/system'));
    timingDisconnected = false;
    latestSystem = system;
    for (const engine of ['clear', 'propainter']) {
      const defaults = engine === 'clear' ? system.inference_defaults : system.propainter_defaults;
      if (defaults && !defaultsLoaded.has(engine)) {
        for (const field of fieldsForEngine(engine)) {
          const value = defaults[field.dataset.option];
          if (value !== undefined) {
            if (field.type === 'checkbox') field.checked = Boolean(value);
            else field.value = value;
          }
        }
        defaultsLoaded.add(engine);
      }
    }
    if (!commonDefaultsLoaded) {
      const crf = system.inference_defaults?.export_crf ?? system.propainter_defaults?.export_crf;
      if (crf !== undefined) {
        exportQuality.value = crf;
        commonDefaultsLoaded = true;
      }
    }
    refreshRoutingState();
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
        finishTiming('任务已结束');
        showError(previous.error);
        resetRetry();
        return;
      }
      const result = await json(response);
      showProgress(result.percent, result.stage, result);
      if (result.state === 'failed' || result.state === 'done') {
        finishTiming(result.state === 'done' ? '处理完成' : '处理失败');
        if (result.error) showError(result.error);
        retryJobId = result.state === 'failed' && result.can_retry === true ? job : null;
        $('retry').hidden = !retryJobId;
        if (result.state === 'done') {
          completedJobId = job;
          $('download-actions').hidden = false;
          $('download-comparison').disabled = false;
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
    timingDisconnected = true;
    firstRepairSample = latestRepairSample = null;
    renderTiming();
    $('gpu-name').textContent = '正在重新连接服务…';
    if (!job) {
      ready = false;
      refreshButtons();
    }
  } finally {
    setTimeout(poll, 1500);
  }
}
poll();
