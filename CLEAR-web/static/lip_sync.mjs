const $ = (id) => document.getElementById(id);
const file = $('file');
const voiceFile = $('voice-file');
const mode = $('duration-mode');
const source = $('source-video');
const voice = $('voice-preview');
const result = $('result-video');
const fullHint = $('duration-hint').textContent;
const storageKey = 'clear-lip-sync-job';
let videoUrl;
let voiceUrl;
let jobId;
try { jobId = sessionStorage.getItem(storageKey); } catch {}
let busy = Boolean(jobId);
let available = false;
let retryable = false;

function rememberJob(id) {
  jobId = id;
  try {
    if (id) sessionStorage.setItem(storageKey, id);
    else sessionStorage.removeItem(storageKey);
  } catch {}
}

function setError(message = '') {
  $('error').textContent = message;
  $('error').hidden = !message;
}

function show(stage, percent = 0) {
  $('status').textContent = stage;
  $('progress').value = percent;
  $('percent').textContent = Math.floor(percent) + '%';
}

function updateControls() {
  for (const input of [file, voiceFile, mode]) input.disabled = busy;
  $('submit').disabled = busy || !available || !file.files.length || !voiceFile.files.length;
  $('retry').hidden = !retryable;
  $('retry').disabled = busy || !available;
}

function updatePlayers() {
  const showSource = !$('source-panel').hidden;
  const showResult = !$('result-panel').hidden;
  $('lip-players').hidden = !(showSource || showResult);
  $('lip-players').classList.toggle('compare', showSource && showResult);
  $('empty').hidden = showSource || showResult;
}

function resetOutput() {
  result.removeAttribute('src');
  result.load();
  $('result-panel').hidden = true;
  $('result-error').hidden = true;
  $('download').hidden = true;
  $('task-timing').hidden = true;
  $('output-length').textContent = '';
  $('elapsed').textContent = '';
  setError();
  retryable = false;
  updatePlayers();
}

function changeInput(element, media, previousUrl) {
  const selected = element.files[0];
  if (previousUrl) URL.revokeObjectURL(previousUrl);
  if (selected) media.src = URL.createObjectURL(selected);
  else {
    media.removeAttribute('src');
    media.load();
  }
  rememberJob(null);
  resetOutput();
  show('等待开始');
  updateControls();
  return selected ? media.src : null;
}

file.onchange = () => {
  $('source-panel').hidden = !file.files.length;
  $('source-error').hidden = true;
  $('file-name').textContent = file.files[0]?.name || '尚未选择视频';
  videoUrl = changeInput(file, source, videoUrl);
};
voiceFile.onchange = () => {
  voice.hidden = !voiceFile.files.length;
  voiceUrl = changeInput(voiceFile, voice, voiceUrl);
};
voice.onerror = () => { voice.hidden = true; };
source.onerror = () => { $('source-error').hidden = false; };
result.onerror = () => { $('result-error').hidden = false; };
for (const media of [source, voice, result]) {
  media.onplay = () => {
    for (const other of [source, voice, result]) if (other !== media) other.pause();
  };
}
mode.onchange = () => {
  $('duration-hint').textContent = mode.value === 'preview' ? '合成视频与配音重叠的部分。' : fullHint;
};

async function json(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || '请求失败，请重试。');
  return data;
}

async function submit(url, body) {
  if (busy || !available) return;
  const previousJob = jobId;
  busy = true;
  rememberJob(null);
  resetOutput();
  updateControls();
  show(body ? '正在上传视频和配音' : '正在重新提交');
  try {
    const response = await json(await fetch(url, { method: 'POST', body }));
    rememberJob(response.id);
    show('等待 GPU 开始处理');
  } catch (error) {
    busy = false;
    if (!body) {
      rememberJob(previousJob);
      retryable = true;
    }
    show('提交失败');
    setError(error.message);
    updateControls();
  }
}

$('lip-form').onsubmit = (event) => {
  event.preventDefault();
  if ($('submit').disabled) return;
  const data = new FormData();
  data.set('mode', 'lip_sync');
  data.set('duration_mode', mode.value);
  data.set('video', file.files[0]);
  data.set('audio', voiceFile.files[0]);
  submit('/api/jobs', data);
};
$('retry').onclick = () => submit('/api/jobs/' + jobId + '/retry');

async function refreshSystem() {
  const data = await json(await fetch('/api/system', { cache: 'no-store' }));
  const engine = data.engines?.musetalk;
  available = Boolean(engine?.ready && !data.draining);
  $('model').textContent = data.draining ? '正在更新服务' : (engine?.model || 'MuseTalk 不可用');
  const gpu = data.gpu;
  $('gpu-name').textContent = gpu?.name || 'GPU 数据不可用';
  $('util-value').textContent = gpu ? gpu.util + '%' : '—';
  $('util').value = gpu?.util || 0;
  $('memory-value').textContent = gpu ? (gpu.used / 1024).toFixed(1) + ' / ' + (gpu.total / 1024).toFixed(1) + ' GiB' : '—';
  $('memory').value = gpu?.total ? 100 * gpu.used / gpu.total : 0;
}

async function refreshJob(id) {
  const response = await fetch('/api/jobs/' + id, { cache: 'no-store' });
  if (id !== jobId) return;
  if (response.status === 404) {
    rememberJob(null);
    busy = false;
    show('找不到上一次任务');
    return;
  }
  const job = await json(response);
  if (id !== jobId) return;
  show(job.stage, job.percent);
  if (job.elapsed_seconds != null) {
    $('task-timing').hidden = false;
    $('elapsed').textContent = '处理用时 ' + job.elapsed_seconds.toFixed(1) + ' 秒';
  }
  if (job.state === 'done') {
    busy = false;
    const url = '/api/jobs/' + id + '/output';
    if (result.getAttribute('src') !== url + '?inline=1') result.src = url + '?inline=1';
    $('result-panel').hidden = false;
    $('download').href = url;
    $('download').hidden = false;
    if (job.output_seconds != null) $('output-length').textContent = '成片 ' + job.output_seconds.toFixed(2) + ' 秒';
    updatePlayers();
  } else if (job.state === 'failed') {
    busy = false;
    retryable = job.can_retry;
    setError(job.error || '合成失败。');
  }
}

async function poll() {
  const tasks = [refreshSystem()];
  if (jobId) tasks.push(refreshJob(jobId));
  const [system, task] = await Promise.allSettled(tasks);
  if (system.status === 'rejected') {
    available = false;
    $('model').textContent = '正在重新连接';
    $('gpu-name').textContent = 'GPU 数据暂不可用';
  }
  if (busy && task?.status === 'rejected') $('status').textContent = '正在重新读取任务状态';
  updateControls();
  setTimeout(poll, 2200);
}

updateControls();
poll();
