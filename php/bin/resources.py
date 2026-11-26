"""Single-device capacity and process accounting for the subtitle queue, in MiB and CPU cores."""

import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time

MIB = 1024 ** 2
KEYS = ('gpu_mib', 'ram_mib', 'cpu_cores', 'disk_mib')
LABELS = dict(zip(KEYS, ('显存', '主机内存', 'CPU 核心', '临时磁盘')))
# Admission estimates, not measured peaks. Tune per host after recording representative clips.
DEFAULTS = {
    'lite': dict(zip(KEYS, (4096, 3072, 4, 512))),
    'clear': dict(zip(KEYS, (20480, 12288, 4, 1024))),
    'propainter': dict(zip(KEYS, (22528, 8192, 4, 1536))),
}
HEADROOM = dict(zip(KEYS, (1536, 2048, 1, 2048)))


def numeric(value, key, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'Invalid scheduler setting {key}={value!r}')
    return value


class Settings:
    def __init__(self, data=None):
        data = data or {}
        unknown = set(data) - {'profiles', 'headroom', 'max_jobs', 'max_bypasses', 'poll_seconds'}
        profiles = data.get('profiles', {})
        if unknown or set(profiles) - set(DEFAULTS):
            raise ValueError(f'Unknown scheduler setting: {unknown or set(profiles) - set(DEFAULTS)}')

        def amounts(base, updates):
            if set(updates) - set(KEYS):
                raise ValueError(f'Unknown resource: {set(updates) - set(KEYS)}')
            return {key: numeric(updates.get(key, default), key) for key, default in base.items()}

        self.profiles = {key: amounts(base, profiles.get(key, {})) for key, base in DEFAULTS.items()}
        self.margin = amounts(HEADROOM, data.get('headroom', {}))
        self.poll = numeric(data.get('poll_seconds', 1), 'poll_seconds', .05)
        self.bypasses = int(numeric(data.get('max_bypasses', 3), 'max_bypasses'))
        self.max_jobs = int(os.environ.get('SUBTITLE_MAX_JOBS', data.get('max_jobs', 3)))
        if self.max_jobs < 1:
            raise ValueError('SUBTITLE_MAX_JOBS must be at least 1')

    @classmethod
    def load(cls):
        path = os.environ.get('SUBTITLE_SCHEDULER_CONFIG')
        return cls(json.loads(Path(path).read_text()) if path else {})

    def estimate(self, folder, mode):
        # Each stage runs in series within its job, so retain the largest prospective working set.
        engines, width, height, input_mib, frames = file_info(str(folder), mode)
        amounts = {k: max(self.profiles[e][k] for e in engines) for k in KEYS}
        if mode == 'lite' and os.environ.get('LITE_DEVICE', 'cuda') == 'cpu':
            amounts['gpu_mib'] = 0
        pixels = width * height
        # CLEAR keeps the source chunk plus overlapping frames and several decoded working copies.
        if pixels:
            buffered_frames = 12 if mode == 'lite' else 100
            amounts['ram_mib'] += math.ceil(buffered_frames * pixels * 3 / MIB)
        # Mixed jobs additionally retain a lossless intermediate until remux. Raw size is an upper estimate.
        copies = 2 if mode == 'lite' else (6 if mode == 'max' else 4)
        scratch = input_mib * copies
        if len(engines) > 1:
            scratch += math.ceil(frames * pixels * 4 / MIB) if frames else input_mib * 10
        amounts['disk_mib'] += math.ceil(scratch)
        return amounts


@lru_cache(maxsize=1024)
def file_info(directory, mode):
    folder = Path(directory)
    try:
        request = json.loads((folder / 'request.json').read_text())
        kinds = {r['type'] for r in request['regions']}
        width, height = map(int, request['dimensions'])
    except (OSError, KeyError, TypeError, ValueError):
        kinds, width, height = {'subtitle'}, 0, 0
    engines = ['lite'] if mode == 'lite' else [
        engine for kind, engine in [('subtitle', 'clear'), ('watermark', 'propainter')] if kind in kinds]
    source = folder / 'source.video'
    try:
        size = source.stat().st_size / MIB
    except OSError:
        size = 0
    frames = 0
    if len(engines) > 1:
        try:
            command = [os.getenv('FFPROBE', 'ffprobe'), '-v', 'error', '-protocol_whitelist', 'file,pipe',
                       '-format_whitelist', 'mov,matroska,webm,avi,flv,mpegts,mpeg,asf,ogg', '-select_streams', 'v:0',
                       '-show_entries', 'stream=nb_frames,avg_frame_rate,duration:format=duration',
                       '-of', 'json', str(source)]
            info = json.loads(subprocess.check_output(command, timeout=8, stderr=subprocess.DEVNULL))
            stream = info['streams'][0]
            if str(stream.get('nb_frames', '')).isdigit():
                frames = int(stream['nb_frames'])
            else:
                num, den = map(float, stream['avg_frame_rate'].split('/'))
                frames = math.ceil(float(stream.get('duration') or info['format']['duration']) * num / den)
        except (OSError, subprocess.SubprocessError, KeyError, IndexError, ZeroDivisionError, ValueError):
            pass
    return engines or ['clear'], width, height, size, max(0, frames)


def read_text(path):
    try:
        return Path(path).read_text(errors='replace').strip()
    except OSError:
        return ''


def command_text(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, errors='replace', timeout=4).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_snapshot():
    target = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
    gpu = dict(uuid=None, total=0, free=0, util=None, processes=None)
    if not target or target == '-1':
        return gpu
    prefix = ['nvidia-smi', '-i', target, '--format=csv,noheader,nounits']
    info = command_text(prefix + ['--query-gpu=uuid,memory.total,memory.free,utilization.gpu'])
    if not info:
        return gpu
    try:
        uuid, total, free, util = (v.strip() for v in next(csv.reader(info.splitlines())))
        gpu.update(uuid=uuid, total=int(total), free=int(free), util=int(util) if util.isdigit() else None)
    except (TypeError, ValueError, StopIteration):
        return gpu
    details = command_text(prefix + ['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'])
    if details is not None:
        try:
            gpu['processes'] = {int(pid): int(mib) for device, pid, mib in csv.reader(details.splitlines())
                                if device.strip() == uuid}
        except ValueError:
            gpu['processes'] = None
    return gpu


@lru_cache(maxsize=1)
def cgroups():
    # A systemd service may live below the mounted v2 root; every ancestor may limit its subtree.
    root = Path('/sys/fs/cgroup')
    for line in read_text('/proc/self/cgroup').splitlines():
        if line.startswith('0::'):
            relative = Path(line[3:].lstrip('/'))
            if '..' not in relative.parts:
                path = root / relative
                return [p for p in [path, *path.parents] if (p == root or root in p.parents) and p.is_dir()]
    return [root]


def linux_memory():
    info = {k: int(v) / 1024 for k, v in re.findall(r'^(MemTotal|MemAvailable):\s*(\d+) kB',
                                                      read_text('/proc/meminfo'), re.M)}
    if 'MemAvailable' not in info:
        return None
    total, free = info['MemTotal'], info['MemAvailable']
    limits = [(base / 'memory.max', base / 'memory.current') for base in cgroups()]
    limits.append(('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'))
    for limit_path, used_path in limits:
        limit, used = read_text(limit_path), read_text(used_path)
        if limit.isdigit() and used.isdigit():
            total = min(total, int(limit) / MIB)
            free = min(free, max(0, int(limit) - int(used)) / MIB)
    return total, free


def memory():
    linux = linux_memory()
    if linux:
        return linux
    # macOS development/smoke tests use real available VM pages; no CUDA capacity is synthesized.
    total = command_text(['sysctl', '-n', 'hw.memsize'])
    vm = command_text(['vm_stat']) or ''
    page = re.search(r'page size of (\d+) bytes', vm)
    reclaimable = re.findall(r'^Pages (?:free|inactive):\s*(\d+)', vm, re.M)
    return (int(total) / MIB if total and total.isdigit() else 0,
            sum(map(int, reclaimable)) * int(page[1]) / MIB if page else 0)


def allowed_cpus():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))


def cpu_limit(allowed):
    limit = len(allowed)
    for base in cgroups():
        quota = read_text(base / 'cpu.max').split()
        if len(quota) == 2 and quota[0].isdigit() and int(quota[1]) > 0:
            limit = min(limit, int(quota[0]) / int(quota[1]))
    quota = read_text('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = read_text('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota.isdigit() and period.isdigit() and int(period) > 0:
        return min(limit, int(quota) / int(period))
    return limit


class Probe:
    def __init__(self, directory):
        self.directory = directory
        self.allowed = allowed_cpus()
        self.cores = cpu_limit(self.allowed)
        self.previous_cpu = {}
        self.idle_fraction = {}
        self.previous_tasks = {}
        self.last = time.monotonic()
        self.previous_cgroup = {}
        self.last_cpu()  # Prime the first tick from counters instead of interpreting uptime as idle time.

    def last_cpu(self):
        samples = {}
        for label, line in re.findall(r'^cpu(\d+)\s+(.+)$', read_text('/proc/stat'), re.M):
            values = [int(s) for s in line.split()[:8]]
            if int(label) in self.allowed:
                samples[int(label)] = (values[3], sum(values))
        idle = 0.0
        for cpu, (quiet, total) in samples.items():
            before = self.previous_cpu.get(cpu)
            if before and total > before[1]:
                self.idle_fraction[cpu] = (quiet - before[0]) / (total - before[1])
            idle += self.idle_fraction.get(cpu, 0)
        self.previous_cpu = samples
        return min(self.cores, idle) if samples else None

    def processes(self, elapsed):
        result = {}
        if not Path('/proc/self/stat').is_file():
            lines = command_text(['ps', '-axo', 'pid=,pgid=,rss=,pcpu=,stat=']) or ''
            for line in lines.splitlines():
                fields = line.split()
                if len(fields) == 5 and not fields[-1].startswith('Z'):
                    pid, group, rss, pct = fields[:4]
                    result[int(pid)] = (int(group), int(rss) / 1024, float(pct))
            return result
        current = {}
        tick = os.sysconf('SC_CLK_TCK')
        page_mib = os.sysconf('SC_PAGE_SIZE') / MIB
        for path in Path('/proc').iterdir():
            if not path.name.isdecimal():
                continue
            stat = read_text(path / 'stat').rsplit(') ', 1)[-1].split()
            if len(stat) < 22 or stat[0] == 'Z':
                continue
            pid = int(path.name)
            key = (pid, stat[19])  # PID plus process birth ticks, resistant to PID reuse.
            ticks = int(stat[11]) + int(stat[12])
            current[key] = ticks
            pct = 100 * (ticks - self.previous_tasks.get(key, ticks)) / tick / elapsed
            result[pid] = (int(stat[2]), int(stat[21]) * page_mib, pct)
        self.previous_tasks = current
        return result

    def sample(self):
        gpu = gpu_snapshot()
        now = time.monotonic()
        elapsed = max(.001, now - self.last)
        self.last = now
        mem_total, mem_free = memory()
        disk = shutil.disk_usage(self.directory)
        processes = self.processes(elapsed)
        cpu = self.last_cpu()
        if cpu is None:
            cpu = self.cores - sum(v[2] for v in processes.values()) / 100
        for base in cgroups():
            quota = read_text(base / 'cpu.max').split()
            usage = re.search(r'^usage_usec (\d+)$', read_text(base / 'cpu.stat'), re.M)
            if usage and len(quota) == 2 and quota[0].isdigit() and int(quota[1]) > 0:
                micros = int(usage[1])
                if base in self.previous_cgroup:
                    limit = int(quota[0]) / int(quota[1])
                    cpu = min(cpu, limit - (micros - self.previous_cgroup[base]) / elapsed / 1e6)
                self.previous_cgroup[base] = micros
        return dict(ts=time.time(), gpu=gpu, processes=processes,
                    free=dict(zip(KEYS, (gpu['free'], mem_free, max(0, cpu), disk.free / MIB))),
                    total=dict(zip(KEYS, (gpu['total'], mem_total, self.cores, disk.total / MIB))))


def job_sample(snapshot, group):
    procs = {pid: entry for pid, entry in snapshot['processes'].items() if entry[0] == group}
    gpu = snapshot['gpu']
    apps = gpu['processes']
    # NVIDIA may expose host PIDs inside a PID namespace. Report an unverified attribution explicitly.
    unseen = apps is not None and any(pid not in snapshot['processes'] for pid in apps)
    gpu_mib = None if apps is None or unseen else sum(apps.get(pid, 0) for pid in procs)
    return dict(ts=snapshot['ts'], gpu_uuid=gpu['uuid'], gpu_total_mib=gpu['total'], gpu_free_mib=gpu['free'],
                gpu_util_pct=gpu['util'], gpu_processes=apps, job_gpu_mib=gpu_mib if procs else None,
                job_rss_mib=sum(p[1] for p in procs.values()) if procs else None,
                job_cpu_pct=sum(p[2] for p in procs.values()) if procs else None, pids=sorted(procs),
                host_available_ram_mib=snapshot['free']['ram_mib'], available_cpu_cores=snapshot['free']['cpu_cores'],
                temp_free_mib=snapshot['free']['disk_mib'])


def working_mib(folder):
    """Allocated blocks for this job's generated files, excluding the uploaded input and linked sources."""
    todo = [folder]
    seen = set()
    size = 0
    while todo:
        directory = todo.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if directory == folder and entry.name == 'source.video':
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        todo.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        if info.st_nlink > 1:
                            key = (info.st_dev, info.st_ino)
                            if key in seen:
                                continue
                            seen.add(key)
                        size += info.st_blocks * 512
        except FileNotFoundError:
            continue
        except OSError:
            return None
    return size / MIB


def remaining(free, margin, running):
    """Live free already excludes usage. Reserve only the unconsumed part of each active estimate."""
    budget = {k: free[k] - margin[k] for k in KEYS}
    for reserve, sample in running:
        used = dict(gpu_mib=sample.get('job_gpu_mib'), ram_mib=sample.get('job_rss_mib'),
                    cpu_cores=(sample.get('job_cpu_pct') or 0) / 100, disk_mib=sample.get('job_disk_growth_mib'))
        for k in KEYS:
            budget[k] -= max(0, reserve[k] - (used[k] or 0))
    return {k: max(0, value) for k, value in budget.items()}
