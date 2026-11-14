"""Route subtitle/watermark regions through resumable model-specific stages."""
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
WEIGHTS = {'clear': 0.6, 'ltx': 0.8, 'propainter': 0.4}
LABELS = {'clear': '字幕修复', 'ltx': 'LTX 字幕修复（测试）', 'propainter': '水印修复'}
PIPELINES = (['clear'], ['ltx'], ['propainter'], ['clear', 'propainter'], ['ltx', 'propainter'])


def report(**fields):
    print('CLEAR_STATUS ' + json.dumps(fields, ensure_ascii=False), flush=True)


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def signature(config):
    serialized = json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def artifact_path(folder, phase, pipeline):
    filename = 'output.mkv' if phase in ('clear', 'ltx') and len(pipeline) > 1 else 'output.mp4'
    return folder / 'stages' / phase / filename


def completed_stages(folder):
    folder = Path(folder)
    try:
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    pipeline = config.get('pipeline', [])
    if pipeline not in PIPELINES:
        return []
    finished = []
    for phase in pipeline:
        artifact = artifact_path(folder, phase, pipeline)
        try:
            marker = json.loads((artifact.parent / 'success.json').read_text(encoding='utf-8'))
            if (marker.get('stage') != phase or marker.get('artifact') != artifact.name
                    or marker.get('config_sha256') != signature(config)
                    or not artifact.is_file() or artifact.stat().st_size <= 0):
                break
        except (OSError, ValueError, TypeError):
            break
        finished.append(phase)
    return finished


def stage_config(config, phase, original):
    kind = 'subtitle' if phase in ('clear', 'ltx') else 'watermark'
    regions = [region for region in config['regions'] if region['type'] == kind]
    if not regions:
        raise ValueError('该修复阶段没有对应类型的选区。')
    inference = config['inference']
    options = dict(inference[phase])
    options.update({key: inference[key] for key in ('start_seconds', 'duration_seconds', 'export_crf')
                    if key in inference})
    result = dict(engine=phase, dimensions=config['dimensions'], regions=regions, inference=options,
                  original_filename=config.get('original_filename', 'video'))
    result['roi'] = [min(region['roi'][0] for region in regions), min(region['roi'][1] for region in regions),
                     max(region['roi'][2] for region in regions), max(region['roi'][3] for region in regions)]
    if len(config['pipeline']) > 1:
        if phase in ('clear', 'ltx'):
            result['lossless_output'] = True
        else:
            result['mask_reference'] = str(original)
    return result


class Progress:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.total_weight = sum(WEIGHTS[phase] for phase in pipeline)
        self.last_percent = 5

    def phase(self, name, status):
        value = status.get('percent', 0)
        value = float(value) if type(value) in (int, float) and math.isfinite(value) else 0
        value = min(100, max(0, value))
        previous = sum(WEIGHTS[phase] for phase in self.pipeline[:self.pipeline.index(name)])
        mapped = int(5 + 89 * (previous + WEIGHTS[name] * value / 100) / self.total_weight)
        self.last_percent = max(self.last_percent, min(94, mapped))
        fields = dict(status)
        fields.update(percent=self.last_percent, phase=name, phase_label=LABELS[name],
                      phase_percent=value, pipeline=self.pipeline)
        report(**fields)


def run_phase(phase, folder, source, python, forward):
    scripts = {'clear': 'worker.py', 'ltx': 'ltx_worker.py', 'propainter': 'propainter_worker.py'}
    command = [str(python), '-u', str(ROOT / scripts[phase]),
               str(folder), str(source)]
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    recent = deque(maxlen=30)
    with subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                recent.append(line.rstrip())
                if line.startswith('CLEAR_STATUS '):
                    try:
                        status = json.loads(line[len('CLEAR_STATUS '):])
                    except ValueError:
                        continue
                    if isinstance(status, dict):
                        forward(status)
            if process.wait():
                raise RuntimeError(f'{LABELS[phase]}失败。' + '\n'.join(recent)[-1400:])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def finalize_output(artifact, original, destination, remux_audio):
    temporary = destination.with_name('output.finalizing.mp4')
    try:
        if remux_audio:
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                       '-i', str(artifact), '-i', str(original), '-map', '0:v:0', '-map', '1:a?',
                       '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(temporary)]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if result.returncode:
                raise RuntimeError('原视频音轨封装失败。' + result.stderr[-800:])
        else:
            shutil.copyfile(artifact, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run(folder, clear_source, propainter_source, propainter_python):
    folder = Path(folder).resolve()
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    pipeline = config['pipeline']
    if pipeline not in PIPELINES:
        raise ValueError('视频修复流程无效。')
    original = folder / 'source.video'
    progress = Progress(pipeline)
    finished = completed_stages(folder)
    sources = {'clear': clear_source, 'propainter': propainter_source,
               'ltx': Path(os.environ.get('LTX_ROOT', '/opt/LTX2-ICEdit-Insight'))}
    pythons = {'clear': sys.executable, 'propainter': propainter_python, 'ltx': sys.executable}
    previous = original
    (folder / 'output.mp4').unlink(missing_ok=True)
    try:
        for phase in pipeline:
            artifact = artifact_path(folder, phase, pipeline)
            stage = artifact.parent
            if phase in finished:
                progress.phase(phase, dict(percent=100, stage='沿用已完成的修复阶段'))
                previous = artifact
                continue
            stage.mkdir(parents=True, exist_ok=True)
            (stage / 'success.json').unlink(missing_ok=True)
            artifact.unlink(missing_ok=True)
            source_link = stage / 'source.video'
            temporary_link = stage / 'source.video.tmp'
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(previous)
            temporary_link.replace(source_link)
            atomic_json(stage / 'config.json', stage_config(config, phase, original))
            progress.phase(phase, dict(percent=0, stage='正在准备修复阶段'))
            run_phase(phase, stage, sources[phase], pythons[phase], lambda status: progress.phase(phase, status))
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise RuntimeError(f'{LABELS[phase]}没有生成有效结果。')
            atomic_json(stage / 'success.json', dict(stage=phase, artifact=artifact.name, config_sha256=signature(config)))
            progress.phase(phase, dict(percent=100, stage='修复阶段完成'))
            previous = artifact
        report(percent=97, stage='正在封装原视频音轨', pipeline=pipeline,
               phase='finalize', phase_label='封装导出', phase_percent=0)
        finalize_output(previous, original, folder / 'output.mp4', remux_audio=len(pipeline) > 1)
        report(percent=100, stage='视频修复完成', pipeline=pipeline,
               phase='finalize', phase_label='封装导出', phase_percent=100)
    finally:
        (folder / 'output.finalizing.mp4').unlink(missing_ok=True)
        for phase in pipeline:
            for name in ('partial.mp4', 'partial.mkv'):
                (folder / 'stages' / phase / name).unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    except Exception as error:
        report(error=str(error)[-1800:])
        raise
