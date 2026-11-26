"""Summarize a frozen L20 run using measured outputs, frame audits and visual review."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROLES = ('baseline', 'lite', 'pro', 'max')
COMPARISONS = (('lite', 'pro', .5), ('pro', 'baseline', 1.2), ('max', 'pro', 1.3))


def load(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def avg(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(mean(values), 3) if values else None


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def telemetry_summary(folder):
    file = folder / 'metrics.jsonl'
    samples = [json.loads(line) for line in file.read_text().splitlines()] if file.is_file() else []
    measured = [s for s in samples if isinstance(s.get('gpu_processes'), dict) and s.get('job_gpu_mib') is not None]
    external = [max(0, sum(s['gpu_processes'].values()) - s['job_gpu_mib']) for s in measured]
    trace = folder / 'stages.jsonl'
    stages = [json.loads(line) for line in trace.read_text().splitlines()] if trace.is_file() else []
    elapsed = defaultdict(float)
    for stage in stages:
        if stage.get('ok') and 'finished_at' in stage:
            elapsed[stage['stage']] += stage['finished_at'] - stage['started_at']
    return dict(gpu_process_samples=len(measured), other_process_peak_mib=max(external, default=None),
                other_process_samples=sum(n > 0 for n in external),
                stage_seconds={k: round(v, 3) for k,v in elapsed.items()})


def one_clip(clip, role, record, folder, notes, telemetry):
    row = dict(clip=clip['id'], style=clip['style'], split=clip['split'], tier=role,
               source_seconds=clip['duration_seconds'], frames=clip['frames'], roi=clip['roi'],
               width=clip['width'], height=clip['height'], fps=clip['fps'], input_sha256=clip['sha256'],
               status=record.get('status', 'pending'))
    trial = load(folder / 'trial.json')
    if (row['status'] not in ('ok', 'reused', 'verified pilot') or not trial.get('validated')
            or trial.get('clip') != clip['id'] or trial.get('input_sha256') != clip['sha256']):
        return row
    row.update(seconds=trial['elapsed_seconds'], peak_vram_mib=trial.get('job_peak_vram_mib'),
               mean_gpu_pct=trial.get('gpu_util_mean_pct'), peak_rss_mib=trial.get('job_peak_rss_mib'),
               source_revision=trial['source_revision'], directory=record['directory'])
    if telemetry is not None:
        row.update(telemetry_summary(telemetry / record['directory']))
    result = folder / trial['output']
    if result.is_file():
        row['output_sha256'] = file_digest(result)
        row['output'] = trial['output']
    audit = load(folder / 'ocr-review.json')
    if audit and audit.get('frames') == clip['frames']:
        row.update(ocr_matched_frames=audit['matched_frames'], ocr_source_text_frames=audit['input_text_frames'])
    inspection = load(folder / 'inspection.json')
    frames = inspection.get('frame_metrics', [])
    if len(frames) == clip['frames']:
        row.update(outside_roi_mae=avg(frames, 'outside_roi_mae'),
                   reference_roi_mae=avg(frames, 'inside_reference_mae'),
                   temporal_reference_roi_mae=avg(frames, 'temporal_reference_mae'),
                   source_reference_roi_mae=avg(frames, 'source_reference_mae'))
    if (folder / 'comparison.mp4').is_file():
        row['comparison'] = 'comparison.mp4'
    verdict = notes.get(clip['id'], {}).get(role)
    if verdict and row.get('output_sha256') and verdict.get('output_sha256') == row['output_sha256']:
        row.update(visual=verdict['verdict'], observation=verdict['note'],
                   text_clean=verdict.get('text_clean'), quality_pass=verdict.get('quality_pass'))
    return row


def batch_data(path, isolated, notes):
    batch = load(path / 'batch.json')
    if not batch:
        return {}
    jobs = []
    for job in batch['jobs']:
        single = isolated.get((job['clip'], job['mode']), {})
        digest = single.get('output_sha256')
        output_digest = job.get('output_sha256')
        quality = None
        basis = None
        if job['state'] != 'done' or not (job.get('same_frames_and_dimensions') and job.get('same_fps')):
            quality, basis = False, 'incomplete_output'
        elif output_digest and output_digest == digest:
            quality = single.get('quality_pass')
            basis = 'identical_isolated_file' if quality is not None else None
        elif output_digest:
            inspected = notes.get('batch_outputs', {}).get(output_digest, {})
            assessment = notes.get(job['clip'], {}).get(job['mode'], {})
            if (inspected.get('clip'), inspected.get('tier')) == (job['clip'], job['mode']):
                quality = inspected.get('quality_pass')
                basis = inspected.get('basis')
            elif assessment.get('output_sha256') == output_digest:
                quality = assessment.get('quality_pass')
                basis = 'reviewed_batch_file'
        jobs.append(dict(id=job['id'], tier=job['mode'], clip=job['clip'], state=job['state'],
                         quality_pass=quality, quality_basis=basis, source_seconds=job['source_seconds'],
                         upload_submit_seconds=job.get('upload_submit_seconds'),
                         wait_seconds=job.get('wait_seconds'), processing_seconds=job.get('processing_seconds'),
                         attempts=len(job['attempts']), peaks=job['peaks'], output_sha256=job.get('output_sha256'),
                         stage_peaks=batch.get('stage_peaks', {}).get(job['id'], {}),
                         matches_isolated=(output_digest == digest) if digest else None))
    stage_max = defaultdict(dict)
    for stages in batch.get('stage_peaks', {}).values():
        for name, samples in stages.items():
            for field in ('job_gpu_mib','job_rss_mib','job_cpu_pct','job_disk_growth_mib','card_used_mib','gpu_util_pct'):
                value = samples.get(field)
                if value is not None:
                    stage_max[name][field] = max(stage_max[name].get(field, 0), value)
    known = [j for j in jobs if j['quality_pass'] is not None or j['state'] == 'failed']
    passed = [j for j in jobs if j['quality_pass'] is True]
    qualified_rate = (60 * sum(j['source_seconds'] for j in passed) / batch['batch_seconds']
                      if len(known) == len(jobs) and batch['batch_seconds'] > 0 else None)
    return dict(capacity=batch['capacity'], seconds=batch['batch_seconds'], completed=batch['completed'],
                quality_passed=len(passed), quality_pending=len(jobs)-len(known),
                qualified_source_minutes_per_hour=qualified_rate,
                failed=batch['failed'], source_minutes_per_hour=batch['completed_source_minutes_per_hour'],
                stage_peaks=stage_max, jobs=jobs,
                overlaps=[dict(jobs=item['jobs'], kind=item['kind'], seconds=item['seconds'])
                          for item in batch['overlaps']],
                three_gpu_seconds=sum(item['seconds'] for item in batch.get('three_gpu_jobs', [])),
                card_peak_mib=max((j['peaks'].get('card_used_mib') or 0 for j in jobs),default=0) or None)


def write_stage_tables(directory, records, queues):
    columns = ('batch', 'id', 'clip', 'tier', 'stage', 'device', 'phase', 'started_at_unix', 'finished_at_unix')
    peak_fields = ('samples', 'job_gpu_mib', 'job_rss_mib', 'job_cpu_pct', 'job_disk_growth_mib',
                   'gpu_util_pct', 'card_util_mean_pct', 'card_used_mib')
    with (directory / 'batch-timeline.csv').open('w', encoding='utf-8-sig', newline='') as times, \
            (directory / 'stage-peaks.csv').open('w', encoding='utf-8-sig', newline='') as peaks:
        timeline = csv.writer(times)
        timeline.writerow(columns)
        resources = csv.writer(peaks)
        resources.writerow((*columns[:5], 'width', 'height', 'regions', *peak_fields))
        for key, record in records.items():
            if not key.startswith('frozen-'):
                continue
            batch = load(queues / record['directory'] / 'batch.json')
            for job in batch.get('jobs', []):
                ident = (record['directory'], job['id'], job['clip'], job['mode'])
                for span in batch['timelines'][job['id']]:
                    if span.get('ok'):
                        timeline.writerow((*ident, span['stage'], span.get('device'), span.get('phase'),
                                           span['started_at'], span['finished_at']))
                probe = job.get('input_probe') or {}
                for stage, measurements in batch.get('stage_peaks', {}).get(job['id'], {}).items():
                    resources.writerow((*ident, stage, probe.get('width'), probe.get('height'),
                                        json.dumps(job['regions'], separators=(',', ':')),
                                        *(measurements.get(k) for k in peak_fields)))


def write_tuning_runs(destination, trials, clips):
    inputs = {c['id']: c['sha256'] for c in clips if c['split'] in ('tune', 'diagnostic')}
    columns = ['trial', 'clip', 'candidate', 'source_revision', 'input_sha256', 'elapsed_seconds',
               'job_peak_vram_mib', 'gpu_util_mean_pct', 'matched_text_frames', 'checkpoint_sha256', 'parameters']
    with destination.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for path in sorted(trials.glob('*/trial.json')):
            label = path.parent.name
            if label.startswith(('final-', 'baseline-')):
                continue
            trial = load(path)
            if not trial.get('validated') or inputs.get(trial.get('clip')) != trial['input_sha256']:
                continue
            request = trial['parameters']
            parameters = request.get('resolved_plan', {}).get('parameters', request.get('inference', {}))
            audit = load(path.with_name('ocr-review.json'))
            writer.writerow(dict(trial, trial=label, matched_text_frames=audit.get('matched_frames'),
                                 parameters=json.dumps(parameters, ensure_ascii=False, sort_keys=True)))


def summarize(args):
    freeze = load(args.acceptance / 'freeze.json')
    if not freeze or freeze.get('corpus_sha256') != file_digest(args.corpus):
        raise ValueError('冻结记录与样片清单不匹配')
    records = load(args.acceptance / 'records.json')
    clips = load(args.corpus)['clips']
    extras = load(args.corpus.with_name('extras.json')).get('clips', [])
    notes = load(args.notes) if args.notes else {}
    rows = []
    for clip in clips + [c for c in extras if c['id'] == 'x01']:
        for role in (ROLES if clip in clips else ('pro','max')):
            record = records.get(role + '-' + clip['id'], {})
            rows.append(one_clip(clip, role, record, args.trials / record.get('directory',''), notes, args.telemetry))
    measures = {(r['clip'], r['tier']): r for r in rows}
    ratios = []
    for role, other, limit in COMPARISONS:
        pairs = [(measures[c['id'],role],measures[c['id'],other]) for c in clips]
        timed = [(a['seconds'],b['seconds']) for a,b in pairs if 'seconds' in a and 'seconds' in b]
        a, b = (map(sum, zip(*timed)) if timed else (0, 0))
        ratio = a / b if b else None
        ratios.append(dict(tier=role, against=other, limit=limit, paired=len(timed), required=len(clips),
                           total_seconds=round(a, 3), reference_seconds=round(b, 3), ratio=ratio,
                           speed_pass=(ratio <= limit) if len(timed)==len(clips) and ratio is not None else None))
    batches = {key: batch_data(args.queues / record['directory'], measures, notes) for key,record in records.items()
               if key.startswith('frozen-')}
    entries = {}
    for role in ROLES[1:]:
        directory = records.get('live-'+role, {}).get('directory', 'live-'+role)
        check = load(args.queues / directory / 'equivalence.json')
        if check:
            entries[role] = dict(checks=check['checks'], jobs=check['jobs'],
                                 output_sha256=check['records']['web'].get('download_sha256'))
    data = dict(freeze=freeze, per_clip=rows, ratios=ratios, batches=batches, entries=entries)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / 'summary.json', data)
    write_stage_tables(args.out, records, args.queues)
    write_tuning_runs(args.out / 'tuning-runs.csv', args.trials, clips + extras)
    columns = ['clip','style','split','tier','status','source_seconds','width','height','fps','frames','roi',
               'input_sha256','seconds','mean_gpu_pct',
               'peak_vram_mib','peak_rss_mib','ocr_source_text_frames','ocr_matched_frames','outside_roi_mae',
               'reference_roi_mae','temporal_reference_roi_mae','source_reference_roi_mae','other_process_peak_mib','other_process_samples','text_clean','quality_pass','visual','observation','directory','output_sha256']
    with (args.out/'per-clip.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print('已记录', sum('seconds' in row for row in rows), '/', len(rows), '次单任务测试')
    return data


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('acceptance','corpus','trials','queues','out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--notes', type=Path)
    cli.add_argument('--telemetry', type=Path, help='每个单任务的 metrics.jsonl 和 stages.jsonl')
    summarize(cli.parse_args())
