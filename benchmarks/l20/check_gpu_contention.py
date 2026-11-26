"""Check whether sampled GPU processes belonged to the measured batch."""
import argparse
import json
from pathlib import Path

from run_queue_batch import GPU_STAGES, merged


def audit(batch_path, job_root):
    batch = json.loads(batch_path.read_text())
    samples = []
    owners = {}
    for job in batch['jobs']:
        for attempt in job['attempts']:
            path = job_root / job['id'] / 'runs' / attempt['run_id'] / 'metrics.jsonl'
            for line in path.read_text().splitlines():
                sample = json.loads(line)
                owners.update({str(pid): job['id'] for pid in sample.get('pids', [])})
                samples.append(sample)

    spans = {job: merged(stages, GPU_STAGES) for job, stages in batch.get('timelines', {}).items()}
    peak = 0
    observed = 0
    external = 0
    concurrent = []
    for sample in samples:
        processes = sample.get('gpu_processes')
        if not isinstance(processes, dict):
            continue
        observed += 1
        other_mib = sum(mib for pid, mib in processes.items() if str(pid) not in owners)
        peak = max(peak, other_mib)
        external += int(other_mib > 0)
        resident_jobs = {owners[str(pid)] for pid, mib in processes.items() if str(pid) in owners and mib > 0}
        concurrent.append(sum(any(begin <= sample['ts'] <= end for begin, end in spans.get(job, []))
                              for job in resident_jobs))
    return dict(jobs=len(batch['jobs']), samples=len(samples), process_samples=observed,
                external_samples=external, external_peak_mib=peak if observed else None,
                max_resident_jobs_in_model_calls=max(concurrent, default=None),
                at_least_two_in_model_calls_samples=sum(n >= 2 for n in concurrent),
                three_in_model_calls_samples=sum(n >= 3 for n in concurrent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('acceptance', 'queues', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--live-batch', type=Path)
    parser.add_argument('--live-data', type=Path)
    args = parser.parse_args()
    if bool(args.live_batch) != bool(args.live_data):
        parser.error('--live-batch and --live-data must be supplied together')
    records = json.loads((args.acceptance / 'records.json').read_text())
    output = {}
    for key, record in records.items():
        if key.startswith('frozen-') and record['status'] in ('ok', 'reused'):
            directory = args.queues / record['directory']
            output[key] = audit(directory / 'batch.json', directory / 'jobs')
    if args.live_batch:
        output['live-mixed-entries'] = audit(args.live_batch, args.live_data)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
