"""Append real operation intervals from any subprocess of one scheduled attempt."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time


def append(path, record):
    # One O_APPEND write per record; several workers may trace the same attempt.
    with path.open('ab', buffering=0) as stream:
        stream.write((json.dumps(record, ensure_ascii=False) + '\n').encode())


def allocator_memory(device):
    torch = sys.modules.get('torch')
    if not str(device).startswith('cuda') or torch is None or not torch.cuda.is_initialized():
        return {}
    try:
        return dict(torch_reserved_mib=torch.cuda.memory_reserved() / 1024**2,
                    torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2)
    except RuntimeError:
        # Preserve the original CUDA exception after a failed operation.
        return {}


@contextmanager
def span(stage, **fields):
    target = os.environ.get('SUBTITLE_TRACE_FILE')
    if not target:
        yield
        return
    path = Path(target)
    clock = time.monotonic_ns()
    record = dict(stage=stage, started_at=time.time(), pid=os.getpid(), span_id=f'{os.getpid()}-{clock}', **fields)
    append(path.with_name('stage_starts.jsonl'), record)
    ok = False
    try:
        yield
        ok = True
    finally:
        record.update(finished_at=time.time(), duration_seconds=(time.monotonic_ns() - clock) / 1e9, ok=ok,
                      **allocator_memory(fields.get('device')))
        try:
            append(path, record)
        except OSError:
            if ok:
                raise
