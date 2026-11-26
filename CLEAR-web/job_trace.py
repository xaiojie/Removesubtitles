"""Append real operation intervals from any subprocess of one scheduled attempt."""

from contextlib import contextmanager
import json
import os
import time


@contextmanager
def span(stage, **fields):
    path = os.environ.get('SUBTITLE_TRACE_FILE')
    if not path:
        yield
        return
    started_at = time.time()
    clock = time.monotonic()
    ok = False
    try:
        yield
        ok = True
    finally:
        record = dict(stage=stage, started_at=started_at, finished_at=time.time(),
                      duration_seconds=time.monotonic() - clock, pid=os.getpid(), ok=ok, **fields)
        # One O_APPEND write per interval; several workers may trace the same attempt.
        with open(path, 'ab', buffering=0) as stream:
            stream.write((json.dumps(record, ensure_ascii=False) + '\n').encode())
