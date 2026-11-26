"""Measure independent videos with the production API and scheduler on one GPU."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import time
import urllib.request
from itertools import combinations
from fractions import Fraction

from clip_limits import check_clip
from run_candidate import json_file, video_info


GPU_STAGES = {"lite_lama", "clear_inference", "propainter_inference"}
CPU_STAGES = {"shot_scan", "lite_ocr", "max_review", "ffmpeg_finalize"}


def lines(file):
    return [json.loads(line) for line in file.read_text().splitlines()] if file.is_file() else []


def merged(spans, stages):
    intervals = sorted((s["started_at"], s["finished_at"]) for s in spans
                       if s["stage"] in stages and s.get("ok"))
    result = []
    for start, end in intervals:
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def intersection(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if lo < hi:
            result.append([lo, hi])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def peaks(samples):
    fields = ("job_gpu_mib", "job_rss_mib", "job_cpu_pct", "job_disk_growth_mib", "gpu_util_pct")
    result = {field: max((s[field] for s in samples if s.get(field) is not None), default=None)
              for field in fields}
    result["card_used_mib"] = max((s["gpu_total_mib"] - s["gpu_free_mib"] for s in samples
                                  if s.get("gpu_total_mib") is not None and s.get("gpu_free_mib") is not None),
                                 default=None)
    return result


def report(data, jobs, started, limit):
    db = sqlite3.connect(data / "queue.sqlite")
    db.row_factory = sqlite3.Row
    timelines = {}
    stage_peaks = {}
    rows = []
    for job in jobs:
        job_id = job["id"]
        folder = data / job_id
        state = dict(db.execute("SELECT state,error FROM jobs WHERE id=?", (job_id,)).fetchone())
        attempts = [dict(x) for x in db.execute(
            "SELECT * FROM job_runs WHERE job_id=? ORDER BY started_at", (job_id,))]
        stages, samples = [], []
        for attempt in attempts:
            run = folder / "runs" / attempt["run_id"]
            stages.extend(lines(run / "stages.jsonl"))
            samples.extend(lines(run / "metrics.jsonl"))
        timelines[job_id] = stages
        stage_peaks[job_id] = {}
        for stage in sorted({s["stage"] for s in stages if s.get("ok")}):
            spans = merged(stages, {stage})
            matching = [sample for sample in samples if any(a <= sample["ts"] <= b for a, b in spans)]
            stage_peaks[job_id][stage] = dict(samples=len(matching), **peaks(matching))
        row = dict(job, **state, attempts=attempts, peaks=peaks(samples))
        if attempts:
            row["wait_seconds"] = sum(a["started_at"] - a["queued_at"] for a in attempts)
            row["processing_seconds"] = sum(a["finished_at"] - a["started_at"] for a in attempts)
        if state["state"] == "done":
            output = folder / "result.mp4"
            before, after = video_info(folder / "source.video"), video_info(output)
            row.update(output=str(output), output_probe=after, input_probe=before,
                       same_frames_and_dimensions=all(before[k] == after[k] for k in ("width", "height", "nb_read_frames")),
                       same_fps=math.isclose(float(Fraction(before["avg_frame_rate"])),
                                             float(Fraction(after["avg_frame_rate"])), rel_tol=.002),
                       output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        rows.append(row)
    db.close()
    overlaps = []
    for a, b in combinations(timelines, 2):
        for kind, aa, bb in (("gpu_gpu", GPU_STAGES, GPU_STAGES), ("gpu_cpu", GPU_STAGES, CPU_STAGES),
                             ("cpu_gpu", CPU_STAGES, GPU_STAGES), ("cpu_cpu", CPU_STAGES, CPU_STAGES)):
            windows = intersection(merged(timelines[a], aa), merged(timelines[b], bb))
            if windows:
                overlaps.append({"jobs": [a, b], "kind": kind, "seconds": sum(y-x for x, y in windows),
                                 "intervals_unix": windows})
    ended = max((x["finished_at"] for row in rows for x in row["attempts"]), default=started)
    wall = max(0, ended - started)
    done = [x for x in rows if x["state"] == "done" and x["same_frames_and_dimensions"] and x["same_fps"]]
    return dict(capacity=limit, jobs=rows, started_at=started, finished_at=ended, batch_seconds=wall,
                completed=len(done), failed=len(rows)-len(done), overlaps=overlaps, timelines=timelines, stage_peaks=stage_peaks,
                completed_source_minutes_per_hour=sum(x["source_seconds"] for x in done)*60/wall if wall else 0)


def main(args):
    root = Path(__file__).resolve().parents[2]
    install = (args.install or root).resolve()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    corpus = {c["id"]: c for c in json.loads(manifest.read_text())["clips"]}
    jobs = []
    for spec in args.jobs:
        mode, clip_id = spec.split(":", 1)
        clip = corpus[clip_id]
        if mode not in ("lite", "pro", "max") or (clip["split"] == "holdout" and not args.acceptance):
            raise ValueError(f"无效档位或未冻结配置的独立样片 {spec}")
        file = (manifest.parent / clip["input"]).resolve(strict=True)
        seconds = check_clip(file)
        if hashlib.sha256(file.read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"样片校验失败 {clip_id}")
        regions = clip.get("regions") or [dict(type="subtitle", roi=clip["roi"])]
        jobs.append(dict(clip=clip_id, mode=mode, file=str(file), source_seconds=float(seconds),
                         sha256=clip["sha256"], regions=regions))
    if len({j["sha256"] for j in jobs}) < 2:
        raise ValueError("并发批次需要至少两个不同视频")
    output.mkdir(parents=True, exist_ok=False)
    data = output / "jobs"
    data.mkdir()
    token = secrets.token_urlsafe(32)
    env = dict(os.environ, SUBTITLE_API_TOKEN=token, SUBTITLE_DATA_DIR=str(data), SUBTITLE_MAX_JOBS=str(args.limit),
               CLEAR_WORKSPACE=str(install), CLEAR_PYTHON=str(install / ".venv/bin/python"),
               PROPAINTER_ROOT=str(install / "ProPainter"), PROPAINTER_PYTHON=str(install / ".venv/bin/python"),
               VSR_PYTHON=str(install / "video-subtitle-remover/.venv/bin/python"),
               VSR_MODEL_DIR=str(install / "video-subtitle-remover/backend/models"), LITE_DEVICE="cuda",
               HF_HUB_OFFLINE="1", PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True")
    # Isolate only the queue and port. Production users and their queue remain available.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        host = f"127.0.0.1:{sock.getsockname()[1]}"
    base = "http://" + host
    api = root / "php/public"
    with (output / "queue.log").open("w") as log:
        server = subprocess.Popen(["php", "-d", "upload_max_filesize=2G", "-d", "post_max_size=2100M", "-S", host, "-t", str(api), str(api / "index.php")],
                                  env=env, stdout=log, stderr=log)
        try:
            req = urllib.request.Request(base+"/api/modes", headers={"Authorization": "Bearer "+token})
            for _ in range(60):
                try:
                    with urllib.request.urlopen(req, timeout=2) as response:
                        catalog = json.load(response)
                    break
                except OSError:
                    if server.poll() is not None:
                        raise RuntimeError("基准接口未启动")
                    time.sleep(.1)
            else:
                raise RuntimeError("基准接口就绪超时")
            for job in jobs:
                args_curl = ["curl", "-fsS", "--max-time", "90", "-H", "Authorization: Bearer "+token,
                             "-F", "video=@"+job["file"], "-F", "mode="+job["mode"],
                             "--form-string", "regions="+json.dumps(job["regions"]),
                             "--form-string", "catalog_sha256="+catalog["catalog_sha256"], base+"/api/jobs"]
                response = json.loads(subprocess.check_output(args_curl))
                job["id"] = response["id"]
            json_file(output / "inputs.json", jobs)
            started = time.time()
            subprocess.run(["php", str(root / "php/bin/worker.php"), "--drain"], check=True,
                           env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            result = report(data, jobs, started, args.limit)
            result["catalog_sha256"] = catalog["catalog_sha256"]
            result["source_revision"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],text=True).strip()
            json_file(output / "batch.json", result)
            print(f"{result['completed']}/{len(jobs)} completed in {result['batch_seconds']:.2f} s", flush=True)
        finally:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--install", type=Path, help="复用现有模型和 Python 环境的安装目录")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(1, 17), required=True)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("jobs", nargs="+", metavar="MODE:CLIP")
    main(parser.parse_args())
