"""Run one isolated candidate and retain its settings, output and real L20 telemetry."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from fractions import Fraction

from clip_limits import check_clip


BASELINE = "b7f26f203dbb6f6a4b780a051d7b147c8023d30d"
PROFILES = {"baseline": "balanced", "clear-fast": "fast", "clear-quality": "quality"}


def json_file(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def video_info(path, *, count_frames=True):
    args = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        args.append("-count_frames")
    args.extend(["-show_entries", "stream=width,height,sample_aspect_ratio,avg_frame_rate,nb_read_frames"
                 ":stream_side_data=rotation", "-of", "json", str(path)])
    return json.loads(subprocess.check_output(args))["streams"][0]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_aspect(probe):
    try:
        numerator, denominator = map(int, probe.get("sample_aspect_ratio", "").split(":"))
    except ValueError:
        return Fraction(1)
    return Fraction(numerator, denominator) if numerator > 0 and denominator > 0 else Fraction(1)


def rotation_degrees(probe):
    return next((int(item["rotation"]) % 360 for item in probe.get("side_data_list", [])
                 if "rotation" in item), 0)


def benchmark_geometry(clip, probe):
    """Apply the Web selection to the corpus's half-open, coded-pixel boxes."""
    width, height = probe["width"], probe["height"]
    if [width, height] != [clip["width"], clip["height"]]:
        raise ValueError("输入编码尺寸与样片清单不符")
    aspect = pixel_aspect(probe)
    # PHP dimensions() and DisplayCapture round half up before applying rotation.
    scaled_width = max(1, math.floor(width * aspect + Fraction(1, 2)))
    rotation = rotation_degrees(probe)
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"无法映射该视频的旋转角度 {rotation}°")
    display = [height, scaled_width] if rotation in (90, 270) else [scaled_width, height]
    coded_regions = clip.get("regions") or [{"type": "subtitle", "roi": clip["roi"]}]
    display_regions = []
    for region in coded_regions:
        left, top, right, bottom = region["roi"]
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"样片选区超出编码尺寸 {region['roi']}")
        # selection.mjs floors the near edge and ceils the far edge in the display raster.
        left = left * scaled_width // width
        right = (right * scaled_width + width - 1) // width
        if rotation == 90:
            box = [top, scaled_width - right, bottom, scaled_width - left]
        elif rotation == 180:
            box = [scaled_width - right, height - bottom, scaled_width - left, height - top]
        elif rotation == 270:
            box = [height - bottom, left, height - top, right]
        else:
            box = [left, top, right, bottom]
        display_regions.append({**region, "roi": box})
    return {"coded_dimensions": [width, height], "coded_regions": coded_regions,
            "sample_aspect_ratio": f"{aspect.numerator}:{aspect.denominator}", "rotation_ccw_degrees": rotation,
            "display_dimensions": display, "display_regions": display_regions,
            "encoded_dimensions": [value + value % 2 for value in display]}


def validate_output(raw, final, geometry):
    same_fps = math.isclose(float(Fraction(raw["avg_frame_rate"])), float(Fraction(final["avg_frame_rate"])),
                            rel_tol=.002)
    if ([final["width"], final["height"]] != geometry["encoded_dimensions"]
            or pixel_aspect(final) != 1 or rotation_degrees(final)
            or raw["nb_read_frames"] != final["nb_read_frames"] or not same_fps):
        raise ValueError(f"成片尺寸、像素比例、方向、帧数或帧率与预期不符 {raw} → {final}，"
                         f"预期编码尺寸 {geometry['encoded_dimensions']}")


def plan(source, regions, options, preset):
    # Resolve through the source being measured, including the pinned baseline checkout.
    helper = ("import json,sys; sys.path.insert(0,sys.argv[1]); from task_options import prepare_task; "
              "print(json.dumps(prepare_task(json.loads(sys.argv[2]), sys.argv[3], 'auto', default_preset=sys.argv[4])))")
    return json.loads(subprocess.check_output([sys.executable, "-c", helper,
                      str(source / "CLEAR-web"), json.dumps(regions), json.dumps(options), preset]))


def telemetry(group):
    now = time.time()
    processes = {}
    ticks = os.sysconf("SC_CLK_TCK")
    for d in Path("/proc").glob("[0-9]*"):
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group:
                continue
            rss = max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE") / 1048576
            processes[int(d.name)] = [rss, (int(fields[11]) + int(fields[12])) / ticks]
        except (OSError, ValueError, IndexError):
            continue
    def smi(kind, fields):
        proc = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or "nvidia-smi query failed")
        return list(csv.reader(proc.stdout.splitlines(), skipinitialspace=True))
    gpu = smi("gpu", "index,utilization.gpu,memory.used,memory.total")
    if len(gpu) != 1:
        raise RuntimeError("本实验需要单卡 L20")
    _, util, used, total = gpu[0]
    apps = smi("compute-apps", "pid,used_gpu_memory")
    memory = {int(pid): int(mib) for pid, mib in apps if pid.isdigit() and mib.isdigit()}
    return {"ts": now, "gpu_util_pct": int(util), "gpu_used_mib": int(used), "gpu_total_mib": int(total),
            "job_gpu_mib": sum(memory.get(pid, 0) for pid in processes), "gpu_processes": memory,
            "job_rss_mib": round(sum(v[0] for v in processes.values()), 1),
            "job_live_cpu_seconds": round(sum(v[1] for v in processes.values()), 2)}


def run(args):
    corpus = json.loads(args.manifest.read_text())
    clip = next((x for x in corpus["clips"] if x["id"] == args.clip), None)
    if not clip:
        raise ValueError(f"unknown clip {args.clip}")
    if clip["split"] == "holdout" and not args.acceptance:
        raise ValueError("独立验收样片只允许在冻结配置后使用 --acceptance")
    src = (args.manifest.resolve().parent / clip["input"]).resolve(strict=True)
    check_clip(src)
    if fingerprint(src) != clip["sha256"]:
        raise ValueError("输入与样片清单签名不符")
    geometry = benchmark_geometry(clip, video_info(src, count_frames=False))
    root, live = args.source.resolve(), args.install.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if args.candidate == "baseline" and (revision != BASELINE or subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "CLEAR", "CLEAR-web", "php"]).returncode):
        raise ValueError("原始基线需要未改动的 b7f26f2 推理源码")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.video").symlink_to(src)
    regions = geometry["display_regions"]
    request = {"dimensions": geometry["display_dimensions"], "regions": regions,
               "original_filename": src.name}
    core = str(live / ".venv/bin/python")
    lite = str(live / "video-subtitle-remover/.venv/bin/python")
    prop = live / "ProPainter"
    env = os.environ | {"CLEAR_WORKSPACE": str(live), "PROPAINTER_ROOT": str(prop), "PROPAINTER_PYTHON": core,
                        "VSR_MODEL_DIR": str(live / "video-subtitle-remover/backend/models"),
                        "LITE_DEVICE": "cuda", "HF_HUB_OFFLINE": "1", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
                        "SUBTITLE_TRACE_FILE": str(output / "stages.jsonl"), "PYTHONUNBUFFERED": "1"}
    env.pop("CLEAR_BENCH_LORA", None)
    env.pop("CLEAR_BENCH_LORA_SHA256", None)
    knobs = json.loads(args.options)
    if args.candidate == "baseline" and (knobs or args.checkpoint):
        raise ValueError("基线不可覆盖参数或权重")
    checkpoint_hash = None
    if args.checkpoint:
        if args.candidate == "ocr-lama":
            raise ValueError("LaMa 不加载 CLEAR 权重")
        checkpoint = args.checkpoint.resolve(strict=True)
        if checkpoint.suffix != ".safetensors":
            raise ValueError("独立权重须为 safetensors")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        env["CLEAR_BENCH_LORA"] = str(checkpoint)
        env["CLEAR_BENCH_LORA_SHA256"] = checkpoint_hash
    if args.candidate in PROFILES or args.candidate == "clear-custom":
        preset = PROFILES.get(args.candidate, "balanced")
        common = {k: knobs[k] for k in ("start_seconds", "duration_seconds", "export_crf") if k in knobs}
        task = plan(root, regions, {"clear": knobs, **common}, preset)
        request.update(task)
        json_file(output / "config.json", request)
        cmd = [core, "-u", str(root / "CLEAR-web/pipeline_worker.py"), str(output), str(root / "CLEAR"),
               str(prop), core]
        result = output / "output.mp4"
    else:
        mode = {"ocr-lama": "lite", "selective": "max", "product-pro": "pro", "product-max": "max"}[args.candidate]
        request["mode"] = mode
        resolver = root / "php/product_modes.php"
        if resolver.is_file():
            request["resolved_plan"] = json.loads(subprocess.check_output(
                ["php", str(resolver), mode, *sorted({r["type"] for r in regions})]))
            for name, changes in knobs.items():
                request["resolved_plan"]["parameters"][name].update(changes)
        elif knobs:
            raise ValueError("此版本的产品入口不支持试验参数")
        json_file(output / "request.json", request)
        cmd = [lite if mode == "lite" else core, "-u", str(root / "php/inference/run.py"), str(output), mode]
        result = output / "result.mp4"
    record = {"candidate": args.candidate, "clip": clip["id"], "style": clip["style"], "split": clip["split"],
              "source_revision": revision, "input_sha256": clip["sha256"], "geometry": geometry, "parameters": request,
              "checkpoint_sha256": checkpoint_hash,
              "source_video_seconds": clip["duration_seconds"], "model_process": "fresh per trial",
              "clock": "worker launch through finalized output, upload and queue excluded"}
    json_file(output / "trial.json", record)
    samples = []
    with (output / "worker.log").open("w", buffering=1) as log, (output / "metrics.jsonl").open("w", buffering=1) as trace:
        started = time.monotonic()
        record["started_at"] = time.time()
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while proc.poll() is None:
                try:
                    sample = telemetry(proc.pid)
                    samples.append(sample)
                    trace.write(json.dumps(sample) + "\n")
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    trace.write(json.dumps({"ts": time.time(), "monitor_error": str(error)}) + "\n")
                try:
                    proc.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait()
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    record.update(stopped_at=time.time(), returncode=proc.returncode, samples=len(samples))
    if samples:
        record.update(job_peak_vram_mib=max(x["job_gpu_mib"] for x in samples),
                      gpu_util_mean_pct=round(sum(x["gpu_util_pct"] for x in samples) / len(samples), 1),
                      job_peak_rss_mib=max(x["job_rss_mib"] for x in samples))
    json_file(output / "trial.json", record)
    if proc.returncode:
        raise RuntimeError(f"推理失败，详见 {output / 'worker.log'}")
    if checkpoint_hash:
        trace = output / "stages.jsonl"
        events = (json.loads(v) for v in trace.read_text().splitlines()) if trace.is_file() else ()
        if not any(e["stage"] == "clear_adapter_load" and e.get("ok") and
                   e.get("checkpoint_sha256") == checkpoint_hash for e in events):
            raise RuntimeError("任务未记录实际加载的试验权重，不能作为权重对照")
    raw, final = video_info(src), video_info(result)
    validate_output(raw, final, geometry)
    record.update(output=result.name, output_sha256=fingerprint(result), output_probe=final, validated=True)
    json_file(output / "trial.json", record)
    print(f"{clip['id']} {args.candidate} {record['elapsed_seconds']} s {result}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source", "install", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--candidate", choices=[*PROFILES, "clear-custom", "ocr-lama", "selective", "product-pro", "product-max"], required=True)
    parser.add_argument("--options", default="{}", help="CLEAR options or parameter groups for the product candidate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--acceptance", action="store_true")
    run(parser.parse_args())
