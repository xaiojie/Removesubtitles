"""Verify distributed metadata against the local video bundle."""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l20"))
from clip_limits import check_clip


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_asset(clip, root, key):
    path = root / clip[key]
    expected_hash = clip["sha256" if key == "input" else "reference_sha256"]
    if fingerprint(path) != expected_hash:
        raise ValueError(f"{path.name} 的 SHA-256 不匹配")
    duration = check_clip(path)
    if abs(float(duration) - clip["duration_seconds"]) > 0.002:
        raise ValueError(f"{path.name} 的实测时长与清单不符")
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ]
    probe = json.loads(subprocess.check_output(command))
    stream = probe["streams"][0]
    if (stream["width"], stream["height"]) != (clip["width"], clip["height"]):
        raise ValueError(f"{path.name} 的图像尺寸与清单不符")
    times = [Fraction(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
    if len(times) != clip["frames"] or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError(f"{path.name} 的解码帧数或时间顺序与清单不符")
    return times


def check_manifest(path, root):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    clips = manifest["clips"]
    if len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError(f"{path.name} 包含重复的样片 ID")
    if "counts" in manifest:
        counts = Counter((clip["style"], clip["split"]) for clip in clips)
        expected = {(style, split): count
                    for style, splits in manifest["counts"].items()
                    for split, count in splits.items()}
        if counts != expected or len({clip["source_sha256"] for clip in clips}) != len(clips):
            raise ValueError(f"{path.name} 的类别数量或独立来源与清单不符")
    for clip in clips:
        times = check_asset(clip, root, "input")
        if clip.get("reference"):
            reference_times = check_asset(clip, root, "reference")
            if any(abs(a - b) > Fraction(1, 1000) for a, b in zip(times, reference_times)):
                raise ValueError(f"{clip['id']} 的参考片与输入显示时间不一致")
        if clip["source_end_frame_exclusive"] - clip["source_start_frame"] != len(times):
            raise ValueError(f"{clip['id']} 的原片帧范围与样片不符")
        if not all(0 <= idx < len(times) for idx in clip["inspect_frames"]):
            raise ValueError(f"{clip['id']} 的复核帧越界")
        for region in [clip["roi"]] + [area["roi"] for area in clip.get("regions", [])]:
            x0, y0, x1, y1 = region
            if not (0 <= x0 < x1 <= clip["width"] and 0 <= y0 < y1 <= clip["height"]):
                raise ValueError(f"{clip['id']} 的选区越界")
    print(f"{path.name} 通过，{len(clips)} 条样片，{sum(c['frames'] for c in clips)} 帧")


def main():
    parser = argparse.ArgumentParser(description="核对清单中的时长、摘要、逐帧时间与选区")
    parser.add_argument("--media-root", type=Path, required=True, help="包含 media 和 refs 的目录")
    parser.add_argument("manifests", nargs="*", type=Path, help="默认核对本目录的 corpus.json")
    args = parser.parse_args()
    for path in args.manifests or [Path(__file__).with_name("corpus.json")]:
        check_manifest(path, args.media_root)


if __name__ == "__main__":
    main()
