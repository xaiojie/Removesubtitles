"""Check the actual time span of a benchmark input before transfer or inference."""
import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path


MAX_SECONDS = 120


class ClipLimitError(ValueError):
    pass


def positive(value):
    try:
        value = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def checked_duration(probe):
    videos = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ClipLimitError("没有可供测试的视频流")
    spans = [positive(probe.get("format", {}).get("duration"))]
    for stream in videos:
        spans.append(positive(stream.get("duration")))
        ticks, unit = positive(stream.get("duration_ts")), positive(stream.get("time_base"))
        if ticks and unit:
            spans.append(ticks * unit)
    duration = max((value for value in spans if value is not None), default=None)
    if duration is None:
        raise ClipLimitError("无法确定测试视频时长")
    if duration > MAX_SECONDS:
        raise ClipLimitError(f"测试片段为 {float(duration):.3f} 秒，超过 {MAX_SECONDS} 秒，请先在本机剪辑")
    return duration


def check_clip(path):
    command = ["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,duration,duration_ts,time_base:stream_disposition=attached_pic",
               "-of", "json", str(Path(path).resolve())]
    try:
        probe = subprocess.run(command, text=True, capture_output=True, check=True, timeout=30)
        return checked_duration(json.loads(probe.stdout))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ClipLimitError(f"无法读取测试视频 {path}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", type=Path)
    for path in parser.parse_args().clips:
        try:
            seconds = check_clip(path)
        except ClipLimitError as error:
            parser.exit(1, f"{path}: {error}\n")
        print(f"{path}: {float(seconds):.3f} s")


if __name__ == "__main__":
    main()
