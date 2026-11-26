"""Prepare aligned visual evidence, including every frame of the final second."""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from clip_limits import check_clip

WEB = Path(__file__).resolve().parents[2] / "CLEAR-web"
sys.path.insert(0, str(WEB))
from comparison_video import create_comparison
from video_capture import open_video_capture


def preview(crops, frame):
    tiles = []
    for image in crops:
        h, w = image.shape[:2]
        scale = min(460 / w, 142 / h)
        thumb = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        tiles.append(thumb)
    line = np.hstack(tiles)
    line = cv2.copyMakeBorder(line, 17, 3, 3, 3, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(line, str(frame), (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return line


def sheet(file, images, columns=1):
    if not images:
        return
    while len(images) % columns:
        images.append(np.zeros_like(images[0]))
    sheet = np.vstack([np.hstack(images[i:i+columns]) for i in range(0, len(images), columns)])
    if not cv2.imwrite(str(file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise OSError(f"无法保存 {file}")


def review(manifest, clip_id, output, folder, region_index, comparison):
    clip = next(c for c in json.loads(manifest.read_text())["clips"] if c["id"] == clip_id)
    source = manifest.parent / clip["input"]
    check_clip(source)
    reference = manifest.parent / clip["reference"] if clip.get("reference") else None
    has_reference = bool(reference and reference.is_file())
    media = [source, output] + ([reference] if has_reference else [])
    caps = [open_video_capture(str(p), cv2.CAP_FFMPEG) for p in media]
    roi = (clip["regions"][region_index]["roi"] if "regions" in clip else clip["roi"])
    left, top, right, bottom = roi
    selection = np.s_[top:bottom, left:right]
    outside = np.ones((clip["height"], clip["width"]), dtype=np.uint8)
    for region in clip.get("regions", [{"roi": roi}]):
        x, y, r, b = region["roi"]
        outside[y:b, x:r] = 0
    folder.mkdir(parents=True, exist_ok=True)
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    final_second = math.ceil(fps)
    chosen = set(clip.get("inspect_frames", [])) | {0, clip["frames"] - 1}
    chosen.update(range(max(0, clip["frames"] - final_second), clip["frames"], max(1, round(fps / 5))))
    frames, tail, rows = {}, [], []
    previous = previous_tile = worst_tile = None
    worst = -1.
    try:
        for n in range(clip["frames"]):
            decoded = [cap.read() for cap in caps]
            if not all(ok for ok, _ in decoded):
                raise ValueError(f"视频在第 {n} 帧提前结束")
            values = [im for _, im in decoded]
            if any(im.shape[:2] != outside.shape for im in values):
                raise ValueError("成片或参考片尺寸与清单不同")
            crops = [im[selection] for im in values]
            orig, repaired = crops[:2]
            diff = cv2.absdiff(orig, repaired)
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            # Inspect unchanged highlights with a local contrast edge; backgrounds can also match.
            low = cv2.erode(gray, np.ones((9, 9), np.uint8))
            strokes = (gray >= 155) & (cv2.subtract(gray, low) >= 72)
            preserved = (float(np.mean(diff[strokes].max(axis=1) < 20)) if strokes.sum() >= 40 else None)
            outer_mae = float(np.mean(cv2.mean(cv2.absdiff(values[0], values[1]), mask=outside)[:3]))
            row = {"frame": n, "source_highlight_retained": preserved, "outside_roi_mae": outer_mae}
            if has_reference:
                row["inside_reference_mae"] = float(cv2.absdiff(repaired, crops[2]).mean())
                row["source_reference_mae"] = float(cv2.absdiff(orig, crops[2]).mean())
            tiny = cv2.resize(values[0], (64, 36), interpolation=cv2.INTER_AREA)
            tile = preview(crops, n)
            if previous is not None and cv2.absdiff(tiny, previous).mean() > 35:
                frames[n-1] = previous_tile
                frames[n] = tile
                row["scene_change"] = True
            if n in chosen:
                frames[n] = tile
            if n >= clip["frames"] - final_second:
                tail.append(tile)
            if preserved is not None and preserved > worst:
                worst, worst_tile = preserved, (n, tile)
            rows.append(row)
            previous, previous_tile = tiny, tile
        if any(cap.grab() for cap in caps):
            raise ValueError("成片或参考片有额外帧")
    finally:
        for cap in caps:
            cap.release()
    if worst_tile:
        frames[worst_tile[0]] = worst_tile[1]
    sheet(folder / "inspection.jpg", [frames[n] for n in sorted(frames)])
    sheet(folder / "last-second.jpg", tail, 2)
    info = {"clip": clip_id, "roi": roi, "columns": ["input", "output"] + (["clean reference"] if has_reference else []),
            "inspection_frames": sorted(frames), "fps": fps, "frame_metrics": rows}
    (folder / "inspection.json").write_text(json.dumps(info, ensure_ascii=False))
    if comparison:
        create_comparison(source, output, folder / "comparison.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--region-index", type=int, default=0)
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    review(args.manifest, args.clip, args.result, args.out, args.region_index, args.comparison)
