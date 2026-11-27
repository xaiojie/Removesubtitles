"""Compare matching full-clip L20 trials. Export timings and media fingerprints without private paths."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

from run_candidate import fingerprint


ORIGINAL_CLEAR = "b7f26f203dbb6f6a4b780a051d7b147c8023d30d"
ORIGINAL_PRODUCT = "2e5bcf2c1e52f1dd4233c8582845a0a954cee4b8"


def read(path):
    return json.loads(path.read_text())


def timing(folder, clip, candidate, revision, frozen=None):
    receipt = folder / "local-validation.json"
    if not receipt.is_file():
        return None
    run, local = read(folder / "trial.json"), read(receipt)
    geom = run.get("geometry", {})
    probe = local["stream"]
    movie = folder / run["output"]
    sha = local["output_sha256"]
    checks = (
        run["validated"], run["returncode"] == 0, run["clip"] == clip["id"],
        run["candidate"] == candidate, run["source_revision"] == revision,
        run["input_sha256"] == clip["sha256"],
        geom.get("coded_regions", run["parameters"]["regions"]) == clip["regions"],
        geom.get("display_regions", clip["regions"]) == run["parameters"]["regions"],
        [probe["width"], probe["height"]] == geom.get("encoded_dimensions", [clip["width"], clip["height"]]),
        int(probe["nb_read_frames"]) == clip["frames"],
        abs(Fraction(probe["avg_frame_rate"]) / Fraction(clip["fps"]) - 1) < .002,
        math.isfinite(run["elapsed_seconds"]) and run["elapsed_seconds"] > 0,
        sha == run.get("output_sha256", sha) and sha == fingerprint(movie),
    )
    if not all(checks):
        raise ValueError(f"Trial or media differs from the selected full clip: {folder.name}")
    if frozen:
        plan = run["parameters"]["resolved_plan"]
        ticket = read(folder.with_name(folder.name + ".conditions.json"))
        conditions = ticket["conditions"]
        if not (ticket.get("output_sha256") == sha and conditions["clip_sha"] == clip["sha256"]
                and conditions["regions"] == clip["regions"] and conditions["revision"] == revision
                and conditions["weights"] == frozen["weights"] and conditions["lite"] == frozen["parameters"]
                and conditions["modes"] == plan["catalog_sha256"]
                and conditions["ocr_graph_sha256"] == frozen["graph_sha256"]):
            raise ValueError(f"Saved input, selection or loaded weight provenance mismatch: {folder.name}")
        events = [json.loads(line) for line in (folder / "stages.jsonl").read_text().splitlines()]
        ocr = [event for event in events if event["stage"] == "lite_ocr"]
        lama = [event for event in events if event["stage"] == "lite_lama"]
        loaded = any(event["stage"] == "lite_ocr_graph_load" and event.get("ok")
                     and event.get("graph_sha256") == frozen["graph_sha256"] for event in events)
        if not (plan["parameters"]["lite"] == frozen["parameters"] and plan["stages"] == ["ocr_lama"]
                and loaded and ocr and lama and all(e.get("ok") and e["device"] == "cuda" for e in ocr + lama)):
            raise ValueError(f"Actual model or parameters differ from the frozen Lite preset: {folder.name}")
    samples = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    observations = [s for s in samples if "gpu_processes" in s and "job_gpu_mib" in s]
    if not observations:
        raise ValueError(f"Missing GPU measurements for {folder.name}")
    outsiders = sum(sum(s["gpu_processes"].values()) > s["job_gpu_mib"] for s in observations)
    return dict(seconds=run["elapsed_seconds"], peak_vram_mib=run["job_peak_vram_mib"],
                mean_gpu_percent=run["gpu_util_mean_pct"], gpu_samples=len(observations),
                monitor_errors=len(samples) - len(observations),
                other_compute_samples=outsiders, sha256=sha)


def paired_ratio(rows, faster, reference):
    common = [r for r in rows if r["split"] in ("tune", "holdout") and
              all((r["runs"].get(k) or {}).get("other_compute_samples") == 0
                  and not r["runs"][k]["monitor_errors"] for k in (faster, reference))]
    measured = sum(r["runs"][faster]["seconds"] for r in common)
    baseline = sum(r["runs"][reference]["seconds"] for r in common)
    return dict(clips=[r["id"] for r in common], seconds=round(measured, 3),
                reference_seconds=round(baseline, 3), ratio=round(measured / baseline, 4) if baseline else None)


def summarize(manifest, trials, frozen):
    clips = read(manifest)["clips"]
    routes = {"lite": (frozen.get("trial_suffix", "lama-border"), "ocr-lama", frozen["source_revision"]),
              "original_lama": ("ocr-lama", "ocr-lama", ORIGINAL_PRODUCT),
              "original_clear_fast": ("clear-fast", "clear-fast", ORIGINAL_CLEAR),
              "pro": ("product-pro", "product-pro", ORIGINAL_PRODUCT),
              "original_standard": ("baseline", "baseline", ORIGINAL_CLEAR),
              "max": ("product-max", "product-max", ORIGINAL_PRODUCT)}
    rows = []
    for clip in clips:
        runs = {}
        for route, (suffix, candidate, revision) in routes.items():
            folder = trials / f"{clip['id']}-{suffix}"
            if route == "original_lama" and clip["id"] == "N03" and not (folder / "local-validation.json").exists():
                folder = trials / "N03-lama-orig-display"
            value = timing(folder, clip, candidate, revision, frozen if route == "lite" else None)
            if value:
                runs[route] = value
        rows.append(dict(id=clip["id"], style=clip["style"], split=clip["split"], frames=clip["frames"],
                         duration_seconds=clip["duration_seconds"], input_sha256=clip["sha256"], runs=runs))
    ratios = {f"{mode}/{against}": paired_ratio(rows, mode, against) for mode, against in (
        ("lite", "original_lama"), ("lite", "original_clear_fast"), ("lite", "pro"),
        ("pro", "original_standard"), ("max", "pro"))}
    return dict(frozen_lite=frozen["source_revision"], main_count=sum(r["split"] in ("tune", "holdout") for r in rows),
                timing="fresh single worker through finalized media, excluding upload and queue",
                comparison="ratio of summed seconds on paired main clips with no sampled outside compute process",
                ratios=ratios, clips=rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "trials", "frozen", "output"):
        cli.add_argument("--" + arg, required=True, type=Path)
    args = cli.parse_args()
    summary = summarize(args.manifest, args.trials, read(args.frozen))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(args.output)
    for label, group in summary["ratios"].items():
        print(label, len(group["clips"]), group["seconds"], group["reference_seconds"], group["ratio"])
