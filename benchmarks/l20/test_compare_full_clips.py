from compare_full_clips import paired_ratio


def test_speed_ratio_requires_the_same_actual_output_geometry_and_frame_rate():
    video = {"width": 1280, "height": 720, "fps": "24", "frames": 720}

    def measure(seconds, output=video):
        return dict(seconds=seconds, other_compute_samples=0, monitor_errors=0, video=output)

    def pair(cid, output):
        return {"id": cid, "split": "tune", "runs": {"lite": measure(40, output), "pro": measure(100)}}

    rows = [pair("same", video), pair("padding", dict(video, height=736)),
            pair("rate", dict(video, fps="24000/1001")), pair("short", dict(video, frames=719)),
            pair("unknown", None)]
    held_out = pair("holdout", video)
    held_out["split"] = "holdout"
    rows.append(held_out)
    summary = paired_ratio(rows, "lite", "pro")
    assert summary["clips"] == ["same", "holdout"]
    assert summary["seconds"] == 80 and summary["reference_seconds"] == 200
    assert summary["ratio"] == .4
    assert {r["id"] for r in summary["unmatched_outputs"]} == {"padding", "rate", "short", "unknown"}
