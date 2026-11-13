from dataclasses import dataclass, fields
import json
import math

from inference_options import InferenceOptions


@dataclass(frozen=True)
class ProPainterOptions:
    start_seconds: float = 0
    duration_seconds: float | None = None
    segment_mode: str = "shots"
    resolution: str = "native"
    export_crf: int = 10
    mask_dilation: int = 8
    neighbor_length: int = 10
    ref_stride: int = 5
    subvideo_length: int = 40
    raft_iter: int = 20
    fp16: bool = True
    watermark_mask_mode: str = "auto"

    def frame_range(self, fps, total):
        return InferenceOptions.frame_range(self, fps, total)

    @classmethod
    def parse(cls, raw):
        values = json.loads(raw)
        names = {field.name for field in fields(cls)}
        if not isinstance(values, dict) or values.keys() - names:
            raise ValueError("ProPainter 推理参数无效。")
        options = cls(**values)
        if type(options.fp16) is not bool:
            raise ValueError("fp16 必须是布尔值。")
        ranges = (
            ("mask_dilation", 0, 32),
            ("neighbor_length", 2, 20),
            ("ref_stride", 1, 60),
            ("subvideo_length", 10, 120),
            ("raft_iter", 1, 40),
            ("export_crf", 0, 30),
        )
        for name, lower, upper in ranges:
            value = getattr(options, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} 必须是 {lower} 到 {upper} 之间的整数。")
        if options.segment_mode not in ("shots", "fixed"):
            raise ValueError("segment_mode 必须是 shots 或 fixed。")
        if options.resolution not in ("native", "auto"):
            raise ValueError("resolution 必须是 native 或 auto。")
        if options.watermark_mask_mode not in ("auto", "rectangle"):
            raise ValueError("watermark_mask_mode 必须是 auto 或 rectangle。")
        start = options.start_seconds
        duration = options.duration_seconds
        if type(start) not in (int, float) or not math.isfinite(start) or not 0 <= start <= 1e9:
            raise ValueError("start_seconds 必须是非负有限数字。")
        if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 1e9):
            raise ValueError("duration_seconds 必须是正的有限数字。")
        return options
