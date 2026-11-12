from dataclasses import dataclass, fields
import json
import math


@dataclass(frozen=True)
class InferenceOptions:
    steps: int = 5
    cfg_scale: float = 1.0
    lora_scale: float = 1.0
    seed: int = 123
    chunk_frames: int = 21
    overlap_frames: int = 4
    context: str = "frame"
    start_seconds: float = 0
    duration_seconds: float | None = None
    resolution: str = "auto"
    export_crf: int = 14
    blend_mode: str = "lines"
    segment_mode: str = "shots"

    def frame_range(self, fps, total):
        first = math.ceil(self.start_seconds * fps - 1e-8)
        if total > 0 and first >= total:
            raise ValueError("开始时间超出视频范围。")
        last = None
        if self.duration_seconds is not None:
            end = math.ceil((self.start_seconds + self.duration_seconds) * fps - 1e-8)
            last = min(end, total) if total > 0 else end
        if last is not None and first >= last:
            raise ValueError('所选时间段没有视频帧。')
        return first, last

    @classmethod
    def parse(cls, raw):
        data = json.loads(raw)
        if not isinstance(data, dict) or data.keys() - {f.name for f in fields(cls)}:
            raise ValueError('inference 必须是有效的推理参数对象。')
        options = cls(**data)
        for name, low, high in (
            ('steps', 1, 24), ('seed', 0, 2**32-1), ('chunk_frames', 5, 33),
            ('overlap_frames', 0, 16), ('export_crf', 0, 30),
        ):
            value = getattr(options, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} 必须是 {low} 到 {high} 之间的整数。')
        for name, low, high in (('cfg_scale', 1, 3), ('lora_scale', 0.5, 2)):
            value = getattr(options, name)
            if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
                raise ValueError(f'{name} 必须在 {low} 到 {high} 之间。')
        if options.chunk_frames % 4 != 1 or options.overlap_frames >= options.chunk_frames:
            raise ValueError('chunk_frames 必须符合 4n+1，overlap_frames 必须小于 chunk_frames。')
        if options.context not in ("roi", "frame"):
            raise ValueError('context 必须是 roi 或 frame。')
        if options.resolution not in ("auto", "native"):
            raise ValueError('resolution 必须是 auto 或 native。')
        start, duration = options.start_seconds, options.duration_seconds
        if type(start) not in (int, float) or not 0 <= start <= 1e9:
            raise ValueError('start_seconds 必须是非负有限数字。')
        if duration is not None and (type(duration) not in (int, float) or not 0 < duration <= 1e9):
            raise ValueError('duration_seconds 必须是正的有限数字。')
        if options.blend_mode not in ("lines", "rectangle"):
            raise ValueError("blend_mode must be lines or rectangle.")
        if options.segment_mode not in ("shots", "fixed"):
            raise ValueError("segment_mode must be shots or fixed.")
        return options
