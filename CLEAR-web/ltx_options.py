"""Validated options for the separately installed LTX test engine."""
from dataclasses import dataclass, fields
import json
import math

from inference_options import InferenceOptions


@dataclass(frozen=True)
class LTXOptions:
    prompt: str = ('Clean natural footage with the original subjects, background, camera motion, '
                   'lighting and fine textures preserved. No subtitles or text overlays.')
    mode: str = 'subtitle_rm'
    lora_scale: float = 1.0
    seed: int = 42
    sigma_profile: str = 'workflow'
    single_stage: bool = True
    chunk_frames: int = 97
    overlap_frames: int = 16
    model_width: int = 0
    model_height: int = 0
    video_conditioning_strength: float = 1.0
    start_seconds: float = 0
    duration_seconds: float | None = None
    export_crf: int = 10

    def frame_range(self, fps, total):
        return InferenceOptions(start_seconds=self.start_seconds,
                                duration_seconds=self.duration_seconds).frame_range(fps, total)

    def model_shape(self, width, height):
        if self.model_width:
            return self.model_width, self.model_height
        scale = min(1, 1248 / max(width, height))
        return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)

    @classmethod
    def parse(cls, raw):
        data = json.loads(raw)
        if not isinstance(data, dict) or data.keys() - {field.name for field in fields(cls)}:
            raise ValueError('LTX 参数包含未知字段。')
        options = cls(**data)
        if not isinstance(options.prompt, str) or not options.prompt.strip() or len(options.prompt) > 6000:
            raise ValueError('LTX 提示词需要 1 到 6000 个字符。')
        if options.mode not in ('subtitle_rm', 'watermark_rm'):
            raise ValueError('LTX 模式必须为 subtitle_rm 或 watermark_rm。')
        if options.sigma_profile not in ('workflow', 'distilled'):
            raise ValueError('LTX sigma 必须为 workflow 或 distilled。')
        if options.single_stage is not True:
            raise ValueError('当前 LTX 网页使用单阶段推理。')
        for name, low, high in (('seed', 0, 2**32-1),
                                ('overlap_frames', 0, 64), ('model_width', 0, 2048),
                                ('model_height', 0, 2048), ('export_crf', 0, 30)):
            value = getattr(options, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'LTX {name} 必须为 {low} 到 {high} 的整数。')
        if (type(options.chunk_frames) is not int or options.chunk_frames < 9
                or options.chunk_frames % 8 != 1 or options.overlap_frames >= options.chunk_frames):
            raise ValueError('LTX 分段帧数需要符合 8n+1，重叠帧数需要小于分段帧数。')
        if bool(options.model_width) != bool(options.model_height) or options.model_width % 32 or options.model_height % 32:
            raise ValueError('LTX 推理宽高需要同时为 0（自动），或同时为 32 的倍数。')
        for name, low, high in (('lora_scale', 0, 2), ('video_conditioning_strength', 0, 2)):
            value = getattr(options, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'LTX {name} 需要在 {low} 到 {high} 之间。')
        InferenceOptions.parse(json.dumps({name: getattr(options, name)
                                          for name in ('start_seconds', 'duration_seconds', 'export_crf')}))
        return options
