
from dataclasses import asdict
import json

from inference_options import InferenceOptions
from propainter_options import ProPainterOptions

COMMON_DEFAULTS = {"start_seconds": 0, "duration_seconds": None, "export_crf": 10}
OPTION_TYPES = {"clear": InferenceOptions, "propainter": ProPainterOptions}


def prepare_task(regions, raw, legacy_engine="auto"):
    kinds = {region["type"] for region in regions}
    pipeline = []
    if "subtitle" in kinds:
        pipeline.append("clear")
    if "watermark" in kinds:
        pipeline.append("propainter")
    if not pipeline:
        raise ValueError("请至少框选一处字幕或水印。")
    if legacy_engine not in ("auto", "clear", "propainter"):
        raise ValueError("处理模型无效。")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("推理参数必须是对象。")
    structured = legacy_engine == "auto" or bool({"clear", "propainter"} & data.keys())
    if structured:
        allowed = set(COMMON_DEFAULTS) | set(OPTION_TYPES)
        if data.keys() - allowed:
            raise ValueError("自动处理参数包含未知字段。")
        common = {name: data.get(name, value) for name, value in COMMON_DEFAULTS.items()}
        groups = data
    else:
        previous = asdict(OPTION_TYPES[legacy_engine].parse(raw))
        common = {name: previous[name] for name in COMMON_DEFAULTS}
        groups = {legacy_engine: previous}
    normalized = dict(common)
    for engine in pipeline:
        group = groups.get(engine, {})
        if not isinstance(group, dict):
            raise ValueError("模型参数必须是对象。")
        values = {**group, **common}
        normalized[engine] = asdict(OPTION_TYPES[engine].parse(json.dumps(values)))
    return {
        "engine": pipeline[0] if len(pipeline) == 1 else "hybrid",
        "pipeline": pipeline,
        "inference": normalized,
    }

