"""Persistent, job-scoped LTX SDK process using its own Python environment.

The web worker owns video decoding and compositing. This process only receives
prepared, resolution-aligned clips and runs the installed LTX pipeline.
"""
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sys


def emit(event, **values):
    print(event + ' ' + json.dumps(values, ensure_ascii=False), flush=True)


def run(source, configuration):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    sys.path.insert(0, str(source))
    import torch
    import run_pipeline as sdk
    from ltx_pipelines.utils import blocks

    options = json.loads(configuration.read_text(encoding='utf-8'))['inference']
    mode = sdk.resolve_mode_settings(options['mode'])
    pipeline = sdk.build_pipeline(mode.lora_path, lora_strength=options['lora_scale'],
                                  streaming_prefetch_count=None)
    emit('CLEAR_STATUS', percent=2, stage='LTX 正在编码提示词，首次加载需要数分钟')
    with torch.inference_mode():
        contexts = pipeline['prompt_encoder']([options['prompt']], streaming_prefetch_count=None)

    class PromptEncoder:
        def __call__(self, prompts, **kwargs):
            if prompts != [options['prompt']]:
                raise ValueError('LTX prompt changed within one job.')
            return contexts

    pipeline['prompt_encoder'] = PromptEncoder()
    # The SDK's default context destroys model tensors on exit. Keep the model
    # contexts alive across clips and release everything when this job exits.
    original_build = sdk.FixedBuilder.build
    models = {}

    def cached_build(builder, device=None, dtype=None, **kwargs):
        key = (builder.model_class_configurator.__name__, str(builder.model_path),
               tuple((lora.path, lora.strength) for lora in builder.loras), str(device), str(dtype))
        if key not in models:
            models[key] = original_build(builder, device=device, dtype=dtype, **kwargs)
        return models[key]

    @contextmanager
    def retained_model(model):
        yield model

    sdk.FixedBuilder.build = cached_build
    original_context = blocks.gpu_model
    blocks.gpu_model = retained_model
    emit('LTX_READY')
    try:
        for line in sys.stdin:
            request = json.loads(line)
            torch.manual_seed(options['seed'])
            torch.cuda.manual_seed_all(options['seed'])
            sdk.run_inference(
                pipeline=pipeline, prompt=options['prompt'], input_video=request['input'],
                output_path=request['output'], width=request['width'], height=request['height'],
                num_frames=request['frames'], frame_rate=request['fps'], seed=options['seed'],
                single_stage=True, streaming_prefetch_count=None,
                sigma_profile=options['sigma_profile'],
                video_conditioning_strength=options['video_conditioning_strength'],
                use_white_mask=mode.use_white_mask,
            )
            emit('LTX_RESULT', output=request['output'])
    finally:
        sdk.FixedBuilder.build = original_build
        blocks.gpu_model = original_context
        models.clear()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
    except Exception as error:
        emit('LTX_ERROR', error=str(error)[-1800:])
        raise
