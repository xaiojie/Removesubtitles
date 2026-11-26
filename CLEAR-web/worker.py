from functools import lru_cache
from itertools import chain, islice
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg

from video_capture import open_video_capture
import numpy as np
from PIL import Image

from inference_options import InferenceOptions
from regions import bounds, regions_from_config
from video_segments import decoded_frames, find_shot_cuts, video_batches
from job_trace import span
from adaptive_processing import prepare_subtitles, composite_subtitle, QualityReport
from subtitle_compositing import camera_path, caption_mask, composite_caption, large_caption_guidance, light_caption_cards, visible_background

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('CLEAR_WORKSPACE', ROOT.parent)).resolve()
MODELS = WORKSPACE / 'CLEAR-models'
WAN = MODELS / 'Wan2.1-T2V-1.3B'
PROMPT = ('Please remove the subtitle text from the video while preserving the character appearance, '
          'background composition, and color style. Do not add any new elements.')


def report(**fields):
    print('CLEAR_STATUS ' + json.dumps(fields), flush=True)


def configure_model_residency(pipe, enabled, device_memory_bytes):
    """Retain generation models only for explicit opt-in on GPUs of at least 20 GiB."""
    if not enabled or device_memory_bytes < 20*1024**3:
        return False
    load_requested_models = pipe.load_models_to_device

    def keep_generation_models(names=None):
        return load_requested_models(list(dict.fromkeys(list(names or []) + ['dit', 'vae'])))

    pipe.load_models_to_device = keep_generation_models
    return True


def load_model(source, lora_scale=1.0, resident_models=False, cached_prompt=None):
    import torch

    sys.path.insert(0, str(source))
    from inference import Wan21InferencePipeline
    from diffsynth.models import model_manager
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用。')
    encoder_path = WAN / 'models_t5_umt5-xxl-enc-bf16.pth'
    stored_embedding = None
    cache_file = None
    if cached_prompt is not None:
        from prompt_cache import embedding_file, read_embedding
        cache_file = embedding_file(WORKSPACE / '.deploy' / 'prompt-cache', encoder_path,
                                    WAN / 'google/umt5-xxl', cached_prompt,
                                    (torch.cuda.get_device_name(0), torch.version.cuda))
        stored_embedding = read_embedding(cache_file)
    model_files = [WAN / 'fun-control.safetensors', WAN / 'Wan2.1_VAE.pth']
    if stored_embedding is None:
        model_files.append(encoder_path)
    original_loader = model_manager.load_state_dict

    def load_weights(path):
        if Path(path) == encoder_path:
            return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        return original_loader(path)

    # Map the 11 GB text encoder so Windows can page its weights directly from disk.
    model_manager.load_state_dict = load_weights
    try:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device='cuda:0',
            model_configs=[
                ModelConfig(path=str(path), skip_download=True, offload_device='cpu')
                for path in model_files
            ],
            tokenizer_config=ModelConfig(path=str(WAN / 'google/umt5-xxl'), skip_download=True),
        )
    finally:
        model_manager.load_state_dict = original_loader
    if pipe.dit is None or pipe.dit.in_dim < 32 or pipe.vae is None or (pipe.text_encoder is None and stored_embedding is None):
        raise RuntimeError('Wan Control、VAE 或文本编码器加载失败。')
    # Reuse CLEAR's adapter loader and keep the large text encoder offloaded.
    adapter = Wan21InferencePipeline.__new__(Wan21InferencePipeline)
    adapter.pipe = pipe
    adapter.lora_rank = 64
    adapter.lora_scale = lora_scale
    adapter.lora_target_modules = ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']
    checkpoint = MODELS / 'CLEAR-mask-free-subtitle-removal.pt'
    variant = os.environ.get('CLEAR_BENCH_LORA')
    checksum = None
    if variant:
        checkpoint = Path(variant).resolve(strict=True)
        if checkpoint.suffix != '.safetensors':
            raise ValueError('独立试验权重须为 safetensors')
        with checkpoint.open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            checksum = digest.hexdigest()
        if checksum != os.environ.get('CLEAR_BENCH_LORA_SHA256'):
            raise ValueError('CLEAR 试验权重与本次实验的 SHA-256 不一致')
    with span('clear_adapter_load', device='cpu', checkpoint=checkpoint.name,
              checkpoint_sha256=checksum, lora_scale=lora_scale):
        adapter.load_lora(str(checkpoint))
    lora_b = [value for name, value in pipe.dit.named_parameters() if '.lora_B.' in name]
    if not lora_b or not all(torch.count_nonzero(value).item() for value in lora_b):
        raise RuntimeError('CLEAR LoRA 权重没有匹配到模型。')
    pipe.dit.eval()
    pipe.enable_vram_management(num_persistent_param_in_dit=1_700_000_000)
    if resident_models:
        configure_model_residency(pipe, True, torch.cuda.get_device_properties(0).total_memory)
    encoder = pipe.prompter.encode_prompt
    if cache_file is not None:
        def encode_fixed(prompt, positive=True, device='cuda'):
            # Only single-prompt CFG 1 jobs call this path. Their embedding is independent of video and LoRA.
            if prompt != cached_prompt:
                raise ValueError('当前任务的 CLEAR 提示词已更改')
            with span('clear_prompt_embedding', device=str(device), cache_hit=stored_embedding is not None):
                if stored_embedding is not None:
                    return stored_embedding.to(device)
                from prompt_cache import save_embedding
                value = encoder(prompt, positive=positive, device=device)
                save_embedding(cache_file, value)
                return value
        pipe.prompter.encode_prompt = lru_cache(maxsize=2)(encode_fixed)
    else:
        pipe.prompter.encode_prompt = lru_cache(maxsize=2)(encoder)
    return pipe


def repair_batch(frames, roi, predict, context="roi", resolution="auto", blend_mode="rectangle"):
    repair_regions(frames, [{"type": "subtitle", "roi": roi}], predict,
                   context, resolution, blend_mode)


def context_crop(regions, width, height, context, large_caption=False):
    x1, y1, x2, y2 = bounds(regions)
    if context == 'frame':
        return 0, 0, width, height
    margin_x = max(24, (x2-x1)//8)
    margin_y = max(48, (y2-y1)//2)
    if context == 'expanded':
        crop_width = min(width, max(round(width*.8), x2-x1+2*margin_x))
        crop_height = min(height, max(round(height*8/15), y2-y1+2*margin_y))
        # Prefer the image's horizontal center while keeping every selection inside.
        left = min(max((width-crop_width)//2, x2-crop_width), x1)
        left = max(0, min(width-crop_width, left))
        top = max(0, min(height-crop_height, (y1+y2-crop_height)//2))
        return left, top, left+crop_width, top+crop_height
    left, right = max(0, x1-margin_x), min(width, x2+margin_x)
    top, bottom = max(0, y1-margin_y), min(height, y2+margin_y)
    if large_caption:
        crop_height = min(height, max(bottom-top, round((right-left)/2)))
        top = max(0, min(height-crop_height, (y1+y2-crop_height)//2))
        bottom = top+crop_height
    return left, top, right, bottom


def model_geometry(width, height, resolution, preserve_aspect=False, large_caption=False):
    if large_caption:
        scale = min(.5, 1024/max(width, height), math.sqrt((832*480)/(width*height)))
        resized = tuple(max(1, round(v*scale)) for v in (width, height))
        return resized, tuple(v+(-v % 16) for v in resized)
    if resolution == 'native':
        return (width, height), (width+(-width % 16), height+(-height % 16))
    scale = min(1, math.sqrt((832*480)/(width*height)), 1024/max(width, height))
    padded = (max(16, round(width*scale/16)*16), max(16, round(height*scale/16)*16))
    if not preserve_aspect:
        return padded, padded
    scale = min(1, padded[0]/width, padded[1]/height)
    resized = (max(1, round(width*scale)), max(1, round(height*scale)))
    return resized, padded


def repair_regions(frames, regions, predict, context="roi", resolution="auto", blend_mode="rectangle",
                   adaptive_processing=False, preset='custom', subtitle_guidance='preset'):
    """Predict once from original frames, then composite only the selected regions."""
    precise = blend_mode == 'precise'
    large_caption = False
    input_hints = [[] for _ in frames]
    observed = [[] for _ in frames]
    if precise:
        fast_geometry = preset == 'fast' and context == 'roi' and resolution == 'native'
        temporal = subtitle_guidance != 'preset'
        inspect_large = fast_geometry or temporal
        camera = None
        analyses = []
        for region in regions:
            if region['type'] != 'subtitle':
                continue
            x1, y1, x2, y2 = region['roi']
            masks = []
            region_hints = []
            for index, frame in enumerate(frames):
                mask = caption_mask(frame, region['roi'])[y1:y2, x1:x2].copy()
                if inspect_large:
                    crop = frame[y1:y2, x1:x2]
                    large, hint = large_caption_guidance(crop, include_outlined=temporal)
                    if temporal:
                        cards = light_caption_cards(crop)
                        if cards is not None:
                            hint = cards if hint is None else cv2.bitwise_or(hint, cards)
                    large_caption |= large and fast_geometry
                    region_hints.append(hint)
                    if hint is not None:
                        mask |= hint
                        input_hints[index].append((region['roi'], hint))
                masks.append(mask)
            if temporal and any(h is not None for h in region_hints):
                crops = [frame[y1:y2, x1:x2] for frame in frames]
                if subtitle_guidance in ('motion', 'motion_lama') and camera is None:
                    camera = camera_path(frames, regions)
                for i, borrowed in enumerate(visible_background(crops, masks, region_hints, camera, (x1, y1))):
                    if borrowed is not None:
                        observed[i].append((region['roi'], *borrowed))
            analyses.append(dict(region=region, masks=masks, needs_model=any(mask.any() for mask in masks)))
    else:
        analyses = prepare_subtitles(frames, regions) if adaptive_processing else []
    active_regions = [item['region'] for item in analyses if item['needs_model']]
    if adaptive_processing or precise:
        active_regions.extend(region for region in regions if region['type'] == 'watermark')
        if not active_regions:
            return dict(background_regions=len(analyses), model_regions=0)
    else:
        active_regions = regions
    height, width = frames[0].shape[:2]
    left, top, right, bottom = context_crop(active_regions, width, height, context, large_caption)
    crop_width, crop_height = right - left, bottom - top
    resized, padded = model_geometry(crop_width, crop_height, resolution, preserve_aspect=precise,
                                     large_caption=large_caption)
    originals = [frame[top:bottom, left:right].copy() for frame in frames]
    guidance_images, guidance_masks = [], []
    use_lama = subtitle_guidance in ('lama', 'motion_lama')
    for original, hints, known in zip(originals, input_hints, observed):
        conditioned = original.copy() if known else original
        for (x1, y1, x2, y2), colors, valid in known:
            conditioned[y1-top:y2-top, x1-left:x2-left][valid] = colors[valid]
        pixels = cv2.cvtColor(conditioned, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(pixels)
        if image.size != resized:
            image = image.resize(resized, Image.Resampling.LANCZOS)
        pixels = np.array(image)
        hint_mask = None
        if hints:
            hint_mask = np.zeros((crop_height, crop_width), np.uint8)
            for (x1, y1, x2, y2), hint in hints:
                hint_mask[y1-top:y2-top, x1-left:x2-left] |= hint
            for (x1, y1, x2, y2), _, valid in known:
                hint_mask[y1-top:y2-top, x1-left:x2-left][valid] = 0
            hint_mask = cv2.resize(hint_mask, resized, interpolation=cv2.INTER_AREA)
            # Suppress opaque lettering in the conditioning image before the single CLEAR pass.
            if not use_lama:
                pixels = cv2.inpaint(pixels, hint_mask, 3, cv2.INPAINT_TELEA)
        guidance_images.append(pixels)
        guidance_masks.append(hint_mask)
    if use_lama and any(mask is not None for mask in guidance_masks):
        from lama_guidance import fill
        fill(guidance_images, guidance_masks)
    inputs = []
    for pixels in guidance_images:
        pixels = cv2.copyMakeBorder(pixels, 0, padded[1]-resized[1], 0, padded[0]-resized[0],
                                    cv2.BORDER_REPLICATE)
        inputs.append(Image.fromarray(pixels))
    # CLEAR can generate color noise for a one-frame shot. Use 4n+1 frames
    # with at least one four-frame interval, repeating only this shot's last frame.
    padding = 4 if len(inputs) == 1 else (1 - len(inputs)) % 4
    inputs.extend([inputs[-1]] * padding)
    outputs = predict(inputs)
    if len(outputs) != len(inputs):
        raise RuntimeError('模型返回的帧数与输入不一致。')
    repaired_frames = []
    for image in outputs[:len(frames)]:
        repaired_image = image.convert('RGB')
        repaired_image = repaired_image.crop((0, 0, *resized))
        if repaired_image.size != (crop_width, crop_height):
            repaired_image = repaired_image.resize((crop_width, crop_height), Image.Resampling.LANCZOS)
        repaired_frames.append(cv2.cvtColor(np.asarray(repaired_image), cv2.COLOR_RGB2BGR))
    watermark_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    subtitle_masks = [np.zeros((crop_height, crop_width), np.uint8) for _ in frames] if precise else []
    analysis_by_roi = {tuple(item['region']['roi']): item for item in analyses}
    for region in active_regions:
        x1, y1, x2, y2 = region["roi"]
        local_roi = (x1 - left, y1 - top, x2 - left, y2 - top)
        lx1, ly1, lx2, ly2 = local_roi
        if region["type"] == "watermark":
            watermark_mask[ly1:ly2, lx1:lx2] = 255
            continue
        if precise:
            for mask, detected in zip(subtitle_masks, analysis_by_roi[tuple(region['roi'])]['masks']):
                mask[ly1:ly2, lx1:lx2] |= detected
            continue
        if adaptive_processing:
            analysis = analysis_by_roi[tuple(region['roi'])]
            for index, (frame, repaired) in enumerate(zip(frames, repaired_frames)):
                prepared = frame[y1:y2, x1:x2].copy()
                mask = analysis['masks'][index]
                if blend_mode == 'rectangle':
                    mask = np.full_like(mask, 255)
                    mask[analysis['background_rows'][index]] = 0
                composite = composite_subtitle(prepared, repaired[ly1:ly2, lx1:lx2], mask)
                rows = analysis['background_rows'][index]
                composite[rows] = prepared[rows]
                frame[y1:y2, x1:x2] = composite
            continue
        generated = [blend_region(original, repaired, local_roi)
                     for original, repaired in zip(originals, repaired_frames)]
        if blend_mode == "lines":
            original_regions = np.stack([original[ly1:ly2, lx1:lx2] for original in originals])
            generated = blend_subtitle_lines(original_regions, np.stack(generated))
        for frame, composite in zip(frames, generated):
            frame[y1:y2, x1:x2] = composite
    if precise:
        for frame, original, repaired, mask, known in zip(frames, originals, repaired_frames, subtitle_masks, observed):
            selected = mask > 0
            composite = composite_caption(original, repaired, mask)
            frame[top:bottom, left:right][selected] = composite[selected]
            for (x1, y1, x2, y2), colors, valid in known:
                frame[y1:y2, x1:x2][valid] = colors[valid]
    # Match all watermark regions to the original background in one Poisson solve.
    selected = watermark_mask > 0
    if selected.any():
        x, y, selected_width, selected_height = cv2.boundingRect(watermark_mask)
        mask = cv2.copyMakeBorder(watermark_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        center = (x + 1 + selected_width // 2, y + 1 + selected_height // 2)
        for frame, original, repaired in zip(frames, originals, repaired_frames):
            composite = repaired
            if min(selected_width, selected_height) >= 5:
                destination = cv2.copyMakeBorder(original, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
                source = cv2.copyMakeBorder(repaired, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
                composite = cv2.seamlessClone(source, destination, mask.copy(), center, cv2.NORMAL_CLONE)[1:-1, 1:-1]
            # Watermarks take priority while every unselected pixel stays untouched.
            frame[top:bottom, left:right][selected] = composite[selected]
    return dict(background_regions=sum(not item['needs_model'] for item in analyses),
                model_regions=len(active_regions))


def prompt_for_regions(regions):
    kinds = {region["type"] for region in regions}
    if kinds == {"subtitle"}:
        return PROMPT
    target = "watermarks and logos" if kinds == {"watermark"} else "subtitle text, watermarks, and logos"
    return (f"Please remove the {target} from the video while preserving the character appearance, "
            "background composition, and color style. Do not add any new elements.")


def blend_subtitle_lines(original, generated):
    luminance = original.min(axis=3)
    probe = ((luminance >= 175) & (original.max(axis=3) - luminance <= 55)
             & (luminance.astype(np.int16) - generated.min(axis=3) >= 24))
    indices = np.arange(len(probe))
    support = sum(probe[np.clip(indices + offset, 0, len(probe) - 1)].astype(np.uint8)
                  for offset in range(-2, 3))
    output = []
    for source, repaired, glyphs in zip(original, generated, probe & (support >= 3)):
        active = (glyphs.sum(axis=1) >= 12).astype(np.uint8)
        active = cv2.morphologyEx(active[:, None], cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))[:, 0]
        edges = np.flatnonzero(np.diff(np.r_[False, active.astype(bool), False]))
        plane = np.zeros(glyphs.shape, dtype=np.float32)
        for top, bottom in zip(edges[::2], edges[1::2]):
            if bottom - top < 6:
                continue
            columns = np.nonzero(glyphs[top:bottom])[1]
            if not columns.size:
                continue
            left, right = np.percentile(columns, [.25, 99.75]).astype(int)
            plane[max(0, top - 8):bottom + 8, max(0, left - 8):right + 9] = 1
        if not plane.any():
            output.append(repaired)
            continue
        core = cv2.dilate(glyphs.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        plane[core] = 1
        alpha = cv2.GaussianBlur(plane, (0, 0), 3)[..., None]
        alpha[core] = 1
        source, repaired = source.astype(np.float32), repaired.astype(np.float32)
        background = 1 - plane
        correction = cv2.GaussianBlur((source - repaired) * background[..., None], (0, 0), 12)
        correction /= np.maximum(cv2.GaussianBlur(background, (0, 0), 12)[..., None], 0.1)
        candidate = repaired + np.clip(correction, -25, 25)
        output.append(np.clip(np.rint(source * (1 - alpha) + candidate * alpha), 0, 255).astype(np.uint8))
    return output


def blend_region(original, repaired, roi):
    x1, y1, x2, y2 = roi
    if min(x2 - x1, y2 - y1) < 5:
        return repaired[y1:y2, x1:x2]
    # Pad edge selections because seamlessClone ignores the outermost mask pixels.
    destination = cv2.copyMakeBorder(original, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    source = cv2.copyMakeBorder(repaired, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    mask = np.zeros(destination.shape[:2], dtype=np.uint8)
    mask[y1+1:y2+1, x1+1:x2+1] = 255
    center = (x1 + 1 + (x2 - x1) // 2, y1 + 1 + (y2 - y1) // 2)
    blended = cv2.seamlessClone(source, destination, mask, center, cv2.NORMAL_CLONE)
    return blended[y1+1:y2+1, x1+1:x2+1]


def run(folder, source):
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    input_file = folder / 'source.video'
    cap = open_video_capture(str(input_file), cv2.CAP_FFMPEG)
    try:
        ok, first = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError('无法读取视频或识别帧率。')
        height, width = first.shape[:2]
        if [width, height] != config['dimensions']:
            raise ValueError('浏览器与解码器读取的画面尺寸不同。请先导出为标准 MP4。')
        regions = regions_from_config(config)
        options = InferenceOptions.parse(json.dumps(config.get("inference", {})))
        frame_range = options.frame_range(fps, total)
        cuts = []
        if options.segment_mode == "shots":
            with span('shot_scan', device='cpu'):
                cuts = scan_shots(input_file, regions, frame_range)
        loaded_model = None

        def pipe(**arguments):
            nonlocal loaded_model
            if loaded_model is None:
                report(stage='正在加载 CLEAR，修复复杂背景字幕')
                with span('clear_model_load', device='cuda', phase='first'):
                    prompt = arguments['prompt'] if options.share_prompt_embedding and options.cfg_scale == 1 else None
                    loaded_model = load_model(source, options.lora_scale, options.resident_models, cached_prompt=prompt)
            with span('clear_inference', device='cuda', phase='first', frames=arguments['num_frames'],
                      steps=arguments['num_inference_steps']):
                return loaded_model(**arguments)

        report(percent=8, stage='正在准备首段视频')
        encode(cap, first, fps, total, pipe, regions, folder, options, frame_range, cuts,
               lossless_output=config.get('lossless_output') is True)
    finally:
        cap.release()


def scan_shots(input_file, roi, frame_range):
    report(percent=5, stage="正在检测切镜")
    start, end = frame_range
    scan = open_video_capture(str(input_file), cv2.CAP_FFMPEG)
    try:
        if not scan.isOpened():
            raise ValueError("无法重新打开视频进行切镜检测。")
        cuts = find_shot_cuts(islice(decoded_frames(scan), start, end), roi)
    finally:
        scan.release()
    cuts = [start + cut for cut in cuts]
    print("SHOT_CUTS", json.dumps(cuts), flush=True)
    return cuts


def encode(cap, first, fps, total, pipe, regions, folder, options, frame_range, cuts, lossless_output=False):
    if regions and not isinstance(regions[0], dict):
        regions = [{"type": "subtitle", "roi": regions}]
    prompt = prompt_for_regions(regions)
    height, width = first.shape[:2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    suffix = "mkv" if lossless_output else "mp4"
    temporary = folder / f"partial.{suffix}"
    command = [
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', f'{width}x{height}',
        '-framerate', str(fps), '-i', 'pipe:0', '-i', str(folder / 'source.video'),
        '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'fast', '-crf', str(options.export_crf),
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(temporary),
    ]
    if lossless_output:
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', f'{width}x{height}',
                   '-framerate', str(fps), '-i', 'pipe:0', '-map', '0:v:0',
                   '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-an', str(temporary)]
    start_frame, end_frame = frame_range
    selected_total = end_frame - start_frame if end_frame is not None else max(0, total - start_frame)
    done = 0
    segment = 0
    quality = QualityReport('clear', fps)
    frames = chain((first,), decoded_frames(cap))
    with subprocess.Popen(command, stdin=subprocess.PIPE) as writer, span(
            'ffmpeg_encode', interval='process_lifetime', process_pid=writer.pid):
        try:
            for carry, pending, selected in video_batches(frames, frame_range, cuts, options):
                if not selected:
                    writer.stdin.write(pending[0].tobytes())
                    continue
                segment += 1
                batch = carry + pending
                before = [frame.copy() for frame in pending] if options.adaptive_processing else []

                def predict(images):
                    def progress(timesteps):
                        for step, timestep in enumerate(timesteps, 1):
                            fraction = (done + len(pending) * (step - 1) / options.steps) / max(selected_total, done + len(pending))
                            report(percent=min(94, int(8 + 86 * fraction)), stage=f'第 {segment} 段 · 去噪 {step}/{options.steps}')
                            yield timestep
                    return pipe(prompt=prompt, control_video=images,
                                width=images[0].width, height=images[0].height,
                                num_frames=len(images), num_inference_steps=options.steps, cfg_scale=options.cfg_scale,
                                seed=options.seed, tiled=options.vae_tiled, tile_size=(30, 52), tile_stride=(15, 26),
                                progress_bar_cmd=progress)

                methods = repair_regions(batch, regions, predict, options.context, options.resolution,
                                         options.blend_mode, options.adaptive_processing, preset=options.preset,
                                         subtitle_guidance=options.subtitle_guidance)
                if options.adaptive_processing:
                    quality.methods.update(methods)
                    for offset, (original, repaired) in enumerate(zip(before, pending)):
                        quality.observe(start_frame + done + offset, original, repaired, regions)
                for frame in pending:
                    writer.stdin.write(frame.tobytes())
                done += len(pending)
                report(percent=min(94, int(8 + 86 * done / max(selected_total, done))), stage=f'已修复 {done} 帧')
            if not done:
                raise ValueError('The selected time range contains no decoded frames.')
            report(percent=97, stage='正在封装视频与音轨')
            with span('ffmpeg_finalize', process_pid=writer.pid):
                writer.stdin.close()
                if writer.wait():
                    raise RuntimeError('FFmpeg 视频编码失败。')
        finally:
            if writer.poll() is None:
                writer.kill()
    temporary.replace(folder / f'output.{suffix}')
    if options.adaptive_processing:
        quality.save(folder)
        report(quality=quality.summary())


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]), Path(sys.argv[2]))
    except Exception as exc:
        report(error='CUDA 显存不足。' if 'out of memory' in str(exc).lower() else str(exc)[:400])
        raise
