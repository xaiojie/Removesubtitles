from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent / 'CLEAR-models'
WAN = MODELS / 'Wan2.1-T2V-1.3B'
PROMPT = ('Please remove the subtitle text from the video while preserving the character appearance, '
          'background composition, and color style. Do not add any new elements.')
STEPS = 5
CHUNK = 21
OVERLAP = 4


def report(**fields):
    print('CLEAR_STATUS ' + json.dumps(fields), flush=True)


def load_model(source):
    import torch

    sys.path.insert(0, str(source))
    from inference import Wan21InferencePipeline
    from diffsynth.models import model_manager
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用。')
    encoder_path = WAN / 'models_t5_umt5-xxl-enc-bf16.pth'
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
                for path in (WAN / 'fun-control.safetensors', WAN / 'Wan2.1_VAE.pth', encoder_path)
            ],
            tokenizer_config=ModelConfig(path=str(WAN / 'google/umt5-xxl'), skip_download=True),
        )
    finally:
        model_manager.load_state_dict = original_loader
    if pipe.dit is None or pipe.dit.in_dim < 32 or pipe.vae is None or pipe.text_encoder is None:
        raise RuntimeError('Wan Control、VAE 或文本编码器加载失败。')
    # Reuse CLEAR's adapter loader and keep the large text encoder offloaded.
    adapter = Wan21InferencePipeline.__new__(Wan21InferencePipeline)
    adapter.pipe = pipe
    adapter.lora_rank = 64
    adapter.lora_scale = 1.0
    adapter.lora_target_modules = ['q', 'k', 'v', 'o', 'ffn.0', 'ffn.2']
    adapter.load_lora(str(MODELS / 'CLEAR-mask-free-subtitle-removal.pt'))
    lora_b = [value for name, value in pipe.dit.named_parameters() if '.lora_B.' in name]
    if not lora_b or not all(torch.count_nonzero(value).item() for value in lora_b):
        raise RuntimeError('CLEAR LoRA 权重没有匹配到模型。')
    pipe.dit.eval()
    pipe.enable_vram_management(num_persistent_param_in_dit=1_700_000_000)
    pipe.prompter.encode_prompt = lru_cache(maxsize=2)(pipe.prompter.encode_prompt)
    return pipe


def repair_batch(frames, roi, predict):
    x1, y1, x2, y2 = roi
    height, width = frames[0].shape[:2]
    margin_x = max(24, (x2 - x1) // 8)
    margin_y = max(48, (y2 - y1) // 2)
    left, top = max(0, x1 - margin_x), max(0, y1 - margin_y)
    right, bottom = min(width, x2 + margin_x), min(height, y2 + margin_y)
    crop_width, crop_height = right - left, bottom - top
    scale = min(1, math.sqrt((832 * 480) / (crop_width * crop_height)), 1024 / max(crop_width, crop_height))
    shape = (max(16, round(crop_width * scale / 16) * 16),
             max(16, round(crop_height * scale / 16) * 16))
    inputs = [
        Image.fromarray(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB))
        .resize(shape, Image.Resampling.LANCZOS)
        for frame in frames
    ]
    # Wan requires 4n+1 frames. Replicate the tail and discard only the padding.
    inputs.extend([inputs[-1]] * (-(len(inputs) - 1) % 4))
    outputs = predict(inputs)
    if len(outputs) != len(inputs):
        raise RuntimeError('模型返回的帧数与输入不一致。')
    local_roi = (x1 - left, y1 - top, x2 - left, y2 - top)
    for frame, image in zip(frames, outputs):
        repaired_image = image.convert('RGB').resize((crop_width, crop_height), Image.Resampling.LANCZOS)
        repaired = cv2.cvtColor(np.asarray(repaired_image), cv2.COLOR_RGB2BGR)
        composite = blend_region(frame[top:bottom, left:right], repaired, local_roi)
        frame[y1:y2, x1:x2] = composite


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
    cap = cv2.VideoCapture(str(input_file), cv2.CAP_FFMPEG)
    try:
        ok, first = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError('无法读取视频或识别帧率。')
        height, width = first.shape[:2]
        if [width, height] != config['dimensions']:
            raise ValueError('浏览器与解码器读取的画面尺寸不同。请先导出为标准 MP4。')
        pipe = load_model(source)
        report(percent=8, stage='正在准备首段视频')
        encode(cap, first, fps, total, pipe, config['roi'], folder)
    finally:
        cap.release()


def encode(cap, first, fps, total, pipe, roi, folder):
    height, width = first.shape[:2]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', f'{width}x{height}',
        '-framerate', str(fps), '-i', 'pipe:0', '-i', str(folder / 'source.video'),
        '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(folder / 'partial.mp4'),
    ]
    done = 0
    segment = 0
    carry = []
    pending = [first]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as writer:
        try:
            while pending:
                while len(carry) + len(pending) < CHUNK:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    pending.append(frame)
                segment += 1
                batch = carry + pending
                next_carry = [frame.copy() for frame in batch[-OVERLAP:]]

                def predict(images):
                    def progress(timesteps):
                        for step, timestep in enumerate(timesteps, 1):
                            fraction = (done + len(pending) * (step - 1) / STEPS) / max(total, done + len(pending))
                            report(percent=min(94, int(8 + 86 * fraction)), stage=f'第 {segment} 段 · 去噪 {step}/{STEPS}')
                            yield timestep
                    return pipe(prompt=PROMPT, control_video=images,
                                width=images[0].width, height=images[0].height,
                                num_frames=len(images), num_inference_steps=STEPS, cfg_scale=1.0,
                                seed=42, tiled=True, tile_size=(30, 52), tile_stride=(15, 26),
                                progress_bar_cmd=progress)

                repair_batch(batch, roi, predict)
                for frame in pending:
                    writer.stdin.write(frame.tobytes())
                done += len(pending)
                report(percent=min(94, int(8 + 86 * done / max(total, done))), stage=f'已修复 {done} 帧')
                carry = next_carry
                ok, frame = cap.read()
                pending = [frame] if ok else []
            report(percent=97, stage='正在封装视频与音轨')
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError('FFmpeg 视频编码失败。')
        finally:
            if writer.poll() is None:
                writer.kill()
    (folder / 'partial.mp4').replace(folder / 'output.mp4')


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]), Path(sys.argv[2]))
    except Exception as exc:
        report(error='CUDA 显存不足。' if 'out of memory' in str(exc).lower() else str(exc)[:400])
        raise
