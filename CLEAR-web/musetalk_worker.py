"""Render MuseTalk 1.5 against the selected audio timeline."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave


FPS = 25
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
BATCH_SIZE = 8
INPUT_FORMATS = {
    'video': 'mov,matroska,webm,avi,mpegts,mpeg,flv',
    'audio': 'wav,mp3,mov,flac,ogg,matroska,webm,aac,aiff',
}
FFMPEG = ['ffmpeg', '-v', 'error', '-nostdin', '-y']


@dataclass(frozen=True)
class Timeline:
    video: Path
    audio: Path
    source_frames: int
    samples: int
    width: int
    height: int

    @property
    def frames(self):
        return (self.samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

    @property
    def seconds(self):
        return self.samples / SAMPLE_RATE


def report(**fields):
    print('CLEAR_STATUS ' + json.dumps(fields, ensure_ascii=False), flush=True)


def run_command(command, timeout):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-800:] or f'{command[0]} 执行失败。')
    return result.stdout


def input_args(path, kind):
    return ['-protocol_whitelist', 'file,pipe', '-format_whitelist', INPUT_FORMATS[kind], '-i', str(path)]


def probe(path):
    return json.loads(run_command([
        'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,duration,nb_frames,width,height',
        '-of', 'json', str(path),
    ], 30))['streams']


def prepare_timeline(folder, work, duration_mode):
    voice = work / 'voice.wav'
    run_command(FFMPEG + input_args(folder / 'source.audio', 'audio') + [
        '-map', '0:a:0', '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_s16le', str(voice),
    ], 180)
    with wave.open(str(voice), 'rb') as wav:
        samples = wav.getnframes()
    if samples < SAMPLE_RATE * 0.3:
        raise ValueError('请选择至少 0.3 秒的配音。')

    video = work / 'video.mp4'
    max_frames = (samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    run_command(FFMPEG + input_args(folder / 'source.video', 'video') + [
        '-map', '0:V:0', '-an', '-frames:v', str(max_frames),
        '-vf', f'fps={FPS},pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-threads', '4', '-crf', '17', str(video),
    ], 300)
    picture = next((stream for stream in probe(video) if stream['codec_type'] == 'video'), {})
    count = int(picture.get('nb_frames', 0))
    if not count:
        raise ValueError('视频没有可用画面。')
    if duration_mode == 'preview' and samples > count * SAMPLES_PER_FRAME:
        samples = count * SAMPLES_PER_FRAME
        if samples < SAMPLE_RATE * 0.3:
            raise ValueError('请选择至少 0.3 秒的重叠画面和配音。')
        trimmed = work / 'preview.wav'
        run_command(FFMPEG + ['-i', str(voice), '-af', f'atrim=end_sample={samples}',
                              '-c:a', 'pcm_s16le', str(trimmed)], 180)
        voice = trimmed
    return Timeline(video, voice, count, samples, picture['width'], picture['height'])


def validate_output(path, timeline):
    streams = {stream['codec_type']: stream for stream in probe(path)}
    picture, voice = streams.get('video', {}), streams.get('audio', {})
    if (int(picture.get('nb_frames', 0)) != timeline.frames
            or abs(float(picture.get('duration', 0)) - timeline.frames / FPS) > 0.001):
        raise RuntimeError('成片画面不完整，无法导出。')
    if abs(float(voice.get('duration', 0)) - timeline.seconds) > 1 / FPS:
        raise RuntimeError('成片配音时长不完整，无法导出。')


def landmark_box(frame, preprocessing):
    """Use MuseTalk's v1.5 landmark crop and fall back to its face detector."""
    import numpy as np

    detection = preprocessing.fa.get_detections_for_batch(np.asarray([frame]))[0]
    if detection is None:
        return None
    result = preprocessing.merge_data_samples(preprocessing.inference_topdown(preprocessing.model, frame))
    keypoints = result.pred_instances.keypoints
    x1, y1, x2, y2 = detection
    if len(keypoints):
        landmarks = keypoints[0][23:91].astype(np.int32)
        min_x = int(landmarks[:, 0].min())
        max_x = int(landmarks[:, 0].max())
        max_y = int(landmarks[:, 1].max())
        upper = max(0, int(2 * landmarks[29, 1] - max_y))
        if 0 <= min_x < max_x and upper < max_y:
            x1, y1, x2, y2 = min_x, upper, max_x, max_y
    height, width = frame.shape[:2]
    box = (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2) + 10))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def is_static_source(timeline):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(timeline.video), cv2.CAP_FFMPEG)
    positions = sorted({round(index * (timeline.source_frames - 1) / 8) for index in range(9)})
    reference = None
    try:
        for position in positions:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, position):
                return False
            ok, frame = capture.read()
            if not ok:
                return False
            thumb = cv2.resize(frame, (64, 40), interpolation=cv2.INTER_AREA)
            if reference is None:
                reference = thumb
                continue
            change = cv2.absdiff(reference, thumb)
            if change.mean() > 0.6 or np.percentile(change, 98) > 4:
                return False
        return reference is not None
    finally:
        capture.release()


def quiet_spans(audio_path):
    import numpy as np

    with wave.open(str(audio_path), 'rb') as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32)
    samples = np.pad(samples, (0, -len(samples) % SAMPLES_PER_FRAME))
    frames = samples.reshape(-1, SAMPLES_PER_FRAME)
    rms = np.sqrt((frames * frames).mean(axis=1))
    # Keep only near-silent spans, even when the narration level is low.
    threshold = min(130, 0.03 * np.percentile(rms, 90))
    changes = np.flatnonzero(np.diff(np.r_[False, rms <= threshold, False]))
    return [(int(start), int(end)) for start, end in zip(changes[::2], changes[1::2])
            if end - start >= 14]


def stabilize_pause_prompts(prompts, spans):
    import torch

    ramp = torch.arange(1, 4, device=prompts.device, dtype=prompts.dtype).view(3, 1, 1) / 4
    for start, end in spans:
        reference = prompts[(start + end - 1) // 2].clone()
        # Let the audio features follow the voice around the pause boundaries.
        left = start + 3 if start > 0 else start
        right = end - 3 if end < len(prompts) else end
        if start > 0:
            prompts[left:left + 3].lerp_(reference, ramp)
            left += 3
        if end < len(prompts):
            prompts[right - 3:right].lerp_(reference, ramp.flip(0))
            right -= 3
        prompts[left:right] = reference


def render(timeline, output):
    import cv2
    import torch
    from transformers import WhisperModel
    from musetalk.utils import preprocessing
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.blending import get_image
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import load_all_model

    cv2.setNumThreads(2)
    device = torch.device('cuda:0')
    with torch.inference_mode():
        vae, unet, position = load_all_model(device=device)
        vae.vae = vae.vae.to(device=device, dtype=torch.float16).eval()
        unet.model = unet.model.to(device=device, dtype=torch.float16).eval()
        position = position.half().to(device).eval()
        parser = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        step = torch.tensor([0], device=device)
        audio = AudioProcessor('models/whisper')
        whisper = WhisperModel.from_pretrained('models/whisper').to(device=device, dtype=torch.float16).eval()
        features, _ = audio.get_audio_feature(str(timeline.audio))
        # Round the feature timeline up to a frame without extending the exported audio.
        # The extra sample avoids floating-point floor rounding inside MuseTalk.
        aligned_samples = timeline.frames * SAMPLES_PER_FRAME + 1
        prompts = audio.get_whisper_chunk(features, device, torch.float16, whisper, aligned_samples, fps=FPS)
        if len(prompts) != timeline.frames:
            raise RuntimeError('配音特征与画面帧数不一致。')
        del whisper, features
        if is_static_source(timeline):
            spans = quiet_spans(timeline.audio)
            if spans:
                stabilize_pause_prompts(prompts, spans)

        command = FFMPEG + [
            '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{timeline.width}x{timeline.height}',
            '-framerate', str(FPS), '-i', 'pipe:0', '-i', str(timeline.audio),
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', '18', '-threads', '4', '-c:a', 'aac', '-b:a', '96k',
            '-movflags', '+faststart', str(output),
        ]
        capture = cv2.VideoCapture(str(timeline.video), cv2.CAP_FFMPEG)
        try:
            with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
                try:
                    render_frames(timeline, capture, encoder.stdin, prompts, vae, unet, position,
                                  parser, step, device, preprocessing, get_image)
                    encoder.stdin.close()
                    if encoder.wait(timeout=120):
                        raise RuntimeError('成片编码失败。')
                finally:
                    if encoder.poll() is None:
                        encoder.kill()
        finally:
            capture.release()


def render_frames(timeline, capture, stream, prompts, vae, unet, position, parser, step, device,
                  preprocessing, blend):
    import cv2
    import torch

    # Only cache face geometry and 256px latents for shorter videos that repeat.
    repeat = timeline.source_frames < timeline.frames
    faces = []
    matched = 0
    reported = 12
    for start in range(0, timeline.frames, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, timeline.frames)
        frames, indices, boxes, latents = [], [], [], []
        for index in range(start, stop):
            source_index = index % timeline.source_frames
            if index and source_index == 0:
                capture.release()
                capture.open(str(timeline.video), cv2.CAP_FFMPEG)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('读取视频帧失败。')
            frames.append(frame)
            if index < timeline.source_frames:
                box = landmark_box(frame, preprocessing)
                latent = None
                if box is not None:
                    x1, y1, x2, y2 = box
                    crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    latent = vae.get_latents_for_unet(crop)
                if repeat:
                    faces.append((box, latent.cpu() if latent is not None else None))
            else:
                box, latent = faces[source_index]
            if box is not None:
                indices.append(index)
                boxes.append(box)
                latents.append(latent.to(device=device, dtype=torch.float16))

        matched += len(indices)
        if not matched and stop >= timeline.source_frames:
            raise ValueError('视频里没有检测到可用的人脸，请选择清晰的单人镜头。')
        if indices:
            prediction = unet.model(torch.cat(latents), step, encoder_hidden_states=position(prompts[indices])).sample
            for index, box, crop in zip(indices, boxes, vae.decode_latents(prediction), strict=True):
                x1, y1, x2, y2 = box
                crop = cv2.resize(crop, (x2 - x1, y2 - y1))
                frames[index - start] = blend(frames[index - start], crop, box, mode='jaw', fp=parser)
        # Emit the original picture for unmatched frames to keep every timestamp intact.
        for frame in frames:
            stream.write(frame.tobytes())
        percent = 16 + math.floor(79 * stop / timeline.frames)
        if percent > reported:
            report(stage='正在同步口型', percent=percent)
            reported = percent


def run(folder):
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    output = folder / 'partial.mp4'
    with TemporaryDirectory(prefix='musetalk-', dir=folder) as name:
        report(stage='正在准备视频和配音', percent=5)
        timeline = prepare_timeline(folder, Path(name), config['duration_mode'])
        report(stage='正在加载 MuseTalk 和提取音频特征', percent=12)
        render(timeline, output)
        report(stage='正在核对画面和配音时长', percent=97)
        validate_output(output, timeline)
        output.replace(folder / 'output.mp4')
        report(output_seconds=timeline.seconds)


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]))
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        report(error=str(error)[-900:])
        raise
