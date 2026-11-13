import math
import subprocess

import cv2
import imageio_ffmpeg


def validate_video_pair(original, processed):
    information = []
    for path in (original, processed):
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            ok, first = capture.read()
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not ok or not math.isfinite(fps) or fps <= 0:
                raise ValueError('无法读取对比视频，请检查原视频文件。')
            height, width = first.shape[:2]
            information.append((width, height, fps, count))
        finally:
            capture.release()
    source, result = information
    if any(abs(source[index] - result[index]) > 1 for index in (0, 1)):
        raise ValueError('所选原视频的画面尺寸与处理结果不一致。')
    if source[3] > 0 and result[3] > 0 and abs(source[3] - result[3]) > 1:
        # MOV headers can count edit-list frames that the decoder never emits.
        decoded_counts = []
        for path in (original, processed):
            capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
            try:
                count = 0
                while capture.grab():
                    count += 1
                decoded_counts.append(count)
            finally:
                capture.release()
        if abs(decoded_counts[0] - decoded_counts[1]) > 1:
            raise ValueError('所选原视频的帧数与处理结果不一致，请选择同一个原视频。')
    return result


def create_comparison(original, processed, destination):
    width, height, fps, _ = validate_video_pair(original, processed)
    temporary = destination.with_name('comparison.partial.mp4')
    timeline = f'setpts=N/({fps:.12g}*TB)'
    filters = (
        f'[0:v]pad={width}:{height}:0:0,setsar=1,settb=AVTB,{timeline}[original];'
        f'[1:v]setsar=1,settb=AVTB,{timeline}[processed];'
        '[original][processed]hstack=inputs=2:shortest=1,'
        'pad=ceil(iw/2)*2:ceil(ih/2)*2[comparison]'
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-i', str(original), '-i', str(processed), '-filter_complex', filters,
        '-map', '[comparison]', '-map', '1:a:0?', '-r', f'{fps:.12g}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '12', '-threads', '8',
        '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', str(temporary),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            raise RuntimeError('对比视频合成失败。' + result.stderr[-500:])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
