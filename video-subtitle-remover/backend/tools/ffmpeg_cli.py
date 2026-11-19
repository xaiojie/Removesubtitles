import os
import shutil


def resolve_ffmpeg():
    name = os.getenv('VSR_FFMPEG') or 'ffmpeg'
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f'找不到可执行的 FFmpeg：{name}。请安装 FFmpeg 或设置 VSR_FFMPEG。')
    return path


class FFmpegCLI:
    @classmethod
    def instance(cls):
        return cls()

    @property
    def ffmpeg_path(self):
        return resolve_ffmpeg()
