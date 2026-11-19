"""配置模块（无图形界面实现）

原版 `backend/config.py` 依赖 qfluentwidgets / PySide6 的 QConfig 读写
`config/config.json`，在无图形界面的环境（Docker、远程 GPU 服务器、Web 服务）里
无法导入。此文件是纯 Python 实现，完整保留配置项名称、默认值与访问方式
（`config.xxx.value` / `config.set(config.xxx, v)`），因此 backend 下的算法代码无需任何修改。

由 `backend/config.py` 在运行时分发使用，通常不需要直接导入本文件。

额外能力：
1. 环境变量覆盖（Web 端为每个任务单独传参，互不干扰）
2. 配置文件路径固定为 仓库根目录/config/config.json（与原版相对路径不同，避免受 cwd 影响）
"""
import configparser
import json
import os
from enum import Enum
from pathlib import Path

from backend.tools.constant import InpaintMode, SubtitleDetectMode

# 项目版本号
VERSION = "1.4.0"
PROJECT_HOME_URL = "https://github.com/YaoFANGUK/video-subtitle-remover"
PROJECT_ISSUES_URL = PROJECT_HOME_URL + "/issues"
PROJECT_RELEASES_URL = PROJECT_HOME_URL + "/releases"
PROJECT_UPDATE_URLS = [
    "https://api.github.com/repos/YaoFANGUK/video-subtitle-remover/releases/latest",
    "https://accelerate.xdow.net/api/repos/YaoFANGUK/video-subtitle-remover/releases/latest",
]

# 硬件加速选项开关
HARDWARD_ACCELERATION_OPTION = True

# 项目的 base 目录 = backend/
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)
REPO_DIR = str(Path(BASE_DIR).parent)
CONFIG_FILE = os.path.join(REPO_DIR, 'config', 'config.json')


class ConfigItem:
    """最小配置项：支持 .value 读写与 .set()，兼容原 qfluentwidgets 用法"""

    __slots__ = ('group', 'key', 'default', '_value', 'enum')

    def __init__(self, group, key, default, enum=None):
        self.group = group
        self.key = key
        self.default = default
        self._value = default
        self.enum = enum

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = self._coerce(v)

    def _coerce(self, v):
        if self.enum is None or isinstance(v, self.enum):
            return v
        if isinstance(v, Enum):
            v = v.name
        if isinstance(v, str):
            if v in self.enum.__members__:
                return self.enum[v]
            for member in self.enum:
                if member.value == v:
                    return member
        raise ValueError(f'Invalid value {v!r} for {self.group}.{self.key}')

    def to_json(self):
        v = self._value
        return v.name if isinstance(v, Enum) else v

    def reset(self):
        self._value = self.default


class Config:
    """原 Config(QConfig) 的无界面替代实现"""

    intefaceTexts = {
        '简体中文': 'ch',
        '繁體中文': 'chinese_cht',
        'English': 'en',
        '한국어': 'ko',
        '日本語': 'japan',
        'Tiếng Việt': 'vi',
        'Español': 'es',
    }

    def __init__(self):
        # 界面/窗口（Web 端不使用，保留以兼容）
        self.interface = ConfigItem("Window", "Interface", "en")
        self.windowX = ConfigItem("Window", "X", None)
        self.windowY = ConfigItem("Window", "Y", None)
        self.windowW = ConfigItem("Window", "Width", 1200)
        self.windowH = ConfigItem("Window", "Height", 1200)

        # 主配置
        self.subtitleSelectionAreas = ConfigItem("Main", "SubtitleSelectionAreas", "0.88,0.99,0.15,0.85")
        self.inpaintMode = ConfigItem("Main", "InpaintMode", InpaintMode.STTN_AUTO, enum=InpaintMode)
        self.subtitleDetectMode = ConfigItem("Main", "SubtitleDetectMode",
                                            SubtitleDetectMode.PP_OCRv5_SERVER, enum=SubtitleDetectMode)
        self.subtitleYXAxisDifferencePixel = ConfigItem("Main", "SubtitleYXAxisDifferencePixel", 10)
        self.subtitleAreaDeviationPixel = ConfigItem("Main", "SubtitleAreaDeviationPixel", 10)
        self.subtitleAreaYAxisDifferencePixel = ConfigItem("Main", "SubtitleAreaYAxisDifferencePixel", 20)
        self.subtitleAreaPixelToleranceYPixel = ConfigItem("Main", "SubtitleAreaPixelToleranceYPixel", 20)
        self.subtitleAreaPixelToleranceXPixel = ConfigItem("Main", "SubtitleAreaPixelToleranceXPixel", 20)
        self.subtitleTimelineBackwardFrameCount = ConfigItem("Main", "SubtitleTimelineBackwardFrameCount", 3)
        self.subtitleTimelineForwardFrameCount = ConfigItem("Main", "subtitleTimelineForwardFrameCount", 3)
        self.hardwareAcceleration = ConfigItem("Main", "HardwareAcceleration", HARDWARD_ACCELERATION_OPTION)
        self.checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", False)
        self.saveDirectory = ConfigItem("Main", "SaveDirectory", "")

        # STTN
        self.sttnNeighborStride = ConfigItem("Sttn", "NeighborStride", 5)
        self.sttnReferenceLength = ConfigItem("Sttn", "ReferenceLength", 10)
        self.sttnMaxLoadNum = ConfigItem("Sttn", "MaxLoadNum", 50)

        # ProPainter
        self.propainterMaxLoadNum = ConfigItem("ProPainter", "MaxLoadNum", 70)

        self._items = [v for v in self.__dict__.values() if isinstance(v, ConfigItem)]

    # ---- 兼容 QConfig 的接口 ----
    def set(self, item, value):
        item.value = value

    def get(self, item):
        return item.value

    def reset(self, item):
        item.reset()

    def save(self, path=None):
        path = path or CONFIG_FILE
        data = {}
        for item in self._items:
            data.setdefault(item.group, {})[item.key] = item.to_json()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def load(self, path=None):
        path = path or CONFIG_FILE
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:  # 配置损坏时使用默认值，不影响启动
            print(f'[config] 读取 {path} 失败, 使用默认配置: {e}')
            return
        for item in self._items:
            if item.group in data and item.key in data[item.group]:
                try:
                    item.value = data[item.group][item.key]
                except Exception as e:
                    print(f'[config] 忽略非法配置 {item.group}.{item.key}={data[item.group][item.key]!r}: {e}')

    def getSttnMaxLoadNum(self):
        return max(self.sttnMaxLoadNum.value,
                   self.sttnNeighborStride.value * self.sttnReferenceLength.value)


class _QConfigShim:
    """兼容原 qconfig.load(CONFIG_FILE, config) 的调用方式"""

    def load(self, file, cfg=None):
        if cfg is not None:
            cfg.load(file)

    def save(self, cfg=None):
        if cfg is not None:
            cfg.save()


qconfig = _QConfigShim()

config = Config()
config.load(CONFIG_FILE)

# ---------------------------------------------------------------------------
# 环境变量覆盖（Web 端每个任务独立进程，通过环境变量传参）
# ---------------------------------------------------------------------------
def _env_int(name):
    v = os.environ.get(name)
    if v is None or v == '':
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _apply_env_overrides():
    mode = os.environ.get('VSR_INPAINT_MODE')
    if mode:
        try:
            key = mode.replace('-', '_').upper()
            config.inpaintMode.value = InpaintMode[key] if key in InpaintMode.__members__ else mode
        except Exception as e:
            print(f'[config] VSR_INPAINT_MODE={mode} 无效: {e}')

    detect = os.environ.get('VSR_DETECT_MODE')
    if detect:
        try:
            key = detect.replace('-', '_').upper()
            config.subtitleDetectMode.value = (SubtitleDetectMode[key]
                                               if key in SubtitleDetectMode.__members__ else detect)
        except Exception as e:
            print(f'[config] VSR_DETECT_MODE={detect} 无效: {e}')

    hw = os.environ.get('VSR_HW_ACCEL')
    if hw is not None and hw != '':
        config.hardwareAcceleration.value = hw not in ('0', 'false', 'False', 'off')

    pairs = [
        ('VSR_YX_DIFF', 'subtitleYXAxisDifferencePixel'),
        ('VSR_AREA_DEVIATION', 'subtitleAreaDeviationPixel'),
        ('VSR_TL_BACKWARD', 'subtitleTimelineBackwardFrameCount'),
        ('VSR_TL_FORWARD', 'subtitleTimelineForwardFrameCount'),
        ('VSR_STTN_NEIGHBOR_STRIDE', 'sttnNeighborStride'),
        ('VSR_STTN_REFERENCE_LENGTH', 'sttnReferenceLength'),
        ('VSR_STTN_MAX_LOAD_NUM', 'sttnMaxLoadNum'),
        ('VSR_PROPAINTER_MAX_LOAD_NUM', 'propainterMaxLoadNum'),
    ]
    for env_name, attr in pairs:
        v = _env_int(env_name)
        if v is not None:
            getattr(config, attr).value = v

    lang = os.environ.get('VSR_LANG')
    if lang:
        config.interface.value = lang


_apply_env_overrides()

# ---------------------------------------------------------------------------
# 界面文案（后端日志用）
# ---------------------------------------------------------------------------
tr = configparser.ConfigParser()
TRANSLATION_FILE = os.path.join(BASE_DIR, 'interface', f"{config.interface.value}.ini")
tr.read(TRANSLATION_FILE, encoding='utf-8')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
