import os
from backend.config import config, BASE_DIR
from backend.tools.constant import SubtitleDetectMode

_MODEL_NAME_MAP = {
    SubtitleDetectMode.PP_OCRv5_MOBILE: "PP-OCRv5_mobile_det",
    SubtitleDetectMode.PP_OCRv5_SERVER: "PP-OCRv5_server_det",
}

class ModelConfig:
    def __init__(self):
        root = os.path.abspath(os.environ.get('VSR_MODEL_DIR') or os.path.join(BASE_DIR, 'models'))
        self.LAMA_MODEL_DIR = os.path.join(root, 'big-lama')
        self.STTN_AUTO_MODEL_PATH = os.path.join(root, 'sttn-auto', 'infer_model.pth')
        self.STTN_DET_MODEL_PATH = os.path.join(root, 'sttn-det', 'sttn.pth')
        self.PROPAINTER_MODEL_DIR = os.path.join(root, 'propainter')
        detector = config.subtitleDetectMode.value
        folder = 'ch_det_fast' if detector == SubtitleDetectMode.PP_OCRv5_MOBILE else 'ch_det'
        self.DET_MODEL_DIR = os.path.join(root, 'V5', folder)
        self.DET_MODEL_NAME = _MODEL_NAME_MAP[config.subtitleDetectMode.value]
