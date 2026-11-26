"""Prepare CLEAR's conditioning images using the same Big-LaMa weights as Lite."""
from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from job_trace import span


@lru_cache(maxsize=1)
def model():
    import torch
    workspace = Path(os.environ.get('CLEAR_WORKSPACE', Path(__file__).resolve().parents[1]))
    root = Path(os.environ.get('VSR_MODEL_DIR', workspace / 'video-subtitle-remover/backend/models'))
    weights = root / 'big-lama/big-lama.pt'
    if not weights.is_file():
        raise ValueError('当前字幕引导需要 Big-LaMa，请用部署脚本下载 Lite 权重')
    with span('guide_lama_load', device='cuda', checkpoint=weights.name):
        return torch.jit.load(str(weights), map_location='cuda').eval()


def fill(rgb, masks):
    import torch
    active = [i for i, mask in enumerate(masks) if mask is not None and mask.any()]
    for offset in range(0, len(active), 4):
        indices = active[offset:offset + 4]
        union = np.maximum.reduce([masks[i] for i in indices])
        x, y, w, h = cv2.boundingRect(union)
        left, top = max(0, x - 52), max(0, y - 52)
        right, bottom = min(union.shape[1], x + w + 52), min(union.shape[0], y + h + 52)
        clips = [im[top:bottom, left:right] for im in (rgb[i] for i in indices)]
        coverage = [masks[i][top:bottom, left:right] for i in indices]
        pad_h, pad_w = -(bottom - top) % 8, -(right - left) % 8
        def padded(array):
            return cv2.copyMakeBorder(array, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        images = torch.from_numpy(np.stack([padded(im) for im in clips])).permute(0, 3, 1, 2).contiguous()
        selection = torch.from_numpy(np.stack([padded(mask) > 0 for mask in coverage]))[:, None]
        with torch.inference_mode(), span('guide_lama', device='cuda', frames=len(indices)):
            repaired = model()(images.to('cuda', dtype=torch.float32).div_(255), selection.to('cuda', dtype=torch.long))
            repaired = repaired.clamp_(0, 1).mul_(255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for im, mask, restored in zip(clips, coverage, repaired):
            selected = mask > 0
            im[selected] = restored[:bottom-top, :right-left][selected]
