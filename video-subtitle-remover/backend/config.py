# -*- coding: utf-8 -*-
"""配置模块入口（Qt / 无界面 运行时切换）

上游实现依赖 qfluentwidgets（PySide6），在没有图形界面的机器上（Docker 容器、
远程 GPU 服务器、只跑 Web 服务的环境）连 import 都会失败，导致 CLI 与 Web 服务
完全跑不起来。本文件按运行环境选择实现，二者对外暴露的名字完全一致
（`config` / `tr` / `BASE_DIR` / `VERSION` / `qconfig` ...），因此：

  * `backend/main.py`、`backend/tools/*` 等算法代码 **无需任何修改**
  * `gui.py` 在桌面环境仍走 Qt 实现，功能不受影响

选择规则（按优先级）：
  1. 环境变量 `VSR_CONFIG_MODE=qt`       → 强制 Qt 版（config_qt.py）
  2. 环境变量 `VSR_CONFIG_MODE=headless` → 强制无界面版（config_headless.py）
  3. 未设置时自动判断：能 import PySide6 且存在图形环境 → Qt 版，否则无界面版

说明：Dockerfile / Web 部署脚本都会显式设置 `VSR_CONFIG_MODE=headless`，
避免“装了 Qt 但没有显示器”时误走 Qt 分支。
"""
import os
import sys

QT_MODE = "qt"
HEADLESS_MODE = "headless"


def _has_display():
    """粗判当前是否有可用的图形环境（决定自动模式下是否用 Qt 版）。"""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _qt_importable():
    try:
        import PySide6  # noqa: F401
        import qfluentwidgets  # noqa: F401
        return True
    except Exception:
        return False


def _choose_mode():
    mode = os.environ.get("VSR_CONFIG_MODE", "").strip().lower()
    if mode in (QT_MODE, HEADLESS_MODE):
        return mode
    if _has_display() and _qt_importable():
        return QT_MODE
    return HEADLESS_MODE


_MODE = _choose_mode()

if _MODE == QT_MODE:
    try:
        from backend.config_qt import *  # noqa: F401,F403
    except Exception:  # Qt 装了但加载失败（缺插件/缺字体等）时退回无界面实现
        _MODE = HEADLESS_MODE

if _MODE == HEADLESS_MODE:
    from backend.config_headless import *  # noqa: F401,F403

# 供调用方判断当前用的是哪套实现（例如日志、调试）
CONFIG_MODE = _MODE
