# RemoveSubtitles

基于 CLEAR 的 Windows 本地字幕移除工具。网页支持视频预览、矩形框选、处理进度、NVIDIA GPU 监控和带音轨的 MP4 导出。

## 运行

需要 Python 3.12、支持 CUDA 的 NVIDIA 显卡和驱动。在仓库根目录运行以下命令。

```powershell
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r CLEAR-web/requirements.txt
```

将以下模型文件放入对应目录。Wan Control、VAE、文本编码器和 tokenizer 可从 [Wan2.1 Fun Control](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control/files) 获取。下载其中的 `diffusion_pytorch_model.safetensors` 后改名为 `fun-control.safetensors`。CLEAR 权重可从 [CLEAR 模型库](https://huggingface.co/charlesw09/CLEAR-mask-free-video-subtitle-removal) 获取。

```text
CLEAR-models/
├── CLEAR-mask-free-subtitle-removal.pt
└── Wan2.1-T2V-1.3B/
    ├── fun-control.safetensors
    ├── Wan2.1_VAE.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── google/umt5-xxl/  （完整 tokenizer 目录）
```

```powershell
& $python CLEAR-web/verify_models.py
& $python -c "import torch; from pathlib import Path; assert torch.ones(1, device='cuda').item() == 1; Path('CLEAR-web/runtime.ready').touch()"
& $python CLEAR-web/app.py
```

浏览器访问 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。也可以双击 `CLEAR-web/start.cmd` 启动。启动脚本优先使用仓库内的 `.venv`，本机既有部署可继续使用 Q 盘的运行环境。

上游源码和许可证见 [CLEAR](https://github.com/silent-commit/CLEAR) 与 [LICENSE](CLEAR/LICENSE)。
