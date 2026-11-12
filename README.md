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

## API

`POST /api/jobs` 接收 multipart 表单。坐标按原视频像素填写，右边界和下边界不包含在选区内。

```powershell
curl.exe -fS http://127.0.0.1:8765/api/jobs -F "video=@input.mp4" -F "roi=[11,1046,720,1222]" -F "dimensions=[720,1280]"
```

可附加名为 `inference` 的 JSON 表单字段。支持 `steps`、`cfg_scale`、`lora_scale`、`seed`、`chunk_frames`、`overlap_frames` 和 `context`。分段长度使用 5–33 范围内的 4n+1 帧，重叠帧数必须小于分段长度。参数有误时返回 HTTP 400，GPU 忙时返回 HTTP 409。

默认使用整帧作为模型输入（`context=frame`），并将生成结果合成到框选区域。`context=roi` 使用围绕选区裁出的局部画面。其他默认值为 5 步、CFG 1.0、LoRA 1.0、种子 123、每段 21 帧及重叠 4 帧。

`blend_mode` 默认为 `lines`，按检测到的字幕行合成并匹配底色。可设为 `rectangle` 使用矩形选区合成。

默认的 `segment_mode=shots` 会先检测原片的切镜，再按镜头划分推理窗口。重叠只在同一镜头内使用，短镜头以本镜头末帧填充。设为 `fixed` 可使用固定帧数分段。切镜检测以选框外的画面变化为主。

用 `start_seconds` 选择起点，用 `duration_seconds` 限制模型处理时长。省略时长或设为 `null` 会处理到视频结束。服务导出完整视频和音轨，时间选区之外的帧直接参与编码。

`resolution` 可选 `auto`（自动缩放）或 `native`（原像素推理）。原像素模式会在边缘填充到模型所需的尺寸倍数。`export_crf` 默认为 14，可设为 18 缩小文件，或设为 0 对转换后的 YUV 4:2:0 画面做无损 H.264 编码。

提交成功返回 HTTP 202 和任务 ID。调用 `GET /api/jobs/<id>` 查询状态、进度与生效的推理参数。状态为 `done` 时，通过 `GET /api/jobs/<id>/output` 下载 MP4。通过 `GET /api/system` 查询 GPU 利用率、显存及默认推理参数。

本机验证实例使用端口 8766。通过 `CLEAR_PORT` 指定端口，通过 `CLEAR_WORKSPACE` 指向现有模型和 CUDA 运行环境所在目录。

[查看处理结果和对照视频](videos/README.md)
