# RemoveSubtitles

基于 CLEAR 和 ProPainter 的视频字幕与水印移除工具，可选安装 LTX 测试模型。网页支持视频预览、多处字幕与水印框选、逐个设置选区的批量队列、分阶段进度、剩余时间估算、NVIDIA GPU 监控，以及带音轨的 MP4 和左右对比视频导出。

## Linux 服务器一键部署

运行 `bash scripts/deploy.sh` 打开分步菜单，或用下面的命令完整安装所选组件。脚本支持国内及官方网络配置、代理、自定义源、断点续传、离线部署和后台服务管理。[查看 Linux 部署教程](docs/linux-deployment.md)

```bash
bash scripts/deploy.sh all --components both --preset balanced --network cn --yes
```

首次安装默认使用“较大上下文”。`--preset fast` 选择快速处理，`--preset quality` 选择强化修复。三档共用 CLEAR 权重。LTX 通过 `--with-ltx` 选装，独立环境和所选模型安装在项目的 `LTX2-ICEdit-Insight/` 目录。

```bash
# 安装 CLEAR、ProPainter 和 LTX 字幕模型
bash scripts/deploy.sh all --with-ltx --ltx-tasks subtitle --network cn --yes

# 已有安装升级源码，等待已接收任务完成
bash scripts/deploy.sh update --wait-idle 1800
```

## Windows 手动运行

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

## 多区域与混合处理

网页根据选区类型自动安排模型。字幕使用 CLEAR，水印使用 ProPainter，两种选区同时存在时先执行 CLEAR，再执行 ProPainter。混合任务的中间视频采用 FFV1 无损编码，水印遮罩参考原视频生成，最终封装原视频音轨。

`POST /api/jobs` 可提交 `engine=auto` 和 JSON 字符串形式的 `regions`，每个区域包含 `type`（`subtitle` 或 `watermark`）及 `roi`。例如下面的字幕和水印选区。

```json
[
  {"type": "subtitle", "roi": [350, 638, 930, 692]},
  {"type": "watermark", "roi": [4, 4, 280, 174]}
]
```

自动模式下，`inference` 接收共用的 `start_seconds`、`duration_seconds`、`export_crf`，以及分别设置模型参数的 `clear` 和 `propainter` 对象。提交 `{}` 使用默认值。ProPainter 默认使用自动精细遮罩、遮罩扩展 8、相邻帧数 10、参考间隔 5、分段长度 40、RAFT 20 次和 FP16。

水印处理需要独立安装 [ProPainter](https://github.com/sczhou/ProPainter) 及其模型权重。通过 `PROPAINTER_ROOT` 指定源码目录，通过 `PROPAINTER_PYTHON` 指定已安装依赖的 Python 解释器。服务会检查权重文件和该目录下的 `runtime.ready` 就绪标记，模型准备情况可通过 `/api/system` 查看。

处理状态包含当前阶段、阶段进度和总体进度。失败任务通过 `POST /api/jobs/<id>/retry` 继续执行，已成功完成且配置一致的阶段会被复用。

处理完成后，通过 `POST /api/jobs/<id>/comparison` 生成左侧原片、右侧处理结果的对比视频，通过同一路径的 GET 请求查询状态，再从 `/api/jobs/<id>/comparison/output` 下载。处理结果与对比文件使用原视频文件名，并分别添加 `_处理后` 和 `_处理前后对比` 后缀。

## API

`POST /api/jobs` 接收 multipart 表单。坐标按原视频像素填写，右边界和下边界不包含在选区内。

```powershell
curl.exe -fS http://127.0.0.1:8765/api/jobs -F "video=@input.mp4" -F "roi=[11,1046,720,1222]" -F "dimensions=[720,1280]"
```

可附加名为 `inference` 的 JSON 表单字段。支持 `steps`、`cfg_scale`、`lora_scale`、`seed`、`chunk_frames`、`overlap_frames` 和 `context`。分段长度使用 5–33 范围内的 4n+1 帧，重叠帧数必须小于分段长度。参数有误时返回 HTTP 400，GPU 忙时返回 HTTP 409。

新任务采用部署配置指定的预设，`CLEAR_PRESET` 默认值为 `balanced`。网页可直接切换三档，手动修改高级参数后显示“自定义”，批量队列分别保存每个视频的方案与选区。

| 预设 | 名称 | 步数 | LoRA | 输入画面 |
|---|---|---|---|---|
| `fast` | 快速 | 5 | 1.2 | 字幕选区与少量周边，原像素推理 |
| `balanced` | 较大上下文 | 5 | 1.0 | 按画面比例扩展周边，原像素推理 |
| `quality` | 强化修复 | 12 | 1.0 | 全画面等比自动缩放，局部重建可能模糊 |

三档均使用 CFG 1.0、种子 20261113、每段 33 帧、重叠 4 帧、CRF 10 和关闭 VAE 分块。`balanced` 在 1280×720 底部字幕样例中使用 1024×384 周边画面，其他画面比例、靠边选区和多选区会调整裁剪范围以覆盖所有字幕。`quality` 的同尺寸样例输入为 848×480，其中内容保持比例，剩余边缘填充。

自动路由 API 可提交以下参数，显式填写的字段优先生效。

```json
{"clear": {"preset": "quality", "steps": 12}}
```

三档采用 `blend_mode=precise`，根据原图生成字形、描边或字幕底板遮罩，再回填并融合颜色。`context` 支持 `roi`、`expanded` 和 `frame`。自定义方案可设 `preset=custom`，使用 `lines` 文字行合成或 `rectangle` 矩形合成。已有任务重试继续使用保存的配置和已成功阶段。

`resident_models=true` 可让足够显存的设备保留 DiT 和 VAE，减少分段之间的搬运。预设会请求这一优化，实际启用以设备容量检查为准，较小显存设备继续使用运行库的模型管理。

默认的 `segment_mode=shots` 会先检测原片的切镜，再按镜头划分推理窗口。重叠只在同一镜头内使用，短镜头以本镜头末帧填充。设为 `fixed` 可使用固定帧数分段。切镜检测以选框外的画面变化为主。

用 `start_seconds` 选择起点，用 `duration_seconds` 限制模型处理时长。省略时长或设为 `null` 会处理到视频结束。服务导出完整视频和音轨，时间选区之外的帧直接参与编码。

`resolution` 由所选预设确定，支持 `native`（原像素推理）和 `auto`（自动缩放）。原像素模式会在边缘填充到模型所需的尺寸倍数。`export_crf` 默认为 10，可设为 18 缩小文件，或设为 0 对转换后的 YUV 4:2:0 画面做无损 H.264 编码。

提交成功返回 HTTP 202 和任务 ID。调用 `GET /api/jobs/<id>` 查询状态、进度与生效的推理参数。状态为 `done` 时，通过 `GET /api/jobs/<id>/output` 下载 MP4。通过 `GET /api/system` 查询 GPU 利用率、显存及默认推理参数。

本机验证实例使用端口 8766。通过 `CLEAR_PORT` 指定端口，通过 `CLEAR_WORKSPACE` 指向现有模型和 CUDA 运行环境所在目录。

[查看处理结果和对照视频](videos/README.md)
