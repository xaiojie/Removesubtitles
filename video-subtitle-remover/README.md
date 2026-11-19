# Video Subtitle Remover · Web 版

基于 [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) 的改造分支：
在原项目（AI 硬字幕 / 水印擦除）基础上，**增加了一套可以直接在浏览器里使用的 Web 界面**，
并让整个程序能在**没有图形界面的 Linux GPU 服务器**上正常运行。

原项目的算法部分（STTN / LaMa / ProPainter / OpenCV 四套重绘模型、PaddleOCR 字幕检测）**完全保留**，
本分支只改动「怎么用」，不改动「怎么算」。桌面端 GUI（`gui.py`）依然可用。

![Web 界面](https://raw.githubusercontent.com/xaiojie/video-subtitle-remover/bed060aa77ac033c9403edcfa40eb72860990485/design/web-ui.png)

---

## 一、这个分支改了什么

| 改动 | 说明 | 涉及文件 |
|---|---|---|
| **Web 界面** | 浏览器上传 → 框选 → 处理 → 下载，全流程 | `web/` |
| **去 Qt 化运行** | 原 `backend/config.py` 依赖 `qfluentwidgets`(PySide6)，服务器上连 import 都失败；现在按环境自动切换实现，算法代码零改动 | `backend/config.py`、`backend/config_qt.py`、`backend/config_headless.py` |
| **任务队列** | 每个任务独立子进程 + 独立进程组，进度/日志实时回传，随时可取消 | `web/app.py` |
| **GPU 状态面板** | 型号、利用率、显存、温度、功耗实时展示（`nvidia-smi`） | `web/app.py` |
| **可视化框选** | 在画面上直接拖框，支持多个区域，坐标自动换算成原分辨率像素 | `web/static/index.html` |
| **对比播放** | 原片与处理后同帧同步播放，支持「并排」和「滑动分割」两种模式 | `web/static/index.html` |
| **重新处理** | 已完成/失败的任务可一键载回上次的框选与算法参数，改完再跑 | `web/static/index.html` |
| **跨平台** | Web 服务在 Windows / Linux / macOS 均可运行（进程创建、进程树终止、ffmpeg 路径按平台处理） | `web/app.py` |

![对比播放](https://raw.githubusercontent.com/xaiojie/video-subtitle-remover/bed060aa77ac033c9403edcfa40eb72860990485/design/web-compare.png)

---

## 二、快速开始

本目录来源为 [xaiojie/video-subtitle-remover 的 bed060a 提交](https://github.com/xaiojie/video-subtitle-remover/tree/bed060aa77ac033c9403edcfa40eb72860990485)。代码与许可证随主仓库分发。安装脚本自动下载模型，默认覆盖所有 Web 模式。仅需部分模式时，可独立运行：

```bash
# 在本目录运行，模型下载到 backend/models/
python3 deploy/download_models.py --mode sttn-auto
python3 deploy/download_models.py --mode lama --detector mobile
python3 deploy/download_models.py --mode lama --detector mobile --verify
```

`--list` 离线列出所需文件和大小；`--verify` 离线校验已有权重；`--base-url` 可指定与上游 backend/models 同目录结构的镜像。默认用原生 Web 对应的 server 检测器；单独安装 mobile 时在网页选择 Mobile。图片任务使用 LaMa 和所选检测器；ProPainter 的单帧回退也需要 LaMa。LaMa 和 ProPainter 权重按上游分片验证后自动合并。自定义模型目录用 `VSR_MODEL_DIR`，下载脚本和推理进程均读取这个变量。

五段竖屏字幕的测试参数、计时口径及成片链接见[实测报告](docs/subtitle-removal-benchmark.md)。报告的裁带逐句流程使用了独立实验脚本；该版本随仓提供的 LaMa 网页选项执行原生流程。


### 方式一：Web 界面（推荐，服务器 / 远程机器）

```bash
git clone --filter=blob:none --sparse https://github.com/xaiojie/Removesubtitles.git
cd Removesubtitles
git sparse-checkout set video-subtitle-remover
cd video-subtitle-remover

# 一键装环境 + 注册 systemd（默认 8800 端口）
bash deploy/install.sh
```

脚本会建两个独立的 venv（依赖隔离，Web 服务不用等 torch 装完就能起）：

- `.venv-web` —— 只有 3 个包（fastapi / uvicorn / python-multipart）
- `.venv` —— 算法环境，torch 需要按显卡手动装，脚本会打印对应命令

**torch 按显卡选版本：**

| 显卡 | CUDA | 安装命令 |
|---|---|---|
| RTX 50 系（5070/5080/5090） | 12.8 | `pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128` |
| RTX 40 系 | 12.6 | `... --index-url https://download.pytorch.org/whl/cu126` |
| RTX 30/20/10 系 | 11.8 | `... --index-url https://download.pytorch.org/whl/cu118` |

> 系统依赖：`apt install ffmpeg libgl1 libglib2.0-0 libsm6 libxext6`。运行时使用系统 FFmpeg，也可设置 VSR_FFMPEG。

装完访问 `http://<服务器IP>:8800/`。不想用 systemd 就手动起：

```bash
VSR_CONFIG_MODE=headless \
VSR_DATA_DIR=./vsr-data \
VSR_PYTHON=./.venv/bin/python \
./.venv-web/bin/python web/serve.py
```

常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `VSR_PORT` / `VSR_PORTS` | `8800` | `serve.py` 支持同时监听多个端口，如 `8800,8801` |
| `VSR_DATA_DIR` | `<repo>/vsr-data` | 上传文件、输出、任务状态存放处 |
| `VSR_PYTHON` | 当前解释器 | 执行算法的 Python（必须是装了 torch 的那个 venv） |
| `VSR_WORKERS` | `1` | 并发任务数（显存够可以调大） |
| `VSR_MAX_UPLOAD_MB` | `8192` | 单文件上传上限 |
| `VSR_FFMPEG` | 自动探测 | 指定 ffmpeg 路径 |

### 方式二：桌面 GUI（原版功能）

```bash
pip install -r requirements.txt
python gui.py
```

`gui.py` 会强制走 Qt 版配置（`VSR_CONFIG_MODE=qt`），界面与上游一致。

### 方式三：命令行

```bash
python backend/main.py -i /data/input.mp4 -o /data/out.mp4 --inpaint-mode sttn-auto \
    -c 641 687 330 950      # 字幕区域：ymin ymax xmin xmax，可重复指定多个
```

---

## 三、Web 界面怎么用

1. **拖入视频** → 自动读取分辨率、时长，并进入框选模式
2. **在画面上拖框**，把字幕（和水印）框住；可以框多个区域
   - 竖屏视频点「⤓ 底部区域」可一键框住常见字幕带，再微调
   - 「📋 复制框选参数」可导出像素/归一化/命令行/JSON 四种格式
3. **选算法**（见下节），点「开始去字幕」
4. 进度条实时显示帧数与速度，完成后可**在线预览、对比播放、下载**

任务状态存在 `VSR_DATA_DIR/tasks_state.json`，服务重启后历史任务仍在；
重启瞬间正在跑的任务会标记为「服务重启，任务已中断」。

---

## 四、算法怎么选（重要）

| 算法 | 原理 | 速度 | 适用 |
|---|---|---|---|
| **STTN 智能擦除** (`sttn-auto`) | **不做字幕检测**，把你框的区域整块重绘 | 最快 | 大部分真人视频；也能擦掉抖音号/水印 logo |
| **STTN 带检测** (`sttn-det`) | OCR 检测文字 → 只擦文字多边形 | 慢（OCR 是瓶颈） | 只想擦字幕、不动其它文字 |
| **LaMa 单帧** (`lama`) | 大模型单帧修复，逐帧处理 | 慢 | 动画/插画类效果最好，字幕区最无痕 |
| **ProPainter** (`propainter`) | 视频级修复，利用时序信息 | 最慢、最吃显存 | 运动非常剧烈、其它算法糊成一团时 |
| **OpenCV** (`opencv`) | 传统 inpaint | 极快 | 纯色背景应急 |

### 框选规则（两者要求**相反**，选错就会「擦不干净」或「误伤画面」）

- **`sttn-auto`：不检测，整框重绘** → 框必须**贴紧**文字，框大了会把画面内容一起抹掉
- **`lama` / `sttn-det` / `propainter`：先 OCR 再擦** → 框可以宽松，但**必须完整包住整行文字**，
  框太窄会导致 `No subtitles detected` 直接报错

> 典型事故：字幕横跨 x=110~640，只框了 x=247~464，OCR 只能认出被切断的字，于是「怎么都处理不干净」。

### 显存参考

| 算法 | 显存 | 说明 |
|---|---|---|
| STTN / LaMa | 4–8 GB | 12GB 卡足够，4K 建议 ≥16GB |
| ProPainter 720p | `MAX_LOAD_NUM=80` 需 **25GB**，`=50` 需 **19GB** | 12GB 卡请把 `PROPAINTER_MAX_LOAD_NUM` 降到 20–30 |

调低 `STTN_MAX_LOAD_NUM` / `STTN_NEIGHBOR_STRIDE` / `STTN_REFERENCE_LENGTH` 同样能省显存（效果略降）。

---

## 五、50 系显卡（RTX 5070 等）注意事项

50 系是 Blackwell（`sm_120`），**必须用 CUDA 12.8 及以上的推理库**：

1. **torch 用 cu128 轮子**（`torch==2.7.0+cu128`），否则 `torch.cuda.is_available()` 为 False
2. **PaddlePaddle 对 `sm_120` 的支持要实测**：官方 GPU 轮子目前主要是 cu118 / cu126，
   如果 `paddlepaddle-gpu` 在你的卡上跑不起来，字幕检测会退回 CPU——
   - 影响：`lama` / `sttn-det` / `propainter` 三种模式（都要先 OCR）
   - 不受影响：`sttn-auto`（跳过检测，纯 GPU 重绘），这也是推荐的默认模式
3. 12GB 显存跑 720p/1080p 的 STTN 没有压力，ProPainter 需要按上表降 `MAX_LOAD_NUM`

---

## 六、实测性能（供参考）

测试环境：NVIDIA A100-SXM4-80GB，Python 3.10，torch 2.6.0+cu124，PaddleOCR 走 CPU。

| 素材 | 模式 | 耗时 | 备注 |
|---|---|---|---|
| 1280×720 / 60s / 1800 帧 | STTN 智能擦除 | **66 秒** | 约 27 帧/秒，纯 GPU |
| 1280×720 / 60s | LaMa | 581 秒 | 其中 OCR 约 450 秒（CPU 是瓶颈） |
| 1280×720 / 12s | ProPainter | 151 秒 | 外推 60 秒约 19 分钟 |
| 852×480 / 21s | LaMa | 165 秒 | 检测 3.7 帧/秒 |

结论：**瓶颈通常不在显卡，而在字幕检测（OCR）**。能用 `sttn-auto` 就用它。

---

## 七、常见问题

**1. 处理卡在 9x% 不动，进程还在但没输出**

视频文件的容器头声明帧数 > 实际可解码帧数时，后端 `FramePrefetcher` 会死锁
（日志停在 `Failed to read frame N`）。抖音/剪映二次导出的视频常见。先重封装：

```bash
ffmpeg -i 源.mp4 -c:v libx264 -crf 18 -preset veryfast -c:a copy fixed.mp4
```

**2. 报 `No subtitles detected`**

说明用的是带检测的模式，而框选区域里 OCR 没认出文字：把框放宽到能完整包住整行；
或者改用 `sttn-auto`（不检测）。

**3. 字幕边缘有一圈白边/残影**

框小了几个像素。字幕通常有描边和阴影，上下各留 5–10px 余量。

**4. 网页打不开 / 任务一直排队**

- 确认 `VSR_PYTHON` 指向装了 torch 的 venv：`curl localhost:8800/api/health`
- 看日志：`journalctl -u vsr-web -f`
- 云服务器记得在安全组放行端口

**5. 想清掉抖音水印**

用 `sttn-auto`，把水印所在的整个矩形（logo + 抖音号 + 昵称）一起框进去——
logo 是图形，OCR 认不出来，只有整框重绘能清掉。

---

## 八、目录结构

```
backend/
  config.py            # 配置入口：按环境在 Qt 版 / 无界面版之间切换
  config_qt.py         # 上游实现（需要 PySide6）
  config_headless.py   # 纯 Python 实现（服务器用，支持 VSR_* 环境变量覆盖）
  main.py              # 算法主流程（未改动）
web/
  app.py               # FastAPI：上传、队列、进度、下载、GPU 状态
  serve.py             # 多端口启动器
  static/index.html    # 单页前端（无构建步骤，原生 JS）
deploy/
  install.sh           # 参考部署脚本
  vsr-web.service      # systemd 单元模板
requirements-web.txt      # Web 服务依赖（3 个包）
requirements-headless.txt # 服务器算法依赖（不含 GUI 包）
```

---

## 九、致谢与许可

- 算法与原始实现：[YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)（Apache-2.0）
- 模型：STTN、LaMa、ProPainter、PaddleOCR
- 本分支的改动同样遵循 Apache-2.0，详见 [LICENSE](LICENSE)
