# 三档视频处理 API

PHP 接收视频、校验选区、管理队列并返回成片。后台按档位调用本仓库的推理程序。每个请求提交一个视频，可以同时指定多个选区。批量处理时逐个提交，保存每个视频返回的任务 ID。

| mode | 用途 | 处理方式 |
| --- | --- | --- |
| Lite（`lite`，快速） | 常规字幕、批量预览 | PP-OCRv5 Mobile 在字幕框内每 6 帧检测一次，框发生明显变化时补检中间帧。LaMa 只回填检测框。短暂闪字可能漏检。 |
| Pro（`pro`，标准，默认） | 常规交付、多处字幕和水印 | 字幕走 CLEAR 的 balanced 预设，5 步、原生分辨率、较大上下文、33 帧分块、4 帧重叠。水印走 ProPainter。混合任务先处理字幕，再处理水印。 |
| Max（`max`，精细） | 粗描边、底板或字幕切换处容易残留的素材 | 完成标准处理后，逐帧对照原片检查选区。可疑片段前后各扩 2 帧，按镜头切开，字幕用 CLEAR 12 步局部精修，水印用 ProPainter 局部精修。无命中时直接交付首遍成片。 |

Max 根据字形、底板和局部对比度判断残留，可能漏掉低对比度水印，也可能把背景纹理当成残字。历史参数与样片见[三档实测依据](../docs/quality-modes.md)。

## 在 Linux 启动

从仓库根目录操作。需要 PHP 8.1 及以上、PDO SQLite、可用的 `proc_open`、FFmpeg（带 libx264）、ffprobe，以及所选档位对应的推理环境。例如在 Ubuntu 24.04 安装接口依赖：

```bash
sudo apt-get install php-cli php-sqlite3 ffmpeg
php -r 'var_dump(PHP_VERSION, in_array("sqlite", PDO::getAvailableDrivers()), function_exists("proc_open"));'
```

已有模型环境可直接复用。新机器按需安装，权重通过脚本下载。

```bash
# Pro 和 Max 用的 CLEAR 和 ProPainter。已有部署可以跳过。
bash scripts/deploy.sh all --dir "$PWD" --components both --preset balanced --yes

# Lite 用的 LaMa 和 Mobile OCR
VSR_MODEL_MODE=lama VSR_DETECTOR=mobile VSR_INSTALL_SERVICE=0 bash scripts/deploy.sh vsr-install
# 按上述安装器提示为 video-subtitle-remover/.venv 安装与显卡驱动匹配的 PyTorch
bash scripts/deploy.sh vsr-models --mode lama --detector mobile --verify
```

开两个终端，进入同一个仓库根目录。两个进程使用同一用户或同一可读写的任务目录。

```bash
# 终端一，启动队列。持续运行，每次只占用一个任务的 GPU 资源
php php/bin/worker.php

# 终端二，启动本机测试用 HTTP 接口。按素材大小调整上传限额
php -d upload_max_filesize=2G -d post_max_size=2100M \
  -S 127.0.0.1:8810 -t php/public php/public/index.php
```

生产环境可用 PHP-FPM，将网站根目录指向 `php/public`，让 `/api/*` 请求进入 `index.php`，放行 Authorization 请求头并设置反向代理上传限额。向公网提供服务时，为 HTTP 进程配置 `SUBTITLE_API_TOKEN` 并使用 HTTPS。通过 systemd 等进程管理工具常驻运行队列。同一任务目录的 PHP 队列串行处理，原有两个 Web 服务各自调度任务，共用一张 GPU 时应统一安排。更新 PHP 代码前应等待当前队列清空，升级后重启 PHP worker。

| 环境变量 | 默认值或用途 |
| --- | --- |
| `SUBTITLE_DATA_DIR` | 默认为 `php/var/`。接口和队列须使用相同绝对路径，保存 SQLite、原片、阶段文件和结果，预留磁盘空间。 |
| `SUBTITLE_API_TOKEN` | 接口密钥，可选。设置后所有接口都要求 `Authorization: Bearer <token>`。 |
| `CLEAR_PYTHON`、`CLEAR_WORKSPACE` | Pro 和 Max 使用的绝对解释器路径、模型工作目录。默认分别为仓库内的 `.venv/bin/python` 和仓库根目录。 |
| `PROPAINTER_ROOT`、`PROPAINTER_PYTHON` | ProPainter 源码和权重目录、解释器。默认使用仓库内的 `ProPainter/` 和Pro 使用的解释器。 |
| `VSR_PYTHON`、`VSR_MODEL_DIR` | 默认使用 `video-subtitle-remover/.venv/bin/python` 和 `video-subtitle-remover/backend/models/`。 |
| `LITE_DEVICE` | 默认为 `cuda`。在无 NVIDIA 显卡的测试机上可设为 `cpu` 运行 Lite。 |

## 接口

| 方法与路径 | 用途 | 成功返回 |
| --- | --- | --- |
| `GET /api/modes` | 列出三档及各自支持的选区类型 | 200，含 `default` 和 `modes` |
| `POST /api/jobs` | 上传视频并提交异步任务 | 202，含 `id`、`mode`、`dimensions` 和 `status_url` |
| `GET /api/jobs/{id}` | 查询状态 | 200，含 `state`、`percent`（0–100）、`stage`、`error` 和 `output_url` |
| `GET /api/jobs/{id}/output` | 下载处理后的 MP4，保留音轨 | 200，文件名为「原视频名_处理后.mp4」 |
| `POST /api/jobs/{id}/retry` | 将失败任务重新排队，复用已上传的视频 | 202，沿用同一个 ID |

提交使用 `multipart/form-data`。文件字段为 `video`，档位字段为 `mode`，可选 `lite`、`pro`、`max`，省略时使用 `pro`。`regions` 是一个 JSON 数组。每个元素的 `type` 取 `subtitle` 或 `watermark`，`roi` 为显示画面的原像素坐标 `[左, 上, 右, 下]`，左上角为原点，右、下边界不包含在框内。浏览器预览若缩放过，需要换算回视频显示尺寸。Lite 只接受字幕选区。以下例子使用一段 1280×720 视频。

```bash
BASE=http://127.0.0.1:8810
AUTH=()
# 已配置 token 时使用下行，并让客户端的 SUBTITLE_API_TOKEN 与服务端相同
# AUTH=(-H "Authorization: Bearer ${SUBTITLE_API_TOKEN}")
curl -sS "${AUTH[@]}" "$BASE/api/modes"

curl -sS "${AUTH[@]}" "$BASE/api/jobs" \
  -F 'video=@./片段.mp4' -F 'mode=pro' \
  -F 'regions=[{"type":"subtitle","roi":[261,622,970,718]},{"type":"watermark","roi":[1015,531,1275,720]}]'

# 把返回的 id 填入变量
ID='替换为返回的任务ID'
curl -sS "${AUTH[@]}" "$BASE/api/jobs/$ID"
# state 为 done 时下载
curl -fS "${AUTH[@]}" "$BASE/api/jobs/$ID/output" -o '片段_处理后.mp4'
# state 为 failed 时可重试
curl -sS -X POST "${AUTH[@]}" "$BASE/api/jobs/$ID/retry"
```

状态依次为 `queued`、`running` 和 `done`，处理出错时为 `failed`，原因见 `error`。成片就绪前下载、重试非失败任务返回 409，找不到任务返回 404，缺少文件返回 400，上传超限返回 413，无效视频或选区、档位不匹配返回 422。错误响应含 `error` 字段。每个任务的 `worker.log` 记录推理日志。Max 档的 `review.json` 记录逐个选区的可疑帧区间，帧从 0 开始，结束帧不包含在区间内。
