# 三档视频处理 API

普通网页和 PHP API 共用档位、SQLite 队列和后台处理程序。PHP 接收视频、校验选区并保存原片和任务参数。后台按档位调用本仓库的推理程序。每个请求提交一个视频，可以同时指定多个选区。批量处理时逐个提交，保存每个视频返回的任务 ID。

| mode | 用途 | 处理方式 |
| --- | --- | --- |
| Lite · 快速（`lite`） | 常规字幕、批量预览 | PP-OCRv5 Mobile 在字幕框内每 6 帧检测一次，框发生明显变化时补检中间帧。LaMa 只回填检测框。短暂闪字可能漏检。 |
| Pro · 标准（`pro`，默认） | 常规交付、多处字幕和水印 | 字幕走 CLEAR 的 balanced 预设，5 步、原生分辨率、较大上下文、33 帧分块、4 帧重叠。水印走 ProPainter。混合任务先处理字幕，再处理水印。 |
| Max · 增加（`max`） | 粗描边、底板或字幕切换处容易残留的素材 | 完成标准处理后，逐帧对照原片检查选区。可疑片段前后各扩 2 帧，按镜头切开，字幕用 CLEAR 12 步局部精修，水印用 ProPainter 局部精修。无命中时直接交付首遍成片。 |

Max 根据字形、底板和局部对比度判断残留，可能漏掉低对比度水印，也可能把背景纹理当成残字。历史参数与样片见[三档实测依据](../docs/quality-modes.md)。

## 在 Linux 启动

从仓库根目录运行下面的命令即可部署原网页、三档 PHP 接口和后台队列。需要 Linux、可用的 NVIDIA 驱动及磁盘空间。脚本安装 PHP、系统 FFmpeg 和两个独立的推理环境，下载并校验模型，启动后通过接口核对三个档位。

```bash
bash scripts/deploy.sh all --components both --with-api --network auto --yes
bash scripts/deploy.sh api-status
```

默认接口地址为 `http://127.0.0.1:8810`。首次启动会生成令牌并写入仅部署用户可读写的 `.deploy/api/api.token`，后续启动和源码升级复用该令牌。以下命令可查询已部署的档位。

```bash
export SUBTITLE_API_TOKEN="$(cat .deploy/api/api.token)"
curl -fsS -H "Authorization: Bearer ${SUBTITLE_API_TOKEN}" http://127.0.0.1:8810/api/modes
```

只管理 PHP 服务可使用 `api-start`、`api-status`、`api-stop`。后台进程支持断开 SSH 后继续运行。机器重启后执行 `api-start`。接口和队列日志分别在 `.deploy/api/http/service.log`、`.deploy/api/worker/service.log`。网页提交的三档任务和直接提交的 API 任务都进入该队列，每个视频有独立任务 ID。LTX 测试、MuseTalk 和显式自定义模型调用沿用网页服务的处理路径。[调度配置与资源记录](../docs/queue-scheduling.md)

一键启动使用 PHP CLI 的 HTTP 服务供测试。生产环境可用 HTTPS 反向代理和 PHP-FPM，对外站点根目录指向 `php/public`，将 `/api/*` 路由到 `index.php`。自行管理时，接口和 `php php/bin/worker.php` 应使用同一任务目录与读写权限。分步安装、离线模型、服务升级及端口设置见 [Linux 部署教程](../docs/linux-deployment.md#php-api)。

| 环境变量 | 默认值或用途 |
| --- | --- |
| `SUBTITLE_DATA_DIR` | 默认为 `php/var/`。网页、PHP 接口和队列须共享该目录及读写权限，保存 SQLite、原片、阶段文件和结果，预留磁盘空间。 |
| `SUBTITLE_API_URL` | 网页后端访问 PHP 服务的地址。部署脚本管理时自动从 `.deploy/api/settings.json` 读取监听地址、端口和数据目录。手动部署若使用 8801，可设置为 `http://127.0.0.1:8801`。 |
| `SUBTITLE_API_TOKEN_FILE` | 网页后端读取令牌的路径，默认为 `.deploy/api/api.token`。浏览器通过同源网页接口访问。 |
| `SUBTITLE_API_TOKEN` | 接口密钥。一键部署自动配置，手动启动可自设。设置后所有接口都要求 `Authorization: Bearer <token>`。 |
| `CLEAR_PYTHON`、`CLEAR_WORKSPACE` | Pro 和 Max 使用的绝对解释器路径、模型工作目录。默认分别为仓库内的 `.venv/bin/python` 和仓库根目录。 |
| `PROPAINTER_ROOT`、`PROPAINTER_PYTHON` | ProPainter 源码和权重目录、解释器。默认使用仓库内的 `ProPainter/` 和 Pro 使用的解释器。 |
| `VSR_PYTHON`、`VSR_MODEL_DIR` | 默认使用 `video-subtitle-remover/.venv/bin/python` 和 `video-subtitle-remover/backend/models/`。 |
| `LITE_DEVICE` | 默认为 `cuda`。在无 NVIDIA 显卡的测试机上可设为 `cpu` 运行 Lite。 |

## 接口

| 方法与路径 | 用途 | 成功返回 |
| --- | --- | --- |
| `GET /api/modes` | 列出三档及各自支持的选区类型 | 200，含 `default`、`modes`、`stages`、`parameters`、`catalog_version` 和 `catalog_sha256` |
| `POST /api/jobs` | 上传视频并提交异步任务 | 202，含 `id`、`mode`、`dimensions`、`status_url` 和 `resolved_plan` |
| `GET /api/jobs/{id}` | 查询状态 | 200，含 `state`、`percent`（0–100）、`stage`、`error` 和 `output_url` |
| `GET /api/jobs/{id}/output` | 下载处理后的 MP4，保留音轨 | 200，文件名为「原视频名_处理后.mp4」 |
| `POST /api/jobs/{id}/retry` | 将失败任务重新排队，复用已上传的视频 | 202，沿用同一个 ID |

提交使用 `multipart/form-data`。文件字段为 `video`，档位字段为 `mode`，可选 `lite`、`pro`、`max`，省略时使用 `pro`。`regions` 是一个 JSON 数组。每个元素的 `type` 取 `subtitle` 或 `watermark`，`roi` 为显示画面的原像素坐标 `[左, 上, 右, 下]`，左上角为原点，右、下边界不包含在框内。浏览器预览若缩放过，需要换算回视频显示尺寸。Lite 只接受字幕选区。以下例子使用一段 1280×720 视频。

```bash
BASE=http://127.0.0.1:8810
AUTH=(-H "Authorization: Bearer ${SUBTITLE_API_TOKEN}")
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

网页后端转交 PHP 时自动读取服务端 Bearer 令牌。通过任一入口提交后，`GET /api/jobs/{id}` 均可读取 `submitted_via`、`resolved_plan` 和进度。任务在创建时冻结档位版本、配置文件 SHA-256、按选区筛选的执行阶段及具体参数。重试沿用原始快照。推理启动后，`execution` 提供本次使用的参数和入口程序校验值。队列支持时，附带 `run_id`、`queue_wait_seconds` 和 `elapsed_seconds`。旧任务的冻结配置可能为空。

提交时可传 `catalog_sha256` 校验选择档位时看到的版本，版本过期返回 409。网页上传时也会传 `dimensions`，与 PHP 从视频读取的显示尺寸不符会返回 422。浏览器可在网页服务的 `/api/jobs/{id}/output?inline=1` 播放，在同一网址去掉查询参数可下载成片。网页服务的 `POST /api/jobs/{id}/comparison` 生成左右对比，同一路径的 GET 返回生成状态和下载地址。

状态依次为 `queued`、`running` 和 `done`，处理出错时为 `failed`，原因见 `error`。成片就绪前下载、重试非失败任务返回 409，找不到任务返回 404，缺少文件返回 400，上传超限返回 413，无效视频或选区、档位不匹配返回 422。错误响应含 `error` 字段。每个任务的 `worker.log` 记录推理日志。Max 档的 `review.json` 记录逐个选区的可疑帧区间，帧从 0 开始，结束帧不包含在区间内。
