# Linux 服务器部署教程

仓库提供 `scripts/deploy.sh` 作为部署入口。直接运行会显示菜单，也可以用命令行分步执行。脚本安装所选模型及依赖，模型下载支持断点续传、备用地址和 SHA-256 校验。

## 1. 服务器准备

推荐 Ubuntu 22.04/24.04、Debian 11/12、Rocky Linux / AlmaLinux / RHEL 8/9、Alibaba Cloud Linux 3 等 glibc 2.28 以上的 Linux。系统包安装支持 `apt-get`、`dnf` 和 `yum`。支持检测 x86_64 和 aarch64，ARM 服务器需要软件源提供对应架构的 CUDA wheel。

推理需要 NVIDIA 驱动。先在服务器执行 `nvidia-smi`，确认驱动正常。脚本会安装 PyTorch 自带的 CUDA 运行库，并执行 CUDA 运算、BF16 和 torchvision 光流相关算子验证。推荐 Ampere 或更新架构的 NVIDIA 显卡。实际兼容性以运行环境验证结果为准。

建议为 CLEAR＋ProPainter 安装准备至少 60 GiB 磁盘和 32 GiB 系统内存，并给视频及临时帧文件留出空间。CLEAR 模型及 tokenizer 下载量约 14.2 GiB，ProPainter 权重约 190 MiB，Python/CUDA 依赖还会占用数 GiB。选装 LTX 字幕或水印用途还会下载约 49.4 GiB 模型，双用途约 49.7 GiB，并创建独立 Python/CUDA 环境。三种模型共存时建议准备 100 GiB 以上可用空间。建议使用 SSD 数据盘作为安装目录。

CLEAR＋ProPainter 建议显存 24 GiB 以上。视频分辨率、选区和分段长度影响实际用量，1080p、4K 视频可能需要更多显存。LTX 使用 BF16/FP8 组件，其显存、主机内存需求随推理尺寸和流式配置变化，部署校验会检查实际 CUDA 能力。三档网页默认选择 Pro · 标准，可按每个视频切换 Lite · 快速或 Max · 增加。

系统 Python 为 3.10～3.12 时直接创建虚拟环境。较旧的系统 Python 可通过 uv 准备 Python 3.11，这一步需要访问 Python 独立发行包的下载源。也可提前安装合适版本，再用 `--python /path/to/python3.11` 指定。

## 2. 获取部署脚本

已有仓库时，在仓库根目录执行。

```bash
bash scripts/deploy.sh
```

新服务器可以先下载入口脚本，再选择安装目录。

```bash
curl -fL https://raw.githubusercontent.com/xaiojie/Removesubtitles/main/scripts/deploy.sh -o deploy.sh
bash deploy.sh --dir /data/Removesubtitles
```

如果服务器访问 GitHub 需要代理，可以先导出自己的代理地址。`--github-prefix` 用于接受“代理前缀/完整 GitHub URL”格式的下载服务。Python 依赖、模型下载和源码下载都支持标准的 `HTTP_PROXY`、`HTTPS_PROXY` 环境变量。

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY="$HTTPS_PROXY"
```

也可以从其他电脑下载仓库压缩包并上传到服务器，解压后执行 `bash scripts/deploy.sh`。

## 3. 一键完整部署

国内网络，部署字幕和水印两个组件，并允许通过服务器公网 IP 访问。

```bash
bash deploy.sh all \
  --dir /data/Removesubtitles \
  --components both \
  --preset balanced \
  --network cn \
  --public-host SERVER_IP \
  --port 8765 \
  --yes
```

把 `SERVER_IP` 换成实际公网 IP。云服务器安全组和系统防火墙需要放行所选端口。成功后访问 `http://SERVER_IP:8765/`。

租机平台提供端口映射时，`--port` 填机器内部监听端口，在平台控制台把公网端口映射到该端口。例如内网 8800 对应公网 10282，程序使用 `--port 8800`，浏览器访问 `http://SERVER_IP:10282/`。

海外或能够直连官方服务的网络可以选择 `--network global`。`--network auto` 会测试 Hugging Face 连通性，选择网络配置。`cn` 默认使用清华 PyPI 源、ModelScope 和 Hugging Face 镜像，`global` 使用官方 PyPI 和模型地址。模型文件还会尝试清单中配置的备用地址。

```bash
bash deploy.sh all --dir /data/Removesubtitles --network global --yes
```

完整流程依次执行系统检查、系统包安装、源码准备、Python/CUDA 环境安装、模型下载、模型与运行环境校验、后台启动。首次执行会把选项保存到安装目录的 `.deploy/config.sh`，后续可继续分步运行。

安装系统包会使用 root 或 sudo。已经准备好系统依赖的普通用户可加 `--skip-system`。建议始终用同一个部署用户管理项目，系统包安装由脚本单独调用 sudo。

<a id="php-api"></a>

### 同时部署 Lite、Pro、Max 的 PHP API

已有完整仓库时，从根目录执行下列命令。只下载了 `deploy.sh` 时把命令里的 `scripts/deploy.sh` 换成 `deploy.sh`，并传入安装目录 `--dir /data/Removesubtitles`。交互菜单的第 13 项也可完整安装。

```bash
bash scripts/deploy.sh all --components both --with-api --api-port 8810 --network auto --yes
bash scripts/deploy.sh api-status
```

| 部署内容 | 脚本行为 |
| --- | --- |
| 系统 | 安装、校验 PHP CLI 8.1+、PDO SQLite、带 libx264 的 FFmpeg 和 ffprobe。Ubuntu 22.04/24.04、Debian 12 等系统可从发行版软件源安装。在默认源缺少 PHP 8.1+ 或 FFmpeg 的系统上，需先配置相应的软件源或安装所需版本，再用 `--skip-system`。 |
| Lite | 在 `video-subtitle-remover/.venv` 内安装 CUDA 版 PyTorch 2.7、PaddleOCR 及算法依赖，仅下载 Mobile OCR 和 LaMa。校验模型文件后，实际调用检测器和 LaMa 做短推理。 |
| Pro / Max | 共用已有的 CLEAR、ProPainter 环境和权重，Max 的局部精修复用同一套模型。完整性校验通过的权重会直接复用。 |
| 服务 | 网页仍由原服务提供。普通三档通过网页后端转交 PHP，直接 API 请求和网页请求进入同一队列。PHP 测试端点默认端口为 8810。PHP 接口省略 `mode` 时使用 `pro`。 |

`--with-api`、`--api-port` 随安装配置保存。默认仅监听本机。测试机需要公网访问时传入 `--public-host SERVER_IP`，同时将平台的外部端口映射到 `--api-port` 指定的内部端口。公网正式服务使用 HTTPS 与 PHP-FPM，配置说明见 [PHP 接口文档](../php/README.md)。

首次启动会把令牌写入安装目录的 `.deploy/api/api.token`，权限为 `0600`。也可以在首次启动前由环境变量 `SUBTITLE_API_TOKEN` 提供。以下命令在服务器本机执行：

```bash
curl -fsS -H "Authorization: Bearer $(cat .deploy/api/api.token)" http://127.0.0.1:8810/api/modes
```

已有项目先按第 13 节更新源码。已有网页追加 API 时，在任务完成后执行 `all --with-api --components both --yes`。如果两个推理环境和权重均已备好，可依次执行 `system`、`verify`、`api-start`，前两步加上 `--with-api --components both`。各阶段使用同一安装目录。只启用自定义模型、LTX 或 MuseTalk 的安装可运行 `api-stop`，再执行 `start --without-api`。普通三档需要同时运行 PHP 接口和队列。

```bash
bash scripts/deploy.sh api-status
bash scripts/deploy.sh api-stop --wait-idle 1800
bash scripts/deploy.sh api-start
bash scripts/deploy.sh update --wait-idle 1800
```

通过部署脚本启用 API 后，`start/stop/restart/status` 同时管理网页和 PHP 服务。停止或更新时，PHP 暂停接收新任务和重试请求（HTTP 503），等待已入队的任务完成。等待超时后恢复接单并保留运行中的进程。更新完成后恢复原先运行的服务。任务默认保存在 `php/var/`，首次启动前可用 `SUBTITLE_DATA_DIR` 指向数据盘的绝对路径。HTTP 和队列日志在 `.deploy/api/{http,worker}/service.log`。这些进程支持关闭 SSH 后运行，服务器重启后执行 `start`。

离线部署时，除 CLEAR 和 ProPainter 外，还要预先准备 Lite 的 `requirements-headless.txt`、对应 Python/平台的 PyTorch 2.7.0 和 torchvision 0.22.0 CUDA wheels 及全部依赖，以及 `video-subtitle-remover/backend/models` 下的 Mobile OCR 与 LaMa 权重。将这些 wheels 放入 `--wheelhouse` 指定的目录。运行 `models --with-api --offline` 只校验本地模型。联网时 Lite 模型支持 `--proxy`、`--github-prefix` 或环境变量 `VSR_MODEL_BASE_URL`，自定义地址的目录结构应与上游 `backend/models` 相同。

## 4. 按步骤安装或恢复

进入安装后的仓库目录，分别执行以下命令。

| 命令 | 用途 |
|---|---|
| `bash scripts/deploy.sh check` | 查看系统、显卡、驱动和磁盘 |
| `bash scripts/deploy.sh system` | 安装 Linux 系统包 |
| `bash scripts/deploy.sh source` | 准备项目及所选 ProPainter 源码 |
| `bash scripts/deploy.sh env` | 创建 Python 环境、安装依赖并验证 CUDA |
| `bash scripts/deploy.sh models` | 下载并校验模型 |
| `bash scripts/deploy.sh verify` | 重新校验模型、tokenizer 和 GPU 运行环境 |
| `bash scripts/deploy.sh update` | 升级源码，保留模型、任务、配置和环境 |
| `bash scripts/deploy.sh start` | 后台启动网页 |
| `bash scripts/deploy.sh status` | 查询脚本管理的服务进程 |
| `bash scripts/deploy.sh stop` | 停止服务及其处理进程 |
| `bash scripts/deploy.sh restart` | 按保存的配置重新启动 |

下载中断后重新执行 `models`。已校验成功的文件会跳过，未完成的 `.part` 文件会续传。服务器忽略 Range 时会重新下载该文件，完成后校验 SHA-256。哈希不匹配的内容会触发重新下载。

停止服务会终止正在处理的视频任务。更新依赖前先停止服务，更新后依次执行 `verify` 和 `start`。

后台启动后可以断开 SSH。日志位于 `.deploy/server.log`、`.deploy/launcher.log` 和 `.deploy/access.log`。

## 5. 只安装需要的模型

字幕使用 CLEAR，水印使用 ProPainter，混合处理依次运行两个模型。两个组件共用 `.venv`，减少重复安装 PyTorch 和 CUDA 库的磁盘开销。

```bash
# 字幕组件
bash scripts/deploy.sh all --components clear --network cn --yes

# 水印组件
bash scripts/deploy.sh all --components propainter --network cn --yes

# 为已有安装补充两个组件
bash scripts/deploy.sh source --components both
bash scripts/deploy.sh env --components both
bash scripts/deploy.sh models --components both
bash scripts/deploy.sh verify --components both
```

ProPainter 默认安装在 `<安装目录>/ProPainter`。已有 ProPainter 环境时，可用 `--propainter-root /data/ProPainter` 指定其源码及权重目录，部署脚本会使用项目共用的 Python 环境运行它。

## 6. 自定义下载源与 CUDA

```bash
bash scripts/deploy.sh all \
  --network cn \
  --pip-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --hf-endpoint https://hf-mirror.com \
  --cuda cu128 \
  --workers 2 \
  --yes
```

PyTorch 固定为 2.8.0，torchvision 固定为 0.23.0。CUDA wheel 支持选择 `cu128` 和 `cu126`。自动选择会考虑驱动展示的 CUDA 版本与 GPU 计算能力，之后通过真实 CUDA 运算校验。

国内 x86_64 的 `cu128` 可从所选 PyPI 镜像获取 PyTorch 包。`cu126` 及其他架构默认使用 PyTorch 官方 wheel 源。自定义 CUDA wheel 镜像可通过 `--torch-index URL` 指定。

```bash
bash scripts/deploy.sh env \
  --cuda cu126 \
  --torch-index https://download.pytorch.org/whl/cu126
```

修改 `--network` 会重新选择该网络配置的默认地址，同一次命令显式传入的 `--pip-index`、`--torch-index` 和 `--hf-endpoint` 优先。代理可通过环境变量或 `--proxy URL` 传入，代理选项仅用于本次运行。

如果准备 Python 3.11 时 GitHub 下载较慢，可使用 uv 支持的 `UV_PYTHON_INSTALL_MIRROR` 环境变量指定 Python 独立发行包镜像，或提前安装系统 Python 3.11。

## 7. 离线服务器

先在相同 Linux 架构、相同 Python 次版本的联网机器上准备仓库、ProPainter 源码、模型和 wheel。以下示例使用 Python 3.11 和 CUDA 12.8。

```bash
bash scripts/deploy.sh source --components both
bash scripts/deploy.sh models --components both

python3.11 -m pip download --dest wheelhouse \
  -r scripts/requirements-linux.txt \
  -r scripts/requirements-propainter.txt \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

把项目、`CLEAR-models/`、`ProPainter/` 和 `wheelhouse/` 上传到离线服务器。上传时排除 `.venv/`、`.deploy/`、`.test-env/` 和历史视频任务目录，目标机器重新创建环境。

离线服务器预先准备 Python 3.11、NVIDIA 驱动、curl、tar、gzip、util-linux 提供的 flock 及必要的系统库，再执行。

```bash
bash scripts/deploy.sh all \
  --dir /data/Removesubtitles \
  --components both \
  --python /usr/bin/python3.11 \
  --cuda cu128 \
  --offline \
  --wheelhouse /data/wheelhouse \
  --skip-system \
  --yes
```

离线模式使用本地源码和 wheel。模型缺失或哈希错误时会指出具体文件。需要在 CPU 服务器上预先下载模型时，可以单独运行 `models`，GPU 运行环境在推理服务器上执行 `env` 和 `verify` 完成验证。

## 8. 多显卡服务器

通过 `--gpu` 选择一张物理 GPU，编号也可替换成 NVIDIA GPU UUID。

```bash
bash scripts/deploy.sh env --gpu 3
bash scripts/deploy.sh verify --gpu 3
bash scripts/deploy.sh start --gpu 3
```

脚本会设置 `CUDA_VISIBLE_DEVICES`，程序里的 `cuda:0` 对应这张被选中的显卡。一个任务使用一张 GPU。网页的 Gunicorn 进程负责上传、进度、播放和对比导出，普通三档由独立 PHP 队列调度。

## 9. 反向代理与 HTTPS

已有 Nginx 或云负载均衡时，可让应用监听回环地址，并登记实际访问域名。

```bash
bash scripts/deploy.sh start \
  --bind 127.0.0.1 \
  --public-host video.example.com \
  --port 8765
```

Nginx 的 HTTP 配置示例。

```nginx
server {
    listen 80;
    server_name video.example.com;
    client_max_body_size 8g;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $http_host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 600s;
    }
}
```

TLS 证书可由 Nginx 或云负载均衡配置。访问域名发生变化时，更新 `--public-host` 后重启服务。保留原始 Host 和协议转发头，可让网页上传的同源校验正常工作。

## 10. 开机自启

`start` 启动的后台服务可跨 SSH 会话运行。需要开机自启时，可创建 `/etc/systemd/system/removesubtitles.service`，将用户和路径替换成实际部署值。

```ini
[Unit]
Description=RemoveSubtitles web service
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
User=ubuntu
WorkingDirectory=/data/Removesubtitles
PIDFile=/data/Removesubtitles/.deploy/server.pid
ExecStart=/bin/bash /data/Removesubtitles/scripts/deploy.sh start
ExecStop=/bin/bash /data/Removesubtitles/scripts/deploy.sh stop
Restart=on-failure
RestartSec=5
TimeoutStartSec=120
KillMode=control-group

[Install]
WantedBy=multi-user.target
```

交给 systemd 管理前先停止手动启动的服务。

```bash
bash scripts/deploy.sh stop
sudo systemctl daemon-reload
sudo systemctl enable --now removesubtitles
sudo systemctl status removesubtitles
```

systemd 管理期间通过 `sudo systemctl restart removesubtitles` 重启。

## 11. 三档字幕方案

网页和 API 从同一个 `GET /api/modes` 读取产品配置，支持 Lite · 快速、Pro · 标准（默认）、Max · 增加。网页批量提交会分别保存每个视频的档位和选区。可以通过任务状态里的 `resolved_plan` 查看该视频对应的处理阶段、参数和档位配置校验值。[三档处理与接口说明](../php/README.md)

通过部署脚本启动时，网页后端自动读取 `.deploy/api/settings.json` 与 `.deploy/api/api.token`。手动部署可用 `SUBTITLE_API_URL` 指定 PHP 内网地址，`SUBTITLE_API_TOKEN_FILE` 指定服务器端密钥文件，`SUBTITLE_DATA_DIR` 指定网页、PHP 接口和队列共享的任务目录。网页需要读取原片和成片，并能在任务目录内写入对比视频。

旧版自定义 CLEAR 接口仍支持 `--preset fast|balanced|quality` 设置模型预设，参数通过 `CLEAR_PRESET` 传给该处理路径。

## 12. 可选安装 LTX

完整安装 CLEAR、ProPainter 和 LTX 字幕用途。

```bash
bash scripts/deploy.sh all \
  --dir /data/Removesubtitles \
  --components both \
  --preset balanced \
  --with-ltx \
  --ltx-tasks subtitle \
  --network cn \
  --public-host SERVER_IP \
  --port 8765 \
  --yes
```

LTX 默认放在 `<安装目录>/LTX2-ICEdit-Insight`，可用 `--ltx-root /data/LTX2-ICEdit-Insight` 指定其他磁盘。它使用自己的 `.venv`，固定 PyTorch 2.8.0、torchvision 0.23.0、torchaudio 2.8.0 和 transformers 4.57.6，CUDA wheel 跟随选定的下载源。

`--ltx-tasks subtitle` 安装去字幕 LoRA，`watermark` 安装去水印 LoRA，`both` 安装两种。各用途共用 Insight FP8 底模和 Gemma 12B 文本编码器。网页会根据已安装用途提供 LTX 模式。

| 组件 | 相对 LTX 目录的位置 |
|---|---|
| Insight FP8 checkpoint | `models/checkpoints/ltx-2.3-edit-insight-dev-fp8.safetensors` |
| Gemma 文本模型、tokenizer、processor | `models/gemma_configs/` |
| 去字幕 LoRA | `models/loras/ltx2.3-train/ltx2.3-ic-subtitles-remove-general.safetensors` |
| 去水印 LoRA | `models/loras/ltx2.3-train/ltx2.3-ic-watermark-remove-general.safetensors` |

Web 使用单阶段 LTX 推理。安装清单固定源码版本、模型 revision、大小与 SHA-256，下载过程支持备用地址和断点续传。源码和权重来自 `joyfoxai/LTX2-ICEdit-Insight`、`joyfox/LTX2.3-ICEdit-Insight`。Gemma 使用 `OpenPipe/gemma-3-12b-it-text-only` 的 BF16 文本权重，其 `model.*` 键名和维度与固定版本的上游文本编码器匹配，processor 配置取自固定版本的 Lightricks 仓库。完整来源与 revision 记录在 `scripts/ltx_manifest.json`。

为已有环境补充 LTX 时，分别运行四步，再在任务结束后重新启动服务。

```bash
bash scripts/deploy.sh source --with-ltx --ltx-tasks both
bash scripts/deploy.sh env --with-ltx
bash scripts/deploy.sh models --with-ltx
bash scripts/deploy.sh verify --with-ltx
bash scripts/deploy.sh restart --with-ltx
```

`verify` 会校验所选文件哈希、safetensors 结构、Gemma 分片索引、上游模型键名及维度、本地 tokenizer/processor、CUDA BF16/FP8 操作和视频编解码，并构造所选用途的管线。成功后发布 `.install/models.complete`、`environment.complete`、`runtime.complete` 和 `selection.json`，供网页判断就绪状态。原安装使用的 `test.complete` 标记继续兼容。

离线准备时，在相同 Linux 架构和 Python 次版本的联网机器中下载 LTX 源码、权重及独立 wheelhouse。

```bash
python3.11 -m pip download --dest ltx-wheelhouse \
  -r scripts/requirements-ltx.txt \
  pip==25.2 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
```

将 LTX 目录连同 `.install/source-*.tar.gz` 和 wheelhouse 上传到目标服务器，在 LTX 助手中使用 `--offline --wheelhouse /data/ltx-wheelhouse`，依次运行 `source`、`env`、`models`、`verify`。主部署脚本使用单个 `--wheelhouse` 选项时，该目录应包含 CLEAR、ProPainter 和所选 LTX 的全部 wheel。目标机重新创建独立环境。

下面以双用途和 Python 3.11 为例，在已有本地源码与模型文件的目标机分步执行。

```bash
python3.11 scripts/deploy_ltx.py source --root /data/LTX2-ICEdit-Insight --tasks both --offline
python3.11 scripts/deploy_ltx.py env --root /data/LTX2-ICEdit-Insight --tasks both --offline --wheelhouse /data/ltx-wheelhouse
python3.11 scripts/deploy_ltx.py models --root /data/LTX2-ICEdit-Insight --tasks both --offline
python3.11 scripts/deploy_ltx.py verify --root /data/LTX2-ICEdit-Insight --tasks both --offline --gpu 0
```

## 13. 已有安装升级

先获取新版部署入口。已有新版仓库可直接使用 `scripts/deploy.sh`。

```bash
curl -fL https://raw.githubusercontent.com/xaiojie/Removesubtitles/main/scripts/deploy.sh -o deploy.sh
bash deploy.sh update --dir /data/Removesubtitles --wait-idle 1800
```

Git 安装默认从 `origin/main` 快进升级，可用 `--update-ref` 指定目标。归档安装会下载新版项目归档。升级保留模型、任务视频、虚拟环境和本地配置，先在暂存目录检查代码及现有环境，再备份并替换程序文件。源码应用或服务启动失败时会恢复原源码与原服务配置。

受部署脚本管理的新版服务通过本机维护接口暂停接收新任务，等待已接收的上传、排队、推理和对比视频任务完成后重启。超过 `--wait-idle` 的等待时间后释放维护状态，保留运行中的服务。维护接口使用安装目录下的私有随机令牌。

旧版服务首次升级时缺少维护接口，脚本会检查任务文件及工作进程，并给出停止方法。请在网页确认任务完成、关闭上传页面后，通过原部署脚本停止服务，再执行新版 `update`，完成后执行 `start`。通过其他方式启动的服务，需要使用其原管理方式停止。

使用 systemd 的实例，在任务完成后通过对应服务单元停止和启动，保持进程纳入原服务管理。

```bash
sudo systemctl stop removesubtitles
bash deploy.sh update --dir /data/Removesubtitles
sudo systemctl start removesubtitles
```

```bash
# 旧版服务任务完成后，使用原启动目录停止
bash /data/Removesubtitles/scripts/deploy.sh stop

# 新版入口升级，再启动
bash deploy.sh update --dir /data/Removesubtitles
bash /data/Removesubtitles/scripts/deploy.sh start --preset balanced
```

`update` 更新程序源码并校验现有环境。依赖安装由 `env` 完成，运行环境和模型校验由 `verify` 完成。Git 工作区存在本地源码修改时，先保存或提交修改再升级。

离线升级归档安装时，可指定本地源码包。

```bash
bash deploy.sh update \
  --dir /data/Removesubtitles \
  --offline \
  --update-archive /data/Removesubtitles-main.tar.gz
```

离线 Git 升级需要提前准备目标 Git revision，并将新版 `deploy.sh` 与 `deploy_helper.py` 放在同一个目录，再使用 `--offline --update-ref REVISION`。

## 14. 故障处理与验证范围

| 现象 | 处理方法 |
|---|---|
| 下载中断 | 重新执行 `models`，保留 `.part` 文件续传 |
| 模型 SHA-256 不匹配 | 重试会重新下载，可切换 `--network` 或指定下载域名 |
| 磁盘不足 | 选择更大的 `--dir` 或 ProPainter 目录，为任务文件留出空间 |
| 找不到 CUDA wheel | 检查 Python 版本、CPU 架构、`--cuda` 与 `--torch-index` |
| CUDA / BF16 验证失败 | 检查 NVIDIA 驱动、GPU 架构与 PyTorch wheel 的兼容性 |
| `runtime.ready` 尚未就绪 | 执行 `verify`，根据实际报错修正依赖或 GPU 环境 |
| 网页返回 Host 校验错误 | 用实际访问 IP 或域名设置 `--public-host` 后重启 |
| 公网无法连接 | 检查监听地址、安全组、防火墙和端口映射 |
| 服务启动后退出 | 查看 `.deploy/server.log` 和 `.deploy/launcher.log` |
| LTX 模型或环境尚未就绪 | 使用保存的 `--with-ltx` 配置运行 `models` 和 `verify`，按具体报错补充文件或依赖 |
| LTX 所选用途的 LoRA 未安装 | 通过 `source --with-ltx --ltx-tasks both` 保存用途选择，再运行 `models` 和 `verify` |
| 升级等待任务超时 | 任务继续执行，可增加 `--wait-idle` 或完成后重试 |
| 旧版服务缺少维护接口 | 确认任务完成、关闭上传页面，通过原启动方式停止服务后升级 |

部署工具的自动测试使用本地 HTTP 服务覆盖续传、忽略 Range、网络中断、校验失败、离线模式和路径检查，也覆盖命令行、预设持久化、访问域名、源码升级与失败恢复、维护队列，以及 LTX 的独立环境和就绪标记。运行时 `verify` 会在目标服务器检查实际依赖、GPU 和模型文件。新版本的完整 Linux GPU 安装与视频推理验收需要在实际服务器上完成。
