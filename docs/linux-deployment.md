# Linux 服务器部署教程

仓库提供 `scripts/deploy.sh` 作为部署入口。直接运行会显示菜单，也可以用命令行分步执行。脚本安装所选模型及依赖，模型下载支持断点续传、备用地址和 SHA-256 校验。

## 1. 服务器准备

推荐 Ubuntu 22.04/24.04、Debian 11/12、Rocky Linux / AlmaLinux / RHEL 8/9、Alibaba Cloud Linux 3 等 glibc 2.28 以上的 Linux。系统包安装支持 `apt-get`、`dnf` 和 `yum`。支持检测 x86_64 和 aarch64，ARM 服务器需要软件源提供对应架构的 CUDA wheel。

推理需要 NVIDIA 驱动。先在服务器执行 `nvidia-smi`，确认驱动正常。脚本会安装 PyTorch 自带的 CUDA 运行库，并执行 CUDA 运算、BF16 和 torchvision 光流相关算子验证。推荐 Ampere 或更新架构的 NVIDIA 显卡。实际兼容性以运行环境验证结果为准。

建议为完整安装准备至少 60 GiB 磁盘和 32 GiB 系统内存，并给视频及临时帧文件留出空间。CLEAR 模型及 tokenizer 下载量约 14.2 GiB，ProPainter 权重约 190 MiB，Python/CUDA 依赖还会占用数 GiB。建议使用 SSD 数据盘作为安装目录。

建议显存 24 GiB 以上。视频分辨率、选区和分段长度影响实际用量，1080p、4K 视频可能需要更多显存。脚本会采用仓库默认推理参数，后续可在网页中调整。

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
  --network cn \
  --public-host SERVER_IP \
  --port 8765 \
  --yes
```

把 `SERVER_IP` 换成实际公网 IP。云服务器安全组和系统防火墙需要放行所选端口。成功后访问 `http://SERVER_IP:8765/`。

海外或能够直连官方服务的网络可以选择 `--network global`。`--network auto` 会测试 Hugging Face 连通性，选择网络配置。`cn` 默认使用清华 PyPI 源、ModelScope 和 Hugging Face 镜像，`global` 使用官方 PyPI 和模型地址。模型文件还会尝试清单中配置的备用地址。

```bash
bash deploy.sh all --dir /data/Removesubtitles --network global --yes
```

完整流程依次执行系统检查、系统包安装、源码准备、Python/CUDA 环境安装、模型下载、模型与运行环境校验、后台启动。首次执行会把选项保存到安装目录的 `.deploy/config.sh`，后续可继续分步运行。

安装系统包会使用 root 或 sudo。已经准备好系统依赖的普通用户可加 `--skip-system`。建议始终用同一个部署用户管理项目，系统包安装由脚本单独调用 sudo。

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

脚本会设置 `CUDA_VISIBLE_DEVICES`，程序里的 `cuda:0` 对应这张被选中的显卡。当前程序一个任务使用一张 GPU，网页采用单任务队列。Gunicorn 固定为一个 worker、八个线程，保证任务队列在同一进程中。八卡服务器的任务并行需要额外的调度实现。

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

## 11. 故障处理与验证范围

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

部署工具的自动测试使用本地 HTTP 服务覆盖续传、忽略 Range、网络中断、校验失败、离线模式和路径检查，也覆盖命令行与访问域名设置。Linux CUDA 全流程安装需要在实际 GPU 服务器上验证，脚本会在部署时执行相应的运行环境检查。
