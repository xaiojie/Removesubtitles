#!/usr/bin/env bash
# Linux server installer. Run without arguments for the interactive menu.
set -euo pipefail

usage() {
    cat <<'HELP'
RemoveSubtitles Linux 服务器部署

用法  bash deploy.sh [命令] [选项]
命令  menu | all | check | system | source | env | models | verify
      update | start | stop | restart | status
      api-start | api-stop | api-status  单独管理三档 PHP 接口和队列
      vsr-models [--mode all|lama|sttn-auto|sttn-det|opencv|propainter] [--detector mobile|server] [--list|--verify]
      vsr-install               安装独立的 VSR Web 服务，详见 video-subtitle-remover/README.md

  --dir PATH                 项目、环境和 CLEAR 模型目录
  --components both|clear|propainter
  --preset fast|balanced|quality
                             CLEAR 默认处理预设，首次安装默认 balanced
  --with-ltx                 额外安装 LTX，使用独立虚拟环境
  --with-api                 同时部署 Lite、Pro、Max 接口及串行推理队列，安装两个组件
  --without-api              后续部署只启动原网页
  --api-port PORT            PHP 接口端口，默认 8810；监听地址沿用 --bind/--public-host
  --ltx-root PATH            LTX 源码、环境和模型目录
  --ltx-tasks subtitle|watermark|both
                             LTX 模型用途，默认 subtitle
  --network auto|cn|global    网络配置，默认 auto
  --python PATH              创建环境所用的 Python 3.10～3.12
  --cuda auto|cu128|cu126     PyTorch CUDA wheel，默认 auto
  --pip-index URL            自定义 Python 软件源
  --torch-index URL          自定义 PyTorch CUDA wheel 源
  --hf-endpoint URL          Hugging Face 下载域名
  --github-prefix URL        GitHub 下载代理前缀
  --proxy URL                本次运行的 HTTP/HTTPS 代理
  --propainter-root PATH     ProPainter 源码和权重目录
  --workers N                同时下载的模型文件数，默认 2，范围 1～8
  --gpu ID                   GPU 编号或 UUID，写入服务配置
  --port PORT                网页端口，默认 8765
  --public-host HOST         网页使用的公网 IP 或域名，自动监听 0.0.0.0
  --bind ADDRESS             自定义监听地址，默认 127.0.0.1
  --offline                  使用本地源码、模型和 wheel 文件
  --wheelhouse PATH          离线 Python wheel 目录
  --update-ref REF           Git 升级目标，默认 main，仅允许快进
  --update-archive PATH       使用本地项目源码压缩包升级
  --wait-idle SECONDS         升级等待任务队列清空的时间，默认 300
  --skip-system              跳过系统软件包安装
  --yes                      全流程免交互执行
  --help                     显示帮助

示例  bash deploy.sh all --dir /data/Removesubtitles --network cn --yes
      bash deploy.sh all --with-api --api-port 8810 --yes
      bash deploy.sh models --components clear --hf-endpoint https://hf-mirror.com
      bash deploy.sh start --public-host SERVER_IP --port 8765
      bash deploy.sh update --dir /data/Removesubtitles --wait-idle 1800

update 保留模型、任务、本地配置与环境，仅更新源码。
升级前使用已有环境检查依赖。需要新依赖时先执行 env，再重新 update。
HELP
}

fail() { printf '错误 %s\n' "$*" >&2; exit 1; }
bad_option() { printf '参数错误 %s\n' "$*" >&2; exit 2; }
log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || bad_option "$1 需要一个值"; }

case "${1:-}" in
    vsr-models|vsr-install)
        vsr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)/video-subtitle-remover"
        vsr_command=$1
        shift
        if [[ $vsr_command == vsr-models ]]; then
            exec python3 "$vsr_dir/deploy/download_models.py" "$@"
        fi
        exec bash "$vsr_dir/deploy/install.sh" "$@"
        ;;
esac

for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then usage; exit 0; fi
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(dirname "$SCRIPT_DIR")
[[ -f "$ROOT/CLEAR-web/app.py" ]] || ROOT="$PWD/Removesubtitles"
arguments=("$@")
for ((position=0; position<${#arguments[@]}; position++)); do
    if [[ "${arguments[$position]}" == --dir ]]; then
        ((position+=1))
        [[ $position -lt ${#arguments[@]} ]] || bad_option '--dir 需要目录'
        ROOT=${arguments[$position]}
    fi
done
[[ "$ROOT" == /* ]] || ROOT="$PWD/$ROOT"
STATE="$ROOT/.deploy"
COMPONENTS=both NETWORK=auto PYTHON_BIN= CUDA_WHEEL=auto
PIP_URL=${PIP_INDEX_URL:-} TORCH_URL= HF_ENDPOINT= GITHUB_PREFIX=
PORT=8765 BIND=127.0.0.1 PUBLIC_HOST= GPU=${CUDA_VISIBLE_DEVICES:-}
PROPAINTER_DIR= DOWNLOAD_WORKERS=2
PRESET=balanced WITH_LTX=0 LTX_DIR= LTX_TASKS=subtitle
WITH_API=0 API_PORT=8810
OFFLINE=0 WHEELHOUSE= SKIP_SYSTEM=0 YES=0 COMMAND=menu
UPDATE_REF=main UPDATE_ARCHIVE= WAIT_IDLE=300
BIND_SET=0 PUBLIC_SET=0 NETWORK_SET=0 PIP_SET=0 TORCH_SET=0 HF_SET=0 CUDA_SET=0
if [[ -f "$STATE/config.sh" ]]; then source "$STATE/config.sh"; fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        menu|all|check|system|source|env|models|verify|update|start|stop|restart|status|api-start|api-stop|api-status) COMMAND=$1; shift ;;
        --dir) need_value "$@"; shift 2 ;;
        --components) need_value "$@"; COMPONENTS=$2; shift 2 ;;
        --preset) need_value "$@"; PRESET=$2; shift 2 ;;
        --with-ltx) WITH_LTX=1; shift ;;
        --with-api) WITH_API=1; shift ;;
        --without-api) WITH_API=0; shift ;;
        --api-port) need_value "$@"; API_PORT=$2; shift 2 ;;
        --ltx-root) need_value "$@"; LTX_DIR=$2; shift 2 ;;
        --ltx-tasks) need_value "$@"; LTX_TASKS=$2; shift 2 ;;
        --network) need_value "$@"; NETWORK=$2; NETWORK_SET=1; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
        --cuda) need_value "$@"; CUDA_WHEEL=$2; CUDA_SET=1; shift 2 ;;
        --pip-index) need_value "$@"; PIP_URL=$2; PIP_SET=1; shift 2 ;;
        --torch-index) need_value "$@"; TORCH_URL=$2; TORCH_SET=1; shift 2 ;;
        --hf-endpoint) need_value "$@"; HF_ENDPOINT=$2; HF_SET=1; shift 2 ;;
        --github-prefix) need_value "$@"; GITHUB_PREFIX=$2; shift 2 ;;
        --proxy) need_value "$@"; export HTTP_PROXY=$2 HTTPS_PROXY=$2 http_proxy=$2 https_proxy=$2; shift 2 ;;
        --propainter-root) need_value "$@"; PROPAINTER_DIR=$2; shift 2 ;;
        --workers) need_value "$@"; DOWNLOAD_WORKERS=$2; shift 2 ;;
        --gpu) need_value "$@"; GPU=$2; shift 2 ;;
        --port) need_value "$@"; PORT=$2; shift 2 ;;
        --bind) need_value "$@"; BIND=$2; BIND_SET=1; shift 2 ;;
        --public-host) need_value "$@"; PUBLIC_HOST=$2; PUBLIC_SET=1; shift 2 ;;
        --wheelhouse) need_value "$@"; WHEELHOUSE=$2; shift 2 ;;
        --update-ref) need_value "$@"; UPDATE_REF=$2; shift 2 ;;
        --update-archive) need_value "$@"; UPDATE_ARCHIVE=$2; shift 2 ;;
        --wait-idle) need_value "$@"; WAIT_IDLE=$2; shift 2 ;;
        --offline) OFFLINE=1; shift ;;
        --skip-system) SKIP_SYSTEM=1; shift ;;
        --yes|-y) YES=1; shift ;;
        *) bad_option "未知选项 $1" ;;
    esac
done

if [[ $NETWORK_SET == 1 ]]; then
    [[ $PIP_SET == 1 ]] || PIP_URL=
    [[ $TORCH_SET == 1 ]] || TORCH_URL=
    [[ $HF_SET == 1 ]] || HF_ENDPOINT=
fi
if [[ $CUDA_SET == 1 && $TORCH_SET == 0 ]]; then TORCH_URL=; fi

[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) || bad_option '端口范围为 1～65535'
[[ "$WITH_API" == 0 || "$WITH_API" == 1 ]] || bad_option 'with-api 配置必须为 0 或 1'
[[ "$API_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && ((API_PORT <= 65535)) || bad_option 'API 端口范围为 1～65535'
if [[ $WITH_API == 1 ]]; then
    ((API_PORT != PORT)) || bad_option '网页与 API 需要使用不同端口'
    [[ "$COMPONENTS" == both ]] || bad_option '三档 API 使用 CLEAR 和 ProPainter，请加 --components both'
fi
[[ "$DOWNLOAD_WORKERS" =~ ^[0-9]+$ ]] && ((DOWNLOAD_WORKERS >= 1 && DOWNLOAD_WORKERS <= 8)) || bad_option 'workers 范围为 1～8'
case "$COMPONENTS" in clear|propainter|both) ;; *) bad_option 'components 必须为 clear、propainter 或 both' ;; esac
case "$PRESET" in fast|balanced|quality) ;; *) bad_option 'preset 必须为 fast、balanced 或 quality' ;; esac
case "$LTX_TASKS" in subtitle|watermark|both) ;; *) bad_option 'ltx-tasks 必须为 subtitle、watermark 或 both' ;; esac
[[ "$WITH_LTX" == 0 || "$WITH_LTX" == 1 ]] || bad_option 'with-ltx 配置必须为 0 或 1'
[[ "$WAIT_IDLE" =~ ^[0-9]+$ ]] || bad_option 'wait-idle 必须为非负整数秒数'
case "$NETWORK" in auto|cn|global) ;; *) bad_option 'network 必须为 auto、cn 或 global' ;; esac
case "$CUDA_WHEEL" in auto|cu128|cu126) ;; *) bad_option 'cuda 必须为 auto、cu128 或 cu126' ;; esac
[[ "$PUBLIC_HOST" != *[[:space:]/]* ]] || bad_option '--public-host 使用 IP 或域名，多个地址用逗号分隔'
if [[ $PUBLIC_SET == 1 && $BIND_SET == 0 ]]; then BIND=0.0.0.0; fi
[[ $(uname -s) == Linux ]] || fail '此脚本用于 Linux 服务器。'
case $(uname -m) in x86_64|aarch64|arm64) ;; *) fail '当前架构缺少已配置的 PyTorch wheel，请使用 x86_64 或 aarch64。' ;; esac
if command -v getconf >/dev/null 2>&1; then
    libc_version=$(getconf GNU_LIBC_VERSION 2>/dev/null) || fail '需要 glibc Linux，PyTorch CUDA wheel 不支持当前 libc。'
    libc_version=${libc_version#* }
    libc_major=${libc_version%%.*}
    libc_minor=${libc_version#*.}
    libc_minor=${libc_minor%%.*}
    ((libc_major > 2 || (libc_major == 2 && libc_minor >= 28))) || fail 'PyTorch 2.8 CUDA wheel 需要 glibc 2.28 或更新版本。'
fi
[[ -n "$PROPAINTER_DIR" ]] || PROPAINTER_DIR="$ROOT/ProPainter"
[[ "$PROPAINTER_DIR" == /* ]] || PROPAINTER_DIR="$PWD/$PROPAINTER_DIR"
[[ -n "$LTX_DIR" ]] || LTX_DIR="$ROOT/LTX2-ICEdit-Insight"
[[ "$LTX_DIR" == /* ]] || LTX_DIR="$PWD/$LTX_DIR"
VENV="$ROOT/.venv/bin/python"
VSR_DIR="$ROOT/video-subtitle-remover"
LITE_PYTHON="$VSR_DIR/.venv/bin/python"
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
INSTALL_LOCKED=0

acquire_install_lock() {
    if [[ $INSTALL_LOCKED == 1 ]]; then return; fi
    command -v flock >/dev/null || fail '缺少 flock，请先执行 system 安装 util-linux。'
    mkdir -p "$STATE"
    exec 9> "$STATE/install.lock"
    flock -n 9 || fail '这个目录已有部署任务正在运行。'
    INSTALL_LOCKED=1
}

save_config() {
    acquire_install_lock
    mkdir -p "$STATE"
    local setting line
    local settings=(COMPONENTS NETWORK PYTHON_BIN CUDA_WHEEL PIP_URL TORCH_URL HF_ENDPOINT GITHUB_PREFIX PORT BIND PUBLIC_HOST GPU PROPAINTER_DIR DOWNLOAD_WORKERS PRESET WITH_LTX LTX_DIR LTX_TASKS WITH_API API_PORT)
    umask 077
    : > "$STATE/config.sh.tmp"
    if [[ -f "$STATE/config.sh" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)= ]]; then
                case " ${settings[*]} " in *" ${BASH_REMATCH[1]} "*) continue ;; esac
            fi
            printf '%s\n' "$line" >> "$STATE/config.sh.tmp"
        done < "$STATE/config.sh"
    fi
    for setting in "${settings[@]}"; do
        printf '%s=%q\n' "$setting" "${!setting}" >> "$STATE/config.sh.tmp"
    done
    mv "$STATE/config.sh.tmp" "$STATE/config.sh"
}

boot_python() {
    local candidate
    for candidate in "$VENV" "$PYTHON_BIN" python3 python3.11 python3.12 python3.10; do
        if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then
            BOOT_PYTHON=$(command -v "$candidate")
            return
        fi
    done
    fail '需要 Python 3.8 以上以准备环境，请先执行 system。'
}

helper() {
    boot_python
    local helper_file=${HELPER_OVERRIDE:-$ROOT/scripts/deploy_helper.py}
    [[ -f "$helper_file" ]] || helper_file="$SCRIPT_DIR/deploy_helper.py"
    [[ -f "$helper_file" ]] || fail '请先执行 source 下载项目源码。'
    local network=$NETWORK endpoint=$HF_ENDPOINT
    [[ "$network" != auto ]] || network=global
    [[ -n "$endpoint" ]] || endpoint=https://huggingface.co
    local optional=()
    [[ $WITH_LTX == 0 ]] || optional+=(--with-ltx)
    "$BOOT_PYTHON" "$helper_file" "$@" --root "$ROOT" --components "$COMPONENTS" --propainter-root "$PROPAINTER_DIR" --network "$network" --hf-endpoint "$endpoint" --github-prefix "$GITHUB_PREFIX" --workers "$DOWNLOAD_WORKERS" --python "$VENV" --port "$PORT" --bind "$BIND" --public-host "$PUBLIC_HOST" --gpu "$GPU" --preset "$PRESET" --ltx-root "$LTX_DIR" ${optional[@]+"${optional[@]}"}
}

api_service() {
    boot_python
    "$BOOT_PYTHON" "$ROOT/scripts/deploy_api.py" "$1" --root "$ROOT" --bind "$BIND" --port "$API_PORT" --propainter-root "$PROPAINTER_DIR" --gpu "$GPU" --wait "$WAIT_IDLE"
}

api_dependencies() {
    php -r 'exit(PHP_VERSION_ID >= 80100 && extension_loaded("pdo_sqlite") && function_exists("proc_open") ? 0 : 1);' || fail '三档 API 需要 PHP 8.1+、PDO SQLite 和 proc_open。'
    command -v ffprobe >/dev/null && command -v ffmpeg >/dev/null || fail '三档 API 需要系统 FFmpeg 和 ffprobe。'
    local encoders
    encoders=$(ffmpeg -hide_banner -encoders 2>/dev/null)
    [[ "$encoders" == *libx264* ]] || fail '系统 FFmpeg 需要包含 libx264 编码器。'
}

lite_models() {
    boot_python
    local args=(--mode lama --detector mobile --model-dir "$VSR_DIR/backend/models")
    if [[ ${1:-} == verify || $OFFLINE == 1 ]]; then args+=(--verify); fi
    if [[ -n ${VSR_MODEL_BASE_URL:-} ]]; then args+=(--base-url "$VSR_MODEL_BASE_URL")
    elif [[ -n $GITHUB_PREFIX ]]; then
        local upstream
        upstream=$("$BOOT_PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); print("https://raw.githubusercontent.com/{repository}/{revision}/backend/models".format(**m))' "$VSR_DIR/deploy/models.json")
        args+=(--base-url "${GITHUB_PREFIX%/}/$upstream")
    fi
    "$BOOT_PYTHON" "$VSR_DIR/deploy/download_models.py" "${args[@]}"
}

ltx_helper() {
    [[ $WITH_LTX == 1 ]] || return 0
    boot_python
    local installer="$ROOT/scripts/deploy_ltx.py"
    [[ -f "$installer" ]] || installer="$SCRIPT_DIR/deploy_ltx.py"
    [[ -f "$installer" ]] || fail '缺少 LTX 部署助手，请先更新项目源码。'
    local interpreter=${PYTHON_BIN:-$BOOT_PYTHON} network=$NETWORK
    [[ "$network" != auto ]] || network=global
    local optional=()
    [[ $OFFLINE == 0 ]] || optional+=(--offline)
    [[ -z "$WHEELHOUSE" ]] || optional+=(--wheelhouse "$WHEELHOUSE")
    [[ -z "$PIP_URL" ]] || optional+=(--pip-index "$PIP_URL")
    local selected_torch=${LTX_TORCH_INDEX:-$TORCH_URL}
    [[ -z "$selected_torch" ]] || optional+=(--torch-index "$selected_torch")
    [[ -z "$GPU" ]] || optional+=(--gpu "$GPU")
    "$BOOT_PYTHON" "$installer" "$@" --root "$LTX_DIR" --python "$interpreter" --network "$network" --hf-endpoint "${HF_ENDPOINT:-https://huggingface.co}" --github-prefix "$GITHUB_PREFIX" --workers "$DOWNLOAD_WORKERS" --tasks "$LTX_TASKS" ${optional[@]+"${optional[@]}"}
}

network_setup() {
    if [[ "$NETWORK" == auto ]]; then
        if [[ $OFFLINE == 1 ]]; then NETWORK=global
        elif curl -fsSL --connect-timeout 3 --max-time 5 -o /dev/null https://huggingface.co/api/models/Wan-AI/Wan2.1-T2V-1.3B; then NETWORK=global
        else NETWORK=cn; fi
    fi
    if [[ -z "$PIP_URL" ]]; then
        if [[ "$NETWORK" == cn ]]; then PIP_URL=https://pypi.tuna.tsinghua.edu.cn/simple
        else PIP_URL=https://pypi.org/simple; fi
    fi
    if [[ -z "$HF_ENDPOINT" ]]; then
        if [[ "$NETWORK" == cn ]]; then HF_ENDPOINT=https://hf-mirror.com
        else HF_ENDPOINT=https://huggingface.co; fi
    fi
    save_config
}

download_archive() {
    local url=$1 target=$2
    [[ $OFFLINE == 0 ]] || fail "离线模式需要预先准备源码 $target"
    local selected=$url
    [[ -z "$GITHUB_PREFIX" ]] || selected="${GITHUB_PREFIX%/}/$url"
    if ! curl -fL --retry 3 --connect-timeout 15 -C - "$selected" -o "$target.part"; then
        if [[ "$selected" != "$url" ]]; then selected=$url; fi
        curl -fL --retry 3 --connect-timeout 15 "$selected" -o "$target.part"
    fi
    mv "$target.part" "$target"
}

step_check() {
    log '检查系统、GPU 和磁盘'
    uname -srmo
    if [[ -f /etc/os-release ]]; then (source /etc/os-release; printf '%s\n' "$PRETTY_NAME"); fi
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else printf '未检测到 NVIDIA 驱动。可先下载模型，推理环境需要可用的 NVIDIA 驱动。\n'; fi
    local disk_path=$ROOT
    while [[ ! -d "$disk_path" ]]; do disk_path=$(dirname "$disk_path"); done
    df -h "$disk_path"
    printf '安装目录 %s\n组件 %s\n网络 %s\n建议为完整环境和任务文件准备 60 GiB 以上磁盘。\n' "$ROOT" "$COMPONENTS" "$NETWORK"
}

step_system() {
    log '安装系统依赖'
    if [[ $OFFLINE == 1 || $SKIP_SYSTEM == 1 ]]; then
        local program
        for program in curl tar gzip flock; do command -v "$program" >/dev/null || fail "缺少系统命令 $program，请用系统软件源预先安装。"; done
        boot_python
        if [[ $WITH_API == 1 ]]; then api_dependencies; fi
        return
    fi
    local privilege=()
    local apt_api=() rpm_api=()
    if [[ $WITH_API == 1 ]]; then
        apt_api=(php-cli php-sqlite3 ffmpeg libgl1)
        rpm_api=(php-cli php-pdo ffmpeg mesa-libGL)
    fi
    if [[ $EUID != 0 ]]; then command -v sudo >/dev/null || fail '安装系统软件包需要 root 或 sudo，也可使用 --skip-system。'; privilege=(sudo); fi
    if command -v apt-get >/dev/null; then
        "${privilege[@]}" apt-get update
        "${privilege[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl tar gzip python3 python3-pip python3-venv libgomp1 util-linux ${apt_api[@]+"${apt_api[@]}"}
    elif command -v dnf >/dev/null; then
        "${privilege[@]}" dnf install -y ca-certificates curl tar gzip python3 python3-pip libgomp util-linux ${rpm_api[@]+"${rpm_api[@]}"}
    elif command -v yum >/dev/null; then
        "${privilege[@]}" yum install -y ca-certificates curl tar gzip python3 python3-pip libgomp util-linux ${rpm_api[@]+"${rpm_api[@]}"}
    else fail '未找到 apt-get、dnf 或 yum，请安装依赖后使用 --skip-system。'; fi
    if [[ $WITH_API == 1 ]]; then api_dependencies; fi
}

step_source() {
    log '准备项目及模型源码'
    acquire_install_lock
    network_setup
    mkdir -p "$ROOT" "$STATE"
    if [[ ! -f "$ROOT/CLEAR-web/app.py" ]]; then
        download_archive https://github.com/xaiojie/Removesubtitles/archive/refs/heads/main.tar.gz "$STATE/project.tar.gz"
        mkdir -p "$STATE/source"
        tar -xzf "$STATE/project.tar.gz" -C "$STATE/source" --strip-components=1 --exclude='*/videos/*'
        [[ -f "$STATE/source/CLEAR-web/app.py" ]] || fail '项目源码压缩包不完整。'
        cp -a "$STATE/source/." "$ROOT/"
        rm -rf "$STATE/source" "$STATE/project.tar.gz"
    fi
    if [[ "$COMPONENTS" != clear && ! -f "$PROPAINTER_DIR/inference_propainter.py" ]]; then
        boot_python
        local revision
        revision=$("$BOOT_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["propainter_revision"])' "$ROOT/scripts/model_manifest.json")
        download_archive "https://github.com/sczhou/ProPainter/archive/$revision.tar.gz" "$STATE/propainter.tar.gz"
        helper extract-propainter --archive "$STATE/propainter.tar.gz"
        printf '%s\n' "$revision" > "$PROPAINTER_DIR/UPSTREAM-COMMIT"
        rm "$STATE/propainter.tar.gz"
    fi
    ltx_helper source
    if [[ $WITH_API == 1 ]]; then
        [[ -f "$ROOT/php/public/index.php" && -f "$ROOT/scripts/deploy_api.py" && -f "$VSR_DIR/deploy/models.json" ]] || fail '缺少三档 API 或 Lite 源码，请先更新仓库（稀疏检出须包含 php、scripts 和 video-subtitle-remover）。'
    fi
}

select_python() {
    local candidate
    if [[ -n "$PYTHON_BIN" ]]; then
        "$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' || fail '--python 需要 Python 3.10～3.12。'
        return
    fi
    for candidate in python3.11 python3.12 python3.10 python3; do
        if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' 2>/dev/null; then
            PYTHON_BIN=$(command -v "$candidate")
            return
        fi
    done
    [[ $OFFLINE == 0 ]] || fail '离线安装请预先准备 Python 3.10～3.12，并用 --python 指定。'
    boot_python
    "$BOOT_PYTHON" -m venv "$STATE/bootstrap"
    "$STATE/bootstrap/bin/python" -m pip install uv --index-url "$PIP_URL"
    export UV_PYTHON_INSTALL_DIR="$STATE/python" UV_PYTHON_BIN_DIR="$STATE/bin"
    "$STATE/bootstrap/bin/uv" python install 3.11
    PYTHON_BIN=$("$STATE/bootstrap/bin/uv" python find 3.11)
}

step_env() {
    log '安装 Python 和 CUDA 依赖'
    [[ -f "$ROOT/scripts/requirements-linux.txt" ]] || fail '请先执行 source。'
    acquire_install_lock
    network_setup
    mkdir -p "$STATE/tmp"
    export TMPDIR="$STATE/tmp"
    select_python
    [[ -x "$VENV" ]] || "$PYTHON_BIN" -m venv "$ROOT/.venv"
    "$VENV" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)' || fail '已有虚拟环境版本不兼容，请换一个安装目录。'
    command -v nvidia-smi >/dev/null || fail '请先安装 NVIDIA 驱动，并确认 nvidia-smi 正常。'
    nvidia-smi >/dev/null || fail 'NVIDIA 驱动不可用。'
    [[ -z "$GPU" ]] || export CUDA_VISIBLE_DEVICES="$GPU"
    local resolved_cuda=$CUDA_WHEEL
    if [[ "$resolved_cuda" == auto ]]; then
        local capability driver_cuda
        local gpu_selector=()
        if [[ -n "$GPU" ]]; then gpu_selector=(-i "${GPU%%,*}"); fi
        capability=$(nvidia-smi ${gpu_selector[@]+"${gpu_selector[@]}"} --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
        driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
        resolved_cuda=cu126
        if [[ "$capability" != 7.0 && "$capability" != 7.2 ]] && [[ "$driver_cuda" == 12.8 || "$driver_cuda" == 12.9 || "$driver_cuda" == 13.* || "$capability" == 1[0-9].* ]]; then resolved_cuda=cu128; fi
    fi
    local resolved_torch=$TORCH_URL
    if [[ -z "$resolved_torch" ]]; then
        resolved_torch="https://download.pytorch.org/whl/$resolved_cuda"
        if [[ "$NETWORK" == cn && "$resolved_cuda" == cu128 && $(uname -m) == x86_64 ]]; then resolved_torch=$PIP_URL; fi
    fi
    local pip_source=(--index-url "$PIP_URL")
    local torch_source=(--index-url "$resolved_torch")
    if [[ $OFFLINE == 1 ]]; then
        [[ -d "$WHEELHOUSE" ]] || fail '--offline env 需要 --wheelhouse 目录。'
        pip_source=(--no-index --find-links "$WHEELHOUSE")
        torch_source=(--no-index --find-links "$WHEELHOUSE")
    elif [[ "$resolved_torch" != "$PIP_URL" ]]; then torch_source+=(--extra-index-url "$PIP_URL"); fi
    rm -f "$ROOT/CLEAR-web/runtime.ready"
    if [[ "$COMPONENTS" != clear ]]; then rm -f "$PROPAINTER_DIR/runtime.ready"; fi
    printf 'torch==2.8.0\ntorchvision==0.23.0\n' > "$STATE/constraints.txt"
    "$VENV" -m pip install --upgrade pip "${pip_source[@]}"
    local installed_cuda
    local cuda_version=12.6
    if [[ "$resolved_cuda" == cu128 ]]; then cuda_version=12.8; fi
    installed_cuda=$("$VENV" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || true)
    if [[ -n "$installed_cuda" && "$installed_cuda" != "$cuda_version" ]]; then
        "$VENV" -m pip uninstall -y torch torchvision
    fi
    "$VENV" -m pip install torch==2.8.0 torchvision==0.23.0 "${torch_source[@]}"
    "$VENV" -m pip install -r "$ROOT/scripts/requirements-linux.txt" -c "$STATE/constraints.txt" "${pip_source[@]}"
    if [[ "$COMPONENTS" != clear ]]; then
        "$VENV" -m pip install -r "$ROOT/scripts/requirements-propainter.txt" -c "$ROOT/CLEAR-web/requirements.txt" -c "$STATE/constraints.txt" "${pip_source[@]}"
    fi
    "$VENV" -m pip check
    "$VENV" -c 'import sys,torch; assert torch.version.cuda == sys.argv[1], "CUDA wheel 与所选版本不一致"; assert torch.cuda.is_available(), "CUDA 不可用"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); torch.ones(1,device="cuda").add_(1)' "$cuda_version"
    helper cuda
    if [[ $WITH_API == 1 ]]; then
        rm -f "$STATE/lite.ready"
        [[ -x "$LITE_PYTHON" ]] || "$PYTHON_BIN" -m venv "$VSR_DIR/.venv"
        local lite_torch=("${torch_source[@]}")
        if [[ $OFFLINE == 0 && -z "$TORCH_URL" ]]; then
            lite_torch=(--index-url "https://download.pytorch.org/whl/$resolved_cuda" --extra-index-url "$PIP_URL")
        fi
        installed_cuda=$("$LITE_PYTHON" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || true)
        if [[ -n "$installed_cuda" && "$installed_cuda" != "$cuda_version" ]]; then
            "$LITE_PYTHON" -m pip uninstall -y torch torchvision
        fi
        "$LITE_PYTHON" -m pip install --upgrade pip "${pip_source[@]}"
        "$LITE_PYTHON" -m pip install torch==2.7.0 torchvision==0.22.0 "${lite_torch[@]}"
        "$LITE_PYTHON" -m pip install -r "$VSR_DIR/requirements-headless.txt" "${pip_source[@]}"
        "$LITE_PYTHON" -m pip check
        "$LITE_PYTHON" -c 'import sys,torch; assert torch.version.cuda == sys.argv[1] and torch.cuda.is_available(), "Lite CUDA 环境不可用"' "$cuda_version"
    fi
    LTX_TORCH_INDEX=$resolved_torch ltx_helper env
    save_config
}

step_models() {
    log '下载所选模型并校验 SHA-256'
    network_setup
    if [[ $OFFLINE == 1 ]]; then helper models --offline; else helper models; fi
    ltx_helper models
    if [[ $WITH_API == 1 ]]; then rm -f "$STATE/lite.ready"; lite_models; fi
}

step_verify() {
    log '校验模型与 CUDA 运行环境'
    acquire_install_lock
    helper verify
    [[ -x "$VENV" ]] || fail '请先执行 env。'
    local helper_file="$ROOT/scripts/deploy_helper.py"
    [[ -z "$GPU" ]] || export CUDA_VISIBLE_DEVICES="$GPU"
    "$VENV" "$helper_file" runtime --root "$ROOT" --components "$COMPONENTS" --propainter-root "$PROPAINTER_DIR"
    ltx_helper verify
    if [[ $WITH_API == 1 ]]; then
        rm -f "$STATE/lite.ready"
        api_dependencies
        lite_models verify
        VSR_MODEL_DIR="$VSR_DIR/backend/models" LITE_DEVICE=cuda "$LITE_PYTHON" "$ROOT/php/inference/lite.py" --verify
        touch "$STATE/lite.ready"
    fi
}

step_start() {
    acquire_install_lock
    [[ -x "$VENV" ]] || fail '请先执行 env。'
    if [[ "$COMPONENTS" != propainter ]]; then [[ -f "$ROOT/CLEAR-web/models.ready" && -f "$ROOT/CLEAR-web/runtime.ready" ]] || fail 'CLEAR 尚未就绪，请执行 models 和 verify。'; fi
    if [[ "$COMPONENTS" != clear ]]; then [[ -f "$PROPAINTER_DIR/models.ready" && -f "$PROPAINTER_DIR/runtime.ready" ]] || fail 'ProPainter 尚未就绪，请执行 models 和 verify。'; fi
    if [[ $WITH_LTX == 1 ]]; then
        [[ -x "$LTX_DIR/.venv/bin/python" && ( -f "$LTX_DIR/.install/runtime.complete" || -f "$LTX_DIR/.install/test.complete" ) ]] || fail 'LTX 尚未就绪，请执行 source、env、models 和 verify。'
    fi
    helper start
    if [[ $WITH_API == 1 ]]; then step_api_start; else save_config; fi
}

step_api_start() {
    acquire_install_lock
    ((API_PORT != PORT)) || fail '网页与 API 需要使用不同端口。'
    [[ -x "$VENV" && -x "$LITE_PYTHON" ]] || fail '缺少推理环境，请执行 env --with-api --components both。'
    [[ -f "$STATE/lite.ready" && -f "$ROOT/CLEAR-web/runtime.ready" && -f "$ROOT/CLEAR-web/models.ready" && -f "$PROPAINTER_DIR/runtime.ready" && -f "$PROPAINTER_DIR/models.ready" ]] || fail '三档 API 需要先运行 models --with-api 和 verify --with-api。'
    api_dependencies
    api_service start
    WITH_API=1
    COMPONENTS=both
    save_config
}

step_stop() {
    acquire_install_lock
    if [[ $WITH_API == 1 || -f "$STATE/api/settings.json" ]]; then api_service stop; fi
    helper stop
}

step_status() {
    helper status
    if [[ $WITH_API == 1 || -f "$STATE/api/settings.json" ]]; then api_service status; fi
}

step_update() {
    log '升级项目源码，保留模型、任务、本地配置与环境'
    acquire_install_lock
    [[ -x "$VENV" ]] || fail '升级前需要已有 Python 环境，请先执行 env。'
    boot_python
    local optional=(--update-ref "$UPDATE_REF" --wait-idle "$WAIT_IDLE") archive=$UPDATE_ARCHIVE
    [[ $OFFLINE == 0 ]] || optional+=(--offline)
    if [[ -n "$UPDATE_ARCHIVE" ]]; then
        optional+=(--archive "$UPDATE_ARCHIVE")
    elif [[ ! -e "$ROOT/.git" ]]; then
        [[ $OFFLINE == 0 ]] || fail '离线升级归档安装需要 --update-archive PATH。'
        download_archive https://github.com/xaiojie/Removesubtitles/archive/refs/heads/main.tar.gz "$STATE/update-source.tar.gz"
        archive="$STATE/update-source.tar.gz"
        optional+=(--archive "$archive")
    fi
    # A freshly downloaded installer must also replace a legacy helper.
    local capability=
    if [[ -f "$SCRIPT_DIR/deploy_helper.py" ]]; then
        capability=$("$BOOT_PYTHON" "$SCRIPT_DIR/deploy_helper.py" capabilities --root "$ROOT" 2>/dev/null) || capability=
    fi
    if [[ "$capability" == update-v1 ]]; then
        HELPER_OVERRIDE="$SCRIPT_DIR/deploy_helper.py"
    elif [[ -n "$archive" ]]; then
        boot_python
        "$BOOT_PYTHON" - "$archive" "$STATE/update-helper.py" <<'PY'
import pathlib
import sys
import tarfile

with tarfile.open(sys.argv[1], 'r:gz') as archive:
    matches = [member for member in archive.getmembers()
               if member.isfile() and not member.issym() and not member.islnk()
               and pathlib.PurePosixPath(member.name).parts[-2:] == ('scripts', 'deploy_helper.py')
               and '..' not in pathlib.PurePosixPath(member.name).parts]
    if len(matches) != 1:
        raise SystemExit('升级源码需要唯一的 scripts/deploy_helper.py。')
    data = archive.extractfile(matches[0]).read()
    compile(data, 'deploy_helper.py', 'exec')
    pathlib.Path(sys.argv[2]).write_bytes(data)
PY
        HELPER_OVERRIDE="$STATE/update-helper.py"
    else
        [[ $OFFLINE == 0 ]] || fail '离线 Git 升级需要将新版 deploy_helper.py 与当前部署脚本放在同一目录。'
        download_archive https://raw.githubusercontent.com/xaiojie/Removesubtitles/main/scripts/deploy_helper.py "$STATE/update-helper.py"
        HELPER_OVERRIDE="$STATE/update-helper.py"
    fi
    helper update ${optional[@]+"${optional[@]}"}
    unset HELPER_OVERRIDE
    save_config
}

step_all() {
    step_check
    if [[ $YES == 0 ]]; then
        read -r -p '开始安装所选组件？[Y/n] ' answer
        [[ "$answer" != n && "$answer" != N ]] || return 0
    fi
    step_system
    step_source
    step_env
    step_models
    step_verify
    step_start
}

menu() {
    [[ -t 0 ]] || { usage; bad_option '非交互环境请指定命令，例如 all --yes'; }
    while true; do
        printf '\nRemoveSubtitles 部署菜单\n目录 %s\n组件 %s  网络 %s  端口 %s\n' "$ROOT" "$COMPONENTS" "$NETWORK" "$PORT"
        printf '1 完整部署\n2 检查配置\n3 安装系统依赖\n4 准备源码\n5 安装 Python/CUDA 环境\n6 下载模型\n7 校验模型与 CUDA\n8 启动服务\n9 状态\n10 停止服务\n11 修改组件、网络和访问地址\n12 升级已有安装\n13 部署网页和三档 PHP API（Lite / Pro / Max）\n0 退出\n'
        read -r -p '请选择 ' choice
        case "$choice" in
            1) step_all ;; 2) step_check ;; 3) step_system ;; 4) step_source ;;
            5) step_env ;; 6) step_models ;; 7) step_verify ;; 8) step_start ;;
            9) step_status ;; 10) step_stop ;;
            13) WITH_API=1; COMPONENTS=both; ((API_PORT != PORT)) || fail '网页与 API 端口重复，请用 --api-port 指定其他端口。'; step_all ;;
            12) step_update ;;
            11)
                read -r -p '组件 clear/propainter/both [both] ' COMPONENTS
                COMPONENTS=${COMPONENTS:-both}
                case "$COMPONENTS" in clear|propainter|both) ;; *) fail '组件值无效。' ;; esac
                if [[ $COMPONENTS != both ]]; then WITH_API=0; fi
                read -r -p '网络 auto/cn/global [auto] ' NETWORK
                NETWORK=${NETWORK:-auto}
                case "$NETWORK" in auto|cn|global) ;; *) fail '网络值无效。' ;; esac
                PIP_URL= TORCH_URL= HF_ENDPOINT=
                read -r -p '公网 IP 或域名，留空仅本机访问 ' PUBLIC_HOST
                [[ "$PUBLIC_HOST" != *[[:space:]/]* ]] || fail '请输入 IP 或域名。'
                BIND=127.0.0.1
                [[ -z "$PUBLIC_HOST" ]] || BIND=0.0.0.0
                save_config ;;
            0) return ;; *) printf '请选择菜单中的编号。\n' ;;
        esac
    done
}

case "$COMMAND" in
    menu) menu ;; all) step_all ;; check) step_check ;; system) step_system ;;
    source) step_source ;; env) step_env ;; models) step_models ;; verify) step_verify ;; update) step_update ;;
    start) step_start ;; stop) step_stop ;; restart) step_stop; step_start ;; status) step_status ;;
    api-start) step_api_start ;; api-stop) acquire_install_lock; api_service stop ;; api-status) api_service status ;;
esac
