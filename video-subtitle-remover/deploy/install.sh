#!/usr/bin/env bash
# 参考部署脚本：Linux + NVIDIA 上把 Web 界面跑起来（systemd 常驻）
#
# 准备模型与两个独立环境，再按配置注册并启动 systemd 服务。
# 算法环境里的 torch / paddlepaddle 需要按你的显卡和 CUDA 版本选择，
# 脚本不替你决定，装完会打印对应命令。
#
# 用法：
#   bash deploy/install.sh              # 默认端口 8800
#   VSR_PORT=8800 bash deploy/install.sh
#   VSR_INSTALL_SERVICE=0 bash deploy/install.sh   # 只装环境，不注册 systemd
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_VENV="${VSR_WEB_VENV:-$REPO_DIR/.venv-web}"
ALGO_VENV="${VSR_VENV:-$REPO_DIR/.venv}"
DATA_DIR="${VSR_DATA_DIR:-$REPO_DIR/vsr-data}"
PORT="${VSR_PORT:-8800}"
PIP_INDEX="${VSR_PIP_INDEX:-https://pypi.org/simple}"
MODEL_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${VSR_MODEL_DIR:-$REPO_DIR/backend/models}")
UNIT=/etc/systemd/system/vsr-web.service

echo "==> 仓库目录: $REPO_DIR"
echo "==> Web 环境: $WEB_VENV"
echo "==> 算法环境: $ALGO_VENV"
echo "==> 数据目录: $DATA_DIR (端口 $PORT)"

command -v "${VSR_FFMPEG:-ffmpeg}" >/dev/null || { echo '请先安装系统 FFmpeg，或设置 VSR_FFMPEG。' >&2; exit 1; }
python3 "$REPO_DIR/deploy/download_models.py" --model-dir "$MODEL_DIR" --mode "${VSR_MODEL_MODE:-all}" --detector "${VSR_DETECTOR:-server}"
python3 --version
mkdir -p "$DATA_DIR"

echo "==> [1/4] 创建 Web 环境并安装依赖"
python3 -m venv "$WEB_VENV"
"$WEB_VENV/bin/pip" install -q --upgrade pip -i "$PIP_INDEX"
"$WEB_VENV/bin/pip" install -q -i "$PIP_INDEX" -r "$REPO_DIR/requirements-web.txt"

echo "==> [2/4] 创建算法环境并安装依赖（不含 torch）"
if [ ! -x "$ALGO_VENV/bin/python" ]; then
  python3 -m venv "$ALGO_VENV"
fi
"$ALGO_VENV/bin/pip" install -q --upgrade pip -i "$PIP_INDEX"
"$ALGO_VENV/bin/pip" install -q -i "$PIP_INDEX" -r "$REPO_DIR/requirements-headless.txt"

echo "==> [3/4] 还需要你手动安装 torch（按显卡选择）："
cat <<EOF

  # 50 系（RTX 50xx，CUDA 12.8）
  $ALGO_VENV/bin/pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  # 40 系（CUDA 12.6）
  $ALGO_VENV/bin/pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
  # 30/20/10 系（CUDA 11.8）
  $ALGO_VENV/bin/pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118

  校验：$ALGO_VENV/bin/python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

EOF

if [ "${VSR_INSTALL_SERVICE:-1}" != "1" ]; then
  echo "==> 跳过 systemd 注册。手动启动："
  echo "    VSR_CONFIG_MODE=headless VSR_DATA_DIR=$DATA_DIR VSR_PYTHON=$ALGO_VENV/bin/python \\"
  echo "      $WEB_VENV/bin/python $REPO_DIR/web/serve.py"
  exit 0
fi

echo "==> [4/4] 写入 systemd 单元并启动"
sed -e "s#__ALGO_VENV__#$ALGO_VENV#g" -e "s#__WEB_VENV__#$WEB_VENV#g" \
    -e "s#__MODEL_DIR__#$MODEL_DIR#g" -e "s#__FFMPEG__#$(command -v "${VSR_FFMPEG:-ffmpeg}")#g" \
    -e "s#__REPO_DIR__#$REPO_DIR#g" -e "s#__DATA_DIR__#$DATA_DIR#g" \
    "$REPO_DIR/deploy/vsr-web.service" > "$UNIT"
sed -i "s#VSR_PORTS=8800#VSR_PORTS=$PORT#" "$UNIT"
systemctl daemon-reload
systemctl enable vsr-web
systemctl restart vsr-web
sleep 2
systemctl --no-pager --lines=5 status vsr-web || true

echo
echo "==> 完成：http://<服务器IP>:$PORT/"
echo "    日志：journalctl -u vsr-web -f   或   $DATA_DIR/vsr-web.log"
