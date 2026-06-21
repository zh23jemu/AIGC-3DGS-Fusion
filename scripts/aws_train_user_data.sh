#!/bin/bash
#
# EC2 GPU 实例启动后执行的训练脚本。
# 该脚本不包含任何 AWS 密钥；请通过 IAM Role 或本机 AWS CLI profile 管理权限。

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/zh23jemu/AIGC-3DGS-Fusion.git}"
WORKDIR="${WORKDIR:-/home/ubuntu/AIGC-3DGS-Fusion}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv python3-pip

if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only
fi

cd "$WORKDIR"

${PYTHON_BIN} -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# 先固定安装 CUDA 12.6 兼容的 PyTorch，再安装项目本身。
# 这样可以避免 `pip install -e .` 根据 `torch>=2.3` 自动解析到更新但不稳定的 CUDA 13 wheel。
.venv/bin/python -m pip install \
  "torch==2.7.1+cu126" \
  "torchvision==0.22.1+cu126" \
  --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install numpy pillow tqdm requests
.venv/bin/python -m pip install --no-deps -e .

.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

.venv/bin/python -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 128
.venv/bin/python -m aigc3dgs.train \
  --data data/toy_scene \
  --out runs/aws_ec2_toy_3dgs \
  --steps 3000 \
  --gaussians 128 \
  --image-size 128
.venv/bin/python -m aigc3dgs.render \
  --checkpoint runs/aws_ec2_toy_3dgs/checkpoints/model_final.pth \
  --data data/toy_scene \
  --out runs/aws_ec2_toy_3dgs/test_renders \
  --image-size 128
