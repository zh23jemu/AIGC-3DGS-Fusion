#!/bin/bash
#
# EC2 GPU 实例启动后执行的训练脚本。
# 该脚本不包含任何 AWS 密钥；请通过 IAM Role 或本机 AWS CLI profile 管理权限。

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/zh23jemu/AIGC-3DGS-Fusion.git}"
WORKDIR="${WORKDIR:-/home/ubuntu/AIGC-3DGS-Fusion}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

sudo apt-get update
sudo apt-get install -y git python3.11 python3.11-venv python3-pip

if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only
fi

cd "$WORKDIR"

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

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
