#!/bin/bash
#
# EC2 GPU 实例启动后执行的质量迭代训练脚本。
# 该脚本不包含任何 AWS 密钥；权限通过本机 AWS CLI profile 或 EC2 IAM Role 管理。

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/zh23jemu/AIGC-3DGS-Fusion.git}"
WORKDIR="${WORKDIR:-/home/ubuntu/AIGC-3DGS-Fusion}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
DATA_DIR="${DATA_DIR:-data/toy_scene}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
VIEWS="${VIEWS:-20}"

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

# 固定 CUDA 12.6 兼容 PyTorch，避免 pip 自动解析到 CUDA 13 wheel。
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

# 只生成一次数据，后续不同质量档共用同一组视角，保证 loss 和渲染可比较。
.venv/bin/python -m aigc3dgs.prepare_toy_dataset --out "${DATA_DIR}" --views "${VIEWS}" --size "${IMAGE_SIZE}"

run_quality_train() {
  local run_name="$1"
  local steps="$2"
  local gaussians="$3"
  local output_dir="runs/${run_name}"

  echo "===== quality train: ${run_name}, steps=${steps}, gaussians=${gaussians}, image_size=${IMAGE_SIZE} ====="
  .venv/bin/python -m aigc3dgs.train \
    --data "${DATA_DIR}" \
    --out "${output_dir}" \
    --steps "${steps}" \
    --gaussians "${gaussians}" \
    --image-size "${IMAGE_SIZE}"
  .venv/bin/python -m aigc3dgs.render \
    --checkpoint "${output_dir}/checkpoints/model_final.pth" \
    --data "${DATA_DIR}" \
    --out "${output_dir}/test_renders" \
    --image-size "${IMAGE_SIZE}"
}

# 迭代策略：
# - q2 在当前推荐配置基础上翻倍高斯和步数，观察是否仍有明显收益。
# - q3 再次加大容量和步数；如果 q3 相对 q2 提升很小，则认为继续堆训练成本收益不明显。
run_quality_train "aws_ec2_toy_3dgs_q2_20k_1024" 20000 1024
run_quality_train "aws_ec2_toy_3dgs_q3_40k_2048" 40000 2048
