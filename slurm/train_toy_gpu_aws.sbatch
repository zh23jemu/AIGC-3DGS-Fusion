#!/bin/bash
#
# AWS GPU 分区训练脚本。
# 注意：aws 分区通常会产生额外费用；本脚本仅在用户明确要求使用 aws 时使用。
# 如需改回默认免费/常规 GPU 分区，可执行：
#   sbatch --partition=gpu slurm/train_toy_gpu_aws.sbatch

#SBATCH --job-name=aigc3dgs-aws
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs/slurm

# 如果集群节点还没有项目虚拟环境，则自动创建。
# CUDA wheel 固定使用 cu126，避免安装 CUDA 13.0 PyTorch wheel。
if [ ! -x ".venv/bin/python" ]; then
  python3.11 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e .
  .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

.venv/bin/python -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 128

.venv/bin/python -m aigc3dgs.train \
  --data data/toy_scene \
  --out runs/aws_toy_3dgs \
  --steps 3000 \
  --gaussians 128 \
  --image-size 128

.venv/bin/python -m aigc3dgs.render \
  --checkpoint runs/aws_toy_3dgs/checkpoints/model_final.pth \
  --data data/toy_scene \
  --out runs/aws_toy_3dgs/test_renders \
  --image-size 128
