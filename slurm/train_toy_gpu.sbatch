#!/bin/bash
#
# 教学版 3DGS 轻量训练 Slurm 脚本。
# 默认使用普通 GPU 分区，避免使用会产生额外费用的 aws 分区。
# 如需临时改分区，可执行：
#   sbatch --partition=gpuHz slurm/train_toy_gpu.sbatch

#SBATCH --job-name=aigc3dgs-toy
#SBATCH --partition=gpu
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

# 始终切换到仓库根目录，保证相对路径和 README 中的命令一致。
cd "$(dirname "$0")/.."

# 日志目录属于训练运行态输出，可重复生成。
mkdir -p runs/slurm

# 如果集群上还没有本地虚拟环境，则创建并安装依赖。
# PyTorch GPU wheel 使用 CUDA 12.6，兼容当前约定的 CUDA 12.x 运行环境。
if [ ! -x ".venv/bin/python" ]; then
  python3.11 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e .
  .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

# 生成小型多视角样例数据；如果已存在，脚本会覆盖同名样例图，确保训练输入完整。
.venv/bin/python -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 128

# 运行轻量 GPU 训练。该配置已在本机 Quadro T1000 上验证可完成。
.venv/bin/python -m aigc3dgs.train \
  --data data/toy_scene \
  --out runs/slurm_toy_3dgs \
  --steps 3000 \
  --gaussians 128 \
  --image-size 128

# 渲染测试视角，作为任务完成后的快速可视化检查。
.venv/bin/python -m aigc3dgs.render \
  --checkpoint runs/slurm_toy_3dgs/checkpoints/model_final.pth \
  --data data/toy_scene \
  --out runs/slurm_toy_3dgs/test_renders \
  --image-size 128
