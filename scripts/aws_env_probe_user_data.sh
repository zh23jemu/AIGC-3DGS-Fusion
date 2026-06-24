#!/usr/bin/env bash
# AWS EC2 GPU 环境探针脚本。
# 用途：启动短生命周期实例，记录 CUDA、Python、conda、docker 等基础环境，
# 为后续 threestudio / Zero123 真实训练选择合适安装路径。

set -euxo pipefail

LOG_DIR="/var/log/aigc-hw3-probe"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/cloud-init.log") 2>&1

echo "===== AIGC HW3 GPU environment probe ====="
date -Is
uname -a

echo "===== NVIDIA ====="
if command -v nvidia-smi; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo "===== CUDA toolkit ====="
if command -v nvcc; then
  nvcc --version || true
else
  echo "nvcc not found"
fi

echo "===== Python candidates ====="
for bin in python python3 python3.10 python3.11 python3.12; do
  if command -v "${bin}"; then
    "${bin}" --version || true
  else
    echo "${bin} not found"
  fi
done

echo "===== Conda / mamba ====="
for bin in conda mamba micromamba; do
  if command -v "${bin}"; then
    "${bin}" --version || true
  else
    echo "${bin} not found"
  fi
done

echo "===== Docker ====="
if command -v docker; then
  docker --version || true
  systemctl is-active docker || true
else
  echo "docker not found"
fi

echo "===== PyTorch quick check ====="
if command -v python3; then
  python3 - <<'PY' || true
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
    print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except Exception as exc:
    print("torch import failed:", repr(exc))
PY
fi

echo "===== Disk ====="
df -h

echo "===== Probe complete ====="
date -Is

shutdown -h now
