#!/usr/bin/env bash
# 在 AWS EC2 GPU 上补齐 HW3 题目一完整链路中的 B/C 资产生成。
#
# 该脚本设计为 user-data 一次性任务：
# 1. 使用 Deep Learning AMI 自带 Docker 与 NVIDIA runtime。
# 2. 克隆 threestudio 和本项目仓库。
# 3. 运行 threestudio DreamFusion-SD 生成文本到 3D 的物体 B。
# 4. 运行 threestudio Stable Zero123 生成单图到 3D 的物体 C。
# 5. 导出 mesh / 测试视频 / 日志，并尝试同步到 S3。
# 6. 任务结束后自动关机，避免实例空转。

set -euxo pipefail

PROJECT_REPO="${PROJECT_REPO:-https://github.com/zh23jemu/AIGC-3DGS-Fusion.git}"
S3_BUCKET="${S3_BUCKET:-}"
RUN_NAME="${RUN_NAME:-hw3_full_assets_$(date +%Y%m%d_%H%M%S)}"
B_PROMPT="${B_PROMPT:-a small blue ceramic rabbit figurine, smooth surface, studio lighting}"
B_STEPS="${B_STEPS:-800}"
C_STEPS="${C_STEPS:-600}"

# g6.xlarge 使用 NVIDIA L4（Ada Lovelace，Compute Capability 8.9）。
# threestudio 依赖的 nerfacc / tiny-cuda-nn / nvdiffrast 都会编译 CUDA 扩展；
# 如果不限制架构，pip 构建阶段会为多代 GPU 生成代码，耗时远超实际训练。
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-89}"

WORK_ROOT="/opt/dlami/nvme/${RUN_NAME}"
if [ ! -d /opt/dlami/nvme ]; then
  WORK_ROOT="/opt/${RUN_NAME}"
fi
LOG_DIR="/var/log/aigc-hw3-full"
RESULT_DIR="${WORK_ROOT}/results"

mkdir -p "${WORK_ROOT}" "${LOG_DIR}" "${RESULT_DIR}"
exec > >(tee -a "${LOG_DIR}/cloud-init.log") 2>&1

echo "===== AIGC HW3 full asset generation ====="
date -Is
nvidia-smi || true
docker --version

cd "${WORK_ROOT}"
git clone --depth 1 "${PROJECT_REPO}" project
git clone --depth 1 https://github.com/threestudio-project/threestudio.git threestudio

# threestudio 官方 Dockerfile 中的 nerfacc / tiny-cuda-nn 依赖会在 PEP517
# build isolation 环境里构建，隔离环境看不到前一步安装的 torch，容易触发
# `ModuleNotFoundError: No module named 'torch'`。这里在构建前做最小 patch：
# 禁用 build isolation，并让 requirements 中的同类依赖复用已安装结果。
python3 - <<'PY'
from pathlib import Path

dockerfile = Path("threestudio/docker/Dockerfile")
text = dockerfile.read_text(encoding="utf-8")
text = text.replace(
    "FROM nvidia/cuda:11.8.0-devel-ubuntu22.04",
    "FROM nvidia/cuda:11.8.0-devel-ubuntu22.04\n"
    "ENV TORCH_CUDA_ARCH_LIST=8.9\n"
    "ENV TCNN_CUDA_ARCHITECTURES=89",
)
text = text.replace(
    "RUN pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2",
    "RUN pip install --no-build-isolation git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2",
)
text = text.replace(
    "RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch",
    "RUN pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch",
)
text = text.replace(
    "COPY requirements.txt /tmp\nRUN cd /tmp && pip install -r requirements.txt",
    "RUN pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git\nCOPY requirements.txt /tmp\nRUN cd /tmp && pip install -r requirements.txt",
)
dockerfile.write_text(text, encoding="utf-8")

req = Path("threestudio/requirements.txt")
lines = req.read_text(encoding="utf-8").splitlines()
patched = []
for line in lines:
    if line.startswith("git+https://github.com/KAIR-BAIR/nerfacc.git"):
        patched.append("# installed in Dockerfile with --no-build-isolation: " + line)
    elif line.startswith("git+https://github.com/NVlabs/tiny-cuda-nn/"):
        patched.append("# installed in Dockerfile with --no-build-isolation: " + line)
    elif line.startswith("git+https://github.com/NVlabs/nvdiffrast.git"):
        patched.append("# installed in Dockerfile with --no-build-isolation: " + line)
    else:
        patched.append(line)
req.write_text("\n".join(patched) + "\n", encoding="utf-8")
print("patched threestudio docker dependencies")
PY

# 准备单图到 3D 的输入。宿主机 AMI 默认未安装 Pillow，因此这里只复制原图；
# RGBA 去背景处理放到 threestudio 容器内部执行。
mkdir -p threestudio/load/images
cp project/data/toy_scene/images/view_000.png threestudio/load/images/hw3_object_c_source.png

cd threestudio

# 构建官方 Dockerfile。threestudio 依赖 tiny-cuda-nn、nvdiffrast 等 CUDA 扩展，
# 使用官方 Dockerfile 比在宿主机直接安装更稳定。
docker build -t aigc-threestudio:hw3 -f docker/Dockerfile .

cat > "${WORK_ROOT}/run_inside_container.sh" <<'EOS'
#!/usr/bin/env bash
set -euxo pipefail
cd /workspace/threestudio

mkdir -p load/zero123 outputs

echo "===== PyTorch check ====="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

echo "===== Prepare RGBA input for Zero123 ====="
python - <<'PY'
from pathlib import Path
from PIL import Image

src = Path("load/images/hw3_object_c_source.png")
dst = Path("load/images/hw3_object_c_rgba.png")
img = Image.open(src).convert("RGBA").resize((256, 256))
pixels = []
for r, g, b, a in img.getdata():
    alpha = 0 if (r < 18 and g < 18 and b < 18) else 255
    pixels.append((r, g, b, alpha))
img.putdata(pixels)
img.save(dst)
print(dst, dst.stat().st_size)
PY

echo "===== Object B: threestudio DreamFusion-SD text-to-3D ====="
python launch.py \
  --config configs/dreamfusion-sd.yaml \
  --train --gpu 0 \
  system.prompt_processor.prompt="${B_PROMPT}" \
  trainer.max_steps="${B_STEPS}" \
  data.width=64 data.height=64 data.batch_size=1 \
  system.freq.guidance_eval=0 \
  2>&1 | tee /workspace/results/object_b_threestudio_train.log

B_TRIAL="$(find outputs -path '*ckpts/last.ckpt' -print | sort | tail -n 1 | sed 's#/ckpts/last.ckpt##')"
echo "${B_TRIAL}" > /workspace/results/object_b_trial.txt
python launch.py \
  --config "${B_TRIAL}/configs/parsed.yaml" \
  --test --gpu 0 \
  resume="${B_TRIAL}/ckpts/last.ckpt" \
  2>&1 | tee /workspace/results/object_b_threestudio_test.log || true
python launch.py \
  --config "${B_TRIAL}/configs/parsed.yaml" \
  --export --gpu 0 \
  resume="${B_TRIAL}/ckpts/last.ckpt" \
  system.exporter_type=mesh-exporter system.exporter.fmt=obj \
  2>&1 | tee /workspace/results/object_b_threestudio_export.log || true

echo "===== Object C: Stable Zero123 single-image-to-3D ====="
python - <<'PY'
from pathlib import Path
import urllib.request

out = Path("load/zero123/stable_zero123.ckpt")
out.parent.mkdir(parents=True, exist_ok=True)
if not out.exists():
    # Hugging Face 的 resolve URL 当前无需 token 时可直接下载；若未来需要授权，
    # 日志会明确失败，后续可改为预先上传 checkpoint 到 S3。
    url = "https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt"
    print("downloading", url)
    urllib.request.urlretrieve(url, out)
print(out, out.stat().st_size)
PY

python launch.py \
  --config configs/stable-zero123.yaml \
  --train --gpu 0 \
  data.image_path=./load/images/hw3_object_c_rgba.png \
  trainer.max_steps="${C_STEPS}" \
  2>&1 | tee /workspace/results/object_c_zero123_train.log

C_TRIAL="$(find outputs -path '*ckpts/last.ckpt' -print | sort | tail -n 1 | sed 's#/ckpts/last.ckpt##')"
echo "${C_TRIAL}" > /workspace/results/object_c_trial.txt
python launch.py \
  --config "${C_TRIAL}/configs/parsed.yaml" \
  --test --gpu 0 \
  resume="${C_TRIAL}/ckpts/last.ckpt" \
  2>&1 | tee /workspace/results/object_c_zero123_test.log || true
python launch.py \
  --config "${C_TRIAL}/configs/parsed.yaml" \
  --export --gpu 0 \
  resume="${C_TRIAL}/ckpts/last.ckpt" \
  system.exporter_type=mesh-exporter system.exporter.fmt=obj \
  2>&1 | tee /workspace/results/object_c_zero123_export.log || true

echo "===== Collect artifacts ====="
mkdir -p /workspace/results/object_b /workspace/results/object_c
cp -r "${B_TRIAL}" /workspace/results/object_b/trial || true
cp -r "${C_TRIAL}" /workspace/results/object_c/trial || true
cp load/images/hw3_object_c_rgba.png /workspace/results/object_c/input_rgba.png || true
find /workspace/results -maxdepth 5 -type f | sort > /workspace/results/artifact_index.txt
EOS

chmod +x "${WORK_ROOT}/run_inside_container.sh"

docker run --gpus all --ipc=host --shm-size=16g \
  -e B_PROMPT="${B_PROMPT}" -e B_STEPS="${B_STEPS}" -e C_STEPS="${C_STEPS}" \
  -v "${WORK_ROOT}/threestudio:/workspace/threestudio" \
  -v "${RESULT_DIR}:/workspace/results" \
  -v "${WORK_ROOT}/run_inside_container.sh:/workspace/run_inside_container.sh:ro" \
  aigc-threestudio:hw3 \
  bash /workspace/run_inside_container.sh

cd "${WORK_ROOT}"
tar -czf "${RESULT_DIR}/${RUN_NAME}.tar.gz" -C "${RESULT_DIR}" .

if [ -n "${S3_BUCKET}" ]; then
  aws s3 sync "${RESULT_DIR}" "s3://${S3_BUCKET}/aigc-3dgs-fusion/${RUN_NAME}/" || true
fi

echo "===== Done ====="
find "${RESULT_DIR}" -maxdepth 3 -type f | sort
date -Is

shutdown -h now
