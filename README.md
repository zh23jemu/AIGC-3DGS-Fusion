# AIGC-3DGS-Fusion

本仓库完成 `HW3_深度学习与空间智能.pdf` 的第一个 3D / AIGC 3D 任务。项目实现一个本机可运行的教学版 3D Gaussian Splatting：从多视角图片学习一组 3D 高斯点，训练后导出模型权重并渲染测试视角。

## 环境配置

建议使用 Python 3.11 或 3.12，并始终通过项目本地 `.venv` 运行。

Windows 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本机已验证 GPU 环境可用：`torch 2.12.1+cu126`，CUDA build 为 `12.6`，显卡为 `Quadro T1000`。如果需要重新安装 CUDA 12.x 版本 PyTorch，可使用：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 数据准备

作业 PDF 推荐 Mip-NeRF 360 数据集，官方页面为：

- https://jonbarron.info/mipnerf360/

可下载官方主包，`garden`、`bicycle`、`counter` 均在主包中。本机已完成主包下载并验证压缩包内包含这三个场景，文件路径为 `data/360_v2.zip`，大小约为 `12.5GB`。完整解压需要较长时间和更多磁盘空间：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.download_dataset --out data --scene garden
```

如果只想下载不解压，可使用：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.download_dataset --out data --scene garden --no-extract
```

官方数据集体积较大。若本机资源不足，可以生成一个小型多视角样例数据，用于本机快速训练和测试：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 96
```

## 训练命令

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.train --data data/toy_scene --out runs/toy_3dgs --steps 800 --gaussians 96 --image-size 96
```

本机已完成两组 GPU 轻量验证训练：

- `800` steps、`96` 个高斯、`96x96` 分辨率：训练阶段约 `119 step/s`，命令墙钟约 `13` 秒，最终 loss 为 `0.00921393`。
- `3000` steps、`128` 个高斯、`128x128` 分辨率：训练阶段约 `90.6 step/s`，命令墙钟约 `44` 秒，最终 loss 为 `0.01173052`。

训练结束后会生成：

- `runs/toy_3dgs/checkpoints/model_final.pth`
- `runs/toy_3dgs/checkpoints/model_final.ply`
- `runs/toy_3dgs/renders/` 中的训练视角渲染图

为了便于提交，本机也整理了一份权重副本：

- `weights/toy_3dgs/model_final.pth`
- `weights/toy_3dgs/model_final.ply`
- `weights/toy_3dgs_gpu/model_final.pth`
- `weights/toy_3dgs_gpu/model_final.ply`
- `weights/toy_3dgs_gpu_quality/model_final.pth`
- `weights/toy_3dgs_gpu_quality/model_final.ply`
- `weights/aws_ec2_toy_3dgs/model_final.pth`
- `weights/aws_ec2_toy_3dgs/model_final.ply`
- `weights/aws_ec2_toy_3dgs_quality/model_final.pth`
- `weights/aws_ec2_toy_3dgs_quality/model_final.ply`
- `weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth`
- `weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.ply`
- `weights/aws_ec2_toy_3dgs_q3_40k_2048/model_final.pth`
- `weights/aws_ec2_toy_3dgs_q3_40k_2048/model_final.ply`

AWS EC2 `g4dn.xlarge` 训练结果已下载到 `weights/aws_ec2_toy_3dgs/`：`3000` steps、`128` 个高斯、`128x128` 分辨率，最终 loss 为 `0.01350455`，训练设备为 `Tesla T4`。

AWS EC2 质量训练结果已下载到 `weights/aws_ec2_toy_3dgs_quality/`：`10000` steps、`512` 个高斯、`128x128` 分辨率，最终 loss 为 `0.00925142`，训练设备为 `Tesla T4`。该版本是当前推荐提交权重。

继续迭代后，当前最佳权重为 `weights/aws_ec2_toy_3dgs_q2_20k_1024/`：`20000` steps、`1024` 个高斯、`128x128` 分辨率，最终 loss 为 `0.00914205`。继续加大到 `40000` steps、`2048` 个高斯后，loss 退化为 `0.01122936`，因此停止继续加码训练，推荐使用 q2 权重作为最终成果。

AWS user-data 脚本默认质量训练档为 `10000` steps、`512` 个高斯、`128x128` 分辨率，输出目录为 `runs/aws_ec2_toy_3dgs_quality`。可通过环境变量 `STEPS`、`GAUSSIANS`、`IMAGE_SIZE`、`RUN_NAME` 覆盖。

## 测试命令

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.render --checkpoint runs/toy_3dgs/checkpoints/model_final.pth --data data/toy_scene --out runs/toy_3dgs/test_renders --image-size 96
```

## Slurm / 集群训练

仓库提供了轻量训练 Slurm 脚本：

```bash
sbatch slurm/train_toy_gpu.sbatch
```

默认分区为 `gpu`，账号为 `gpo-ifv7xx`，QOS 为 `normal`。如果需要临时覆盖分区，可使用：

```bash
sbatch --partition=gpuHz slurm/train_toy_gpu.sbatch
```

不建议默认使用 `aws` 分区，除非明确接受额外费用。

如果明确要提交到 `aws` 分区：

```bash
sbatch slurm/train_toy_gpu_aws.sbatch
```

## AWS EC2 公有云训练

如果使用 AWS EC2 GPU 实例训练，请参考：

- `scripts/aws_ec2_train.md`
- `scripts/aws_train_user_data.sh`

不要把 AWS Access Key 或 Secret Access Key 写入仓库。建议使用 AWS CLI profile 或 EC2 IAM Role。

## 目录说明

- `src/aigc3dgs/`：核心代码。
- `slurm/`：集群训练提交脚本。
- `scripts/`：辅助说明和提交提示。
- `data/`：本地数据目录，不纳入版本管理。
- `runs/`：训练输出目录，不纳入版本管理。
- `weights/`：整理后的模型权重目录，小型作业交付权重纳入版本管理。
- `report.md`：实验报告。
