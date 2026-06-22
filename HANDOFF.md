# 换机续作交接文档

更新时间：2026-06-22  
项目路径：`C:\coding\AIGC-3DGS-Fusion`

## 1. 当前项目状态

本项目用于完成 `HW3_深度学习与空间智能.pdf` 的第一个 3D / AIGC 3D 方向任务。当前实现采用教学版 3D Gaussian Splatting：用 PyTorch 学习一组 3D 高斯点，通过相机投影和 2D splatting 重建多视角图像。

当前已完成：

- 代码：`src/aigc3dgs/`
- 环境说明：`requirements.txt`、`environment.yml`、`pyproject.toml`
- README：`README.md`
- 实验报告：`report.md`
- 小型样例数据：`data/toy_scene/`
- 模型权重：`weights/`
- Slurm 脚本：`slurm/train_toy_gpu.sbatch`、`slurm/train_toy_gpu_aws.sbatch`
- Slurm 提交说明：`scripts/submit_slurm.md`

当前 Git 状态：

- 工作区已清理干净。
- 最近提交：
  - `654942a chore: 补充AWS训练提交脚本`
  - `7bfb0fa feat: 初始化3DGS作业实现`
- 尚未配置远程仓库，`git remote -v` 当前为空。

## 2. 数据与权重

已提交的小型样例数据：

- `data/toy_scene/transforms.json`
- `data/toy_scene/images/view_000.png` 到 `view_019.png`

已提交的模型权重：

- `weights/toy_3dgs/`：早期 CPU 轻量训练权重。
- `weights/toy_3dgs_gpu/`：GPU 版 800 steps 权重。
- `weights/toy_3dgs_gpu_quality/`：GPU 版 3000 steps、128 高斯、128x128 分辨率权重。
- `weights/aws_ec2_toy_3dgs/`：AWS EC2 `g4dn.xlarge` 训练权重，3000 steps、128 高斯、128x128 分辨率，final loss `0.01350455`，device `cuda`。
- `weights/aws_ec2_toy_3dgs_quality/`：AWS EC2 `g4dn.xlarge` 质量训练权重，10000 steps、512 高斯、128x128 分辨率，final loss `0.00925142`，device `cuda`，当前推荐作为最终提交权重。
- `weights/aws_ec2_toy_3dgs_q2_20k_1024/`：AWS EC2 质量迭代最佳权重，20000 steps、1024 高斯、128x128 分辨率，final loss `0.00914205`，device `cuda`，当前最终推荐成果。
- `weights/aws_ec2_toy_3dgs_q3_40k_2048/`：AWS EC2 质量迭代对照权重，40000 steps、2048 高斯、128x128 分辨率，final loss `0.01122936`，相比 q2 退化，因此不作为最终推荐。

官方 Mip-NeRF 360 主包曾在本机下载到：

- `data/360_v2.zip`

该文件约 12.5GB，已被 `.gitignore` 忽略，不在 Git 提交中。换电脑后如果需要官方数据，重新运行：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.download_dataset --out data --scene garden --no-extract
```

压缩包中已确认推荐场景图片数量：

- `garden`：185 张
- `bicycle`：194 张
- `counter`：240 张

## 3. 新电脑恢复步骤

克隆或复制项目后，先进入仓库根目录。

Windows 推荐使用 Python 3.11：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

如果新电脑有 NVIDIA GPU，安装 CUDA 12.6 版 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

验证 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

如果没有 GPU，也可以使用 CPU 版 PyTorch，但训练会慢一些。

## 4. 常用命令

生成样例数据：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 128
```

本机 GPU 训练推荐命令：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.train --data data/toy_scene --out runs/final_gpu_train --steps 3000 --gaussians 128 --image-size 128
```

测试渲染：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.render --checkpoint runs/final_gpu_train/checkpoints/model_final.pth --data data/toy_scene --out runs/final_gpu_train/test_renders --image-size 128
```

使用已提交的高质量 GPU 权重重新渲染：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.render --checkpoint weights/toy_3dgs_gpu_quality/model_final.pth --data data/toy_scene --out runs/handoff_check_renders --image-size 128
```

## 5. 本机训练实测

原电脑 GPU：

- GPU：Quadro T1000
- 显存：4GB
- PyTorch：`2.12.1+cu126`
- CUDA build：`12.6`

实测训练：

- `96` 高斯、`96x96`、`800 steps`：约 `119 step/s`，墙钟约 13 秒，final loss `0.00921393`。
- `128` 高斯、`128x128`、`3000 steps`：约 `90.6 step/s`，墙钟约 44 秒，final loss `0.01173052`。

如果在本机跑当前教学版官方单场景降采样实验，粗略估计：

- 低配试跑：几分钟到十几分钟。
- 中等配置：`256x256`、几千高斯、`10k-30k steps`，约 30 分钟到 2 小时。

如果换成原版 3DGS 全量 Mip-NeRF 360，单场景估计 2 到 6 小时以上，并且 4GB 显存很可能成为主要风险。

## 6. Slurm / AWS 提交

常规 GPU 分区：

```bash
sbatch slurm/train_toy_gpu.sbatch
```

AWS 分区：

```bash
sbatch slurm/train_toy_gpu_aws.sbatch
```

注意：`aws` 分区通常会产生额外费用；只有明确接受费用时才使用。脚本默认账号和 QOS：

- account：`gpo-ifv7xx`
- qos：`normal`

如果集群提示 account/partition 不匹配，先查：

```bash
sacctmgr show assoc user=$USER format=User,Account,Partition,QOS
```

AWS CLI 状态：

- 原电脑已安装 AWS CLI v2.35.9。
- 已验证 `aigc-3dgs` profile。
- AWS Account：`553432479592`
- IAM User ARN：`arn:aws:iam::553432479592:user/grafana`
- Region：`ap-northeast-1`

继续启动 EC2 训练前，还需确认：

- GPU 实例配额。
- Deep Learning AMI ID。
- EC2 key pair。
- 安全组。
- 子网。

## 7. 远程仓库待办

当前远程 public GitHub 仓库：

- https://github.com/zh23jemu/AIGC-3DGS-Fusion

换电脑后可直接克隆：

```bash
git clone https://github.com/zh23jemu/AIGC-3DGS-Fusion.git
```

## 8. 关键风险

- 当前实现是教学版 3DGS，不是原版 CUDA rasterizer 3DGS。
- 官方 Mip-NeRF 360 数据集很大，主包约 12.5GB，没有纳入 Git。
- 原版 3DGS 全量训练对显存要求更高，4GB 显存不稳。
- `runs/`、`.venv/`、`data/360_v2.zip` 已被忽略，不会随 Git 迁移。
- 如果换电脑后要继续训练，优先重新创建 `.venv`，不要复制旧 `.venv`。
