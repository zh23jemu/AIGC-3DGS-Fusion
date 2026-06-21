# HW3 第一个任务实验报告

## 1. 任务目标

本实验选择作业第一个 3D / AIGC 3D 方向任务中的 3D Gaussian Splatting 路线。目标是从多视角图像中训练一个可微 3D 高斯场景表示，并导出可复现的代码、实验报告和模型权重。

## 2. 数据集

作业 PDF 推荐使用 Mip-NeRF 360 数据集，推荐场景包括 `garden`、`bicycle`、`counter`。官方页面为 https://jonbarron.info/mipnerf360/ 。

本机已完成官方主包 `360_v2.zip` 下载，路径为 `data/360_v2.zip`，文件大小约 `12.5GB`。压缩包结构验证显示其中包含 `garden`、`bicycle`、`counter` 场景。由于完整解压和原版大场景训练耗时较长，本次本机训练验证使用仓库脚本生成的小型多视角样例数据。样例数据包含 20 个围绕同一合成目标的相机视角，每个视角保存 RGB 图像和相机位姿参数。

## 3. 方法

本实验实现教学版 3D Gaussian Splatting。模型由一组可学习 3D 高斯点组成，每个高斯包含空间位置、颜色、不透明度和尺度参数。训练时将 3D 高斯点投影到当前相机视角，在图像平面上生成 2D 高斯权重并进行 alpha 合成，最后用渲染图和真实图之间的重建损失优化模型。

## 4. 网络结构与超参数

- Network Architecture：可学习 3D Gaussian 参数集合，包括位置、颜色、尺度和不透明度。
- Batch Size：每次随机采样 1 个视角。
- Learning Rate：`0.03`。
- Optimizer：Adam。
- Epochs / Iterations：本机验证训练 `250` steps。
- Loss Function：RGB 图像均方误差 MSE，同时加入轻量尺度正则项抑制高斯过度扩张。

## 5. 训练与测试命令

生成样例数据：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 96
```

训练：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.train --data data/toy_scene --out runs/toy_3dgs --steps 800 --gaussians 96 --image-size 96
```

测试渲染：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.render --checkpoint runs/toy_3dgs/checkpoints/model_final.pth --data data/toy_scene --out runs/toy_3dgs/test_renders --image-size 96
```

## 6. 实验结果

本机 GPU 环境已调通，PyTorch 版本为 `2.12.1+cu126`，CUDA build 为 `12.6`，`torch.cuda.is_available()` 为 `True`，显卡为 `Quadro T1000`。本次使用 GPU 完成轻量训练验证，配置如下：

- 训练视角：20
- 轻量验证一：96x96 分辨率，96 个 3D 高斯，800 steps，训练阶段约 `119 step/s`，命令墙钟约 `13` 秒，最终损失 `0.00921393`。
- 轻量验证二：128x128 分辨率，128 个 3D 高斯，3000 steps，训练阶段约 `90.6 step/s`，命令墙钟约 `44` 秒，最终损失 `0.01173052`。

官方 Mip-NeRF 360 主包中推荐场景图片数量为：

- `garden`：185 张图片。
- `bicycle`：194 张图片。
- `counter`：240 张图片。

训练完成后，模型权重保存为：

- `runs/toy_3dgs/checkpoints/model_final.pth`
- `runs/toy_3dgs/checkpoints/model_final.ply`
- `weights/toy_3dgs/model_final.pth`
- `weights/toy_3dgs/model_final.ply`
- `weights/toy_3dgs_gpu/model_final.pth`
- `weights/toy_3dgs_gpu/model_final.ply`
- `weights/toy_3dgs_gpu_quality/model_final.pth`
- `weights/toy_3dgs_gpu_quality/model_final.ply`

渲染结果保存在：

- `runs/toy_3dgs/renders/`
- `runs/toy_3dgs/test_renders/`

## 7. 分析

教学版实现保留了 3DGS 的核心思想：用显式高斯点表示场景，并通过可微渲染直接优化图像重建误差。与原版 3DGS 相比，本实现没有依赖 CUDA rasterizer 扩展，训练速度和真实大场景质量较弱，但环境更轻量，适合在 Windows 本机快速复现实验流程。

如果后续切换到 GPU 环境，可以使用作业推荐的 Mip-NeRF 360 `garden`、`bicycle` 或 `counter` 场景，并将训练步数、图像分辨率和高斯数量提高，以获得更接近原版 3DGS 的重建质量。
