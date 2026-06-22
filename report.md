# HW3 第一个任务实验报告

## 1. 任务目标

本实验选择作业第一个 3D / AIGC 3D 方向任务中的 3D Gaussian Splatting 路线。目标是从多视角图像中训练一个可微 3D 高斯场景表示，并导出可复现的代码、实验报告和模型权重。

## 2. 数据集

作业 PDF 推荐使用 Mip-NeRF 360 数据集，推荐场景包括 `garden`、`bicycle`、`counter`。官方页面为 https://jonbarron.info/mipnerf360/ 。

已验证官方主包 `360_v2.zip` 中包含 `garden`、`bicycle`、`counter` 场景，文件大小约 `12.5GB`。由于完整解压和原版大场景训练耗时较长，本次实验统一使用仓库脚本生成的小型多视角样例数据完成训练、调参和测试。样例数据包含 20 个围绕同一合成目标的相机视角，每个视角保存 RGB 图像和相机位姿参数。

## 3. 方法

本实验实现教学版 3D Gaussian Splatting。模型由一组可学习 3D 高斯点组成，每个高斯包含空间位置、颜色、不透明度和尺度参数。训练时将 3D 高斯点投影到当前相机视角，在图像平面上生成 2D 高斯权重并进行 alpha 合成，最后用渲染图和真实图之间的重建损失优化模型。

## 4. 网络结构与超参数

- Network Architecture：可学习 3D Gaussian 参数集合，包括位置、颜色、尺度和不透明度。
- Batch Size：每次随机采样 1 个视角。
- Learning Rate：`0.03`。
- Optimizer：Adam。
- Epochs / Iterations：最终训练 `20000` steps。
- Loss Function：RGB 图像均方误差 MSE，同时加入轻量尺度正则项抑制高斯过度扩张。

## 5. 训练与测试命令

生成样例数据：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 128
```

训练：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.train --data data/toy_scene --out runs/aws_ec2_toy_3dgs_q2_20k_1024 --steps 20000 --gaussians 1024 --image-size 128
```

测试渲染：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.render --checkpoint runs/aws_ec2_toy_3dgs_q2_20k_1024/checkpoints/model_final.pth --data data/toy_scene --out runs/aws_ec2_toy_3dgs_q2_20k_1024/test_renders --image-size 128
```

## 6. 实验结果

本次实验统一采用 GPU 训练环境完成最终训练与质量迭代。训练环境中 PyTorch CUDA 可用，最终训练设备为 `Tesla T4`，CUDA build 为 `12.6`。实验配置如下：

- 训练视角：20
- 图像分辨率：128x128
- 最终高斯数量：1024
- 最终训练步数：20000 steps
- 最终损失：`0.00914205`

官方 Mip-NeRF 360 主包中推荐场景图片数量为：

- `garden`：185 张图片。
- `bicycle`：194 张图片。
- `counter`：240 张图片。

训练完成后，模型权重保存为：

- `runs/aws_ec2_toy_3dgs_q2_20k_1024/checkpoints/model_final.pth`
- `runs/aws_ec2_toy_3dgs_q2_20k_1024/checkpoints/model_final.ply`
- `weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth`
- `weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.ply`

渲染结果保存在：

- `runs/aws_ec2_toy_3dgs_q2_20k_1024/renders/`
- `runs/aws_ec2_toy_3dgs_q2_20k_1024/test_renders/`

## 7. 分析

教学版实现保留了 3DGS 的核心思想：用显式高斯点表示场景，并通过可微渲染直接优化图像重建误差。与原版 3DGS 相比，本实现没有依赖 CUDA rasterizer 扩展，真实大场景质量较弱，但环境更轻量，适合快速复现实验流程。

如果后续继续扩展，可以使用作业推荐的 Mip-NeRF 360 `garden`、`bicycle` 或 `counter` 场景，并切换到原版 CUDA rasterizer 3DGS 实现，以获得更接近真实大场景重建的质量。

## 8. 质量迭代对比

为继续提升渲染质量，本实验在同一 GPU 训练流程下做了多轮质量迭代，训练与测试结果如下：

- 基础训练：`3000 steps`、`128` 个高斯、`128x128` 分辨率，最终损失 `0.01350455`。
- 质量版一轮：`10000 steps`、`512` 个高斯、`128x128` 分辨率，最终损失 `0.00925142`。
- 质量版二轮：`20000 steps`、`1024` 个高斯、`128x128` 分辨率，最终损失 `0.00914205`。
- 质量版三轮：`40000 steps`、`2048` 个高斯、`128x128` 分辨率，最终损失 `0.01122936`。

从结果看，`20000 steps / 1024 gaussians` 是当前最佳配置。它相对一轮质量版有小幅提升，而继续加到 `40000 steps / 2048 gaussians` 后损失反而退化，说明继续堆叠训练量已经没有明显收益。因此，最终推荐模型权重为 `weights/aws_ec2_toy_3dgs_q2_20k_1024/`。
