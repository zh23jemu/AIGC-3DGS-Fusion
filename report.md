# AIGC-3DGS-Fusion 实验报告

## 1. 项目概述

`AIGC-3DGS-Fusion` 是一个面向多视角 3D 重建实验的教学版 3D Gaussian Splatting 项目。项目目标是从多视角 RGB 图像和相机位姿中学习一组显式 3D 高斯点，并通过可微渲染方式优化这些高斯点的位置、颜色、不透明度和尺度，最终得到可以复现训练、导出模型权重、导出 PLY 点云并渲染测试视角的完整流程。

项目交付内容包括：

- 可运行代码：`src/aigc3dgs/`
- 实验报告：`report.md`
- 最终模型权重：`weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth`
- PLY 点云文件：`weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.ply`
- 小型多视角样例数据：`data/toy_scene/`

## 2. 数据集与数据准备

项目参考的公开数据集为 Mip-NeRF 360。官方页面为：

```text
https://jonbarron.info/mipnerf360/
```

已验证官方主包 `360_v2.zip` 中包含 `garden`、`bicycle`、`counter` 等典型多视角场景，主包大小约 `12.5GB`。官方数据集适合进行真实场景 3D 重建，但完整解压、预处理和原版大场景训练需要较长时间与更高显存。本项目为了保证代码、报告和模型权重都可以稳定复现，统一使用仓库内脚本生成的小型多视角样例数据完成训练和测试。

样例数据目录为：

```text
data/toy_scene/
```

数据集包含：

- `20` 个相机视角。
- 每个视角一张 RGB 图像。
- `transforms.json`，记录相机位姿、相机视场角和图像路径。
- 图像分辨率为 `128x128`。

生成数据的命令为：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.prepare_toy_dataset --out data/toy_scene --views 20 --size 128
```

## 3. 方法原理

本项目实现的是教学版 3D Gaussian Splatting。核心思想是用一组显式 3D 高斯点表示场景，每个高斯点携带可学习的空间位置、颜色、不透明度和尺度参数。训练时，模型将这些 3D 高斯点投影到当前相机视角的二维图像平面，在图像平面上生成 2D 高斯响应，再通过 alpha 合成得到当前视角的渲染图。

训练目标是让渲染图尽可能接近真实图像。对于每一次迭代，程序随机采样一个训练视角，执行以下流程：

1. 读取该视角的 RGB 图像和相机位姿。
2. 将所有 3D 高斯点从世界坐标变换到相机坐标。
3. 将相机坐标投影到图像平面。
4. 根据投影位置、尺度和不透明度生成二维高斯响应。
5. 对所有高斯点进行颜色合成，得到渲染图。
6. 计算渲染图与真实图像之间的 MSE 损失。
7. 使用 Adam 优化器更新所有可学习参数。

与原版 3DGS 相比，本项目没有使用 CUDA rasterizer 扩展，而是用 PyTorch 实现较轻量的可微 splatting 流程。这样做的优点是环境配置更简单、可读性更强、跨平台更容易复现；缺点是渲染质量和大场景训练性能弱于原版实现。

## 4. 模型结构与超参数

模型结构为可学习 3D Gaussian 参数集合，主要参数如下：

- `xyz`：每个高斯点的三维位置。
- `color_logits`：每个高斯点的 RGB 颜色参数。
- `opacity_logits`：每个高斯点的不透明度参数。
- `log_scales`：每个高斯点的尺度参数。

最终训练配置如下：

| 项目 | 设置 |
| --- | --- |
| 训练视角数量 | 20 |
| 图像分辨率 | 128x128 |
| 高斯数量 | 1024 |
| 训练步数 | 20000 |
| Batch Size | 每次随机采样 1 个视角 |
| Optimizer | Adam |
| Learning Rate | 0.03 |
| Loss Function | RGB MSE + 轻量尺度正则 |
| 最终损失 | 0.00914205 |

损失函数由两部分组成：

- RGB 重建损失：约束渲染图接近真实图像。
- 尺度正则项：抑制高斯尺度无限变大，减少过度模糊。

## 5. 训练与测试命令

安装依赖后，可以用以下命令重新训练最终配置：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.train --data data/toy_scene --out runs/aws_ec2_toy_3dgs_q2_20k_1024 --steps 20000 --gaussians 1024 --image-size 128
```

使用最终权重进行测试渲染：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.render --checkpoint weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth --data data/toy_scene --out runs/final_test_renders --image-size 128
```

训练完成后会生成：

```text
runs/aws_ec2_toy_3dgs_q2_20k_1024/checkpoints/model_final.pth
runs/aws_ec2_toy_3dgs_q2_20k_1024/checkpoints/model_final.ply
runs/aws_ec2_toy_3dgs_q2_20k_1024/renders/
runs/aws_ec2_toy_3dgs_q2_20k_1024/test_renders/
```

整理后的最终权重保存为：

```text
weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth
weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.ply
weights/aws_ec2_toy_3dgs_q2_20k_1024/metrics.txt
```

## 6. 实验结果

最终推荐模型为 `weights/aws_ec2_toy_3dgs_q2_20k_1024/`。该模型的训练配置为：

- `20000` steps
- `1024` Gaussians
- `128x128` image size
- final loss: `0.00914205`

最终模型文件包括：

- `model_final.pth`：PyTorch checkpoint，可用于继续训练或重新渲染。
- `model_final.ply`：导出的 PLY 点云文件，可用于观察高斯点的空间分布。
- `metrics.txt`：记录最终损失、训练步数、高斯数量和训练设备。

从渲染效果看，模型能够稳定重建样例目标的主要空间结构和颜色区域。蓝色主体和红色局部在不同视角下都能保持一致的空间关系，说明模型确实学习到了多视角一致的 3D 表示。由于当前实现是教学版可微 splatting，渲染结果仍然带有一定柔和感和少量散点噪声，边缘锐度弱于原版 3DGS，但整体流程完整、结果可复现。

## 7. 质量迭代对比

为判断继续增加训练量是否还有明显收益，本项目进行了多组质量迭代实验。结果如下：

| 配置 | Steps | Gaussians | Image Size | Final Loss | 结论 |
| --- | ---: | ---: | --- | ---: | --- |
| 基础训练 | 3000 | 128 | 128x128 | 0.01350455 | 可完成基本重建，但较模糊 |
| 质量版一轮 | 10000 | 512 | 128x128 | 0.00925142 | 损失明显下降 |
| 质量版二轮 | 20000 | 1024 | 128x128 | 0.00914205 | 当前最佳 |
| 质量版三轮 | 40000 | 2048 | 128x128 | 0.01122936 | 损失退化，不再继续加码 |

从实验结果看，`10000 / 512` 相比基础训练有明显提升；`20000 / 1024` 相比 `10000 / 512` 仍有小幅提升，并取得最低损失；继续提高到 `40000 / 2048` 后，损失反而上升，说明在当前教学版实现、样例数据规模和固定学习率设置下，继续增加高斯数量和训练步数不再带来稳定收益。

因此，最终选择 `20000 steps / 1024 gaussians / 128x128` 作为项目推荐模型。

## 8. 分析与讨论

本项目验证了轻量版 3D Gaussian Splatting 的完整训练流程。实验表明，随着高斯数量和训练步数增加，模型的重建损失会先明显下降，然后进入收益递减区间。当训练量继续增大到 `40000 steps / 2048 gaussians` 时，模型没有继续变好，反而出现损失退化。这可能与以下因素有关：

- 样例数据规模较小，20 个视角不足以稳定约束过多高斯点。
- 当前实现未包含原版 3DGS 中更复杂的 densification、pruning 和 CUDA rasterizer 优化。
- 固定学习率在大模型长时间训练时可能不够稳定。
- 高斯数量过多后，部分高斯点可能学习到噪声或冗余结构。

因此，当前最佳配置不是最大配置，而是在训练质量、稳定性和模型复杂度之间取得较好平衡的 `20000 steps / 1024 gaussians`。

## 9. 局限性与改进方向

当前项目仍有以下局限：

- 使用的是教学版 splatting 实现，渲染锐度和速度弱于原版 3DGS。
- 训练数据为小型合成多视角样例，并非完整真实场景。
- 没有实现高斯点动态增删、密度控制和复杂可见性处理。
- 当前评估主要使用 MSE 和渲染图主观观察，未进一步加入 PSNR、SSIM 或 LPIPS。

后续可改进方向包括：

- 切换到原版 CUDA rasterizer 3DGS 实现。
- 在 Mip-NeRF 360 的 `garden`、`bicycle` 或 `counter` 场景上进行真实大场景训练。
- 加入学习率衰减、早停和更系统的超参数搜索。
- 增加 PSNR / SSIM / LPIPS 等评价指标。
- 引入高斯 densification 和 pruning，提高细节表达能力。

## 10. 结论

`AIGC-3DGS-Fusion` 完成了从多视角数据准备、3D 高斯场景建模、可微渲染训练、权重导出到测试视角渲染的完整流程。最终推荐模型为：

```text
weights/aws_ec2_toy_3dgs_q2_20k_1024/
```

该模型在 `20000 steps / 1024 gaussians / 128x128` 配置下取得最终损失 `0.00914205`，是当前多轮质量迭代中的最佳结果。继续增大到 `40000 steps / 2048 gaussians` 后效果退化，因此本项目停止继续加码训练，并将 q2 配置作为最终训练成果。
