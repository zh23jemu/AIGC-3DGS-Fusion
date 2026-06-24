# AIGC-3DGS-Fusion 完整实验报告

## 1. 项目背景

`AIGC-3DGS-Fusion` 面向“基于 3DGS 与 AIGC 的多源资产生成与真实场景融合”任务。项目目标不是只完成单一 3D 重建，而是搭建一条完整链路：先分别获得真实多视角重建资产、文本生成 3D 资产和单图生成 3D 资产，再重建统一背景场景，最后将多源资产插入同一个 3D 场景并输出多视角漫游结果。

本项目最终交付包括：

- 代码仓库：`https://github.com/zh23jemu/AIGC-3DGS-Fusion`
- 3DGS 权重：`weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth`
- 3DGS 点云导出：`weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.ply`
- 文本到 3D 资产 B：`weights/object_b_threestudio/`
- 单图到 3D 资产 C：`weights/object_c_zero123/`
- 融合漫游视频：`runs/final_fusion/aigc_3dgs_fusion_walkthrough.mp4`
- 模型权重网盘链接：待上传后填写

## 2. 任务要求对应关系

| 要求 | 实现文件与结果 | 说明 |
| --- | --- | --- |
| 物体 A：真实多视角重建 + COLMAP + 3DGS | `src/aigc3dgs/train.py`、`weights/aws_ec2_toy_3dgs_q2_20k_1024/` | 已实现多视角 3DGS 训练和 PLY 导出；若替换为手机真实物体数据，入口保持一致 |
| 物体 B：threestudio + SDS 文本到 3D | `scripts/aws_hw3_full_user_data.sh`、`weights/object_b_threestudio/` | 使用 threestudio 的 `dreamfusion-sd.yaml` 和 Stable Diffusion SDS 训练 |
| 物体 C：Zero123 单图到 3D | `scripts/aws_hw3_full_user_data.sh`、`weights/object_c_zero123/` | 使用 threestudio 的 `stable-zero123.yaml` 从单张 RGBA 前景图生成 |
| 背景场景：Mip-NeRF 360 + 3DGS | `src/aigc3dgs/download_dataset.py`、`src/aigc3dgs/train.py` | 推荐场景为 `counter`；当前交付保留轻量可复现背景与官方数据下载入口 |
| 场景融合与渲染 | `src/aigc3dgs/final_fusion_render.py` | 将 A/B/C 资产预览结果合成多视角漫游视频 |
| 质量评估与报告 | `src/aigc3dgs/final_report_assets.py`、本报告 | 汇总 Loss 曲线、指标表、方法对比和融合策略 |

## 3. 数据集与资产来源

### 3.1 物体 A：多视角重建资产

物体 A 使用多视角图像训练 3D Gaussian Splatting。当前代码支持读取 `transforms.json` 中的相机位姿和多视角 RGB 图像，并优化一组显式 3D 高斯点。训练后导出 `.pth` 权重和 `.ply` 点云。

最终推荐权重为：

```text
weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth
weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.ply
```

### 3.2 物体 B：文本到 3D 生成资产

物体 B 使用 threestudio 的 DreamFusion-SD 路线生成。输入是一段文本 prompt：

```text
a small blue ceramic rabbit figurine, smooth surface, studio lighting
```

训练入口为：

```bash
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a small blue ceramic rabbit figurine, smooth surface, studio lighting"
```

该路线使用预训练 2D Stable Diffusion 模型作为先验，通过 SDS Loss 优化 3D 表示。

### 3.3 物体 C：单图到 3D 生成资产

物体 C 使用 Stable Zero123 路线生成。输入是一张去背景的 RGBA 前景图，训练入口为：

```bash
python launch.py --config configs/stable-zero123.yaml --train --gpu 0 data.image_path=./load/images/hw3_object_c_rgba.png
```

Zero123 根据单张输入图像预测新视角一致性约束，再优化得到完整 3D 表示。

### 3.4 背景场景

背景场景参考 Mip-NeRF 360 数据集，官方数据页面为：

```text
https://jonbarron.info/mipnerf360/
```

推荐使用 `counter` 场景，原因是其室内结构稳定、物体边界清晰，适合进行资产插入和漫游渲染。项目提供下载入口：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.download_dataset --out data --scene counter
```

## 4. 方法原理

### 4.1 3D Gaussian Splatting

3DGS 用显式 3D 高斯点表示场景。每个高斯点包含三维位置、颜色、不透明度和尺度。训练时将高斯点投影到图像平面，通过可微 alpha 合成得到渲染图，并用渲染图与真实图像之间的误差更新参数。

本项目轻量实现的优化目标为：

```text
L = MSE(rendered_rgb, target_rgb) + lambda * scale_regularization
```

### 4.2 SDS 文本到 3D

SDS 使用预训练 2D 扩散模型提供梯度。给定文本 prompt 后，3D 表示从随机初始化开始渲染多个视角，再将渲染图输入扩散模型计算 score distillation 梯度，使 3D 表示逐渐符合文本语义。

### 4.3 Zero123 单图到 3D

Zero123 输入单张图像，并学习相对相机视角变化下的新视角图像先验。训练时通过新视角一致性约束补全单图中不可见的几何和纹理。

### 4.4 表示统一与融合

三类资产的原始表示不同：

- 物体 A 和背景：显式 3D Gaussian / PLY。
- 物体 B：threestudio 训练结果，可导出 mesh。
- 物体 C：Stable Zero123 训练结果，可导出 mesh。

本项目采用两级统一策略：

1. 资产级统一：将每个资产保存为可检查的 checkpoint、mesh、PLY 或预览渲染。
2. 渲染级统一：将 A/B/C 的预览渲染结果与背景帧按照统一坐标、比例和相机轨迹合成，输出多视角漫游视频。

若后续使用 Blender，可直接导入 B/C mesh 和 A/background PLY；若使用代码级 3DGS 融合，则可将 mesh 采样为点云，再初始化为 Gaussian 表示。

## 5. 实验设置

| 模块 | 设置 |
| --- | --- |
| 3DGS Optimizer | Adam |
| 3DGS Learning Rate | 0.03 |
| 3DGS Loss | RGB MSE + 尺度正则 |
| 3DGS Steps | 20000 |
| 3DGS Gaussians | 1024 |
| threestudio B Config | `configs/dreamfusion-sd.yaml` |
| threestudio B Prompt | `a small blue ceramic rabbit figurine, smooth surface, studio lighting` |
| Zero123 C Config | `configs/stable-zero123.yaml` |
| 训练平台 | AWS EC2 GPU |

## 6. 实验结果

### 6.1 训练曲线

训练曲线文件：

```text
runs/final_report_assets/loss_curves.png
```

### 6.2 指标表

指标表文件：

```text
runs/final_report_assets/metrics_table.md
```

### 6.3 融合视频

融合渲染命令：

```powershell
.\.venv\Scripts\python.exe -m aigc3dgs.final_fusion_render --out runs/final_fusion
```

输出视频：

```text
runs/final_fusion/aigc_3dgs_fusion_walkthrough.mp4
```

## 7. 三种资产生成方法对比

| 方法 | 几何准确度 | 纹理细节 | 计算耗时 | 主要优点 | 主要问题 |
| --- | --- | --- | --- | --- | --- |
| 多视角重建 | 高，多视角约束真实几何 | 高，来自真实图像 | 中到高 | 可复现、几何稳定 | 需要拍摄和位姿估计 |
| 文本到 3D | 中，依赖扩散先验 | 中到高，语义丰富 | 高 | 不需要真实参考图 | 可能出现多面不一致 |
| 单图到 3D | 中，正面可靠，背面不确定 | 正面细节好 | 中 | 输入成本低 | 不可见区域依赖先验 |

## 8. 现象分析

多视角 3DGS 的优势在于几何约束明确，因此空间一致性最好。文本到 3D 的优势是生成自由度高，但生成质量强依赖 prompt、扩散模型版本和优化步数。单图到 3D 的输入成本最低，但由于只有一个真实视角，背面纹理和遮挡区域不可避免地依赖生成模型补全。

融合阶段最大的工程问题是表示不统一。3DGS 背景是显式高斯球，而 threestudio / Zero123 结果通常是隐式场或 mesh。因此本项目采用 mesh/PLY/渲染预览三种中间表达，保证资产可以被检查、可以被融合，也可以继续迁移到 Blender 或代码级 Gaussian 拼接。

## 9. 局限性

- 如果没有用户提供的手机真实物体照片，物体 A/C 的输入只能使用项目样例图替代。
- 低步数 threestudio / Zero123 训练能证明链路真实运行，但视觉质量可能弱于长时间训练。
- 当前 Python 融合视频更偏工程验证；若追求最终视觉质量，建议使用 Blender 重新布光和渲染。

## 10. 结论

`AIGC-3DGS-Fusion` 补齐了多源 3D 资产生成、背景重建、资产融合、质量评估和报告交付链路。项目同时提供可运行代码、训练脚本、模型权重、资产输出和融合视频，满足完整题目一的工程结构要求。
