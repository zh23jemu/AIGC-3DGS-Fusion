# AIGC-3DGS-Fusion 项目记忆

## 项目目标

完成 `HW3_深度学习与空间智能.pdf` 中第一个 3D / AIGC 3D 方向任务，交付可运行代码、Markdown 实验报告和模型权重。

## 技术栈

- Python + PyTorch。
- 教学版 3D Gaussian Splatting：使用可学习 3D 高斯点，经相机投影后进行可微 2D splatting。
- 数据优先使用作业推荐的 Mip-NeRF 360 数据集；若本机网络或磁盘不适合完整下载，则使用仓库脚本生成的小型多视角样例数据完成本机训练验证。

## 当前架构

- `src/aigc3dgs/`：训练、渲染、数据下载、样例数据生成与模型工具代码。
- `README.md`：环境配置、数据准备、训练和测试命令。
- `report.md`：实验报告正文。
- `requirements.txt` / `environment.yml`：本机 Python 环境依赖说明。

## 开发规范

- 保持最小修改，优先保证作业交付物完整可复现。
- 新增代码使用较详细中文注释说明关键逻辑。
- 本项目始终使用 `.venv` 中的 Python 运行脚本。
- 不提交大型数据集、训练输出和本地虚拟环境。

## TODO

- 如需使用官方数据训练，解压 `data/360_v2.zip` 并准备 `garden`、`bicycle` 或 `counter` 场景。
- 如需更高质量结果，可在当前 GPU 环境继续增加训练步数、分辨率和高斯数量，但需注意 Quadro T1000 只有 4GB 显存。
- 最终提交前确认是否需要把模型权重上传到 Google Drive。

## 当前进度

已完成本机可运行的轻量教学版 3DGS 项目骨架，并用样例数据完成一次 CPU 训练验证。

## 风险问题

- 官方 Mip-NeRF 360 数据集体积较大，本机下载和训练可能受网络、磁盘与 GPU 条件限制。
- Windows 本机编译原版 3DGS CUDA 扩展风险较高，因此当前实现优先保证作业管线可复现。

## Current Status

项目已完成代码、文档、样例数据 GPU 训练验证、模型权重产出、本地 Git 提交、GitHub public 仓库推送、集群训练脚本、AWS EC2 训练说明、AWS CLI/profile 配置验证，并新增换机续作交接文档。AWS 公有云质量训练已使用 `g4dn.xlarge` 完成，产物已下载到本地 `weights/aws_ec2_toy_3dgs_quality/`，当前推荐该权重作为最终提交结果。

## Recent Changes

- 创建项目级长期维护说明。
- 确认第一个任务推荐数据集为 Mip-NeRF 360，推荐场景包括 `garden`、`bicycle`、`counter`。
- 新增轻量 PyTorch 3DGS 实现、数据生成、数据下载、训练和渲染脚本。
- 使用 `data/toy_scene` 完成 250 steps CPU 训练，生成 `model_final.pth` 和 `model_final.ply`。
- 更新 `.gitignore`，保留小样例数据和最终权重作为作业交付物，仅忽略官方大数据包、训练中间输出、虚拟环境缓存和 egg-info 元数据。
- 将 `.venv` 中 PyTorch 从 CPU 版切换为 `torch 2.12.1+cu126`，GPU 验证通过，显卡为 `Quadro T1000`。
- 使用 GPU 完成 800 steps、96 高斯、96x96 分辨率训练，训练阶段约 119 step/s，最终 loss 为 `0.00921393`。
- 使用 GPU 完成 3000 steps、128 高斯、128x128 分辨率训练，训练阶段约 90.6 step/s，最终 loss 为 `0.01173052`。
- 已统计官方推荐场景图片数量：`garden` 185、`bicycle` 194、`counter` 240。
- 按全局 Slurm 约定新增 `slurm/train_toy_gpu.sbatch` 和 `scripts/submit_slurm.md`，用于在集群登录节点提交训练。
- 根据用户明确要求新增 `slurm/train_toy_gpu_aws.sbatch`，用于提交到 `aws` 分区，并保留费用提醒与分区覆盖说明。
- 新增 `HANDOFF.md`，记录换电脑后恢复环境、继续训练、提交远程仓库和 Slurm/AWS 作业的步骤。
- 创建并推送 GitHub public 仓库：`https://github.com/zh23jemu/AIGC-3DGS-Fusion`，默认分支为 `main`。
- 新增 AWS EC2 公有云训练说明和 user-data 脚本，明确不保存 AWS 密钥，建议使用 AWS CLI profile 或 EC2 IAM Role。
- 本机已安装 AWS CLI v2.35.9，并验证 `aigc-3dgs` profile 可访问账号 `553432479592`，当前 region 为 `ap-northeast-1`。
- 选择 `g4dn.xlarge` 作为 AWS 训练实例，首次实例 `i-00affb8095794efa5` 验证到 `Tesla T4` 与 CUDA 可用，但 `torch 2.12.1+cu130` 在 Python 环境中触发 `sys.get_int_max_str_digits` 缺失，训练失败。
- 已停止失败实例 `i-00affb8095794efa5`，避免继续产生 GPU 实例费用。
- 更新 `scripts/aws_train_user_data.sh`：EC2 默认使用 `python3.10`，固定安装 `torch==2.7.1+cu126` / `torchvision==0.22.1+cu126`，再用 `--no-deps` 安装本项目，避免 PyTorch 被自动升级。
- 使用修正版 user-data 在 `i-0ad699fcb01e110b8` 上完成 AWS EC2 `g4dn.xlarge` 训练，训练设备为 `Tesla T4`，配置为 3000 steps、128 高斯、128x128 分辨率，最终 loss 为 `0.01350455`。
- 已通过 EC2 Instance Connect 注入一次性 SSH key，下载远端权重、PLY、metrics 和渲染图；权重整理到 `weights/aws_ec2_toy_3dgs/`，渲染图保留在本地 `runs/aws_ec2_toy_3dgs/`。
- 已对两台 EC2 实例 `i-00affb8095794efa5` 和 `i-0ad699fcb01e110b8` 发起终止，避免继续产生实例与 EBS 费用。
- 将 `scripts/aws_train_user_data.sh` 参数化，默认质量训练配置提升为 `10000` steps、`512` 高斯、`128x128` 分辨率，输出目录为 `runs/aws_ec2_toy_3dgs_quality`。
- 使用新实例 `i-02ddca258856fc5bf` 完成 AWS EC2 质量训练，训练设备为 `Tesla T4`，配置为 10000 steps、512 高斯、128x128 分辨率，最终 loss 为 `0.00925142`。
- 已通过 EC2 Instance Connect 下载质量版权重、PLY、metrics 和渲染图；权重整理到 `weights/aws_ec2_toy_3dgs_quality/`，渲染图保留在本地 `runs/aws_ec2_toy_3dgs_quality/`。
- 已对质量训练实例 `i-02ddca258856fc5bf` 发起终止，避免继续产生实例与 EBS 费用。

## Next TODO

- 换电脑后可直接从 GitHub public 仓库拉取项目，并按 `HANDOFF.md` 恢复 `.venv`、验证 GPU。
- AWS/Slurm 提交需要在有 `sbatch` 的集群环境执行。
- 确认质量训练实例 `i-02ddca258856fc5bf` 最终进入 `terminated` 状态。
- 提交并推送 AWS 质量版权重与文档更新。

## Open Issues

- 本机 GPU 可用，但 Quadro T1000 只有 4GB 显存，完整大场景训练可能受显存限制。
- 官方 Mip-NeRF 360 主包已下载到 `data/360_v2.zip`，约 12.5GB，尚未解压。
- 当前 Windows 本机没有 `sbatch`，不能直接提交 Slurm 作业；需在集群登录节点运行提交命令。
- 用户曾在对话中暴露 AWS Access Key/Secret，必须在 AWS IAM 中禁用/删除后重新生成，不能继续使用暴露密钥。
- 本地 `runs/aws_ec2_toy_3dgs/` 中保存了 AWS 渲染图，但 `runs/` 仍按 `.gitignore` 作为可再生成输出不提交。

## Architecture Decisions

- 使用轻量 PyTorch 实现替代原版 3DGS CUDA 扩展，降低 Windows 本机交付风险。
- 模型权重同时保存为 `.pth` 和 `.ply`，便于代码复现与报告展示。
- `data/toy_scene` 与 `weights/toy_3dgs` 属于当前交付物；`runs/` 属于可再生成训练输出。
- GPU 版最终权重整理到 `weights/toy_3dgs_gpu`，CPU 版早期权重仍保留在 `weights/toy_3dgs` 便于对比。
- 更高分辨率 GPU 质量测试权重整理到 `weights/toy_3dgs_gpu_quality`。
- AWS EC2 训练权重整理到 `weights/aws_ec2_toy_3dgs`，作为云端训练结果交付物入库。
- AWS EC2 质量训练权重整理到 `weights/aws_ec2_toy_3dgs_quality`，作为当前推荐最终交付权重入库。
- 新增 Slurm 训练脚本 `slurm/train_toy_gpu.sbatch`，默认使用 `gpu` 分区、`gpo-ifv7xx` 账号和 `normal` QOS，避免默认使用 `aws` 分区。
