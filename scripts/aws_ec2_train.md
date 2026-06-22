# AWS EC2 公有云训练说明

## 安全提醒

不要把 AWS Access Key 或 Secret Access Key 写入仓库、脚本、README、终端历史或聊天记录。若密钥已经暴露，应立即在 AWS IAM 中禁用/删除并重新生成。

推荐方式：

- 优先给 EC2 实例绑定 IAM Role。
- 本机启动实例时使用 AWS CLI profile。
- 不在项目文件中保存任何密钥。

## 1. 安装并配置 AWS CLI

Windows 可安装 AWS CLI v2，然后配置 profile：

```powershell
aws configure --profile aigc-3dgs
```

按提示输入新的 Access Key、Secret、Region 和输出格式。建议 region 使用离你近且有 GPU 实例配额的区域，例如：

```text
ap-northeast-1
```

检查登录身份：

```powershell
aws sts get-caller-identity --profile aigc-3dgs
```

当前本机已验证的 profile 信息：

- profile：`aigc-3dgs`
- account：`553432479592`
- region：`ap-northeast-1`
- IAM user：`arn:aws:iam::553432479592:user/grafana`

## 2. 推荐实例

本项目轻量训练建议：

- `g4dn.xlarge`：1 张 NVIDIA T4，16GB 显存，性价比适合本实验。
- AMI：Ubuntu 22.04 深度学习 AMI，或普通 Ubuntu 22.04 加 NVIDIA 驱动环境。
- Python / PyTorch：EC2 启动脚本默认使用 `python3.10`，并固定安装 `torch==2.7.1+cu126`、`torchvision==0.22.1+cu126`，避免自动解析到 CUDA 13 wheel。
- 默认质量训练参数：`10000` steps、`512` 个高斯、`128x128` 分辨率，输出目录为 `runs/aws_ec2_toy_3dgs_quality`。

如果使用普通 Ubuntu AMI，需要额外安装 NVIDIA 驱动；为了减少配置风险，建议使用 AWS Deep Learning AMI。

## 3. 启动实例示例

以下命令需要替换：

- `<AMI_ID>`：Deep Learning AMI ID。
- `<KEY_NAME>`：你的 EC2 key pair 名称。
- `<SECURITY_GROUP_ID>`：允许 SSH 的安全组。
- `<SUBNET_ID>`：目标子网。

```powershell
aws ec2 run-instances `
  --profile aigc-3dgs `
  --image-id <AMI_ID> `
  --instance-type g4dn.xlarge `
  --key-name <KEY_NAME> `
  --security-group-ids <SECURITY_GROUP_ID> `
  --subnet-id <SUBNET_ID> `
  --user-data file://scripts/aws_train_user_data.sh `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=aigc-3dgs-training}]" `
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=80,VolumeType=gp3}"
```

查看实例：

```powershell
aws ec2 describe-instances --profile aigc-3dgs --filters "Name=tag:Name,Values=aigc-3dgs-training" --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,PublicIp:PublicIpAddress}"
```

## 4. 查看训练日志

SSH 到实例后：

```bash
tail -f /var/log/cloud-init-output.log
```

训练输出位于：

- `/home/ubuntu/AIGC-3DGS-Fusion/runs/aws_ec2_toy_3dgs/checkpoints/model_final.pth`
- `/home/ubuntu/AIGC-3DGS-Fusion/runs/aws_ec2_toy_3dgs/checkpoints/model_final.ply`
- `/home/ubuntu/AIGC-3DGS-Fusion/runs/aws_ec2_toy_3dgs/test_renders/`

## 5. 停止或终止实例

训练结束后务必停止或终止实例，避免继续计费。

停止：

```powershell
aws ec2 stop-instances --profile aigc-3dgs --instance-ids <INSTANCE_ID>
```

终止：

```powershell
aws ec2 terminate-instances --profile aigc-3dgs --instance-ids <INSTANCE_ID>
```

## 6. 费用与配额风险

- `g4dn.xlarge` 会产生 GPU 实例费用和 EBS 存储费用。
- 新账号可能没有 GPU 实例配额，需要先申请 EC2 On-Demand G/VT 实例配额。
- 使用完实例后及时停止或终止。

## 7. 当前执行记录

- 已选择 `g4dn.xlarge` 作为本项目 AWS GPU 训练资源，原因是 T4 16GB 显存足够运行当前教学版 3DGS 训练，费用低于 `g5` / `g6`。
- 首次实例 `i-00affb8095794efa5` 已验证 GPU 和 CUDA 可用，但因 pip 自动安装到 `torch 2.12.1+cu130` 后触发 Python 兼容问题，训练未完成，实例已停止以避免继续计费。
- 已修正 `scripts/aws_train_user_data.sh`，固定 CUDA 12.6 PyTorch 版本并使用 `--no-deps` 安装项目，后续重新启动实例应使用修正版脚本。
- 为提升渲染质量，已将 user-data 默认训练参数提升为 `10000` steps、`512` 个高斯，并通过环境变量 `STEPS`、`GAUSSIANS`、`IMAGE_SIZE`、`RUN_NAME` 支持覆盖。
- 新增 `scripts/aws_train_quality_sweep_user_data.sh`，用于继续迭代质量训练，默认依次运行 `20000 steps / 1024 gaussians` 与 `40000 steps / 2048 gaussians` 两档，用于判断继续增大训练量是否还有明显收益。
