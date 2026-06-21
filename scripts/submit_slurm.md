# Slurm 训练提交说明

本项目默认按集群约定使用 `gpu` 分区、`gpo-ifv7xx` 账号和 `normal` QOS。

提交命令：

```bash
sbatch slurm/train_toy_gpu.sbatch
```

如果需要覆盖分区，例如短时任务使用 `gpuHz`：

```bash
sbatch --partition=gpuHz slurm/train_toy_gpu.sbatch
```

不要默认使用 `aws` 分区，除非明确接受额外费用。

如果明确要提交到 `aws` 分区，可使用：

```bash
sbatch slurm/train_toy_gpu_aws.sbatch
```

该脚本仍保留 `--account=gpo-ifv7xx` 和 `--qos=normal`；如果集群提示 account/partition 不匹配，先用：

```bash
sacctmgr show assoc user=$USER format=User,Account,Partition,QOS
```

确认可用组合。

训练完成后主要产物位于：

- `runs/slurm_toy_3dgs/checkpoints/model_final.pth`
- `runs/slurm_toy_3dgs/checkpoints/model_final.ply`
- `runs/slurm_toy_3dgs/test_renders/`

AWS 分区脚本的训练产物位于：

- `runs/aws_toy_3dgs/checkpoints/model_final.pth`
- `runs/aws_toy_3dgs/checkpoints/model_final.ply`
- `runs/aws_toy_3dgs/test_renders/`
