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

训练完成后主要产物位于：

- `runs/slurm_toy_3dgs/checkpoints/model_final.pth`
- `runs/slurm_toy_3dgs/checkpoints/model_final.ply`
- `runs/slurm_toy_3dgs/test_renders/`
