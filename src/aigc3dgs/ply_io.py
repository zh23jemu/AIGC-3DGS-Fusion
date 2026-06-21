"""模型权重导出工具。"""

from __future__ import annotations

from pathlib import Path

import torch

from .gaussian_model import GaussianScene


def save_ply(model: GaussianScene, path: str | Path) -> None:
    """将学习到的 3D 高斯中心和颜色导出为 ASCII PLY 点云。

    PLY 文件可被 MeshLab、CloudCompare、Blender 等工具查看。这里导出的字段
    包含位置、颜色、不透明度和尺度，便于报告中说明模型权重内容。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = model.xyz.detach().cpu()
    colors = (model.colors.detach().cpu() * 255.0).clamp(0, 255).to(torch.uint8)
    opacities = model.opacities.detach().cpu().squeeze(1)
    scales = model.scales.detach().cpu().squeeze(1)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float opacity\n")
        f.write("property float scale\n")
        f.write("end_header\n")
        for p, c, opacity, scale in zip(xyz, colors, opacities, scales):
            f.write(
                f"{p[0].item():.6f} {p[1].item():.6f} {p[2].item():.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"{opacity.item():.6f} {scale.item():.6f}\n"
            )


def checkpoint_dict(model: GaussianScene, image_size: int) -> dict:
    """生成可被 `render.py` 重新加载的模型检查点。"""

    return {
        "num_gaussians": int(model.xyz.shape[0]),
        "image_size": int(image_size),
        "state_dict": model.state_dict(),
    }
