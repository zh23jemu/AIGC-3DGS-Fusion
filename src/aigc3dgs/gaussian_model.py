"""教学版 3D Gaussian Splatting 模型与渲染器。"""

from __future__ import annotations

import math

import torch
from torch import nn

from .cameras import CameraView


class GaussianScene(nn.Module):
    """用一组可学习 3D 高斯表示场景。

    每个高斯包含：
    - `xyz`：世界坐标位置。
    - `color_logits`：经过 sigmoid 后得到 RGB 颜色。
    - `opacity_logits`：经过 sigmoid 后得到不透明度。
    - `log_scales`：经过 exp 后得到屏幕空间高斯尺度。

    这个实现没有使用原版 3DGS 的 CUDA rasterizer，而是用 PyTorch 张量运算在
    小分辨率图像上完成可微 splatting，便于在普通 Windows 本机复现。
    """

    def __init__(self, num_gaussians: int, radius: float = 1.0) -> None:
        super().__init__()
        self.xyz = nn.Parameter(torch.randn(num_gaussians, 3) * radius * 0.35)
        self.color_logits = nn.Parameter(torch.randn(num_gaussians, 3) * 0.4)
        self.opacity_logits = nn.Parameter(torch.full((num_gaussians, 1), -1.0))
        self.log_scales = nn.Parameter(torch.full((num_gaussians, 1), math.log(5.0)))

    @property
    def colors(self) -> torch.Tensor:
        """返回约束到 `[0, 1]` 的 RGB 颜色。"""

        return torch.sigmoid(self.color_logits)

    @property
    def opacities(self) -> torch.Tensor:
        """返回约束到 `[0, 1]` 的不透明度。"""

        return torch.sigmoid(self.opacity_logits)

    @property
    def scales(self) -> torch.Tensor:
        """返回正数尺度，并限制最小值避免数值不稳定。"""

        return torch.exp(self.log_scales).clamp(min=1.0, max=32.0)


def render(model: GaussianScene, view: CameraView, image_size: int) -> torch.Tensor:
    """将 3D 高斯模型渲染到指定相机视角。

    渲染流程：
    1. 将 3D 高斯中心从世界坐标变换到相机坐标。
    2. 使用针孔相机模型投影到图像平面。
    3. 对每个像素计算所有高斯的 2D 权重。
    4. 使用 alpha 风格的归一化加权合成 RGB 图像。
    """

    device = model.xyz.device
    c2w = view.camera_to_world.to(device)
    w2c = torch.linalg.inv(c2w)

    ones = torch.ones((model.xyz.shape[0], 1), device=device)
    points_h = torch.cat([model.xyz, ones], dim=1)
    cam = (w2c @ points_h.T).T[:, :3]

    z = -cam[:, 2].clamp(max=-0.05)
    valid = z > 0.05
    x = view.focal * (cam[:, 0] / z) + image_size * 0.5
    y = view.focal * (-cam[:, 1] / z) + image_size * 0.5

    ys, xs = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=torch.float32),
        torch.arange(image_size, device=device, dtype=torch.float32),
        indexing="ij",
    )

    dx = xs[None, :, :] - x[:, None, None]
    dy = ys[None, :, :] - y[:, None, None]
    sigma = model.scales.squeeze(1)[:, None, None]
    weights = torch.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
    weights = weights * model.opacities.squeeze(1)[:, None, None]
    weights = weights * valid[:, None, None].float()

    colors = model.colors.view(-1, 3, 1, 1)
    rgb = (weights[:, None, :, :] * colors).sum(dim=0)
    alpha = weights.sum(dim=0, keepdim=True).clamp(min=1e-6)

    # 使用浅灰背景可以避免空白区域对训练产生过强黑色偏置。
    image = rgb / alpha.clamp(min=1.0)
    background = torch.full_like(image, 0.92)
    alpha01 = alpha.clamp(0.0, 1.0)
    return image * alpha01 + background * (1.0 - alpha01)
