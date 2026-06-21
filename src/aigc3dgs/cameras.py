"""相机与数据集读取工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


@dataclass
class CameraView:
    """保存单个视角的图像路径与相机参数。

    参数说明：
    - image_path：该视角对应的 RGB 图像文件。
    - camera_to_world：4x4 相机到世界坐标变换矩阵。
    - focal：针孔相机焦距，单位为像素。
    - width / height：图像宽高。
    """

    image_path: Path
    camera_to_world: torch.Tensor
    focal: float
    width: int
    height: int


def load_views(data_dir: str | Path, image_size: int | None = None) -> list[tuple[CameraView, torch.Tensor]]:
    """读取 `transforms.json` 格式的小型多视角数据集。

    返回值中的图像张量形状为 `[3, H, W]`，数值范围为 `[0, 1]`。如果传入
    `image_size`，会在读取时统一缩放，保证训练和测试分辨率一致。
    """

    root = Path(data_dir)
    meta_path = root / "transforms.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    loaded: list[tuple[CameraView, torch.Tensor]] = []
    for frame in meta["frames"]:
        image_path = root / frame["file_path"]
        image = Image.open(image_path).convert("RGB")
        if image_size is not None:
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)

        width, height = image.size
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
        tensor = tensor.view(height, width, 3).permute(2, 0, 1) / 255.0

        focal = float(frame.get("focal", meta.get("focal", width)))
        if image_size is not None:
            # 样例数据使用方形图像，缩放后焦距按宽度比例同步缩放。
            original_width = float(frame.get("width", meta.get("width", width)))
            focal *= width / max(original_width, 1.0)

        view = CameraView(
            image_path=image_path,
            camera_to_world=torch.tensor(frame["transform_matrix"], dtype=torch.float32),
            focal=focal,
            width=width,
            height=height,
        )
        loaded.append((view, tensor))

    return loaded


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor | None = None) -> torch.Tensor:
    """构造相机到世界矩阵。

    这里采用常见 OpenGL 风格约定：相机局部坐标的 `-Z` 方向看向目标点。
    """

    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    true_up = torch.cross(right, forward, dim=0)

    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, 0] = right
    matrix[:3, 1] = true_up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye
    return matrix
