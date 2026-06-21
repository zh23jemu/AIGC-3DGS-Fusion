"""加载模型权重并渲染测试视角。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .cameras import load_views
from .gaussian_model import GaussianScene, render


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """保存渲染结果图像。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染教学版 3DGS 模型。")
    parser.add_argument("--checkpoint", required=True, help="训练得到的 .pth 权重路径。")
    parser.add_argument("--data", default="data/toy_scene", help="包含 transforms.json 的数据目录。")
    parser.add_argument("--out", default="runs/toy_3dgs/test_renders", help="渲染输出目录。")
    parser.add_argument("--image-size", type=int, default=None, help="输出图像分辨率，默认使用权重记录值。")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = int(args.image_size or ckpt.get("image_size", 96))

    model = GaussianScene(int(ckpt["num_gaussians"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    views = load_views(args.data, image_size=image_size)
    out = Path(args.out)
    for idx, (view, _) in enumerate(views):
        with torch.no_grad():
            image = render(model, view, image_size)
        save_image(image, out / f"render_{idx:03d}.png")

    print(f"已渲染 {len(views)} 个视角到 {out}")


if __name__ == "__main__":
    main()
