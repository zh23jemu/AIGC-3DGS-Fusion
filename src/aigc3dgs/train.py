"""训练教学版 3D Gaussian Splatting 模型。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .cameras import load_views
from .gaussian_model import GaussianScene, render
from .ply_io import checkpoint_dict, save_ply


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """保存 `[3, H, W]` 图像张量到 PNG 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练教学版 3D Gaussian Splatting。")
    parser.add_argument("--data", default="data/toy_scene", help="包含 transforms.json 的数据目录。")
    parser.add_argument("--out", default="runs/toy_3dgs", help="训练输出目录。")
    parser.add_argument("--steps", type=int, default=800, help="训练迭代步数。")
    parser.add_argument("--gaussians", type=int, default=96, help="可学习 3D 高斯数量。")
    parser.add_argument("--image-size", type=int, default=96, help="训练图像分辨率。")
    parser.add_argument("--lr", type=float, default=0.03, help="Adam 学习率。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    views = load_views(args.data, image_size=args.image_size)
    if not views:
        raise RuntimeError("未读取到任何训练视角，请先准备数据。")

    model = GaussianScene(args.gaussians).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = Path(args.out)

    losses: list[float] = []
    progress = tqdm(range(1, args.steps + 1), desc="training")
    for step in progress:
        view, target = random.choice(views)
        target = target.to(device)

        pred = render(model, view, args.image_size)
        mse = torch.mean((pred - target) ** 2)
        scale_reg = 0.0005 * torch.mean(model.scales ** 2)
        loss = mse + scale_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        value = float(loss.detach().cpu())
        losses.append(value)
        if step % 20 == 0 or step == 1:
            progress.set_postfix(loss=f"{value:.5f}", mse=f"{float(mse.detach().cpu()):.5f}")

    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_dict(model, args.image_size), ckpt_dir / "model_final.pth")
    save_ply(model, ckpt_dir / "model_final.ply")

    render_dir = out / "renders"
    for idx, (view, _) in enumerate(views[: min(6, len(views))]):
        with torch.no_grad():
            image = render(model, view, args.image_size)
        save_image(image, render_dir / f"train_view_{idx:03d}.png")

    with (out / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"final_loss={losses[-1]:.8f}\n")
        f.write(f"steps={args.steps}\n")
        f.write(f"gaussians={args.gaussians}\n")
        f.write(f"device={device}\n")

    print(f"训练完成，权重已保存到 {ckpt_dir}")


if __name__ == "__main__":
    main()
