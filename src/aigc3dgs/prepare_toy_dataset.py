"""生成本机快速训练用的小型多视角数据集。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .cameras import look_at


def draw_view(angle: float, size: int) -> Image.Image:
    """根据相机角度绘制一个有视差变化的合成目标。

    该数据不追求真实照片级复杂度，而是让训练脚本在本机能快速验证：
    多视角输入、相机位姿、可微高斯渲染和权重保存这条链路是否完整。
    """

    image = Image.new("RGB", (size, size), (235, 238, 242))
    draw = ImageDraw.Draw(image)

    cx = size * 0.5 + math.cos(angle) * size * 0.08
    cy = size * 0.52 + math.sin(angle * 1.7) * size * 0.035
    shadow = [cx - size * 0.26, cy + size * 0.20, cx + size * 0.26, cy + size * 0.29]
    draw.ellipse(shadow, fill=(175, 182, 190))

    body = [cx - size * 0.20, cy - size * 0.16, cx + size * 0.20, cy + size * 0.18]
    draw.ellipse(body, fill=(55, 125, 215), outline=(24, 70, 150), width=max(1, size // 64))

    highlight_x = cx - math.cos(angle) * size * 0.08
    highlight = [
        highlight_x - size * 0.08,
        cy - size * 0.12,
        highlight_x + size * 0.03,
        cy - size * 0.02,
    ]
    draw.ellipse(highlight, fill=(145, 205, 255))

    red_x = cx + math.sin(angle) * size * 0.16
    draw.rectangle(
        [red_x - size * 0.07, cy + size * 0.03, red_x + size * 0.07, cy + size * 0.15],
        fill=(230, 85, 70),
        outline=(145, 35, 30),
        width=max(1, size // 80),
    )

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="生成教学版 3DGS 的小型多视角数据。")
    parser.add_argument("--out", default="data/toy_scene", help="输出数据目录。")
    parser.add_argument("--views", type=int, default=20, help="生成视角数量。")
    parser.add_argument("--size", type=int, default=96, help="图像宽高。")
    args = parser.parse_args()

    out = Path(args.out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(args.views):
        angle = 2.0 * math.pi * i / args.views
        image = draw_view(angle, args.size)
        file_path = f"images/view_{i:03d}.png"
        image.save(out / file_path)

        eye = torch.tensor([math.sin(angle) * 2.4, 0.35, math.cos(angle) * 2.4], dtype=torch.float32)
        target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        c2w = look_at(eye, target)

        frames.append(
            {
                "file_path": file_path,
                "transform_matrix": c2w.tolist(),
                "focal": float(args.size * 0.92),
                "width": args.size,
                "height": args.size,
            }
        )

    meta = {"width": args.size, "height": args.size, "focal": float(args.size * 0.92), "frames": frames}
    with (out / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"已生成 {args.views} 个视角到 {out}")


if __name__ == "__main__":
    main()
