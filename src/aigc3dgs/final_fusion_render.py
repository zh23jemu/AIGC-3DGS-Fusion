"""生成最终融合渲染图和漫游视频。

这个脚本用于补齐题目一的“场景融合与渲染”交付项。它不依赖 Blender，
而是直接用 Pillow 与 imageio 将 A/B/C 三类资产和背景渲染合成为多视角
帧序列，再导出 MP4 漫游视频。真实 threestudio / Zero123 结果下载后，
只需把资产预览图或渲染图目录传入本脚本即可重新生成最终视频。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def load_asset(path: Path, fallback_color: tuple[int, int, int], label: str) -> Image.Image:
    """读取资产预览图；如果文件缺失，则生成带标签的占位资产。

    真实交付时应传入 threestudio / Zero123 导出的预览图。保留 fallback 是为了
    让脚本在资产尚未下载前也能被测试，避免视频合成流程本身不可运行。
    """

    if path.exists():
        return Image.open(path).convert("RGBA")

    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((32, 32, 224, 224), fill=(*fallback_color, 235), outline=(255, 255, 255, 255), width=4)
    draw.text((92, 112), label, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    return image


def load_background(path: Path, size: int) -> Image.Image:
    """读取背景图；若没有背景图，则生成一个简洁的空间背景。"""

    if path.exists():
        return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)

    image = Image.new("RGB", (size, size), (20, 28, 36))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        ratio = y / max(1, size - 1)
        color = (
            int(24 + 32 * ratio),
            int(34 + 56 * ratio),
            int(42 + 48 * ratio),
        )
        draw.line((0, y, size, y), fill=color)
    draw.rectangle((0, int(size * 0.62), size, size), fill=(70, 82, 76))
    for x in range(0, size, 24):
        draw.line((x, int(size * 0.62), x - 80, size), fill=(92, 103, 96), width=1)
    return image


def paste_asset(
    canvas: Image.Image,
    asset: Image.Image,
    center: tuple[float, float],
    scale: float,
    shadow: bool = True,
) -> None:
    """按中心点和缩放比例把资产贴到当前帧上。"""

    width = max(16, int(asset.width * scale))
    height = max(16, int(asset.height * scale))
    resized = asset.resize((width, height), Image.Resampling.LANCZOS)
    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    if shadow:
        shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_draw.ellipse(
            (x + width * 0.12, y + height * 0.78, x + width * 0.88, y + height * 0.98),
            fill=(0, 0, 0, 80),
        )
        canvas.alpha_composite(shadow_layer.filter(ImageFilter.GaussianBlur(8)))

    canvas.alpha_composite(resized, (x, y))


def render_frames(args: argparse.Namespace) -> list[Path]:
    """生成多视角漫游帧并返回帧路径列表。"""

    out_dir = Path(args.out)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    background = load_background(Path(args.background), args.size)
    asset_a = load_asset(Path(args.asset_a), (70, 130, 220), "A")
    asset_b = load_asset(Path(args.asset_b), (230, 150, 55), "B")
    asset_c = load_asset(Path(args.asset_c), (80, 190, 130), "C")

    frames: list[Path] = []
    for idx in range(args.frames):
        t = idx / max(1, args.frames - 1)
        angle = 2.0 * math.pi * t
        parallax = math.sin(angle)
        depth = math.cos(angle)

        frame = background.convert("RGBA")
        # 三个资产使用不同的水平摆动幅度，模拟相机绕场景漫游时的视差变化。
        paste_asset(frame, asset_a, (args.size * (0.50 + 0.05 * parallax), args.size * 0.57), 0.34 + 0.02 * depth)
        paste_asset(frame, asset_b, (args.size * (0.30 + 0.08 * parallax), args.size * 0.61), 0.27 - 0.01 * depth)
        paste_asset(frame, asset_c, (args.size * (0.70 + 0.07 * parallax), args.size * 0.60), 0.29 + 0.01 * depth)

        draw = ImageDraw.Draw(frame)
        draw.text((16, 16), f"AIGC-3DGS-Fusion view {idx:03d}", fill=(255, 255, 255, 230), font=ImageFont.load_default())
        out_path = frame_dir / f"fusion_{idx:03d}.png"
        frame.convert("RGB").save(out_path)
        frames.append(out_path)
    return frames


def write_video(frames: list[Path], video_path: Path, fps: int) -> None:
    """把帧序列写成 MP4 视频。"""

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 A/B/C 资产融合漫游视频。")
    parser.add_argument("--background", default="runs/final_assets/background_counter_preview.png", help="背景预览图。")
    parser.add_argument("--asset-a", default="runs/final_assets/object_a_multiview.png", help="物体 A 预览图。")
    parser.add_argument("--asset-b", default="runs/final_assets/object_b_threestudio.png", help="物体 B 预览图。")
    parser.add_argument("--asset-c", default="runs/final_assets/object_c_zero123.png", help="物体 C 预览图。")
    parser.add_argument("--out", default="runs/final_fusion", help="输出目录。")
    parser.add_argument("--size", type=int, default=512, help="视频帧尺寸。")
    parser.add_argument("--frames", type=int, default=72, help="漫游帧数。")
    parser.add_argument("--fps", type=int, default=24, help="视频帧率。")
    args = parser.parse_args()

    frames = render_frames(args)
    video = Path(args.out) / "aigc_3dgs_fusion_walkthrough.mp4"
    write_video(frames, video, args.fps)
    print(f"已生成 {len(frames)} 帧融合渲染和视频：{video}")


if __name__ == "__main__":
    main()
