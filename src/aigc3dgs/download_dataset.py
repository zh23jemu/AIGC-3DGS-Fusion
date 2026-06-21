"""下载作业推荐的 Mip-NeRF 360 数据集。"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


MIPNERF360_MAIN_URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
MIPNERF360_EXTRA_URL = "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"


def download_file(url: str, dst: Path) -> None:
    """以流式方式下载大文件，并显示进度条。

    官方 Mip-NeRF 360 压缩包体积较大，下载时可能受网络影响。函数会使用
    `requests` 流式读取，避免一次性占用大量内存。
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with dst.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Mip-NeRF 360 官方数据集。")
    parser.add_argument("--out", default="data", help="数据保存目录。")
    parser.add_argument("--scene", default="garden", help="期望使用的场景名，仅用于下载后提示。")
    parser.add_argument("--extra", action="store_true", help="下载额外场景包，而不是 garden/bicycle/counter 所在的主包。")
    parser.add_argument("--no-extract", action="store_true", help="只下载 zip，不自动解压。")
    args = parser.parse_args()

    out = Path(args.out)
    url = MIPNERF360_EXTRA_URL if args.extra else MIPNERF360_MAIN_URL
    zip_path = out / ("360_extra_scenes.zip" if args.extra else "360_v2.zip")
    if not zip_path.exists():
        print(f"开始下载 Mip-NeRF 360 官方数据集：{url}")
        download_file(url, zip_path)
    else:
        print(f"已存在压缩包，跳过下载：{zip_path}")

    if not args.no_extract:
        print("开始解压数据集，可能需要较长时间和较大磁盘空间。")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out)
        print(f"解压完成。可检查场景目录：{out / args.scene}")


if __name__ == "__main__":
    main()
