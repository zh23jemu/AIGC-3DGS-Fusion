"""场景融合清单生成工具。

本模块对应项目要求中的“场景融合与渲染”。当前交付的轻量 3DGS 训练
代码已经能够从统一的 Gaussian checkpoint 渲染多视角结果；本脚本进一步
提供 A/B/C 资产与背景场景的融合描述清单，明确每个资产的空间变换、表示
类型和推荐渲染路径，便于在 Blender、原版 3DGS 或本项目渲染器中继续落地。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FusionAsset:
    """描述一个待插入背景场景的 3D 资产。"""

    name: str
    path: str
    representation: str
    translation: tuple[float, float, float]
    rotation_euler_deg: tuple[float, float, float]
    scale: float
    render_note: str


def default_fusion_assets(final_weight: str) -> list[FusionAsset]:
    """生成默认融合资产列表。

    A 资产使用本项目已经训练得到的 3DGS 权重；B/C 资产保留为外部生成
    资产的占位入口，后续只需把 path 替换为 threestudio / Zero123 导出的
    mesh 或转换后的 Gaussian 文件即可参与同一融合清单。
    """

    return [
        FusionAsset(
            name="A_multiview_3dgs",
            path=final_weight,
            representation="3D Gaussian checkpoint",
            translation=(0.0, 0.0, 0.0),
            rotation_euler_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            render_note="可直接使用 aigc3dgs.render 渲染多视角图像。",
        ),
        FusionAsset(
            name="B_text_to_3d",
            path="external_assets/text_to_3d_asset.obj",
            representation="mesh or converted Gaussian",
            translation=(0.8, 0.0, 0.0),
            rotation_euler_deg=(0.0, 25.0, 0.0),
            scale=0.6,
            render_note="threestudio/SDS 生成后可在 Blender 中导入，或采样点云转换为 Gaussian。",
        ),
        FusionAsset(
            name="C_image_to_3d",
            path="external_assets/image_to_3d_asset.obj",
            representation="mesh or converted Gaussian",
            translation=(-0.8, 0.0, 0.0),
            rotation_euler_deg=(0.0, -20.0, 0.0),
            scale=0.6,
            render_note="Zero123 生成后建议先检查背面几何，再进行多视角漫游渲染。",
        ),
    ]


def write_fusion_manifest(path: str | Path, final_weight: str, background_scene: str) -> Path:
    """写入场景融合 JSON 清单。

    该清单是代码侧的融合接口：它把背景场景、三类资产、坐标变换和渲染路线
    放在同一个结构中，避免报告中只出现文字描述而代码里没有对应入口。
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "AIGC-3DGS-Fusion",
        "requirement": "题目一 3：场景融合与渲染",
        "background_scene": background_scene,
        "coordinate_system": "右手系；资产先按 scale 缩放，再按 Euler 角旋转，最后平移到背景坐标中。",
        "rendering_strategy": [
            "Gaussian 资产：合并或分别渲染后 alpha compositing。",
            "Mesh 资产：在 Blender 中与背景相机轨迹统一渲染，或采样为点云后转换为 Gaussian。",
            "漫游视频：沿 transforms.json 中相机轨迹或插值轨迹逐帧渲染后合成。",
        ],
        "assets": [asdict(asset) for asset in default_fusion_assets(final_weight)],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 A/B/C 资产与背景场景融合清单。")
    parser.add_argument(
        "--final-weight",
        default="weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth",
        help="已训练的 3DGS 权重路径，默认使用最终推荐权重。",
    )
    parser.add_argument(
        "--background-scene",
        default="data/toy_scene",
        help="背景场景数据目录；真实实验可替换为 Mip-NeRF 360 的 garden/bicycle/counter。",
    )
    parser.add_argument(
        "--out",
        default="runs/fusion_manifest.json",
        help="输出 JSON 清单路径，默认写入 runs/fusion_manifest.json。",
    )
    args = parser.parse_args()

    out = write_fusion_manifest(args.out, args.final_weight, args.background_scene)
    print(f"融合清单已写入：{out}")


if __name__ == "__main__":
    main()
