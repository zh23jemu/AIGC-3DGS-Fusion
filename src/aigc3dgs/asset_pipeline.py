"""题目一资产准备流程清单生成工具。

本模块对应项目要求中的“3D 资产准备”部分。它不会伪装已经完成外部
threestudio / Zero123 的长时间训练，而是把三类资产的输入、推荐工具、
统一表示和交付检查项写成机器可读 JSON，便于报告、融合脚本和后续真实
生产流程复用。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AssetBranch:
    """描述一个 3D 资产来源分支。

    字段说明：
    - name：资产编号，与题目中的 A / B / C 对应。
    - source：资产来源类型。
    - recommended_tool：推荐使用的外部工具或本项目脚本。
    - input_requirement：准备该资产时需要的输入数据。
    - output_format：建议输出格式，方便后续统一融合。
    - project_status：本交付包内的实现状态。
    - notes：关键注意事项，帮助接手者继续扩展。
    """

    name: str
    source: str
    recommended_tool: str
    input_requirement: str
    output_format: str
    project_status: str
    notes: str


def default_asset_branches() -> list[AssetBranch]:
    """返回题目一要求的 A/B/C 三类 3D 资产准备方案。"""

    return [
        AssetBranch(
            name="A",
            source="真实多视角重建资产",
            recommended_tool="COLMAP + 3DGS；本项目可用 aigc3dgs.train 训练轻量 3DGS 权重",
            input_requirement="同一真实物体或小场景的多视角图片、相机内参和外参",
            output_format=".pth 3DGS checkpoint + .ply Gaussian/point-cloud export",
            project_status="已在 data/toy_scene 上实现多视角 3DGS 训练与 PLY 导出",
            notes="真实数据接入时先用 COLMAP 求位姿，再转换为 transforms.json 后训练。",
        ),
        AssetBranch(
            name="B",
            source="文本生成 3D 资产",
            recommended_tool="threestudio + Stable-DreamFusion / SDS",
            input_requirement="文本提示词、负向提示词、训练步数、NeRF/mesh/3DGS 导出配置",
            output_format="mesh(.obj/.glb) 或转换后的 .ply/.pth Gaussian 表示",
            project_status="本交付包提供接入清单和融合接口，未内置 threestudio 长训练结果",
            notes="生成 mesh 后可在 Blender 中放置，也可采样点云并初始化为 Gaussian。",
        ),
        AssetBranch(
            name="C",
            source="单图生成 3D 资产",
            recommended_tool="Zero123 / Zero123++ / image-to-3D pipeline",
            input_requirement="单张物体参考图、前景 mask、相机假设或生成视角配置",
            output_format="mesh(.obj/.glb) 或转换后的 .ply/.pth Gaussian 表示",
            project_status="本交付包提供接入清单和融合接口，未内置 Zero123 长训练结果",
            notes="单图生成结果通常几何背面不确定，评估时需单独说明视角一致性风险。",
        ),
    ]


def write_manifest(path: str | Path) -> Path:
    """把 A/B/C 三类资产准备方案写入 JSON 文件。"""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "AIGC-3DGS-Fusion",
        "requirement": "题目一 1：3D 资产准备",
        "unified_representation": "优先统一为 3D Gaussian checkpoint 与 PLY；mesh 资产可通过 Blender 或点采样转换后融合。",
        "assets": [asdict(branch) for branch in default_asset_branches()],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成题目一 A/B/C 资产准备流程清单。")
    parser.add_argument(
        "--out",
        default="runs/asset_pipeline_manifest.json",
        help="输出 JSON 清单路径，默认写入 runs/asset_pipeline_manifest.json。",
    )
    args = parser.parse_args()

    out = write_manifest(args.out)
    print(f"资产准备清单已写入：{out}")


if __name__ == "__main__":
    main()
