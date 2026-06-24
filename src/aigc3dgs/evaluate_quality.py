"""质量评估表生成工具。

本模块对应项目要求中的“质量评估与技术报告”。它把多视角重建、文本
生成 3D、单图生成 3D 三条路线放到同一张 Markdown 表中，对几何准确性、
纹理细节、计算成本和当前交付状态进行对比，避免评估只停留在报告文字中。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityRow:
    """一行质量评估记录。"""

    method: str
    geometry: str
    texture: str
    compute_cost: str
    reproducibility: str
    project_evidence: str


def default_quality_rows() -> list[QualityRow]:
    """返回题目一 4 要求的三类方法对比。"""

    return [
        QualityRow(
            method="多视角重建：COLMAP + 3DGS / 本项目轻量 3DGS",
            geometry="多视角约束强，几何一致性最好；依赖相机位姿质量",
            texture="来自真实图片，纹理可信度高，边缘锐度取决于 rasterizer 与高斯密度",
            compute_cost="中等到较高；真实大场景需要较长 GPU 训练",
            reproducibility="高；数据、位姿、训练参数固定后可重复",
            project_evidence="已训练 q2 权重，final_loss=0.00914205，导出 .pth 和 .ply",
        ),
        QualityRow(
            method="文本生成 3D：threestudio + SDS",
            geometry="可生成无参考资产，但几何可能受 prompt 和先验影响",
            texture="语义丰富，细节可能出现多面不一致或过度平滑",
            compute_cost="较高；SDS 优化通常需要较多迭代",
            reproducibility="中等；受随机种子、扩散模型版本和 prompt 影响",
            project_evidence="提供 asset_pipeline 与 fusion_manifest 接入位置",
        ),
        QualityRow(
            method="单图生成 3D：Zero123",
            geometry="正面一致性较好，背面和遮挡区域不确定",
            texture="与输入图正面相似，侧后方纹理依赖生成先验",
            compute_cost="中等；通常低于从零文本优化，高于普通重建渲染",
            reproducibility="中等；依赖输入 mask、视角假设和模型版本",
            project_evidence="提供 asset_pipeline 与 fusion_manifest 接入位置",
        ),
    ]


def to_markdown(rows: list[QualityRow]) -> str:
    """把评估记录转换为 Markdown 表格。"""

    lines = [
        "# AIGC-3DGS-Fusion 质量评估表",
        "",
        "| 方法 | 几何准确性 | 纹理细节 | 计算成本 | 可复现性 | 项目证据 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.method} | {row.geometry} | {row.texture} | "
            f"{row.compute_cost} | {row.reproducibility} | {row.project_evidence} |"
        )
    lines.append("")
    lines.append("最终推荐权重：`weights/aws_ec2_toy_3dgs_q2_20k_1024/model_final.pth`。")
    return "\n".join(lines)


def write_quality_report(path: str | Path) -> Path:
    """写出质量评估 Markdown 文件。"""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(default_quality_rows()), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="生成题目一质量评估 Markdown 表。")
    parser.add_argument(
        "--out",
        default="runs/quality_evaluation.md",
        help="输出 Markdown 路径，默认写入 runs/quality_evaluation.md。",
    )
    args = parser.parse_args()

    out = write_quality_report(args.out)
    print(f"质量评估表已写入：{out}")


if __name__ == "__main__":
    main()
