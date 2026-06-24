"""Build Markdown, LaTeX and PDF versions of the course report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ASSETS = REPORT_DIR / "assets"
OUTPUTS = ROOT / "outputs"
METRICS_PATH = OUTPUTS / "metrics.json"
FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")
PDF_OUTPUT = REPORT_DIR / "course_report_submit_v3.pdf"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("请先运行 python src/run_experiment.py 生成 outputs/metrics.json")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def model_rows(metrics: dict):
    rows = []
    for name, info in metrics["models"].items():
        stat = info["stats"]
        m = info["metrics"]
        rows.append([
            name,
            str(stat["parameters"]),
            str(stat["ops_per_sample"]),
            pct(m["accuracy"]),
            f"{m['bbox_mae']:.4f}",
            f"{m['mean_iou']:.4f}",
        ])
    return rows


def get_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def rounded_box(draw: ImageDraw.ImageDraw, xy, fill, outline, width=3, radius=20):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(30, 30, 30), spacing=8):
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + ((y1 - y0) - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x0 + ((x1 - x0) - w) / 2, y), line, font=font, fill=fill)
        y += h + spacing


def arrow(draw: ImageDraw.ImageDraw, start, end, fill=(60, 90, 140), width=4):
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        pts = [(ex, ey), (ex - direction * 18, ey - 9), (ex - direction * 18, ey + 9)]
    else:
        direction = 1 if ey >= sy else -1
        pts = [(ex, ey), (ex - 9, ey - direction * 18), (ex + 9, ey - direction * 18)]
    draw.polygon(pts, fill=fill)


def build_baseline_flow_png() -> None:
    ASSETS.mkdir(exist_ok=True)
    img = PILImage.new("RGB", (1500, 720), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(42)
    font = get_font(29)
    small = get_font(23)
    draw.text((42, 34), "图1  基线实验流程：标准特征提取与监督训练", font=title_font, fill=(20, 40, 70))

    blue = (225, 237, 252)
    orange = (255, 239, 217)
    green = (226, 244, 232)
    red = (255, 230, 230)
    outline = (75, 105, 160)

    boxes = {
        "input": (70, 165, 295, 265),
        "voxel": (375, 165, 625, 265),
        "feature": (710, 130, 980, 300),
        "heads": (1070, 120, 1390, 310),
        "label": (710, 435, 980, 555),
        "loss": (1070, 430, 1390, 565),
    }
    rounded_box(draw, boxes["input"], blue, outline)
    centered_text(draw, boxes["input"], "原始点云\n或体素网格", font)
    rounded_box(draw, boxes["voxel"], blue, outline)
    centered_text(draw, boxes["voxel"], "体素化与\n特征编码", font)
    rounded_box(draw, boxes["feature"], orange, (180, 115, 40))
    centered_text(draw, boxes["feature"], "基线模型\n标准特征提取层\n参数量较大", small)
    rounded_box(draw, boxes["heads"], green, (70, 140, 90))
    centered_text(draw, boxes["heads"], "分类头 + 框回归头\n输出目标类别\n和 3D 框参数", small)
    rounded_box(draw, boxes["label"], blue, outline)
    centered_text(draw, boxes["label"], "真实标签\n类别 + 3D 框", font)
    rounded_box(draw, boxes["loss"], red, (170, 80, 80))
    centered_text(draw, boxes["loss"], "监督损失\nLcls + lambda_box Lbox\n反向更新基线模型", small)

    arrow(draw, (295, 215), (375, 215))
    arrow(draw, (625, 215), (710, 215))
    arrow(draw, (980, 215), (1070, 215))
    arrow(draw, (1230, 310), (1230, 430))
    arrow(draw, (980, 495), (1070, 495))
    arrow(draw, (1070, 495), (930, 300), fill=(170, 80, 80), width=3)

    draw.text((84, 625), "基线作用：先验证任务可学习性，并作为后续轻量模型和知识蒸馏实验的对照与教师模型。", font=small, fill=(80, 80, 80))
    img.save(ASSETS / "baseline_flow.png")


def build_method_flow_png() -> None:
    ASSETS.mkdir(exist_ok=True)
    img = PILImage.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(42)
    font = get_font(29)
    small = get_font(24)
    draw.text((42, 34), "图2  创新实验流程：轻量化结构与知识蒸馏", font=title_font, fill=(20, 40, 70))

    blue = (225, 237, 252)
    green = (226, 244, 232)
    orange = (255, 239, 217)
    purple = (237, 230, 250)
    outline = (75, 105, 160)

    boxes = {
        "input": (70, 160, 300, 260),
        "voxel": (380, 160, 650, 260),
        "teacher": (760, 95, 1035, 215),
        "student": (760, 285, 1035, 425),
        "distill": (1120, 215, 1405, 345),
        "output": (1120, 455, 1405, 585),
    }
    rounded_box(draw, boxes["input"], blue, outline)
    centered_text(draw, boxes["input"], "原始点云\n或体素网格", font)
    rounded_box(draw, boxes["voxel"], blue, outline)
    centered_text(draw, boxes["voxel"], "体素化与\n特征编码", font)
    rounded_box(draw, boxes["teacher"], orange, (180, 115, 40))
    centered_text(draw, boxes["teacher"], "基线教师模型\n标准特征提取", font)
    rounded_box(draw, boxes["student"], green, (70, 140, 90))
    centered_text(draw, boxes["student"], "轻量学生模型\n分组变换 + 通道混合", font)
    rounded_box(draw, boxes["distill"], purple, (115, 85, 160))
    centered_text(draw, boxes["distill"], "知识蒸馏约束\nLout / Lfeature / LFSP", small)
    rounded_box(draw, boxes["output"], blue, outline)
    centered_text(draw, boxes["output"], "目标类别\n和 3D 框预测", font)

    arrow(draw, (300, 210), (380, 210))
    arrow(draw, (650, 190), (760, 155))
    arrow(draw, (650, 230), (760, 355))
    arrow(draw, (1035, 155), (1120, 250))
    arrow(draw, (1035, 355), (1120, 300))
    arrow(draw, (900, 425), (1120, 520))
    arrow(draw, (1260, 345), (1260, 455))

    draw.text((84, 625), "设计思路：保留点云检测的基本输入-特征-预测流程，用轻量模块降低复杂度，再用教师模型提供软标签和中间特征监督。", font=small, fill=(80, 80, 80))
    img.save(ASSETS / "method_flow.png")


def build_results_png(metrics: dict) -> None:
    ASSETS.mkdir(exist_ok=True)
    img = PILImage.new("RGB", (1600, 980), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(48)
    font = get_font(34)
    small = get_font(26)
    tiny = get_font(22)
    draw.text((55, 42), "图3  定量指标对比：复杂度、分类准确率与框误差", font=title_font, fill=(20, 40, 70))

    names = list(metrics["models"].keys())
    display_names = ["基线模型", "轻量模型", "轻量+蒸馏"]
    base_params = metrics["models"]["Baseline teacher"]["stats"]["parameters"]
    base_ops = metrics["models"]["Baseline teacher"]["stats"]["ops_per_sample"]
    colors_used = [(70, 116, 180), (238, 137, 42), (82, 160, 88)]

    card_fill = (248, 250, 253)
    card_edge = (205, 215, 230)
    panels = {
        "complex": (60, 145, 1540, 395),
        "acc": (60, 435, 770, 910),
        "mae": (830, 435, 1540, 910),
    }
    for xy in panels.values():
        draw.rounded_rectangle(xy, radius=24, fill=card_fill, outline=card_edge, width=3)

    x0, y0, x1, y1 = panels["complex"]
    draw.text((x0 + 32, y0 + 24), "A. 模型复杂度压缩（基线=100%，越低越好）", font=font, fill=(30, 40, 60))
    bar_x = x0 + 265
    bar_w = 1040
    for i, name in enumerate(names):
        stat = metrics["models"][name]["stats"]
        rel_param = stat["parameters"] / base_params * 100
        rel_ops = stat["ops_per_sample"] / base_ops * 100
        y = y0 + 85 + i * 52
        draw.text((x0 + 35, y - 4), display_names[i], font=small, fill=(30, 30, 30))
        draw.rectangle((bar_x, y, bar_x + bar_w, y + 20), fill=(230, 235, 242))
        draw.rectangle((bar_x, y, bar_x + int(bar_w * rel_param / 100), y + 20), fill=colors_used[i])
        draw.text((bar_x + bar_w + 18, y - 6), f"参数 {rel_param:.1f}%", font=tiny, fill=(50, 50, 50))
        y2 = y + 24
        draw.rectangle((bar_x, y2, bar_x + bar_w, y2 + 20), fill=(230, 235, 242))
        draw.rectangle((bar_x, y2, bar_x + int(bar_w * rel_ops / 100), y2 + 20), fill=colors_used[i])
        draw.text((bar_x + bar_w + 18, y2 - 6), f"乘加 {rel_ops:.1f}%", font=tiny, fill=(50, 50, 50))

    x0, y0, x1, y1 = panels["acc"]
    draw.text((x0 + 32, y0 + 24), "B. 分类准确率（越高越好）", font=font, fill=(30, 40, 60))
    axis_bottom = y1 - 105
    axis_left = x0 + 85
    chart_h = 310
    draw.line((axis_left, axis_bottom, x1 - 45, axis_bottom), fill=(90, 90, 90), width=3)
    draw.line((axis_left, axis_bottom - chart_h, axis_left, axis_bottom), fill=(90, 90, 90), width=3)
    for i, name in enumerate(names):
        acc = metrics["models"][name]["metrics"]["accuracy"] * 100
        h = int(chart_h * acc / 100)
        bx = axis_left + 70 + i * 165
        by = axis_bottom - h
        draw.rounded_rectangle((bx, by, bx + 95, axis_bottom), radius=10, fill=colors_used[i])
        draw.text((bx + 2, by - 35), f"{acc:.2f}%", font=small, fill=(25, 25, 25))
        centered_text(draw, (bx - 28, axis_bottom + 16, bx + 123, axis_bottom + 75), display_names[i], tiny, fill=(60, 60, 60), spacing=2)

    x0, y0, x1, y1 = panels["mae"]
    draw.text((x0 + 32, y0 + 24), "C. BBox MAE（越低越好）", font=font, fill=(30, 40, 60))
    max_mae = 0.16
    axis_bottom = y1 - 105
    axis_left = x0 + 85
    chart_h = 310
    draw.line((axis_left, axis_bottom, x1 - 45, axis_bottom), fill=(90, 90, 90), width=3)
    draw.line((axis_left, axis_bottom - chart_h, axis_left, axis_bottom), fill=(90, 90, 90), width=3)
    for i, name in enumerate(names):
        mae = metrics["models"][name]["metrics"]["bbox_mae"]
        h = int(chart_h * mae / max_mae)
        bx = axis_left + 70 + i * 165
        by = axis_bottom - h
        draw.rounded_rectangle((bx, by, bx + 95, axis_bottom), radius=10, fill=colors_used[i])
        draw.text((bx - 2, by - 35), f"{mae:.4f}", font=small, fill=(25, 25, 25))
        centered_text(draw, (bx - 28, axis_bottom + 16, bx + 123, axis_bottom + 75), display_names[i], tiny, fill=(60, 60, 60), spacing=2)

    img.save(ASSETS / "results_visual.png")


def build_visual_assets(metrics: dict) -> None:
    build_baseline_flow_png()
    build_method_flow_png()
    build_results_png(metrics)
    qualitative_src = OUTPUTS / "qualitative_results.png"
    if qualitative_src.exists():
        shutil.copyfile(qualitative_src, ASSETS / "qualitative_results.png")


def build_markdown(metrics: dict) -> str:
    rows = model_rows(metrics)
    paper = metrics["paper_reference"]
    local_table = "\n".join(
        ["| 模型 | 参数量 | 单样本乘加量 | 分类准确率 | BBox MAE | mean IoU |",
         "|---|---:|---:|---:|---:|---:|"]
        + [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in rows]
    )
    compression = metrics["models"]["Baseline teacher"]["stats"]["parameters"] / metrics["models"]["Lightweight student"]["stats"]["parameters"]
    op_compression = metrics["models"]["Baseline teacher"]["stats"]["ops_per_sample"] / metrics["models"]["Lightweight student"]["stats"]["ops_per_sample"]

    return f"""# 基于轻量化网络与知识蒸馏的 3D 点云目标检测方法设计与分析

## 摘要

3D 点云目标检测是自动驾驶、机器人感知和三维场景理解中的重要任务。与普通二维图像相比，点云数据具有稀疏、不规则和空间结构明显等特点，因此检测模型通常需要较复杂的特征提取结构。复杂模型可以提升检测效果，但也会带来参数量大、计算量高、部署困难等问题。本文以 Li 等人在 Applied Sciences 2023 发表的 A Lightweight Model for 3D Point Cloud Object Detection 为参考[1]，借鉴其中轻量化稀疏卷积和知识蒸馏的思想，设计了一个面向课程实验的 3D 体素目标检测方案。本文没有把目标设定为照搬原论文工程系统，而是在课程条件下围绕“基线实现、轻量化改进、蒸馏增强、结果比较”建立完整实验流程。

本文首先构建了一个小规模合成 3D 体素数据集，将目标分为 car、pedestrian 和 cyclist 三类，并为每个样本设置类别标签和归一化三维框。基线模型采用标准特征提取结构，用于提供基本检测能力；创新实验一将大规模特征变换替换为分组变换和通道混合结构，以减少模型复杂度；创新实验二在轻量模型基础上加入知识蒸馏，使学生模型学习基线模型的输出分布、中间特征和特征关系。实验结果显示，轻量模型参数量约压缩到基线的 1/{compression:.1f}，单样本乘加量约压缩到 1/{op_compression:.1f}；加入知识蒸馏后，分类准确率由 {rows[1][3]} 提升到 {rows[2][3]}，BBox MAE 由 {rows[1][4]} 降低到 {rows[2][4]}。结果说明，在模型规模显著降低的情况下，蒸馏训练可以帮助轻量模型恢复部分性能。

关键词：3D 点云；目标检测；轻量化模型；知识蒸馏；课程实验

## 1 绪论

### 1.1 研究背景

点云由激光雷达等传感器采集得到，能够描述物体在三维空间中的位置和几何结构。在自动驾驶场景中，车辆需要识别周围的汽车、行人和骑行者，并给出它们的三维框位置。如果检测模型太大，即使精度较高，也很难部署到车载或边缘设备上。因此，如何在尽量保持检测效果的同时降低参数量和计算量，是一个比较实际的问题。

在三维检测任务中，模型不仅要判断目标类别，还要预测目标在三维空间中的位置和尺度。与二维目标检测相比，三维框多了深度、高度和空间方向等信息，数据表达也更复杂。点云通常是稀疏的，很多空间位置没有点；同时点的数量和分布会受传感器距离、遮挡、反射材料等因素影响。因此，三维检测模型往往需要较强的特征提取能力。

但是，强特征提取能力经常意味着更大的模型。对于课程实验来说，如果直接选择一个完整的自动驾驶检测框架，不仅数据量很大，而且还会涉及体素化、anchor 生成、稀疏卷积库、后处理和官方评测协议等工程细节。本文更关注一个清晰的问题：在一个可控的小规模 3D 检测任务中，能否先实现一个基线模型，再用轻量化结构降低复杂度，并用知识蒸馏弥补轻量化带来的性能下降。

### 1.2 本文工作内容

本文围绕参考论文中的核心思想开展课程实验，但重点是设计自己的方案并进行验证。具体来说，本文首先设计了一个小规模 3D 体素目标检测任务，使模型同时完成类别预测和三维框回归；然后实现标准基线模型，作为后续比较的基础；在此基础上设计轻量化学生模型，用分组变换和通道混合减少参数量与计算量；随后在轻量模型上加入知识蒸馏，包括输出蒸馏、特征蒸馏和关系蒸馏；最后从定量指标和定性效果图两个角度比较基线实验与创新实验。

### 1.3 参考论文的作用

本文参考的论文提出了 LW-Sconv 模块和三段式知识蒸馏[1]。论文中的基线是 SECOND[2]，提升方法是在 3D sparse convolution 中加入分组卷积、因子化卷积和深度卷积思想，再让轻量学生模型学习教师模型的输出、特征和特征流关系[1]。该论文对本文的主要启发不是某一个具体代码细节，而是一个研究思路：先建立可靠基线，再从模型结构和训练策略两个方向降低复杂度并恢复性能。

## 2 研究现状

三维目标检测方法大致可以分为点方法、体素方法和混合方法。PointNet 直接处理点集合，VoxelNet 和 SECOND 将点云划分成体素后用卷积提取特征，PointPillars 则把点云划分成柱状结构后投影到鸟瞰图。随着模型越来越深，检测效果通常会提高，但计算量也明显增加。论文中提到的 SECOND 已经使用稀疏卷积提升效率，不过仍然有较高 FLOPs[2]。

轻量化模型设计在二维图像任务中比较常见，例如 MobileNet 使用 depth-wise separable convolution[4]，ShuffleNet 使用 group convolution 和 channel shuffle。论文的主要贡献就是把这些轻量化思想迁移到 3D sparse convolution 中[1]。同时，知识蒸馏让小模型学习大模型的“软知识”，不仅学习真实标签，也学习教师模型的输出分布和中间特征[3]。

### 2.1 基于点的检测方法

基于点的方法直接以点集作为输入，典型代表是 PointNet 及其后续方法。这类方法的优点是保留了点云原始结构，不需要把点云强行转换成规则网格。但点云是无序集合，点的数量也不固定，因此模型需要设计对点排列不敏感的特征聚合方式。PointNet 使用多层感知机提取每个点的特征，再通过最大池化得到全局特征。这种方法结构直观，但对局部几何关系的刻画能力有限。

### 2.2 基于体素的检测方法

基于体素的方法先把三维空间划分为规则网格，再把点云映射到体素中。VoxelNet 和 SECOND 属于这一类。体素化之后，模型可以使用卷积结构提取空间特征，整体流程更接近传统卷积神经网络。缺点是三维体素网格可能非常稀疏，如果使用普通三维卷积，会在大量空位置上浪费计算。SECOND 通过稀疏卷积减少无效计算，是三维检测中比较有代表性的基线方法。

### 2.3 基于 BEV 表示的方法

还有一些方法将点云投影到鸟瞰图，也就是 BEV 表示。PointPillars 将点云划分为柱状结构，再投影到二维平面上处理。BEV 表示的优点是计算更接近二维卷积，速度较快，也便于自动驾驶场景中理解车辆周围平面布局。缺点是高度信息会被压缩，需要通过特征编码尽量保留三维结构。

### 2.4 模型轻量化与知识蒸馏

模型轻量化的目标是在尽量少损失性能的情况下减少参数量、计算量和存储开销。常见方式包括剪枝、量化、低秩分解、深度可分离卷积、分组卷积和通道混合等。本文主要借鉴分组变换和通道混合思想[1]。知识蒸馏则是一种训练策略，它让小模型学习大模型的输出分布和中间特征[3]。与只学习硬标签相比，蒸馏提供的信息更丰富。例如教师模型可能认为某个样本 80% 像汽车、15% 像骑行者、5% 像行人，这种软分布比单纯的 one-hot 标签更能体现类别之间的相似关系。

## 3 方法

### 3.1 参考方法的启发

参考论文的基本流程是：原始点云首先体素化，经 VFE 编码后进入 3D backbone；3D 特征沿 z 轴变换到 BEV 表示，再经过 2D backbone 和检测头输出类别与框。其关键启发在于，检测模型不一定只能靠堆叠更复杂的网络提升效果，也可以通过重新设计卷积计算方式来降低复杂度。

LW-Sconv 模块先用 1 x 1 x 1 分组卷积降低通道计算，再通过 transpose 和 reshape 促进不同组之间的信息流动，最后用 3 x 3 x 3 depth-wise sparse convolution 扩大感受野。这个设计给本文的启发是：在课程实验中也可以把“大矩阵特征提取”替换为“分组变换 + 通道混合”的轻量结构。

论文还使用三部分蒸馏损失：

- Lout：让学生模型学习教师模型的输出软标签，包括分类和框回归输出。
- Lfeature：让学生模型模仿进入检测头前的特征图。
- LFSP：用 FSP 矩阵描述特征流关系，让学生模型模仿教师模型的特征提取过程。

### 3.2 任务定义

本文将课程实验任务定义为一个简化的 3D 体素目标检测问题。每个输入样本是一个三维体素网格，网格中的值表示该位置是否存在点以及点的强度。模型需要输出两个结果：第一是目标类别，类别包括 car、pedestrian 和 cyclist；第二是目标三维框，使用中心位置和尺寸进行表示，即 center_z、center_y、center_x、size_z、size_y、size_x。为了便于训练，所有框参数都归一化到 0 到 1 的范围内。

这个任务虽然比真实自动驾驶数据集简单，但仍保留了 3D 检测的两个核心部分：类别识别和空间定位。分类任务考察模型能否理解不同物体的几何形态，框回归任务考察模型能否估计目标在三维空间中的位置和大小。

### 3.3 创新实验总体流程

本文方案保留“点云体素化 - 特征提取 - 类别与三维框预测”的主线，同时将参考论文中的轻量化和蒸馏思想转化为课程环境中可以验证的模型结构。创新实验流程如图1所示。

![创新实验总体方法流程图](assets/method_flow.png)

在具体模型上，基线模型使用标准特征提取层处理体素输入，并同时输出目标类别和归一化三维框。轻量模型将第一层大规模参数矩阵替换为分组线性变换，再加入通道混合层，模拟 group convolution 与 channel shuffle 的作用。蒸馏增强模型在普通监督损失之外，引入输出蒸馏、特征蒸馏和关系蒸馏，使轻量学生模型不仅学习真实标签，也学习基线模型形成的软分布和中间表达。

### 3.4 损失函数设计

本文模型同时完成分类和三维框回归，因此监督损失由分类损失和框回归损失组成。分类部分使用交叉熵损失，框回归部分使用均方误差。总监督损失可以写成：Lsup = Lcls + lambda_box Lbox。其中 Lcls 衡量类别预测和真实类别之间的差异，Lbox 衡量预测框和真实框之间的误差，lambda_box 用来控制框回归在总损失中的权重。

知识蒸馏部分由三项组成。第一项是输出蒸馏 Lout，使学生模型的类别分布和框输出接近教师模型。第二项是特征蒸馏 Lfeature，使学生模型的中间特征接近教师模型的中间特征。第三项是关系蒸馏 Lrelation，用特征之间的相似关系约束学生模型。最终创新模型的训练目标可以概括为：L = Lsup + alpha_out Lout + alpha_feature Lfeature + alpha_relation Lrelation。这样设计的目的，是让学生模型在容量较小的情况下，仍然尽可能学习教师模型已经形成的判别信息。

### 3.5 评价指标原理

本文使用四类指标评价实验结果。第一是分类准确率，即预测类别正确的样本比例。第二是 BBox MAE，即预测三维框参数和真实框参数之间的平均绝对误差，数值越低表示框回归越准确。第三是 mean IoU，即预测框和真实框的平均三维交并比，数值越高表示空间重叠越好。第四是模型复杂度，包括参数量和单样本乘加量，用于衡量模型大小和计算成本。

需要说明的是，分类准确率、BBox MAE 和 mean IoU 关注的是不同方面。一个模型可能分类更准，但框重叠不一定更好；也可能框参数平均误差降低，但某些样本的 IoU 仍然较低。因此，结果分析不能只看单一指标，而要结合多项指标和定性效果图一起判断。

### 3.6 基线实验流程

基线实验用于回答“这个任务能否被一个标准模型学习”这一问题。它不引入轻量化结构，也不使用知识蒸馏，而是采用直接的监督训练方式。基线实验流程如图2所示。

![基线实验流程图](assets/baseline_flow.png)

### 3.7 数据构造与输入表示

为了验证方法是否具有合理趋势，本文构建了一个小规模合成 3D 体素数据集。每个样本被表示为 8 x 12 x 12 的体素网格，目标类别包括 car、pedestrian 和 cyclist。不同类别由不同几何结构表示：car 采用扁长的长方体结构，pedestrian 采用高而窄的柱状结构，cyclist 采用两个轮子加车架的稀疏结构。这样设计是为了让三类目标在几何形态上有明显差异，同时仍然保留点云稀疏和带噪声的特点。

数据中还加入少量随机噪声点，用来模拟真实点云中的离群点和背景干扰。每个样本的标签包括类别标签和三维框参数。三维框用中心点和尺寸表示，分别对应 z、y、x 三个方向。由于体素网格尺寸固定，框参数被归一化到 0 到 1，便于模型学习。

### 3.8 基线模型结构

基线模型的作用是提供一个可以比较的标准实现。本文的基线模型采用标准特征提取结构：首先将三维体素网格展平成向量，然后经过两层全连接特征变换，得到一个中间特征向量。最后模型分成两个输出头：分类头输出三类目标的 logits，框回归头输出六个归一化框参数。

这种基线结构虽然没有真实 SECOND 那样复杂，但适合作为课程实验中的起点。它参数量较大，表达能力较强，能够学习输入体素和输出标签之间的基本映射。后续轻量化模型和蒸馏模型都与它进行比较，因此基线必须先单独训练并评估。

### 3.9 基线训练过程

基线模型使用监督学习训练。每个批次输入体素特征，模型输出类别预测和框预测，然后计算分类交叉熵和框回归误差。训练过程中，分类损失推动模型区分 car、pedestrian 和 cyclist，框回归损失推动模型预测目标中心和大小。由于框定位对检测任务很重要，本文对框回归损失设置了较高权重，使模型不仅关注分类，也学习空间定位。

基线实验的意义有两点。第一，它验证了任务本身是否可以被模型学习；如果基线模型都无法取得合理结果，则说明数据设计或训练流程存在问题。第二，它为创新实验提供教师模型。知识蒸馏中的学生模型需要学习教师模型的软标签和中间特征，因此基线模型不只是对照组，也是蒸馏训练中的知识来源。

### 3.10 基线实验观察

从最终结果看，基线模型分类准确率为 {rows[0][3]}，BBox MAE 为 {rows[0][4]}，mean IoU 为 {rows[0][5]}。这说明标准特征提取结构能够较好地区分三类目标，并且对三维框有一定定位能力。同时，基线模型参数量为 {rows[0][1]}，单样本乘加量为 {rows[0][2]}，明显高于后续轻量模型。也就是说，基线模型效果较稳定，但复杂度较高。

### 3.11 创新实验设计

### 3.12 创新实验一：轻量化学生模型

第一个创新实验是轻量化结构设计。参考 LW-Sconv 的思想，本文没有直接使用完整的大矩阵特征变换，而是将输入特征按组划分，每组只连接到一部分瓶颈通道。这样做类似于分组卷积，可以显著减少参数量。随后，模型对分组后的通道进行混合，使不同组之间的信息能够重新流动，避免每一组只看到局部输入导致表达能力下降。

如果使用普通全连接层，输入维度为 D，输出维度为 H，则参数量大约为 D x H。分组结构把输入和输出拆成 g 组，每组只需要 (D/g) x (H/g) 的参数，总参数量约为 D x H / g。虽然实际模型还包含后续混合层和输出头，但第一层大规模参数已经被明显压缩。这正是轻量化模型参数量降低的主要原因。

轻量模型的风险在于容量变小后表达能力下降。因为每个组最开始只能处理一部分输入，模型可能难以捕捉全局结构。通道混合层就是为了解决这个问题：它让分组后的特征重新组合，使后续特征可以同时包含不同输入区域的信息。

### 3.13 创新实验二：知识蒸馏增强

第二个创新实验是在轻量模型上加入知识蒸馏。蒸馏训练的基本思想是：基线模型作为教师，轻量模型作为学生。学生不只学习真实标签，还学习教师模型的输出分布和中间表达。对于分类任务，教师模型的 softmax 输出比 one-hot 标签包含更多信息。例如某个样本真实类别是 cyclist，但教师模型可能同时给 car 较高概率，这说明这两个类别在某些几何结构上有相似性。学生学习这种软分布，有助于获得更平滑的决策边界。

在本文实验中，蒸馏损失分为三部分。输出蒸馏约束学生的分类输出和框输出接近教师；特征蒸馏约束学生中间特征接近教师；关系蒸馏约束学生样本间特征相似关系接近教师。三者分别对应“学结果”“学表示”“学关系”。这种训练方式比只使用真实标签更充分，也更符合参考论文中三段式知识迁移的思想。

### 3.14 基线实现与创新实验的区别

基线实现和创新实验的区别可以从两个层面理解。结构层面上，基线模型使用标准特征变换，参数量较大；创新实验一使用分组瓶颈结构，参数量和计算量显著降低。训练层面上，基线模型只使用真实标签进行监督；创新实验二在真实标签之外加入教师模型提供的软信息，使轻量模型能够获得额外指导。

因此，本文的比较不是简单地比较三个模型谁更高，而是回答三个问题：第一，标准基线能否完成这个 3D 检测任务；第二，轻量化结构能在多大程度上降低复杂度；第三，知识蒸馏能否在轻量化之后恢复一部分性能。这样的实验逻辑也更接近一般论文中的 ablation study。

## 4 实验与结果

### 4.1 数据集划分

本文共生成 960 个样本，其中 720 个用于训练，240 个用于验证。训练集和验证集使用不同随机种子生成，避免模型只记住某一批固定样本。每个样本包含一个目标，目标类别在三类之间随机选择。由于本文任务是课程级方法验证，数据规模没有追求很大，而是强调每个样本标签清晰、结构可控、能够稳定比较不同模型。

### 4.2 模型设置

实验设置三个模型。第一是 Baseline teacher，对应基线实现。第二是 Lightweight student，对应仅使用轻量化结构的学生模型。第三是 Lightweight + KD，对应加入知识蒸馏训练的轻量学生模型。三个模型使用相同的数据集和评价指标，保证比较具有一致性。

### 4.3 评价方式

评价分为定量和定性两部分。定量评价包括参数量、单样本乘加量、分类准确率、BBox MAE 和 mean IoU。参数量和乘加量用于衡量模型复杂度，分类准确率用于衡量类别判断能力，BBox MAE 和 mean IoU 用于衡量空间定位能力。定性评价通过 BEV 投影图展示真实框和预测框的位置关系，能够直观看出模型预测是否贴近目标。

### 4.4 定量结果

{local_table}

![复杂度和性能指标可视化](assets/results_visual.png)

从表格和图3可以看到，轻量模型的参数量从 {rows[0][1]} 降到 {rows[1][1]}，单样本乘加量从 {rows[0][2]} 降到 {rows[1][2]}，压缩幅度比较明显。按参数量计算，轻量模型约为基线的 1/{compression:.1f}；按乘加量计算，轻量模型约为基线的 1/{op_compression:.1f}。这说明分组瓶颈结构确实达到了降低复杂度的目标。

从检测效果看，普通轻量模型分类准确率为 {rows[1][3]}，低于基线模型的 {rows[0][3]}，说明单纯压缩模型会带来一定性能损失。加入知识蒸馏后，Lightweight + KD 的分类准确率提升到 {rows[2][3]}，超过普通轻量模型，也略高于基线模型。BBox MAE 方面，普通轻量模型为 {rows[1][4]}，蒸馏模型降至 {rows[2][4]}，说明蒸馏对框参数平均误差也有一定改善。

mean IoU 的变化没有分类准确率那么理想。蒸馏模型 mean IoU 为 {rows[2][5]}，略低于普通轻量模型的 {rows[1][5]}。这说明本文蒸馏策略主要改善了类别判断和框参数平均误差，但对框的空间重叠质量提升不足。原因可能是本文使用的框回归损失是参数级误差，而不是直接优化 IoU，因此模型可能在中心和尺寸的平均误差上变小，但局部样本的重叠效果没有同步提升。

### 4.5 定性效果图

为了更直观地观察检测效果，本文将验证样本投影到 BEV 平面，并同时绘制真实框和预测框。绿色框表示真实框，红色框表示模型预测框。图4选取 car、pedestrian 和 cyclist 三类代表样本，对比了基线模型、普通轻量模型和蒸馏模型的预测效果。

![定性检测效果图](assets/qualitative_results.png)

从图4可以看到，模型不仅给出类别预测，也给出了对应的目标位置。对于 car 和 cyclist 这类形状较明显的样本，蒸馏模型的预测框与真实框重叠较好，说明学生模型能够从教师模型中学习到一定定位能力。对于 pedestrian 这类较小目标，预测仍然更困难，因为目标在 BEV 平面上的占用区域很小，少量误差就会导致 IoU 明显下降。这也解释了为什么本文整体 mean IoU 不如分类准确率好看。

### 4.6 基线与创新实验比较

基线模型的优势是结构直接、容量较大、训练稳定。它在定位指标上表现最好，mean IoU 达到 {rows[0][5]}，说明较大的模型容量对框回归有帮助。但基线模型的缺点也很明显：参数量和计算量都远高于轻量模型。

普通轻量模型的优势是复杂度最低，参数量只有 {rows[1][1]}，乘加量只有 {rows[1][2]}。它的问题是单纯压缩后性能下降，分类准确率和框回归误差都不如基线。这个结果符合轻量化模型的常见规律：模型变小以后，表达能力会受到限制。

蒸馏增强模型在轻量模型基础上加入教师指导，分类准确率和 BBox MAE 都优于普通轻量模型。它说明知识蒸馏可以缓解轻量化带来的部分性能损失。不过，蒸馏模型的 mean IoU 没有超过普通轻量模型，说明当前蒸馏方式还没有完全解决空间定位质量问题。

## 5 总结与展望

### 5.1 与参考论文结果的关系

参考论文在 KITTI 上报告的 BEV mAP 中，SECOND 为 {paper['SECOND_KITTI_BEV_mAP']}，LW-SECOND-3 为 {paper['LW_SECOND_3_KITTI_BEV_mAP']}，加入三部分知识蒸馏后为 {paper['LW_SECOND_3_KD_KITTI_BEV_mAP']}。论文还显示，FLOPs 从 {paper['SECOND_FLOPs_G']}G 降到 {paper['LW_SECOND_3_FLOPs_G']}G，参数量从 {paper['SECOND_params_M']}M 降到 {paper['LW_SECOND_3_params_M']}M。由此可以看出，参考论文提供的核心经验是：单纯压缩会带来精度下降，而知识蒸馏可以帮助轻量模型恢复一部分效果。

本文实验与参考论文在数据规模和模型工程上不同，但实验逻辑是一致的。本文同样先建立基线，再进行轻量化压缩，最后使用蒸馏方法改善轻量模型。不同之处在于，本文采用的是课程环境中可控的小规模体素任务，因此结果不能与 KITTI 或 nuScenes 直接比较。本文更关注趋势验证：轻量化是否能显著降低复杂度，蒸馏是否能改善轻量模型。实验结果表明，这两个趋势都能在本文任务中观察到。

### 5.2 讨论与不足

参考论文比较有价值的一点是，它没有简单地追求“大模型更准”，而是把部署成本放进了问题本身。实际应用中，模型能否运行、能否实时运行，和精度同样重要。LW-Sconv 模块的设计也说明，很多二维视觉中的轻量化思想并不是只能用于图像任务，只要理解卷积计算的本质，就可以迁移到三维稀疏卷积中。

本文实验也暴露了几个问题。第一，真实 3D 检测系统工程量很大，数据处理、体素化、anchor、NMS、评估协议都很复杂，课程实验只能抽取其中最核心的思想进行验证。第二，知识蒸馏不是“必然提升所有指标”，本文中 mean IoU 没有提升，说明蒸馏损失权重和框回归设计还需要进一步调参。第三，轻量化模型虽然参数少，但如果实现不够高效，理论 FLOPs 降低不一定等于真实速度提升，这也是以后继续做实验时需要验证的。

### 5.3 数据集规模限制

本文使用的是合成体素数据，优点是结构可控、便于快速验证，缺点是与真实自动驾驶点云仍有明显差距。真实点云中存在遮挡、远距离稀疏、传感器噪声、复杂背景和多目标重叠等问题，而本文每个样本只包含一个目标。因此，本文结果只能说明方法在简化任务中有效，不能直接代表真实道路场景效果。

### 5.4 模型结构限制

本文使用的模型是课程级简化结构，没有实现完整的 3D sparse convolution、BEV backbone、anchor 生成和 NMS 后处理。因此，它更适合说明轻量化和蒸馏的原理，而不是替代真实检测框架。若后续继续完善，可以基于 OpenPCDet 或类似框架，将轻量模块放入真实 3D backbone 中验证。

### 5.5 指标表现分析

本文分类准确率较高，但 mean IoU 相对一般，说明模型对类别模式学习较好，对精确框定位仍有不足。造成这一现象的原因可能有三点。第一，体素分辨率较低，目标框位置变化一个格子就会明显影响 IoU。第二，框回归使用的是参数误差，而不是 IoU loss。第三，蒸馏损失更偏向输出分布和中间特征，对空间框重叠的直接约束不够。后续可以尝试加入 IoU loss 或者对中心点、尺寸设置不同权重。

### 5.6 可改进方向

后续可以从三个方向改进。第一，扩大数据规模并加入更多变化，例如目标旋转、多目标场景和不同噪声强度。第二，改进模型结构，例如加入简单的三维卷积或 BEV 卷积，使模型更好地保留空间邻域信息。第三，改进蒸馏策略，例如分别蒸馏分类头和回归头，或者增加与 IoU 相关的蒸馏目标。这样可以进一步提高定位指标，使实验更接近真实检测任务。

### 5.7 总结

本文围绕轻量化 3D 点云目标检测完成了一次课程实验设计。报告首先分析了参考论文中的 SECOND 基线、LW-Sconv 轻量模块和三段式知识蒸馏方法；随后结合课程条件，设计了一个小规模 3D 体素目标检测任务，并比较了标准基线模型、轻量化模型和蒸馏增强模型。

实验结果表明，轻量模型可以大幅减少参数量和计算量，但单纯轻量化会带来一定性能损失。加入知识蒸馏后，轻量模型分类准确率和 BBox MAE 得到改善，说明教师模型提供的软标签和特征信息对学生模型有帮助。定性效果图也表明，模型能够在 BEV 投影中给出较直观的目标框预测。

整体来看，本文完成的不是对某个开源工程的直接照搬，而是围绕参考论文思想进行的一次方法设计、基线实现、创新实验和结果比较。通过本次作业可以看到，人工智能模型设计不只是追求更高精度，还需要考虑模型复杂度、训练方式、部署条件和评价指标之间的平衡。

## 代码可用性声明

本文相关实验代码已整理并开源至 GitHub，仓库地址为：https://github.com/ShaoZidi/ai-course-3d-pointcloud-lightweight-kd 。

## 参考文献

[1] Li, Z.; Li, Y.; Wang, Y.; Xie, G.; Qu, H.; Lyu, Z. A Lightweight Model for 3D Point Cloud Object Detection. Applied Sciences, 2023, 13(11), 6754.

[2] Yan, Y.; Mao, Y.; Li, B. SECOND: Sparsely Embedded Convolutional Detection. Sensors, 2018.

[3] Hinton, G.; Vinyals, O.; Dean, J. Distilling the Knowledge in a Neural Network. arXiv, 2015.

[4] Howard, A. G. et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv, 2017.
"""


def polish_report_structure(markdown_text: str) -> str:
    """Reshape the generated draft into the required course-report structure."""

    chapter3 = """## 3 方法

### 3.1 任务定义与总体思路

本文将实验任务定义为一个小规模 3D 体素目标检测问题。输入是固定大小的三维体素网格，输出包括目标类别和归一化三维框参数。目标类别包括 car、pedestrian 和 cyclist，三维框使用 center_z、center_y、center_x、size_z、size_y、size_x 表示。这个任务虽然比真实自动驾驶点云检测简单，但仍保留了类别识别和空间定位两个核心目标。

参考论文给本文的主要启发是：三维检测模型不一定只能通过堆叠更复杂的网络提升效果，也可以通过轻量化结构降低复杂度，再用知识蒸馏恢复部分性能[1]。因此，本文方法分为两条线：第一条是基线实验，用标准模型验证任务可学习性；第二条是创新实验，在轻量模型上加入蒸馏训练，与基线进行比较。

### 3.2 基线实验

（a）输入与标签。基线实验的输入为体素化后的 3D 网格。每个样本包含一个目标，标签由目标类别和三维框参数组成。由于体素网格大小固定，三维框参数被归一化到 0 到 1，便于模型学习。

（b）模型结构。基线模型采用标准特征提取结构：首先将三维体素网格转换为特征向量，然后经过标准特征提取层得到中间表示，最后分别接分类头和框回归头。分类头输出三类目标的预测结果，框回归头输出 6 个三维框参数。基线模型参数量较大，表达能力较强，适合作为对照模型。

（c）训练方式。基线模型采用监督学习训练，只使用真实标签进行约束。分类部分使用交叉熵损失，框回归部分使用均方误差。总损失可以写为 Lsup = Lcls + lambda_box Lbox，其中 lambda_box 用来控制框回归损失的权重。基线实验流程如图1所示。

![基线实验流程图](assets/baseline_flow.png)

### 3.3 创新实验

（a）轻量化学生模型。创新实验首先将基线模型中的大规模特征变换替换为分组变换和通道混合结构。分组变换相当于让不同特征组分别进行计算，可以减少参数量和乘加量；通道混合则用于缓解分组后信息不流通的问题，使不同组之间的特征重新融合。

（b）知识蒸馏增强。轻量模型虽然复杂度低，但容量变小后容易损失性能。因此，本文进一步加入知识蒸馏。蒸馏时，基线模型作为教师模型，轻量模型作为学生模型[3]。学生模型不仅学习真实标签，也学习教师模型的输出分布、中间特征和特征关系[1]。这样可以让轻量模型在参数较少的情况下获得更多训练信息。

（c）创新实验与基线的区别。基线实验只使用标准模型和真实标签；创新实验则同时改变模型结构和训练方式。结构上，创新实验使用轻量化分组特征提取；训练上，创新实验加入教师模型提供的软标签和中间特征监督。创新实验总体流程如图2所示。

![创新实验总体方法流程图](assets/method_flow.png)

### 3.4 损失函数与评价指标

本文模型同时进行分类和三维框回归。分类损失用于约束目标类别预测，框回归损失用于约束目标位置和大小。知识蒸馏部分包括三项：Lout 约束学生模型输出接近教师模型，Lfeature 约束学生模型中间特征接近教师模型，Lrelation 约束学生模型样本间特征关系接近教师模型。综合来看，创新实验的目标函数可以表示为：L = Lsup + alpha_out Lout + alpha_feature Lfeature + alpha_relation Lrelation。

评价指标包括参数量、单样本乘加量、分类准确率、BBox MAE 和 mean IoU。参数量和乘加量衡量模型复杂度；分类准确率衡量类别判断能力；BBox MAE 衡量三维框参数误差；mean IoU 衡量预测框与真实框的空间重叠程度。由于这些指标关注角度不同，本文在结果分析中同时使用定量表格、指标图和定性效果图。

"""

    chapter5 = """## 5 总结

本文围绕轻量化 3D 点云目标检测完成了一次课程实验。报告按照课程要求，先介绍任务背景和研究现状，再给出方法设计，随后进行实验验证和结果分析。本文没有把参考论文作为照搬对象，而是借鉴其中轻量化结构和知识蒸馏的思想[1]，设计了基线实验和创新实验，并对二者进行了比较。

实验结果表明，轻量化学生模型能够显著减少参数量和单样本乘加量，但单纯压缩会带来一定性能下降；加入知识蒸馏后，轻量模型的分类准确率和 BBox MAE 得到改善，说明教师模型提供的软标签和中间特征对学生模型有帮助。定性效果图也显示，模型可以在 BEV 投影中给出较直观的目标框预测。

本文仍有不足。由于课程时间和算力限制，实验使用的是小规模合成体素数据，而不是真实 KITTI 或 nuScenes 数据集；模型结构也没有实现完整稀疏卷积检测框架。因此，本文结果主要用于验证方法思路，而不能直接代表真实自动驾驶场景效果。总体来看，本次实验完成了基线实现、方法优化和方案验证，符合课程报告中对实际内容和方法思考的要求。

"""

    start3 = markdown_text.index("## 3 方法")
    start4 = markdown_text.index("## 4 实验与结果")
    markdown_text = markdown_text[:start3] + chapter3 + markdown_text[start4:]

    start5 = markdown_text.index("## 5 总结与展望")
    start_refs = markdown_text.index("## 参考文献")
    code_availability = """## 代码可用性声明

本文相关实验代码已整理并开源至 GitHub，仓库地址为：https://github.com/ShaoZidi/ai-course-3d-pointcloud-lightweight-kd 。

"""
    markdown_text = markdown_text[:start5] + chapter5 + code_availability + markdown_text[start_refs:]
    return markdown_text


def build_tex(markdown_text: str) -> str:
    body = markdown_text
    replacements = [
        ("# ", "\\section*{"),
        ("## ", "\\section{"),
        ("### ", "\\subsection{"),
    ]
    lines = []
    in_table = False
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            if in_table:
                lines.append("\\end{tabular}\\end{center}")
                in_table = False
            lines.append("")
            continue
        if line.startswith("|"):
            if line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not in_table:
                lines.append("\\begin{center}\\small\\begin{tabular}{lrrrrr}\\hline")
                in_table = True
            lines.append(" & ".join(cells).replace("%", "\\%") + " \\\\")
            if cells[0] == "模型":
                lines.append("\\hline")
            continue
        if in_table:
            lines.append("\\end{tabular}\\end{center}")
            in_table = False
        if line.startswith("![") and "](" in line:
            alt = line[2:line.index("]")]
            path = line[line.index("(") + 1:line.rindex(")")]
            lines.append("\\begin{figure}[htbp]\\centering")
            lines.append("\\includegraphics[width=0.95\\linewidth]{" + path.replace("_", "\\_") + "}")
            lines.append("\\caption{" + alt.replace("_", "\\_") + "}")
            lines.append("\\end{figure}")
            continue
        if raw.startswith("# "):
            lines.append("\\begin{center}\\LARGE " + raw[2:] + "\\end{center}")
        elif raw.startswith("## "):
            lines.append("\\section{" + raw[3:] + "}")
        elif raw.startswith("### "):
            lines.append("\\subsection{" + raw[4:] + "}")
        elif raw.startswith("- "):
            lines.append("\\noindent " + raw.replace("_", "\\_") + "\\\\")
        else:
            lines.append(raw.replace("_", "\\_").replace("%", "\\%") + "\n")
    if in_table:
        lines.append("\\end{tabular}\\end{center}")
    return "\\documentclass[UTF8]{ctexart}\n\\usepackage[a4paper,margin=2.5cm]{geometry}\n\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\graphicspath{{report/}{./}}\n\\title{基于轻量化网络与知识蒸馏的 3D 点云目标检测方法设计与分析}\n\\author{人工智能课程大作业}\n\\date{2026年6月}\n\\begin{document}\n" + "\n".join(lines) + "\n\\end{document}\n"


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("SimHei", 9)
    canvas.drawCentredString(A4[0] / 2.0, 1.0 * cm, str(doc.page))
    canvas.restoreState()


def paragraph_lines(markdown_text: str):
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("|"):
            continue
        if line.startswith("![") and "](" in line:
            alt = line[2:line.index("]")]
            path = line[line.index("(") + 1:line.rindex(")")]
            yield ("image", alt + "|" + path)
            continue
        if line.startswith("# "):
            yield ("title", line[2:])
        elif line.startswith("## "):
            yield ("heading", line[3:])
        elif line.startswith("### "):
            yield ("subheading", line[4:])
        elif line.startswith("- "):
            yield ("body", "- " + line[2:])
        else:
            yield ("body", line)


def build_pdf(markdown_text: str, metrics: dict) -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"找不到中文字体：{FONT_PATH}")
    pdfmetrics.registerFont(TTFont("SimHei", str(FONT_PATH)))
    styles = getSampleStyleSheet()
    title = ParagraphStyle("cn-title", parent=styles["Title"], fontName="SimHei", fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=18)
    heading = ParagraphStyle("cn-heading", parent=styles["Heading1"], fontName="SimHei", fontSize=14, leading=20, spaceBefore=12, spaceAfter=8)
    subheading = ParagraphStyle("cn-subheading", parent=styles["Heading2"], fontName="SimHei", fontSize=12, leading=18, spaceBefore=8, spaceAfter=5)
    body = ParagraphStyle("cn-body", parent=styles["BodyText"], fontName="SimHei", fontSize=10.5, leading=17, firstLineIndent=21, alignment=TA_JUSTIFY, wordWrap="CJK")

    doc = SimpleDocTemplate(str(PDF_OUTPUT), pagesize=A4, rightMargin=2.2 * cm, leftMargin=2.2 * cm, topMargin=2.1 * cm, bottomMargin=1.8 * cm)
    story = []
    inserted_local_table = False

    for kind, text in paragraph_lines(markdown_text):
        if kind == "title":
            story.append(Paragraph(text, title))
            story.append(Spacer(1, 0.2 * cm))
        elif kind == "heading":
            story.append(Paragraph(text, heading))
        elif kind == "subheading":
            story.append(Paragraph(text, subheading))
            if text == "4.4 定量结果" and not inserted_local_table:
                data = [["模型", "参数量", "乘加量", "准确率", "BBox MAE", "mean IoU"]] + model_rows(metrics)
                tbl = Table(data, colWidths=[4.0 * cm, 2.0 * cm, 2.3 * cm, 2.0 * cm, 2.2 * cm, 2.0 * cm])
                tbl.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 0.25 * cm))
                inserted_local_table = True
        elif kind == "image":
            alt, rel_path = text.split("|", 1)
            image_path = REPORT_DIR / rel_path
            img = RLImage(str(image_path))
            max_width = A4[0] - 4.4 * cm
            max_height = 13.2 * cm if "qualitative_results" in rel_path else 16.0 * cm
            scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
            story.append(img)
            story.append(Paragraph(alt, ParagraphStyle("caption", parent=body, fontName="SimHei", fontSize=9, leading=13, alignment=TA_CENTER, firstLineIndent=0)))
            story.append(Spacer(1, 0.18 * cm))
        else:
            story.append(Paragraph(text, body))
            story.append(Spacer(1, 0.08 * cm))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    metrics = load_metrics()
    build_visual_assets(metrics)
    markdown_text = polish_report_structure(build_markdown(metrics))
    (REPORT_DIR / "course_report.md").write_text(markdown_text, encoding="utf-8")
    (REPORT_DIR / "course_report.tex").write_text(build_tex(markdown_text), encoding="utf-8")
    build_pdf(markdown_text, metrics)
    print("Saved", REPORT_DIR / "course_report.md")
    print("Saved", REPORT_DIR / "course_report.tex")
    print("Saved", PDF_OUTPUT)


if __name__ == "__main__":
    main()
