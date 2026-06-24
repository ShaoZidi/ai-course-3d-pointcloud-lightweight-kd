"""Run the reproducible course experiment and save metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, make_dataset
from src.metrics import box_iou_3d
from src.models import BaselineMLP, LightweightMLP
from src.train import evaluate_model, train_supervised, train_with_distillation


OUTPUTS = ROOT / "outputs"


def percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def _font(size: int):
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _box_to_xy(box, grid_shape, cell):
    _, height, width = grid_shape
    cy, cx = float(box[1]) * height, float(box[2]) * width
    sy, sx = max(float(box[4]) * height, 0.3), max(float(box[5]) * width, 0.3)
    x0 = (cx - sx / 2.0) * cell
    y0 = (cy - sy / 2.0) * cell
    x1 = (cx + sx / 2.0) * cell
    y1 = (cy + sy / 2.0) * cell
    return [x0, y0, x1, y1]


def _draw_bev_panel(draw, origin, bev, gt_box, pred_box, title, pred_label, iou, grid_shape, cell=18):
    x0, y0 = origin
    panel_w = grid_shape[2] * cell
    panel_h = grid_shape[1] * cell
    font_title = _font(24)
    font_small = _font(18)
    draw.text((x0, y0), title, font=font_title, fill=(25, 45, 80))
    top = y0 + 34
    draw.rectangle((x0, top, x0 + panel_w, top + panel_h), fill=(250, 252, 255), outline=(160, 170, 185), width=2)
    for yy in range(grid_shape[1]):
        for xx in range(grid_shape[2]):
            value = float(bev[yy, xx])
            if value > 0:
                shade = int(245 - min(value, 1.0) * 155)
                draw.rectangle(
                    (x0 + xx * cell + 1, top + yy * cell + 1, x0 + (xx + 1) * cell - 1, top + (yy + 1) * cell - 1),
                    fill=(shade, shade, shade),
                )
    for xx in range(grid_shape[2] + 1):
        draw.line((x0 + xx * cell, top, x0 + xx * cell, top + panel_h), fill=(230, 235, 242))
    for yy in range(grid_shape[1] + 1):
        draw.line((x0, top + yy * cell, x0 + panel_w, top + yy * cell), fill=(230, 235, 242))

    gt_xy = _box_to_xy(gt_box, grid_shape, cell)
    gt_xy = [gt_xy[0] + x0, gt_xy[1] + top, gt_xy[2] + x0, gt_xy[3] + top]
    draw.rectangle(gt_xy, outline=(42, 160, 80), width=4)
    if pred_box is not None:
        pred_xy = _box_to_xy(pred_box, grid_shape, cell)
        pred_xy = [pred_xy[0] + x0, pred_xy[1] + top, pred_xy[2] + x0, pred_xy[3] + top]
        draw.rectangle(pred_xy, outline=(210, 70, 70), width=4)
    draw.text((x0, top + panel_h + 10), "绿:真实框  红:预测框", font=font_small, fill=(60, 60, 60))
    if pred_label is not None:
        draw.text((x0, top + panel_h + 35), f"预测: {pred_label}  IoU={iou:.2f}", font=font_small, fill=(60, 60, 60))


def save_qualitative_figure(models, val_data, grid_shape, output_path: Path) -> None:
    x = val_data["x"]
    y_class = val_data["y_class"]
    y_box = val_data["y_box"]
    samples = []
    kd_model = models.get("Lightweight + KD")
    kd_out = kd_model.forward(x, cache=False) if kd_model is not None else None
    for cls in range(len(CLASS_NAMES)):
        candidates = np.where(y_class == cls)[0]
        if kd_out is not None:
            pred_cls = np.argmax(kd_out["logits"][candidates], axis=1)
            ious = box_iou_3d(kd_out["box"][candidates], y_box[candidates])
            correct_bonus = (pred_cls == cls).astype(np.float32) * 0.4
            idx = int(candidates[np.argmax(ious + correct_bonus)])
        else:
            idx = int(candidates[0])
        samples.append(idx)

    cell = 18
    panel_w = grid_shape[2] * cell
    col_gap = 70
    row_gap = 92
    left = 55
    top = 105
    row_h = grid_shape[1] * cell + 92
    width = left * 2 + 4 * panel_w + 3 * col_gap
    height = top + len(samples) * row_h + (len(samples) - 1) * row_gap + 35
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(36)
    draw.text((left, 34), "图4  定性效果图：BEV 投影中的真实框与预测框对比", font=title_font, fill=(25, 45, 80))

    for r, idx in enumerate(samples):
        row_y = top + r * (row_h + row_gap)
        voxel = x[idx].reshape(grid_shape)
        bev = voxel.max(axis=0)
        draw.text((left, row_y - 36), f"样本类别：{CLASS_NAMES[int(y_class[idx])]}", font=_font(24), fill=(40, 40, 40))
        _draw_bev_panel(draw, (left, row_y), bev, y_box[idx], None, "输入 + GT", None, 0.0, grid_shape, cell)
        for c, (name, model) in enumerate(models.items(), start=1):
            out = model.forward(x[idx:idx + 1], cache=False)
            pred_box = out["box"][0]
            pred_cls = int(np.argmax(out["logits"], axis=1)[0])
            iou = float(box_iou_3d(pred_box[None, :], y_box[idx:idx + 1])[0])
            _draw_bev_panel(
                draw,
                (left + c * (panel_w + col_gap), row_y),
                bev,
                y_box[idx],
                pred_box,
                name,
                CLASS_NAMES[pred_cls],
                iou,
                grid_shape,
                cell,
            )
    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight 3D point cloud course experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Check imports and model construction without training.")
    args = parser.parse_args()

    OUTPUTS.mkdir(exist_ok=True)
    grid_shape = (8, 12, 12)
    train = make_dataset(n_samples=720, seed=2026, grid_shape=grid_shape)
    val = make_dataset(n_samples=240, seed=2027, grid_shape=grid_shape)
    input_dim = train["x"].shape[1]

    baseline = BaselineMLP(input_dim=input_dim, hidden_dim=96, feature_dim=32, seed=10)
    lightweight_plain = LightweightMLP(input_dim=input_dim, bottleneck_dim=24, feature_dim=32, groups=4, seed=11)
    lightweight_kd = LightweightMLP(input_dim=input_dim, bottleneck_dim=24, feature_dim=32, groups=4, seed=11)

    if args.dry_run:
        print("dry run ok")
        print("input_dim", input_dim)
        print("baseline_parameters", baseline.count_parameters_and_ops()["parameters"])
        print("lightweight_parameters", lightweight_plain.count_parameters_and_ops()["parameters"])
        return

    baseline_train = train_supervised(baseline, train, val, epochs=24, batch_size=32, lr=0.045, seed=20, box_weight=3.0)
    plain_train = train_supervised(lightweight_plain, train, val, epochs=18, batch_size=32, lr=0.050, seed=21, box_weight=3.0)
    kd_train = train_with_distillation(
        lightweight_kd,
        baseline,
        train,
        val,
        epochs=18,
        batch_size=32,
        lr=0.050,
        seed=22,
        box_weight=3.0,
        alpha_out=0.55,
        alpha_feature=0.10,
        alpha_relation=0.01,
    )

    qualitative_path = OUTPUTS / "qualitative_results.png"
    save_qualitative_figure(
        {
            "Baseline": baseline,
            "Lightweight": lightweight_plain,
            "Lightweight + KD": lightweight_kd,
        },
        val,
        grid_shape,
        qualitative_path,
    )

    results = {
        "task": "synthetic 3D voxel object classification and box regression",
        "class_names": list(CLASS_NAMES),
        "grid_shape": list(grid_shape),
        "models": {
            "Baseline teacher": {
                "stats": baseline.count_parameters_and_ops(),
                "metrics": baseline_train["final"],
                "history": baseline_train["history"],
            },
            "Lightweight student": {
                "stats": lightweight_plain.count_parameters_and_ops(),
                "metrics": plain_train["final"],
                "history": plain_train["history"],
            },
            "Lightweight + KD": {
                "stats": lightweight_kd.count_parameters_and_ops(),
                "metrics": kd_train["final"],
                "history": kd_train["history"],
            },
        },
        "paper_reference": {
            "SECOND_KITTI_BEV_mAP": 63.3,
            "LW_SECOND_3_KITTI_BEV_mAP": 48.7,
            "LW_SECOND_3_KD_KITTI_BEV_mAP": 62.7,
            "SECOND_FLOPs_G": 69.8,
            "LW_SECOND_3_FLOPs_G": 18.5,
            "SECOND_params_M": 5.34,
            "LW_SECOND_3_params_M": 0.67,
        },
        "qualitative_figure": str(qualitative_path.relative_to(ROOT)),
    }

    metrics_path = OUTPUTS / "metrics.json"
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "| 模型 | 参数量 | 单样本乘加量 | 分类准确率 | BBox MAE | mean IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, info in results["models"].items():
        stats = info["stats"]
        metrics = info["metrics"]
        lines.append(
            f"| {name} | {stats['parameters']} | {stats['ops_per_sample']} | "
            f"{percent(metrics['accuracy'])} | {metrics['bbox_mae']:.4f} | {metrics['mean_iou']:.4f} |"
        )
    table_path = OUTPUTS / "results_table.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Saved", metrics_path)
    print("Saved", table_path)
    print("Saved", qualitative_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
