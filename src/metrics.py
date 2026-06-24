"""Evaluation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np


def box_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    a_min = box_a[:, :3] - box_a[:, 3:] / 2.0
    a_max = box_a[:, :3] + box_a[:, 3:] / 2.0
    b_min = box_b[:, :3] - box_b[:, 3:] / 2.0
    b_max = box_b[:, :3] + box_b[:, 3:] / 2.0
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = inter_size[:, 0] * inter_size[:, 1] * inter_size[:, 2]
    a_vol = np.prod(np.maximum(a_max - a_min, 0.0), axis=1)
    b_vol = np.prod(np.maximum(b_max - b_min, 0.0), axis=1)
    return inter_vol / (a_vol + b_vol - inter_vol + 1e-9)


def evaluate_predictions(logits: np.ndarray, boxes: np.ndarray, y_class: np.ndarray, y_box: np.ndarray) -> Dict[str, float]:
    pred_class = np.argmax(logits, axis=1)
    accuracy = float(np.mean(pred_class == y_class))
    bbox_mae = float(np.mean(np.abs(boxes - y_box)))
    mean_iou = float(np.mean(box_iou_3d(boxes, y_box)))
    return {
        "accuracy": accuracy,
        "bbox_mae": bbox_mae,
        "mean_iou": mean_iou,
    }

