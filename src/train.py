"""Training loops for the small NumPy experiment."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.losses import (
    cross_entropy_and_grad,
    feature_loss_and_grad,
    kd_logits_loss_and_grad,
    mse_and_grad,
    relation_loss_and_grad,
)
from src.metrics import evaluate_predictions


def iterate_batches(x, y_class, y_box, batch_size: int, rng: np.random.Generator):
    order = rng.permutation(x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        idx = order[start:start + batch_size]
        yield x[idx], y_class[idx], y_box[idx]


def evaluate_model(model, data: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = model.forward(data["x"], cache=False)
    return evaluate_predictions(out["logits"], out["box"], data["y_class"], data["y_box"])


def train_supervised(
    model,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    epochs: int = 18,
    batch_size: int = 32,
    lr: float = 0.04,
    seed: int = 0,
    box_weight: float = 2.0,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for xb, yc, yb in iterate_batches(train_data["x"], train_data["y_class"], train_data["y_box"], batch_size, rng):
            out = model.forward(xb, cache=True)
            ce, grad_logits = cross_entropy_and_grad(out["logits"], yc)
            box_loss, grad_box = mse_and_grad(out["box"], yb)
            model.backward(out["cache"], grad_logits, box_weight * grad_box, lr)
            total_loss += ce + box_weight * box_loss
            batches += 1
        if epoch == 1 or epoch == epochs or epoch % 6 == 0:
            metrics = evaluate_model(model, val_data)
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(total_loss / max(batches, 1))
            history.append(metrics)
    return {"history": history, "final": evaluate_model(model, val_data)}


def train_with_distillation(
    student,
    teacher,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    epochs: int = 18,
    batch_size: int = 32,
    lr: float = 0.04,
    seed: int = 1,
    alpha_out: float = 0.45,
    alpha_feature: float = 0.20,
    alpha_relation: float = 0.08,
    temperature: float = 2.0,
    box_weight: float = 2.0,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for xb, yc, yb in iterate_batches(train_data["x"], train_data["y_class"], train_data["y_box"], batch_size, rng):
            s = student.forward(xb, cache=True)
            t = teacher.forward(xb, cache=False)
            ce, grad_logits = cross_entropy_and_grad(s["logits"], yc)
            box_loss, grad_box = mse_and_grad(s["box"], yb)

            kd_loss, grad_kd_logits = kd_logits_loss_and_grad(s["logits"], t["logits"], temperature)
            box_kd_loss, grad_kd_box = mse_and_grad(s["box"], t["box"])
            feat_loss, grad_feat = feature_loss_and_grad(s["feature"], t["feature"])
            rel_loss, grad_rel = relation_loss_and_grad(s["feature"], t["feature"])

            grad_logits = grad_logits + alpha_out * grad_kd_logits
            grad_box = box_weight * grad_box + alpha_out * grad_kd_box
            grad_feature = alpha_feature * grad_feat + alpha_relation * grad_rel
            student.backward(s["cache"], grad_logits, grad_box, lr, grad_feature=grad_feature)

            total_loss += ce + box_weight * box_loss + alpha_out * (kd_loss + box_kd_loss) + alpha_feature * feat_loss + alpha_relation * rel_loss
            batches += 1
        if epoch == 1 or epoch == epochs or epoch % 6 == 0:
            metrics = evaluate_model(student, val_data)
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(total_loss / max(batches, 1))
            history.append(metrics)
    return {"history": history, "final": evaluate_model(student, val_data)}
