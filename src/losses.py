"""Loss helpers for supervised learning and knowledge distillation."""

from __future__ import annotations

from typing import Dict

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_and_grad(logits: np.ndarray, y_class: np.ndarray):
    probs = softmax(logits)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y_class] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(n), y_class] -= 1.0
    grad /= n
    return float(loss), grad


def mse_and_grad(pred: np.ndarray, target: np.ndarray):
    diff = pred - target
    loss = float(np.mean(diff * diff))
    grad = 2.0 * diff / diff.size
    return loss, grad


def kd_logits_loss_and_grad(student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float = 2.0):
    student_probs = softmax(student_logits, temperature)
    teacher_probs = softmax(teacher_logits, temperature)
    loss = np.mean(np.sum(teacher_probs * (np.log(teacher_probs + 1e-12) - np.log(student_probs + 1e-12)), axis=1))
    grad = (student_probs - teacher_probs) / student_logits.shape[0]
    grad *= temperature
    return float(loss), grad


def relation_loss_and_grad(student_feat: np.ndarray, teacher_feat: np.ndarray):
    dim = min(student_feat.shape[1], teacher_feat.shape[1])
    s = student_feat[:, :dim]
    t = teacher_feat[:, :dim]
    gs = (s @ s.T) / max(dim, 1)
    gt = (t @ t.T) / max(dim, 1)
    diff = gs - gt
    loss = float(np.mean(diff * diff))
    grad_shared = (4.0 / (s.shape[0] * s.shape[0] * max(dim, 1))) * (diff @ s)
    grad = np.zeros_like(student_feat)
    grad[:, :dim] = grad_shared
    return loss, grad


def feature_loss_and_grad(student_feat: np.ndarray, teacher_feat: np.ndarray):
    dim = min(student_feat.shape[1], teacher_feat.shape[1])
    diff = student_feat[:, :dim] - teacher_feat[:, :dim]
    loss = float(np.mean(diff * diff))
    grad = np.zeros_like(student_feat)
    grad[:, :dim] = 2.0 * diff / diff.size
    return loss, grad


def distillation_loss_parts(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    student_feat: np.ndarray,
    teacher_feat: np.ndarray,
    student_box: np.ndarray,
    teacher_box: np.ndarray,
    temperature: float = 2.0,
) -> Dict[str, float]:
    """Return scalar loss parts inspired by Lout, Lfeature and LFSP."""

    out_loss, _ = kd_logits_loss_and_grad(student_logits, teacher_logits, temperature)
    box_loss, _ = mse_and_grad(student_box, teacher_box)
    feat_loss, _ = feature_loss_and_grad(student_feat, teacher_feat)
    rel_loss, _ = relation_loss_and_grad(student_feat, teacher_feat)
    return {
        "Lout": out_loss + box_loss,
        "Lfeature": feat_loss,
        "LFSP": rel_loss,
    }

