"""Small NumPy models for the course project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _he(rng: np.random.Generator, fan_in: int, shape: Tuple[int, ...]) -> np.ndarray:
    return (rng.normal(size=shape) * np.sqrt(2.0 / max(fan_in, 1))).astype(np.float32)


@dataclass
class BaselineMLP:
    input_dim: int
    hidden_dim: int = 96
    feature_dim: int = 32
    num_classes: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = _he(rng, self.input_dim, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.w2 = _he(rng, self.hidden_dim, (self.hidden_dim, self.feature_dim))
        self.b2 = np.zeros(self.feature_dim, dtype=np.float32)
        self.w_cls = _he(rng, self.feature_dim, (self.feature_dim, self.num_classes))
        self.b_cls = np.zeros(self.num_classes, dtype=np.float32)
        self.w_box = _he(rng, self.feature_dim, (self.feature_dim, 6))
        self.b_box = np.zeros(6, dtype=np.float32)

    def forward(self, x: np.ndarray, cache: bool = False) -> Dict[str, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        h1 = relu(z1)
        z2 = h1 @ self.w2 + self.b2
        feat = relu(z2)
        logits = feat @ self.w_cls + self.b_cls
        box_raw = feat @ self.w_box + self.b_box
        box = sigmoid(box_raw)
        out = {"logits": logits, "box": box, "feature": feat}
        if cache:
            out["cache"] = {"x": x, "z1": z1, "h1": h1, "z2": z2, "feat": feat, "box_raw": box_raw, "box": box}
        return out

    def backward(self, cached: Dict[str, np.ndarray], grad_logits: np.ndarray, grad_box: np.ndarray, lr: float, grad_feature=None) -> None:
        c = cached
        grad_box_raw = grad_box * c["box"] * (1.0 - c["box"])
        gw_cls = c["feat"].T @ grad_logits
        gb_cls = grad_logits.sum(axis=0)
        gw_box = c["feat"].T @ grad_box_raw
        gb_box = grad_box_raw.sum(axis=0)
        grad_feat = grad_logits @ self.w_cls.T + grad_box_raw @ self.w_box.T
        if grad_feature is not None:
            grad_feat = grad_feat + grad_feature
        grad_z2 = grad_feat * (c["z2"] > 0)
        gw2 = c["h1"].T @ grad_z2
        gb2 = grad_z2.sum(axis=0)
        grad_h1 = grad_z2 @ self.w2.T
        grad_z1 = grad_h1 * (c["z1"] > 0)
        gw1 = c["x"].T @ grad_z1
        gb1 = grad_z1.sum(axis=0)
        self._apply(
            lr,
            (self.w_cls, gw_cls), (self.b_cls, gb_cls), (self.w_box, gw_box), (self.b_box, gb_box),
            (self.w2, gw2), (self.b2, gb2), (self.w1, gw1), (self.b1, gb1),
        )

    def _apply(self, lr: float, *pairs) -> None:
        for param, grad in pairs:
            np.clip(grad, -5.0, 5.0, out=grad)
            param -= lr * grad.astype(np.float32)

    def count_parameters_and_ops(self) -> Dict[str, int]:
        weight_ops = self.w1.size + self.w2.size + self.w_cls.size + self.w_box.size
        params = weight_ops + self.b1.size + self.b2.size + self.b_cls.size + self.b_box.size
        return {"parameters": int(params), "ops_per_sample": int(weight_ops)}


@dataclass
class LightweightMLP:
    input_dim: int
    bottleneck_dim: int = 24
    feature_dim: int = 32
    groups: int = 4
    num_classes: int = 3
    seed: int = 1

    def __post_init__(self) -> None:
        if self.input_dim % self.groups != 0:
            raise ValueError("input_dim must be divisible by groups")
        if self.bottleneck_dim % self.groups != 0:
            raise ValueError("bottleneck_dim must be divisible by groups")
        rng = np.random.default_rng(self.seed)
        self.group_in = self.input_dim // self.groups
        self.group_out = self.bottleneck_dim // self.groups
        self.w_group = _he(rng, self.group_in, (self.groups, self.group_in, self.group_out))
        self.b_group = np.zeros((self.groups, self.group_out), dtype=np.float32)
        self.w_mix = _he(rng, self.bottleneck_dim, (self.bottleneck_dim, self.feature_dim))
        self.b_mix = np.zeros(self.feature_dim, dtype=np.float32)
        self.w_cls = _he(rng, self.feature_dim, (self.feature_dim, self.num_classes))
        self.b_cls = np.zeros(self.num_classes, dtype=np.float32)
        self.w_box = _he(rng, self.feature_dim, (self.feature_dim, 6))
        self.b_box = np.zeros(6, dtype=np.float32)

    def _shuffle(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        return x.reshape(n, self.groups, self.group_out).transpose(0, 2, 1).reshape(n, self.bottleneck_dim)

    def _unshuffle(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        return x.reshape(n, self.group_out, self.groups).transpose(0, 2, 1).reshape(n, self.bottleneck_dim)

    def forward(self, x: np.ndarray, cache: bool = False) -> Dict[str, np.ndarray]:
        parts = x.reshape(x.shape[0], self.groups, self.group_in)
        z_group = np.einsum("ngi,gio->ngo", parts, self.w_group) + self.b_group
        a_group = relu(z_group).reshape(x.shape[0], self.bottleneck_dim)
        shuffled = self._shuffle(a_group)
        z_mix = shuffled @ self.w_mix + self.b_mix
        feat = relu(z_mix)
        logits = feat @ self.w_cls + self.b_cls
        box_raw = feat @ self.w_box + self.b_box
        box = sigmoid(box_raw)
        out = {"logits": logits, "box": box, "feature": feat}
        if cache:
            out["cache"] = {
                "x": x, "parts": parts, "z_group": z_group, "a_group": a_group,
                "shuffled": shuffled, "z_mix": z_mix, "feat": feat, "box": box,
            }
        return out

    def backward(self, cached: Dict[str, np.ndarray], grad_logits: np.ndarray, grad_box: np.ndarray, lr: float, grad_feature=None) -> None:
        c = cached
        grad_box_raw = grad_box * c["box"] * (1.0 - c["box"])
        gw_cls = c["feat"].T @ grad_logits
        gb_cls = grad_logits.sum(axis=0)
        gw_box = c["feat"].T @ grad_box_raw
        gb_box = grad_box_raw.sum(axis=0)
        grad_feat = grad_logits @ self.w_cls.T + grad_box_raw @ self.w_box.T
        if grad_feature is not None:
            grad_feat = grad_feat + grad_feature
        grad_z_mix = grad_feat * (c["z_mix"] > 0)
        gw_mix = c["shuffled"].T @ grad_z_mix
        gb_mix = grad_z_mix.sum(axis=0)
        grad_shuffled = grad_z_mix @ self.w_mix.T
        grad_a_group = self._unshuffle(grad_shuffled)
        grad_z_group = grad_a_group.reshape(c["z_group"].shape) * (c["z_group"] > 0)
        gw_group = np.einsum("ngi,ngo->gio", c["parts"], grad_z_group)
        gb_group = grad_z_group.sum(axis=0)
        self._apply(
            lr,
            (self.w_cls, gw_cls), (self.b_cls, gb_cls), (self.w_box, gw_box), (self.b_box, gb_box),
            (self.w_mix, gw_mix), (self.b_mix, gb_mix), (self.w_group, gw_group), (self.b_group, gb_group),
        )

    def _apply(self, lr: float, *pairs) -> None:
        for param, grad in pairs:
            np.clip(grad, -5.0, 5.0, out=grad)
            param -= lr * grad.astype(np.float32)

    def count_parameters_and_ops(self) -> Dict[str, int]:
        weight_ops = self.w_group.size + self.w_mix.size + self.w_cls.size + self.w_box.size
        params = weight_ops + self.b_group.size + self.b_mix.size + self.b_cls.size + self.b_box.size
        return {"parameters": int(params), "ops_per_sample": int(weight_ops)}

