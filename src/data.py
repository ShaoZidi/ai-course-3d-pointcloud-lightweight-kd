"""Synthetic 3D voxel data used by the course project.

The real paper evaluates on KITTI and nuScenes. Those datasets are too large for
this small assignment environment, so this module creates a deterministic toy
task with three object types. The goal is not to replace KITTI, but to make the
baseline/lightweight/distillation pipeline executable.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


CLASS_NAMES = ("car", "pedestrian", "cyclist")


def _sample_box(rng: np.random.Generator, cls: int, grid_shape: Tuple[int, int, int]):
    depth, height, width = grid_shape
    if cls == 0:
        base = np.array([2, 3, 8], dtype=np.int32)
    elif cls == 1:
        base = np.array([6, 2, 2], dtype=np.int32)
    else:
        base = np.array([3, 4, 7], dtype=np.int32)

    jitter = rng.integers(-1, 2, size=3)
    size = np.maximum(base + jitter, 1)
    size = np.minimum(size, np.array(grid_shape) - 2)
    low = size // 2 + 1
    high = np.array(grid_shape) - (size - size // 2) - 1
    center = np.array([rng.integers(low[i], high[i] + 1) for i in range(3)])
    return center.astype(np.int32), size.astype(np.int32)


def _draw_box(grid: np.ndarray, center: np.ndarray, size: np.ndarray, value: float) -> None:
    start = center - size // 2
    end = start + size
    z0, y0, x0 = start
    z1, y1, x1 = end
    grid[z0:z1, y0:y1, x0:x1] = value


def _draw_cyclist(grid: np.ndarray, center: np.ndarray, size: np.ndarray) -> None:
    depth, height, width = grid.shape
    z_ground = max(center[0] - size[0] // 2, 0)
    x_left = int(np.clip(center[2] - size[2] // 3, 1, width - 2))
    x_right = int(np.clip(center[2] + size[2] // 3, 1, width - 2))
    y_mid = int(np.clip(center[1], 1, height - 2))

    for x in (x_left, x_right):
        grid[z_ground, y_mid - 1:y_mid + 2, x - 1:x + 2] = 1.0

    frame_z = min(z_ground + 1, depth - 1)
    grid[frame_z, y_mid, x_left:x_right + 1] = 0.9
    rider_z0 = min(frame_z + 1, depth - 1)
    rider_z1 = min(rider_z0 + 2, depth)
    grid[rider_z0:rider_z1, y_mid:y_mid + 1, center[2]:center[2] + 1] = 1.0
    grid[min(rider_z1, depth - 1), max(y_mid - 1, 0):min(y_mid + 2, height), center[2]:center[2] + 1] = 0.8


def make_dataset(
    n_samples: int = 600,
    seed: int = 0,
    grid_shape: Tuple[int, int, int] = (8, 12, 12),
    noise_prob: float = 0.015,
) -> Dict[str, np.ndarray]:
    """Create a deterministic synthetic voxel dataset.

    Each sample contains one object. The labels are class id and a normalized
    box vector: center_z, center_y, center_x, size_z, size_y, size_x.
    """

    rng = np.random.default_rng(seed)
    depth, height, width = grid_shape
    x = np.zeros((n_samples, depth, height, width), dtype=np.float32)
    y_class = np.zeros(n_samples, dtype=np.int64)
    y_box = np.zeros((n_samples, 6), dtype=np.float32)
    normalizer = np.array([depth, height, width, depth, height, width], dtype=np.float32)

    for i in range(n_samples):
        cls = int(rng.integers(0, len(CLASS_NAMES)))
        center, size = _sample_box(rng, cls, grid_shape)
        grid = np.zeros(grid_shape, dtype=np.float32)
        if cls == 2:
            _draw_cyclist(grid, center, size)
        else:
            _draw_box(grid, center, size, 1.0)

        noise = rng.random(grid_shape) < noise_prob
        grid = np.maximum(grid, noise.astype(np.float32) * 0.35)
        x[i] = grid
        y_class[i] = cls
        y_box[i] = np.concatenate([center, size]).astype(np.float32) / normalizer

    return {
        "x": x.reshape(n_samples, -1),
        "y_class": y_class,
        "y_box": y_box,
        "grid_shape": np.array(grid_shape, dtype=np.int64),
        "class_names": np.array(CLASS_NAMES),
    }
