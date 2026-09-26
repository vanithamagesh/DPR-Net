"""Skeletons and scale-normalised vessel radius."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize

# Median FOV diameters (pixels) used to express radii in DRIVE-equivalent pixels.
FOV_DIAMETER = {"drive": 537.99, "chase": 914.54}
THIN_RADIUS = 2.0


def _erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_skeleton(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Differentiable soft skeleton (clDice)."""
    skel = F.relu(x - F.max_pool2d(_erode(x), 3, 1, 1))
    for _ in range(iterations):
        x = _erode(x)
        delta = F.relu(x - F.max_pool2d(_erode(x), 3, 1, 1))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def local_radius(vessels: np.ndarray) -> np.ndarray:
    """Radius r(q) of every vessel pixel: distance-transform value at the nearest skeleton pixel."""
    vessels = vessels.astype(bool)
    dist = ndimage.distance_transform_edt(vessels)
    skel = skeletonize(vessels)
    if not skel.any():
        return np.zeros(vessels.shape, np.float32)
    _, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    r = dist[iy, ix].astype(np.float32)
    r[~vessels] = 0
    return r


def normalised_radius(vessels: np.ndarray, dataset: str) -> np.ndarray:
    """r~(q) = r(q) * d_DRIVE / d_dataset."""
    return local_radius(vessels) * FOV_DIAMETER["drive"] / FOV_DIAMETER[dataset]
