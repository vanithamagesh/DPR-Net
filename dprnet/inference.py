"""Whole-image prediction with eight-view test-time augmentation (flips and transposes)."""
from __future__ import annotations

import numpy as np
import torch

from .data import features


def _views():
    for t in (False, True):
        for dims in ((), (-1,), (-2,), (-2, -1)):
            yield t, dims


@torch.inference_mode()
def predict(model, image: np.ndarray, device, tta: bool = True) -> np.ndarray:
    x = torch.from_numpy(features(image).transpose(2, 0, 1)).float()[None].to(device)
    probs = []
    for t, dims in (_views() if tta else [(False, ())]):
        v = x.transpose(-2, -1) if t else x
        v = torch.flip(v, dims) if dims else v
        p = model(v)["logits"].sigmoid()
        p = torch.flip(p, dims) if dims else p
        probs.append(p.transpose(-2, -1) if t else p)
    return torch.stack(probs).mean(0)[0, 0].float().cpu().numpy()
