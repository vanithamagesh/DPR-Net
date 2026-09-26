"""l = 0.75 BCE + 0.25 Dice on each output; L = 0.8 l(z2) + 0.15 l(z0) + 0.05 l(aux) + lambda(t) clDice."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import soft_skeleton


def masked_mean(x, valid):
    return (x * valid).sum() / valid.sum().clamp_min(1)


def soft_dice(p, y, valid):
    d = (1, 2, 3)
    return 1 - ((2 * (p * y * valid).sum(d) + 1) / ((p * valid).sum(d) + (y * valid).sum(d) + 1)).mean()


def region_loss(logits, y, valid):
    bce = masked_mean(F.binary_cross_entropy_with_logits(logits, y, reduction="none"), valid)
    return 0.75 * bce + 0.25 * soft_dice(logits.sigmoid(), y, valid)


def cldice_loss(p, y, skel_y, valid, margin: int = 4):
    core = valid.clone()
    core[..., :margin, :] = 0; core[..., -margin:, :] = 0
    core[..., :, :margin] = 0; core[..., :, -margin:] = 0
    sp, sy = soft_skeleton(p, 3) * core, skel_y * core
    d = (1, 2, 3)
    tprec = ((sp * y).sum(d) + 1) / (sp.sum(d) + 1)
    tsens = ((sy * p).sum(d) + 1) / (sy.sum(d) + 1)
    return 1 - (2 * tprec * tsens / (tprec + tsens + 1e-8)).mean()


def total_loss(out, y, valid, skel_y, cl_weight: float):
    loss = (0.80 * region_loss(out["logits"], y, valid)
            + 0.15 * region_loss(out["initial_logits"], y, valid)
            + 0.05 * region_loss(out["aux_logits"], y, valid))
    if cl_weight > 0:
        loss = loss + cl_weight * cldice_loss(out["logits"].sigmoid(), y, skel_y, valid)
    return loss
