"""Pooled pixel metrics, scale-normalised thin-vessel recall, equal-precision comparison and radius bins."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from .geometry import THIN_RADIUS, normalised_radius


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def scores(tp, fp, fn, tn):
    e = 1e-12
    return {"accuracy": (tp + tn) / (tp + fp + fn + tn + e), "sensitivity": tp / (tp + fn + e),
            "specificity": tn / (tn + fp + e), "precision": tp / (tp + fp + e),
            "f1": 2 * tp / (2 * tp + fp + fn + e), "iou": tp / (tp + fp + fn + e)}


def pooled_metrics(probs, gts, fovs, threshold):
    """All pixel counts pooled over the test images inside the FOV."""
    tot = np.zeros(4, np.int64)
    ps, ys = [], []
    for p, y, v in zip(probs, gts, fovs):
        v = v.astype(bool)
        tot += confusion(p[v] >= threshold, y[v])
        ps.append(p[v]); ys.append(y[v].astype(bool))
    out = scores(*tot)
    p, y = np.concatenate(ps), np.concatenate(ys)
    out["roc_auc"] = float(roc_auc_score(y, p)); out["pr_auc"] = float(average_precision_score(y, p))
    return out


def best_threshold(probs, gts, fovs, grid=np.arange(0.20, 0.801, 0.01)):
    """Threshold maximising the pooled F1 score (used on the validation images only)."""
    p = np.concatenate([a[v.astype(bool)] for a, v in zip(probs, fovs)])
    y = np.concatenate([a[v.astype(bool)] for a, v in zip(gts, fovs)]).astype(bool)
    best = max((2 * (b & y).sum() / (b.sum() + y.sum() + 1e-12), float(round(t, 2)))
               for t in grid for b in [p >= t])
    return best[1], float(best[0])


class ThinVesselEvaluator:
    """Thin-vessel recall with radii scaled to DRIVE-equivalent pixels (thin: r~ <= 2)."""

    def __init__(self, gts, fovs, dataset):
        self.gts = [g.astype(bool) for g in gts]
        self.fovs = [f.astype(bool) for f in fovs]
        self.radii = [normalised_radius(g, dataset) for g in self.gts]
        self.thin = [g & (r <= THIN_RADIUS) & f for g, r, f in zip(self.gts, self.radii, self.fovs)]

    def at(self, probs, threshold):
        tp = fp = thin_hit = thin_all = 0
        per_image = []
        for p, g, f, t in zip(probs, self.gts, self.fovs, self.thin):
            b = (p >= threshold) & f
            tp += (b & g).sum(); fp += (b & ~g).sum()
            h = (b & t).sum(); thin_hit += h; thin_all += t.sum()
            per_image.append(100.0 * h / max(t.sum(), 1))
        return {"precision": tp / max(tp + fp, 1), "thin_recall": 100.0 * thin_hit / max(thin_all, 1),
                "per_image": per_image}

    def at_precision(self, probs, target_precision, lo=0.01, hi=0.99, iters=40):
        """Move the threshold (bisection) until the pooled precision equals ``target_precision``."""
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.at(probs, mid)["precision"] < target_precision:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        return {"threshold": t, **self.at(probs, t)}

    def by_radius(self, probs, threshold, edges=(0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 99)):
        hits = np.zeros(len(edges) - 1); alls = np.zeros(len(edges) - 1)
        for p, g, f, r in zip(probs, self.gts, self.fovs, self.radii):
            b = (p >= threshold) & f
            idx = np.digitize(r[g & f], edges) - 1
            np.add.at(alls, idx, 1); np.add.at(hits, idx, b[g & f])
        return {f"{edges[i]}-{edges[i + 1]}": 100.0 * hits[i] / max(alls[i], 1) for i in range(len(hits))}


def paired_wilcoxon(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "a_better": 0, "n": len(a)}
    stat, p = wilcoxon(a, b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "a_better": int((a > b).sum()), "n": len(a)}
