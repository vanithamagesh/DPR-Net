"""Predict the test images with eight-view TTA, save probability maps and pooled metrics.

  python test.py --dataset drive --data-root data/DRIVE --ckpt runs/drive_dprnet/final.pt --out results/drive_dprnet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dprnet import build
from dprnet.data import discover, load
from dprnet.inference import predict
from dprnet.metrics import pooled_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["drive", "chase"], required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, help="override the validation threshold stored in the checkpoint")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build(ck["arch"]); model.load_state_dict(ck["model"]); model.to(a.device).eval()
    thr = a.threshold if a.threshold is not None else ck["threshold"]
    (a.out / "prob").mkdir(parents=True, exist_ok=True); (a.out / "binary").mkdir(exist_ok=True)
    probs, gts, fovs = [], [], []
    for s in discover(a.dataset, a.data_root, "test"):
        img, y, v = load(s)
        p = predict(model, img, a.device, tta=not a.no_tta)
        np.save(a.out / "prob" / f"{s.name}.npy", p.astype(np.float32))
        Image.fromarray(((p >= thr) & (v > 0)).astype(np.uint8) * 255).save(a.out / "binary" / f"{s.name}.png")
        probs.append(p); gts.append(y); fovs.append(v)
        print(s.name, flush=True)
    m = pooled_metrics(probs, gts, fovs, thr)
    m.update({"threshold": thr, "checkpoint": str(a.ckpt), "arch": ck["arch"], "tta_views": 1 if a.no_tta else 8})
    (a.out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
