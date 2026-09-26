"""Train DPR-Net (or the baseline) in three stages.

  phase 1 : 3000 steps, lr 1e-3, clDice weight ramped 0 -> 0.05 over the first quarter
  phase 2 : 1500 steps, lr 2e-4, a quarter of the patch centres on thin-vessel centrelines
  refit   :  500 steps, lr 1e-4, all training images (validation images included)

The threshold is selected on the validation images after phase 2 and kept fixed.

Example:
  python train.py --dataset drive --data-root data/DRIVE --model dprnet --out runs/drive_dprnet
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dprnet import build, count_parameters
from dprnet.data import PatchDataset, discover, load
from dprnet.inference import predict
from dprnet.losses import total_loss
from dprnet.metrics import best_threshold


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_stage(model, ema, samples, args, steps, lr, thin_share, cl_ramp, tag, log):
    ds = PatchDataset(samples, args.dataset, steps * args.batch, args.patch, args.seed + len(tag), thin_share)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.02)
    start = time.time()
    for step, (x, y, v, s) in enumerate(dl, 1):
        x, y, v, s = (t.to(args.device) for t in (x, y, v, s))
        model.train()
        cl = 0.05 * min(1.0, step / (0.25 * steps)) if cl_ramp else 0.05
        loss = total_loss(model(x), y, v, s, cl)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at {tag} step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        with torch.no_grad():
            decay = min(args.ema, 1 - 1 / (step + 1))
            for e, p in zip(ema.parameters(), model.parameters()):
                e.lerp_(p, 1 - decay)
            for e, b in zip(ema.buffers(), model.buffers()):
                e.copy_(b)
        if step == 1 or step % args.log_every == 0 or step == steps:
            rec = {"stage": tag, "step": step, "loss": round(float(loss), 5), "lr": sched.get_last_lr()[0],
                   "seconds": round(time.time() - start, 1)}
            log.append(rec); print(json.dumps(rec), flush=True)


@torch.no_grad()
def select_threshold(model, samples, device):
    model.eval()
    probs, gts, fovs = [], [], []
    for s in samples:
        img, y, v = load(s)
        probs.append(predict(model, img, device, tta=True)); gts.append(y); fovs.append(v)
    return best_threshold(probs, gts, fovs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["drive", "chase"], required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model", choices=["dprnet", "baseline"], default="dprnet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps1", type=int, default=3000); ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--steps2", type=int, default=1500); ap.add_argument("--lr2", type=float, default=2e-4)
    ap.add_argument("--steps3", type=int, default=500); ap.add_argument("--lr3", type=float, default=1e-4)
    ap.add_argument("--thin-share", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--ema", type=float, default=0.99); ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train_s = discover(args.dataset, args.data_root, "train")
    val_s = discover(args.dataset, args.data_root, "val")
    model = build(args.model).to(args.device)
    ema = copy.deepcopy(model).eval()
    n = count_parameters(model)
    print(f"{args.model}: {n:,} parameters", flush=True)
    log = []

    run_stage(model, ema, train_s, args, args.steps1, args.lr1, 0.0, True, "phase1", log)
    run_stage(model, ema, train_s, args, args.steps2, args.lr2, args.thin_share, False, "phase2", log)
    model.load_state_dict(ema.state_dict())
    threshold, val_f1 = select_threshold(ema, val_s, args.device)
    print(f"validation threshold {threshold:.2f} (F1 {val_f1:.4f})", flush=True)
    torch.save({"model": ema.state_dict(), "arch": args.model, "threshold": threshold},
               args.out / "phase2.pt")

    if args.steps3 > 0:
        run_stage(model, ema, train_s + val_s, args, args.steps3, args.lr3, args.thin_share, False, "refit", log)
    torch.save({"model": ema.state_dict(), "arch": args.model, "threshold": threshold,
                "dataset": args.dataset, "parameters": n}, args.out / "final.pt")
    (args.out / "train_log.json").write_text(json.dumps(
        {"config": {k: str(v) for k, v in vars(args).items()}, "parameters": n, "threshold": threshold,
         "validation_f1": val_f1, "train": [s.name for s in train_s], "validation": [s.name for s in val_s],
         "log": log}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
