"""Thin-vessel evaluation of several models on the same test set.

Reads the probability maps written by test.py (``<dir>/prob/<case>.npy``). The first model is the
reference: every other model is also scored at the precision of the reference (equal precision).

  python evaluate_thin.py --dataset drive --data-root data/DRIVE \
      --model DPR-Net results/drive_dprnet 0.35 --model Baseline results/drive_baseline 0.45 \
      --out results/drive_thin.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dprnet.data import discover, load
from dprnet.metrics import ThinVesselEvaluator, paired_wilcoxon, pooled_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["drive", "chase"], required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model", nargs=3, action="append", metavar=("NAME", "DIR", "THRESHOLD"), required=True)
    ap.add_argument("--second-observer", action="store_true", help="also score the second observer")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    samples = discover(a.dataset, a.data_root, "test")
    data = [load(s) for s in samples]
    gts, fovs = [d[1] for d in data], [d[2] for d in data]
    ev = ThinVesselEvaluator(gts, fovs, a.dataset)
    models = [(n, [np.load(Path(d) / "prob" / f"{s.name}.npy") for s in samples], float(t)) for n, d, t in a.model]
    if a.second_observer:
        from PIL import Image
        obs = [(np.asarray(Image.open(s.label2).convert("L")) > 127).astype(np.float32) for s in samples]
        models.append(("Second observer", obs, 0.5))

    ref_name, ref_probs, ref_thr = models[0]
    ref = ev.at(ref_probs, ref_thr)
    report = {"dataset": a.dataset, "reference": ref_name, "models": {}}
    for name, probs, thr in models:
        own = ev.at(probs, thr)
        entry = {"threshold": thr, "pooled": pooled_metrics(probs, gts, fovs, thr) if name != "Second observer" else None,
                 "precision": own["precision"], "thin_recall": own["thin_recall"],
                 "thin_recall_per_image": own["per_image"], "by_radius": ev.by_radius(probs, thr)}
        if name not in (ref_name, "Second observer"):
            eq = ev.at_precision(probs, ref["precision"])
            entry["equal_precision"] = {"threshold": eq["threshold"], "precision": eq["precision"],
                                        "thin_recall": eq["thin_recall"]}
            entry["wilcoxon_vs_reference"] = paired_wilcoxon(ref["per_image"], own["per_image"])
        report["models"][name] = entry
        print(f"{name:16s} precision {own['precision']:.4f}  thin recall {own['thin_recall']:.2f}"
              + (f"  | equal precision {entry['equal_precision']['thin_recall']:.2f}" if "equal_precision" in entry else ""))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
