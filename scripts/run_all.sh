#!/usr/bin/env bash
# Train and test DPR-Net and the baseline on DRIVE and CHASE_DB1, then run the thin-vessel evaluation.
# Usage: bash scripts/run_all.sh [DATA_DIR]   (DATA_DIR contains DRIVE/ and CHASE_DB1/)
set -euo pipefail
DATA=${1:-data}
for ds in drive chase; do
  root=$DATA/DRIVE; [ "$ds" = chase ] && root=$DATA/CHASE_DB1
  for m in dprnet baseline; do
    python train.py --dataset $ds --data-root $root --model $m --out runs/${ds}_${m}
    python test.py  --dataset $ds --data-root $root --ckpt runs/${ds}_${m}/final.pt --out results/${ds}_${m}
  done
  t1=$(python -c "import json;print(json.load(open('results/${ds}_dprnet/metrics.json'))['threshold'])")
  t2=$(python -c "import json;print(json.load(open('results/${ds}_baseline/metrics.json'))['threshold'])")
  extra=""; [ "$ds" = chase ] && extra="--second-observer"
  python evaluate_thin.py --dataset $ds --data-root $root \
    --model DPR-Net results/${ds}_dprnet $t1 --model Baseline results/${ds}_baseline $t2 \
    $extra --out results/${ds}_thin.json
done
