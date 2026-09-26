# DPR-Net: Recovering Thin Retinal Vessels with a Lightweight Dual-Domain Segmentation Network

PyTorch implementation of DPR-Net and of the scale-normalised, equal-precision thin-vessel evaluation.

DPR-Net combines

- **dual-domain encoder blocks**: a spatial residual block plus a Fourier path that corrects log-amplitude and phase (zero-initialised, so every block starts as a plain residual block);
- **pinwheel-edge attention** on the skip connections: four asymmetrically padded 1×5 / 5×1 convolutions merged by a 2×2 convolution, Sobel magnitudes at 0°, 45°, 90° and 135°, and squeeze-and-excitation;
- a **selective fusion block** at the bottleneck: 3×3 and dilated 5×5 depthwise branches with spatial and channel selection, then three channel self-attention layers (4 heads, top-k 80 %);
- a **gated decoder** (skip gate in [0.5, 1]), a full-resolution **line branch** (1×9 and 9×1 depthwise), and
- an **uncertainty-driven refinement unit** applied twice with shared weights.

The **baseline** is the same network without the frequency paths, the pinwheel-edge attention and the selective fusion block.

## Results (paper)

Pooled over the test images inside the FOV.

| Model | Params (M) | DRIVE Se | DRIVE F1 | CHASE_DB1 Se | CHASE_DB1 F1 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Baseline | 0.59 | 0.8339 | 0.8260 | 0.7886 | 0.8115 |
| DPR-Net | 1.14 | 0.8548 | 0.8300 | 0.8059 | 0.8190 |

Thin-vessel recall (%) — normalised radius ≤ 2 DRIVE-equivalent pixels:

| Model | DRIVE, own threshold | DRIVE, equal precision | CHASE, own threshold | CHASE, equal precision |
|:--|:--:|:--:|:--:|:--:|
| Baseline | 76.53 | 77.92 | 64.83 | 65.43 |
| DPR-Net | 78.98 | 78.98 | 67.18 | 67.18 |

## Installation

```bash
git clone https://github.com/<user>/DPR-Net.git
cd DPR-Net
pip install -r requirements.txt
python count_params.py
```

## Data

The datasets are not redistributed here. Download them from the official sources and unpack them unchanged:

- DRIVE: https://drive.grand-challenge.org/
- CHASE_DB1: https://researchdata.kingston.ac.uk/96/

```
data/
├── DRIVE/
│   ├── training/{images,1st_manual,mask}/
│   └── test/{images,1st_manual,mask}/
└── CHASE_DB1/
    ├── Image_01L.jpg  Image_01L_1stHO.png  Image_01L_2ndHO.png
    └── ...
```

Splits:

- **DRIVE**: 20 training images (22, 27, 34 and 39 held out for validation) and 20 test images.
- **CHASE_DB1**: images 1–10 (01L–10R) for training, with 03L, 05R, 08L and 10R held out for validation. Images 11–14 (11L–14R) are used for testing.
- The CHASE_DB1 FOV mask is estimated by thresholding the red channel.

## Training

```bash
python train.py --dataset drive --data-root data/DRIVE --model dprnet   --out runs/drive_dprnet
python train.py --dataset drive --data-root data/DRIVE --model baseline --out runs/drive_baseline
```

Schedule (defaults):

| Stage | Steps | Learning rate | Notes |
|:--|:--:|:--:|:--|
| Phase 1 | 3000 | 1e-3 | clDice weight ramped 0 → 0.05 over the first quarter |
| Phase 2 | 1500 | 2e-4 | 25 % of patch centres on thin-vessel centrelines |
| Refit | 500 | 1e-4 | all training images; validation threshold kept fixed |

Other settings:

- 128×128 patches, batch size 8
- AdamW with cosine decay
- gradient clipping at 2
- EMA of the weights (0.99)
- seed 73

Loss: ℓ = 0.75 BCE + 0.25 Dice; L = 0.8 ℓ(z⁽²⁾) + 0.15 ℓ(z⁽⁰⁾) + 0.05 ℓ(z_aux) + λ(t) clDice.

## Testing (eight-view TTA)

```bash
python test.py --dataset drive --data-root data/DRIVE --ckpt runs/drive_dprnet/final.pt --out results/drive_dprnet
```

This writes probability maps (`prob/*.npy`), binary maps (`binary/*.png`) and `metrics.json`. The metrics are accuracy, Se, Sp, precision, F1, IoU, ROC-AUC and PR-AUC.

## Thin-vessel evaluation

```bash
python evaluate_thin.py --dataset chase --data-root data/CHASE_DB1 \
    --model DPR-Net results/chase_dprnet 0.35 \
    --model Baseline results/chase_baseline 0.45 \
    --second-observer --out results/chase_thin.json
```

How it works:

- The local radius of a vessel pixel is the distance-transform value at the nearest skeleton pixel.
- It is scaled to DRIVE-equivalent pixels by the ratio of the median FOV diameters (537.99 px for DRIVE, 914.54 px for CHASE_DB1). A pixel is thin when the scaled radius is at most 2.
- The first model given is the reference. Every other model is also scored at the reference's pooled precision (equal precision).
- The output also includes recall by radius bin and a two-sided Wilcoxon signed-rank test on the per-image thin-vessel recall.

To run everything:

```bash
bash scripts/run_all.sh data
```

## Repository layout

```
dprnet/model.py       network (DPRNet, baseline via build("baseline"))
dprnet/data.py        dataset discovery, pre-processing, patch sampler
dprnet/geometry.py    soft skeleton, local and normalised radius
dprnet/losses.py      region loss and clDice
dprnet/metrics.py     pooled metrics, thin-vessel evaluator, Wilcoxon test
dprnet/inference.py   eight-view test-time augmentation
train.py  test.py  evaluate_thin.py  count_params.py
```

## Citation

The citation will be added once the paper is published.

## License

MIT
